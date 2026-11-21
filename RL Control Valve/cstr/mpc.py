"""Nonlinear MPC baseline (CasADi + IPOPT) for the Van de Vusse CSTR.

Fair-comparison rules: the MPC sees exactly what PI and PPO see (delayed, lagged, noisy
measurements via info["meas"]) and only knows the NOMINAL model. It never reads the true state,
the true disturbances or the perturbed parameters.

Pieces
  Observer : nominal model run in parallel with the plant + the same sensor dynamics (lag,
             delays), corrected by measurement innovations. Two integrating disturbance states
             (on dcB/dt and dT/dt) give offset-free tracking under model mismatch and drifts.
  NMPC     : multiple shooting, N steps of dt = 10 s, RK4. Objective mirrors the RL reward:
             cB tracking + move penalty (same weight) + soft T <= T_soft. Hard input bounds and
             the same rate limits as the actuator.
"""
from __future__ import annotations

import math
from collections import deque

import casadi as ca
import numpy as np

from . import vdv

P = vdv.NOMINAL


def _f_ca(x, u, d):
    """Nominal 4-state model (cA, cB, T, TK) in CasADi, time in hours. d = (d_cB, d_T)."""
    cA, cB, T, TK = x[0], x[1], x[2], x[3]
    F, Q = u[0], u[1]
    Tk = T + vdv.KELVIN
    k1 = P.k10 * ca.exp(-P.E1 / Tk)
    k2 = P.k20 * ca.exp(-P.E2 / Tk)
    k3 = P.k30 * ca.exp(-P.E3 / Tk)
    r1, r2, r3 = k1 * cA, k2 * cB, k3 * cA * cA
    UA = P.kw * P.AR
    dcA = F * (P.cA0 - cA) - r1 - r3
    dcB = -F * cB + r1 - r2 + d[0]
    dT = (F * (P.Tin - T) + UA * (TK - T) / (P.rho * P.Cp * P.VR)
          - (r1 * P.dH_AB + r2 * P.dH_BC + r3 * P.dH_AD) / (P.rho * P.Cp) + d[1])
    dTK = (Q + UA * (T - TK)) / (P.mK * P.CpK)
    return ca.vertcat(dcA, dcB, dT, dTK)


def _rk4_ca(x, u, d, h, n):
    hs = h / n
    for _ in range(n):
        k1 = _f_ca(x, u, d)
        k2 = _f_ca(x + hs / 2 * k1, u, d)
        k3 = _f_ca(x + hs / 2 * k2, u, d)
        k4 = _f_ca(x + hs * k3, u, d)
        x = x + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class NMPC:
    def __init__(self, N=20, move_penalty=0.05, T_soft=130.0, n_rk=2,
                 L=(0.3, 0.3, 0.3, 0.5), Ki_cB=4.0, Ki_T=15.0):
        self.N = N
        self.move_penalty = move_penalty
        self.T_soft = T_soft
        self.L = np.array(L)
        self.Ki_cB, self.Ki_T = Ki_cB, Ki_T
        self.n_rk = n_rk
        self.solver = None
        self.fails = 0

    # ------------------------------------------------------------------ build NLP
    def _build(self, env):
        N = self.N
        lo, hi = env.u_low, env.u_high
        span = hi - lo
        dt_h = env.dt_s / 3600.0
        x = ca.SX.sym("x", 4)
        a = ca.SX.sym("a", 2)          # normalised input in [-1, 1]
        d = ca.SX.sym("d", 2)
        u_phys = lo + (a + 1) / 2 * span
        F_step = ca.Function("F", [x, a, d], [_rk4_ca(x, u_phys, d, dt_h, self.n_rk)])

        X = ca.SX.sym("X", 4, N + 1)
        A = ca.SX.sym("A", 2, N)
        S = ca.SX.sym("S", N)          # T slack
        p = ca.SX.sym("p", 4 + 2 + 1 + 2)   # x0, a_prev, sp, d
        x0, a_prev, sp, dd = p[0:4], p[4:6], p[6], p[7:9]

        rate_a = 2 * env.rate / span   # rate limit in normalised units
        J = 0
        g, lbg, ubg = [], [], []
        g.append(X[:, 0] - x0); lbg += [0] * 4; ubg += [0] * 4
        for k in range(N):
            g.append(X[:, k + 1] - F_step(X[:, k], A[:, k], dd)); lbg += [0] * 4; ubg += [0] * 4
            du = A[:, k] - (a_prev if k == 0 else A[:, k - 1])
            g.append(du); lbg += list(-rate_a); ubg += list(rate_a)
            g.append(X[2, k + 1] - S[k]); lbg += [-ca.inf]; ubg += [self.T_soft]
            e = (X[1, k + 1] - sp) / 0.1
            J += e * e
            J += self.move_penalty * ca.sumsqr(du / rate_a)
            J += 2.0 * (S[k] / 5.0) ** 2 + 10.0 * S[k]
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(A, -1, 1), S)
        nlp = {"x": w, "f": J, "g": ca.vertcat(*g), "p": p}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
                "ipopt.max_iter": 100, "ipopt.tol": 1e-6,
                "ipopt.warm_start_init_point": "yes"}
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        nX, nA = 4 * (N + 1), 2 * N
        self.lbw = np.concatenate([np.tile([0, 0, 50, 50], N + 1), -np.ones(nA), np.zeros(N)])
        self.ubw = np.concatenate([np.tile([10, 10, 250, 250], N + 1), np.ones(nA), np.full(N, 100)])
        self.lbg, self.ubg = np.array(lbg, float), np.array(ubg, float)
        self.nX, self.nA = nX, nA
        self.F_np = F_step

    # ------------------------------------------------------------------ controller API
    def reset(self, env):
        if self.solver is None:
            self._build(env)
        m = env.meas
        self.xh = np.array([m["cA"], m["cB"], m["T"], m["TK"]], float)
        self.d = np.zeros(2)
        self.aT, self.aTK = env.aT, env.aTK
        self.T_lag, self.TK_lag = self.xh[2], self.xh[3]
        self.T_buf = deque([self.xh[2]] * (env.T_delay + 1), maxlen=env.T_delay + 1)
        self.an_buf = deque([tuple(self.xh[:2])] * (env.an_delay + 1), maxlen=env.an_delay + 1)
        self.a_prev = env.u_to_action(env.u).astype(float)
        self.w0 = None
        self.first = True

    def _observe(self, env, meas):
        if not self.first:   # predict with the input that was applied over the last interval
            self.xh = np.array(self.F_np(self.xh, self.a_prev, self.d)).ravel()
            self.T_lag = self.aT * self.T_lag + (1 - self.aT) * self.xh[2]
            self.TK_lag = self.aTK * self.TK_lag + (1 - self.aTK) * self.xh[3]
            self.T_buf.append(self.T_lag)
            self.an_buf.append((self.xh[0], self.xh[1]))
        self.first = False
        y_hat = np.array([self.an_buf[0][0], self.an_buf[0][1], self.T_buf[0], self.TK_lag])
        y = np.array([meas["cA"], meas["cB"], meas["T"], meas["TK"]])
        e = y - y_hat
        corr = self.L * e
        self.xh += corr
        # shift the sensor-model memories by the same correction so they stay consistent
        self.T_lag += corr[2]; self.TK_lag += corr[3]
        self.T_buf = deque([v + corr[2] for v in self.T_buf], maxlen=self.T_buf.maxlen)
        self.an_buf = deque([(c + corr[0], b + corr[1]) for c, b in self.an_buf], maxlen=self.an_buf.maxlen)
        self.d[0] += self.Ki_cB * e[1]
        self.d[1] += self.Ki_T * e[2]
        self.d = np.clip(self.d, [-20, -500], [20, 500])
        self.xh[:2] = np.maximum(self.xh[:2], 0.0)

    def sync_applied(self, env):
        """Shadow mode: after env.step() with SOMEONE ELSE's action, tell the observer which input
        was really applied over the last interval (so its model prediction stays correct)."""
        self.a_prev = env.u_to_action(env.u).astype(float)

    def act(self, env, info):
        self._observe(env, info["meas"])
        p = np.concatenate([self.xh, self.a_prev, [info["cB_sp"]], self.d])
        if self.w0 is None:
            self.w0 = np.concatenate([np.tile(self.xh, self.N + 1), np.tile(self.a_prev, self.N),
                                      np.zeros(self.N)])
        sol = self.solver(x0=self.w0, p=p, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        ok = self.solver.stats()["success"]
        w = np.array(sol["x"]).ravel()
        A = w[self.nX:self.nX + self.nA].reshape(2, self.N, order="F").T   # (N, 2)
        if not ok:
            self.fails += 1
        if np.all(np.isfinite(A)):
            a = np.clip(A[0], -1, 1)     # IPOPT's last iterate is still feasible-ish if not converged
        else:
            a = self.a_prev
            A = np.tile(self.a_prev, (self.N, 1))
            w = self.w0
        # warm start: shift solution one step
        X = w[:self.nX].reshape(4, self.N + 1, order="F")
        S = w[self.nX + self.nA:]
        Xs = np.hstack([X[:, 1:], X[:, -1:]])
        As = np.vstack([A[1:], A[-1:]])
        self.w0 = np.concatenate([Xs.ravel(order="F"), As.T.ravel(order="F"), np.r_[S[1:], S[-1:]]])
        # what the actuator will actually do (rate limit) is what the observer should assume
        u_target = env.action_to_u(a)
        du = np.clip(u_target - env.u, -env.rate, env.rate)
        u_applied = np.clip(env.u + du, env.u_low, env.u_high)
        self.a_prev = env.u_to_action(u_applied).astype(float)
        return a.astype(np.float32)
