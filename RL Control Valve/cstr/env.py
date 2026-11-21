"""Gymnasium environment: Van de Vusse CSTR with realistic sensors, actuators and disturbances.

Control task: track a set-point for the product concentration cB while keeping the reactor
temperature below a safety limit.

Actions (2, in [-1, 1]) : target dilution rate F/V and jacket duty QK, mapped to their ranges,
                          then rate-limited and saturated (actuator limits).
Observations (11)       : scaled measurements (delayed / lagged / noisy), set-point, error,
                          temperature trend, current actuator positions, safety margin.

Modes
    nominal     : no disturbances, no model mismatch (only set-point changes + sensor noise)
    randomized  : mild domain randomisation (default for training)
    perturbed   : stronger disturbances + model mismatch that training never saw (for testing)
"""
from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import vdv

MODES = ("nominal", "randomized", "randomized_wide", "perturbed", "hard")
DT_S = 10.0          # control interval [s]
H = 1.0 / 3600.0     # 1 s in hours


class VdVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode: str = "randomized", episode_min: float = 60.0, dt_s: float = DT_S,
                 n_sub: int = 10, sp_range=(0.80, 1.05), T_soft: float = 130.0, T_hard: float = 150.0,
                 T_delay_s: float = 20.0, T_lag_s: float = 30.0, TK_lag_s: float = 10.0,
                 analyzer_delay_s: float = 30.0, rate_F: float = 1.5, rate_Q: float = 400.0,
                 move_penalty: float = 0.05, integral_obs: bool = False):
        super().__init__()
        assert mode in MODES, mode
        self.mode = mode
        self.dt_s = dt_s
        self.dt_h = dt_s * H
        self.n_sub = n_sub
        self.max_steps = int(round(episode_min * 60.0 / dt_s))
        self.sp_range = sp_range
        self.T_soft, self.T_hard = T_soft, T_hard
        self.T_delay = max(0, int(round(T_delay_s / dt_s)))
        self.an_delay = max(0, int(round(analyzer_delay_s / dt_s)))
        self.aT = math.exp(-dt_s / T_lag_s) if T_lag_s > 0 else 0.0
        self.aTK = math.exp(-dt_s / TK_lag_s) if TK_lag_s > 0 else 0.0
        self.u_low = np.array(vdv.U_LOW, dtype=np.float64)
        self.u_high = np.array(vdv.U_HIGH, dtype=np.float64)
        self.rate = np.array([rate_F, rate_Q], dtype=np.float64)
        self.move_penalty = move_penalty
        # error memory: exponential moving averages of the scaled cB error (5 min and 20 min),
        # the RL analogue of integral action / disturbance estimation
        self.integral_obs = integral_obs
        self.a_fast = math.exp(-dt_s / 300.0)
        self.a_slow = math.exp(-dt_s / 1200.0)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(13 if integral_obs else 11,),
                                            dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def action_to_u(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        return self.u_low + 0.5 * (a + 1.0) * (self.u_high - self.u_low)

    def u_to_action(self, u):
        u = np.asarray(u, dtype=np.float64)
        return (2.0 * (u - self.u_low) / (self.u_high - self.u_low) - 1.0).astype(np.float32)

    def _sample_scenario(self):
        r = self.np_random
        U = r.uniform
        if self.mode == "nominal":
            s = dict(kin=1.0, dH=1.0, dTin0=0.0, tin_sigma=0.0, cA0f=1.0,
                     foul0=1.0, foul_drop=0.0, cap0=1.0, cap_drop=0.0, noise=1.0)
        elif self.mode == "randomized":
            s = dict(kin=U(0.95, 1.05), dH=U(0.95, 1.05), dTin0=U(-5, 5), tin_sigma=2.0,
                     cA0f=U(0.93, 1.07), foul0=U(0.85, 1.0), foul_drop=U(0.0, 0.10),
                     cap0=1.0, cap_drop=U(0.0, 0.10), noise=1.0)
        elif self.mode == "randomized_wide":   # training mode covering the whole 'perturbed' range
            s = dict(kin=U(0.85, 1.15), dH=U(0.90, 1.10), dTin0=U(-8, 8), tin_sigma=4.0,
                     cA0f=U(0.88, 1.12), foul0=U(0.75, 1.0), foul_drop=U(0.0, 0.15),
                     cap0=U(0.90, 1.0), cap_drop=U(0.0, 0.15), noise=U(1.0, 1.5))
        elif self.mode == "perturbed":  # test set (seen in training only by 'randomized_wide')
            s = dict(kin=U(0.85, 1.15), dH=U(0.90, 1.10), dTin0=U(-8, 8), tin_sigma=4.0,
                     cA0f=U(0.88, 1.12), foul0=U(0.75, 0.90), foul_drop=U(0.05, 0.15),
                     cap0=U(0.90, 1.0), cap_drop=U(0.05, 0.15), noise=1.5)
        else:  # 'hard': HELD-OUT test set, beyond every training range (never used for training,
            #   model selection or MPC tuning). Kinetics drawn from 15-22% off nominal, either side.
            kin = U(1.15, 1.22) if r.uniform() < 0.5 else U(0.78, 0.85)
            s = dict(kin=kin, dH=U(0.85, 1.15), dTin0=U(-10, 10), tin_sigma=5.0,
                     cA0f=U(0.85, 1.15), foul0=U(0.68, 0.78), foul_drop=U(0.10, 0.20),
                     cap0=U(0.85, 0.92), cap_drop=U(0.10, 0.20), noise=2.0)
        return s

    def _next_sp_change(self):
        return self.t_step + int(self.np_random.uniform(12, 30) * 60.0 / self.dt_s)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.sc = self._sample_scenario()
        self.p = vdv.NOMINAL.scaled(self.sc["kin"], self.sc["dH"])
        self.cA0 = vdv.NOMINAL.cA0 * self.sc["cA0f"]
        self.tin_walk = 0.0
        self.Tin = vdv.NOMINAL.Tin + self.sc["dTin0"]
        self.foul = self.sc["foul0"]
        self.cap = self.sc["cap0"]

        # start at the steady state of a random input near nominal
        u0 = np.array([r.uniform(8.0, 20.0), r.uniform(-3000.0, -300.0)])
        x, ok = vdv.steady_state(u0[0], u0[1] * self.cap, self.p, self.cA0, self.Tin, self.foul,
                                 guess=(2.1, 1.0, 0.6, 0.6, 112.0, 110.0))
        if not ok or x[4] > self.T_soft - 5:
            u0 = np.array(vdv.U_NOM)
            x, _ = vdv.steady_state(u0[0], u0[1], self.p, self.cA0, self.Tin, self.foul)
        self.x = tuple(float(v) for v in x)
        self.u = u0.copy()
        self.t_step = 0
        self.sp = float(r.uniform(*self.sp_range))
        self.sp_next = self._next_sp_change()

        # sensor states
        self.T_lagged = self.x[4]
        self.TK_lagged = self.x[5]
        self.T_buf = deque([self.x[4]] * (self.T_delay + 1), maxlen=self.T_delay + 1)
        self.an_buf = deque([(self.x[0], self.x[1])] * (self.an_delay + 1), maxlen=self.an_delay + 1)
        self.meas = self._measure()
        self.T_meas_prev = self.meas["T"]
        self.ie_fast = self.ie_slow = 0.0
        return self._obs(), self._info()

    def _measure(self):
        n = self.sc["noise"]
        r = self.np_random
        cA, cB = self.an_buf[0]
        return dict(cA=cA + n * 0.01 * r.standard_normal(),
                    cB=cB + n * 0.01 * r.standard_normal(),
                    T=self.T_buf[0] + n * 0.1 * r.standard_normal(),
                    TK=self.TK_lagged + n * 0.1 * r.standard_normal())

    def _obs(self):
        m = self.meas
        uF, uQ = self.u_to_action(self.u)
        o = np.array([
            (m["cA"] - 2.14) / 1.0,
            (m["cB"] - 1.0) / 0.2,
            (m["T"] - 114.0) / 10.0,
            (m["TK"] - 112.0) / 10.0,
            (self.sp - 1.0) / 0.2,
            (self.sp - m["cB"]) / 0.1,
            (m["T"] - self.T_meas_prev) / 1.0,
            uF, uQ,
            (self.T_soft - m["T"]) / 10.0,
            self.t_step / self.max_steps,
        ], dtype=np.float32)
        if self.integral_obs:
            o = np.concatenate([o, np.array([self.ie_fast, self.ie_slow], dtype=np.float32)])
        return np.clip(o, -10.0, 10.0)

    def _info(self):
        cA, cB, cC, cD, T, TK = self.x
        return dict(cA=cA, cB=cB, cC=cC, cD=cD, T=T, TK=TK, F=float(self.u[0]), Q=float(self.u[1]),
                    cB_sp=self.sp, Tin=self.Tin, foul=self.foul, cap=self.cap, meas=dict(self.meas),
                    t_min=self.t_step * self.dt_s / 60.0)

    def step(self, action):
        # --- actuator: target -> rate limit -> saturation
        target = self.action_to_u(action)
        du = np.clip(target - self.u, -self.rate, self.rate)
        u_new = np.clip(self.u + du, self.u_low, self.u_high)
        du = u_new - self.u
        self.u = u_new

        # --- disturbances (slow drifts)
        frac = 1.0 / self.max_steps
        self.foul = max(0.3, self.foul - self.sc["foul_drop"] * frac)
        self.cap = max(0.3, self.cap - self.sc["cap_drop"] * frac)
        if self.sc["tin_sigma"] > 0:
            tau = 0.5  # h
            self.tin_walk += (-self.tin_walk / tau) * self.dt_h + \
                self.sc["tin_sigma"] * math.sqrt(self.dt_h) * self.np_random.standard_normal()
        self.Tin = vdv.NOMINAL.Tin + self.sc["dTin0"] + self.tin_walk

        # --- plant
        x = vdv.rk4(self.x, float(self.u[0]), float(self.u[1]) * self.cap, self.dt_h, self.n_sub,
                    self.p, self.cA0, self.Tin, self.foul)
        bad = not all(math.isfinite(v) for v in x)
        if bad:
            x = self.x
        self.x = tuple(max(0.0, v) if i < 4 else v for i, v in enumerate(x))
        self.t_step += 1

        # --- set-point schedule
        if self.t_step >= self.sp_next:
            self.sp = float(self.np_random.uniform(*self.sp_range))
            self.sp_next = self._next_sp_change()

        # --- sensors: thermocouple lag -> transport delay ; analyzer delay
        T, TK = self.x[4], self.x[5]
        self.T_lagged = self.aT * self.T_lagged + (1 - self.aT) * T
        self.TK_lagged = self.aTK * self.TK_lagged + (1 - self.aTK) * TK
        self.T_buf.append(self.T_lagged)
        self.an_buf.append((self.x[0], self.x[1]))
        self.T_meas_prev = self.meas["T"]
        self.meas = self._measure()
        e_meas = (self.sp - self.meas["cB"]) / 0.1
        self.ie_fast = self.a_fast * self.ie_fast + (1 - self.a_fast) * e_meas
        self.ie_slow = self.a_slow * self.ie_slow + (1 - self.a_slow) * e_meas

        # --- reward (uses true states)
        e = (self.x[1] - self.sp) / 0.1
        reward = -min(e * e, 5.0)
        over = max(0.0, (T - self.T_soft) / 5.0)
        reward -= 2.0 * over * over
        reward -= self.move_penalty * float(np.sum((du / self.rate) ** 2))
        terminated = bad or T > self.T_hard
        if terminated:
            reward -= 50.0
        truncated = self.t_step >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()


if __name__ == "__main__":
    import time
    env = VdVEnv()
    obs, _ = env.reset(seed=0)
    n, t0 = 0, time.perf_counter()
    while n < 5000:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        n += 1
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    print(f"{n / dt:.0f} env steps/s (single env, random actions)")
