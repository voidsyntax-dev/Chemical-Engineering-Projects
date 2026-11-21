"""Van de Vusse CSTR benchmark (Chen et al. 1995; Klatt & Engell 1998).

Reactions
    A  -> B  -> C        (B is the desired product, C is the over-reaction product)
    2A -> D              (side product)

States  x = [cA, cB, cC, cD, T, TK]   [mol/L]x4, [degC], [degC]
Inputs  F  = F/V  (dilution rate)      [1/h]
        Q  = QK   (jacket heat duty)   [kJ/h], negative = cooling
Time unit is HOURS everywhere in this module.

cC and cD are not part of the original 4-state benchmark. They are carried along
so that the mass balance can be checked: cA + cB + cC + 2*cD -> cA0 at steady state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

KELVIN = 273.15


@dataclass(frozen=True)
class Params:
    k10: float = 1.287e12   # 1/h
    k20: float = 1.287e12   # 1/h
    k30: float = 9.043e9    # L/(mol h)
    E1: float = 9758.3      # K   (activation energy / R)
    E2: float = 9758.3      # K
    E3: float = 8560.0      # K
    dH_AB: float = 4.2      # kJ/mol  (endothermic)
    dH_BC: float = -11.0    # kJ/mol
    dH_AD: float = -41.85   # kJ/mol
    rho: float = 0.9342     # kg/L
    Cp: float = 3.01        # kJ/(kg K)
    kw: float = 4032.0      # kJ/(h m2 K)
    AR: float = 0.215       # m2
    VR: float = 10.0        # L
    mK: float = 5.0         # kg
    CpK: float = 2.0        # kJ/(kg K)
    cA0: float = 5.1        # mol/L   feed concentration
    Tin: float = 104.9      # degC    feed temperature

    def scaled(self, kin=1.0, dH=1.0) -> "Params":
        """Return a copy with pre-exponential factors and heats of reaction scaled (model mismatch)."""
        return replace(self, k10=self.k10 * kin, k20=self.k20 * kin, k30=self.k30 * kin,
                       dH_AB=self.dH_AB * dH, dH_BC=self.dH_BC * dH, dH_AD=self.dH_AD * dH)


NOMINAL = Params()
U_NOM = (14.19, -1113.5)                 # F/V [1/h], QK [kJ/h]
X_NOM_LIT = (2.14, 1.09, 114.2, 112.9)   # cA, cB, T, TK  (literature steady state)
U_LOW = (3.0, -9000.0)
U_HIGH = (35.0, 0.0)


def rate_constants(T: float, p: Params = NOMINAL):
    Tk = T + KELVIN
    return (p.k10 * math.exp(-p.E1 / Tk),
            p.k20 * math.exp(-p.E2 / Tk),
            p.k30 * math.exp(-p.E3 / Tk))


def rhs(x, F, Q, p: Params = NOMINAL, cA0=None, Tin=None, foul=1.0):
    """Time derivatives (per hour). Plain floats for speed inside the RL env."""
    cA, cB, cC, cD, T, TK = x
    if cA0 is None:
        cA0 = p.cA0
    if Tin is None:
        Tin = p.Tin
    k1, k2, k3 = rate_constants(T, p)
    r1 = k1 * cA
    r2 = k2 * cB
    r3 = k3 * cA * cA
    UA = p.kw * p.AR * foul
    q_rxn = r1 * p.dH_AB + r2 * p.dH_BC + r3 * p.dH_AD          # kJ/(L h)
    dcA = F * (cA0 - cA) - r1 - r3
    dcB = -F * cB + r1 - r2
    dcC = -F * cC + r2
    dcD = -F * cD + 0.5 * r3
    dT = F * (Tin - T) + UA * (TK - T) / (p.rho * p.Cp * p.VR) - q_rxn / (p.rho * p.Cp)
    dTK = (Q + UA * (T - TK)) / (p.mK * p.CpK)
    return (dcA, dcB, dcC, dcD, dT, dTK)


def rk4(x, F, Q, dt_h, n_sub, p: Params = NOMINAL, cA0=None, Tin=None, foul=1.0):
    """Fixed-step RK4 over dt_h hours with n_sub substeps (inputs held constant)."""
    h = dt_h / n_sub
    x = tuple(x)
    for _ in range(n_sub):
        a = rhs(x, F, Q, p, cA0, Tin, foul)
        b = rhs(tuple(xi + 0.5 * h * ki for xi, ki in zip(x, a)), F, Q, p, cA0, Tin, foul)
        c = rhs(tuple(xi + 0.5 * h * ki for xi, ki in zip(x, b)), F, Q, p, cA0, Tin, foul)
        d = rhs(tuple(xi + h * ki for xi, ki in zip(x, c)), F, Q, p, cA0, Tin, foul)
        x = tuple(xi + h / 6.0 * (ai + 2 * bi + 2 * ci + di)
                  for xi, ai, bi, ci, di in zip(x, a, b, c, d))
    return x


def simulate_ivp(x0, u_of_t, t_end_h, p: Params = NOMINAL, t_eval=None, method="LSODA",
                 cA0=None, Tin=None, foul=1.0, rtol=1e-8, atol=1e-10):
    """Reference integrator (SciPy). u_of_t(t) -> (F, Q)."""
    from scipy.integrate import solve_ivp

    def f(t, x):
        F, Q = u_of_t(t)
        return rhs(x, F, Q, p, cA0, Tin, foul)

    return solve_ivp(f, (0.0, t_end_h), np.asarray(x0, float), method=method,
                     t_eval=t_eval, rtol=rtol, atol=atol)


def steady_state(F, Q, p: Params = NOMINAL, cA0=None, Tin=None, foul=1.0, guess=None):
    """Solve rhs = 0 for the full 6-state model. Returns (x_ss, converged)."""
    from scipy.optimize import fsolve

    if guess is None:
        guess = (2.0, 1.0, 0.5, 0.5, 110.0, 108.0)
    sol, info, ier, _ = fsolve(lambda x: rhs(x, F, Q, p, cA0, Tin, foul), np.asarray(guess, float),
                               full_output=True, xtol=1e-12)
    ok = ier == 1 and np.all(np.isfinite(sol)) and np.all(sol[:4] >= -1e-9)
    return sol, bool(ok)


def jacobian(x, F, Q, p: Params = NOMINAL, cA0=None, Tin=None, foul=1.0, eps=1e-6):
    x = np.asarray(x, float)
    f0 = np.asarray(rhs(x, F, Q, p, cA0, Tin, foul))
    J = np.zeros((6, 6))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = eps * max(1.0, abs(x[i]))
        J[:, i] = (np.asarray(rhs(x + dx, F, Q, p, cA0, Tin, foul)) - f0) / dx[i]
    return J


def cB_at_fixed_T(F, T, p: Params = NOMINAL, cA0=None):
    """Analytic steady-state cA, cB when the reactor temperature is held at T (perfect T loop)."""
    cA0 = p.cA0 if cA0 is None else cA0
    Tk = np.asarray(T, float) + KELVIN
    k1 = p.k10 * np.exp(-p.E1 / Tk)
    k2 = p.k20 * np.exp(-p.E2 / Tk)
    k3 = p.k30 * np.exp(-p.E3 / Tk)
    b = F + k1
    cA = (-b + np.sqrt(b * b + 4.0 * k3 * F * cA0)) / (2.0 * k3)
    cB = k1 * cA / (F + k2)
    return cA, cB
