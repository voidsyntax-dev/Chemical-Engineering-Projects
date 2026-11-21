"""Decentralised PI baseline: cB <- F/V loop, T <- QK loop (velocity form, inherently anti-windup)."""
from __future__ import annotations

import numpy as np


class DualPI:
    def __init__(self, Kc_cB=20.0, Ti_cB_min=4.0, Kc_T=-300.0, Ti_T_min=3.0, T_sp=114.0):
        # Tuned by grid search on the nominal model (mean |cB error| ~ 0.054 mol/L).
        # Set-points 0.80-1.05 lie on the low-F branch of cB(F) where the gain is positive; near
        # the cB maximum (~F/V 14.7 at 114 C) the gain goes to zero and flips sign - a PI can't
        # handle that, which is one of the things RL is expected to do better.
        # Higher T_sp (e.g. 120) is infeasible at low F/V because the jacket can only cool.
        # Kc_cB : (1/h) per (mol/L)   -> +cB error raises F/V (operating left of the cB maximum)
        # Kc_T  : (kJ/h) per K        -> T above set-point makes QK more negative (more cooling)
        self.Kc_cB, self.Ti_cB = Kc_cB, Ti_cB_min
        self.Kc_T, self.Ti_T = Kc_T, Ti_T_min
        self.T_sp = T_sp

    def reset(self, env):
        self.u = np.array(env.u, dtype=float)
        self.e_prev = None
        self.dt_min = env.dt_s / 60.0

    def act(self, env, info):
        m = info["meas"]
        e1 = info["cB_sp"] - m["cB"]
        e2 = m["T"] - self.T_sp          # positive -> too hot -> more cooling (Kc_T < 0)
        if self.e_prev is None:
            self.e_prev = (e1, e2)
        d1 = self.Kc_cB * ((e1 - self.e_prev[0]) + self.dt_min / self.Ti_cB * e1)
        d2 = self.Kc_T * ((e2 - self.e_prev[1]) + self.dt_min / self.Ti_T * e2)
        self.e_prev = (e1, e2)
        self.u = np.clip(self.u + np.array([d1, d2]), env.u_low, env.u_high)
        return env.u_to_action(self.u)
