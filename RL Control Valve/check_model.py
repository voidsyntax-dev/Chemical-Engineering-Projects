"""Build-order steps 1-3: verify the Van de Vusse model before any control work.

1. Nominal steady state matches the literature values.
2. Mass balance closes (cA + cB + cC + 2 cD -> cA0) during a dynamic run.
3. The RL env's RK4 integrator agrees with SciPy LSODA.
4. Stability (Jacobian eigenvalues) and the selectivity trade-off map cB(F/V, T).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cstr import vdv

OUT = Path("results")
OUT.mkdir(exist_ok=True)
p = vdv.NOMINAL

# 1 ---------------------------------------------------------------------------------------
x_ss, ok = vdv.steady_state(*vdv.U_NOM)
print("1) Nominal steady state (F/V=14.19 1/h, QK=-1113.5 kJ/h)  converged:", ok)
for name, val, lit in zip(["cA", "cB", "T", "TK"], x_ss[[0, 1, 4, 5]], vdv.X_NOM_LIT):
    print(f"   {name:3s} = {val:8.3f}   literature {lit:8.3f}   diff {val - lit:+.3f}")
print(f"   cC = {x_ss[2]:.3f}, cD = {x_ss[3]:.3f}")

# 2 + 3 -----------------------------------------------------------------------------------
def u_prog(t):  # step changes in both inputs
    F = 14.19 if t < 0.25 else (20.0 if t < 0.6 else 8.0)
    Q = -1113.5 if t < 0.4 else -3000.0
    return F, Q

x0 = (1.0, 0.0, 0.0, 0.0, 105.0, 105.0)       # start far from SS
t_eval = np.linspace(0, 1.0, 361)              # 1 h, 10 s grid
ref = vdv.simulate_ivp(x0, u_prog, 1.0, t_eval=t_eval)
Y = ref.y
total = Y[0] + Y[1] + Y[2] + 2 * Y[3]
# d(total)/dt = F (cA0 - total): integrate the right-hand side and compare with the change in total
from scipy.integrate import solve_ivp
tot_ref = solve_ivp(lambda t, y: [u_prog(t)[0] * (p.cA0 - y[0])], (0, 1.0), [total[0]],
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=1e-3).y[0]
resid = np.max(np.abs(total - tot_ref))
print(f"\n2) Mass balance: total A-equivalent at end = {total[-1]:.4f} (cA0 = {p.cA0}); "
      f"max closure error vs in-out balance = {resid:.2e} mol/L")

x = x0
err = 0.0
for k in range(360):
    F, Q = u_prog(t_eval[k])
    x = vdv.rk4(x, F, Q, 10 / 3600, 10)
    err = max(err, np.max(np.abs(np.array(x) - Y[:, k + 1])))
print(f"3) RK4 (10 s, 10 substeps) vs LSODA: max abs deviation = {err:.2e}")

# 4 ---------------------------------------------------------------------------------------
ev = np.linalg.eigvals(vdv.jacobian(x_ss, *vdv.U_NOM))
print(f"\n4) Eigenvalues at nominal SS (1/h): {np.round(np.sort(ev.real), 2)}  ->",
      "STABLE" if np.all(ev.real < 0) else "UNSTABLE")

Fs = np.linspace(3, 35, 321)
Ts = np.linspace(100, 135, 141)
FF, TT = np.meshgrid(Fs, Ts)
_, CB = vdv.cB_at_fixed_T(FF, TT)
i, j = np.unravel_index(np.argmax(CB), CB.shape)
print(f"   Max steady-state cB = {CB[i, j]:.3f} mol/L at F/V = {FF[i, j]:.1f} 1/h, T = {TT[i, j]:.1f} C")
for T in (105, 110, 114.2, 120, 125, 130):
    _, cb = vdv.cB_at_fixed_T(Fs, T)
    k = np.argmax(cb)
    print(f"   T = {T:5.1f} C : best cB = {cb[k]:.3f} at F/V = {Fs[k]:5.1f}; "
          f"dcB/dF at nominal F = {np.interp(14.19, Fs[1:], np.diff(cb) / np.diff(Fs)):+.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
cs = ax[0].contourf(FF, TT, CB, levels=20, cmap="viridis")
fig.colorbar(cs, ax=ax[0], label="cB [mol/L]")
ax[0].plot(FF[i, j], TT[i, j], "r*", ms=12, label="max cB")
ax[0].plot(*vdv.U_NOM[:1], x_ss[4], "wo", label="nominal")
ax[0].set(xlabel="F/V [1/h]", ylabel="T [C]", title="Steady-state cB (selectivity trade-off)")
ax[0].legend()
for name, row in zip(["cA", "cB", "cC", "2cD"], [Y[0], Y[1], Y[2], 2 * Y[3]]):
    ax[1].plot(ref.t * 60, row, label=name)
ax[1].plot(ref.t * 60, total, "k--", label="total")
ax[1].set(xlabel="t [min]", ylabel="mol/L", title="Dynamic run (mass balance)")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT / "model_check.png", dpi=120)
print(f"\nSaved {OUT / 'model_check.png'}")
