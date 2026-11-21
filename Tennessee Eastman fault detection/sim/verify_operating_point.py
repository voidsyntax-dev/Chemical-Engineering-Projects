"""Step 1: verify the simulator reproduces the Downs & Vogel (1993) base-case operating point."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_sim import simulate, COLS

# Published base case (Downs & Vogel 1993, Table 5 / Mode 1 steady state), XMEAS 1-22
BASE = {1: 0.25052, 2: 3664.0, 3: 4509.3, 4: 9.3477, 5: 26.902, 6: 42.339, 7: 2705.0, 8: 75.0, 9: 120.40,
        10: 0.33712, 11: 80.109, 12: 50.0, 13: 2633.7, 14: 25.160, 15: 50.0, 16: 3102.2, 17: 22.949,
        18: 65.731, 19: 230.31, 20: 341.43, 21: 94.599, 22: 77.297}
N = 500  # 25 h, same length as Rieth training runs
x = simulate(np.zeros((N, 20), dtype=np.int32))
assert np.isfinite(x).all(), "simulation produced NaN/inf"
rows, worst = [], 0.0
for k, ref in BASE.items():
    first, mean, sd = float(x[0, k - 1]), float(x[:, k - 1].mean()), float(x[:, k - 1].std())
    rel = abs(mean - ref) / abs(ref) * 100
    rows.append(dict(xmeas=k, published=ref, sim_t0=first, sim_mean_25h=mean, sim_std_25h=sd, rel_err_mean_pct=rel))
    worst = max(worst, rel)
    print(f"XMEAS({k:2d}) published {ref:10.4f} | t0 {first:10.4f} | mean(25h) {mean:10.4f} +- {sd:8.4f} | err {rel:6.2f}%")
out = dict(n_samples=N, worst_rel_err_pct=worst, median_rel_err_pct=float(np.median([r['rel_err_mean_pct'] for r in rows])),
           rows=rows, note="Zero disturbances (IDV all 0). Random measurement noise on; simulator's fixed internal seed.")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "sim_verification.json"), "w"), indent=1)
print("worst rel err %:", round(worst, 3), "| median:", round(out["median_rel_err_pct"], 3))
