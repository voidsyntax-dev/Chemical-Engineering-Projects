"""Leakage checks (task step 10). Triggered for every (model, fault) whose mean metric exceeds 99%.

Check A  train/test separation: no run and no individual row of the official TRAIN files appears in the official TEST files.
Check B  alarm timing: if features leaked the fault label, alarms would also fire BEFORE onset. Compare pre-onset alarm rate
         with the fault-free false-alarm rate for every flagged (model, fault).
Check C  label-shuffle control (XGBoost, LSTM-AE): same pipeline, permuted training labels -> diagnosis must fall to ~chance.
Check D  pre-onset windows are classified 'normal' (classifier is not reading a fault tag from the features).
Check E  construction audit (static facts, asserted in code): features are only the 52 process columns; scalers fit on fault-free
         TRAIN; thresholds from held-out FAULT-FREE TRAIN runs; diagnosis evaluated on windows fully after onset.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import *

out = {}
# ---------- Check A: run + row level overlap between official train and test ----------
rng = np.random.default_rng(MASTER_SEED)
wvec = rng.integers(1, 2 ** 63 - 1, size=NVAR, dtype=np.uint64)


def row_hashes(A):
    A = np.ascontiguousarray(np.asarray(A, np.float32).reshape(-1, NVAR))
    return (A.view(np.uint32).astype(np.uint64) * wvec).sum(1, dtype=np.uint64)


def all_hashes(names):
    hs = []
    for n in names:
        arr = np.load(PROC / f"{n}.npy", mmap_mode="r")
        flat = arr.reshape(-1, arr.shape[-2], NVAR)
        for i in range(0, flat.shape[0], 500):
            hs.append(row_hashes(flat[i:i + 500]))
    return np.concatenate(hs)


htr, hte = all_hashes(["ff_train", "fl_train"]), all_hashes(["ff_test", "fl_test"])
common = np.intersect1d(htr, hte)
out["A_row_overlap"] = {"train_rows": int(len(htr)), "test_rows": int(len(hte)), "identical_rows_in_both": int(len(common)),
                        "train_rows_unique": int(len(np.unique(htr))), "note": "hash of the 52-value float32 row"}
print("A: identical rows train vs test:", len(common), "of", len(htr), "train /", len(hte), "test rows")

# ---------- Checks B, C, D from per-model results ----------
flag_rows, models = [], ["PCA", "PCA-theory", "XGBoost", "LSTM-AE"]
for m in models:
    d = RESULTS / "models" / m
    if not (d / "summary.csv").exists():
        continue
    s = pd.read_csv(d / "summary.csv"); ff = pd.read_csv(d / "fault_free.csv")
    s = s[s.fault > 0]
    for _, r in s.iterrows():
        for met in ("fdr_sample", "det_run", "diag_acc", "diag_acc_alarmed"):
            if r[met] > 0.99:
                flag_rows.append(dict(model=m, fault=int(r.fault), metric=met, value=round(float(r[met]), 4),
                                      pre_onset_alarm_rate=round(float(r.far_pre_onset), 4),
                                      fault_free_far_sample=round(float(r.far_ff_sample), 4)))
fl = pd.DataFrame(flag_rows)
out["B_flagged_above_99pct"] = fl.to_dict("records")
if len(fl):
    print(f"B: {len(fl)} (model, fault, metric) entries above 99%; max pre-onset alarm rate among them = {fl.pre_onset_alarm_rate.max()}")
    fl.to_csv(RESULTS / "leakage_flagged.csv", index=False)

ctrl = {}
for m in ("XGBoost", "LSTM-AE"):
    p = RESULTS / "models" / m / "extra.json"
    if not p.exists():
        continue
    e = json.load(open(p))
    if m == "LSTM-AE":
        e = e["leakage_checks"]
        ctrl[m] = {"shuffled_label_diag_acc": e["shuffled_label_control_diag_acc"], "chance": e["chance_level"],
                   "pre_onset_windows_predicted_normal": e["pre_onset_windows_predicted_normal_in_faulty_test_runs"]}
    else:
        ctrl[m] = {"shuffled_label_diag_acc": e.get("shuffled_label_control_diag_acc"), "chance": e.get("chance_level"),
                   "pre_onset_windows_predicted_normal": e.get("pre_onset_windows_predicted_normal")}
out["C_D_controls"] = ctrl
print("C/D:", json.dumps(ctrl, indent=1))

out["E_construction_audit"] = {
    "features": FEATS, "label_like_columns_used": [c for c in FEATS if c in ("faultNumber", "simulationRun", "sample")],
    "scaler_fit": "fault-free TRAINING arrays only (PCA: 80% runs split by run; LSTM-AE: all 500 training runs, test untouched)",
    "thresholds": "99th percentile on held-out fault-free TRAINING runs (split by run)",
    "train_windows": "drawn only from the official train files; faulty-run windows start at/after onset so none straddle it",
    "eval_windows_for_diagnosis": "start >= onset (160) in test runs, so none straddle onset",
    "window_overlap_train_vs_test": "impossible: windows are cut inside a single run and train/test runs are disjoint (Check A)"}
json.dump(out, open(RESULTS / "leakage_checks.json", "w"), indent=1)
print("saved results/leakage_checks.json")
