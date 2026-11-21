"""Figures + descriptive evidence for why faults 3, 9, 15 are hard (computed on test data, descriptive only; nothing is tuned on it)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import *

FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
MODELS = ["PCA", "PCA-theory", "XGBoost", "LSTM-AE"]
COL = {"PCA": "#4C78A8", "PCA-theory": "#9ECAE9", "XGBoost": "#F58518", "LSTM-AE": "#54A24B"}
S = {m: pd.read_csv(RESULTS / "models" / m / "summary.csv") for m in MODELS if (RESULTS / "models" / m / "summary.csv").exists()}
hrs = np.arange(T_TEST) * SAMPLE_MIN / 60

# ---- 1. T2 / SPE examples (PCA, seed 0, test run 1 of each fault) ----
z = np.load(RESULTS / "scores" / "pca_seed0.npz")
lt, ls = z["lim_PCA"]
for name, faults in (("detectable", [1, 4, 6, 13]), ("hard", [3, 9, 15])):
    fig, ax = plt.subplots(2, len(faults), figsize=(3.3 * len(faults), 5), sharex=True)
    for j, f in enumerate(faults):
        for i, (key, lim, lab) in enumerate((("t2", lt, "T²"), ("spe", ls, "SPE (Q)"))):
            a = ax[i, j]
            a.semilogy(hrs, z[f"f{f}_{key}"][0], lw=.7, color="#444")
            a.axhline(lim, color="#d62728", ls="--", lw=1); a.axvline(ONSET_TEST * SAMPLE_MIN / 60, color="k", ls=":", lw=1)
            if i == 0: a.set_title(f"Fault {f}")
            if j == 0: a.set_ylabel(lab)
            if i == 1: a.set_xlabel("time (h)")
    fig.suptitle("PCA monitoring statistics, one test run per fault (dashed = 99% limit from held-out normal runs, dotted = fault onset at 8 h)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / f"pca_t2_spe_{name}.png"); plt.close(fig)

# ---- 2. contribution plots (signature centroids from faulty TRAINING runs) ----
C, labels = z["lib_C"], z["lib_labels"]
names = [n.replace("xmeas_", "XMEAS").replace("xmv_", "XMV") for n in FEATS]
fig, ax = plt.subplots(2, 4, figsize=(13, 5.2))
for j, f in enumerate([1, 4, 6, 13]):
    row = C[list(labels).index(f)]
    for i, (v, lab) in enumerate(((row[:NVAR], "T² contribution"), (row[NVAR:], "SPE contribution"))):
        top = np.argsort(v)[::-1][:8]
        ax[i, j].barh([names[k] for k in top][::-1], v[top][::-1], color=COL["PCA"])
        ax[i, j].set_title(f"Fault {f} - {lab}" if i == 0 else lab, fontsize=8)
fig.suptitle("Contribution plots: mean normalised contribution of the top-8 variables (faulty training runs, post-onset)", fontsize=9)
fig.tight_layout(); fig.savefig(FIGS / "pca_contributions.png"); plt.close(fig)

# ---- 3. detection rate by fault (all models) ----
fig, ax = plt.subplots(figsize=(12, 3.8)); w = 0.2
for k, m in enumerate(S):
    d = S[m][S[m].fault > 0]
    ax.bar(d.fault + (k - 1.5) * w, d.fdr_sample, w, yerr=d.fdr_sample_ci95.fillna(0), color=COL[m], label=m, capsize=1.5)
ax.set_xticks(FAULTS); ax.set_xlabel("fault"); ax.set_ylabel("fraction of post-onset samples in alarm")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.12)); fig.tight_layout()
fig.savefig(FIGS / "detection_rate_by_fault.png"); plt.close(fig)

# ---- 4. faults 3, 9, 15: detection vs false-alarm baseline ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, (key, ffkey, ttl) in zip(ax, (("fdr_sample", "far_ff_sample", "sample level"), ("det_run", "far_ff_run", "run level"))):
    xs = np.arange(3)
    for k, m in enumerate(S):
        d = S[m].set_index("fault")
        a.bar(xs + (k - 1.5) * w, [d.loc[f, key] for f in (3, 9, 15)], w, color=COL[m], label=m)
        a.hlines(d.loc[3, ffkey], -.5, 2.5, color=COL[m], ls="--", lw=1)
    a.set_xticks(xs); a.set_xticklabels(["fault 3", "fault 9", "fault 15"]); a.set_title(f"Detection rate ({ttl}); dashed = same model's fault-free false-alarm rate", fontsize=8)
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(FIGS / "faults_3_9_15_vs_far.png"); plt.close(fig)

# ---- 5. XGBoost / LSTM-AE example scores ----
paths = {"XGBoost": RESULTS / "scores" / "xgb_seed0.npz", "LSTM-AE": RESULTS / "scores" / "lstm_seed0.npz"}
fig, ax = plt.subplots(2, 3, figsize=(11, 4.6), sharex=True)
for i, (m, p) in enumerate(paths.items()):
    if not p.exists(): continue
    zz = np.load(p)
    for j, f in enumerate([1, 10, 3]):
        sc = zz[f"f{f}_score"][0] if m == "XGBoost" else zz[f"fault{f}_scores"][0]
        x = hrs[len(hrs) - len(sc):]
        ax[i, j].plot(x, sc, lw=.7, color=COL[m]); ax[i, j].axhline(float(zz["thr"]), color="#d62728", ls="--", lw=1)
        ax[i, j].axvline(ONSET_TEST * SAMPLE_MIN / 60, color="k", ls=":", lw=1); ax[i, j].set_title(f"{m}, fault {f}", fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "model_scores_examples.png"); plt.close(fig)

# ---- 6. descriptive evidence: how much do the measurements actually change after onset? ----
fl = np.load(PROC / "fl_test.npy", mmap_mode="r")
rows = []
for f in FAULTS:
    X = np.asarray(fl[f - 1], np.float64)          # float64: float32 sums of ~120.4-valued columns are inaccurate
    pre, post = X[:, :ONSET_TEST].reshape(-1, NVAR), X[:, ONSET_TEST + 20:].reshape(-1, NVAR)
    sd = pre.std(0) + 1e-9
    d = np.abs(post.mean(0) - pre.mean(0)) / sd
    vr = np.abs(np.log(post.std(0) / sd))
    rows.append(dict(fault=f, max_mean_shift_in_sd=float(d.max()), max_abs_log_std_ratio=float(vr.max()),
                     top_var=FEATS[int(d.argmax())]))
X = np.asarray(np.load(PROC / "ff_test.npy", mmap_mode="r"), np.float64)      # reference: fault-free runs, same split of time
pre, post = X[:, :ONSET_TEST].reshape(-1, NVAR), X[:, ONSET_TEST + 20:].reshape(-1, NVAR); sd = pre.std(0) + 1e-9
d = np.abs(post.mean(0) - pre.mean(0)) / sd
rows.insert(0, dict(fault=0, max_mean_shift_in_sd=float(d.max()), max_abs_log_std_ratio=float(np.abs(np.log(post.std(0) / sd)).max()),
                    top_var=FEATS[int(d.argmax())]))
ev = pd.DataFrame(rows); ev.to_csv(RESULTS / "fault_signal_strength.csv", index=False)
print(ev.round(3).to_string(index=False))
