"""Evaluation metrics shared by every model (and mirrored in the Kaggle notebook).

All arrays are in SAMPLE-index space, shape (runs, 960). Window models pad the first W-1 samples with "no alarm".
Definitions (per fault, over the 500 official test runs):
  fdr_sample        fraction of post-onset samples (index >= 160) that are in alarm            (the usual TEP "FDR")
  det_run           fraction of runs with >=1 alarm at/after onset
  delay_min         mean over DETECTED runs of (first alarm sample - 160) * 3 min
  far_pre_onset     alarm rate on the pre-onset part (samples < 160) of the same faulty runs
  diag_acc          fraction of post-onset samples (index >= diag_from) whose predicted class == true fault
  diag_acc_alarmed  same, restricted to samples that are in alarm
Fault-free FAR: alarm rate over all (scored) samples of the 500 fault-free test runs, and run-level (>=1 alarm in a run).
"""
import numpy as np
import pandas as pd
from scipy import stats
from .config import ONSET_TEST, SAMPLE_MIN, K_CONSEC


def consecutive(raw, k=K_CONSEC):
    """alarm[t] = raw[t] & ... & raw[t-k+1]  (causal)."""
    a = raw.copy()
    for j in range(1, k):
        a[:, j:] &= raw[:, :-j]
        a[:, :j] = False
    return a


def fault_metrics(alarm, pred, fault, diag_from, valid_from=0):
    post = alarm[:, ONSET_TEST:]
    det = post.any(1)
    out = dict(
        fdr_sample=float(post.mean()), det_run=float(det.mean()),
        delay_min=float(post.argmax(1)[det].mean() * SAMPLE_MIN) if det.any() else np.nan,
        far_pre_onset=float(alarm[:, valid_from:ONSET_TEST].mean()))
    if pred is not None:
        ok = pred[:, diag_from:] == fault
        al = alarm[:, diag_from:]
        out["diag_acc"] = float(ok.mean())
        out["diag_acc_alarmed"] = float(ok[al].mean()) if al.any() else np.nan
    else:
        out["diag_acc"] = out["diag_acc_alarmed"] = np.nan
    return out


def fault_free_metrics(alarm, pred, valid_from=0):
    a = alarm[:, valid_from:]
    out = dict(far_sample=float(a.mean()), far_run=float(a.any(1).mean()))
    out["normal_pred_acc"] = float((pred[:, valid_from:] == 0).mean()) if pred is not None else np.nan
    return out


def mean_ci(v, level=0.95):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h)


METRIC_COLS = ["fdr_sample", "det_run", "delay_min", "far_pre_onset", "diag_acc", "diag_acc_alarmed"]


def summarise(per_seed: pd.DataFrame, ff: pd.DataFrame):
    """Mean and 95% t-interval (across seeds) per fault + macro-average row (fault == 0 means 'ALL')."""
    far_m, far_h = mean_ci(ff.far_sample)
    farr_m, farr_h = mean_ci(ff.far_run)
    rows = []
    for f, d in per_seed.groupby("fault"):
        r = {"fault": int(f), "n_seeds": d.seed.nunique()}
        for c in METRIC_COLS:
            r[c], r[c + "_ci95"] = mean_ci(d[c])
        r.update(far_ff_sample=far_m, far_ff_sample_ci95=far_h, far_ff_run=farr_m, far_ff_run_ci95=farr_h)
        rows.append(r)
    s = pd.DataFrame(rows).sort_values("fault").reset_index(drop=True)
    # overall = macro average over faults, per seed, then CI across seeds
    r = {"fault": 0, "n_seeds": per_seed.seed.nunique()}
    for c in METRIC_COLS:
        per = per_seed.groupby("seed")[c].mean()
        r[c], r[c + "_ci95"] = mean_ci(per)
    r.update(far_ff_sample=far_m, far_ff_sample_ci95=far_h, far_ff_run=farr_m, far_ff_run_ci95=farr_h)
    return pd.concat([s, pd.DataFrame([r])], ignore_index=True)


def save_model_results(name, rows, ffrows, extra=None):
    """Write results/models/<name>/{per_seed_fault,fault_free,summary}.csv (+extra.json)."""
    import json
    from .config import RESULTS
    d = RESULTS / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    per, ff = pd.DataFrame(rows), pd.DataFrame(ffrows)
    per.to_csv(d / "per_seed_fault.csv", index=False)
    ff.to_csv(d / "fault_free.csv", index=False)
    summ = summarise(per, ff)
    summ.to_csv(d / "summary.csv", index=False)
    if extra:
        json.dump(extra, open(d / "extra.json", "w"), indent=1, default=float)
    return summ


def split_runs(rng, n, val_frac):
    p = rng.permutation(n)
    k = int(round((1 - val_frac) * n))
    return np.sort(p[:k]), np.sort(p[k:])
