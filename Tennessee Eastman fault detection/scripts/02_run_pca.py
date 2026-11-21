"""PCA baseline (T^2 + SPE) with two kinds of control limits, evaluated on the official test runs.

PCA            : limits = 99th percentile of T^2 / SPE on held-out FAULT-FREE validation runs (split by run, from the training file)
PCA-theory     : textbook limits (F-distribution for T^2, Jackson-Mudholkar for SPE), 99% level, from training normal data
Alarm rule for both: (T^2 > limit OR SPE > limit) for K=3 consecutive samples.
Diagnosis: contribution vectors (T^2 and SPE) matched to fault signature centroids built from faulty TRAINING runs.
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import *
from tepfdd.pca_model import PCAMonitor, CentroidLibrary
from tepfdd.metrics import consecutive, fault_metrics, fault_free_metrics, save_model_results, split_runs

ff_tr = np.load(PROC / "ff_train.npy", mmap_mode="r"); ff_te = np.load(PROC / "ff_test.npy", mmap_mode="r")
fl_tr = np.load(PROC / "fl_train.npy", mmap_mode="r"); fl_te = np.load(PROC / "fl_test.npy", mmap_mode="r")
VARIANTS = ["PCA", "PCA-theory"]
rows = {v: [] for v in VARIANTS}; ffrows = {v: [] for v in VARIANTS}; extra = {v: {} for v in VARIANTS}
(RESULTS / "scores").mkdir(parents=True, exist_ok=True)


def score_block(mon, lib, X):                 # X (R, T, 52) -> t2, spe, pred each (R, T)
    R, T, _ = X.shape
    F, t2, spe = mon.contribution_features(np.asarray(X, np.float32).reshape(-1, NVAR).astype(np.float64))
    return t2.reshape(R, T), spe.reshape(R, T), lib.predict(F).reshape(R, T)


for seed in SEEDS:
    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED + seed)
    tr_r, va_r = split_runs(rng, ff_tr.shape[0], VAL_FRAC)
    sig_r, _ = split_runs(rng, fl_tr.shape[1], VAL_FRAC)
    mon = PCAMonitor(0.90).fit(np.asarray(ff_tr[tr_r], np.float64).reshape(-1, NVAR))
    t2v, spev, _, _ = mon.stats(np.asarray(ff_tr[va_r], np.float64).reshape(-1, NVAR))
    lims = {"PCA": (np.quantile(t2v, THR_Q), np.quantile(spev, THR_Q)), "PCA-theory": mon.theory_limits(THR_Q)}
    # fault signature library from faulty TRAINING runs (train-split runs only), post-onset samples
    lib = CentroidLibrary()
    F0, _, _ = mon.contribution_features(np.asarray(ff_tr[tr_r][:, ::5], np.float64).reshape(-1, NVAR))
    lib.add(0, F0)
    for f in FAULTS:
        Xf = np.asarray(fl_tr[f - 1, sig_r, ONSET_TRAIN:], np.float64).reshape(-1, NVAR)
        lib.add(f, mon.contribution_features(Xf)[0])
    lib.finalize()
    # ---- evaluation on official test runs (limits and library are frozen) ----
    t2, spe, pred = score_block(mon, lib, ff_te)
    for v in VARIANTS:
        lt, ls = lims[v]
        a = consecutive((t2 > lt) | (spe > ls))
        ffrows[v].append(dict(seed=seed, model=v, lim_t2=lt, lim_spe=ls, n_pc=mon.k, **fault_free_metrics(a, pred)))
    if seed == SEEDS[0]:
        keep = {"ff_t2": t2[:3], "ff_spe": spe[:3], "k": mon.k, "lib_C": lib.C, "lib_labels": lib.labels}
        for v in VARIANTS: keep[f"lim_{v}"] = np.array(lims[v])
    for f in FAULTS:
        t2, spe, pred = score_block(mon, lib, fl_te[f - 1])
        for v in VARIANTS:
            lt, ls = lims[v]
            a = consecutive((t2 > lt) | (spe > ls))
            rows[v].append(dict(seed=seed, model=v, fault=f, **fault_metrics(a, pred, f, diag_from=ONSET_TEST)))
        if seed == SEEDS[0] and f in (1, 2, 4, 5, 6, 7, 10, 13, 3, 9, 15):
            keep[f"f{f}_t2"], keep[f"f{f}_spe"] = t2[:3], spe[:3]
    if seed == SEEDS[0]:
        np.savez_compressed(RESULTS / "scores" / "pca_seed0.npz", **keep)
    print(f"seed {seed}: k={mon.k} PCs, emp limits T2={lims['PCA'][0]:.2f} SPE={lims['PCA'][1]:.2f}; "
          f"theory T2={lims['PCA-theory'][0]:.2f} SPE={lims['PCA-theory'][1]:.2f}; {time.time()-t0:.0f}s", flush=True)

for v in VARIANTS:
    s = save_model_results(v, rows[v], ffrows[v], extra={"cpv": 0.90, "window": 1, "K_CONSEC": K_CONSEC, "THR_Q": THR_Q})
    print(f"\n== {v} ==")
    print(s[["fault", "fdr_sample", "det_run", "delay_min", "far_ff_sample", "diag_acc"]].round(3).to_string(index=False))
