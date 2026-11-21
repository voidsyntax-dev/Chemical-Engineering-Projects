"""Gradient-boosted classifier on windowed features: detection AND diagnosis.

Classes: 0 = normal, 1..20 = fault id. Training windows: fault-free TRAINING runs (label 0) and windows lying FULLY after the
onset in faulty TRAINING runs (label = fault). Runs are split 80/20 by run (seed-dependent); the 20% validation runs are used
for early stopping and for the alarm threshold only. Detection score = 1 - P(normal); threshold = 99th percentile of that score
on fault-free validation runs; alarm = K=3 consecutive exceedances. Diagnosis = argmax over fault classes of the same model.
"""
import sys, json, time
from pathlib import Path
import numpy as np, xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import *
from tepfdd.xgb_model import window_features
from tepfdd.metrics import consecutive, fault_metrics, fault_free_metrics, save_model_results, split_runs

W = WINDOW
ff_tr = np.load(PROC / "ff_train.npy", mmap_mode="r"); ff_te = np.load(PROC / "ff_test.npy", mmap_mode="r")
fl_tr = np.load(PROC / "fl_train.npy", mmap_mode="r"); fl_te = np.load(PROC / "fl_test.npy", mmap_mode="r")
mu = np.asarray(ff_tr, np.float64).reshape(-1, NVAR).mean(0)       # scaling is irrelevant for trees; kept for parity/inspection
SMOKE = "--smoke" in sys.argv
N_PER_FF, N_PER_FL = (50, 16) if not SMOKE else (10, 4)
PARAMS = dict(objective="multi:softprob", num_class=len(FAULTS) + 1, tree_method="hist", max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.6, min_child_weight=3, n_jobs=-1, eval_metric="mlogloss")
ROUNDS = 400 if not SMOKE else 20


def sample_windows(rng, feats_run, runs, lo, n):
    """feats_run: (R, Tw, d) features indexed by window-end t = i + W - 1. Draw n random windows per run with end >= lo."""
    first = lo - (W - 1)
    r = np.repeat(runs, n)
    i = rng.integers(first, feats_run.shape[1], size=len(r))
    return feats_run[r, i]


def build_set(rng, ff_runs, fl_runs):
    Xs, ys = [], []
    Xs.append(sample_windows(rng, window_features(ff_tr[:], W), ff_runs, W - 1, N_PER_FF)); ys.append(np.zeros(len(Xs[-1]), int))
    for f in FAULTS:
        fe = window_features(np.asarray(fl_tr[f - 1]), W)
        Xs.append(sample_windows(rng, fe, fl_runs, ONSET_TRAIN + W - 1, N_PER_FL)); ys.append(np.full(len(Xs[-1]), f))
    return np.concatenate(Xs), np.concatenate(ys)


def predict_run_block(booster, X):
    """X (R, T, 52) -> score (R, T), pred (R, T) in sample-index space (first W-1 samples: no score)."""
    R, T, _ = X.shape
    fe = window_features(X, W).reshape(-1, 4 * NVAR)
    P = booster.predict(xgb.DMatrix(fe))
    sc = np.zeros((R, T), np.float32); pr = np.zeros((R, T), np.int8)
    sc[:, W - 1:] = (1 - P[:, 0]).reshape(R, T - W + 1)
    pr[:, W - 1:] = P.argmax(1).reshape(R, T - W + 1)
    return sc, pr


def train(seed, tr_r, va_r, fl_tr_r, fl_va_r, shuffle=False):
    rng = np.random.default_rng(MASTER_SEED + seed + 77)
    Xt, yt = build_set(rng, tr_r, fl_tr_r)
    Xv, yv = build_set(rng, va_r, fl_va_r)
    if shuffle:
        yt = rng.permutation(yt)
    dtr, dva = xgb.DMatrix(Xt, label=yt), xgb.DMatrix(Xv, label=yv)
    b = xgb.train({**PARAMS, "seed": MASTER_SEED + seed}, dtr, ROUNDS, evals=[(dva, "val")],
                  early_stopping_rounds=20, verbose_eval=False)
    return b


rows, ffrows, extra = [], [], {}
for seed in (SEEDS[:1] if SMOKE else SEEDS):
    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED + seed)
    tr_r, va_r = split_runs(rng, ff_tr.shape[0], VAL_FRAC)
    fl_tr_r, fl_va_r = split_runs(rng, fl_tr.shape[1], VAL_FRAC)
    b = train(seed, tr_r, va_r, fl_tr_r, fl_va_r)
    sv, _ = predict_run_block(b, np.asarray(ff_tr[va_r]))
    thr = float(np.quantile(sv[:, W - 1:], THR_Q))
    print(f"seed {seed}: trained {b.best_iteration + 1} rounds, thr={thr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    # ---- evaluation on official test runs (model + threshold frozen) ----
    sc, pr = predict_run_block(b, np.asarray(ff_te))
    ffrows.append(dict(seed=seed, model="XGBoost", thr=thr, rounds=b.best_iteration + 1,
                       **fault_free_metrics(consecutive(sc > thr), pr, valid_from=W - 1)))
    keep = {"ff_score": sc[:3], "thr": thr}
    pre_norm, diag_sh = [], []
    for f in FAULTS:
        sc, pr = predict_run_block(b, np.asarray(fl_te[f - 1]))
        rows.append(dict(seed=seed, model="XGBoost", fault=f,
                         **fault_metrics(consecutive(sc > thr), pr, f, diag_from=ONSET_TEST + W - 1, valid_from=W - 1)))
        pre_norm.append(float((pr[:, W - 1:ONSET_TEST - W + 1] == 0).mean()))   # windows fully before onset
        if seed == SEEDS[0]:
            keep[f"f{f}_score"] = sc[:3]
    if seed == SEEDS[0]:
        np.savez_compressed(RESULTS / "scores" / "xgb_seed0.npz", **keep)
        extra["pre_onset_windows_predicted_normal"] = float(np.mean(pre_norm))
        # leakage control: identical pipeline with permuted training labels -> diagnosis must fall to ~chance
        bs = train(seed, tr_r, va_r, fl_tr_r, fl_va_r, shuffle=True)
        accs = []
        for f in FAULTS:
            _, pr = predict_run_block(bs, np.asarray(fl_te[f - 1])[:100])
            accs.append(float((pr[:, ONSET_TEST + W - 1:] == f).mean()))
        extra["shuffled_label_control_diag_acc"] = float(np.mean(accs)); extra["chance_level"] = 1 / (len(FAULTS) + 1)
        imp = b.get_score(importance_type="gain")
        names = [f"{s}:{v}" for s in ("mean", "std", "last", "trend") for v in FEATS]
        extra["top_features_gain"] = sorted(((names[int(k[1:])], float(v)) for k, v in imp.items()), key=lambda t: -t[1])[:15]
    print(f"seed {seed} evaluated ({time.time()-t0:.0f}s)", flush=True)

extra.update(window=W, rounds_max=ROUNDS, params={k: v for k, v in PARAMS.items()}, K_CONSEC=K_CONSEC, THR_Q=THR_Q)
if not SMOKE:
    s = save_model_results("XGBoost", rows, ffrows, extra)
    print(s[["fault", "fdr_sample", "det_run", "delay_min", "far_ff_sample", "diag_acc"]].round(3).to_string(index=False))
else:
    print(json.dumps(rows[:3], indent=1, default=float))
