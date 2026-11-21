"""Is XGBoost's above-baseline detection of fault 3 (and 5, 19) real signal or a run-level artifact?

Binary XGBoost (normal vs fault f), trained ONLY on the official training files (fault-free train windows vs windows fully after
onset in faulty train runs; fixed hyper-parameters, no tuning). Evaluated on the TEST files with two AUCs:
  AUC_vs_faultfree : fault-free test windows vs post-onset windows of faulty test runs          (what a detector is asked to do)
  AUC_within_run   : PRE-onset windows vs POST-onset windows of the SAME faulty test runs        (controls for any run-level nuisance:
                     same runs, same simulator seed, only the fault switches on)
If the signal were a nuisance difference between fault-free and faulty runs, AUC_within_run would be ~0.5 while AUC_vs_faultfree is high.
Also reports the detection rate at the false-alarm level of the main experiment (threshold = 99th pct on held-out fault-free train runs).
"""
import json, sys
from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import *
from tepfdd.xgb_model import window_features
from tepfdd.metrics import split_runs, consecutive

W = WINDOW
ff_tr = np.load(PROC / "ff_train.npy", mmap_mode="r"); ff_te = np.load(PROC / "ff_test.npy", mmap_mode="r")
fl_tr = np.load(PROC / "fl_train.npy", mmap_mode="r"); fl_te = np.load(PROC / "fl_test.npy", mmap_mode="r")
names = [f"{s}:{v}" for s in ("mean", "std", "last", "trend") for v in FEATS]
rng = np.random.default_rng(MASTER_SEED + 999)
tr_r, va_r = split_runs(rng, 500, VAL_FRAC)
fe_ff = window_features(np.asarray(ff_tr), W)
Xn = fe_ff[np.repeat(tr_r, 80), rng.integers(0, fe_ff.shape[1], 80 * len(tr_r))]
Xn_va = fe_ff[va_r].reshape(-1, 4 * NVAR)
fe_fft = window_features(np.asarray(ff_te), W)
res = {}
for f in (3, 9, 15, 5, 19, 1):
    fe = window_features(np.asarray(fl_tr[f - 1]), W)
    Xf = fe[np.repeat(tr_r, 40), rng.integers(ONSET_TRAIN + 1, fe.shape[1], 40 * len(tr_r))]   # window start >= onset+1: fully post-onset
    X = np.concatenate([Xn, Xf]); y = np.r_[np.zeros(len(Xn)), np.ones(len(Xf))]
    b = xgb.train(dict(objective="binary:logistic", tree_method="hist", max_depth=5, learning_rate=0.1, subsample=.8, colsample_bytree=.6,
                       n_jobs=-1, seed=MASTER_SEED), xgb.DMatrix(X, label=y), 200)
    pr = lambda A: b.predict(xgb.DMatrix(A.reshape(-1, 4 * NVAR)))
    fte = window_features(np.asarray(fl_te[f - 1]), W)                # (500, 941, d); window i ends at sample i+W-1
    pre = fte[:, :ONSET_TEST - W + 1:3]                               # windows fully before onset (end < 160)
    post = fte[:, ONSET_TEST::3]                                      # windows starting at/after onset
    ffw = fe_fft[:, ::3]
    p_pre, p_post, p_ff = pr(pre), pr(post), pr(ffw)
    thr = np.quantile(pr(Xn_va), THR_Q)
    res[f] = dict(AUC_vs_faultfree=float(roc_auc_score(np.r_[np.zeros(len(p_ff)), np.ones(len(p_post))], np.r_[p_ff, p_post])),
                  AUC_within_run=float(roc_auc_score(np.r_[np.zeros(len(p_pre)), np.ones(len(p_post))], np.r_[p_pre, p_post])),
                  frac_post_above_thr=float((p_post > thr).mean()), frac_faultfree_above_thr=float((p_ff > thr).mean()),
                  frac_pre_onset_above_thr=float((p_pre > thr).mean()),
                  top_features=[(names[int(k[1:])], round(v, 1)) for k, v in sorted(b.get_score(importance_type="gain").items(), key=lambda t: -t[1])[:6]])
    print(f, json.dumps(res[f]), flush=True)
json.dump(res, open(RESULTS / "hard_fault_check.json", "w"), indent=1)
