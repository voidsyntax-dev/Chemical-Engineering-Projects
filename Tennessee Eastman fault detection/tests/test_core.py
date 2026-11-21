"""Fast unit tests for the pieces whose correctness the results depend on (no data files needed)."""
import numpy as np
import pandas as pd

from tepfdd.config import ONSET_TEST, SAMPLE_MIN, T_TEST, NVAR
from tepfdd.metrics import consecutive, fault_metrics, fault_free_metrics, mean_ci, split_runs, summarise
from tepfdd.pca_model import PCAMonitor, CentroidLibrary
from tepfdd.xgb_model import window_features


def test_consecutive_requires_k_in_a_row():
    raw = np.array([[0, 1, 1, 1, 0, 1, 1]], bool)
    out = consecutive(raw, 3)
    assert out.tolist() == [[False, False, False, True, False, False, False]]


def test_fault_metrics_known_case():
    a = np.zeros((2, T_TEST), bool)
    a[0, ONSET_TEST + 10:] = True                       # run 0 alarms 10 samples after onset, stays on
    pred = np.full((2, T_TEST), 7, np.int8)
    m = fault_metrics(a, pred, fault=7, diag_from=ONSET_TEST)
    n_post = T_TEST - ONSET_TEST
    assert m["det_run"] == 0.5
    assert np.isclose(m["fdr_sample"], 0.5 * (n_post - 10) / n_post)
    assert np.isclose(m["delay_min"], 10 * SAMPLE_MIN)
    assert m["far_pre_onset"] == 0.0 and m["diag_acc"] == 1.0
    ff = fault_free_metrics(a, np.zeros_like(pred))
    assert ff["normal_pred_acc"] == 1.0 and ff["far_run"] == 0.5


def test_window_features_match_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 60, 4))
    W = 10
    F = window_features(X, W)
    assert F.shape == (3, 60 - W + 1, 16)
    r, i = 2, 17                                        # window covers samples i .. i+W-1
    w = X[r, i:i + W]
    expected = np.concatenate([w.mean(0), w.std(0), w[-1], w[-1] - w[0]])
    assert np.allclose(F[r, i], expected, atol=1e-5)


def test_pca_monitor_behaves_on_gaussian_data():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(NVAR, NVAR))
    X = rng.normal(size=(20000, NVAR)) @ A
    mon = PCAMonitor(0.90).fit(X)
    t2, spe, _, _ = mon.stats(X)
    assert 1 <= mon.k < NVAR
    assert abs(t2.mean() - mon.k) / mon.k < 0.05       # E[T^2] = k for in-control data
    lt, ls = mon.theory_limits(0.99)
    assert (t2 > lt).mean() < 0.03 and lt > mon.k
    F, _, _ = mon.contribution_features(X[:50])
    assert np.allclose(F[:, :NVAR].sum(1), 1, atol=1e-4) and np.allclose(F[:, NVAR:].sum(1), 1, atol=1e-4)


def test_centroid_library_recovers_signature():
    rng = np.random.default_rng(2)
    lib = CentroidLibrary()
    for lab in (0, 1, 2):
        base = np.zeros(8); base[lab] = 1.0
        lib.add(lab, np.abs(base + 0.05 * rng.normal(size=(100, 8))).astype(np.float32))
    lib.finalize()
    probe = np.zeros((1, 8), np.float32); probe[0, 2] = 1.0
    assert lib.predict(probe)[0] == 2


def test_split_runs_is_disjoint_and_complete():
    tr, va = split_runs(np.random.default_rng(0), 500, 0.2)
    assert len(tr) == 400 and len(va) == 100 and not set(tr) & set(va) and set(tr) | set(va) == set(range(500))


def test_mean_ci_and_summarise():
    m, h = mean_ci([1.0, 1.0, 1.0])
    assert m == 1.0 and h == 0.0
    rows = [dict(seed=s, fault=f, fdr_sample=0.5, det_run=1.0, delay_min=3.0, far_pre_onset=0.0, diag_acc=0.9, diag_acc_alarmed=0.9)
            for s in range(3) for f in (1, 2)]
    ff = pd.DataFrame([dict(seed=s, far_sample=0.01, far_run=0.3) for s in range(3)])
    out = summarise(pd.DataFrame(rows), ff)
    assert list(out.fault) == [1, 2, 0] and np.isclose(out.loc[out.fault == 0, "fdr_sample"].iloc[0], 0.5)
