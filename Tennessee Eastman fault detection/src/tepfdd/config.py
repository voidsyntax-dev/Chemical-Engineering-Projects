"""Project-wide constants. Everything that affects results lives here so runs are reproducible."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

MASTER_SEED = 20240607           # base seed; run seeds are MASTER_SEED + k
SEEDS = [0, 1, 2, 3, 4]          # per-seed offsets (also used by the Kaggle notebook)

FEATS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
NVAR = len(FEATS)                # 52
FAULTS = list(range(1, 21))      # Rieth et al. provides faults 1-20

ONSET_TRAIN = 20                 # 0-based index of first faulty sample in training runs (1 h)
ONSET_TEST = 160                 # ... in testing runs (8 h)
T_TRAIN, T_TEST = 500, 960
SAMPLE_MIN = 3.0                 # minutes per sample

WINDOW = 20                      # window length for window-based models
K_CONSEC = 3                     # alarm = K consecutive exceedances (fixed a priori)
THR_Q = 0.99                     # alarm threshold quantile on held-out FAULT-FREE validation runs
VAL_FRAC = 0.2                   # fraction of runs held out (split by run)

DATAVERSE = {
    "doi": "10.7910/DVN/6C3JR1", "version": "1.0 (released 2017-07-06)",
    "files": {
        "TEP_FaultFree_Training.RData": "ec126484534331f85001d8c4ebce6d17",
        "TEP_FaultFree_Testing.RData": "38ad9810fc871026157086ae2c2f0ee9",
        "TEP_Faulty_Training.RData": "c5f594d54c47e620ff877feb58407fda",
        "TEP_Faulty_Testing.RData": "556bdb64c83021bc0c5f92e427753565",
    },
}
