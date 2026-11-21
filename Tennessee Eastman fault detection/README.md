# Tennessee Eastman Process - fault detection and diagnosis

PCA (T² / SPE), XGBoost on windowed features, and an LSTM autoencoder, evaluated on the official
Rieth et al. (2017) train/test split with a fixed alarm rule, 5 seeds, and no tuning on test data.

**Full results, per-fault tables, plots, leakage checks, published-results comparison and limitations: [results/REPORT.md](results/REPORT.md).**

| mean over 20 faults | samples in alarm | runs detected | false alarms (samples) | diagnosis acc. |
|---|---|---|---|---|
| PCA | 62.6% | 89.7% | 0.06% | 64.3% |
| LSTM autoencoder (Kaggle) | 74.6% | 95.6% | 0.66% | 81.8% |
| XGBoost | 82.7% | 98.0% | 0.20% | 86.5% |

The 99% target is missed on 14 of 20 faults. Faults 3, 9 and 15 stay far below it for every model (best: XGBoost, 10.6% / 3.1% / 0.5%
of post-onset samples). See the report for the caveats before quoting any number.

## Layout
| path | purpose |
|---|---|
| `src/tepfdd/` | config (constants, seeds, checksums), metrics, PCA monitor, XGBoost window features, published numbers |
| `scripts/01..08_*.py` | pipeline, run in order (below) |
| `kaggle/tep_neural_kaggle.ipynb` | LSTM-AE + diagnosis head for a Kaggle GPU (source `notebook_src.py`, built by `build_notebook.py`); downloaded outputs in `kaggle/outputs/` |
| `sim/` | Downs & Vogel Fortran build script, Python wrapper, operating-point check |
| `vendor/tep2py-master/` | upstream Fortran + f2py signature (MIT / University of Illinois licences, see `THIRD_PARTY_NOTICES.md`) |
| `results/` | per-model CSVs, scores, figures, `REPORT.md`, leakage and diagnostic JSON |
| `tests/` | unit tests (`python -m pytest`) |
| `data/` | **not committed** (4.6 GB): official `.RData` files and the float32 `.npy` cache |

## Reproduce
Python 3.11 (3.15 is too new for the torch/xgboost wheels). Needs about 16 GB RAM for the data preparation step.
```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# 1. download the 4 .RData files of doi:10.7910/DVN/6C3JR1 (v1.0) into data\raw\  (MD5s: src/tepfdd/config.py)
.venv\Scripts\python scripts\01_prepare_data.py
.venv\Scripts\python scripts\02_run_pca.py
.venv\Scripts\python scripts\03_run_xgb.py          # about 45 min on a 22-thread CPU
# 2. run kaggle/tep_neural_kaggle.ipynb on Kaggle (GPU, Internet on), unzip tep_neural_outputs.zip into kaggle\outputs\
.venv\Scripts\python scripts\04_import_neural.py
.venv\Scripts\python scripts\05_leakage_checks.py
.venv\Scripts\python scripts\08_hard_fault_check.py
.venv\Scripts\python scripts\06_figures.py
.venv\Scripts\python scripts\07_report.py
```
The committed `results/` already contains the outputs of this pipeline, so the report can be read without re-running anything.

Simulator check (optional): install a gfortran toolchain (`winget install BrechtSanders.WinLibs.POSIX.UCRT`), then
```powershell
powershell -File sim\build_ext.ps1
.venv\Scripts\python sim\verify_operating_point.py   # worst error vs Downs & Vogel base case: 0.24%
```
The simulator is used only for verification; every model is trained and evaluated on the official Rieth files.

## Reproducibility notes
- Master seed `20240607`; run seeds 0-4 (`src/tepfdd/config.py`). Test data are never used for thresholds, early stopping or model selection.
- The Kaggle notebook sets torch seeds, but GPU kernels are not bit-wise deterministic, so a re-run can differ slightly.
- Two analyses were added after seeing the main results (the within-run diagnostic for faults 3/5/19 and the "meaningfully above baseline"
  criterion); they are labelled as post-hoc in the report and change no model or metric.

