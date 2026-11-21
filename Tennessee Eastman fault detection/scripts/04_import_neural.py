"""Import the Kaggle-trained LSTM-AE results (kaggle/outputs/) into results/models/LSTM-AE/ in the common format."""
import json, shutil, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import ROOT, RESULTS
from tepfdd.metrics import save_model_results

src = ROOT / "kaggle" / "outputs"
per = pd.read_csv(src / "neural_per_seed_fault_metrics.csv")
ff = pd.read_csv(src / "neural_per_seed_fault_free.csv")
per.insert(1, "model", "LSTM-AE"); ff.insert(1, "model", "LSTM-AE")
extra = {"leakage_checks": json.load(open(src / "neural_leakage_checks.json")),
         "provenance": json.load(open(src / "neural_provenance.json")), "trained_on": "Kaggle GPU notebook kaggle/tep_neural_kaggle.ipynb"}
s = save_model_results("LSTM-AE", per.to_dict("records"), ff.to_dict("records"), extra)
(RESULTS / "scores").mkdir(parents=True, exist_ok=True)
shutil.copy(src / "neural_example_scores_seed0.npz", RESULTS / "scores" / "lstm_seed0.npz")
print(s[["fault", "fdr_sample", "det_run", "delay_min", "far_ff_sample", "diag_acc"]].round(3).to_string(index=False))
