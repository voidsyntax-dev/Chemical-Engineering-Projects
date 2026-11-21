"""Convert the official Rieth et al. (2017) .RData files into float32 .npy arrays (no resplitting, no other change).

Arrays written to data/processed/:
  ff_train (500, 500, 52)       fault-free training runs   [run, sample, variable]
  ff_test  (500, 960, 52)       fault-free testing runs
  fl_train (20, 500, 500, 52)   faulty training runs        [fault-1, run, sample, variable]
  fl_test  (20, 500, 960, 52)   faulty testing runs
Only operations: sort by (fault, run, sample), keep the 52 process columns, cast float64 -> float32.
The columns faultNumber / simulationRun / sample are used for ordering and then dropped (never a model input).
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pyreadr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tepfdd.config import RAW, PROC, FEATS, FAULTS, DATAVERSE

PROC.mkdir(parents=True, exist_ok=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def to_array(df, faults, T):
    df = df[["faultNumber", "simulationRun", "sample"] + FEATS].sort_values(
        ["faultNumber", "simulationRun", "sample"], kind="stable")
    assert sorted(int(f) for f in df.faultNumber.unique()) == faults
    runs = df.simulationRun.nunique()
    assert df["sample"].nunique() == T and runs == 500, (runs, df["sample"].nunique())
    X = df[FEATS].to_numpy(np.float32)
    assert X.shape[0] == len(faults) * runs * T
    return X.reshape(len(faults), runs, T, len(FEATS))


manifest = {"dataverse": DATAVERSE, "arrays": {}, "preprocessing": [
    "sort by (faultNumber, simulationRun, sample)", "keep xmeas_1..41 + xmv_1..11 (52 columns)", "float64 -> float32",
    "NO resplitting: official training/testing files kept separate", "NO scaling here (scalers are fit on fault-free TRAIN only, inside each model script)"]}
for name, md in DATAVERSE["files"].items():
    got = md5(RAW / name)
    assert got == md, f"MD5 mismatch {name}: {got}"
    manifest.setdefault("md5_verified", {})[name] = got
spec = {"ff_train": ("TEP_FaultFree_Training.RData", [0], 500), "ff_test": ("TEP_FaultFree_Testing.RData", [0], 960),
        "fl_train": ("TEP_Faulty_Training.RData", FAULTS, 500), "fl_test": ("TEP_Faulty_Testing.RData", FAULTS, 960)}
for key, (fname, faults, T) in spec.items():
    t0 = time.time()
    df = next(iter(pyreadr.read_r(str(RAW / fname)).values()))
    X = to_array(df, faults, T); del df
    if key.startswith("ff"):
        X = X[0]
    assert np.isfinite(X).all(), key
    np.save(PROC / f"{key}.npy", X)
    manifest["arrays"][key] = {"shape": list(X.shape), "dtype": "float32", "seconds": round(time.time() - t0)}
    print(key, X.shape, f"{time.time()-t0:.0f}s", flush=True); del X
json.dump(manifest, open(PROC / "manifest.json", "w"), indent=1)
print("done")
