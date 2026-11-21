# %% [markdown]
# # TEP neural detector + diagnosis (LSTM autoencoder) — Kaggle notebook
#
# Part of the Tennessee Eastman fault detection project. The CPU models (PCA, XGBoost) run locally; this notebook trains the **LSTM autoencoder detector** and an **MLP diagnosis classifier** on the autoencoder's features, then writes result files to `/kaggle/working` for download.
#
# **Before running:** Settings -> Accelerator = GPU (P100/T4), Internet = On (needed to download the Rieth dataset from Harvard Dataverse, unless you attached it as an input).
#
# **Protocol (no test-set tuning):**
# * Official Rieth et al. (2017) train/test split. Test files are used only for final evaluation.
# * AE trains on fault-free *training* runs only; a held-out set of fault-free training runs (split by run) gives the alarm threshold (99th percentile) and early stopping.
# * Alarm rule (fixed a priori, not tuned): window score > threshold for **3 consecutive** windows.
# * Diagnosis classifier trains on faulty *training* runs (train/val split by run); evaluated on the faulty *test* runs.
# * 5 seeds; each seed redraws the run splits and re-initialises the networks. CI = t-interval across seeds.
# * Fault onset: sample 20 (1 h) in training runs, sample 160 (8 h) in testing runs; sampling interval 3 min.

# %%
import os, sys, json, glob, hashlib, time, platform, subprocess, shutil
import numpy as np, pandas as pd, torch, torch.nn as nn
try:
    import pyreadr
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyreadr"], check=True)
    import pyreadr
import requests

SMOKE = os.environ.get("TEP_SMOKE") == "1"        # tiny synthetic run to test the code path
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
W, K_CONSEC, THR_Q = 20, 3, 0.99                   # window length, consecutive-alarm rule, threshold quantile
HID, LATENT, BS = 64, 16, 512
EP_AE, EP_CLF = (1, 2) if SMOKE else (20, 30)
ONSET_TRAIN, ONSET_TEST, SAMPLE_MIN = 20, 160, 3.0
FEATS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
NF_FEAT = len(FEATS)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

# %% [markdown]
# ## 1. Data (Rieth et al. 2017, Harvard Dataverse doi:10.7910/DVN/6C3JR1)
# Looks in `/kaggle/input` first; otherwise downloads from Dataverse and verifies the MD5 that Dataverse reports. Provenance goes to `provenance.json`.

# %%
DOI = "doi:10.7910/DVN/6C3JR1"
NAMES = ["TEP_FaultFree_Training.RData", "TEP_FaultFree_Testing.RData",
         "TEP_Faulty_Training.RData", "TEP_Faulty_Testing.RData"]
prov = {"doi": DOI, "files": {}, "torch": torch.__version__, "pandas": pd.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "smoke": SMOKE}

def locate_files():
    paths, meta = {}, None
    for n in NAMES:
        hits = glob.glob(f"/kaggle/input/**/{n}", recursive=True)
        if hits:
            paths[n] = hits[0]
    missing = [n for n in NAMES if n not in paths]
    if missing:
        # File ids/MD5s as listed by the Dataverse API for dataset version 1.0 (released 2017-07-06).
        # Hard-coded because the metadata endpoint can return a non-JSON page to cloud IPs.
        KNOWN = {"TEP_FaultFree_Testing.RData": (3031240, "38ad9810fc871026157086ae2c2f0ee9"),
                 "TEP_FaultFree_Training.RData": (3031241, "ec126484534331f85001d8c4ebce6d17"),
                 "TEP_Faulty_Testing.RData": (3031243, "556bdb64c83021bc0c5f92e427753565"),
                 "TEP_Faulty_Training.RData": (3031242, "c5f594d54c47e620ff877feb58407fda")}
        prov["dataverse_version"] = "1.0"
        hdr = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-download"}
        for n in missing:
            fid, md5 = KNOWN[n]; dst = f"/tmp/{n}"
            for attempt in range(3):
                print("downloading", n, "attempt", attempt + 1)
                h = hashlib.md5()
                try:
                    with requests.get(f"https://dataverse.harvard.edu/api/access/datafile/{fid}",
                                      headers=hdr, stream=True, timeout=120) as resp:
                        resp.raise_for_status()
                        with open(dst, "wb") as fh:
                            for chunk in resp.iter_content(1 << 20):
                                fh.write(chunk); h.update(chunk)
                    if h.hexdigest() == md5: break
                    print("  MD5 mismatch, retrying")
                except Exception as ex:
                    print("  download error:", ex)
            else:
                raise RuntimeError(f"Could not fetch {n} from Dataverse. Turn Internet ON in the notebook settings, "
                                   "or attach the four .RData files as a Kaggle dataset (Add Input) and re-run.")
            prov["files"][n] = {"md5": md5, "bytes": os.path.getsize(dst), "source": "dataverse"}
            paths[n] = dst
    for n in NAMES:
        prov["files"].setdefault(n, {"source": "kaggle_input", "path": paths[n], "bytes": os.path.getsize(paths[n])})
    return paths

def to_arrays(df):
    df = df[["faultNumber", "simulationRun", "sample"] + FEATS]
    df = df.sort_values(["faultNumber", "simulationRun", "sample"], kind="stable")
    faults = sorted(int(f) for f in df["faultNumber"].unique())
    runs = df["simulationRun"].nunique(); T = df["sample"].nunique()
    X = df[FEATS].to_numpy(np.float32)
    assert X.shape[0] == len(faults) * runs * T, "unexpected row count"
    return X.reshape(len(faults), runs, T, NF_FEAT), faults

def make_synth(nf, runs, T, onset, seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((nf, runs, T, NF_FEAT)).astype(np.float32)
    X[..., 1:, :] += 0.5 * X[..., :-1, :]
    for j in range(nf):
        X[j, :, onset:, 3 * j:3 * j + 3] += 3.0
    return X

if SMOKE:
    FAULT_IDS = [1, 2, 3]
    ff_tr = make_synth(1, 40, 500, 10 ** 9, 1)[0]; ff_te = make_synth(1, 20, 960, 10 ** 9, 2)[0]
    fl_tr = make_synth(3, 40, 500, ONSET_TRAIN, 3); fl_te = make_synth(3, 10, 960, ONSET_TEST, 4)
else:
    paths = locate_files()
    loaded = {}
    for n in NAMES:
        t0 = time.time()
        df = next(iter(pyreadr.read_r(paths[n]).values()))
        print(n, df.shape, f"{time.time()-t0:.0f}s")
        X, faults = to_arrays(df); del df
        loaded[n] = (X, faults)
    ff_tr, ff_te = loaded[NAMES[0]][0][0], loaded[NAMES[1]][0][0]
    fl_tr, fl_te = loaded[NAMES[2]][0], loaded[NAMES[3]][0]
    FAULT_IDS = loaded[NAMES[2]][1]
    assert loaded[NAMES[3]][1] == FAULT_IDS
    del loaded
NCLS = max(FAULT_IDS) + 1
print("fault-free train/test:", ff_tr.shape, ff_te.shape, "| faulty train/test:", fl_tr.shape, fl_te.shape, "| faults:", FAULT_IDS)
assert fl_tr.shape[2] == 500 and fl_te.shape[2] == 960
prov["faults"] = FAULT_IDS; prov["shapes"] = {"ff_tr": ff_tr.shape, "ff_te": ff_te.shape, "fl_tr": fl_tr.shape, "fl_te": fl_te.shape}

# %% [markdown]
# ## 2. Leakage groundwork + scaling
# Run fingerprints (hash of each run's first 5 rows) are computed on the raw arrays so we can prove no run appears in both train and test. The scaler is fit on the 500 fault-free **training** runs only.

# %%
def fingerprints(X):
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    return {hashlib.md5(np.ascontiguousarray(x[:5]).tobytes()).hexdigest() for x in X}
FP_TRAIN = fingerprints(ff_tr) | fingerprints(fl_tr)
FP_TEST = fingerprints(ff_te) | fingerprints(fl_te)
RUN_OVERLAP = len(FP_TRAIN & FP_TEST)
print("train/test run-fingerprint overlap:", RUN_OVERLAP, "(must be 0)")

mu = ff_tr.reshape(-1, NF_FEAT).mean(0); sd = ff_tr.reshape(-1, NF_FEAT).std(0)
sd = np.maximum(sd, 1e-3)
def prep(X):
    return torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)
XFF_TR, XFF_TE = prep(ff_tr), prep(ff_te)
XFL_TR = [prep(fl_tr[j]) for j in range(len(FAULT_IDS))]
XFL_TE = [prep(fl_te[j]) for j in range(len(FAULT_IDS))]
del ff_tr, ff_te, fl_tr, fl_te
ARW = torch.arange(W, device=dev)
def gather(X, r, e):
    return X[r[:, None], e[:, None] - (W - 1) + ARW[None, :]]

# %% [markdown]
# ## 3. Models

# %%
class LSTMAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.LSTM(NF_FEAT, HID, batch_first=True); s.to_z = nn.Linear(HID, LATENT)
        s.from_z = nn.Linear(LATENT, HID); s.dec = nn.LSTM(HID, HID, batch_first=True); s.out = nn.Linear(HID, NF_FEAT)
    def forward(s, x):
        _, (h, _) = s.enc(x); z = s.to_z(h[-1])
        d = s.from_z(z).unsqueeze(1).expand(-1, x.shape[1], -1).contiguous()
        o, _ = s.dec(d)
        return s.out(o), z

def make_clf():
    fd = LATENT + NF_FEAT
    return nn.Sequential(nn.BatchNorm1d(fd), nn.Linear(fd, 256), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NCLS)).to(dev)

@torch.no_grad()
def feats(model, win):
    rec, z = model(win); se = (rec - win) ** 2
    return torch.cat([z, se.mean(1).sqrt()], 1), se.mean((1, 2))   # classifier features, window score (SPE-like)

@torch.no_grad()
def run_model(model, clf, X, rows=8192):
    model.eval()
    if clf is not None: clf.eval()
    R, T, F_ = X.shape; Tw = T - W + 1
    sc = torch.empty(R, Tw); pr = torch.zeros(R, Tw, dtype=torch.int8)
    rb = max(1, rows // Tw)
    for i in range(0, R, rb):
        xb = X[i:i + rb]; b = xb.shape[0]
        win = xb.unfold(1, W, 1).permute(0, 1, 3, 2).reshape(b * Tw, W, F_)
        f, s = feats(model, win)
        sc[i:i + b] = s.view(b, Tw).cpu()
        if clf is not None: pr[i:i + b] = clf(f).argmax(1).view(b, Tw).to(torch.int8).cpu()
    return sc.numpy(), pr.numpy()

def train_ae(seed, tr_runs, va_runs):
    torch.manual_seed(seed)
    model = LSTMAE().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
    tr_t = torch.as_tensor(tr_runs, device=dev); ends = torch.arange(W - 1, XFF_TR.shape[1], device=dev)
    best, state, bad = 1e9, None, 0
    for ep in range(EP_AE):
        model.train()
        n = len(tr_t) * len(ends); sel = torch.randperm(n, device=dev)[: n // 2]
        r, e = tr_t[sel // len(ends)], ends[sel % len(ends)]
        for i in range(0, len(sel), BS):
            xb = gather(XFF_TR, r[i:i + BS], e[i:i + BS])
            rec, _ = model(xb); loss = ((rec - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vl = run_model(model, None, XFF_TR[torch.as_tensor(va_runs, device=dev)])[0].mean()
        print(f"  seed {seed} AE epoch {ep+1}/{EP_AE} val_loss {vl:.4f}")
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4: break
    model.load_state_dict(state); model.eval()
    return model

def build_clf_set(model, ff_runs, fl_runs, rng, n_ff=50, n_fl=16):
    F, Y = [], []
    def add(X, runs, lo, n, label):
        r = np.repeat(runs, n); e = rng.integers(lo, X.shape[1], size=len(r))
        rt, et = torch.as_tensor(r, device=dev), torch.as_tensor(e, device=dev)
        for i in range(0, len(r), 8192):
            F.append(feats(model, gather(X, rt[i:i + 8192], et[i:i + 8192]))[0])
        Y.append(np.full(len(r), label))
    add(XFF_TR, ff_runs, W - 1, n_ff, 0)
    for j, f in enumerate(FAULT_IDS):                      # only windows fully AFTER fault onset get the fault label
        add(XFL_TR[j], fl_runs, ONSET_TRAIN + W - 1, n_fl, f)
    return torch.cat(F), torch.as_tensor(np.concatenate(Y), device=dev)

def train_clf(seed, model, ff_tr_runs, ff_va_runs, fl_tr_runs, fl_va_runs, shuffle_labels=False):
    rng = np.random.default_rng(seed + 1000); torch.manual_seed(seed)
    Ft, Yt = build_clf_set(model, ff_tr_runs, fl_tr_runs, rng)
    Fv, Yv = build_clf_set(model, ff_va_runs, fl_va_runs, rng)
    if shuffle_labels: Yt = Yt[torch.randperm(len(Yt), device=dev)]
    clf = make_clf(); opt = torch.optim.Adam(clf.parameters(), 1e-3); best, state = -1, None
    for ep in range(EP_CLF):
        clf.train(); perm = torch.randperm(len(Ft), device=dev)
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            loss = nn.functional.cross_entropy(clf(Ft[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad(): acc = (clf(Fv).argmax(1) == Yv).float().mean().item()
        if acc > best: best, state = acc, {k: v.clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(state); clf.eval()
    print(f"  seed {seed} clf best val acc {best:.4f}{' (SHUFFLED labels control)' if shuffle_labels else ''}")
    return clf

# %% [markdown]
# ## 4. Metrics

# %%
def alarm(sc, thr):
    raw = sc > thr; a = raw.copy()
    for k in range(1, K_CONSEC):
        a[:, k:] &= raw[:, :-k]; a[:, :k] = False
    return a

I0 = ONSET_TEST - (W - 1)       # first window (index into score array) containing a faulty sample
def fault_metrics(sc, pr, f, thr):
    a = alarm(sc, thr); post = a[:, I0:]; det = post.any(1)
    ok = pr[:, ONSET_TEST:] == f; al = a[:, ONSET_TEST:]
    return dict(fdr_sample=post.mean(), det_run=det.mean(),
                delay_min=(post.argmax(1)[det].mean() * SAMPLE_MIN) if det.any() else np.nan,
                far_pre_onset=a[:, :I0].mean(), diag_acc=ok.mean(),
                diag_acc_alarmed=ok[al].mean() if al.any() else np.nan)

def split(rng, n):
    p = rng.permutation(n); k = int(0.8 * n); return p[:k], p[k:]

# %% [markdown]
# ## 5. Train + evaluate over seeds
# Test data is touched only in the evaluation block, after the threshold and both networks are frozen.

# %%
rows, ffrows, ART = [], [], {}
for seed in SEEDS:
    t0 = time.time(); rng = np.random.default_rng(seed)
    ff_tr_r, ff_va_r = split(rng, XFF_TR.shape[0]); fl_tr_r, fl_va_r = split(rng, XFL_TR[0].shape[0])
    model = train_ae(seed, ff_tr_r, ff_va_r)
    thr = float(np.quantile(run_model(model, None, XFF_TR[torch.as_tensor(ff_va_r, device=dev)])[0], THR_Q))
    clf = train_clf(seed, model, ff_tr_r, ff_va_r, fl_tr_r, fl_va_r)
    # ---- evaluation on official TEST runs ----
    sc0, pr0 = run_model(model, clf, XFF_TE); a0 = alarm(sc0, thr)
    ffrows.append(dict(seed=seed, thr=thr, far_sample=a0.mean(), far_run=a0.any(1).mean(), normal_pred_acc=(pr0 == 0).mean()))
    keep = {}
    for j, f in enumerate(FAULT_IDS):
        sc, pr = run_model(model, clf, XFL_TE[j])
        rows.append(dict(seed=seed, fault=f, **fault_metrics(sc, pr, f, thr)))
        if seed == SEEDS[0]: keep[f] = (sc[:5], pr[:5])
    if seed == SEEDS[0]:
        ART = dict(model=model, clf=clf, ff_tr_r=ff_tr_r, ff_va_r=ff_va_r, fl_tr_r=fl_tr_r, fl_va_r=fl_va_r, thr=thr, ff_scores=sc0[:5], fault_scores=keep)
    torch.save({"ae": model.state_dict(), "clf": clf.state_dict(), "thr": thr}, f"{OUT}/neural_models_seed{seed}.pt")
    print(f"seed {seed} done in {time.time()-t0:.0f}s | thr {thr:.4f} | FAR(sample) {a0.mean():.4f}")
res = pd.DataFrame(rows); ffres = pd.DataFrame(ffrows)
res.to_csv(f"{OUT}/neural_per_seed_fault_metrics.csv", index=False); ffres.to_csv(f"{OUT}/neural_per_seed_fault_free.csv", index=False)

# %% [markdown]
# ## 6. Summary table (mean ± 95% t-interval across seeds)

# %%
from scipy import stats
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return v.mean(), h
cols = ["fdr_sample", "det_run", "delay_min", "far_pre_onset", "diag_acc", "diag_acc_alarmed"]
far_m, far_h = ci(ffres.far_sample); farr_m, farr_h = ci(ffres.far_run)
summ = []
for f in FAULT_IDS:
    d = res[res.fault == f]; row = {"fault": f}
    for c in cols:
        m, h = ci(d[c]); row[c] = m; row[c + "_ci95"] = h
    row["far_fault_free_sample"] = far_m; row["far_fault_free_sample_ci95"] = far_h
    summ.append(row)
summ = pd.DataFrame(summ); summ.to_csv(f"{OUT}/neural_summary_by_fault.csv", index=False)
overall = {c: ci(summ[c])[0] for c in cols}; overall.update(far_fault_free_sample=far_m, far_fault_free_run=farr_m, seeds=len(SEEDS))
json.dump(overall, open(f"{OUT}/neural_overall.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(summ[["fault", "fdr_sample", "fdr_sample_ci95", "det_run", "delay_min", "far_fault_free_sample", "diag_acc", "diag_acc_ci95"]].to_string(index=False))
print("\nOVERALL (macro mean over faults):", overall)
print("\nFaults 3, 9, 15 (detection rate vs false-alarm baseline):")
print(summ[summ.fault.isin([3, 9, 15])][["fault", "fdr_sample", "fdr_sample_ci95", "det_run", "far_fault_free_sample", "far_pre_onset"]].to_string(index=False))
print("A detector firing at random with the same alarm rate would show fdr_sample ~= far_fault_free_sample. "
      "A fault is only 'detected above baseline' if fdr_sample minus its CI half-width clearly exceeds the FAR.")

# %% [markdown]
# ## 7. Leakage checks (run for seed 0; any metric > 99% is flagged)

# %%
flag = summ[(summ.fdr_sample > .99) | (summ.det_run > .99) | (summ.diag_acc > .99)].fault.tolist()
print("faults with a metric > 99%:", flag)
chk = {"flagged_faults": flag, "features_used": FEATS,
       "label_columns_in_features": [c for c in FEATS if c in ("faultNumber", "simulationRun", "sample")],
       "train_test_run_fingerprint_overlap": RUN_OVERLAP,
       "scaler_fit_on": "fault-free TRAINING runs only (official train file)",
       "ae_train_val_split": "by simulation run (80/20), validation set used only for early stopping and the alarm threshold",
       "clf_train_val_split": "by simulation run (80/20)",
       "clf_train_windows": "fully after fault onset (start >= sample 20), label=fault id; fault-free windows label 0",
       "diag_eval_windows": "fully after test onset (start >= sample 160) so no window straddles onset"}
m, c = ART["model"], ART["clf"]
# (a) before onset the classifier must say 'normal' -- if it names the fault early, features leak the label
pre_pred = []
for j, f in enumerate(FAULT_IDS):
    _, pr = run_model(m, c, XFL_TE[j]); pre_pred.append((pr[:, :I0] == 0).mean())
chk["pre_onset_windows_predicted_normal_in_faulty_test_runs"] = float(np.mean(pre_pred))
# (b) shuffled-label control: same pipeline, labels permuted -> accuracy must collapse to ~chance
c_sh = train_clf(SEEDS[0], m, ART["ff_tr_r"], ART["ff_va_r"], ART["fl_tr_r"], ART["fl_va_r"], shuffle_labels=True)
acc_sh = np.mean([(run_model(m, c_sh, XFL_TE[j])[1][:, ONSET_TEST:] == f).mean() for j, f in enumerate(FAULT_IDS)])
chk["shuffled_label_control_diag_acc"] = float(acc_sh); chk["chance_level"] = 1.0 / len(FAULT_IDS)
chk["verdict"] = ("no leakage indicators" if RUN_OVERLAP == 0 and not chk["label_columns_in_features"]
                  and chk["pre_onset_windows_predicted_normal_in_faulty_test_runs"] > 0.5 and acc_sh < 0.2 else "INSPECT MANUALLY")
json.dump(chk, open(f"{OUT}/neural_leakage_checks.json", "w"), indent=1)
print(json.dumps(chk, indent=1))

# %% [markdown]
# ## 8. Save artefacts (download `tep_neural_outputs.zip` from the Output tab)

# %%
np.savez_compressed(f"{OUT}/neural_example_scores_seed{SEEDS[0]}.npz", thr=ART["thr"], ff_scores=ART["ff_scores"],
                    **{f"fault{f}_scores": v[0] for f, v in ART["fault_scores"].items()},
                    **{f"fault{f}_pred": v[1] for f, v in ART["fault_scores"].items()})
prov.update(settings=dict(W=W, K_CONSEC=K_CONSEC, THR_Q=THR_Q, HID=HID, LATENT=LATENT, EP_AE=EP_AE, EP_CLF=EP_CLF, SEEDS=SEEDS,
                          onset_train=ONSET_TRAIN, onset_test=ONSET_TEST), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
json.dump(prov, open(f"{OUT}/neural_provenance.json", "w"), indent=1, default=str)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ex = [f for f in (1, 6, 3) if f in ART["fault_scores"]]
    fig, ax = plt.subplots(len(ex), 1, figsize=(9, 2.6 * len(ex)), squeeze=False)
    for a_, f in zip(ax[:, 0], ex):
        a_.semilogy(ART["fault_scores"][f][0][0]); a_.axhline(ART["thr"], c="r", ls="--"); a_.axvline(I0, c="k", ls=":")
        a_.set_title(f"Fault {f}, test run 1: AE window score (dashed=threshold, dotted=onset)")
    plt.tight_layout(); plt.savefig(f"{OUT}/neural_example_scores.png", dpi=120)
except Exception as e:
    print("plot skipped:", e)
if not SMOKE:
    shutil.make_archive("/kaggle/working/tep_neural_outputs", "zip", OUT) if OUT == "/kaggle/working" else None
print("done. files:", sorted(os.listdir(OUT)))
