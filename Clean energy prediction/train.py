"""Train and evaluate day-ahead (1-24 h) forecasts of Belgian solar, onshore and offshore wind.

    python train.py

Inputs
    data/train.parquet   forecasts issued before 2025-03-01 whose targets were already observed
    data/test.parquet    forecasts issued 2025-03-01 to 2025-05-31 (held out)
    One row per (issue_time, lead). Weather columns are archived ECMWF IFS forecasts that were
    available at issue_time; lagged generation is from >= 1 h before issue_time (see LEAKAGE.md).

Outputs
    models/<tech>_<point|q10|q50|q90>.txt   LightGBM boosters
    printed test scores: nMAE / nRMSE (% of capacity), skill vs baselines, interval coverage

Targets are capacity factors w.r.t. the capacity known at issue time; errors are normalised by
the capacity at target time. Solar is scored on daytime hours only; offshore excludes curtailed
hours and Elia's missing-wind-farm window.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
DATA, MODELS = ROOT / "data", ROOT / "models"
TECHS = ["solar", "onshore", "offshore"]
QUANTILES = [0.1, 0.5, 0.9]
OBS_LATENCY_H = 1   # generation is known up to 1 h before issue time
VALID_DAYS = 60     # last part of the training data used to choose the number of boosting rounds

COMMON = ["lead", "issue_hour", "hour", "doy_sin", "doy_cos"]
FEATURES = {
    "solar": COMMON + ["zenith", "cos_zenith", "ghi_clear", "solar_ghi", "solar_dir", "solar_dif",
                       "solar_cloud", "solar_t2m", "solar_cloud_spread", "solar_kt_fc",
                       "solar_cloud_trend3", "solar_cf_last", "solar_cf_samehour", "solar_k_samehour"],
    **{t: COMMON + [f"{t}_{c}" for c in ["ws100", "ws100_cubed", "ws10", "shear", "rho", "wpd",
                                         "wd_sin", "wd_cos", "t2m", "ws100_trend3",
                                         "cf_last", "cf_last3", "cf_samehour"]]
       for t in ["onshore", "offshore"]},
}
PARAMS = {
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "device_type": "cpu",  # pip LightGBM has no Intel-GPU support; CPU trains in seconds here
    "seed": 42, "deterministic": True, "verbose": -1,
}


# --- scoring helpers ---------------------------------------------------------------------

def score_mask(f, tech):
    """Rows used for training targets and scoring."""
    m = f[f"{tech}_y"].notna()
    if tech == "solar":
        m &= f.is_day  # night-time zeros would flatter the scores
    if tech == "offshore":
        m &= ~f.offshore_curtailed & ~f.offshore_gap
    return m


def to_mw(f, tech, cf):
    return np.clip(cf, 0, 1) * f[f"{tech}_cap_issue"].values


def metrics(f, tech, pred_mw):
    err = (pred_mw - f[f"{tech}_mw"].values) / f[f"{tech}_cap_target"].values
    return {"nMAE_%": 100 * np.nanmean(np.abs(err)), "nRMSE_%": 100 * np.sqrt(np.nanmean(err ** 2))}


# --- baselines (capacity factors) ---------------------------------------------------------

def baselines(train, test, tech):
    """Reference forecasts in MW. Elia's own forecast is an external yardstick: it is not
    matched to our issue times (published D-1 11:00 Brussels time)."""
    out = {"persistence (same hour)": to_mw(test, tech, test[f"{tech}_cf_samehour"].values)}
    if tech == "solar":
        # same-hour persistence rescaled by the change in clear-sky irradiance
        scaled = test.solar_cf_samehour.values * test.solar_cs_ratio.values
        out["clear-sky persistence"] = to_mw(test, tech, np.where(np.isfinite(scaled), scaled,
                                                                  test.solar_cf_samehour.values))
    else:
        out["persistence (last value)"] = to_mw(test, tech, test[f"{tech}_cf_last"].values)
        # power curve: monotone function of forecast 100 m wind speed alone
        tr = train[score_mask(train, tech) & train[f"{tech}_ws100"].notna()]
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(tr[f"{tech}_ws100"], tr[f"{tech}_y"])
        out["power curve (ws100)"] = to_mw(test, tech, iso.predict(test[f"{tech}_ws100"].values))
    out["Elia day-ahead (reference)"] = test[f"{tech}_elia_da"].values
    return out


# --- LightGBM ---------------------------------------------------------------------------

def fit(train, tech, params):
    """Early-stop on the last VALID_DAYS (leak-free inner split), then refit on all rows."""
    X, y = train[FEATURES[tech]], train[f"{tech}_y"]
    cut = train.target_time.max() - pd.Timedelta(days=VALID_DAYS)
    fit_rows = train.target_time <= cut - pd.Timedelta(hours=OBS_LATENCY_H)
    val_rows = train.issue_time >= cut
    probe = lgb.train(params, lgb.Dataset(X[fit_rows], y[fit_rows]), 3000,
                      valid_sets=[lgb.Dataset(X[val_rows], y[val_rows])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(params, lgb.Dataset(X, y), max(probe.best_iteration, 50))


def train_tech(train, tech):
    tr = train[score_mask(train, tech)]
    models = {"point": fit(tr, tech, dict(PARAMS, objective="regression"))}
    for q in QUANTILES:
        models[f"q{int(q * 100)}"] = fit(tr, tech, dict(PARAMS, objective="quantile", alpha=q))
    return models


def predict(models, test, tech):
    """Capacity factors; quantiles sorted per row so they never cross; solar = 0 at night."""
    X = test[FEATURES[tech]]
    out = pd.DataFrame({"point": models["point"].predict(X)}, index=test.index)
    qs = np.sort(np.column_stack([models[f"q{int(q * 100)}"].predict(X) for q in QUANTILES]), axis=1)
    for i, q in enumerate(QUANTILES):
        out[f"q{int(q * 100)}"] = qs[:, i]
    out = out.clip(0, 1)
    if tech == "solar":
        out[~test.is_day.values] = 0.0
    return out


def main():
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    print(f"Train: {len(train):,} rows, targets {train.target_time.min():%Y-%m-%d} to "
          f"{train.target_time.max():%Y-%m-%d} | Test: {len(test):,} rows, issued "
          f"{test.issue_time.min():%Y-%m-%d} to {test.issue_time.max():%Y-%m-%d}\n")
    MODELS.mkdir(exist_ok=True)
    reference = {"solar": "clear-sky persistence", "onshore": "power curve (ws100)",
                 "offshore": "power curve (ws100)"}

    for tech in TECHS:
        models = train_tech(train, tech)
        for name, m in models.items():
            m.save_model(str(MODELS / f"{tech}_{name}.txt"))
        p = predict(models, test, tech)

        te = test[score_mask(test, tech)]
        preds = {name: b[score_mask(test, tech).values] for name, b in baselines(train, test, tech).items()}
        preds["LightGBM"] = to_mw(te, tech, p.loc[te.index, "point"].values)
        ok = np.all([np.isfinite(v) for v in preds.values()], axis=0)  # same rows for every model
        te = te[ok]
        scores = pd.DataFrame({name: metrics(te, tech, v[ok]) for name, v in preds.items()}).T
        scores["skill_vs_ref_%"] = 100 * (1 - scores["nRMSE_%"] / scores.loc[reference[tech], "nRMSE_%"])

        y = te[f"{tech}_mw"].values
        q = {c: to_mw(te, tech, p.loc[te.index, c].values) for c in ["q10", "q50", "q90"]}
        inside = np.mean((y >= q["q10"]) & (y <= q["q90"]))
        below = [np.mean(y < q[c]) for c in ["q10", "q50", "q90"]]

        label = {"solar": "Solar PV (daytime hours only)", "onshore": "Onshore wind",
                 "offshore": "Offshore wind (no curtailment / missing-farm window)"}[tech]
        print(f"{label}: {ok.sum():,} scored hours, rounds "
              f"{', '.join(str(m.num_trees()) for m in models.values())}")
        print(scores.round(2).to_string())
        print(f"10-90 % interval holds {100 * inside:.1f} % of actuals (target 80 %); share below "
              f"q10/q50/q90: {' / '.join(f'{100 * b:.1f}' for b in below)} % (target 10/50/90)\n")


if __name__ == "__main__":
    main()
