# Leakage: why it matters here and how the pipeline avoids it

## The problem

A day-ahead generation forecast is made at an **issue time**. At that moment the forecaster has
(1) the latest numerical weather prediction (NWP) run that has actually been published, and
(2) generation measured up to shortly before the issue time. Nothing else.

When a model is trained on historical data, it is easy to give it information that would not
have existed at the issue time. The model then learns a relationship that is sharper than reality,
scores well in the backtest, and does worse in operation. For weather-driven forecasting the
biggest trap is the weather itself. Measured or reanalysed weather (e.g. ERA5) at the target
hour is **what happened**, not **what was forecast**. A model trained on it learns, for example,
that a cloud-free hour gives high solar output with near-certainty. In operation it receives a
forecast, which may have the clouds in the wrong place, and its errors grow. Because generation
depends nonlinearly on weather (wind power ∝ v³ below rated speed), even modest weather errors
are amplified.

In this project, that error was measured directly. The same LightGBM model, trained and tested
with ERA5 weather instead of forecasts, has a 14 % lower nRMSE for onshore wind
(3.39 % vs 3.94 % of capacity) and 11 % lower for offshore (7.60 % vs 8.57 %). That gap is what a
reanalysis-trained model would have wrongly claimed. (Measured in an earlier, fuller evaluation run whose code has since been removed from this folder.)

## The leakage channels and what the pipeline does about each

| # | Channel | What would go wrong | What the pipeline does | Where it lives now |
|---|---|---|---|---|
| 1 | **Weather at target time** | Training on observed/reanalysis weather = perfect foresight | Model inputs are **archived NWP forecasts** only. ERA5 is used only for the oracle comparison and for labelling weather regimes in the evaluation | dataset build (code removed; result is `data/*.parquet`) |
| 2 | **A forecast that wasn't out yet** | Using the run initialised at 00 UTC for a forecast "issued" at 00 UTC, although ECMWF publishes it ~6-7 h later. "Historical forecast" archives that stitch together the latest available run per hour hide this completely | Each archived ECMWF IFS run is fetched by its **initialisation time** (Open-Meteo Single Runs API) and stored with `init_time`. A run is usable only **7 h after initialisation**: issue times are 07 and 19 UTC, and lead *L* uses NWP step 7 + *L*. An `assert` checks `issue_time - init_time ≥ 7 h` for every row | dataset build (code removed; result is `data/*.parquet`); `issue_time`, `lead` columns |
| 3 | **Generation observed after the issue time** | Lags such as "generation 24 h before the target" reach past the issue time for long leads | Generation is assumed known up to the hour ending **1 h before** issue (`OBS_LATENCY_H`). The same-hour lag goes back 24 h for leads 1-23 and **48 h for lead 24**. An `assert` checks every lag timestamp ≤ issue time − 1 h | dataset build (code removed; result is `data/*.parquet`) |
| 4 | **Train/test split** | A random split puts tomorrow's hour in the training set. A split on *target* time is subtler: a test forecast issued at 19:00 on day *d* predicts hours on *d*+1, while the training set still contains targets from the evening of *d*, which weren't observed yet at that issue time | `time_split`: **test = forecasts issued at or after the cut; train = targets observed at least 1 h before it**. Used for the train/test split and for the early-stopping split inside training. (The target-time version was in an earlier draft of this pipeline; fixing it changed onshore nRMSE only from 3.95 % to 3.94 %, but it was a real leak) | split into `data/train.parquet` / `data/test.parquet`; `train.py: fit` (inner split) |
| 5 | **Model selection on the test set** | Tuning hyperparameters or choosing features until the test score looks good | Hyperparameters are fixed in advance (`train.py: PARAMS`). The number of boosting rounds is chosen on the last 60 days of the *training* data. The test period (forecasts issued from 1 Mar 2025) is never used for any choice | `train.py: PARAMS, fit` |
| 6 | **Interval calibration on the test set** | Calibrating quantiles on the data they are evaluated on guarantees good-looking coverage | Conformal offsets are estimated from the **rolling-origin backtest's out-of-sample errors only**, which come entirely before the test period | earlier evaluation (code removed); `train.py` reports raw intervals only |
| 7 | **Capacity at the target time** | Installed capacity grows by ~35 % over the period. Scaling predictions by the capacity at the target hour uses a number from the future | The model predicts generation divided by the capacity **known at issue time** and is converted back to MW with that same capacity. Capacity at the target time is used only to normalise the error metrics | `<tech>_y`, `<tech>_cap_issue` columns; `train.py: to_mw` |
| 8 | **Weights for the weather points** | Weighting the provinces' weather by capacity uses Elia data | Monthly mean capacities are used. This includes a few weeks after some issue times (see "remaining risks"), but capacity is administrative data that changes slowly and is known in advance | dataset build (code removed; result is `data/*.parquet`) |

## Evidence that it works

- **Hard checks**: the asserts in channels 2 and 3 ran when the datasets were built; every row passed.
- **Lead-time behaviour**: the correlation between the last observed hour and the target falls
  steadily with lead (onshore: 0.95 at 1 h → 0.39 at 24 h), as it must if no future generation
  sneaks in. Forecast-wind features stay informative at all leads (≈0.9).
- **Alignment**: forecast irradiance correlates best with measured solar at zero offset
  (0.968, vs 0.93 at ±1 h), confirming that the hour-ending timestamps line up.
- **Oracle gap**: the ERA5 model is clearly better for wind. If the forecast inputs had leaked
  observed weather, the gap would be close to zero.

## Remaining risks (where the setup may still be optimistic)

- **Availability assumption**: 7 h after initialisation is a reasonable estimate for ECMWF HRES,
  not a measured value for every run. If a run were late, an operational system would fall back
  to the previous run; this pipeline doesn't simulate that.
- **The target itself is partly modelled**: Elia's "measured" solar (and DSO-connected wind) is
  an upscaled estimate, not a meter reading. If Elia's estimation uses weather information, part
  of the target may be easier to predict from weather than real output would be.
- **Monthly capacity weights** (channel 8) use a full-month mean, so a few weeks of future capacity
  information enter the weather aggregation. The effect is expected to be negligible.
- **Data revisions**: Elia data were downloaded in October 2026 and may have been corrected since
  first publication. An operational forecaster would have seen the unrevised values.
- **Design choices made with knowledge of the whole period**: the study period, the weather
  points, and the decision to exclude Elia's missing-offshore-farm window were made after looking
  at the data. None of them uses the test outcomes, but they aren't fully "blind" either.
- **ERA5 is not ground truth**: for solar, the ERA5 "oracle" is *worse* than the forecast model,
  because ERA5's coarse grid (≈30 km) and cloud representation give poorer irradiance than ECMWF's
  short-range forecast. The oracle gap is therefore a rough lower bound on the cost of forecast
  error for wind, and not meaningful for solar.
