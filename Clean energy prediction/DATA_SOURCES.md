# Data sources

All data were downloaded from the public APIs below (the download and preprocessing scripts have since been
removed; their output is `data/train.parquet` and `data/test.parquet`). Nothing is fabricated or synthetic.
All timestamps are stored in **UTC**. Retrieved on 2026-10-31.

## Summary

| Role | Source | Period used | Resolution | Registration |
|---|---|---|---|---|
| Target: solar PV generation | Elia open data `ods032` | 2024-03-14 → 2025-06-01 | 15 min | None |
| Target: wind generation (onshore / offshore) | Elia open data `ods031` | 2024-03-14 → 2025-06-01 | 15 min | None |
| **Model inputs**: archived NWP forecasts | Open-Meteo Single Runs API, ECMWF IFS HRES 9 km | runs 2024-03-14 → 2025-05-31 | hourly | None |
| Reference only (never a model input): reanalysis | Open-Meteo Historical Weather API, ERA5 | same (not included in the datasets) | hourly | None |

The study period starts on 2024-03-14 because that is the first ECMWF run archived on
the Single Runs API. Generation data go back further, but without archived forecasts
they could only be paired with reanalysis weather, which would leak future information.

---

## 1. Elia — solar PV generation (`ods032`)

- **Title:** Photovoltaic power production estimation and forecast on Belgian grid (Historical)
- **URL:** https://opendata.elia.be/explore/dataset/ods032/
- **API used:** `https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032/exports/parquet`
- **Licence:** "Elia Open Data Licence", attribution to *Elia Transmission Belgium SA* (from the dataset
  metadata). The licence page (https://opendata.elia.be/pages/licence/) was returning a "technical issue"
  page when this file was written, so the full terms are **still to be verified**.
- **Full time range available:** 2020-10-01 → present (the last record seen was 2026-10-25 22:45 UTC).
- **Resolution:** 15 minutes. `datetime` is the start of the quarter-hour, in UTC.
- **Spatial breakdown:** Belgium total, the regions Flanders and Wallonia, and 11 provinces
  (including Brussels). The provinces add up exactly to the Belgium total (checked: max difference 1e-12 MW).
- **Fields kept:** `measured` (MW), `monitoredcapacity` (MW), Elia's day-ahead forecast plus its
  P10/P90 (`dayaheadforecast`, `dayaheadconfidence10/90`), and the D-1 11:00 forecast (`dayahead11hforecast`).
- **Units:** MW (quarter-hour average).
- **Known quality issues:**
  - `measured` is **an estimate, not a meter reading**: Elia extrapolates from a sample of metered
    installations, because most rooftop PV is not metered. The target therefore has its own error.
  - In the April 2024 test month, `measured` is **missing at 21:45 UTC every day** in every region.
    This is a systematic artefact at night, when output is 0, so it has no effect on daytime evaluation.
  - Monitored capacity grows over time (9.09 → 9.19 GW within April 2024; about 11.2 GW by mid-2025).
    All errors are therefore normalised by the capacity at that time, not by a fixed number.

## 2. Elia — wind generation (`ods031`)

- **Title:** Wind power production estimation and forecast on Belgian grid (Historical)
- **URL:** https://opendata.elia.be/explore/dataset/ods031/
- **Licence:** as for `ods032`.
- **Full time range available:** 2014-12-31 → present (the last record seen was 2026-10-30 22:45 UTC).
- **Resolution:** 15 minutes, UTC, `datetime` = start of the interval.
- **Breakdown:** 5 groups: Offshore (Federal, Elia-connected), and Onshore Flanders/Wallonia × Elia/DSO-connected.
  There is no national total row; we sum the groups ourselves.
  Capacity in mid-2025: offshore ≈ 2.26 GW, onshore ≈ 3.29 GW.
- **Targets used:** **offshore** and **onshore** wind are modelled separately, because they respond
  differently to the weather and have different data problems.
- **Units:** MW.
- **Known quality issues:**
  - Elia states that **one offshore wind farm is missing from the published data from 2025-04-03 to
    2025-09-30**, while the monitored capacity may still include it. Handling: those hours are flagged
    (`offshore_gap`) and excluded from offshore training and scoring.
  - **Curtailment:** `decrementalbidid` is non-empty when offshore output was reduced on Elia's request
    (e.g. 12 quarter-hours in April 2024). The weather did not cause these reductions, so they are flagged.
  - Onshore values for DSO-connected turbines are partly estimated, as for solar.

## 3. Open-Meteo Single Runs API — archived ECMWF IFS forecasts (model inputs)

- **URL / docs:** https://single-runs-api.open-meteo.com/v1/forecast · https://openmeteo.substack.com/p/single-runs-api
- **Model:** `ecmwf_ifs` (ECMWF IFS HRES, about 9 km grid)
- **Licence:** Open-Meteo API data are CC BY 4.0 ("Weather data by Open-Meteo.com",
  https://open-meteo.com/en/licence). The underlying ECMWF data are under ECMWF's open-data CC BY 4.0 licence.
  The free API is for non-commercial use.
- **Time range:** IFS runs at 00/06/12/18 UTC are archived from 2024-03-14. We use the **00 and 12 UTC runs**.
- **Resolution:** hourly, 32 steps per run (step 0–31 h).
- **Variables:** `shortwave_radiation` (GHI), `direct_radiation`, `diffuse_radiation` (W/m², mean over
  the **preceding** hour), `cloud_cover` (%), `temperature_2m` (°C), `relative_humidity_2m` (%),
  `surface_pressure` (hPa), `wind_speed_10m`, `wind_speed_100m` (m/s), `wind_direction_100m` (°).
- **Points:** 6 land points, each standing for one or more provinces, plus 1 offshore point:
  West+East Flanders (51.03 N, 3.40 E), Antwerp (51.20, 4.55), Limburg (51.00, 5.35), Flemish+Walloon Brabant and Brussels (50.80, 4.55), Hainaut (50.45, 3.95), Liège+Namur+Luxembourg (50.40, 5.30), offshore wind zone (51.62, 2.90).
- **How the issue time is enforced:** each row keeps the run's `init_time`. A run is assumed usable only
  **7 h after initialisation** (ECMWF dissemination finishes roughly 6–7 h after init). Forecasts are therefore
  issued at 07:00 and 19:00 UTC, and lead *L* hours uses NWP step 7 + *L*. See `LEAKAGE.md`.
- **Known quality issues:** radiation is null at step 0 (it is a preceding-hour mean), which we never use.
  A single grid point is only a rough stand-in for a whole province or wind-farm cluster.

## 4. Open-Meteo Historical Weather API — ERA5 reanalysis (reference only)

- **URL / docs:** https://archive-api.open-meteo.com/v1/archive · https://open-meteo.com/en/docs/historical-weather-api
- **Licence:** Copernicus Climate Change Service (C3S) licence; attribution required: "Contains modified
  Copernicus Climate Change Service information".
- **Resolution:** hourly, about 0.25°. Same points and variables as the NWP data.
- **Use:** **never used as a model input** and not included in `data/*.parquet`. It was used only in an
  earlier evaluation (code since removed) to (a) measure how far the forecast weather differed from what
  actually happened, and (b) train an "oracle" model to show how much forecast error costs.
- **Known quality issues:** ERA5 is coarse (~25–31 km) and smooths out convective cloud. On Open-Meteo it
  becomes available about 5 days after real time.

## Sources considered and rejected

| Source | Why not used |
|---|---|
| Grid-India NLDC PSP reports (India) | Usable (15-min all-India SCADA from Dec 2024), but only a national total and no capacity column. Belgium was chosen instead (see project discussion). |
| Kerala / Agali / Ramakkalmedu, KSEB, SLDC Kerala | No public hourly generation data found. |
| Open Power System Data | Ends in 2020, before archived NWP forecasts exist, so leak-free features are impossible. |
| ENTSO-E Transparency Platform | Needs an account and an API token. |
| EIA-930 (US) | No installed capacity in the data, and known data-quality problems. Kept as a fallback. |
| NREL solar/wind datasets | Modelled or synthetic, not measured. |
| Open-Meteo Previous Runs API | Only fixed 24 h offsets (day0…day7); does not give exact issue times for 1–24 h leads. |
| Renewables.ninja | Needs registration for API access; optional sanity check only. |
