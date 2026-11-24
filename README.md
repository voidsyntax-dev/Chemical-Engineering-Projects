# Chemical Engineering Projects

A collection of chemical engineering process simulations and data science applications. Each project models an industrial process, from feed conditions through separation and product recovery.  

## Projects

### Soft Sensor for Ethylbenzene Production

This project builds a soft sensor that estimates the outlet composition of a CSTR producing ethylbenzene and diethylbenzene from benzene and ethylene. The composition is hard to measure online, so the sensor predicts it from easy-to-measure temperatures, pressures, flows, level, and controller setpoints.

The workflow runs in stages:

1. **Data:** Dynamic history from an Aspen Dynamics model of the reactor. Inputs are varied with random step changes in the reactor temperature and level setpoints, the flow setpoints, and the feed and utility conditions. Each run is logged at 0.01 h.
2. **Normal-operation filter:** Rows where the reactor, level, or setpoints leave the normal band are removed, so the model learns only the operating range it will be used in.
3. **Lag features:** Past values of each measurement are added, computed within continuous segments so no feature crosses a gap in the data.
4. **Time-based split:** The first 70% of time is used for training and the last 30% for testing. Random splitting is avoided because neighboring samples are highly correlated.
5. **Model comparison:** A mean baseline, Ridge, PLS, random forest, SVR, a multilayer perceptron, and gradient boosting are trained for each target. The MLP architecture for D-ETBENZ was tuned by time-ordered cross-validation on the training data. The ETBENZ MLP uses a fixed architecture of (32, 16).
6. **Deployment:** The selected model for each target is saved with its scaler, feature list, lag settings, normal-band limits, and sampling interval, so the same preprocessing can be applied to new data.

**Targets:** outlet mole fractions of ethylbenzene (ETBENZ) and diethylbenzene (D-ETBENZ).

#### Results (held-out test period, last 30% of time)

| Target | Selected model | RMSE | MAE | R² |
|---|---|---|---|---|
| ETBENZ | MLP (32, 16) | 0.004900 | 0.003548 | 0.979 |
| D-ETBENZ | MLP (16,) | 0.000042 | 0.000027 | 0.892 |

Mean baseline for comparison: ETBENZ R² of −0.001, D-ETBENZ R² of −0.048. Ridge scored 0.973 for ETBENZ and 0.733 for D-ETBENZ.

---

### Reinforcement Learning Control of a Van de Vusse CSTR

This project trains a reinforcement learning (RL) controller for a jacketed CSTR running the Van de Vusse reactions (A → B → C, 2A → D), where B is the desired product. The controller manipulates the feed rate and jacket cooling duty to hold the product concentration at a set-point while keeping the reactor below 130 °C. It is benchmarked against an industrial PI scheme and a nonlinear model predictive controller (MPC).

The workflow runs in stages:

1. **Process model:** Material, energy and jacket balances with Arrhenius kinetics, a side reaction, and a cooling-only jacket, using literature parameters (Klatt & Engell, 1998). The model is checked against the published steady state, a mass-balance closure test, and a reference ODE solver.
2. **Realistic plant environment:** The model is wrapped as a Gymnasium environment with a 20 s dead time and thermocouple lag on temperature, a 30 s analyzer delay on concentrations, measurement noise, valve saturation and rate limits. Each episode adds feed-temperature drift, fouling, coolant capacity loss and kinetic mismatch.
3. **Baselines:** A dual PI controller (concentration ← feed rate, temperature ← jacket duty) tuned by grid search, and a nonlinear MPC (CasADi + IPOPT, 200 s horizon) with a disturbance observer for offset-free tracking.
4. **RL training:** A PPO agent is first trained from scratch on a GPU, then retrained as MPC-guided PPO. This combines behaviour cloning from MPC demonstrations, PPO with an imitation term that decays over training, and DAgger rounds in which the MPC corrects the agent's own trajectories.
5. **Robustness:** The agent gets error-memory inputs (5- and 20-min error averages) and is trained over a wide range of kinetic mismatch (±15%). It is then tested on a held-out "hard" set (kinetics 15–22% off nominal) never used in training or model selection.

**Objective:** track the product concentration cB while respecting the temperature limit and limiting valve movement.

#### Results (20 identical scenarios per set, mean |cB − set-point| in mol/L)

| Test set | PI | Nonlinear MPC | MPC-guided PPO |
|---|---|---|---|
| Nominal plant | 0.078 | 0.011 | 0.013 |
| Perturbed plant | 0.093 | 0.030 | 0.032 |
| Hard, held out | 0.110 | 0.046 | 0.046 |
| Compute per decision | ~0 | 10–34 ms | 0.6 ms |

On the held-out set the RL controller matches nonlinear MPC on tracking at about 1/20–1/50 of the computation per decision, with no online optimisation. No controller had a thermal runaway or time above 130 °C in any test. PI error is 3–8× higher, because the two loops interact, the product gain changes sign near the optimum, and the process has dead time.

---

### Fault Detection and Diagnosis for the Tennessee Eastman Process

This project builds fault detection and diagnosis systems for the Tennessee Eastman Process (TEP), a standard benchmark plant with a reactor, condenser, separator, stripper and recycle loop under closed-loop control. The controllers hide faults, so the monitors must pick up small changes in 52 measured and manipulated variables and name the fault behind them.

The workflow runs in stages:

1. **Data and simulator check:** The Rieth et al. (2017) dataset is used with its official train and test split, with 500 runs for each of 20 faults. The Downs and Vogel Fortran model is compiled and reproduces the published normal operating point within 0.24%.
2. **PCA baseline:** T² and SPE statistics with control limits set only from held-out normal runs. Diagnosis uses contribution plots.
3. **Learned models:** XGBoost on windowed features, and an LSTM autoencoder trained on normal data only with a classifier head for diagnosis (trained on a Kaggle GPU).
4. **Fixed alarm rule:** Every model alarms when its score stays above the 99th-percentile limit for 3 samples in a row. Test data never sets a threshold or picks a model.
5. **Evaluation and leakage checks:** Detection rate, false-alarm rate, delay and diagnosis accuracy for each fault, with 95% intervals over 5 seeds. Results above 99% go through leakage checks.

**Faults:** 20 process faults, including steps, random variations, drifts and valve sticking.

| Model | Samples in alarm | False alarms | Diagnosis accuracy |
|---|---|---|---|
| PCA | 62.6% | 0.06% | 64.3% |
| LSTM autoencoder | 74.6% | 0.66% | 81.8% |
| XGBoost | 82.7% | 0.20% | 86.5% |

No model reaches 99% detection on 14 of 20 faults. Faults 3, 9 and 15 are the hardest: the best detection is 10.6%, 3.1% and 1.1%. The leakage checks found no problem.

Dataset: Rieth et al. (2017), Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation (Harvard Dataverse).

---

### Day-Ahead Solar and Wind Generation Forecasting for Belgium

This project forecasts Belgium's national solar PV, onshore wind and offshore wind generation 1 to 24 hours ahead, with 10/50/90% prediction intervals. The focus is avoiding leakage: every input was really available when the forecast was issued. In particular, the weather inputs are the forecasts actually published by then, not the weather that later happened.

The workflow runs in stages:

Data: Elia open data gives generation and monitored capacity every 15 minutes, averaged to hourly. The weather inputs are archived ECMWF IFS 9 km forecast runs (00 and 12 UTC) at 7 points over Belgium, from the Open-Meteo Single Runs API. Forecasts were issued from March 2024 to May 2025.
Issue-time enforcement: A forecast run is used only 7 h after it starts, so forecasts are issued at 07:00 and 19:00 UTC, and lead L uses forecast step 7 + L. Lagged generation comes only from at least 1 h before issue time. Assertions check both rules on every row.
Feature engineering: Solar features are irradiance, cloud cover, clear-sky index and sun position (pvlib). Wind features are 100 m wind speed and its cube, air density, wind power density, shear and direction. Lagged capacity factors are added for all three. The targets are capacity factors based on the capacity known at issue time.
Leak-free split: The test set holds forecasts issued from 1 March 2025. The training set holds only targets observed before that date. The number of boosting rounds is chosen on the last 60 days of training data with the same rule. Hyperparameters are fixed in advance.
Models and baselines: Each technology gets one LightGBM point model and three quantile models. They are compared with same-hour persistence, last-value persistence, clear-sky-scaled persistence for solar, and an isotonic power curve on forecast wind speed for wind.
Scoring: nMAE and nRMSE are in % of capacity. Solar is scored on daytime hours only. Offshore leaves out curtailed hours and a window when one wind farm is missing from Elia's data.

Targets: hourly solar PV, onshore wind and offshore wind generation for Belgium.

Results (held-out test period, March–May 2025, nRMSE in % of capacity)
Model	Solar (daytime)	Onshore wind	Offshore wind
Persistence (same hour)	12.55	12.80	26.08
Strongest simple baseline	12.60	4.39	11.11
LightGBM	5.81 (+54%)	3.94 (+10%)	8.57 (+23%)
Elia's own day-ahead forecast	5.05	4.08	8.85

The strongest simple baseline is clear-sky persistence for solar and the power curve for wind. LightGBM beats Elia's operational forecast for both onshore and offshore wind, and comes within 0.8 points of it for solar. Training the same model on reanalysis weather would have overstated its wind accuracy by 11–14%. The 10–90% intervals contain 72–74% of actual values against an 80% target.

Data: Elia Open Data (ods031, ods032) and ECMWF IFS forecasts via the Open-Meteo Single Runs API.

---

### Shell-and-Tube Heat Exchanger Design Tool

This project builds a Python package for the preliminary thermal, hydraulic and mechanical design of shell-and-tube heat exchangers. Given the two process streams, a target duty and pressure-drop limits, it either rates a fixed geometry or searches for the smallest exchanger that meets every constraint. It runs as a library, a JSON command-line tool, or a local web page.

The workflow runs in stages:

1. **Input checks:** Streams, geometry and constraints are read from JSON with strict validation for unknown fields, types and NaN values. The hot duty, cold duty and target duty must agree within 2 %, so an inconsistent energy balance is rejected before any calculation.
2. **Thermal model:** Heat duty, LMTD with the F-correction for one shell pass and multiple tube passes, and tube-side coefficients that blend continuously through the laminar–turbulent transition, with Sieder–Tate wall correction. Rating uses effectiveness–NTU, so both stream energy balances close.
3. **Shell side:** Bell–Delaware method following Taborek/Serth, with the five correction factors for baffle cut, leakage, bypass, unequal spacing and laminar flow. Shell pressure drop uses the window, cross-flow and end-zone terms. A Kern estimate is kept for comparison.
4. **Hydraulics:** Tube-side pressure drop with Colebrook–White friction, plus return losses.
5. **Design solver:** Searches tube count, tube passes, shell diameter and baffle spacing. The shell diameter is derived from the tube count so the bundle physically fits. A design is accepted only if it meets duty, outlet temperatures, both pressure-drop limits and the TEMA baffle-spacing limits.
6. **Mechanical screening and reporting:** Tube-sheet thickness, a vortex-shedding vibration screen and a cost proxy. Results go to a text report and an Excel report, and the web interface shows pass/fail checks in Design and Rate modes.

**Objective:** meet the target duty with the least heat-transfer area while staying within the pressure-drop limits.

#### Results (sample case: 4.18 kg/s process stream cooled 150 → 100 °C by 2.0 kg/s water heated 20 → 80 °C)

| Item | Value |
|---|---|
| Duty (target 501.6 kW) | 501.6 kW |
| Selected geometry | 48 tubes (19.05 mm OD, 5 m), 4 tube passes, 0.22 m shell ID |
| Heat-transfer area | 14.4 m² |
| LMTD / F-correction | 74.9 K / 0.903 |
| h tube / h shell / U | 4,072 / 831 / 516 W/(m² K) |
| Tube ΔP (limit 200 kPa) | 14.2 kPa |
| Shell ΔP (limit 100 kPa) | 18.9 kPa |
| Bell–Delaware factors Jc, Jl, Jb | 1.06, 0.71, 0.72 |

The solver reduced the 240-tube, 0.65 m trial geometry to a 48-tube design that passes all checks: duty, outlet temperatures, pressure drop, baffle spacing and vibration. Both outlet temperatures match the specification to within 0.0001 K. The 146 tests cover the thermal, Bell–Delaware, rating, mechanical, CLI and web-interface modules.

---

### Predictive Maintenance of Milling Machine

This project develops a machine learning model that predicts equipment failure from operating conditions, so that maintenance can be scheduled before a breakdown occurs. 

The workflow runs in stages:

1. **Data and quality checks:** 10,000 machine readings covering air and process temperature, rotational speed, torque, tool wear, and machine type. Conflicting failure labels are identified and handled.
2. **Feature engineering:** derived variables such as temperature difference, mechanical power, and torque-to-speed ratio that capture operating conditions linked to failure.
3. **Model comparison:** eleven models, including a baseline, are trained and tuned, with validation designed to avoid data leakage and to respect the time order of the data.
4. **Threshold selection:** the decision threshold is set to minimise expected cost, where a missed failure is ten times as costly as a false alarm.
5. **Deployment:** the selected gradient boosting model is saved with its threshold and feature settings, along with an input-validated prediction function.

On held-out data, the model catches about 84% of failures with about 95% precision.  
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).

---

### Crude and Vacuum Refinery Flowsheet with Heat Integration (DWSIM)

This project builds a steady-state model of a refinery crude and vacuum distillation complex in DWSIM 10. It is based on a HYSYS CDU flowsheet and covers desalting, crude and vacuum distillation, light-ends recovery, sweetening and sulphur recovery. It then adds a heat-recovery network to cut the fired-heater duty. The whole flowsheet is built by Python scripts, so it can be rebuilt and changed without redrawing.

The workflow runs in stages:

1. **Crude basis:** A heavy, sour, salty crude (Arab-Heavy type, 100 kbpd, 27.6 °API, 2.85 wt% sulphur, 11 PTB salt) is defined from a supplied assay. It uses 19 real compounds and 11 pseudo-components with the given MW, Tc, Pc and acentric factor, and the Peng-Robinson property package.
2. **Flash fix:** DWSIM's default flash returned false all-vapour results above about 250 °C for this crude. Switching to the Nested Loops v2 flash fixed it.
3. **Hybrid process model:** The desalter, pre-fractionator, atmospheric tower and vacuum tower are chains of cut-point splitters, because rigorous columns would not converge on a wide-boiling feed. The two debutanisers, the deethaniser and the depropaniser are rigorous columns and all converge. The model also has a fired heater with a firebox, amine treating, MEROX and a Claus unit with tail-gas incinerator.
4. **Heat integration:** Thirteen exchangers recover heat from the hot product draws, the debutaniser bottoms and the pre-fractionator overhead into three crude preheat trains. The network was built by greedy stream matching on enthalpy curves with a minimum approach of about 15 K. It is not a pinch-analysis design.
5. **Closing the loops:** Thermal recycles between trains and the sulphur-unit feed are real DWSIM Recycle blocks. The firebox burns part of the plant's own sweet fuel gas, so no copied streams remain.
6. **Layout and deliverables:** A hand-designed process-flow canvas puts the crude train on a main line, with the light ends above it and the product coolers below. Checks cover mass closure, sulphur balance and solver status. Outputs are the `.dwxmz` model, an exchanger table and a technical report.

**Products:** sweet naphtha, kerosene, diesel, atmospheric gas oil, light and heavy vacuum gas oil, vacuum residue, propane, C4 LPG, fuel gas and liquid sulphur.

#### Results (steady-state solution)

| Item | Utility-heated case | Heat-integrated |
|---|---|---|
| Crude-train heater duty (E-100, H101, FH-100, E-102) | 138.5 MW | 27.0 MW (−81 %) |
| Including the 17.2 MW vacuum furnace | 155.7 MW | 44.2 MW (−72 %) |
| Heat recovered | – | 112.2 MW (13 exchangers) |
| Minimum approach | – | 13.9 K |

Product yields (wt% of crude): naphtha 8.7, kerosene 14.2, diesel 13.9, AGO 5.3, LVGO 7.4, HVGO 15.5, vacuum residue 33.8. Mass closure is 100.000 %, and sulphur in equals sulphur out at 16,795 kg/h. All units and all four rigorous columns converge.

---

### Biodiesel production Unit

This project models a base-catalyzed biodiesel plant that converts vegetable oil and methanol into fatty acid methyl esters (FAME), the main component of biodiesel, with glycerol as a byproduct.

The process runs in stages:

1. **Feed preparation:** Oil and methanol are mixed and pumped through a heat exchanger to bring them to reaction temperature. A sodium hydroxide (NaOH) catalyst is added to the stream.
2. **Transesterification:** The mixture enters a reactor, where triglycerides in the oil react with methanol to form methyl esters and glycerol.
3. **Separation:** The reactor output goes to a methanol column, which recovers excess methanol for recycling. The remaining stream passes to an ester column, which separates the crude FAME from the heavier phase.
4. **Purification:** The crude ester is neutralized with phosphoric acid (H3PO4), washed with water in a wash column, and filtered to remove solids.
5. **Recovery and recycle:** Unreacted oil and methanol are recovered and recycled, and the water and glycerol streams are separated out.

The flowsheet gives the final FAME product, glycerol, and aqueous waste streams as outputs.

## Repository Structure
```
Chemical-Engineering-Projects/
├── Biodiesel production Unit/                                                  # Biodiesel production process model
├── Crude and Vacuum Refinery Flowsheet with Heat Integration (DWSIM)/          # Crude + Vacuum distillation process model
├── Predictive Maintenance of Milling machine/                                  # Dataset, model and report
├── Soft Sensor for Ethylbenzene Production/                                    # CSTR Simulation, Model files
├── RL Control of Van de Vusse CSTR/                                            # Gymnasium env, PI/MPC baselines, PPO training, models
├── Tennessee Eastman Fault Detection/                                          # Simulator check, PCA/XGBoost/LSTM models, report
├── Shell-and-Tube Heat Exchanger Design/                                       # Design package, CLI, web interface, examples, tests
├── Solar and Wind Forecasting Belgium/                                         # Datasets, LightGBM training script, leakage notes
├── LICENSE
└── README.md
```

## Requirements

- **Crude + Vacuum Distillation Unit:** DWSIM 10
- **Biodiesel production Unit:** ASPEN PLUS V15
- **Predictive Maintenance of a Milling Machine:** requirements.txt (pip install -r requirements.txt)
- **Soft Sensor for Ethylbenzene Production:** requirements.txt (pip install -r requirements.txt)
- **Fault Detection and Diagnosis for the Tennessee Eastman Process:** requirements.txt (pip install -r requirements.txt)
- **Reinforcement Learning Control of a Van de Vusse CSTR:** requirements.txt (pip install -r requirements.txt)
- **Shell-and-Tube Heat Exchanger Design Tool:** requirements.txt (pip install -r requirements.txt)
- **Day-Ahead Solar and Wind Generation Forecasting for Belgium:** requirements.txt (pip install -r requirements.txt)


