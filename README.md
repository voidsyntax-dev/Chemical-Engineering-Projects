# Chemical Engineering Projects

A collection of chemical engineering process simulations and data science applications. Each project models an industrial process, from feed conditions through separation and product recovery.  

## Projects

### Crude Distillation Unit

This project simulates an atmospheric crude oil distillation unit, which separates crude oil into fractions by boiling point.

The process runs in stages:

1. **Preheating:** Crude oil is preheated through a train of heat exchangers, using the hot product streams to recover energy.
2. **Furnace:** The crude is heated to the required temperature in a fired heater (H-101), fired with fuel gas and air.
3. **Pre-fractionation:** A pre-fractionating column removes the lightest components before the main separation.
4. **Stabilization:** A debutanizer column removes light gases (LPG and fuel gas) from the naphtha, producing a stabilized liquid.
5. **Atmospheric fractionation:** The atmospheric unit separates the stream into naphtha, kerosene, diesel, atmospheric gas oil (AGO), and atmospheric residue.

The model covers the main product streams and the energy integration between them.

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

---

### Predictive Maintenance of a Milling Machine

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

### Soft Sensor for Diethylbenzene Production

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


## Repository Structure
```
Chemical-Engineering-Projects/
├── Biodiesel production Unit/                   # Biodiesel production process model
├── Crude Distillation Unit/                     # Crude distillation process model
├── Predictive Maintenance of Milling machine/   # Dataset, model and report
├── Soft sensor model on CSTR outlet/            # CSTR Simulation, Model files
├── LICENSE
└── README.md
```


## Requirements

- **Crude Distillation Unit:** Aspen HYSYS V15
- **Biodiesel production Unit:** ASPEN PLUS V15
- **Predictive Maintenance of a Milling Machine:** requirements.txt (pip install -r requirements.txt)
