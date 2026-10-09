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

## Repository Structure
```
Chemical-Engineering-Projects/
├── Biodiesel production Unit/   # Biodiesel production process model
├── Crude Distillation Unit/     # Crude distillation process model
├── Predictive Maintenance/      # Dataset, model and report
├── LICENSE
└── README.md
```


## Requirements

- **Crude Distillation Unit:** Aspen HYSYS V15
- **Biodiesel production Unit:** ASPEN PLUS V15
- **Predictive Maintenance of a Milling Machine:** requirements.txt (pip install -r requirements.txt)
