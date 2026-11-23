# Crude + Vacuum Refinery â€” DWSIM model

**Open (heat-integrated, current):** `Crude_Vacuum_Refinery_HeatIntegrated.dwxmz` in DWSIM 10 (results stored in the file).
**Utility-heated baseline:** `Crude_Vacuum_Refinery_utility_baseline.dwxmz` (same process, no heat recovery).
**Rebuild:** `.venv\Scripts\python.exe build_scripts\build_refinery.py` (Python 3.11 + pythonnet, ~3 min; `set HEAT_INTEGRATION=0` for the baseline).
Block diagram: `flowsheet_final.png`. Verified exchanger table: `heat_integration_report.txt` (made by `build_scripts\hx_report.py`).

## Heat integration (13 exchangers) - scope and caveats
13 DWSIM heat exchangers recover **112 MW** into three crude preheat trains: pre-desalter crude (H7-H13), desalted crude (H3-H6)
and stabilised crude to the atmospheric furnace (H1-H2). Hot sources: kerosene, diesel, AGO, LVGO, HVGO, vacuum residue, plus
debutaniser-2 naphtha (H12) and pre-fractionator overhead vapour (H13). Minimum approach 13.9 K.

**Crude-train heater duty** (E-100, H101, FH-100, E-102) falls from 138.5 MW to **27.0 MW (-81 %)**.
This excludes the 17.2 MW vacuum furnace, which is unchanged. **Including it**, total fired/heater duty goes from 155.7 MW to
44.2 MW, a **72 % reduction**.

Read these numbers with the following caveats:
- **Draw temperatures are optimistic.** The cut-point splitters deliver every product at flash-zone temperature: kerosene, diesel and AGO
  at 343.5 C, LVGO, HVGO and vacuum residue at 388 C. Real side draws are much cooler (e.g. kerosene roughly 190-230 C, diesel 270-300 C),
  so the available high-grade heat, and therefore the 112 MW and -81 %, are overstated compared with a real tower.
- **Not a pinch design.** The network came from greedy stream matching on enthalpy curves (MITA >= 15 K, hot outlets not below product-spec
  temperatures), not from a pinch-analysis target. Some matches are thermodynamically wasteful, e.g. H3 uses 388 C LVGO to heat crude from
  125 C (a ~234 K hot-end approach). The remaining 18.7 MW E-102 trim and 17.2 MW vacuum furnace are therefore not proven unrecoverable;
  a pinch-based redesign or realistic draw temperatures would change both the recovered heat and what is left.
- Exchanger areas, fouling and pressure-drop design were not done.

How it is solved: a hot stream used in two trains forms a thermal recycle, closed with DWSIM Recycle blocks (tower draws to exchanger hot sides,
cross-train tear streams, Claus feed). FH-100 burns part of the plant's own sweet fuel gas through a splitter. Vacuum residue is pumped to 12 bar and
HVGO/LVGO to 5 bar because DWSIM's flash fails for heavy vacuum cuts at low pressure in some temperature windows.
Product slate, mass balance (100.000 %) and sulphur balance (16,795 kg/h in = out) are unchanged by the integration.

## Basis
- Crude per the supplied Arab-Heavy-type assay: 100 kbpd, 589,000 kg/h, 27.6 Â°API, 2.85 wt % S, ~11 PTB salt, 0.25 wt % BS&W
  (`assay.py`: 19 real compounds + 11 coarse pseudo-components with the supplied MW / Tc / Pc / Ï‰).
- Property package: Peng-Robinson. DWSIM flash set to `NestedLoops_v2` (the default flash returned false "all vapour" results
  for this heavy crude above ~250 Â°C).

## What is modelled
| Section | Modelling |
|---|---|
| Desalting | wash water + mixing valve + desalter (component split: 97.8 % water, 95 % salt to brine) |
| Preheat / heater | E-100, H101 (utility duties E0, E1 â€” no heat-integration network), FH-100 fired heater with real fuel-gas + air firebox (conversion reactor, flue gas) |
| Pre-fractionator | cut-point splitter (150 Â°C), condenser E2 + drum, reboiler E3 |
| Debutanisers 1 & 2 | **rigorous DWSIM columns** (20 bar gas-fed / 14 bar liquid-fed) |
| Atmospheric unit | cut-point tower: OVHD / kerosene / diesel / AGO / residue, condenser E11 |
| Vacuum unit | vacuum furnace + cut-point tower at 0.08 bar: slop / LVGO / HVGO / vacuum residue |
| Gas plant | LPG amine treater, **rigorous deethaniser (32 bar) and depropaniser (17 bar)**, off-gas + fuel-gas amine treater |
| MEROX | naphtha sweetening reactor; LPG mercaptan extractors + caustic regenerator + disulfide separator |
| Claus | thermal reactor, sulphur condensers, one catalytic stage, tail-gas incinerator |

## Known limitations (read before relying on numbers)
0. DWSIM's solver rejects a flowsheet beyond a certain connected size ("infinite loop" error), so heat, pump and reactor
   energy-stream objects are not drawn; their duties are read from the unit operations instead.
1. **Crude and vacuum towers are cut-point splitters, not rigorous trays** â€” DWSIM's rigorous solvers would not converge on this
   wide-boiling feed. No tray hydraulics, pumparounds or stripping steam; product coolers carry the energy balance.
2. **Claus feed** is linked through a Recycle block (amine acid gas -> `Claus Feed`); air flows to Claus/incinerator/FH-100 are
   still set by the build script's tune functions (re-run it after changing upstream conditions).
   Claus has only 1 catalytic stage (a 2nd stage tripped the same solver limit) â†’ sulphur recovery â‰ˆ 84 %, below a real 3-stage unit.
3. Amine and caustic circuits are lumped component splitters (no lean/rich solvent loops). Water KO decanters are component splits.
4. Dissolved H2S is only 80 ppmw (no hydrotreating in scope), so the sulphur unit is small (~39 kg/h S).
5. Pseudo-component sulphur is a reporting attribute (mass fraction per cut), not a reacting species.
6. Liquid sulphur leaves the thermal reactor stream hot (â‰ˆ540 Â°C mixed header) â€” no sulphur pit / cooler modelled.

## Results (see `build_refinery.py` output)
Mass balance closes to 100.000 %; sulphur balance closes (16,795 kg/h in = out).
Yields (wt % of crude): sweet naphtha 8.7, kerosene 14.2, diesel 13.9, AGO 5.3, LVGO 7.4, HVGO 15.5, vacuum residue 33.8,
propane 0.2, C4 LPG 0.4, fuel gas 0.3.
