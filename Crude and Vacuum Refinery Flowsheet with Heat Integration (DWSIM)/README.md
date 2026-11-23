# Crude + Vacuum Refinery â€” DWSIM model

**Open (heat-integrated, current):** `Crude_Vacuum_Refinery_HeatIntegrated.dwxmz` in DWSIM 10 (results stored in the file).
**Utility-heated baseline:** `Crude_Vacuum_Refinery_utility_baseline.dwxmz` (same process, no heat recovery).
**Rebuild:** `.venv\Scripts\python.exe build_scripts\build_refinery.py` (Python 3.11 + pythonnet, ~3 min; `set HEAT_INTEGRATION=0` for the baseline).
Block diagram: `flowsheet_final.png`. Verified exchanger table: `heat_integration_report.txt` (made by `build_scripts\hx_report.py`).

## Heat integration (original 11 exchangers)
11 DWSIM heat exchangers recover **107 MW** from the hot product draws (kerosene, diesel, AGO, LVGO, HVGO, vacuum residue)
into three crude trains: pre-desalter crude (H7â€“H11), desalted crude (H3â€“H6) and stabilised crude to the atmospheric furnace (H1â€“H2).
Minimum approach 14.2 K. Heater duty falls from **138.5 MW to 31.6 MW (âˆ’77 %)**: E-100 trim 5.9, H101 trim 3.9, FH-100 3.1, E-102 trim 18.7 MW.
FH-100 fuel scales down with its 3.1 MW duty. Product slate, mass and sulphur balances are unchanged from the baseline.
How it is solved: a hot stream used in two crude trains (kerosene, diesel, HVGO, vacuum residue) forms a true thermal recycle, so
the second use is fed by a *tear copy* of the first use's outlet. The build iterates the tear temperatures (converged to 0.06 K).
Exchanger targets were designed on enthalpy curves (`hx_network.json`, greedy design, MITA â‰¥ 15 K, hot outlets not below product-spec temperatures).
Product trim coolers remain on LVGO (11 MW less heat than the utility case) etc.; vacuum-residue is pumped to 12 bar and HVGO/LVGO to 5 bar
because DWSIM's flash fails for the heavy vacuum cuts at low pressure in some temperature windows.

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
Mass balance closes to 100.000 %; sulphur balance closes (16,860 kg/h in = out).
Yields (wt % of crude): sweet naphtha 8.7, kerosene 14.2, diesel 13.9, AGO 5.3, LVGO 7.4, HVGO 15.5, vacuum residue 33.8,
propane 0.2, C4 LPG 0.4, fuel gas 0.3.