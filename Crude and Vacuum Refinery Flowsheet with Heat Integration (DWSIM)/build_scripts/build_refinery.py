"""Crude + vacuum refinery model in DWSIM (hybrid: cut-point towers for crude/vacuum, rigorous columns for light ends)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flowsheet_tools import *
import assay
from System.Collections.Generic import Dictionary
from pfd_layout import LANES
from pfd_layout import apply_pfd

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # models are saved one level up
HI = os.environ.get("HEAT_INTEGRATION", "1") == "1"   # 1 = heat-recovery network, 0 = utility preheat/cooling
EXTRA =["Nitrogen", "Oxygen", "Carbon dioxide", "Sulfur dioxide", "Hydrogen", "Sulfur",
         "Dimethyl disulfide", "Diethyl disulfide", "1-(Butyldisulfanyl)Butane"]

S = {}      # streams by tag
U = {}      # units by tag
E = {}      # energy streams by tag


def mole_to_mass(P, x):
    m = {k: v * P.mw[k] for k, v in x.items()}
    t = sum(m.values())
    return {k: v / t for k, v in m.items()}


def make_plant():
    w, ps = assay.wt_fractions(coarse=True)
    P = Plant("Peng-Robinson (PR)")
    P.add_real(list(assay.REAL_WT) + EXTRA)
    for n, t, sg, wt, s, mw, tc, pc, om in ps:
        P.add_pseudo(n, t + 273.15, sg, mw, tc, pc, om)
    P.finish_compounds()
    P.w_crude = w
    P.pseudo_names = [p[0] for p in ps]
    return P


def add_combustion(P):
    fs = P.fs

    def conv(name, stoich, base, x=99.9):
        d = Dictionary[str, float]()
        for k, v in stoich.items():
            d[k] = v
        r = fs.CreateConversionReaction(name, name, d, base, "Vapor", str(x))
        fs.AddReaction(r)
        return r
    rs = [conv("CH4 comb", {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}, "Methane"),
          conv("C2H6 comb", {"Ethane": -1, "Oxygen": -3.5, "Carbon dioxide": 2, "Water": 3}, "Ethane"),
          conv("C3H8 comb", {"Propane": -1, "Oxygen": -5, "Carbon dioxide": 3, "Water": 4}, "Propane"),
          conv("H2 comb", {"Hydrogen": -1, "Oxygen": -0.5, "Water": 1}, "Hydrogen"),
          conv("H2S comb", {"Hydrogen sulfide": -1, "Oxygen": -1.5, "Sulfur dioxide": 1, "Water": 1}, "Hydrogen sulfide")]
    rset = fs.CreateReactionSet("Combustion", "Fuel gas combustion")
    fs.AddReactionSet(rset)
    for i, r in enumerate(rs):
        fs.AddReactionToSet(r.ID, rset.ID, True, i)
    P.rset_comb = rset


# ------------------------------------------------------------------ A: feed, desalter, preheat, FH-100
def section_A(P):
    w = P.w_crude
    S["Crude Oil"] = P.stream("Crude Oil", 40, 3.0, assay.FEED_KG_H, w)
    S["Wash Water"] = P.stream("Wash Water", 80, 16.0, 0.045 * assay.FEED_KG_H, {"Water": 1.0})

    S["Crude Oil P"] = P.stream("Crude Oil P")
    U["P-100"], E["W-P100"] = pump(P, "P-100", S["Crude Oil"], S["Crude Oil P"], 18.0 if HI else 16.0)
    S["Crude Oil Pre"] = P.stream("Crude Oil Pre")
    if not HI:
        U["E-100"], E["E0"] = heater(P, "E-100 Preheat", S["Crude Oil P"], S["Crude Oil Pre"], T=145.0, dP=1.0)
    S["Desalter Feed Mix"] = P.stream("Desalter Feed Mix")
    U["M-100"] = mixer(P, "M-100 Wash Water Mixer", [S["Crude Oil Pre"], S["Wash Water"]], S["Desalter Feed Mix"])
    S["Desalter Feed"] = P.stream("Desalter Feed")
    U["V-100"] = valve(P, "V-100 Mixing Valve", S["Desalter Feed Mix"], S["Desalter Feed"], dP=1.0)
    S["Desalted Crude"] = P.stream("Desalted Crude")
    S["Brine"] = P.stream("Brine")
    # electrostatic desalter: ~97.8 % of the water and 95 % of the salt leave as brine; ~0.1 wt% BS&W remains
    f2 = {c: 0.0 for c in P.comps}
    f2["Water"] = 0.978
    f2["Sodium Chloride"] = 0.95
    U["D-100"] = splitter_components(P, "D-100 Desalter", S["Desalter Feed"], S["Desalted Crude"], S["Brine"], f2)
    S["Crude Oil H1"] = P.stream("Crude Oil H1")
    if not HI:
        U["H101"], E["E1"] = heater(P, "H101", S["Desalted Crude"], S["Crude Oil H1"], T=205.0, dP=1.0)

    # fired heater FH-100: process side + combustion side
    S["Crude Oil H2"] = P.stream("Crude Oil H2")
    U["FH-100"], E["FH-100 duty"] = heater(P, "FH-100", S["Crude Oil H1"], S["Crude Oil H2"], T=255.0, dP=3.0)
    fuel_x = {"Methane": 0.55, "Ethane": 0.20, "Propane": 0.08, "Hydrogen": 0.15, "Hydrogen sulfide": 0.02}
    S["Fuel Gas"] = P.stream("Fuel Gas", 40, 4.0, 1500.0, mole_to_mass(P, fuel_x))
    o2_per_mol = 0.55 * 2 + 0.20 * 3.5 + 0.08 * 5 + 0.15 * 0.5 + 0.02 * 1.5
    S["Air"] = P.stream("Air", 25, 1.2, 25000.0, {"Oxygen": 0.233, "Nitrogen": 0.767})
    S["Combustion Feed"] = P.stream("Combustion Feed")
    U["M-FH"] = mixer(P, "M-FH Burner", [S["Fuel Gas"], S["Air"]], S["Combustion Feed"])
    S["Flue Gas"] = P.stream("Flue Gas")
    S["Flue Gas liq"] = P.stream("Flue Gas liq")
    rc = P.add(OT.RCT_Conversion, "FH-100 Firebox")
    P.connect(S["Combustion Feed"], rc, 0, 0)
    P.connect(rc, S["Flue Gas"], 0, 0)
    P.connect(rc, S["Flue Gas liq"], 1, 0)
    E["FH-100 fire"] = P.energy("FH-100 fire duty")
    P.connect(rc, E["FH-100 fire"], 2, 0)
    rc.ReactionSetID = P.rset_comb.ID
    set_enum(rc, "ReactorOperationMode", "OutletTemperature")
    rc.OutletTemperature = 320.0 + 273.15
    U["FH-100 Firebox"] = rc
    P.fuel_o2_per_mol = o2_per_mol
    P.fuel_x = fuel_x


def tune_fuel(P):
    """Scale fuel/air so firebox heat release matches the crude duty (flowsheet is linear in fuel flow)."""
    q_proc = E["FH-100 duty"].EnergyFlow
    q_fire = E["FH-100 fire"].EnergyFlow
    if not q_fire or not q_proc:
        print("  FH-100 not yet calculated; fuel not tuned")
        return
    q_proc, q_fire = abs(q_proc), abs(q_fire)
    k = q_proc / q_fire
    if "SP-FG" in U:                       # fuel now comes from the plant's own sweet fuel gas through the splitter
        U["SP-FG"].StreamFlowSpec = U["SP-FG"].StreamFlowSpec * k
        S["Air"].Phases[0].Properties.massflow = S["Air"].Phases[0].Properties.massflow * k
    else:
        for tag in ("Fuel Gas", "Air"):
            S[tag].Phases[0].Properties.massflow = S[tag].Phases[0].Properties.massflow * k
    print(f"  FH-100: process duty {q_proc/1000:.2f} MW, firebox {q_fire/1000:.2f} MW -> fuel/air scaled x{k:.3f}")


# ------------------------------------------------------------------ B: PF tower, debutanisers, atmospheric unit
def heavy_split(P, tcut, width, pseudo_heavy=True, over=None):
    """Heavy-fraction map for a TBP cut; pseudos go 100% heavy at cuts below their range when pseudo_heavy."""
    f = cut_fraction(P, tcut, width)
    f["Sodium Chloride"] = 1.0
    if pseudo_heavy:
        for n in P.pseudo_names:
            f[n] = 1.0
    if over:
        f.update(over)
    return f


def section_B(P):
    # letdown to PF pressure
    S["PF Feed"] = P.stream("PF Feed")
    U["V-101"] = valve(P, "V-101 PF Letdown", S["Crude Oil H2"], S["PF Feed"], Pout=4.5)
    # PRE-FRACTIONATING COLUMN (cut-point): light ends + naphtha overhead, stabilised crude bottoms
    S["PF Overhead"] = P.stream("PF Overhead")
    S["PF Bottoms"] = P.stream("PF Bottoms")
    f = heavy_split(P, 150.0, 8.0)
    U["PF"] = splitter_components(P, "PRE-FRACTIONATING COLUMN", S["PF Feed"], S["PF Overhead"], S["PF Bottoms"], f)
    # partial condenser E2 + reflux drum
    S["PF Overhead Cooled"] = P.stream("PF Overhead Cooled")
    cond_in = S["PF Overhead"]
    if HI:      # overhead vapour first preheats the crude (exchanger H13, built in heat_integration), then is condensed
        S["PF Overhead HX"] = P.stream("PF Overhead HX")
        cond_in = S["PF Overhead HX"]
    U["E2"], E["E2"] = heater(P, "PF Condenser", cond_in, S["PF Overhead Cooled"], T=55.0, dP=0.3, cooler=True)
    S["PF OVHD Vapor"] = P.stream("PF OVHD Vapor")
    S["PF OVHD Liquid"] = P.stream("PF OVHD Liquid")
    S["PF Sour Water"] = P.stream("PF Sour Water")
    U["PF Drum"] = flash_drum(P, "PF Reflux Drum", S["PF Overhead Cooled"], vapor=S["PF OVHD Vapor"],
                              liquid=S["PF OVHD Liquid"], heavy=S["PF Sour Water"])
    # reboiler E3 keeps stabilised crude at 250 C
    S["Stabilised Crude"] = P.stream("Stabilised Crude")
    U["E3"], E["E3"] = heater(P, "PF Reboiler", S["PF Bottoms"], S["Stabilised Crude"], T=250.0)
    # E-102 furnace to atmospheric flash-zone temperature
    S["Stabilised Crude P"] = P.stream("Stabilised Crude P")
    U["P-102"], E["W-P102"] = pump(P, "P-102", S["Stabilised Crude"], S["Stabilised Crude P"], 10.0 if HI else 9.0)
    S["Stabilised Crude H1"] = P.stream("Stabilised Crude H1")
    if not HI:
        U["E-102"], E["E10"] = heater(P, "E-102 Atm Furnace", S["Stabilised Crude P"], S["Stabilised Crude H1"], T=355.0, dP=4.0)
    # ATMOSPHERIC UNIT (cut-point): OVHD / Kerosene / Diesel / AGO / residue
    S["Atm Feed"] = P.stream("Atm Feed")
    U["V-102"] = valve(P, "V-102 Atm Flash", S["Stabilised Crude H1"], S["Atm Feed"], Pout=2.2)
    stages = [("OVHD", 165.0, 8.0), ("Kerosene", 245.0, 8.0), ("Diesel", 340.0, 8.0), ("AGO", 385.0, 8.0)]
    rest = S["Atm Feed"]
    prods = {}
    for i, (nm, tcut, wd) in enumerate(stages):
        prod = P.stream("Atm %s Hot" % nm)
        nxt = P.stream("Atm Residue" if i == len(stages) - 1 else "Atm Rest %d" % (i + 1))
        sep = splitter_components(P, "Atm Cut %d (%s)" % (i + 1, nm), rest, prod, nxt, heavy_split(P, tcut, wd, pseudo_heavy=False, over={"Water": 0.0, "Hydrogen sulfide": 0.0}))
        U["Atm Cut %d" % (i + 1)] = sep
        prods[nm] = prod
        rest = nxt
    S["Atm Residue"] = rest
    # product coolers
    S["OVHD Cooled"] = P.stream("OVHD Cooled")
    U["E11"], E["E11"] = heater(P, "Atm Condenser", prods["OVHD"], S["OVHD Cooled"], T=45.0, dP=0.5, cooler=True)
    S["Atm OVHD Gas"] = P.stream("Atm OVHD Gas")
    S["OVHD Liquid"] = P.stream("OVHD Liquid")
    S["Atm Sour Water"] = P.stream("Atm Sour Water")
    U["Atm Drum"] = flash_drum(P, "Atm Reflux Drum", S["OVHD Cooled"], vapor=S["Atm OVHD Gas"],
                               liquid=S["OVHD Liquid"], heavy=S["Atm Sour Water"])
    S["Kerosene"] = P.stream("Kerosene")
    S["Diesel"] = P.stream("Diesel")
    S["AGO"] = P.stream("AGO")
    if not HI:
        U["Ck"], E["Ek"] = heater(P, "Kerosene Cooler", prods["Kerosene"], S["Kerosene"], T=60.0, cooler=True)
        U["Cd"], E["Ed"] = heater(P, "Diesel Cooler", prods["Diesel"], S["Diesel"], T=70.0, cooler=True)
        U["Ca"], E["Ea"] = heater(P, "AGO Cooler", prods["AGO"], S["AGO"], T=90.0, cooler=True)


# ------------------------------------------------------------------ C: debutanisers (rigorous)
LIGHT_C4 = ["Methane", "Ethane", "Propane", "Isobutane", "n-Butane", "Hydrogen sulfide", "Methyl mercaptan",
            "Hydrogen", "Nitrogen", "Carbon dioxide"]


def section_C_feeds(P):
    S["PF Vapor Comp"] = P.stream("PF Vapor Comp")
    U["K-100"], E["W-K100"] = compressor(P, "K-100 PF Vapor Compressor", S["PF OVHD Vapor"], S["PF Vapor Comp"], 21.5)
    S["PF Vapor Cooled"] = P.stream("PF Vapor Cooled")
    U["E-100"], E["E4"] = heater(P, "E-100", S["PF Vapor Comp"], S["PF Vapor Cooled"], T=40.0, dP=0.3, cooler=True)
    S["PF Vapor C1"] = P.stream("PF Vapor C1")
    S["PF Vapor Sour Water"] = P.stream("PF Vapor Sour Water")
    water = {c: 0.0 for c in P.comps}
    water["Water"] = 0.999
    U["WKO-1"] = splitter_components(P, "WKO-1 Water Decanter", S["PF Vapor Cooled"], S["PF Vapor C1"], S["PF Vapor Sour Water"], water)
    S["PF Liquid P"] = P.stream("PF Liquid P")
    U["P-103"], E["W-P103"] = pump(P, "P-103", S["PF OVHD Liquid"], S["PF Liquid P"], 15.5)
    S["PF Liquid Cooled"] = P.stream("PF Liquid Cooled")
    U["E-101"], E["E9"] = heater(P, "E-101", S["PF Liquid P"], S["PF Liquid Cooled"], T=40.0, dP=0.3, cooler=True)
    S["PF OVHD Liquid C1"] = P.stream("PF OVHD Liquid C1")
    S["PF Liquid Sour Water"] = P.stream("PF Liquid Sour Water")
    U["WKO-2"] = splitter_components(P, "WKO-2 Water Decanter", S["PF Liquid Cooled"], S["PF OVHD Liquid C1"], S["PF Liquid Sour Water"], water)


def section_C_cols(P):
    for tag in ("Naphtha", "Naphtha 2", "DB1 LPG liquid", "DB1 Off gas", "DB2 LPG liquid", "DB2 Off gas", "LPG + FG", "LPG + FG S2"):
        S[tag] = P.stream(tag)
    U["DEBUTANISER"], E["E5"], E["E6"] = rigorous_column(
        P, "DEBUTANISER", S["PF Vapor C1"], S["DB1 LPG liquid"], S["Naphtha"], 12, 6, 20.0, 21.0, 1.0,
        condenser="Partial_Condenser", vapor=S["DB1 Off gas"])
    U["DEBUTANISER 2"], E["E7"], E["E8"] = rigorous_column(
        P, "DEBUTANISER 2", S["PF OVHD Liquid C1"], S["DB2 LPG liquid"], S["Naphtha 2"], 20, 10, 14.0, 15.0, 2.0,
        condenser="Partial_Condenser", vapor=S["DB2 Off gas"])
    if HI:      # naphtha bottoms heat the crude (exchanger H12) before the naphtha cooler
        U["RC-N2"] = recycle(P, S["Naphtha 2"], S["Naphtha 2 HX in"], "RC Naphtha 2")
    U["M-DB1"] = mixer(P, "M-DB1 LPG+FG", [S["DB1 LPG liquid"], S["DB1 Off gas"]], S["LPG + FG"])
    U["M-DB2"] = mixer(P, "M-DB2 LPG+FG S2", [S["DB2 LPG liquid"], S["DB2 Off gas"]], S["LPG + FG S2"])


def tune_C(P):
    for nm, feed in (("DEBUTANISER", S["PF Vapor C1"]), ("DEBUTANISER 2", S["PF OVHD Liquid C1"])):
        heavy = [c for c in P.comps if c not in LIGHT_C4]
        nb, n = tune_column(U[nm], feed, heavy, P.comps)
        ph = feed.Phases[0]
        vk = 1.0 if nm == "DEBUTANISER" else 2.0
        nv = vk * n * (ph.Compounds["Methane"].MoleFraction + ph.Compounds["Ethane"].MoleFraction)
        U[nm].VaporFlowRate = nv
        U[nm].VaporFlowRateUnit = "mol/s"
        print(f"  {nm}: feed {n:.2f} mol/s, bottoms spec {nb:.3f} mol/s, vapour product {nv:.2f} mol/s")


# ------------------------------------------------------------------ D: vacuum distillation unit (cut-point)
def cut_chain(P, prefix, feed, stages, residue_name):
    """Series of TBP cut-point splitters. stages: [(name, tcut_C, width_C, overrides)] -> (hot products dict, residue)."""
    rest, prods = feed, {}
    for i, (nm, tcut, wd, ov) in enumerate(stages):
        prod = P.stream("%s %s Hot" % (prefix, nm))
        last = i == len(stages) - 1
        nxt = P.stream(residue_name if last else "%s Rest %d" % (prefix, i + 1))
        U["%s Cut %d" % (prefix, i + 1)] = splitter_components(
            P, "%s Cut %d (%s)" % (prefix, i + 1, nm), rest, prod, nxt, heavy_split(P, tcut, wd, pseudo_heavy=False, over=ov))
        prods[nm] = prod
        rest = nxt
    S[residue_name] = rest
    return prods, rest


def section_D(P):
    S["VDU Feed P"] = P.stream("VDU Feed P")
    U["P-104"], E["W-P104"] = pump(P, "P-104", S["Atm Residue"], S["VDU Feed P"], 8.0)
    S["VDU Feed H"] = P.stream("VDU Feed H")
    U["VF-100"], E["E-VF"] = heater(P, "VF-100 Vacuum Furnace", S["VDU Feed P"], S["VDU Feed H"], T=405.0, dP=4.0)
    S["VDU Flash Zone"] = P.stream("VDU Flash Zone")
    U["V-103"] = valve(P, "V-103 Vacuum Flash", S["VDU Feed H"], S["VDU Flash Zone"], Pout=0.08)
    ov = {"Water": 0.0, "Hydrogen sulfide": 0.0}
    prods, res = cut_chain(P, "VDU", S["VDU Flash Zone"],
                           [("OVHD", 330.0, 8.0, ov), ("LVGO", 440.0, 10.0, ov), ("HVGO", 555.0, 10.0, ov)], "Vacuum Residue Hot")
    S["VDU OVHD Cooled"] = P.stream("VDU OVHD Cooled")
    U["E-VO"], E["E-VO"] = heater(P, "VDU Condenser", prods["OVHD"], S["VDU OVHD Cooled"], T=45.0, dP=-1.42, cooler=True)
    S["Vacuum Gas"] = P.stream("Vacuum Gas")
    S["Slop Distillate"] = P.stream("Slop Distillate")
    S["VDU Sour Water"] = P.stream("VDU Sour Water")
    U["VDU Drum"] = flash_drum(P, "VDU Overhead Drum", S["VDU OVHD Cooled"], vapor=S["Vacuum Gas"], liquid=S["Slop Distillate"],
                               heavy=S["VDU Sour Water"])
    S["LVGO"] = P.stream("LVGO")
    S["HVGO"] = P.stream("HVGO")
    S["Vacuum Residue"] = P.stream("Vacuum Residue")
    if not HI:
        U["C-LVGO"], E["E-LVGO"] = heater(P, "LVGO Cooler", prods["LVGO"], S["LVGO"], T=90.0, dP=-2.92, cooler=True)
        U["C-HVGO"], E["E-HVGO"] = heater(P, "HVGO Cooler", prods["HVGO"], S["HVGO"], T=130.0, dP=-2.92, cooler=True)
        U["C-VR"], E["E-VR"] = heater(P, "VR Cooler", res, S["Vacuum Residue"], T=200.0, dP=-2.92, cooler=True)


# ------------------------------------------------------------------ E: gas plant (amine, deethaniser, depropaniser)
def amine_unit(P, tag, feed, treated, acid, h2s=0.998, co=0.0015):
    """Amine contactor + regenerator lumped as a component splitter: H2S to acid gas, small hydrocarbon co-absorption."""
    f = {c: 0.0 for c in P.comps}
    f["Hydrogen sulfide"] = h2s
    f["Carbon dioxide"] = 0.9
    f["Methyl mercaptan"] = 0.15
    for c in ("Methane", "Ethane", "Propane"):
        f[c] = co
    return splitter_components(P, tag, feed, treated, acid, f)


def section_E_feeds(P):
    S["LPG + FG L"] = P.stream("LPG + FG L")
    U["V-104"] = valve(P, "V-104 DB1 LPG Letdown", S["LPG + FG"], S["LPG + FG L"], Pout=14.0)
    S["Raw LPG + FG"] = P.stream("Raw LPG + FG")
    U["M-LPG"] = mixer(P, "M-LPG", [S["LPG + FG L"], S["LPG + FG S2"], S["Atm OVHD Gas"]], S["Raw LPG + FG"])
    S["Sweet LPG + FG"] = P.stream("Sweet LPG + FG")
    S["Acid Gas 1"] = P.stream("Acid Gas 1")
    U["AM-1"] = amine_unit(P, "AM-1 LPG Amine Treater", S["Raw LPG + FG"], S["Sweet LPG + FG"], S["Acid Gas 1"])
    S["LPG Cooled"] = P.stream("LPG Cooled")
    U["E-103"], E["E-103"] = heater(P, "E-103", S["Sweet LPG + FG"], S["LPG Cooled"], T=30.0, dP=0.2, cooler=True)
    S["LPG Flash Gas"] = P.stream("LPG Flash Gas")
    S["LPG Liquid"] = P.stream("LPG Liquid")
    S["LPG Water"] = P.stream("LPG Water")
    U["D-103"] = flash_drum(P, "D-103 LPG Feed Drum", S["LPG Cooled"], vapor=S["LPG Flash Gas"], liquid=S["LPG Liquid"], heavy=S["LPG Water"])
    S["LPG Liquid Dry"] = P.stream("LPG Liquid Dry")
    S["LPG Dryer Water"] = P.stream("LPG Dryer Water")
    wd = {c: 0.0 for c in P.comps}
    wd["Water"] = 0.99999
    U["WKO-4"] = splitter_components(P, "WKO-4 LPG Dryer", S["LPG Liquid"], S["LPG Liquid Dry"], S["LPG Dryer Water"], wd)
    S["DE Feed"] = P.stream("DE Feed")
    U["P-105"], E["W-P105"] = pump(P, "P-105", S["LPG Liquid Dry"], S["DE Feed"], 34.0)


def section_E_de(P):
    S["DE Dist"] = P.stream("DE Distillate (C1/C2 rich)")
    S["C3+ Bottoms"] = P.stream("C3+ Bottoms")
    U["DEETHANISER"], E["E-DEc"], E["E-DEr"] = rigorous_column(
        P, "DEETHANISER", S["DE Feed"], S["DE Dist"], S["C3+ Bottoms"], 20, 10, 32.0, 33.0, 3.0, condenser="Total_Condenser")
    S["DE Off Gas"] = P.stream("DE Off Gas")
    U["V-106"] = valve(P, "V-106 DE Distillate Letdown", S["DE Dist"], S["DE Off Gas"], Pout=10.0)

def section_E_dp(P):
    S["DP Feed"] = P.stream("DP Feed")
    U["V-105"] = valve(P, "V-105 DE Bottoms Letdown", S["C3+ Bottoms"], S["DP Feed"], Pout=17.5)
    S["Propane"] = P.stream("Propane")
    S["C4 LPG"] = P.stream("C4 LPG")
    U["DEPROPANISER"], E["E-DPc"], E["E-DPr"] = rigorous_column(
        P, "DEPROPANISER", S["DP Feed"], S["Propane"], S["C4 LPG"], 24, 12, 16.5, 17.5, 2.5)


LIGHT_DE = ["Methane", "Ethane", "Hydrogen", "Nitrogen", "Carbon dioxide"]
LIGHT_DP = ["Methane", "Ethane", "Propane", "Hydrogen", "Nitrogen", "Carbon dioxide", "Hydrogen sulfide"]


def tune_DE(P):
    # deethaniser: bottoms = everything except C1/C2; vapour product = all of C1+C2 (condenser vapour)
    feed = S["DE Feed"]
    ph = feed.Phases[0]
    n = ph.Properties.molarflow
    nl = n * sum(ph.Compounds[c].MoleFraction for c in LIGHT_DE)
    U["DEETHANISER"].Specs["R"].SpecValue = max(n - 1.5 * nl, 1e-6)   # distillate carries C1/C2 plus ~0.5x propane
    print(f"  DEETHANISER: feed {n:.2f} mol/s, C1+C2 {nl:.3f} mol/s")


def tune_DP(P):
    fd = S["C3+ Bottoms"].Phases[0]
    n2 = fd.Properties.molarflow
    npro = n2 * sum(fd.Compounds[c].MoleFraction for c in LIGHT_DP)
    U["DEPROPANISER"].Specs["R"].SpecValue = max(n2 - npro, 1e-6)
    print(f"  DEPROPANISER: feed {n2:.2f} mol/s, C3- {npro:.3f} mol/s")


# ------------------------------------------------------------------ F: MEROX (naphtha sweetening, LPG extraction) and fuel-gas amine
THIOLS = {"Methyl mercaptan": "Dimethyl disulfide", "Ethyl mercaptan": "Diethyl disulfide",
          "Butane-1-Thiol": "1-(Butyldisulfanyl)Butane"}


def add_reactions(P):
    fs = P.fs

    def conv(name, stoich, base, x, phase="Liquid"):
        d = Dictionary[str, float]()
        for k, v in stoich.items():
            d[k] = v
        r = fs.CreateConversionReaction(name, name, d, base, phase, str(x))
        fs.AddReaction(r)
        return r

    def rset(name, reactions):
        s = fs.CreateReactionSet(name, name)
        fs.AddReactionSet(s)
        for i, r in enumerate(reactions):
            fs.AddReactionToSet(r.ID, s.ID, True, i)
        return s
    # MEROX oxidation: 4 RSH + O2 -> 2 RSSR + 2 H2O
    ox = [conv("MEROX " + t, {t: -4, "Oxygen": -1, d: 2, "Water": 2}, t, 99.0, "Mixture") for t, d in THIOLS.items()]
    P.rset_merox = rset("MEROX oxidation", ox)
    # Claus: thermal reactor (1/3 H2S burnt, then Claus reaction), two catalytic stages, tail-gas incinerator
    th1 = conv("Claus H2S oxidation", {"Hydrogen sulfide": -1, "Oxygen": -1.5, "Sulfur dioxide": 1, "Water": 1}, "Oxygen", 100.0, "Vapor")
    hc = [conv("Claus CH4 burn", {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}, "Methane", 99.9, "Vapor"),
          conv("Claus C2H6 burn", {"Ethane": -1, "Oxygen": -3.5, "Carbon dioxide": 2, "Water": 3}, "Ethane", 99.9, "Vapor"),
          conv("Claus C3H8 burn", {"Propane": -1, "Oxygen": -5, "Carbon dioxide": 3, "Water": 4}, "Propane", 99.9, "Vapor")]
    cl = {k: conv("Claus reaction " + k, {"Hydrogen sulfide": -2, "Sulfur dioxide": -1, "Sulfur": 3, "Water": 2},
                  "Hydrogen sulfide", x, "Vapor") for k, x in (("thermal", 62.0), ("cat1", 70.0), ("cat2", 60.0))}
    hc.append(conv("Claus MeSH burn", {"Methyl mercaptan": -1, "Oxygen": -3, "Carbon dioxide": 1, "Sulfur dioxide": 1, "Water": 2}, "Methyl mercaptan", 99.9, "Vapor"))
    P.rset_claus_th = rset("Claus thermal", hc + [th1, cl["thermal"]])
    P.rset_claus_c1 = rset("Claus cat 1", [cl["cat1"]])
    P.rset_claus_c2 = rset("Claus cat 2", [cl["cat2"]])
    inc = [conv("Tail gas H2S burn", {"Hydrogen sulfide": -1, "Oxygen": -1.5, "Sulfur dioxide": 1, "Water": 1}, "Hydrogen sulfide", 99.9, "Vapor"),
           conv("Tail gas MeSH burn", {"Methyl mercaptan": -1, "Oxygen": -3, "Carbon dioxide": 1, "Sulfur dioxide": 1, "Water": 2}, "Methyl mercaptan", 99.9, "Vapor")]
    P.rset_incin = rset("Tail gas incineration", inc)


def reactor(P, tag, inlet, vap_out, liq_out, rset, T_c, dP=0.0):
    r = P.add(OT.RCT_Conversion, tag)
    P.connect(inlet, r, 0, 0)
    P.connect(r, vap_out, 0, 0)
    P.connect(r, liq_out, 1, 0)
    if USE_ENERGY_STREAMS:
        e = P.energy("Q_" + tag)
        P.connect(r, e, 2, 0)
    else:
        e = DutyRef(r)
    r.ReactionSetID = rset.ID
    set_enum(r, "ReactorOperationMode", "OutletTemperature")
    r.OutletTemperature = T_c + 273.15
    r.DeltaP = dP * 1e5
    return r, e


def section_F_naphtha(P):
    # bring each naphtha stream to 45 C / 8 bar before the header
    hdr = []
    for tag, pump_needed in (("Naphtha", False), ("Naphtha 2", False), ("OVHD Liquid", True)):
        src = S["Naphtha 2 HX"] if (HI and tag == "Naphtha 2") else S[tag]
        if tag != "OVHD Liquid":
            cooled = P.stream(tag + " Cooled 45C")
            U["EN-" + tag], E["EN-" + tag] = heater(P, "EN-%s Cooler" % tag, src, cooled, T=45.0, dP=0.5, cooler=True)
            low = P.stream(tag + " 8 bar")
            U["VN-" + tag] = valve(P, "VN-%s Letdown" % tag, cooled, low, Pout=8.0)
            hdr.append(low)
        else:
            pumped = P.stream(tag + " 8 bar")
            U["PN-" + tag], E["WN-" + tag] = pump(P, "PN-%s Pump" % tag, src, pumped, 8.0)
            hdr.append(pumped)
    S["Combined Naphtha"] = P.stream("Combined Naphtha")
    U["M-NAP"] = mixer(P, "M-NAP Naphtha Header", hdr, S["Combined Naphtha"])
    S["Naphtha Cooled"] = P.stream("Naphtha Cooled")
    U["E-NAP"], E["E-NAP"] = heater(P, "E-NAP Naphtha Trim Cooler", S["Combined Naphtha"], S["Naphtha Cooled"], T=40.0, dP=0.3, cooler=True)
    S["MEROX Air N"] = P.stream("MEROX Air (Naphtha)", 30, 8.0, 10.0, {"Oxygen": 0.233, "Nitrogen": 0.767})
    S["Naphtha + Air"] = P.stream("Naphtha + Air")
    U["M-NAP2"] = mixer(P, "M-NAP2", [S["Naphtha Cooled"], S["MEROX Air N"]], S["Naphtha + Air"])
    S["MEROX Vent N"] = P.stream("MEROX Vent (Naphtha)")
    S["Sweet Naphtha Raw"] = P.stream("Sweet Naphtha Raw")
    U["R-MX1"], E["R-MX1"] = reactor(P, "R-MX1 Naphtha Sweetening Reactor", S["Naphtha + Air"], S["MEROX Vent N"], S["Sweet Naphtha Raw"],
                                     P.rset_merox, 40.0)
    S["Sweet Naphtha"] = P.stream("Sweet Naphtha")
    S["Naphtha Water"] = P.stream("Naphtha Water")
    w = {c: 0.0 for c in P.comps}
    w["Water"] = 0.999
    U["WKO-3"] = splitter_components(P, "WKO-3 Naphtha Water Wash/Decant", S["Sweet Naphtha Raw"], S["Sweet Naphtha"], S["Naphtha Water"], w)

def tune_F_naphtha(P):
    """Air for the naphtha sweetener: 1.5 x stoichiometric O2 (4 RSH + O2)."""
    ph = S["Naphtha Cooled"].Phases[0]
    n = ph.Properties.molarflow
    n_rsh = n * sum(ph.Compounds[t].MoleFraction for t in THIOLS)
    o2 = 1.5 * n_rsh / 4.0
    air_mol = o2 / 0.21
    S["MEROX Air N"].Phases[0].Properties.massflow = max(air_mol * 28.96e-3, 1e-6)
    print(f"  Naphtha MEROX: thiols {n_rsh:.4f} mol/s -> air {air_mol*28.96e-3*3600:.2f} kg/h")


def section_F_lpg(P):
    """LPG MEROX: caustic extraction of mercaptans, oxidation regeneration to disulfide oil."""
    for nm, src, ex in (("C3", "Propane", {"Methyl mercaptan": 0.98, "Ethyl mercaptan": 0.97}),
                        ("C4", "C4 LPG", {"Methyl mercaptan": 0.98, "Ethyl mercaptan": 0.97, "Butane-1-Thiol": 0.90})):
        S["Sweet " + src] = P.stream("Sweet " + src)
        S["RSH Caustic " + nm] = P.stream("RSH Caustic (%s)" % nm)
        U["EX-" + nm] = splitter_components(P, "EX-%s LPG Mercaptan Extractor" % nm, S[src], S["Sweet " + src], S["RSH Caustic " + nm], ex)
    S["RSH Caustic"] = P.stream("RSH Caustic Total")
    U["M-CAUS"] = mixer(P, "M-CAUS", [S["RSH Caustic C3"], S["RSH Caustic C4"]], S["RSH Caustic"])
    S["MEROX Air L"] = P.stream("MEROX Air (Regenerator)", 30, 18.0, 1.0, {"Oxygen": 0.233, "Nitrogen": 0.767})
    S["Regen Feed"] = P.stream("Regen Feed")
    U["M-REG"] = mixer(P, "M-REG", [S["RSH Caustic"], S["MEROX Air L"]], S["Regen Feed"])
    S["Regen Vent"] = P.stream("Regen Vent")
    S["Regen Liquid"] = P.stream("Regen Liquid")
    U["R-MX2"], E["R-MX2"] = reactor(P, "R-MX2 Caustic Regenerator Oxidiser", S["Regen Feed"], S["Regen Vent"], S["Regen Liquid"],
                                     P.rset_merox, 40.0)
    S["Disulfide Oil"] = P.stream("Disulfide Oil")
    d = {c: 0.0 for c in P.comps}
    for t, ds in THIOLS.items():
        d[ds] = 1.0
    d["Water"] = 0.0
    S["Regen Water"] = P.stream("Regen Water")
    U["DS-1"] = splitter_components(P, "DS-1 Disulfide Separator", S["Regen Liquid"], S["Regen Water"], S["Disulfide Oil"], d)


def tune_F_lpg(P):
    n_rsh = 0.0
    for key in ("RSH Caustic",):
        ph = S[key].Phases[0]
        n = ph.Properties.molarflow or 0.0
        n_rsh = n * sum(ph.Compounds[t].MoleFraction for t in THIOLS)
    o2 = 1.5 * n_rsh / 4.0
    S["MEROX Air L"].Phases[0].Properties.massflow = max(o2 / 0.21 * 28.96e-3, 1e-6)
    print(f"  LPG MEROX: thiols to regenerator {n_rsh:.4f} mol/s")


# ------------------------------------------------------------------ G: fuel-gas amine + Claus
def section_G1(P):
    S["Off Gas Header"] = P.stream("Off Gas Header")
    S["DE Off Gas Warm"] = P.stream("DE Off Gas Warm")
    U["E-OG"], E["E-OG"] = heater(P, "E-OG Off Gas Warmer", S["DE Off Gas"], S["DE Off Gas Warm"], T=30.0, dP=0.2)
    S["LPG Flash Gas L"] = P.stream("LPG Flash Gas (10 bar)")
    U["V-107"] = valve(P, "V-107 Flash Gas Letdown", S["LPG Flash Gas"], S["LPG Flash Gas L"], Pout=9.5)
    U["M-FG"] = mixer(P, "M-FG Off Gas Header", [S["DE Off Gas Warm"], S["LPG Flash Gas L"]], S["Off Gas Header"])
    S["Sweet Fuel Gas"] = P.stream("Sweet Fuel Gas")
    S["Acid Gas 2"] = P.stream("Acid Gas 2")
    U["AM-2"] = amine_unit(P, "AM-2 Fuel Gas Amine Treater", S["Off Gas Header"], S["Sweet Fuel Gas"], S["Acid Gas 2"], h2s=0.999)
    # plant fuel gas: the firebox burns part of the sweet fuel gas, the rest is exported
    S["Sweet FG Export"] = P.stream("Sweet Fuel Gas Export")
    fg0 = S["Fuel Gas"].Phases[0].Properties.massflow * 3600
    U["SP-FG"] = stream_splitter(P, "SP-FG Fuel Gas Splitter", S["Sweet Fuel Gas"], S["Fuel Gas"], S["Sweet FG Export"], fg0)
    S["Claus Feed Calc"] = P.stream("Claus Feed Calc (amine acid gas total)")
    S["Acid Gas 1 L"] = P.stream("Acid Gas 1 (2 bar)")
    S["Acid Gas 2 L"] = P.stream("Acid Gas 2 (2 bar)")
    U["V-AG1"] = valve(P, "V-AG1 Acid Gas Letdown", S["Acid Gas 1"], S["Acid Gas 1 L"], Pout=2.0)
    U["V-AG2"] = valve(P, "V-AG2 Acid Gas Letdown", S["Acid Gas 2"], S["Acid Gas 2 L"], Pout=2.0)
    U["M-CL"] = mixer(P, "M-CL Acid Gas Header", [S["Acid Gas 1 L"], S["Acid Gas 2 L"]], S["Claus Feed Calc"])


def section_G2(P):
    # The sulphur unit gets its own battery-limit feed stream, set from the solved amine acid gas. DWSIM cannot
    # order a flowsheet chain this deep (reports an infinite loop), so the SRU is decoupled from the crude-unit chain.
    cf = S["Claus Feed Calc"].Phases[0]
    mf = {c: (cf.Compounds[c].MassFraction or 0.0) for c in P.comps}
    S["Claus Feed"] = P.stream("Claus Feed", cf.Properties.temperature - 273.15, cf.Properties.pressure / 1e5,
                               (cf.Properties.massflow or 0.0) * 3600, mf)
    U["RC-Claus"] = recycle(P, S["Claus Feed Calc"], S["Claus Feed"], "RC Claus Feed")   # real connection to the amine acid gas
    ph0 = S["Claus Feed"].Phases[0]
    n0 = ph0.Properties.molarflow or 0.0
    o2_0 = n0 * (0.5 * ph0.Compounds["Hydrogen sulfide"].MoleFraction + 2 * ph0.Compounds["Methane"].MoleFraction
                 + 3.5 * ph0.Compounds["Ethane"].MoleFraction + 5 * ph0.Compounds["Propane"].MoleFraction
                 + 3 * ph0.Compounds["Methyl mercaptan"].MoleFraction) * 1.15
    S["Claus Air"] = P.stream("Claus Air Feed", 40, 3.0, max(o2_0 / 0.21 * 28.96e-3 * 3600, 5.0), {"Oxygen": 0.233, "Nitrogen": 0.767})
    S["Claus Th Feed"] = P.stream("Claus Thermal Feed")
    U["M-CL2"] = mixer(P, "M-CL2 Burner Mixer", [S["Claus Feed"], S["Claus Air"]], S["Claus Th Feed"])
    S["Claus Th Out"] = P.stream("Claus Thermal Out")
    S["Claus Th Liq"] = P.stream("Claus Thermal Liq")
    U["R-CL1"], E["R-CL1"] = reactor(P, "R-CL1 Thermal Reactor", S["Claus Th Feed"], S["Claus Th Out"], S["Claus Th Liq"], P.rset_claus_th, 1100.0)
    stage_in = S["Claus Th Out"]
    sulfur_streams = [S["Claus Th Liq"]]
    sf = {c: 0.0 for c in P.comps}
    sf["Sulfur"] = 1.0
    for i, (T_con, T_reac, rs) in enumerate(((160.0, 250.0, P.rset_claus_c1),), start=1):
        P  # readability
        cond = P.stream("Claus Cond %d Gas" % (i - 1)); sl = P.stream("Liquid Sulfur %d" % (i - 1))
        U["CD%d" % (i - 1)], E["CD%d" % (i - 1)] = heater(P, "CD-%d Sulfur Condenser" % (i - 1), stage_in, P.stream("Claus Cond %d Out" % (i - 1)), T=T_con, dP=0.1, cooler=True)
        co = P.objs["Claus Cond %d Out" % (i - 1)]
        U["SS%d" % (i - 1)] = splitter_components(P, "SS-%d Sulfur Separator" % (i - 1), co, cond, sl, sf)
        sulfur_streams.append(sl)
        ro = P.stream("Claus Cat %d Out" % i); rl = P.stream("Claus Cat %d Liq" % i)
        # reactor holds its own outlet temperature, so its duty includes the reheat
        U["R-CL%d" % (i + 1)], E["R-CL%d" % (i + 1)] = reactor(P, "R-CL%d Catalytic Reactor %d (incl. reheat)" % (i + 1, i), cond, ro, rl, rs, T_reac)
        sulfur_streams.append(rl)
        stage_in = ro
    cond = P.stream("Claus Cond 2 Gas")
    sl = P.stream("Liquid Sulfur 2")
    U["CD2"], E["CD2"] = heater(P, "CD-3 Final Sulfur Condenser", stage_in, P.stream("Claus Cond 2 Out"), T=125.0, dP=0.1, cooler=True)
    U["SS2"] = splitter_components(P, "SS-3 Sulfur Separator", P.objs["Claus Cond 2 Out"], cond, sl, sf)
    sulfur_streams.append(sl)
    S["Liquid Sulfur"] = P.stream("Liquid Sulfur")
    U["M-S"] = mixer(P, "M-S Sulfur Header", sulfur_streams + [], S["Liquid Sulfur"])
    S["Claus Tail Gas"] = cond
    S["Incinerator Air"] = P.stream("Incinerator Air Feed", 25, 2.0, 100.0, {"Oxygen": 0.233, "Nitrogen": 0.767})
    S["Incin Feed"] = P.stream("Incin Feed")
    U["M-INC"] = mixer(P, "M-INC", [S["Claus Tail Gas"], S["Incinerator Air"]], S["Incin Feed"])
    S["Stack Gas"] = P.stream("Stack Gas")
    S["Stack Liq"] = P.stream("Stack Liq")
    U["INC"], E["INC"] = reactor(P, "INC-1 Tail Gas Incinerator", S["Incin Feed"], S["Stack Gas"], S["Stack Liq"], P.rset_incin, 650.0)


def tune_G(P):
    ph = S["Claus Feed"].Phases[0]
    n = ph.Properties.molarflow or 0.0
    h2s = n * ph.Compounds["Hydrogen sulfide"].MoleFraction
    hc_o2 = n * (2 * ph.Compounds["Methane"].MoleFraction + 3.5 * ph.Compounds["Ethane"].MoleFraction + 5 * ph.Compounds["Propane"].MoleFraction)
    mesh_o2 = 3.0 * n * ph.Compounds["Methyl mercaptan"].MoleFraction
    o2 = 0.5 * h2s + hc_o2 + mesh_o2   # 1/3 of the H2S to SO2 (1.5 mol O2 per mol) + hydrocarbon/mercaptan burn
    S["Claus Air"].Phases[0].Properties.massflow = max(o2 / 0.21 * 28.96e-3, 1e-6)
    tg = S["Claus Tail Gas"].Phases[0]
    ntg = tg.Properties.molarflow or 0.0
    o2i = ntg * (1.5 * tg.Compounds["Hydrogen sulfide"].MoleFraction + 3.0 * tg.Compounds["Methyl mercaptan"].MoleFraction) * 1.3 + 1e-6
    S["Incinerator Air"].Phases[0].Properties.massflow = max(o2i / 0.21 * 28.96e-3, 1e-6)
    print(f"  Claus: H2S feed {h2s:.4f} mol/s -> air {o2/0.21*28.96e-3*3600:.1f} kg/h")


def report(P, tags):
    print("%-26s %8s %8s %11s" % ("stream", "T C", "P bar", "kg/h"))
    for t in tags:
        s = S[t]
        p = s.Phases[0].Properties
        try:
            print("%-26s %8.1f %8.2f %11.0f" % (t, p.temperature - 273.15, p.pressure / 1e5, p.massflow * 3600))
        except Exception:
            print(t, "n/a")


def sulfur_wt(P, s):
    ph = s.Phases[0]
    tot = ph.Properties.massflow
    if not tot:
        return 0.0
    return 100 * sum(ph.Compounds[c].MassFraction * assay.sulfur_fraction(c) for c in P.comps)


S_EXTRA = {"Sulfur": 1.0, "Sulfur dioxide": 32.06 / 64.06, "Dimethyl disulfide": 64.12 / 94.2,
           "Diethyl disulfide": 64.12 / 122.25, "1-(Butyldisulfanyl)Butane": 64.12 / 178.36}


def s_frac(c):
    return S_EXTRA.get(c, assay.sulfur_fraction(c))


def stream_sulfur_kgh(P, s):
    ph = s.Phases[0]
    m = (ph.Properties.massflow or 0.0) * 3600
    return m * sum((ph.Compounds[c].MassFraction or 0.0) * s_frac(c) for c in P.comps)


def stream_mass(s):
    return (s.Phases[0].Properties.massflow or 0.0) * 3600


FEEDS = ["Crude Oil", "Wash Water", "Fuel Gas", "Air", "MEROX Air N", "MEROX Air L", "Claus Air", "Incinerator Air"]
PRODUCTS = ["Brine", "PF Sour Water", "PF Vapor Sour Water", "PF Liquid Sour Water", "Atm Sour Water", "VDU Sour Water", "LPG Water",
            "Naphtha Water", "Regen Water", "LPG Dryer Water", "Flue Gas", "Flue Gas liq", "Sweet Naphtha", "MEROX Vent N", "Kerosene", "Diesel", "AGO",
            "Slop Distillate", "Vacuum Gas", "LVGO", "HVGO", "Vacuum Residue", "Sweet Propane", "Sweet C4 LPG", "Sweet Fuel Gas",
            "Disulfide Oil", "Regen Vent", "Liquid Sulfur", "Stack Gas", "Stack Liq", "Claus Th Liq"]


def results(P):
    print("\n=== KEY STREAMS ===")
    rows = ["Crude Oil", "Desalted Crude", "Brine", "Crude Oil H1", "Crude Oil H2", "PF OVHD Vapor", "PF OVHD Liquid", "Stabilised Crude H1",
            "OVHD Liquid", "Kerosene", "Diesel", "AGO", "Atm Residue", "Naphtha", "Naphtha 2", "LPG + FG", "LPG + FG S2", "Sweet Naphtha",
            "Slop Distillate", "LVGO", "HVGO", "Vacuum Residue", "Raw LPG + FG", "Acid Gas 1", "Acid Gas 2", "DE Off Gas", "Sweet Fuel Gas",
            "C3+ Bottoms", "Sweet Propane", "Sweet C4 LPG", "Disulfide Oil", "Claus Feed", "Liquid Sulfur", "Stack Gas"]
    print("%-22s %7s %7s %10s %7s" % ("stream", "T C", "P bar", "kg/h", "S wt%"))
    for t in rows:
        s = S[t]
        p = s.Phases[0].Properties
        m = stream_mass(s)
        sw = 100 * stream_sulfur_kgh(P, s) / m if m > 1e-9 else 0.0
        print("%-22s %7.1f %7.2f %10.1f %7.3f" % (t, p.temperature - 273.15, p.pressure / 1e5, m, sw))
    linked = "SP-FG" in U
    FEEDS_ = [t for t in FEEDS if not (linked and t == "Fuel Gas")]
    PRODUCTS_ = ["Sweet FG Export" if (linked and t == "Sweet Fuel Gas") else t for t in PRODUCTS]
    fin = sum(stream_mass(S[t]) for t in FEEDS_)
    fout = sum(stream_mass(S[t]) for t in PRODUCTS_)
    print("\nOverall mass balance: in %.0f kg/h, out %.0f kg/h, closure %.3f %%" % (fin, fout, 100 * fout / fin))
    s_in = sum(stream_sulfur_kgh(P, S[t]) for t in FEEDS_)
    s_out = {t: stream_sulfur_kgh(P, S[t]) for t in PRODUCTS_}
    print("Sulphur balance: in %.0f kg/h (crude %.0f, fuel gas %.1f); out %.0f kg/h" % (s_in, stream_sulfur_kgh(P, S["Crude Oil"]), stream_sulfur_kgh(P, S["Fuel Gas"]), sum(s_out.values())))
    for t, v in sorted(s_out.items(), key=lambda kv: -kv[1])[:10]:
        print("   %-18s %9.1f kg/h S (%.1f %%)" % (t, v, 100 * v / s_in))
    print("\nVolumetric yields on crude (wt%% of crude): ")
    for t in ("Sweet Naphtha", "Kerosene", "Diesel", "AGO", "LVGO", "HVGO", "Vacuum Residue", "Sweet Propane", "Sweet C4 LPG", "Sweet Fuel Gas"):
        print("   %-16s %6.2f wt%%" % (t, 100 * stream_mass(S[t]) / stream_mass(S["Crude Oil"])))
    if linked:
        print("   %-16s %6.2f wt%%  (fuel gas to FH-100 firebox: %.0f kg/h)" % ("FG export", 100 * stream_mass(S["Sweet FG Export"]) / stream_mass(S["Crude Oil"]), stream_mass(S["Fuel Gas"])))
    print("\nDuties (kW):")
    for k in ("E0", "E1", "FH-100 duty", "E2", "E3", "E10", "E11", "E-VF", "E-DEc", "E-DEr", "E-DPc", "E-DPr", "E5", "E6", "E7", "E8"):
        try:
            print("   %-12s %10.0f" % (k, E[k].EnergyFlow))
        except Exception:
            pass


def check(P, label):
    for k, u in U.items():
        try:
            if not u.Calculated:
                print("  [%s] first uncalculated unit: %s  %s" % (label, k, (u.ErrorMessage or "")[:150]))
                try:
                    errs = P.interf.CalculateFlowsheet4(P.fs)
                    for x in list(errs)[:3]:
                        print("      solver error:", str(x)[:300].replace("\r", " ").replace("\n", " "))
                except Exception as ex:
                    print("      (could not fetch solver errors)", str(ex)[:100])
                return False
        except Exception:
            pass
    print("  [%s] all units calculated" % label)
    return True


if __name__ == "__main__":
    t0 = time.time()
    P = make_plant()
    add_combustion(P)
    add_reactions(P)
    P.section = LANES[0]
    section_A(P)
    P.section = LANES[2]
    section_B(P)
    if HI:
        P.section = LANES[3]
        section_D(P)
        P.section = LANES[1]
        from heat_integration import integrate, update_tears
        HX, TEARS = integrate(P, S, U, E)
    P.section = LANES[4]
    section_C_feeds(P)
    P.solve("A-B + C feeds")
    check(P, "crude train + heat recovery")
    if HI:
        for it in range(6):
            dT = update_tears(TEARS)
            print("  heat-recovery tear pass %d: max hot-side change %.2f K" % (it + 1, dT))
            if dT < 0.5:
                break
            P.solve("heat recovery pass %d" % (it + 1))
        check(P, "heat recovery converged")
    tune_fuel(P)
    section_C_cols(P)
    tune_C(P)
    P.solve("debutanisers")
    if not HI:
        P.section = LANES[3]
        section_D(P)
    P.section = LANES[5]
    section_E_feeds(P)
    P.solve("VDU + LPG feed")
    section_E_de(P)
    tune_DE(P)
    P.solve("deethaniser")
    section_E_dp(P)
    tune_DP(P)
    P.solve("depropaniser")
    tune_DE(P)
    tune_DP(P)
    P.solve("gas plant")
    P.section = LANES[6]
    section_F_naphtha(P)
    P.solve("naphtha MEROX")
    check(P, "naphtha MEROX")
    tune_F_naphtha(P)
    P.solve("naphtha MEROX tuned")
    section_F_lpg(P)
    P.solve("LPG MEROX")
    check(P, "LPG MEROX")
    tune_F_lpg(P)
    P.solve("LPG MEROX tuned")
    P.section = LANES[7]
    section_G1(P)
    P.solve("fuel gas amine")
    check(P, "fuel gas amine")
    P.section = LANES[8]
    section_G2(P)
    P.solve("Claus")
    check(P, "Claus")
    for it in range(2):
        tune_G(P)
        P.solve("Claus tune %d" % it)
    tune_fuel(P)
    P.solve("final")
    print("\nColumn status:")
    for nm in ("DEBUTANISER", "DEBUTANISER 2", "DEETHANISER", "DEPROPANISER"):
        print("  ", nm, "calc=", U[nm].Calculated, (U[nm].ErrorMessage or "")[:100])
    bad = [k for k, u in U.items() if hasattr(u, "Calculated") and not u.Calculated]
    print("Unit ops not calculated:", bad if bad else "none")
    results(P)
    if HI:
        print("\n=== HEAT-RECOVERY NETWORK ===")
        tot = 0.0
        for n in range(1, 14):
            hx = U["HX%d" % n]
            q = hx.HeatDuty or 0.0
            tot += q
            print("%-44s Q=%8.0f kW  hot out %6.1f C  cold out %6.1f C  MITA %5.1f K  calc=%s" % (
                hx.GraphicObject.Tag, q, hx.HotSideOutletTemperature - 273.15, hx.ColdSideOutletTemperature - 273.15,
                hx.CalculatedMITA, hx.Calculated))
        print("Total recovered: %.0f kW" % tot)
        trim = {k: abs(E[k].EnergyFlow or 0.0) for k in ("E0", "E1", "FH-100 duty", "E10")}
        print("Remaining heater duties (kW):", {k: round(v) for k, v in trim.items()}, "| baseline: E0 36938, E1 31642, FH-100 22841, E10 47104")
        print("Heater duty now %.0f kW vs 138525 kW baseline (-%.0f%%)" % (sum(trim.values()), 100 * (1 - sum(trim.values()) / 138525.0)))
    path = os.path.join(OUT_DIR, "Crude_Vacuum_Refinery_HeatIntegrated.dwxmz" if HI else "Crude_Vacuum_Refinery.dwxmz")
    import json as _json
    by, centre = apply_pfd(P)
    print('layout: %d objects, x up to %d, y %d..%d' % (len(by), max(g.X for g in by.values()), min(g.Y for g in by.values()), max(g.Y for g in by.values())))
    P.save(path)
    print("saved", path, "| total build+solve %.0fs" % (time.time() - t0))
















