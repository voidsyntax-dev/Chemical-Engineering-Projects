"""Crude/product heat-recovery network (11 exchangers) built into the DWSIM flowsheet.

Hot-side feeds are copies of the solved product-draw streams (hot_streams_baseline.json). That is exact here: the
cut-point tower draw states depend only on the furnace outlet temperatures, which are specified, not on the preheat.
The copies break the structural recycle (preheat -> tower -> products -> preheat) that DWSIM's solver cannot order.
Exchanger targets come from hx_network.json (greedy MITA >= 15 K design on enthalpy curves).
"""
import json, os
from flowsheet_tools import *

HERE = os.path.dirname(os.path.abspath(__file__))
HOT_KEY = {"K": "Atm Kerosene Hot", "D": "Atm Diesel Hot", "A": "Atm AGO Hot",
           "L": "VDU LVGO Hot", "HV": "VDU HVGO Hot", "V": "Vacuum Residue Hot"}
HOT_NAME = {"K": "Kerosene", "D": "Diesel", "A": "AGO", "L": "LVGO", "HV": "HVGO", "V": "Vacuum Residue"}
PRODUCT_T = {"K": 60.0, "D": 70.0, "A": 90.0, "L": 90.0, "HV": 130.0, "V": 200.0}


def exchanger(P, tag, hot_in, hot_out, cold_in, cold_out, tc_out_c, dPh_bar, dPc_bar, th_out_c=None):
    hx = P.add(OT.HeatExchanger, tag)
    P.connect(hot_in, hx, 0, 0)
    P.connect(cold_in, hx, 0, 1)
    P.connect(hx, hot_out, 0, 0)
    P.connect(hx, cold_out, 1, 0)
    if th_out_c is not None:               # hot-side outlet specified, cold-side outlet calculated
        set_enum(hx, "CalculationMode", "CalcTempColdOut")
        hx.HotSideOutletTemperature = th_out_c + 273.15
    else:
        set_enum(hx, "CalculationMode", "CalcTempHotOut")
        hx.ColdSideOutletTemperature = tc_out_c + 273.15
    hx.HotSidePressureDrop = dPh_bar * 1e5
    hx.ColdSidePressureDrop = dPc_bar * 1e5
    return hx


def integrate(P, S, U, E):
    net = json.load(open(os.path.join(HERE, "hx_network.json")))
    hot = json.load(open(os.path.join(HERE, "hot_streams_baseline.json")))
    cur = {}                                  # current hot-stream object per hot id
    for h, key in HOT_KEY.items():
        d = hot[key]
        cur[h] = P.stream("HOT %s (draw copy)" % HOT_NAME[h], d["T_C"], d["P_bar"], d["kg_h"], d["w"])
        S["HOT copy " + h] = cur[h]
        U["RC-" + h] = recycle(P, P.objs[key], cur[h], "RC %s draw" % HOT_NAME[h])    # tower draw -> exchanger hot side
    # vacuum residue is a liquid at the draw: pump it up before any exchange
    vr_p = P.stream("VR Pumped")
    U["P-VR"], E["W-P-VR"] = pump(P, "P-VR Vacuum Residue Pump", cur["V"], vr_p, 12.0)
    cur["V"] = vr_p
    # cold-side trains: (inlet stream, outlet stream) per train
    cold_in = {"C1": S["Crude Oil P"], "C2": S["Desalted Crude"], "C3": S["Stabilised Crude P"]}
    cold_end = {"C1": "Crude Oil Pre (HX)", "C2": "Crude Oil H1 (HX)", "C3": "Stabilised Crude H1 (HX)"}
    order = sorted(net, key=lambda r: (r["c"] != "C1", r["n"]))   # keep crude flow order within a train
    trains = {"C1": [], "C2": [], "C3": []}
    for r in net:
        trains[r["c"]].append(r)
    # hot-stream sequencing requires the high-temperature use first: C3, then C2, then C1 (as numbered in the design)
    hx_obj = {}
    inter = []                                # (copy stream, upstream hot outlet) pairs updated between passes
    prev_out = {}
    for r in sorted(net, key=lambda r: r["n"]):
        h, c, n = r["h"], r["c"], r["n"]
        dPh = 0.0 if (h in ("L", "HV") and c in ("C2", "C3")) else 0.2
        hot_out = P.stream("HX%d %s out" % (n, HOT_NAME[h]))
        cold_out = P.stream("HX%d %s out" % (n, c + " crude"))
        if h in prev_out:
            # a later use of an already-used hot stream sits in a different train: tear it with a feed copy
            d = hot[HOT_KEY[h]]
            p_copy = 0.08 if (h == "HV" and c == "C1") else (12.0 if h == "V" else (5.0 if h == "HV" else d["P_bar"] - 0.2))
            cp = P.stream("HX%d %s in (tear copy)" % (n, HOT_NAME[h]), r["th_in"], p_copy, d["kg_h"], d["w"])
            inter.append((cp, prev_out[h]))
            U["RCT%d" % n] = recycle(P, prev_out[h], cp, "RC HX%d %s tear" % (n, HOT_NAME[h]))
            cur[h] = cp
        if h == "HV" and c == "C1":
            hp = P.stream("HVGO Pumped")
            U["P-HVGO"], E["W-P-HVGO"] = pump(P, "P-HVGO HVGO Pump", cur["HV"], hp, 5.0)
            cur["HV"] = hp
        hx = exchanger(P, "H%d %s / %s" % (n, HOT_NAME[h], {"C1": "crude", "C2": "desalted crude", "C3": "stab. crude"}[c]),
                       cur[h], hot_out, cold_in[c], cold_out, r["tc_out"], dPh, 0.5)
        U["HX%d" % n] = hx
        hx_obj[n] = hx
        prev_out[h] = hot_out
        cur[h] = hot_out
        cold_in[c] = cold_out    # trim heaters on each cold train and hand-offs to the existing flowsheet streams
    # light-end heat recovery: debutaniser-2 naphtha bottoms, then pre-fractionator overhead vapour, heat the pre-desalter crude
    ex = json.load(open(os.path.join(HERE, "hot_streams_extra.json")))
    for n, key, hin_tag, hot_out, th_out, dPh in ((12, "Naphtha 2", "Naphtha 2 HX in", None, 145.0, 0.2),
                                                  (13, "PF Overhead", "PF Overhead HX in", S["PF Overhead HX"], 150.0, 0.2)):
        d = ex[key]
        hot_in = P.stream(hin_tag, d["T_C"], d["P_bar"], d["kg_h"], d["w"])
        if hot_out is None:
            hot_out = P.stream("Naphtha 2 HX")
            S["Naphtha 2 HX"] = hot_out
            S["Naphtha 2 HX in"] = hot_in
        else:
            U["RC-PFO"] = recycle(P, S["PF Overhead"], hot_in, "RC PF Overhead")
        cold_out = P.stream("HX%d C1 crude out" % n)
        nm = "H%d %s / crude" % (n, key)
        hx = exchanger(P, nm, hot_in, hot_out, cold_in["C1"], cold_out, None, dPh, 0.3, th_out_c=th_out)
        U["HX%d" % n] = hx
        hx_obj[n] = hx
        cold_in["C1"] = cold_out
    U["E-100"], E["E0"] = heater(P, "E-100 Preheat Trim", cold_in["C1"], S["Crude Oil Pre"], T=145.0, dP=0.5)
    U["H101"], E["E1"] = heater(P, "H101 Trim Heater", cold_in["C2"], S["Crude Oil H1"], T=250.0, dP=0.5)
    U["E-102"], E["E10"] = heater(P, "E-102 Atm Furnace Trim", cold_in["C3"], S["Stabilised Crude H1"], T=355.0, dP=4.0)
    # LVGO also needs a pump out of vacuum before its trim cooler
    lp = P.stream("LVGO Pumped")
    U["P-LVGO"], E["W-P-LVGO"] = pump(P, "P-LVGO LVGO Pump", cur["L"], lp, 5.0)
    cur["L"] = lp
    # product trim coolers (same names/product streams as the utility-cooled case)
    spec = {"K": ("Kerosene Cooler", "Ck", "Ek"), "D": ("Diesel Cooler", "Cd", "Ed"), "A": ("AGO Cooler", "Ca", "Ea"),
            "L": ("LVGO Cooler", "C-LVGO", "E-LVGO"), "HV": ("HVGO Cooler", "C-HVGO", "E-HVGO"), "V": ("VR Cooler", "C-VR", "E-VR")}
    for h, (tag, uk, ek) in spec.items():
        U[uk], E[ek] = heater(P, tag, cur[h], S[HOT_NAME[h]], T=PRODUCT_T[h], dP=0.2, cooler=True)
    return hx_obj, inter


def update_tears(inter):
    '''Copy upstream hot-outlet temperatures onto the tear feeds; return the largest change (K).'''
    worst = 0.0
    for cp, src in inter:
        t_new = src.Phases[0].Properties.temperature
        t_old = cp.Phases[0].Properties.temperature
        if src.Calculated and t_new is not None and t_new > 0:
            worst = max(worst, abs(t_new - t_old))
            cp.Phases[0].Properties.temperature = t_new
    return worst



