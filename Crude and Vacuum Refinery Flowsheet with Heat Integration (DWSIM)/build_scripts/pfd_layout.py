"""Designed PFD layout: every unit operation has a hand-set grid position; streams are drawn between their units.

Main process line (row 0, left to right): crude -> preheat train -> desalter -> hot preheat -> fired heater -> pre-fractionator
-> atmospheric furnace train -> atmospheric unit -> vacuum furnace -> vacuum unit -> sulphur recovery.
Above it: pre-fractionator overheads, debutanisers, LPG recovery, MEROX, fuel-gas amine.  Below it: hot product coolers, firebox.
"""
from collections import defaultdict

PX, PY = 170, 135            # grid pitch (px per column / row)
X0, Y0 = 200, 1300
LANES = ["1 Feed, desalting, fired heater", "2 Heat-recovery network", "3 Pre-fractionator + atmospheric unit",
         "4 Vacuum unit", "5 Debutanisers", "6 LPG recovery: amine, deethaniser, depropaniser",
         "7 MEROX sweetening", "8 Fuel-gas amine treating", "9 Sulphur recovery (Claus, incinerator)"]
LANES_ORDER = []             # kept for compatibility with relayout previews

# (column, row) of each unit operation; row 0 is the main line, negative = above, positive = below
POS = {
    # --- row 0: crude train ------------------------------------------------------------------------------------------
    "P-100": (0, 0),
    "H7 AGO / crude": (1.2, 0), "H8 Kerosene / crude": (2.4, 0), "H9 Diesel / crude": (3.6, 0),
    "H10 HVGO / crude": (4.8, 0), "H11 Vacuum Residue / crude": (6.0, 0), "E-100 Preheat Trim": (7.2, 0),
    "M-100 Wash Water Mixer": (8.4, 0), "V-100 Mixing Valve": (9.4, 0), "D-100 Desalter": (10.6, 0),
    "H3 LVGO / desalted crude": (11.8, 0), "H4 Kerosene / desalted crude": (13.0, 0), "H5 Diesel / desalted crude": (14.2, 0),
    "H6 Vacuum Residue / desalted crude": (15.4, 0), "H101 Trim Heater": (16.6, 0), "FH-100": (17.8, 0),
    "V-101 PF Letdown": (19.0, 0), "PRE-FRACTIONATING COLUMN": (20.2, 0), "PF Reboiler": (21.4, 0), "P-102": (22.4, 0),
    "H1 HVGO / stab. crude": (23.6, 0), "H2 Vacuum Residue / stab. crude": (24.8, 0),
    "E-102 Atm Furnace Trim": (26.0, 0), "V-102 Atm Flash": (27.0, 0),
    "Atm Cut 1 (OVHD)": (28.2, 0), "Atm Cut 2 (Kerosene)": (29.2, 0), "Atm Cut 3 (Diesel)": (30.2, 0), "Atm Cut 4 (AGO)": (31.2, 0),
    "P-104": (32.4, 0), "VF-100 Vacuum Furnace": (33.4, 0), "V-103 Vacuum Flash": (34.4, 0),
    "VDU Cut 1 (OVHD)": (35.6, 0), "VDU Cut 2 (LVGO)": (36.6, 0), "VDU Cut 3 (HVGO)": (37.6, 0),
    # --- below: hot product coolers / pumps, firebox -----------------------------------------------------------------
    "AGO Cooler": (1.2, 2.2), "Kerosene Cooler": (2.4, 2.2), "Diesel Cooler": (3.6, 2.2),
    "P-HVGO HVGO Pump": (4.0, 3.3), "HVGO Cooler": (5.0, 2.2), "VR Cooler": (6.2, 2.2),
    "P-LVGO LVGO Pump": (11.4, 2.2), "LVGO Cooler": (12.4, 2.2),
    "P-VR Vacuum Residue Pump": (23.8, 2.2),
    "M-FH Burner": (16.6, 2.2), "FH-100 Firebox": (17.8, 2.2),
    # --- above: pre-fractionator overheads ---------------------------------------------------------------------------
    "PF Condenser": (21.0, -1.2), "PF Reflux Drum": (22.0, -1.2),
    "Atm Condenser": (28.6, -1.2), "Atm Reflux Drum": (29.6, -1.2),
    "VDU Condenser": (35.8, -1.2), "VDU Overhead Drum": (36.8, -1.2),
    # --- debutanisers ------------------------------------------------------------------------------------------------
    "K-100 PF Vapor Compressor": (23.4, -2.4), "E-100": (24.4, -2.4), "WKO-1 Water Decanter": (25.4, -2.4),
    "DEBUTANISER": (26.8, -2.4), "M-DB1 LPG+FG": (28.2, -2.4),
    "P-103": (23.4, -4.0), "E-101": (24.4, -4.0), "WKO-2 Water Decanter": (25.4, -4.0),
    "DEBUTANISER 2": (26.8, -4.0), "M-DB2 LPG+FG S2": (28.2, -4.0),
    # --- LPG recovery -------------------------------------------------------------------------------------------------
    "V-104 DB1 LPG Letdown": (29.4, -2.4), "M-LPG": (30.6, -3.2), "AM-1 LPG Amine Treater": (31.8, -3.2),
    "E-103": (33.0, -3.2), "D-103 LPG Feed Drum": (34.0, -3.2), "WKO-4 LPG Dryer": (35.2, -3.2), "P-105": (36.2, -3.2),
    "DEETHANISER": (37.5, -3.2), "V-105 DE Bottoms Letdown": (38.7, -3.2), "DEPROPANISER": (40.0, -3.2),
    # --- LPG MEROX ---------------------------------------------------------------------------------------------------
    "EX-C3 LPG Mercaptan Extractor": (41.4, -2.5), "EX-C4 LPG Mercaptan Extractor": (41.4, -4.0),
    "M-CAUS": (42.6, -3.2), "M-REG": (43.6, -3.2), "R-MX2 Caustic Regenerator Oxidiser": (44.8, -3.2), "DS-1 Disulfide Separator": (46.0, -3.2),
    # --- naphtha MEROX -----------------------------------------------------------------------------------------------
    "EN-Naphtha Cooler": (29.4, -6.4), "VN-Naphtha Letdown": (30.4, -6.4),
    "EN-Naphtha 2 Cooler": (29.4, -5.4), "VN-Naphtha 2 Letdown": (30.4, -5.4),
    "PN-OVHD Liquid Pump": (30.4, -7.4),
    "M-NAP Naphtha Header": (31.8, -6.4), "E-NAP Naphtha Trim Cooler": (33.0, -6.4), "M-NAP2": (34.0, -6.4),
    "R-MX1 Naphtha Sweetening Reactor": (35.2, -6.4), "WKO-3 Naphtha Water Wash/Decant": (36.6, -6.4),
    # --- fuel-gas amine ----------------------------------------------------------------------------------------------
    "V-AG1 Acid Gas Letdown": (32.8, -1.6), "V-107 Flash Gas Letdown": (34.4, -1.6), "V-106 DE Distillate Letdown": (38.0, -1.6),
    "E-OG Off Gas Warmer": (39.0, -1.6), "M-FG Off Gas Header": (40.2, -1.6), "AM-2 Fuel Gas Amine Treater": (41.4, -1.6),
    "V-AG2 Acid Gas Letdown": (42.6, -1.6), "M-CL Acid Gas Header": (43.8, -1.6),
    # --- sulphur recovery (fed from the amine acid gas via a copy stream) -------------------------------------------
    "M-CL2 Burner Mixer": (40.0, 0.0), "R-CL1 Thermal Reactor": (41.0, 0.0), "CD-0 Sulfur Condenser": (42.0, 0.0),
    "SS-0 Sulfur Separator": (43.0, 0.0), "R-CL2 Catalytic Reactor 1 (incl. reheat)": (44.2, 0.0),
    "CD-3 Final Sulfur Condenser": (45.4, 0.0), "SS-3 Sulfur Separator": (46.4, 0.0), "M-INC": (47.4, 0.0),
    "INC-1 Tail Gas Incinerator": (48.4, 0.0), "M-S Sulfur Header": (45.0, 1.7),
}

# two extra light-end exchangers on the pre-desalter crude train: everything from col 7 on moves right to make room
POS = {k: ((c + 2.4 if c >= 7.0 else c), r) for k, (c, r) in POS.items()}
POS["H12 Naphtha 2 / crude"] = (7.2, 0)
POS["H13 PF Overhead / crude"] = (8.4, 0)
POS["SP-FG Fuel Gas Splitter"] = (44.4, -0.8)

# streams that should sit on the hot-side (below) of an exchanger
HX_HOT_DY = 78


def apply_pfd(P):
    G = list(P.fs.GraphicObjects.Values)
    by = {g.Tag: g for g in G}
    kind = {t: (P.objs[t].GetType().Name if t in P.objs else "") for t in by}
    is_stream = {t: kind[t] in ("MaterialStream", "EnergyStream") for t in by}
    ins, outs = {}, {}
    for g in G:
        ins[g.Tag] = [ic.AttachedConnector.AttachedFrom.Tag if (ic.IsAttached and ic.AttachedConnector is not None) else None
                      for ic in g.InputConnectors]
        outs[g.Tag] = [oc.AttachedConnector.AttachedTo.Tag if (oc.IsAttached and oc.AttachedConnector is not None) else None
                       for oc in g.OutputConnectors]
    # sizes
    for t, g in by.items():
        if kind[t] == "DistillationColumn":
            g.Width, g.Height = 56, 130
        elif is_stream[t]:
            g.Width, g.Height = 20, 20
        elif kind[t]:
            g.Width, g.Height = 40, 40
    centre = {}
    unplaced = []
    for t in by:
        if is_stream[t]:
            continue
        if t in POS:
            c, r = POS[t]
            centre[t] = (X0 + c * PX, Y0 + r * PY)
        else:
            unplaced.append(t)
    def prod_unit(s):        # unit that makes stream s
        for t, o in outs.items():
            if not is_stream[t] and s in o:
                return t
        return None

    def cons_unit(s):
        for t, i in ins.items():
            if not is_stream[t] and s in i:
                return t
        return None

    # recycle (tear) blocks: sit just below-left of the unit their outlet stream feeds
    for t in list(unplaced):
        if kind[t] == "Recycle":
            o = outs[t][0] if outs[t] else None
            v = cons_unit(o) if o else None
            if v in centre:
                centre[t] = (centre[v][0] - 120, centre[v][1] + 150)
                unplaced.remove(t)
    # unit ops without a hand position: park below the lane (should be none)
    for i, t in enumerate(unplaced):
        centre[t] = (X0 + i * 80, Y0 + 9 * PY)

    used = []
    def free(x, y):
        for (ux, uy) in used:
            if abs(ux - x) < 70 and abs(uy - y) < 26:
                return False
        return True

    def place(s, x, y):
        step = 0
        while not free(x, y + step) and step < 400:
            step = (-step + 30) if step <= 0 else -step
        centre[s] = (x, y + step)
        used.append(centre[s])

    streams = [t for t in by if is_stream[t]]
    # streams between two placed units first (mid-point), then feeds, then products
    for s in streams:
        u, v = prod_unit(s), cons_unit(s)
        if u in centre and v in centre:
            (ux, uy), (vx, vy) = centre[u], centre[v]
            place(s, (ux + vx) / 2.0, (uy + vy) / 2.0)
    for s in streams:
        if s in centre:
            continue
        u, v = prod_unit(s), cons_unit(s)
        if v in centre and u is None:                               # feed: left of its consumer
            vx, vy = centre[v]
            idx = ins[v].index(s)
            dy = 0.0
            if kind[v] == "HeatExchanger":
                dy = HX_HOT_DY if idx == 0 else 0.0
            elif kind[v] == "EnergyStream" or "Qr_" in s:
                dy = 90.0
            elif idx > 0:
                dy = (-1 if idx % 2 else 1) * 62 * ((idx + 1) // 2)
            place(s, vx - 105, vy + dy)
        elif u in centre and v is None:                              # product: right of its producer
            ux, uy = centre[u]
            idx = outs[u].index(s)
            n = len([o for o in outs[u] if o is not None])
            if kind[u] == "HeatExchanger":
                dy = HX_HOT_DY if idx == 0 else 0.0
                place(s, ux + 100, uy + dy)
            elif "Qc_" in s:
                place(s, ux + 100, uy - 80)
            else:
                dy = (idx - (n - 1) / 2.0) * 62
                place(s, ux + 105, uy + dy)
    iso = {"OVHD Liquid Cooled 45C": (29.6, -7.4), "Spent Air / Off-gas": (45.9, -2.1)}
    for s in streams:                                                # isolated / leftovers
        if s not in centre:
            c, r = iso.get(s, (0.0, 11.0))
            place(s, X0 + c * PX, Y0 + r * PY)
    for t, g in by.items():
        cx, cy = centre[t]
        g.X = int(cx - g.Width / 2)
        g.Y = int(cy - g.Height / 2)
    return by, centre


