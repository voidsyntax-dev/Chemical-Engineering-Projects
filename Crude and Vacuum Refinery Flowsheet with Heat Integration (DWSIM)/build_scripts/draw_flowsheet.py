import os
"""Approximate block-flow diagram of the DWSIM refinery model built so far -> flowsheet_final.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(24, 15))
ax.set_xlim(0, 48)
ax.set_ylim(0, 30)
ax.axis("off")

COL = {"unit": "#dbe9f6", "col": "#ffe9b3", "heat": "#f9d6d5", "plan": "#eeeeee", "prod": "#d9f2d9"}
boxes = {}


def box(key, x, y, w, h, text, kind="unit", dashed=False, fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=COL[kind], ec="#444",
                       lw=1.4, ls="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    boxes[key] = (x, y, w, h)


def anchor(key, side):
    x, y, w, h = boxes[key]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def arrow(a, sa, b, sb, label=None, dashed=False, off=(0, 0.25), rad=0.0):
    p0, p1 = anchor(a, sa), anchor(b, sb)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#1a3a6e",
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], label, fontsize=7.5, ha="center", color="#1a3a6e")


# ---- row 1: feed / desalting / furnace (y ~ 24)
box("crude", 0.3, 24, 3, 2, "Crude Oil\n100 kbpd\n27.6 API, 2.85 wt% S", "prod")
box("p100", 4.2, 24, 2.2, 2, "P-100\nPump\n16 bar")
box("e100", 7.3, 24, 2.6, 2, "Heat recovery C1\nH7-H11 (5 HX)\n+ E-100 trim 145 C", "heat")
box("mix", 10.8, 24, 2.6, 2, "M-100 / V-100\nWash water mix\n+ mixing valve")
box("desal", 14.3, 24, 3.2, 2, "D-100 DESALTER\n(salt/BS&W split)", "col")
box("h101", 18.4, 24, 2.2, 2, "Heat recovery C2\nH3-H6 + H101 trim\n250 C", "heat")
box("fh", 21.5, 24, 3.6, 2, "FH-100 Fired heater\n255 C\n(fuel gas + air firebox)", "heat")
box("ww", 10.8, 27.5, 2.6, 1.4, "Wash water\n4.5 wt%", "prod")
box("brine", 14.3, 27.5, 3.2, 1.4, "Brine\n(~27 t/h)", "prod")
box("fuel", 21.5, 27.5, 1.6, 1.4, "Fuel gas", "prod")
box("air", 23.5, 27.5, 1.6, 1.4, "Air", "prod")
box("flue", 26.0, 27.5, 2.0, 1.4, "Flue gas\n320 C", "prod")
for a, sa, b, sb in [("crude", "r", "p100", "l"), ("p100", "r", "e100", "l"), ("e100", "r", "mix", "l"), ("mix", "r", "desal", "l"),
                     ("desal", "r", "h101", "l"), ("h101", "r", "fh", "l")]:
    arrow(a, sa, b, sb)
arrow("ww", "b", "mix", "t")
arrow("desal", "t", "brine", "b")
arrow("fuel", "b", "fh", "t")
arrow("air", "b", "fh", "t")
arrow("fh", "t", "flue", "l", rad=-0.2)

# ---- row 2: PF column + branches (y ~ 16)
box("pf", 27.5, 22.2, 4.2, 3.8, "PRE-FRACTIONATING\nCOLUMN\n(cut-point, 150 C)\ncondenser E2\nreboiler E3", "col", fs=9)
arrow("fh", "r", "pf", "l", "Crude Oil H2\n+ V-101 letdown", off=(0, 0.35))
box("db1", 33.5, 26.5, 4.2, 2.6, "DEBUTANISER\n(rigorous, 20 bar)\nE-100 + K-100 + WKO-1", "col")
box("db2", 33.5, 22.0, 4.2, 2.6, "DEBUTANISER 2\n(rigorous, 14 bar)\nP-103 + E-101 + WKO-2", "col")
arrow("pf", "t", "db1", "l", "PF OVHD Vapor", off=(0, 0.35), rad=-0.25)
arrow("pf", "r", "db2", "l", "PF OVHD Liquid", off=(0, 0.35))
box("lpg1", 39.3, 27.3, 3.2, 1.4, "LPG + FG", "prod")
box("nap1", 43.0, 27.3, 3.0, 1.4, "Naphtha", "prod")
box("lpg2", 39.3, 22.8, 3.2, 1.4, "LPG + FG S2", "prod")
box("nap2", 43.0, 22.8, 3.0, 1.4, "Naphtha 2", "prod")
arrow("db1", "r", "lpg1", "l")
arrow("db1", "t", "nap1", "t", rad=-0.35)
arrow("db2", "r", "lpg2", "l")
arrow("db2", "b", "nap2", "b", rad=0.35)

# ---- row 3: atmospheric unit (y ~ 12)
box("e102", 27.5, 17.2, 4.2, 2.2, "P-102 + HX H1,H2\n(HVGO, VR)\n+ E-102 trim 355 C", "heat")
arrow("pf", "b", "e102", "t", "Stabilised Crude", off=(1.6, 0))
box("atm", 33.5, 12.0, 4.6, 6.8, "ATMOSPHERIC UNIT\n(cut-point)\nCut 1: <165 C\nCut 2: Kero 165-245 C\nCut 3: Diesel 245-340 C\nCut 4: AGO 340-385 C\nResidue 385 C+\ncondenser E11", "col")
arrow("e102", "r", "atm", "l", "Stab. Crude H1\n(V-102 flash)", off=(0, 0.45))
box("ovhd", 39.3, 17.8, 3.2, 1.2, "OVHD Liquid", "prod")
box("kero", 39.3, 16.1, 3.2, 1.2, "Kerosene", "prod")
box("dsl", 39.3, 14.4, 3.2, 1.2, "Diesel", "prod")
box("ago", 39.3, 12.7, 3.2, 1.2, "AGO", "prod")
for k, y in (("ovhd", 18.4), ("kero", 16.7), ("dsl", 15.0), ("ago", 13.3)):
    ax.add_patch(FancyArrowPatch((37.9, y), (39.2, y), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#1a3a6e"))

# ---- row 4: vacuum unit (y ~ 5)
box("vf", 33.5, 7.3, 4.6, 2.4, "P-104 + VF-100\nVacuum furnace 405 C", "heat")
arrow("atm", "b", "vf", "t", "Atm Residue", off=(1.1, 0))
box("vdu", 39.0, 3.2, 5.0, 6.8, "VACUUM UNIT\n(cut-point, 0.08 bar)\nOVHD <330 C\nLVGO 330-440 C\nHVGO 440-555 C\nVacuum residue 555 C+\ncondenser + drum", "col")
arrow("vf", "r", "vdu", "l", "V-103 flash", off=(0, 0.3))
box("lvgo", 44.8, 7.6, 2.8, 1.0, "LVGO", "prod")
box("hvgo", 44.8, 6.0, 2.8, 1.0, "HVGO", "prod")
box("vr", 44.8, 4.4, 2.8, 1.0, "Vacuum Residue", "prod")
box("vslop", 44.8, 2.8, 2.8, 1.0, "Slop / Vac. gas", "prod")
for y in (8.1, 6.5, 4.9, 3.3):
    ax.add_patch(FancyArrowPatch((44.0, y), (44.7, y), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#1a3a6e"))

# ---- row 5: gas plant (y ~ 12 -> left)
box("mlpg", 18.0, 14.0, 3.2, 2.2, "M-LPG\nraw LPG + FG\n+ V-104 letdown")
box("am1", 22.0, 14.0, 3.6, 2.2, "AM-1 Amine\ntreater\n(H2S removal)", "col")
box("acid", 22.0, 10.5, 3.6, 1.6, "Acid gas -> Claus", "prod")
box("e103", 18.0, 10.5, 3.2, 2.2, "E-103 cool +\nD-103 drum\n+ P-105 28 bar", "heat")
box("de", 13.0, 10.5, 4.0, 3.2, "DEETHANISER\n(rigorous, 27 bar)", "col")
box("dp", 7.8, 10.5, 4.2, 3.2, "DEPROPANISER\n(rigorous, 17 bar)", "col")
box("offgas", 13.0, 6.8, 4.0, 1.4, "Off gas (C1/C2)\n-> fuel gas", "prod")
box("c3", 7.8, 6.8, 4.2, 1.4, "Propane", "prod")
box("c4", 2.8, 6.8, 4.0, 1.4, "C4 LPG", "prod")
arrow("lpg1", "b", "mlpg", "t", "", rad=0.0)
arrow("lpg2", "b", "mlpg", "t", "", rad=0.1)
arrow("mlpg", "r", "am1", "l")
arrow("am1", "b", "acid", "t")
arrow("am1", "l", "e103", "t", rad=0.0)
arrow("e103", "l", "de", "r")
arrow("de", "l", "dp", "r", "C3+")
arrow("de", "b", "offgas", "t")
arrow("dp", "b", "c3", "t")
arrow("dp", "l", "c4", "t", rad=0.3)

# ---- planned (not yet built)
box("merox", 2.8, 14.2, 4.0, 2.2, "MEROX\n(naphtha sweetening +\nLPG mercaptan extraction)", "col")
box("claus", 2.8, 17.2, 4.0, 2.2, "CLAUS SULPHUR UNIT\n(thermal + catalytic stage,\ntail-gas incinerator)", "col")
box("fg", 8.0, 14.2, 4.0, 2.2, "AM-2 Fuel gas amine\ntreater", "col")

ax.text(0.3, 29.4, "Crude + Vacuum Refinery in DWSIM v10  -  APPROXIMATE block flow (as built, heat-integrated)", fontsize=15, weight="bold")
ax.text(0.3, 28.6, "Blue = unit ops, yellow = columns/towers, red = heaters/coolers, green = streams/products, all units shown are built and solving.",
        fontsize=9.5)
ax.text(0.3, 0.6, "Heat recovery: 11 exchangers (hot kero/diesel/AGO/LVGO/HVGO/VR vs crude), 107 MW recovered, min approach 14 K  |  heater duty 138.5 -> 31.6 MW (-77 %)", fontsize=11, weight="bold", color="#8a1c1c")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flowsheet_final.png"), dpi=110)
print("saved")


