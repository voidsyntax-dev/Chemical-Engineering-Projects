"""Export the flowsheet canvas (positions saved in the .dwxmz) as vector SVG + PDF.  usage: export_flowsheet.py"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dwlib import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fs = Automation3().LoadFlowsheet2(os.path.join(ROOT, "Crude_Vacuum_Refinery_HeatIntegrated.dwxmz"))
kinds = {o.GraphicObject.Tag: o.GetAsObject().GetType().Name for o in fs.SimulationObjects.Values}
G = {g.Tag: g for g in fs.GraphicObjects.Values}
col = {"MaterialStream": "#cfe3f7", "EnergyStream": "#fff3b0", "DistillationColumn": "#ffe3a3", "HeatExchanger": "#f6c9c6",
       "Heater": "#f6c9c6", "Cooler": "#bfe0f5", "Pump": "#dddddd", "Valve": "#dddddd", "Mixer": "#dcd0f0", "Vessel": "#d6ecd6",
       "ComponentSeparator": "#ffe3a3", "Compressor": "#dddddd", "Recycle": "#c9f0c9", "Splitter": "#dcd0f0"}
xs = [g.X for g in G.values()]; ys = [g.Y for g in G.values()]
W = (max(xs) - min(xs) + 200) / 100.0; H = (max(ys) - min(ys) + 200) / 100.0
fig, ax = plt.subplots(figsize=(W, H))
for g in G.values():
    k = kinds.get(g.Tag, "?"); s = k in ("MaterialStream", "EnergyStream")
    ax.add_patch(plt.Rectangle((g.X, -g.Y - g.Height), g.Width, g.Height, fc=col.get(k, "#ffe3a3" if "Reactor" in k else "#eee"),
                               ec="#555", lw=0.5, zorder=2))
    ax.text(g.X + g.Width / 2, -g.Y - g.Height - 3, g.Tag[:34], fontsize=3.2 if s else 4.5, ha="center", va="top", zorder=3,
            weight=None if s else "bold")
for g in G.values():
    for oc in g.OutputConnectors:
        if oc.IsAttached and oc.AttachedConnector is not None:
            h = oc.AttachedConnector.AttachedTo
            ax.annotate("", xy=(h.X, -h.Y - h.Height / 2), xytext=(g.X + g.Width, -g.Y - g.Height / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.4, color="#1a3a6e", mutation_scale=5), zorder=1)
ax.set_aspect("equal"); ax.autoscale(); ax.axis("off"); plt.tight_layout(pad=0.2)
for ext in ("svg", "pdf"):
    p = os.path.join(ROOT, "Flowsheet." + ext); plt.savefig(p); print("wrote", p)
