"""Apply the designed PFD layout to a saved model and render a preview PNG.  usage: relayout.py [save]"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dwlib import *
import pfd_layout
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILE = os.path.join(ROOT, "Crude_Vacuum_Refinery_HeatIntegrated.dwxmz")
PREV = os.path.join(ROOT, "layout_preview"); os.makedirs(PREV, exist_ok=True)
interf = Automation3()
fs = interf.LoadFlowsheet2(FILE)


class Stub:
    pass


P = Stub()
P.fs = fs
P.objs = {o.GraphicObject.Tag: o.GetAsObject() for o in fs.SimulationObjects.Values}
by, centre = pfd_layout.apply_pfd(P)
missing = [t for t in pfd_layout.POS if t not in by]
unplaced = [t for t, o in P.objs.items() if o.GetType().Name not in ("MaterialStream", "EnergyStream") and t not in pfd_layout.POS]
print("extent x %d..%d  y %d..%d" % (min(g.X for g in by.values()), max(g.X for g in by.values()),
                                      min(g.Y for g in by.values()), max(g.Y for g in by.values())))
print("POS names not in model:", missing)
print("units without a position:", unplaced)
iso=[t for t,g in by.items() if not any(ic.IsAttached for ic in g.InputConnectors) and not any(oc.IsAttached for oc in g.OutputConnectors)]
print("isolated objects:", iso)
kinds = {t: o.GetType().Name for t, o in P.objs.items()}
col = {"MaterialStream": "#cfe3f7", "EnergyStream": "#fff3b0", "DistillationColumn": "#ffe3a3", "HeatExchanger": "#f6c9c6", "Heater": "#f6c9c6",
       "Cooler": "#bfe0f5", "Pump": "#e0e0e0", "Valve": "#e0e0e0", "Mixer": "#dcd0f0", "Vessel": "#d6ecd6",
       "ComponentSeparator": "#ffe3a3", "Compressor": "#e0e0e0"}
fig, ax = plt.subplots(figsize=(80, 26))
for g in by.values():
    k = kinds.get(g.Tag, "?")
    c = col.get(k, "#ffe3a3" if "Reactor" in k else "#eeeeee")
    ax.add_patch(plt.Rectangle((g.X, -g.Y - g.Height), g.Width, g.Height, fc=c, ec="#555", lw=.7, zorder=2))
    fs_ = 6.2 if k not in ("MaterialStream", "EnergyStream") else 4.2
    ax.text(g.X + g.Width / 2, -g.Y - g.Height - 4, g.Tag[:30], fontsize=fs_, ha="center", va="top", zorder=3,
            weight="bold" if k not in ("MaterialStream", "EnergyStream") else None)
for g in by.values():
    for oc in g.OutputConnectors:
        if oc.IsAttached and oc.AttachedConnector is not None:
            h = oc.AttachedConnector.AttachedTo
            ax.annotate("", xy=(h.X, -h.Y - h.Height / 2), xytext=(g.X + g.Width, -g.Y - g.Height / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=.6, color="#1a3a6e", mutation_scale=6), zorder=1)
ax.autoscale(); ax.set_aspect("equal"); ax.axis("off"); plt.tight_layout()
plt.savefig(os.path.join(PREV, "layout_pfd.png"), dpi=34)
for name,(x0,x1) in {"a":(0,3000),"b":(2900,5900),"c":(5800,8800)}.items():
    fig.set_size_inches((x1-x0)/120.0, 1650/120.0)
    ax.set_aspect("equal"); ax.set_xlim(x0,x1); ax.set_ylim(-1800,-150)
    plt.savefig(os.path.join(PREV, "crop_%s.png" % name), dpi=70)
if len(sys.argv) > 1 and sys.argv[1] == "save":
    interf.SaveFlowsheet(fs, FILE, True)
    print("saved", FILE)


