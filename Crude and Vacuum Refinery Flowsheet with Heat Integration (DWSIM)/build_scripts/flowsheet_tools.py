"""Higher-level builders on top of dwlib.Plant: heaters, valves, mixers, flash drums, cut-point towers."""
import math
from dwlib import *
from DWSIM.UnitOperations.UnitOperations.Auxiliary import ComponentSeparationSpec, SeparationSpec


import os
USE_ENERGY_STREAMS = os.environ.get("ENERGY_STREAMS", "0") == "1"   # DWSIM cannot order very large flowsheets; energy-stream objects are optional


class DutyRef:
    """Stand-in for an energy stream: reads the duty straight from the unit operation (kW)."""
    def __init__(self, unit):
        self.unit = unit

    @property
    def EnergyFlow(self):
        for a in ("DeltaQ", "EnergyFlow"):
            try:
                v = getattr(self.unit, a)
                if v is not None:
                    return v
            except Exception:
                pass
        return None


def set_enum(obj, prop, val):
    t = obj.GetType().GetProperty(prop).PropertyType
    obj.GetType().GetProperty(prop).SetValue(obj, System.Enum.Parse(t, val))


def new_stream(P, tag):
    return P.stream(tag)


def recycle(P, src, dst, tag=None):
    """DWSIM Recycle block: src (calculated) -> dst (guess feed of the downstream loop). Makes the tear a real connection."""
    r = P.add(OT.OT_Recycle, tag or ("RC " + dst.GraphicObject.Tag))
    P.connect(src, r, 0, 0)
    P.connect(r, dst, 0, 0)
    return r


def stream_splitter(P, tag, inlet, out1, out2, mass_kg_h):
    """Two-way stream splitter with a mass-flow specification on outlet 1."""
    s = P.add(OT.NodeOut, tag)
    P.connect(inlet, s, 0, 0)
    P.connect(s, out1, 0, 0)
    P.connect(s, out2, 1, 0)
    set_enum(s, "OperationMode", "StreamMassFlowSpec")
    s.StreamFlowSpec = mass_kg_h / 3600.0
    return s


def heater(P, tag, inlet, out, T=None, dP=0.0, vf=None, Q_kw=None, cooler=False):
    """Heater/Cooler. Specify outlet T (C) or vapour fraction or duty (kW). Returns (unit, energy_stream)."""
    h = P.add(OT.Cooler if cooler else OT.Heater, tag)
    P.connect(inlet, h, 0, 0)
    P.connect(h, out, 0, 0)
    if USE_ENERGY_STREAMS:
        e = P.energy("Q_" + tag)
        P.connect(e, h, 0, 1)
    else:
        e = DutyRef(h)
    if T is not None:
        set_enum(h, "CalcMode", "OutletTemperature")
        h.OutletTemperature = T + 273.15
    elif vf is not None:
        set_enum(h, "CalcMode", "OutletVaporFraction")
        h.OutletVaporFraction = vf
    else:
        set_enum(h, "CalcMode", "HeatRemoved" if cooler else "HeatAdded")
        h.DeltaQ = Q_kw
    h.DeltaP = dP * 1e5
    return h, e


def valve(P, tag, inlet, out, dP=None, Pout=None):
    v = P.add(OT.Valve, tag)
    P.connect(inlet, v, 0, 0)
    P.connect(v, out, 0, 0)
    if Pout is not None:
        set_enum(v, "CalcMode", "OutletPressure")
        v.OutletPressure = Pout * 1e5
    else:
        set_enum(v, "CalcMode", "DeltaP")
        v.DeltaP = dP * 1e5
    return v


def pump(P, tag, inlet, out, Pout_bar, eff=0.75):
    p = P.add(OT.Pump, tag)
    P.connect(inlet, p, 0, 0)
    P.connect(p, out, 0, 0)
    if USE_ENERGY_STREAMS:
        e = P.energy("W_" + tag)
        P.connect(e, p, 0, 1)
    else:
        e = DutyRef(p)
    set_enum(p, "CalcMode", "OutletPressure")
    p.Pout = Pout_bar * 1e5
    p.Eficiencia = eff * 100.0 if eff <= 1.0 else eff
    return p, e


def mixer(P, tag, inlets, out):
    m = P.add(OT.Mixer, tag)
    for i, s in enumerate(inlets):
        P.connect(s, m, 0, i)
    P.connect(m, out, 0, 0)
    set_enum(m, "PressureCalculation", "Minimum")
    return m


def flash_drum(P, tag, inlet, vapor=None, liquid=None, heavy=None, T=None, p_bar=None):
    """Two/three-phase separator (adiabatic by default)."""
    v = P.add(OT.Vessel, tag)
    P.connect(inlet, v, 0, 0)
    if vapor is not None:
        P.connect(v, vapor, 0, 0)
    if liquid is not None:
        P.connect(v, liquid, 1, 0)
    if heavy is not None:
        P.connect(v, heavy, 2, 0)
    return v


def splitter_components(P, tag, inlet, out1, out2, frac_to_out2, with_energy=True):
    """ComponentSeparator. frac_to_out2: {compound: fraction (0-1) of inlet MASS flow going to Outlet 2}."""
    s = P.add(OT.ComponentSeparator, tag)
    P.connect(inlet, s, 0, 0)
    P.connect(s, out1, 0, 0)
    P.connect(s, out2, 1, 0)
    s.SpecifiedStreamIndex = 1
    for c in P.comps:
        f = float(frac_to_out2.get(c, 0.0))
        sp = ComponentSeparationSpec()
        sp.ComponentID = c
        sp.SepSpec = SeparationSpec.PercentInletMassFlow
        sp.SpecValue = 100.0 * min(1.0, max(0.0, f))
        sp.SpecUnit = "%"
        s.ComponentSepSpecs[c] = sp
    return s


def compressor(P, tag, inlet, out, Pout_bar, eff=75.0):
    c = P.add(OT.Compressor, tag)
    P.connect(inlet, c, 0, 0)
    P.connect(c, out, 0, 0)
    if USE_ENERGY_STREAMS:
        e = P.energy("W_" + tag)
        P.connect(e, c, 0, 1)
    else:
        e = DutyRef(c)
    set_enum(c, "CalcMode", "OutletPressure")
    c.POut = Pout_bar * 1e5
    c.AdiabaticEfficiency = eff
    return c, e


def rigorous_column(P, tag, feed, dist, bott, nst, feed_stage, ptop_bar, pbot_bar, rr,
                    solver="Wang-Henke (Bubble Point)", maxit=100, condenser="Total_Condenser", vapor=None):
    """Rigorous DWSIM distillation column (specs: reflux ratio + bottoms molar flow, set later by tune_column)."""
    from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import ColumnSpec
    col = P.add(OT.DistillationColumn, tag)
    col.SetNumberOfStages(nst)
    qc = P.energy("Qc_" + tag)
    qr = P.energy("Qr_" + tag)
    col.ConnectFeed(feed, feed_stage)
    col.ConnectDistillate(dist)
    col.ConnectBottoms(bott)
    if vapor is not None:
        col.ConnectVaporProduct(vapor)
    col.ConnectCondenserDuty(qc)
    col.ConnectReboilerDuty(qr)
    set_enum(col, "CondenserType", condenser)
    col.SetTopPressure(ptop_bar * 1e5)
    col.SetBottomStagePressure(pbot_bar * 1e5)
    col.Specs["C"].SType = ColumnSpec.SpecType.Stream_Ratio
    col.Specs["C"].SpecValue = rr
    col.Specs["R"].SType = ColumnSpec.SpecType.Product_Molar_Flow_Rate
    col.Specs["R"].SpecUnit = "mol/s"
    col.Specs["R"].SpecValue = 1.0
    col.SolvingMethodName = solver
    col.MaxIterations = maxit
    return col, qc, qr


def tune_column(col, feed, bottoms_comps, comps):
    """Set the bottoms molar-flow spec = moles of `bottoms_comps` in the (solved) feed."""
    ph = feed.Phases[0]
    n = ph.Properties.molarflow
    nb = n * sum(ph.Compounds[c].MoleFraction for c in bottoms_comps)
    col.Specs["R"].SpecValue = max(nb, 1e-6)
    return nb, n


def nbp_c(P, comp):
    return P.fs.SelectedCompounds[comp].Normal_Boiling_Point - 273.15


def cut_fraction(P, tcut_c, width_c=7.0, skip=()):
    """Fraction of each component going to the HEAVY side of a cut at tcut (logistic TBP overlap)."""
    out = {}
    for c in P.comps:
        tb = nbp_c(P, c)
        if c in skip:
            continue
        x = (tb - tcut_c) / width_c
        out[c] = 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))
    return out



