"""Small helper layer over the DWSIM automation API (pythonnet) for the refinery model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dwsim_boot import boot
boot()
import System
from System import Array, Double
from DWSIM.Automation import Automation3
from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType as OT
from DWSIM.Thermodynamics.Utilities.PetroleumCharacterization import PseudoEstimator, PseudoCompoundInput

R_BBL = 0.158987294928  # m3 per bbl


class Plant:
    def __init__(self, pp="Peng-Robinson (PR)"):
        self.interf = Automation3()
        self.fs = self.interf.CreateFlowsheet()
        self.pp_name = pp
        self.objs = {}
        self.lane = {}          # object tag -> flowsheet section (used by layout.py)
        self.section = None
        self._x = 50
        self._y = 50
        self.comps = []

    # ---------------- compounds ----------------
    def add_real(self, names):
        for n in names:
            if n not in self.comps:
                self.fs.AddCompound(n)
                self.comps.append(n)

    def add_pseudo(self, name, nbp_k, sg, mw=None, tc=None, pc_bar=None, omega=None):
        inp = PseudoCompoundInput()
        inp.Name = name
        inp.NBP = Double(nbp_k)
        inp.SG = Double(sg)
        inp.MW = Double(mw if mw else PseudoEstimator.MWFromNBP(nbp_k))
        if tc:
            inp.Tc = Double(tc)
        if pc_bar:
            inp.Pc = Double(pc_bar * 1e5)
        if omega is not None:
            inp.AF = Double(omega)
        cp = PseudoEstimator().Estimate(inp, 1)
        cp.Name = name
        # force the assay-supplied characterisation values
        if mw:
            cp.Molar_Weight = mw
        if tc:
            cp.Critical_Temperature = tc
        if pc_bar:
            cp.Critical_Pressure = pc_bar * 1e5
        if omega is not None:
            cp.Acentric_Factor = omega
            cp.Chao_Seader_Acentricity = omega
        self.fs.SelectedCompounds.Add(name, cp)
        self.comps.append(name)
        return cp

    def finish_compounds(self):
        self.pp = self.fs.CreateAndAddPropertyPackage(self.pp_name)
        # DWSIM's default flash wrongly returns "all vapour" once a heavy liquid forms (VF > ~0.35 with this
        # crude); the Nested-Loops v2 implementation behaves correctly.
        from DWSIM.Interfaces.Enums import FlashSetting
        d = self.pp.GetType().GetProperty("FlashSettings").GetValue(self.pp)
        d[System.Enum.Parse(FlashSetting, "NestedLoops_v2")] = "True"
        self.mw = {n: self.fs.SelectedCompounds[n].Molar_Weight for n in self.comps}

    # ---------------- objects ----------------
    def add(self, kind, tag, x=None, y=None):
        if x is None:
            self._x += 80
            x = self._x
        if y is None:
            y = self._y
        o = self.fs.AddObject(kind, x, y, tag)
        o = o.GetAsObject() if hasattr(o, "GetAsObject") else o
        self.objs[tag] = o
        if self.section:
            self.lane[tag] = self.section
        return o

    def stream(self, tag, T_C=25.0, P_bar=1.01325, mass_kg_h=0.0, w=None, x=None, y=None):
        """Material stream with composition given as mass fractions dict (normalised)."""
        ms = self.add(OT.MaterialStream, tag, x, y)
        ms.Phases[0].Properties.temperature = T_C + 273.15
        ms.Phases[0].Properties.pressure = P_bar * 1e5
        if w:
            self.set_mass_comp(ms, w)
        ms.Phases[0].Properties.massflow = mass_kg_h / 3600.0
        return ms

    def set_mass_comp(self, ms, w):
        tot = sum(w.values())
        n = {k: v / tot / self.mw[k] for k, v in w.items()}
        nt = sum(n.values())
        for c in self.comps:
            ms.Phases[0].Compounds[c].MoleFraction = n.get(c, 0.0) / nt

    def energy(self, tag):
        return self.add(OT.EnergyStream, tag)

    def connect(self, a, b, pa=0, pb=0):
        self.fs.ConnectObjects(a.GraphicObject, b.GraphicObject, pa, pb)

    def solve(self, label=""):
        errs = self.interf.CalculateFlowsheet2(self.fs)
        msgs = [str(e) for e in errs] if errs is not None else []
        if msgs:
            print(f"[solve {label}] {len(msgs)} message(s):")
            for m in msgs[:15]:
                print("   ", m[:300])
        return msgs

    def save(self, path):
        self.interf.SaveFlowsheet(self.fs, path, True)


def ms_summary(ms):
    p = ms.Phases[0].Properties
    return dict(T=p.temperature - 273.15, P=p.pressure / 1e5, kgh=(p.massflow or 0) * 3600,
                vf=(ms.Phases[2].Properties.molarfraction if ms.Phases[2] else None))
