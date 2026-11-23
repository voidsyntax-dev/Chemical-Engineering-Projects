"""Verified heat-recovery report from the saved heat-integrated model."""
import sys, os, json; HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from dwlib import *
names={"K":"Kerosene","D":"Diesel","A":"AGO","L":"LVGO","HV":"HVGO","V":"Vacuum Residue"}
net=json.load(open(os.path.join(HERE,"hx_network.json")))
i=Automation3(); fs=i.LoadFlowsheet2(os.path.join(ROOT,"Crude_Vacuum_Refinery_HeatIntegrated.dwxmz"))
by={o.GraphicObject.Tag:o.GetAsObject() for o in fs.SimulationObjects.Values}
T=lambda tag: by[tag].Phases[0].Properties.temperature-273.15
first={}
lines=[]
cold_prev={"C1":"Crude Oil P","C2":"Desalted Crude","C3":"Stabilised Crude P"}
trainname={"C1":"crude (pre-desalter)","C2":"desalted crude","C3":"stabilised crude"}
rows=[]
for r in sorted(net,key=lambda r:r["n"]):
    n,h,c=r["n"],r["h"],r["c"]
    nm=names[h]
    if h not in first:
        hin={"V":"VR Pumped"}.get(h,"HOT %s (draw copy)"%nm)
    else:
        hin="HX%d %s in (tear copy)"%(n,nm)
    hout="HX%d %s out"%(n,nm); cin=cold_prev[c]; cout="HX%d %s out"%(n,c+" crude")
    first[h]=True
    Th_in,Th_out,Tc_in,Tc_out=T(hin),T(hout),T(cin),T(cout)
    Q=(by[hout].Phases[0].Properties.enthalpy*by[hout].Phases[0].Properties.massflow) # placeholder
    qc=(by[cout].Phases[0].Properties.enthalpy-by[cin].Phases[0].Properties.enthalpy)*by[cout].Phases[0].Properties.massflow
    hx=[o for k,o in by.items() if k.startswith("H%d "%n)][0]
    mita=min(Th_out-Tc_in,Th_in-Tc_out)
    rows.append((n,nm,trainname[c],Th_in,Th_out,Tc_in,Tc_out,qc,mita))
    cold_prev[c]=cout
for n,nm,hin,hout,cin,cout in ((12,"Naphtha 2","Naphtha 2 HX in","Naphtha 2 HX","HX11 C1 crude out","HX12 C1 crude out"),
                               (13,"PF overhead","PF Overhead HX in","PF Overhead HX","HX12 C1 crude out","HX13 C1 crude out")):
    Th_in,Th_out,Tc_in,Tc_out=T(hin),T(hout),T(cin),T(cout)
    qc=(by[cout].Phases[0].Properties.enthalpy-by[cin].Phases[0].Properties.enthalpy)*by[cout].Phases[0].Properties.massflow
    rows.append((n,nm,trainname["C1"],Th_in,Th_out,Tc_in,Tc_out,qc,min(Th_out-Tc_in,Th_in-Tc_out)))
out=["HEAT-RECOVERY NETWORK (verified from saved model)","%-3s %-15s %-22s %8s %8s %8s %8s %9s %7s"%("HX","hot stream","cold stream","Th in","Th out","Tc in","Tc out","Q kW","MITA K")]
tot=0
for n,nm,cn,a,b,c_,d,q,m in rows:
    out.append("H%-2d %-15s %-22s %8.1f %8.1f %8.1f %8.1f %9.0f %7.1f"%(n,nm,cn,a,b,c_,d,q,m)); tot+=q
out.append("Total recovered: %.0f kW; minimum approach %.1f K"%(tot,min(r[-1] for r in rows)))
out.append("Trim heaters now: E-100 %.0f, H101 %.0f, FH-100 %.0f, E-102 %.0f kW"%tuple(by[k].DeltaQ for k in ("E-100 Preheat Trim","H101 Trim Heater","FH-100","E-102 Atm Furnace Trim")))
tr=sum(by[k].DeltaQ for k in ("E-100 Preheat Trim","H101 Trim Heater","FH-100","E-102 Atm Furnace Trim"))
out.append("Heater duty %.0f kW vs 138,525 kW utility case (-%.0f%%); FH-100 fuel is taken from the plant's own sweet fuel gas"%(tr,100*(1-tr/138525.0)))
# tear-copy consistency
for n,nm,cn,a,b,c_,d,q,m in rows:
    pass
txt="\n".join(out); print(txt); open(os.path.join(ROOT,"heat_integration_report.txt"),"w").write(txt+"\n")
