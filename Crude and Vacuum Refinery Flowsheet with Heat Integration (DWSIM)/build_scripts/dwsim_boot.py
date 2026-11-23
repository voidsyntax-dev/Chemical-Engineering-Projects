"""Bootstrap DWSIM (.NET 10) automation from Python via pythonnet."""
import os, sys, json

DWSIM_DIR = os.path.join(os.environ["LOCALAPPDATA"], "Programs", "DWSIM")
HERE = os.path.dirname(os.path.abspath(__file__))
RTCONFIG = os.path.join(HERE, "dwsim.runtimeconfig.json")


def boot():
    if not os.path.exists(RTCONFIG):
        with open(RTCONFIG, "w") as f:
            json.dump({"runtimeOptions": {"tfm": "net10.0",
                       "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"},
                       "rollForward": "LatestMinor"}}, f)
    from pythonnet import load
    load("coreclr", runtime_config=RTCONFIG)
    import clr
    sys.path.append(DWSIM_DIR)
    os.environ["PATH"] = DWSIM_DIR + os.pathsep + os.environ["PATH"]
    os.chdir(DWSIM_DIR)  # DWSIM resolves data files relative to its dir
    for dll in ["CapeOpen", "DWSIM.Automation", "DWSIM.Interfaces", "DWSIM.GlobalSettings",
                "DWSIM.SharedClasses", "DWSIM.Thermodynamics", "DWSIM.UnitOperations",
                "DWSIM.FlowsheetBase", "DWSIM.FlowsheetSolver", "DWSIM.MathOps",
                "DWSIM.XMLSerializer", "DWSIM.ExtensionMethods"]:
        clr.AddReference(os.path.join(DWSIM_DIR, dll + ".dll"))
    from DWSIM.GlobalSettings import Settings
    Settings.EnableParallelProcessing = False   # deterministic, thread-safe solving of heavy-component flashes
    Settings.MaxDegreeOfParallelism = 1
    return clr


if __name__ == "__main__":
    clr = boot()
    from DWSIM.Automation import Automation3
    interf = Automation3()
    fs = interf.CreateFlowsheet()
    print("flowsheet ok:", fs)
    print("compounds in db:", len(fs.AvailableCompounds))
    print("prop packages:", len(fs.PropertyPackages) if hasattr(fs, "PropertyPackages") else "n/a")
