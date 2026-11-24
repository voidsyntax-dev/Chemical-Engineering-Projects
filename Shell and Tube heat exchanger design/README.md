# Shell-and-Tube Heat Exchanger Design Tool

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Status](https://img.shields.io/badge/status-preliminary%20design-orange)

A Python package for preliminary thermal, hydraulic and mechanical design of shell-and-tube heat exchangers. All calculations are in SI units. You can use it as a library, from the command line, or through a small local web interface.

> **Disclaimer:** This tool is for preliminary design and screening only. Final designs must be checked against the applicable TEMA Standards and reviewed by a qualified engineer.

---

## Features

- **Thermal design:** heat duty, LMTD, F-correction factor (1 shell pass, 2+ tube passes), tube-side and Kern shell-side coefficients, overall U, required area and tube count.
- **Bell–Delaware shell side:** ideal cross-flow coefficient with J-factor corrections (Jc, Jl, Jb, Js, Jr) and shell-side pressure-drop components.
- **Hydraulics:** tube-side pressure drop with Colebrook–White friction and Sieder–Tate viscosity correction.
- **Rating and design:** rates a given geometry, or searches tube count, passes, shell diameter and baffle spacing under pressure-drop and duty constraints.
- **Mechanical screening:** fixed tube-sheet thickness, TEMA baffle-spacing limits and vortex-shedding vibration screen.
- **Cost proxy:** a relative figure for comparing design options.
- **Reports:** plain-text and Excel (`.xlsx`) design reports.
- **Interfaces:** a JSON-driven CLI and a local web page that uses only the standard library.

## Project structure

```
.
├── src/she/                # Package source
│   ├── properties.py       # Fluid property container / temperature correlations
│   ├── thermal.py          # Duty, LMTD, F, h, U, area
│   ├── geometry.py         # Bundle / shell geometry
│   ├── bell_delaware.py    # Bell–Delaware shell-side method
│   ├── hydraulics.py       # Tube-side pressure drop
│   ├── rating.py           # Rating and iterative design solver
│   ├── mechanical.py       # Tube sheet, baffles, vibration
│   ├── cost.py             # Cost proxy
│   ├── reporting.py        # Text and Excel reports
│   ├── cli.py              # Command-line interface
│   ├── webapp.py           # Local web server
│   └── static/index.html   # Web UI
├── examples/               # Worked scripts, phases 1–4
├── tests/                  # pytest suite
├── sample_input.json       # Example CLI / web input
├── requirements.txt        # Pinned dependencies
└── pytest.ini
```

## Installation

You need Python 3.12 (the version used for development).

```bash
python -m venv .venv
```

Activate the virtual environment:

| Shell              | Command                         |
|--------------------|---------------------------------|
| Linux / macOS      | `source .venv/bin/activate`     |
| Windows PowerShell | `.venv\Scripts\Activate.ps1`    |
| Windows cmd        | `.venv\Scripts\activate.bat`    |

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

The package uses a `src/` layout. When you run the CLI or the web app with `python -m`, put `src` on the Python path:

- Linux / macOS: `export PYTHONPATH=src`
- Windows PowerShell: `$env:PYTHONPATH="src"`

The example scripts and the tests find `src/` on their own, so they don't need this.

### Command line

```bash
python -m she.cli sample_input.json --text-report report.txt --excel-report report.xlsx
```

The input is a JSON file. See [`sample_input.json`](sample_input.json) for the full schema.

- Temperatures are in K, mass flows in kg/s, dimensions in m, pressures in Pa and duties in W.
- The hot duty (`m·cp·ΔT`), the cold duty and `target_duty_w` must agree within 2 %, or the input is rejected.
- `hot_stream_side` (`"shell"` or `"tube"`) says which side the hot stream is on.
- Unknown fields are treated as errors.

### Web interface

```bash
python -m she.webapp --open          # optional: --port 8000
```

This starts a server on `http://127.0.0.1:8000/`, reachable only from your machine. The page offers:

- a form editor and a raw JSON editor
- **Design** mode, where the solver chooses tubes, passes, shell and baffles
- **Rate** mode, which evaluates exactly the geometry you enter
- pass/fail checks, warnings and a live energy-balance indicator
- Excel report download

### Examples

Run the scripts from any directory:

```bash
python examples/phase1_basic_design.py       # Thermal fundamentals
python examples/phase2_bell_delaware.py      # Kern vs Bell–Delaware
python examples/phase3_rating_and_design.py  # Rating and design solver
python examples/phase4_full_workflow.py      # JSON input -> reports
```

### As a library

```python
from she import FluidProperties, BundleGeometry, RatingInput, DesignConstraints, solve_design

water   = FluidProperties(323, 980, 4180, 0.0008, 0.6)
process = FluidProperties(398, 850, 2400, 0.0012, 0.12)
case  = RatingInput(501_600, 423.15, 373.15, 293.15, 353.15, water, process, 2.0, 4.18, 5.0, 0.016)
trial = BundleGeometry(0.65, 0.01905, 0.0254, 240, baffle_spacing_m=0.2)

solution = solve_design(case, trial, 2, DesignConstraints(max_tube_pressure_drop_pa=200_000,
                                                         max_shell_pressure_drop_pa=100_000))
print(solution.geometry.tube_count, solution.rating.overall_u_w_m2_k)
```

## Testing

```bash
pytest
```

`pytest.ini` puts `src` on the path and collects tests from `tests/`.

## Limitations

- One shell pass only. A single tube pass is treated as counter-current; otherwise the number of tube passes must be even.
- No phase change (no condensation or boiling).
- Bell–Delaware covers 30° and 90° tube layouts only.
- Tube-sheet thickness, vibration (vortex shedding only) and return/nozzle losses are rough screens.
- The cost value is only a proxy for comparing options.
