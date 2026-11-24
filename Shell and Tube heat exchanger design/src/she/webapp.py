"""Local web interface: ``python -m she.webapp`` serves a page on http://127.0.0.1:8000.

Standard library only.  The server binds to the loopback interface, rejects
cross-origin browser requests, and accepts the same JSON document as the CLI
(plus an optional ``mechanical`` object that is used only for the tube-sheet
screen).
"""

import argparse
import json
import math
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .cli import parse_input, parse_json
from .mechanical import check_baffle_spacing, fixed_tubesheet_thickness, screen_tube_vibration
from .rating import DesignSolution, RatingInput, RatingResult, rate_exchanger, solve_design
from .reporting import generate_report
from .thermal import tube_flow_regime

STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "sample_input.json"
MAX_BODY_BYTES = 1_000_000
DEFAULT_DESIGN_PRESSURE_PA = 1.0e6  # assumed; the design case carries no mechanical design pressure
DEFAULT_ALLOWABLE_STRESS_PA = 1.1e8  # assumed


def _clean(value: Any) -> Any:
    """Make a structure JSON-safe: non-finite floats become None, numpy scalars become floats."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    number = float(value)
    return number if math.isfinite(number) else None


def _split_mechanical(data: Any) -> tuple[Any, float, float]:
    """Remove the UI-only ``mechanical`` object from the document and return its two values."""
    if not isinstance(data, dict) or "mechanical" not in data:
        return data, DEFAULT_DESIGN_PRESSURE_PA, DEFAULT_ALLOWABLE_STRESS_PA
    data = dict(data)
    mech = data.pop("mechanical")
    if not isinstance(mech, dict) or set(mech) - {"design_pressure_pa", "allowable_stress_pa"}:
        raise ValueError("mechanical must be an object with design_pressure_pa and/or allowable_stress_pa")
    values = []
    for key, default in (("design_pressure_pa", DEFAULT_DESIGN_PRESSURE_PA), ("allowable_stress_pa", DEFAULT_ALLOWABLE_STRESS_PA)):
        value = mech.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"mechanical.{key} must be a finite positive number")
        values.append(float(value))
    return data, values[0], values[1]


def _warnings(case: RatingInput, rating: RatingResult, geometry_passes: int) -> list[str]:
    notes = []
    tube_re = rating.tube_pressure.reynolds
    regime = tube_flow_regime(tube_re)
    if regime == "laminar":
        notes.append(f"Tube-side flow is laminar (Re {tube_re:.0f}); h uses the constant Nu=3.66 and is sensitive to maldistribution and free convection.")
    elif regime == "transitional":
        notes.append(f"Tube-side Re {tube_re:.0f} is in the transitional range (2300-10000); h is a linear blend, a screening estimate.")
    if rating.shell.reynolds <= 100:
        notes.append(f"Shell-side flow is laminar (Re {rating.shell.reynolds:.0f}); laminar Bell-Delaware terms apply.")
    if 0 < rating.correction_factor < 0.8:
        notes.append(f"LMTD correction factor F={rating.correction_factor:.2f} is low (guideline: below about 0.8 is uneconomic); consider more shell passes.")
    if not math.isfinite(rating.correction_factor):
        notes.append("The specified outlet temperatures cannot be reached in one shell pass (temperature cross).")
    if rating.shell.factors["Jb"] < 0.6:
        notes.append(f"Bundle-bypass factor Jb={rating.shell.factors['Jb']:.2f} is low; add seal strips or reduce the bundle-to-shell gap.")
    if rating.shell.factors["Jl"] < 0.6:
        notes.append(f"Leakage factor Jl={rating.shell.factors['Jl']:.2f} is low; baffle clearances or spacing are leaking a lot of flow.")
    if geometry_passes == 1:
        notes.append("One tube pass is treated as pure counter-current (F=1).")
    notes.append("Vibration check covers vortex-shedding resonance only; fluid-elastic instability is not screened.")
    notes.append(f"Tube-sheet value uses an assumed design pressure and allowable stress and is not the TEMA bending equation.")
    return notes


def _payload(case: RatingInput, geometry, passes: int, rating: RatingResult, pressure_pa: float, stress_pa: float,
             limits, iterations: int | None) -> dict[str, Any]:
    baffle = check_baffle_spacing(geometry.shell_inner_diameter_m, geometry.baffle_spacing_m)
    span = case.tube_length_m / max(1, round(case.tube_length_m / geometry.baffle_spacing_m))
    vibration = screen_tube_vibration(rating.shell_velocity_m_s, span, geometry.tube_outer_diameter_m, case.tube_inner_diameter_m)
    tubesheet = fixed_tubesheet_thickness(geometry.shell_inner_diameter_m, pressure_pa, stress_pa)
    checks = [
        {"name": "Duty meets target", "pass": rating.meets_duty, "detail": f"{rating.duty_w / 1e3:.2f} kW vs {case.target_duty_w / 1e3:.2f} kW"},
        {"name": "Outlet temperatures met", "pass": rating.meets_outlets,
         "detail": f"hot {rating.hot_outlet_k:.2f} K (spec {case.hot_outlet_k:.2f}), cold {rating.cold_outlet_k:.2f} K (spec {case.cold_outlet_k:.2f})"},
        {"name": "Tube-side pressure drop", "pass": rating.tube_pressure_drop_pa <= limits.max_tube_pressure_drop_pa,
         "detail": f"{rating.tube_pressure_drop_pa / 1e3:.2f} kPa (limit {limits.max_tube_pressure_drop_pa / 1e3:.0f} kPa)"},
        {"name": "Shell-side pressure drop", "pass": rating.shell_pressure_drop_pa <= limits.max_shell_pressure_drop_pa,
         "detail": f"{rating.shell_pressure_drop_pa / 1e3:.2f} kPa (limit {limits.max_shell_pressure_drop_pa / 1e3:.0f} kPa)"},
        {"name": "Bundle fits shell", "pass": geometry.bundle_fits(),
         "detail": f"OTL {geometry.bundle_diameter_m * 1e3:.0f} mm in {geometry.shell_inner_diameter_m * 1e3:.0f} mm shell"},
        {"name": "TEMA baffle spacing", "pass": baffle.passed, "detail": f"{geometry.baffle_spacing_m * 1e3:.0f} mm; allowed {baffle.minimum_m * 1e3:.0f}-{baffle.maximum_m * 1e3:.0f} mm"},
        {"name": "Vortex-shedding screen", "pass": vibration.passed,
         "detail": f"fs/fn = {vibration.frequency_ratio:.3f} (fn {vibration.natural_frequency_hz:.0f} Hz, span {span * 1e3:.0f} mm)"},
    ]
    return _clean({
        "feasible": all(c["pass"] for c in checks[:6]),
        "iterations": iterations,
        "geometry": {
            "tube_count": geometry.tube_count, "tube_passes": passes, "shell_inner_diameter_m": geometry.shell_inner_diameter_m,
            "tube_outer_diameter_m": geometry.tube_outer_diameter_m, "tube_inner_diameter_m": case.tube_inner_diameter_m,
            "tube_pitch_m": geometry.tube_pitch_m, "tube_length_m": case.tube_length_m, "layout": geometry.tube_layout,
            "baffle_spacing_m": geometry.baffle_spacing_m, "baffle_cut_fraction": geometry.baffle_cut_fraction,
            "number_of_baffles": rating.shell.number_of_baffles, "bundle_diameter_m": geometry.bundle_diameter_m,
            "area_m2": rating.area_m2, "tubesheet_thickness_m": tubesheet,
        },
        "thermal": {
            "target_duty_w": case.target_duty_w, "duty_w": rating.duty_w, "overall_u_w_m2_k": rating.overall_u_w_m2_k,
            "tube_h_w_m2_k": rating.tube_h_w_m2_k, "shell_h_w_m2_k": rating.shell_h_w_m2_k,
            "area_m2": rating.area_m2, "required_area_m2": rating.required_area_m2,
            "area_margin_pct": (rating.area_m2 / rating.required_area_m2 - 1) * 100 if rating.required_area_m2 else None,
            "lmtd_k": rating.lmtd_k, "correction_factor": rating.correction_factor, "ntu": rating.ntu,
            "effectiveness": rating.effectiveness, "hot_outlet_k": rating.hot_outlet_k, "cold_outlet_k": rating.cold_outlet_k,
        },
        "hydraulic": {
            "tube_velocity_m_s": rating.tube_pressure.velocity_m_s, "tube_reynolds": rating.tube_pressure.reynolds,
            "tube_regime": tube_flow_regime(rating.tube_pressure.reynolds), "tube_friction_factor": rating.tube_pressure.friction_factor,
            "tube_dp_pa": rating.tube_pressure_drop_pa,
            "tube_dp_parts_pa": {"friction": rating.tube_pressure.friction_pa, "return": rating.tube_pressure.return_loss_pa, "nozzles": rating.tube_pressure.nozzle_loss_pa},
            "shell_velocity_m_s": rating.shell_velocity_m_s, "shell_reynolds": rating.shell.reynolds,
            "shell_dp_pa": rating.shell_pressure_drop_pa, "shell_dp_parts_pa": rating.shell.pressure_components_pa,
        },
        "bell_delaware": {"ideal_h_w_m2_k": rating.shell.ideal_h_w_m2_k, "factors": rating.shell.factors, "pressure_factors": rating.shell.pressure_factors},
        "checks": checks,
        "warnings": _warnings(case, rating, passes),
    })


def run_design(data: Any) -> dict[str, Any]:
    data, pressure, stress = _split_mechanical(data)
    case, geometry, passes, constraints = parse_input(data)
    solution = solve_design(case, geometry, passes, constraints)
    return _payload(case, solution.geometry, solution.tube_passes, solution.rating, pressure, stress, constraints, solution.iterations)


def run_rating(data: Any) -> dict[str, Any]:
    data, pressure, stress = _split_mechanical(data)
    case, geometry, passes, constraints = parse_input(data)
    rating = rate_exchanger(case, geometry, passes, constraints)
    return _payload(case, geometry, passes, rating, pressure, stress, constraints, None)


def build_report(data: Any) -> bytes:
    data, pressure, stress = _split_mechanical(data)
    case, geometry, passes, constraints = parse_input(data)
    solution: DesignSolution = solve_design(case, geometry, passes, constraints)
    with tempfile.TemporaryDirectory() as tmp:
        generate_report(solution, case, Path(tmp) / "r.txt", Path(tmp) / "r.xlsx", pressure, stress)
        return (Path(tmp) / "r.xlsx").read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "she-web"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        pass

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(_clean(payload)).encode("utf-8"), "application/json; charset=utf-8")

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc == self.headers.get("Host")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/sample":
            try:
                self._send(200, SAMPLE_PATH.read_bytes(), "application/json; charset=utf-8")
            except OSError:
                self._json(404, {"error": "sample_input.json not found"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not self._same_origin():
            self._json(403, {"error": "cross-origin request refused"})
            return
        routes = {"/api/design": run_design, "/api/rate": run_rating}
        if path not in routes and path != "/api/report":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("request body must be between 1 byte and 1 MB")
            data = parse_json(self.rfile.read(length).decode("utf-8"))
            if path == "/api/report":
                self._send(200, build_report(data), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           {"Content-Disposition": 'attachment; filename="heat_exchanger_report.xlsx"'})
            else:
                self._json(200, routes[path](data))
        except (ValueError, RuntimeError, ArithmeticError, UnicodeDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # unexpected: report, do not crash the server
            self._json(500, {"error": f"internal error: {type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local web interface for the shell-and-tube design tool")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
