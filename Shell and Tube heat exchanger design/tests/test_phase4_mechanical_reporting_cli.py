import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook

from she.cli import load_input
from she.cost import exchanger_cost_proxy
from she.mechanical import check_baffle_spacing, fixed_tubesheet_thickness, screen_tube_vibration
from she.rating import solve_design
from she.reporting import generate_report

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = json.loads((ROOT / "sample_input.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def case_and_solution():
    case, trial, passes, constraints = load_input(ROOT / "sample_input.json")
    return case, solve_design(case, trial, passes, constraints)


# ---------------------------------------------------------------- mechanical
def test_tube_sheet_equation_as_implemented():
    # NOTE: this checks the arithmetic of the implemented screening formula t = D sqrt(P/(S E - 0.6 P)) + CA only.
    # It is NOT the TEMA bending equation; no published value is available offline.
    assert fixed_tubesheet_thickness(0.6, 1e6, 110e6) == pytest.approx(0.6 * (1e6 / (110e6 - 0.6e6)) ** 0.5, rel=1e-12)
    assert fixed_tubesheet_thickness(0.6, 1e6, 110e6, corrosion_allowance_m=0.003) == pytest.approx(0.0573644 + 0.003, rel=1e-5)
    with pytest.raises(ValueError):
        fixed_tubesheet_thickness(0.6, 1e8, 1e8, 0.5)  # allowable stress too low


@pytest.mark.parametrize("spacing,passes", [
    (0.12, True), (0.11, True), (0.55, True),  # exactly 0.2 D and exactly D are inside the limits
    (0.1099, False), (0.0508, False), (0.56, False),
])
def test_tema_baffle_limits(spacing, passes):
    assert check_baffle_spacing(0.55, spacing).passed is passes


def test_absolute_minimum_spacing_applies_to_small_shells():
    check = check_baffle_spacing(0.15, 0.04)  # 0.2 * 0.15 = 0.03 < 50.8 mm
    assert check.minimum_m == pytest.approx(0.0508) and not check.passed


def test_vibration_resonance_band():
    do, di = 0.01905, 0.016
    low = screen_tube_vibration(1.0, 0.2, do, di)
    assert low.passed and low.frequency_ratio < 0.8
    fn = low.natural_frequency_hz
    # velocity that puts the shedding frequency exactly at fn (ratio 1) must fail; 1.5x fn must pass again
    v_res = fn * do / 0.2
    assert not screen_tube_vibration(v_res, 0.2, do, di).passed
    assert screen_tube_vibration(1.5 * v_res, 0.2, do, di).passed
    assert fn == pytest.approx(3.141592653589793 / (2 * 0.2**2) * ((2e11 * 3.141592653589793 * (do**4 - di**4) / 64)
                                                                   / (7850 * 3.141592653589793 * (do**2 - di**2) / 4)) ** 0.5)
    with pytest.raises(ValueError):
        screen_tube_vibration(1.0, 0.2, 0.016, 0.02)


def test_cost_proxy():
    assert exchanger_cost_proxy(0.6, 5, 10, 1.5) == pytest.approx(1.5 * (3.141592653589793 * 0.6 * 5 + 10))
    with pytest.raises(ValueError):
        exchanger_cost_proxy(0.6, 5, 0)


# ---------------------------------------------------------------- report
def test_report_text_and_excel_match_internal_results(case_and_solution, tmp_path):
    case, solution = case_and_solution
    rows = generate_report(solution, case, tmp_path / "r.txt", tmp_path / "r.xlsx")
    r = solution.rating
    expected = {
        "rated_duty_w": r.duty_w, "overall_u_w_m2_k": r.overall_u_w_m2_k, "area_m2": r.area_m2,
        "tube_pressure_drop_pa": r.tube_pressure_drop_pa, "shell_pressure_drop_pa": r.shell_pressure_drop_pa,
        "cold_outlet_k": r.cold_outlet_k, "hot_outlet_k": r.hot_outlet_k, "tube_count": solution.geometry.tube_count,
        "shell_inner_diameter_m": solution.geometry.shell_inner_diameter_m, "tube_reynolds": r.tube_pressure.reynolds,
        "Jc": r.shell.factors["Jc"], "Jb": r.shell.factors["Jb"],
    }
    by_key = {key: value for _, key, value in rows}
    text = dict(line.split(": ", 1) for line in (tmp_path / "r.txt").read_text(encoding="utf-8").splitlines() if ": " in line)
    sheet = {row[1].value: row[2].value for row in load_workbook(tmp_path / "r.xlsx")["Summary"].iter_rows(min_row=2)}
    for key, value in expected.items():
        assert by_key[key] == pytest.approx(value)
        assert float(text[key]) == pytest.approx(value)
        assert sheet[key] == pytest.approx(value)
    assert text["duty_check"] == "PASS" and sheet["outlet_temperature_check"] == "PASS"
    assert {"Summary", "Geometry", "Checks"}.issubset(load_workbook(tmp_path / "r.xlsx").sheetnames)


# ---------------------------------------------------------------- CLI
def run_cli(path, tmp_path):
    return subprocess.run(
        [sys.executable, "-m", "she.cli", str(path), "--text-report", str(tmp_path / "o.txt"), "--excel-report", str(tmp_path / "o.xlsx")],
        capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_cli_end_to_end_on_sample(tmp_path):
    result = run_cli(ROOT / "sample_input.json", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "o.txt").exists() and (tmp_path / "o.xlsx").exists()


def _mutated(tmp_path, mutate=None, raw=None):
    data = copy.deepcopy(SAMPLE)
    if mutate:
        mutate(data)
    path = tmp_path / "in.json"
    path.write_text(raw if raw is not None else json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("name,mutate,message", [
    ("string number", lambda d: d["hot_stream"].__setitem__("density_kg_m3", "abc"), "hot_stream.density_kg_m3 must be a number"),
    ("null number", lambda d: d["hot_stream"].__setitem__("mass_flow_kg_s", None), "hot_stream.mass_flow_kg_s must be a number"),
    ("bool number", lambda d: d.__setitem__("tube_length_m", True), "tube_length_m must be a number"),
    ("null count", lambda d: d["geometry"].__setitem__("tube_count", None), "geometry.tube_count must be a number"),
    ("fractional count", lambda d: d["geometry"].__setitem__("tube_count", 240.7), "geometry.tube_count must be a whole number"),
    ("fractional passes", lambda d: d.__setitem__("tube_passes", 2.9), "tube_passes must be a whole number"),
    ("missing field", lambda d: d["cold_stream"].pop("viscosity_pa_s"), "missing required field: cold_stream.viscosity_pa_s"),
    ("missing object", lambda d: d.pop("geometry"), "geometry is required"),
    ("unknown top key", lambda d: d.__setitem__("tube_lenght_m", 5), "unknown field(s) in input: tube_lenght_m"),
    ("unknown constraint key", lambda d: d["constraints"].__setitem__("max_tube_pressure_drop", 1), "unknown field(s) in constraints"),
    ("bad layout", lambda d: d["geometry"].__setitem__("tube_layout", "hex"), "geometry.tube_layout must be one of"),
    ("bad objective", lambda d: d["constraints"].__setitem__("objective", "cheap"), "constraints.objective must be one of"),
    ("bad side", lambda d: d.__setitem__("hot_stream_side", "both"), "hot_stream_side must be one of"),
    ("negative flow", lambda d: d["hot_stream"].__setitem__("mass_flow_kg_s", -1), "mass_flow_kg_s must be a finite positive"),
    ("hot outlet above inlet", lambda d: d["hot_stream"].__setitem__("outlet_temperature_k", 450), "hot stream must cool"),
    ("cold range zero", lambda d: d["cold_stream"].__setitem__("outlet_temperature_k", 293.15), "hot stream must cool"),
    ("energy imbalance", lambda d: d["hot_stream"].__setitem__("mass_flow_kg_s", 8.0), "energy balance fails"),
    ("target mismatch", lambda d: d.__setitem__("target_duty_w", 100000), "target_duty_w"),
    ("Di > Do", lambda d: d.__setitem__("tube_inner_diameter_m", 0.03), "tube_outer_diameter_m"),
])
def test_cli_rejects_bad_input_with_clear_message(tmp_path, name, mutate, message):
    result = run_cli(_mutated(tmp_path, mutate), tmp_path)
    assert result.returncode == 2, (name, result.stderr)
    assert "Traceback" not in result.stderr
    if message:
        assert message in result.stderr


@pytest.mark.parametrize("raw,message", [
    ("{bad", "invalid JSON"),
    ("[]", "input root must be a JSON object"),
    ("{}", "hot_stream is required"),
    (json.dumps(SAMPLE).replace('"target_duty_w": 501600', '"target_duty_w": NaN'), "NaN is not allowed"),
])
def test_cli_rejects_malformed_documents(tmp_path, raw, message):
    result = run_cli(_mutated(tmp_path, raw=raw), tmp_path)
    assert result.returncode == 2 and message in result.stderr and "Traceback" not in result.stderr


def test_cli_reports_missing_file_and_infeasible_design(tmp_path):
    missing = run_cli(tmp_path / "nope.json", tmp_path)
    assert missing.returncode == 2 and "cannot read input file" in missing.stderr
    infeasible = run_cli(_mutated(tmp_path, lambda d: d["constraints"].__setitem__("max_shell_pressure_drop_pa", 1)), tmp_path)
    assert infeasible.returncode == 2 and "design solver failed" in infeasible.stderr


def test_cli_tube_side_hot_stream_option(tmp_path):
    def swap(d):
        d["hot_stream_side"] = "tube"
        d["hot_stream"]["mass_flow_kg_s"], d["cold_stream"]["mass_flow_kg_s"] = 4.18, 2.0
    result = run_cli(_mutated(tmp_path, swap), tmp_path)
    assert result.returncode == 0, result.stderr


def test_cli_wall_viscosity_fields_are_accepted(tmp_path):
    result = run_cli(_mutated(tmp_path, lambda d: d["cold_stream"].__setitem__("wall_viscosity_pa_s", 0.0006)), tmp_path)
    assert result.returncode == 0, result.stderr
