"""Plain-text and Excel report generation."""

from pathlib import Path

from openpyxl import Workbook

from .cost import exchanger_cost_proxy
from .mechanical import BaffleSpacingCheck, VibrationScreening, check_baffle_spacing, fixed_tubesheet_thickness, screen_tube_vibration
from .rating import DesignSolution, RatingInput
from .thermal import tube_flow_regime


def _rows(solution: DesignSolution, case: RatingInput, baffle: BaffleSpacingCheck, vibration: VibrationScreening, tubesheet_m: float, cost: float) -> list[tuple[str, str, object]]:
    g = solution.geometry
    r = solution.rating
    return [
        ("DESIGN SUMMARY", "tube_count", g.tube_count),
        ("DESIGN SUMMARY", "tube_passes", solution.tube_passes),
        ("DESIGN SUMMARY", "shell_inner_diameter_m", g.shell_inner_diameter_m),
        ("DESIGN SUMMARY", "tube_outer_diameter_m", g.tube_outer_diameter_m),
        ("DESIGN SUMMARY", "tube_pitch_m", g.tube_pitch_m),
        ("DESIGN SUMMARY", "tube_length_m", case.tube_length_m),
        ("DESIGN SUMMARY", "baffle_spacing_m", g.baffle_spacing_m),
        ("THERMAL", "target_duty_w", case.target_duty_w),
        ("THERMAL", "rated_duty_w", r.duty_w),
        ("THERMAL", "hot_outlet_k", r.hot_outlet_k),
        ("THERMAL", "cold_outlet_k", r.cold_outlet_k),
        ("THERMAL", "ntu", r.ntu),
        ("THERMAL", "effectiveness", r.effectiveness),
        ("THERMAL", "required_area_m2", r.required_area_m2),
        ("THERMAL", "tube_mass_flow_kg_s", case.tube_mass_flow_kg_s),
        ("THERMAL", "shell_mass_flow_kg_s", case.shell_mass_flow_kg_s),
        ("THERMAL", "area_m2", r.area_m2),
        ("THERMAL", "lmtd_k", r.lmtd_k),
        ("THERMAL", "correction_factor", r.correction_factor),
        ("THERMAL", "tube_h_w_m2_k", r.tube_h_w_m2_k),
        ("THERMAL", "shell_h_w_m2_k", r.shell_h_w_m2_k),
        ("THERMAL", "overall_u_w_m2_k", r.overall_u_w_m2_k),
        ("HYDRAULIC", "tube_pressure_drop_pa", r.tube_pressure_drop_pa),
        ("HYDRAULIC", "shell_pressure_drop_pa", r.shell_pressure_drop_pa),
        ("HYDRAULIC", "tube_velocity_m_s", r.tube_pressure.velocity_m_s),
        ("HYDRAULIC", "tube_reynolds", r.tube_pressure.reynolds),
        ("HYDRAULIC", "tube_regime", tube_flow_regime(r.tube_pressure.reynolds)),
        ("HYDRAULIC", "shell_reynolds", r.shell.reynolds),
        ("HYDRAULIC", "number_of_baffles", r.shell.number_of_baffles),
        ("HYDRAULIC", "shell_velocity_m_s", r.shell_velocity_m_s),
        *[("BELL-DELAWARE", name, value) for name, value in r.shell.factors.items()],
        ("CHECKS", "duty_check", "PASS" if r.meets_duty else "FAIL"),
        ("CHECKS", "outlet_temperature_check", "PASS" if r.meets_outlets else "FAIL"),
        ("CHECKS", "pressure_drop_check", "PASS" if r.meets_pressure_drop else "FAIL"),
        ("CHECKS", "baffle_spacing_check", "PASS" if baffle.passed else "FAIL"),
        ("CHECKS", "baffle_spacing_message", baffle.message),
        ("CHECKS", "tube_sheet_thickness_m", tubesheet_m),
        ("CHECKS", "vibration_check", "PASS" if vibration.passed else "FAIL"),
        ("CHECKS", "vibration_message", vibration.message),
        ("CHECKS", "cost_proxy", cost),
    ]


def generate_report(
    solution: DesignSolution,
    case: RatingInput,
    text_path: str | Path,
    excel_path: str | Path,
    tubesheet_design_pressure_pa: float = 1.0e6,  # ASSUMED, not part of the design case
    tubesheet_allowable_stress_pa: float = 1.1e8,
) -> list[tuple[str, str, object]]:
    """Write a TEMA-style summary as text and an Excel workbook.

    Paths are filesystem paths.  Numerical values retain the package's SI
    units.  The report includes geometry, duty, flow-derived values, U,
    pressures, thermal/mechanical checks, and the cost proxy.
    """
    baffle = check_baffle_spacing(solution.geometry.shell_inner_diameter_m, solution.geometry.baffle_spacing_m)
    vibration = screen_tube_vibration(rating_velocity(solution), case.tube_length_m / max(1, round(case.tube_length_m / solution.geometry.baffle_spacing_m)), solution.geometry.tube_outer_diameter_m, case.tube_inner_diameter_m)
    tubesheet = fixed_tubesheet_thickness(solution.geometry.shell_inner_diameter_m, tubesheet_design_pressure_pa, tubesheet_allowable_stress_pa)
    cost = exchanger_cost_proxy(solution.geometry.shell_inner_diameter_m, case.tube_length_m, solution.rating.area_m2)
    rows = _rows(solution, case, baffle, vibration, tubesheet, cost)
    text_file = Path(text_path)
    text_file.parent.mkdir(parents=True, exist_ok=True)
    with text_file.open("w", encoding="utf-8") as stream:
        current = None
        for section, key, value in rows:
            if section != current:
                stream.write(f"\n[{section}]\n")
                current = section
            stream.write(f"{key}: {value}\n")
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["Section", "Item", "Value"])
    for row in rows:
        summary.append(list(row))
    geometry_sheet = workbook.create_sheet("Geometry")
    geometry_sheet.append(["Parameter", "Value", "Unit"])
    for key, value, unit in (("shell_inner_diameter", solution.geometry.shell_inner_diameter_m, "m"), ("tube_outer_diameter", solution.geometry.tube_outer_diameter_m, "m"), ("tube_pitch", solution.geometry.tube_pitch_m, "m"), ("tube_count", solution.geometry.tube_count, "count"), ("baffle_spacing", solution.geometry.baffle_spacing_m, "m")):
        geometry_sheet.append([key, value, unit])
    checks = workbook.create_sheet("Checks")
    checks.append(["Check", "Result"])
    for key, value in (("duty", "PASS" if solution.rating.meets_duty else "FAIL"), ("pressure_drop", "PASS" if solution.rating.meets_pressure_drop else "FAIL"), ("baffle_spacing", "PASS" if baffle.passed else "FAIL"), ("vibration", "PASS" if vibration.passed else "FAIL")):
        checks.append([key, value])
    excel_file = Path(excel_path)
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(excel_file)
    return rows


def rating_velocity(solution: DesignSolution) -> float:
    """Return shell crossflow velocity in m/s for report screening."""
    return solution.rating.shell_velocity_m_s
