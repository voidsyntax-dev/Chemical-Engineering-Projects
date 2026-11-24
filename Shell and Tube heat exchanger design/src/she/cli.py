"""Command-line interface for JSON-driven exchanger design."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .geometry import BundleGeometry
from .properties import FluidProperties
from .rating import DesignConstraints, RatingInput, solve_design
from .reporting import generate_report

_STREAM_KEYS = {"mass_flow_kg_s", "inlet_temperature_k", "outlet_temperature_k", "density_kg_m3",
                "specific_heat_j_kg_k", "viscosity_pa_s", "conductivity_w_m_k", "wall_viscosity_pa_s"}
_GEOMETRY_KEYS = {"shell_inner_diameter_m", "tube_outer_diameter_m", "tube_pitch_m", "tube_count",
                  "baffle_cut_fraction", "baffle_spacing_m", "tube_layout", "tube_to_baffle_clearance_m",
                  "shell_to_baffle_clearance_m", "seal_strips"}
_CONSTRAINT_KEYS = {"max_tube_pressure_drop_pa", "max_shell_pressure_drop_pa", "minimum_duty_fraction",
                    "objective", "max_iterations", "max_shell_diameter_m"}
_TOP_KEYS = {"target_duty_w", "hot_stream", "cold_stream", "hot_stream_side", "tube_length_m",
             "tube_inner_diameter_m", "tube_passes", "geometry", "constraints", "wall_conductivity_w_m_k",
             "fouling_inner_m2_k_w", "fouling_outer_m2_k_w"}


def _object(mapping: dict[str, Any], key: str, allowed: set[str], path: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}{key} is required and must be a JSON object")
    _no_unknown(value, allowed, f"{path}{key}")
    return value


def _no_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown field(s) in {where}: {', '.join(unknown)}")


def _number(mapping: dict[str, Any], key: str, where: str, default: float | None = None) -> float:
    if key not in mapping:
        if default is None:
            raise ValueError(f"missing required field: {where}{key}")
        return default
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}{key} must be a number, got {type(value).__name__} {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where}{key} must be finite")
    return float(value)


def _optional_number(mapping: dict[str, Any], key: str, where: str) -> float | None:
    return None if mapping.get(key) is None else _number(mapping, key, where)


def _integer(mapping: dict[str, Any], key: str, where: str, default: int | None = None) -> int:
    value = _number(mapping, key, where, None if default is None else float(default))
    if value != int(value):
        raise ValueError(f"{where}{key} must be a whole number, got {value}")
    return int(value)


def _text(mapping: dict[str, Any], key: str, where: str, default: str, choices: set[str]) -> str:
    value = mapping.get(key, default)
    if value not in choices:
        raise ValueError(f"{where}{key} must be one of {sorted(choices)}, got {value!r}")
    return value


def _fluid(stream: dict[str, Any], where: str) -> FluidProperties:
    mean_temperature = 0.5 * (_number(stream, "inlet_temperature_k", where) + _number(stream, "outlet_temperature_k", where))
    return FluidProperties(
        mean_temperature,
        _number(stream, "density_kg_m3", where),
        _number(stream, "specific_heat_j_kg_k", where),
        _number(stream, "viscosity_pa_s", where),
        _number(stream, "conductivity_w_m_k", where),
    )


def _reject_constant(name: str) -> None:
    raise ValueError(f"invalid JSON input: {name} is not allowed")


def parse_json(text: str) -> Any:
    """Parse JSON text, rejecting ``NaN``/``Infinity`` and reporting the error position."""
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {exc.msg} at line {exc.lineno}, column {exc.colno}") from exc


def load_input(path: str | Path) -> tuple[RatingInput, BundleGeometry, int, DesignConstraints]:
    """Read and validate a JSON input file, returning design objects."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read input file: {exc}") from exc
    return parse_input(parse_json(text))


def parse_input(data: Any) -> tuple[RatingInput, BundleGeometry, int, DesignConstraints]:
    """Validate a decoded JSON document, returning design objects.

    Unknown fields, wrong types, non-finite numbers and fractional integers
    are rejected with the offending field named.
    """
    if not isinstance(data, dict):
        raise ValueError("input root must be a JSON object")
    _no_unknown(data, _TOP_KEYS, "input")
    hot = _object(data, "hot_stream", _STREAM_KEYS, "")
    cold = _object(data, "cold_stream", _STREAM_KEYS, "")
    geometry_data = _object(data, "geometry", _GEOMETRY_KEYS, "")
    constraints_data = data.get("constraints", {})
    if not isinstance(constraints_data, dict):
        raise ValueError("constraints must be a JSON object")
    _no_unknown(constraints_data, _CONSTRAINT_KEYS, "constraints")
    hot_side = _text(data, "hot_stream_side", "", "shell", {"shell", "tube"})
    tube_stream, shell_stream = (cold, hot) if hot_side == "shell" else (hot, cold)
    tube_label, shell_label = ("cold_stream.", "hot_stream.") if hot_side == "shell" else ("hot_stream.", "cold_stream.")
    case = RatingInput(
        _number(data, "target_duty_w", ""),
        _number(hot, "inlet_temperature_k", "hot_stream."),
        _number(hot, "outlet_temperature_k", "hot_stream."),
        _number(cold, "inlet_temperature_k", "cold_stream."),
        _number(cold, "outlet_temperature_k", "cold_stream."),
        _fluid(tube_stream, tube_label),
        _fluid(shell_stream, shell_label),
        _number(tube_stream, "mass_flow_kg_s", tube_label),
        _number(shell_stream, "mass_flow_kg_s", shell_label),
        _number(data, "tube_length_m", ""),
        _number(data, "tube_inner_diameter_m", ""),
        _number(data, "wall_conductivity_w_m_k", "", 16.0),
        _number(data, "fouling_inner_m2_k_w", "", 0.0002),
        _number(data, "fouling_outer_m2_k_w", "", 0.0001),
        _optional_number(tube_stream, "wall_viscosity_pa_s", tube_label),
        _optional_number(shell_stream, "wall_viscosity_pa_s", shell_label),
        hot_on_shell_side=hot_side == "shell",
    )
    g = "geometry."
    geometry = BundleGeometry(
        _number(geometry_data, "shell_inner_diameter_m", g),
        _number(geometry_data, "tube_outer_diameter_m", g),
        _number(geometry_data, "tube_pitch_m", g),
        _integer(geometry_data, "tube_count", g),
        _number(geometry_data, "baffle_cut_fraction", g, 0.25),
        _number(geometry_data, "baffle_spacing_m", g),
        _text(geometry_data, "tube_layout", g, "triangular", {"triangular", "square"}),
        _number(geometry_data, "tube_to_baffle_clearance_m", g, 0.0008),
        _number(geometry_data, "shell_to_baffle_clearance_m", g, 0.003),
        None,
        None,
        _integer(geometry_data, "seal_strips", g, 0),
    )
    c = "constraints."
    defaults = DesignConstraints()
    constraints = DesignConstraints(
        _number(constraints_data, "max_tube_pressure_drop_pa", c, defaults.max_tube_pressure_drop_pa),
        _number(constraints_data, "max_shell_pressure_drop_pa", c, defaults.max_shell_pressure_drop_pa),
        _number(constraints_data, "minimum_duty_fraction", c, defaults.minimum_duty_fraction),
        _text(constraints_data, "objective", c, "area", {"area", "pressure_drop", "cost"}),
        _integer(constraints_data, "max_iterations", c, defaults.max_iterations),
        _number(constraints_data, "max_shell_diameter_m", c, defaults.max_shell_diameter_m),
    )
    return case, geometry, _integer(data, "tube_passes", "", 2), constraints


def main(argv: list[str] | None = None) -> int:
    """Run a design from JSON and write text and Excel reports."""
    parser = argparse.ArgumentParser(description="Preliminary shell-and-tube heat-exchanger design")
    parser.add_argument("input", help="JSON input file")
    parser.add_argument("--text-report", default="report.txt")
    parser.add_argument("--excel-report", default="report.xlsx")
    args = parser.parse_args(argv)
    try:
        case, geometry, passes, constraints = load_input(args.input)
        solution = solve_design(case, geometry, passes, constraints)
        generate_report(solution, case, args.text_report, args.excel_report)
    except (ValueError, RuntimeError, OSError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(f"Design complete: {args.text_report}, {args.excel_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
