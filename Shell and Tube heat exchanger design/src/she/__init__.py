"""Preliminary shell-and-tube heat-exchanger design calculations."""

from .properties import FluidProperties, fluid_properties_at_temperature
from .thermal import (
    RequiredArea,
    correction_factor_1_2plus,
    heat_duty,
    kern_shell_side_h,
    lmtd_counterflow,
    overall_u,
    required_area_and_tubes,
    tube_side_h,
)
from .geometry import BundleGeometry, bundle_diameter_from_tube_count, tube_count_from_shell_id
from .bell_delaware import BellDelawareResult, bell_delaware_shell_side, ideal_crossflow_h
from .hydraulics import PressureDropResult, colebrook_white, sieder_tate_correction, tube_side_pressure_drop
from .rating import DesignConstraints, DesignSolution, RatingInput, RatingResult, constraint_cost, rate_exchanger, solve_design
from .mechanical import BaffleSpacingCheck, VibrationScreening, check_baffle_spacing, fixed_tubesheet_thickness, screen_tube_vibration
from .cost import exchanger_cost_proxy
from .reporting import generate_report

__all__ = [
    "FluidProperties",
    "fluid_properties_at_temperature",
    "RequiredArea",
    "heat_duty",
    "lmtd_counterflow",
    "correction_factor_1_2plus",
    "tube_side_h",
    "kern_shell_side_h",
    "overall_u",
    "required_area_and_tubes",
    "BundleGeometry",
    "bundle_diameter_from_tube_count",
    "tube_count_from_shell_id",
    "BellDelawareResult",
    "ideal_crossflow_h",
    "bell_delaware_shell_side",
    "PressureDropResult",
    "colebrook_white",
    "sieder_tate_correction",
    "tube_side_pressure_drop",
    "RatingInput",
    "RatingResult",
    "DesignConstraints",
    "DesignSolution",
    "rate_exchanger",
    "solve_design",
    "constraint_cost",
    "BaffleSpacingCheck",
    "VibrationScreening",
    "fixed_tubesheet_thickness",
    "check_baffle_spacing",
    "screen_tube_vibration",
    "exchanger_cost_proxy",
    "generate_report",
]
