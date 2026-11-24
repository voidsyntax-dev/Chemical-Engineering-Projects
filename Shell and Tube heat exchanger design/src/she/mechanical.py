"""Preliminary mechanical and vibration screening checks."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BaffleSpacingCheck:
    """TEMA spacing check and its allowable bounds in m."""

    passed: bool
    minimum_m: float
    maximum_m: float
    message: str


@dataclass(frozen=True)
class VibrationScreening:
    """Tube vibration screening result."""

    passed: bool
    crossflow_velocity_m_s: float
    natural_frequency_hz: float
    vortex_frequency_hz: float
    frequency_ratio: float
    message: str


def fixed_tubesheet_thickness(
    shell_inner_diameter_m: float,
    design_pressure_pa: float,
    allowable_stress_pa: float,
    weld_efficiency: float = 1.0,
    corrosion_allowance_m: float = 0.0,
) -> float:
    """Calculate fixed-tubesheet thickness using the TEMA screening equation.

    Diameter and corrosion allowance are m; pressure and allowable stress are
    Pa; weld efficiency is dimensionless; result is m.  The equation
    ``t=D sqrt(P/(S E - 0.6P))+CA`` follows the fixed-tubesheet form in TEMA
    Standards, 10th ed., RCB-7.2.2.  It is a screening calculation and does
    not replace a detailed tube-sheet stress analysis.
    """
    if shell_inner_diameter_m <= 0 or design_pressure_pa <= 0 or allowable_stress_pa <= 0:
        raise ValueError("diameter, pressure, and allowable stress must be positive")
    if not 0 < weld_efficiency <= 1 or corrosion_allowance_m < 0:
        raise ValueError("weld efficiency must be in (0, 1] and corrosion allowance non-negative")
    denominator = allowable_stress_pa * weld_efficiency - 0.6 * design_pressure_pa
    if denominator <= 0:
        raise ValueError("allowable stress is insufficient for design pressure")
    return shell_inner_diameter_m * math.sqrt(design_pressure_pa / denominator) + corrosion_allowance_m


def check_baffle_spacing(
    shell_inner_diameter_m: float,
    baffle_spacing_m: float,
    minimum_fraction: float = 0.2,
    maximum_fraction: float = 1.0,
    absolute_minimum_m: float = 0.0508,
) -> BaffleSpacingCheck:
    """Check TEMA baffle spacing limits.

    Shell diameter and spacing are m.  TEMA screening limits are at least 20 %
    of shell ID and 50.8 mm, and no more than one shell diameter; the returned
    bounds and message make these assumptions explicit.
    """
    if shell_inner_diameter_m <= 0 or baffle_spacing_m <= 0:
        raise ValueError("shell diameter and baffle spacing must be positive")
    if not 0 < minimum_fraction < maximum_fraction:
        raise ValueError("spacing fractions are invalid")
    minimum = max(absolute_minimum_m, minimum_fraction * shell_inner_diameter_m)
    maximum = maximum_fraction * shell_inner_diameter_m
    slack = 1e-9 * shell_inner_diameter_m  # absorb floating-point noise at the limits
    passed = minimum - slack <= baffle_spacing_m <= maximum + slack
    message = "pass" if passed else f"spacing must be between {minimum:.4g} and {maximum:.4g} m"
    return BaffleSpacingCheck(passed, minimum, maximum, message)


def screen_tube_vibration(
    crossflow_velocity_m_s: float,
    tube_span_m: float,
    tube_outer_diameter_m: float,
    tube_inner_diameter_m: float,
    youngs_modulus_pa: float = 2.0e11,
    tube_density_kg_m3: float = 7850.0,
    strouhal_number: float = 0.2,
) -> VibrationScreening:
    """Screen vortex-shedding resonance of a simply supported tube span.

    Velocity, span, and diameters are m/s and m; modulus is Pa; density is
    kg/m³; Strouhal number is dimensionless.  The natural frequency is the
    first simply-supported Euler--Bernoulli estimate.  Risk is flagged when
    vortex shedding lies within ±20 % of the natural frequency, following a
    conservative preliminary resonance screen; tube-support damping and fluid
    added mass require a final TEMA/FEA review.
    """
    for name, value in (("crossflow_velocity_m_s", crossflow_velocity_m_s), ("tube_span_m", tube_span_m), ("tube_outer_diameter_m", tube_outer_diameter_m), ("tube_inner_diameter_m", tube_inner_diameter_m), ("youngs_modulus_pa", youngs_modulus_pa), ("tube_density_kg_m3", tube_density_kg_m3), ("strouhal_number", strouhal_number)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if tube_inner_diameter_m >= tube_outer_diameter_m:
        raise ValueError("tube inner diameter must be less than outer diameter")
    second_moment = math.pi * (tube_outer_diameter_m**4 - tube_inner_diameter_m**4) / 64
    mass_per_length = tube_density_kg_m3 * math.pi * (tube_outer_diameter_m**2 - tube_inner_diameter_m**2) / 4
    natural = math.pi / (2 * tube_span_m**2) * math.sqrt(youngs_modulus_pa * second_moment / mass_per_length)
    vortex = strouhal_number * crossflow_velocity_m_s / tube_outer_diameter_m
    ratio = vortex / natural
    passed = not 0.8 <= ratio <= 1.2
    message = "pass" if passed else "resonance risk: vortex shedding is within ±20% of natural frequency"
    return VibrationScreening(passed, crossflow_velocity_m_s, natural, vortex, ratio, message)
