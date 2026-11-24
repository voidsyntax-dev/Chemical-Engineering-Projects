"""Tube-side friction and pressure-drop calculations."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PressureDropResult:
    """Tube-side pressure-drop components in Pa."""

    total_pa: float
    friction_pa: float
    return_loss_pa: float
    nozzle_loss_pa: float
    velocity_m_s: float
    reynolds: float
    friction_factor: float


RE_LAMINAR_MAX = 2300.0  # laminar friction 64/Re up to here
RE_TURBULENT_MIN = 4000.0  # Colebrook--White from here; linear blend in between


def _colebrook_turbulent(
    reynolds: float,
    relative_roughness: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    f = 1 / (-1.8 * math.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds)) ** 2  # Haaland guess
    for _ in range(max_iterations):
        next_f = 1 / (-2 * math.log10(relative_roughness / 3.7 + 2.51 / (reynolds * math.sqrt(f)))) ** 2
        if abs(next_f - f) < tolerance:
            return next_f
        f = next_f
    raise RuntimeError("Colebrook solver did not converge within max_iterations")


def colebrook_white(
    reynolds: float,
    roughness_m: float,
    diameter_m: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Return the Darcy friction factor (not Fanning: f_D = 4 f_F).

    Reynolds number is dimensionless; roughness and diameter are m.
    ``Re <= 2300`` uses ``64/Re``; ``Re >= 4000`` solves Colebrook (1939) by
    fixed-point iteration from the Haaland explicit estimate (Moody, Trans.
    ASME 66, 1944).  Between 2300 and 4000 the factor is interpolated linearly
    in Re between the two limits, so f is continuous; that range is a
    screening estimate only.
    """
    if reynolds <= 0 or roughness_m < 0 or diameter_m <= 0:
        raise ValueError("reynolds and diameter must be positive; roughness cannot be negative")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    if reynolds <= RE_LAMINAR_MAX:
        return 64 / reynolds
    relative_roughness = roughness_m / diameter_m
    if reynolds >= RE_TURBULENT_MIN:
        return _colebrook_turbulent(reynolds, relative_roughness, tolerance, max_iterations)
    f_laminar = 64 / RE_LAMINAR_MAX
    f_turbulent = _colebrook_turbulent(RE_TURBULENT_MIN, relative_roughness, tolerance, max_iterations)
    weight = (reynolds - RE_LAMINAR_MAX) / (RE_TURBULENT_MIN - RE_LAMINAR_MAX)
    return f_laminar + weight * (f_turbulent - f_laminar)


def tube_side_pressure_drop(
    mass_flow_kg_s: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    tube_length_m: float,
    tube_inner_diameter_m: float,
    number_of_tubes: int,
    tube_passes: int = 1,
    roughness_m: float = 1.5e-6,
    return_loss_coefficient: float = 1.5,
    nozzle_loss_coefficient: float = 1.0,
) -> PressureDropResult:
    """Calculate tube friction, return, and nozzle pressure losses.

    Mass flow is kg/s, density kg/m³, viscosity Pa·s, lengths and roughness m,
    and the result components are Pa.  Tube count must be divisible by the
    number of passes.  Darcy--Weisbach friction is evaluated over the total
    tube path length; return-loss K is applied once per pass transition and
    nozzle K is applied at both nozzles.  This is a preliminary mechanical
    estimate and fittings should be replaced by vendor data for final design.
    """
    for name, value in (("mass_flow_kg_s", mass_flow_kg_s), ("density_kg_m3", density_kg_m3), ("viscosity_pa_s", viscosity_pa_s), ("tube_length_m", tube_length_m), ("tube_inner_diameter_m", tube_inner_diameter_m)):
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    if number_of_tubes < 1 or tube_passes < 1 or number_of_tubes % tube_passes:
        raise ValueError("number_of_tubes must be divisible by positive tube_passes")
    if roughness_m < 0 or return_loss_coefficient < 0 or nozzle_loss_coefficient < 0:
        raise ValueError("roughness and loss coefficients cannot be negative")
    tubes_per_pass = number_of_tubes / tube_passes
    flow_area = tubes_per_pass * math.pi * tube_inner_diameter_m**2 / 4
    velocity = mass_flow_kg_s / (density_kg_m3 * flow_area)
    reynolds = density_kg_m3 * velocity * tube_inner_diameter_m / viscosity_pa_s
    f = colebrook_white(reynolds, roughness_m, tube_inner_diameter_m)
    dynamic_pressure = density_kg_m3 * velocity**2 / 2
    friction = f * (tube_length_m * tube_passes / tube_inner_diameter_m) * dynamic_pressure
    return_loss = max(0, tube_passes - 1) * return_loss_coefficient * dynamic_pressure
    nozzle_loss = 2 * nozzle_loss_coefficient * dynamic_pressure
    return PressureDropResult(friction + return_loss + nozzle_loss, friction, return_loss, nozzle_loss, velocity, reynolds, f)


def sieder_tate_correction(
    h_w_m2_k: float,
    bulk_viscosity_pa_s: float,
    wall_viscosity_pa_s: float,
) -> float:
    """Apply the Sieder--Tate viscosity ratio correction to h.

    Coefficient is W/(m²·K), viscosities are Pa·s, and the returned coefficient
    is W/(m²·K).  The correction is ``h (μ_b/μ_w)^0.14``.
    """
    if h_w_m2_k <= 0 or bulk_viscosity_pa_s <= 0 or wall_viscosity_pa_s <= 0:
        raise ValueError("coefficient and viscosities must be positive")
    return h_w_m2_k * (bulk_viscosity_pa_s / wall_viscosity_pa_s) ** 0.14
