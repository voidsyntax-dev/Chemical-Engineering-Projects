"""Thermal design correlations used in the preliminary design stages."""

from dataclasses import dataclass
import math


# Tube-side regime limits and constants (Incropera, Fundamentals of Heat and Mass Transfer).
RE_LAMINAR_MAX = 2300.0
RE_TURBULENT_MIN = 10_000.0
NU_LAMINAR_CONSTANT_WALL_T = 3.66
_DB_EXPONENT_HEATING = 0.4
_DB_EXPONENT_COOLING = 0.3
SIEDER_TATE_EXPONENT = 0.14  # Sieder & Tate (1936), (mu_bulk/mu_wall)^0.14
_R_UNITY_TOLERANCE = 1e-8  # |R-1| below which the R=1 limit of the F equation is used


@dataclass(frozen=True)
class RequiredArea:
    """Required outside tube area and corresponding integer tube count."""

    area_m2: float
    number_of_tubes: int

    def __iter__(self):
        yield self.area_m2
        yield self.number_of_tubes


def _positive(name: str, value: float) -> None:
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite positive number")


def heat_duty(
    mass_flow_kg_s: float,
    specific_heat_j_kg_k: float,
    delta_temperature_k: float,
) -> float:
    """Return heat duty magnitude from mass flow and temperature change.

    Mass flow is kg/s, heat capacity is J/(kg·K), and temperature change is K.
    The returned duty is W and is the magnitude; sign convention belongs to
    the stream energy balance.
    """
    _positive("mass_flow_kg_s", mass_flow_kg_s)
    _positive("specific_heat_j_kg_k", specific_heat_j_kg_k)
    if not math.isfinite(delta_temperature_k) or delta_temperature_k == 0:
        raise ValueError("delta_temperature_k must be finite and non-zero")
    return abs(mass_flow_kg_s * specific_heat_j_kg_k * delta_temperature_k)


def lmtd_counterflow(
    hot_inlet_k: float,
    hot_outlet_k: float,
    cold_inlet_k: float,
    cold_outlet_k: float,
) -> float:
    """Calculate counter-current log-mean temperature difference.

    All temperatures are in K and the result is in K.  The terminal
    differences are ``hot_inlet-cold_outlet`` and ``hot_outlet-cold_inlet``;
    both must be positive.  Equal terminal differences use their common value
    instead of evaluating the removable 0/0 logarithmic expression.
    """
    d1 = hot_inlet_k - cold_outlet_k
    d2 = hot_outlet_k - cold_inlet_k
    if not all(math.isfinite(value) for value in (d1, d2)) or d1 <= 0 or d2 <= 0:
        raise ValueError("counterflow terminal temperature differences must be positive")
    if math.isclose(d1, d2, rel_tol=1e-12, abs_tol=1e-12):
        return float((d1 + d2) / 2)
    return float((d1 - d2) / math.log(d1 / d2))


def correction_factor_1_2plus(
    P: float,
    R: float,
    shell_passes: int = 1,
) -> float:
    """Return the LMTD correction factor F for one shell pass, even tube passes.

    Definitions (dimensionless, in terms of the TUBE-SIDE fluid ``t`` and the
    SHELL-SIDE fluid ``T``; F is the same if the roles are swapped)::

        P = (t_out - t_in) / (T_in - t_in)     thermal effectiveness, 0 <= P < P_max
        R = (T_in - T_out) / (t_out - t_in)    capacity-rate ratio,  R >= 0

    The equation is the Bowman, Mueller & Nagle (1940) / Underwood closed form
    (Kern, *Process Heat Transfer*, 1950)::

        F = sqrt(R^2+1)/(R-1) * ln[(1-P)/(1-P R)]
            / ln{[2 - P(R+1-s)] / [2 - P(R+1+s)]},   s = sqrt(R^2+1)

    At R = 1 the first factor is the limit of ``ln[(1-P)/(1-PR)]/(R-1)``, which
    is ``P/(1-P)``, so::

        F(P, R=1) = sqrt(2) P/(1-P) / ln{[2 - P(2-sqrt2)] / [2 - P(2+sqrt2)]}

    P = 0 returns 1.  A temperature cross is impossible in one shell pass, so
    ``P >= P_max = 2/(R+1+s)`` raises ``ValueError`` (e.g. P_max = 0.586 at
    R = 1); in particular P = R = 1 is infeasible and is rejected.
    """
    if shell_passes != 1:
        raise ValueError("this closed form supports exactly one shell pass")
    if not all(math.isfinite(value) for value in (P, R)) or not (0 <= P < 1 and R >= 0):
        raise ValueError("P must be in [0, 1) and R must be non-negative")
    if P == 0.0:
        return 1.0
    s = math.sqrt(R * R + 1.0)
    p_max = 2.0 / (R + 1.0 + s)
    if P >= p_max:
        raise ValueError(
            f"P={P:.4g} exceeds the maximum {p_max:.4g} achievable with one shell pass at R={R:.4g}"
            " (temperature cross); use more shell passes"
        )
    denominator_log = math.log((2 - P * (R + 1 - s)) / (2 - P * (R + 1 + s)))
    if abs(R - 1.0) < _R_UNITY_TOLERANCE:
        factor = math.sqrt(2.0) * P / (1.0 - P) / denominator_log
    else:
        factor = (s / (R - 1.0)) * math.log((1 - P) / (1 - P * R)) / denominator_log
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("P and R produce an invalid correction factor")
    return min(1.0, float(factor))


def effectiveness_ntu(ntu: float, capacity_ratio: float, tube_passes: int = 2) -> float:
    """Return heat-exchanger effectiveness for given NTU and Cr = Cmin/Cmax.

    ``tube_passes == 1`` is treated as pure counter-current flow; two or more
    (even) tube passes use the one-shell-pass equation, Incropera Table 11.3
    / Kays & London.  NTU = UA/Cmin is dimensionless.
    """
    if ntu < 0 or not 0 <= capacity_ratio <= 1 or not math.isfinite(ntu):
        raise ValueError("ntu must be non-negative and capacity_ratio in [0, 1]")
    if ntu == 0:
        return 0.0
    if tube_passes == 1:
        if abs(1.0 - capacity_ratio) < 1e-12:
            return ntu / (1.0 + ntu)
        e = math.exp(-ntu * (1.0 - capacity_ratio))
        return (1.0 - e) / (1.0 - capacity_ratio * e)
    q = math.sqrt(1.0 + capacity_ratio**2)
    e = math.exp(-ntu * q)
    return 2.0 / (1.0 + capacity_ratio + q * (1.0 + e) / (1.0 - e))


def tube_side_nusselt(reynolds: float, prandtl: float, heating: bool = True) -> float:
    """Return tube-side Nusselt number, continuous across the flow regimes.

    * Re <= 2300 (laminar): constant 3.66 (fully developed, constant wall
      temperature; entry effects ignored, which is conservative).
    * Re >= 10000 (turbulent): Dittus--Boelter ``0.023 Re^0.8 Pr^n`` with
      n = 0.4 when the tube fluid is HEATED and n = 0.3 when it is COOLED.
    * 2300 < Re < 10000 (transitional): linear interpolation in Re between the
      two limits.  This is a screening estimate that keeps h continuous; it is
      not a validated transition correlation.
    """
    if reynolds <= 0 or prandtl <= 0:
        raise ValueError("reynolds and prandtl must be positive")
    exponent = _DB_EXPONENT_HEATING if heating else _DB_EXPONENT_COOLING
    if reynolds <= RE_LAMINAR_MAX:
        return NU_LAMINAR_CONSTANT_WALL_T
    if reynolds >= RE_TURBULENT_MIN:
        return 0.023 * reynolds**0.8 * prandtl**exponent
    nu_turbulent = 0.023 * RE_TURBULENT_MIN**0.8 * prandtl**exponent
    weight = (reynolds - RE_LAMINAR_MAX) / (RE_TURBULENT_MIN - RE_LAMINAR_MAX)
    return NU_LAMINAR_CONSTANT_WALL_T + weight * (nu_turbulent - NU_LAMINAR_CONSTANT_WALL_T)


def tube_flow_regime(reynolds: float) -> str:
    """Return 'laminar', 'transitional' or 'turbulent' for tube-side Reynolds number."""
    if reynolds <= RE_LAMINAR_MAX:
        return "laminar"
    return "turbulent" if reynolds >= RE_TURBULENT_MIN else "transitional"


def tube_side_h(
    mass_flow_kg_s: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    conductivity_w_m_k: float,
    specific_heat_j_kg_k: float,
    tube_inner_diameter_m: float,
    number_of_tubes: int,
    tube_passes: int = 1,
    heating: bool = True,
    wall_viscosity_pa_s: float | None = None,
) -> float:
    """Calculate tube-side film coefficient in W/(m²·K).

    Flow and geometry are SI: kg/s, kg/m³, Pa·s, W/(m·K), J/(kg·K), and m.
    ``heating`` is True when the tube fluid is being heated (exponent 0.4) and
    False when it is being cooled (0.3); see :func:`tube_side_nusselt` for the
    regime treatment.  If ``wall_viscosity_pa_s`` (viscosity at the wall
    temperature, Pa·s) is given, the Sieder--Tate factor
    ``(mu_bulk/mu_wall)^0.14`` multiplies h in every regime.
    """
    for name, value in (
        ("mass_flow_kg_s", mass_flow_kg_s),
        ("density_kg_m3", density_kg_m3),
        ("viscosity_pa_s", viscosity_pa_s),
        ("conductivity_w_m_k", conductivity_w_m_k),
        ("specific_heat_j_kg_k", specific_heat_j_kg_k),
        ("tube_inner_diameter_m", tube_inner_diameter_m),
    ):
        _positive(name, value)
    if number_of_tubes < 1 or tube_passes < 1 or number_of_tubes % tube_passes:
        raise ValueError("number_of_tubes must be positive and divisible by tube_passes")
    area = number_of_tubes / tube_passes * math.pi * tube_inner_diameter_m**2 / 4
    velocity = mass_flow_kg_s / (density_kg_m3 * area)
    reynolds = density_kg_m3 * velocity * tube_inner_diameter_m / viscosity_pa_s
    prandtl = specific_heat_j_kg_k * viscosity_pa_s / conductivity_w_m_k
    nusselt = tube_side_nusselt(reynolds, prandtl, heating)
    if wall_viscosity_pa_s is not None:
        _positive("wall_viscosity_pa_s", wall_viscosity_pa_s)
        nusselt *= (viscosity_pa_s / wall_viscosity_pa_s) ** SIEDER_TATE_EXPONENT
    return nusselt * conductivity_w_m_k / tube_inner_diameter_m


def kern_shell_side_h(
    mass_flow_kg_s: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    conductivity_w_m_k: float,
    specific_heat_j_kg_k: float,
    shell_inner_diameter_m: float,
    tube_outer_diameter_m: float,
    tube_pitch_m: float,
    baffle_spacing_m: float,
    tube_layout: str = "triangular",
) -> float:
    """Estimate shell-side coefficient using the Kern method, W/(m²·K).

    Inputs use kg/s, kg/m³, Pa·s, W/(m·K), J/(kg·K), and m.  The ideal
    cross-flow area is ``shell diameter × baffle spacing × open pitch ratio``
    and the equivalent diameter is the standard Kern pitch-cell hydraulic
    diameter.  The correlation is from Kern, *Process Heat Transfer*,
    McGraw-Hill (1950), and is a preliminary estimate.
    """
    for name, value in (
        ("mass_flow_kg_s", mass_flow_kg_s),
        ("density_kg_m3", density_kg_m3),
        ("viscosity_pa_s", viscosity_pa_s),
        ("conductivity_w_m_k", conductivity_w_m_k),
        ("specific_heat_j_kg_k", specific_heat_j_kg_k),
        ("shell_inner_diameter_m", shell_inner_diameter_m),
        ("tube_outer_diameter_m", tube_outer_diameter_m),
        ("tube_pitch_m", tube_pitch_m),
        ("baffle_spacing_m", baffle_spacing_m),
    ):
        _positive(name, value)
    if tube_pitch_m <= tube_outer_diameter_m:
        raise ValueError("tube_pitch_m must exceed tube_outer_diameter_m")
    if tube_layout not in {"triangular", "square"}:
        raise ValueError("tube_layout must be 'triangular' or 'square'")
    if tube_layout == "triangular":
        cell_area = math.sqrt(3) * tube_pitch_m**2 / 2
        open_ratio = 1 - tube_outer_diameter_m / tube_pitch_m
    else:
        cell_area = tube_pitch_m**2
        open_ratio = 1 - tube_outer_diameter_m / tube_pitch_m
    flow_area = shell_inner_diameter_m * baffle_spacing_m * open_ratio
    equivalent_diameter = 4 * (cell_area - math.pi * tube_outer_diameter_m**2 / 4) / (math.pi * tube_outer_diameter_m)
    if flow_area <= 0 or equivalent_diameter <= 0:
        raise ValueError("geometry gives no positive shell cross-flow area")
    velocity = mass_flow_kg_s / (density_kg_m3 * flow_area)
    reynolds = density_kg_m3 * velocity * equivalent_diameter / viscosity_pa_s
    prandtl = specific_heat_j_kg_k * viscosity_pa_s / conductivity_w_m_k
    if reynolds <= 0:
        raise ValueError("shell Reynolds number must be positive")
    return 0.36 * conductivity_w_m_k / equivalent_diameter * reynolds**0.55 * prandtl ** (1 / 3)


def overall_u(
    tube_side_h_w_m2_k: float,
    shell_side_h_w_m2_k: float,
    tube_inner_diameter_m: float,
    tube_outer_diameter_m: float,
    wall_conductivity_w_m_k: float,
    fouling_inner_m2_k_w: float = 0.0,
    fouling_outer_m2_k_w: float = 0.0,
) -> float:
    """Return clean/dirty outside-area overall U in W/(m²·K).

    Film coefficients are W/(m²·K), diameters and wall conductivity are m and
    W/(m·K), and fouling resistances are m²·K/W on their respective surfaces.
    Cylindrical wall and area-ratio effects are included.
    """
    for name, value in (
        ("tube_side_h_w_m2_k", tube_side_h_w_m2_k),
        ("shell_side_h_w_m2_k", shell_side_h_w_m2_k),
        ("tube_inner_diameter_m", tube_inner_diameter_m),
        ("tube_outer_diameter_m", tube_outer_diameter_m),
        ("wall_conductivity_w_m_k", wall_conductivity_w_m_k),
    ):
        _positive(name, value)
    if tube_outer_diameter_m < tube_inner_diameter_m:
        raise ValueError("tube_outer_diameter_m must be at least tube_inner_diameter_m")
    if fouling_inner_m2_k_w < 0 or fouling_outer_m2_k_w < 0:
        raise ValueError("fouling resistances cannot be negative")
    resistance = (
        1 / shell_side_h_w_m2_k
        + fouling_outer_m2_k_w
        + tube_outer_diameter_m / tube_inner_diameter_m * (1 / tube_side_h_w_m2_k + fouling_inner_m2_k_w)
        + tube_outer_diameter_m * math.log(tube_outer_diameter_m / tube_inner_diameter_m) / (2 * wall_conductivity_w_m_k)
    )
    return 1 / resistance


def required_area_and_tubes(
    duty_w: float,
    overall_u_w_m2_k: float,
    lmtd_k: float,
    tube_length_m: float,
    tube_outer_diameter_m: float,
    correction_factor: float = 1.0,
) -> RequiredArea:
    """Solve outside heat-transfer area and integer tube count.

    Duty is W, U is W/(m²·K), LMTD and tube length are K and m, and the tube
    outside diameter is m.  ``correction_factor`` is dimensionless.  Area is
    based on outside tube surface and tube count is rounded up.
    """
    for name, value in (
        ("duty_w", duty_w),
        ("overall_u_w_m2_k", overall_u_w_m2_k),
        ("lmtd_k", lmtd_k),
        ("tube_length_m", tube_length_m),
        ("tube_outer_diameter_m", tube_outer_diameter_m),
        ("correction_factor", correction_factor),
    ):
        _positive(name, value)
    area = duty_w / (overall_u_w_m2_k * correction_factor * lmtd_k)
    tube_area = math.pi * tube_outer_diameter_m * tube_length_m
    return RequiredArea(area, math.ceil(area / tube_area))
