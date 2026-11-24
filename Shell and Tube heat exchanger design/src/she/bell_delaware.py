"""Bell--Delaware shell-side heat-transfer and pressure-drop method.

Equations follow Taborek (Heat Exchanger Design Handbook, 3.3.5) as presented
in Serth, *Process Heat Transfer* (Ch. 5/6).  The book itself was not
available while writing this module: the equation structure was cross-checked
against Saif & Tariq, J. Mech. Eng. Sci. 19(4), 2025, and the coefficient
tables against the pychemqt implementation of Serth's method.  Coefficients
flagged ``# CHECK`` differ between secondary sources and should be confirmed
against the book.  Only 30 degree (triangular) and 90 degree (square) layouts
are supported.  Units are SI (m, m², kg/s, Pa, W/(m²·K)); clearances are
DIAMETRAL.  The viscosity-ratio (wall) correction is applied by the caller.
"""

from dataclasses import dataclass
import math

from .geometry import BundleGeometry
from .properties import FluidProperties

RE_LAMINAR_LIMIT = 100.0  # Bell--Delaware laminar/turbulent switch for J and R factors

# (upper Re bound, a1, a2, b1, b2) per layout; a3,a4,b3,b4 are per layout.
_COEFFICIENTS = {
    "triangular": (
        ((10.0, 1.40, -0.667, 48.0, -1.0),
         (100.0, 1.36, -0.657, 45.1, -0.973),
         (1000.0, 0.593, -0.477, 4.57, -0.476),
         (10000.0, 0.321, -0.388, 0.486, -0.152),
         (math.inf, 0.321, -0.388, 0.372, -0.123)),  # CHECK b2: sources give -0.120 to -0.126
        (1.45, 0.519, 7.0, 0.5),
    ),
    "square": (
        ((10.0, 0.970, -0.667, 35.0, -1.0),
         (100.0, 0.900, -0.631, 32.1, -0.963),
         (1000.0, 0.408, -0.460, 6.09, -0.602),
         (10000.0, 0.107, -0.266, 0.0815, 0.022),
         (math.inf, 0.370, -0.395, 0.391, -0.148)),
        (1.187, 0.370, 6.30, 0.378),
    ),
}


@dataclass(frozen=True)
class ShellGeometry:
    """Derived Bell--Delaware geometry (m, m²; angles in rad)."""

    shell_diameter_m: float
    tube_outer_diameter_m: float
    tube_pitch_m: float
    layout: str
    tube_count: int
    baffle_cut_fraction: float
    baffle_spacing_m: float
    inlet_spacing_m: float
    outlet_spacing_m: float
    number_of_baffles: int
    seal_strip_pairs: int
    otl_diameter_m: float
    theta_ds: float
    theta_ctl: float
    fw: float
    fc: float
    crossflow_area_m2: float  # S_m
    window_area_m2: float  # S_w
    window_diameter_m: float  # D_w
    rows_crossflow: float  # N_c (between baffle tips)
    rows_window: float  # N_cw
    shell_baffle_leak_m2: float  # S_sb
    tube_baffle_leak_m2: float  # S_tb
    bypass_area_m2: float  # S_b


@dataclass(frozen=True)
class BellDelawareResult:
    """Bell--Delaware result, with coefficients in SI units."""

    ideal_h_w_m2_k: float
    corrected_h_w_m2_k: float
    pressure_drop_pa: float
    factors: dict[str, float]  # Jc, Jl, Jb, Js, Jr
    pressure_factors: dict[str, float]  # Rl, Rb, Rs
    pressure_components_pa: dict[str, float]  # crossflow, window, end_zones
    crossflow_area_m2: float
    reynolds: float
    number_of_baffles: int


def shell_geometry(
    geometry: BundleGeometry,
    tube_length_m: float | None = None,
    number_of_baffles: int | None = None,
) -> ShellGeometry:
    """Build the Bell--Delaware geometry; baffle count comes from tube length and spacing.

    ``N_b = round((L - L_bi - L_bo)/B) + 1`` for straight tube length ``L``;
    pass ``number_of_baffles`` instead to override.
    """
    ds = geometry.shell_inner_diameter_m
    do = geometry.tube_outer_diameter_m
    p = geometry.tube_pitch_m
    bc = geometry.baffle_cut_fraction
    spacing = geometry.baffle_spacing_m
    inlet = geometry.inlet_baffle_spacing_m or spacing
    outlet = geometry.outlet_baffle_spacing_m or spacing
    if inlet <= 0 or outlet <= 0:
        raise ValueError("inlet and outlet baffle spacing must be positive")
    if not geometry.bundle_fits():
        raise ValueError(
            f"tube bundle (outer tube limit {geometry.bundle_diameter_m:.3f} m) does not fit "
            f"shell ID {ds:.3f} m with the minimum clearance"
        )
    if number_of_baffles is None:
        if tube_length_m is None or tube_length_m <= 0:
            raise ValueError("tube_length_m (or number_of_baffles) is required to count baffles")
        number_of_baffles = round((tube_length_m - inlet - outlet) / spacing) + 1
    if number_of_baffles < 1:
        raise ValueError("tube length is too short for even one baffle at this spacing")
    otl = geometry.bundle_diameter_m
    ctl = otl - do
    theta_ds = 2 * math.acos(1 - 2 * bc)
    cos_arg = min(1.0, ds * (1 - 2 * bc) / ctl)
    theta_ctl = 2 * math.acos(cos_arg)
    fw = (theta_ctl - math.sin(theta_ctl)) / (2 * math.pi)
    fc = 1 - 2 * fw
    pp = 0.866 * p if geometry.tube_layout == "triangular" else p  # pitch parallel to flow
    sm = spacing * (ds - otl + (ctl) * (p - do) / p)
    swg = ds**2 / 8 * (theta_ds - math.sin(theta_ds))
    swt = geometry.tube_count * fw * math.pi * do**2 / 4
    sw = swg - swt
    if sm <= 0 or sw <= 0:
        raise ValueError("geometry gives non-positive crossflow or window area")
    wetted = math.pi * do * geometry.tube_count * fw + ds * theta_ds / 2
    ssb = ds * geometry.shell_to_baffle_clearance_m * (math.pi - theta_ds / 2) / 2
    stb = math.pi / 4 * ((do + geometry.tube_to_baffle_clearance_m) ** 2 - do**2) * geometry.tube_count * (1 - fw)
    return ShellGeometry(
        ds, do, p, geometry.tube_layout, geometry.tube_count, bc, spacing, inlet, outlet,
        number_of_baffles, geometry.seal_strips, otl, theta_ds, theta_ctl, fw, fc, sm, sw,
        4 * sw / wetted, ds * (1 - 2 * bc) / pp, 0.8 * bc * ds / pp, ssb, stb, spacing * (ds - otl),
    )


def _coefficients(layout: str, reynolds: float) -> tuple[float, float, float, float, float, float, float, float]:
    rows, (a3, a4, b3, b4) = _COEFFICIENTS[layout]
    for upper, a1, a2, b1, b2 in rows:
        if reynolds < upper:
            return a1, a2, a3, a4, b1, b2, b3, b4
    raise AssertionError("unreachable")


def ideal_j_and_f(layout: str, reynolds: float, pitch_ratio: float) -> tuple[float, float]:
    """Return ideal-tube-bank Colburn j and friction factor f (Taborek).

    ``j = a1 (1.33/(p/d))^a Re^a2``, ``a = a3/(1+0.14 Re^a4)`` and likewise
    ``f = b1 (1.33/(p/d))^b Re^b2`` with ``b = b3/(1+0.14 Re^b4)``.  Re is based
    on tube OD and the crossflow mass velocity.
    """
    if reynolds <= 0 or pitch_ratio <= 1:
        raise ValueError("reynolds must be positive and pitch ratio above 1")
    a1, a2, a3, a4, b1, b2, b3, b4 = _coefficients(layout, reynolds)
    a = a3 / (1 + 0.14 * reynolds**a4)
    b = b3 / (1 + 0.14 * reynolds**b4)
    ratio = 1.33 / pitch_ratio
    return a1 * ratio**a * reynolds**a2, b1 * ratio**b * reynolds**b2


def ideal_crossflow_h(
    mass_flow_kg_s: float,
    fluid: FluidProperties,
    geometry: BundleGeometry,
    tube_length_m: float | None = None,
) -> tuple[float, float, float]:
    """Return ideal-bank h in W/(m²·K), Reynolds number (tube OD) and S_m in m²."""
    if mass_flow_kg_s <= 0:
        raise ValueError("mass_flow_kg_s must be positive")
    if min(fluid.density_kg_m3, fluid.viscosity_pa_s, fluid.conductivity_w_m_k, fluid.specific_heat_j_kg_k) <= 0:
        raise ValueError("fluid properties must be positive")
    sg = shell_geometry(geometry, tube_length_m, number_of_baffles=1 if tube_length_m is None else None)
    return _ideal_h(mass_flow_kg_s, fluid, sg)[:3]


def _ideal_h(mass_flow_kg_s: float, fluid: FluidProperties, sg: ShellGeometry) -> tuple[float, float, float, float, float]:
    g = mass_flow_kg_s / sg.crossflow_area_m2
    re = g * sg.tube_outer_diameter_m / fluid.viscosity_pa_s
    pr = fluid.specific_heat_j_kg_k * fluid.viscosity_pa_s / fluid.conductivity_w_m_k
    j, f = ideal_j_and_f(sg.layout, re, sg.tube_pitch_m / sg.tube_outer_diameter_m)
    return j * fluid.specific_heat_j_kg_k * g * pr ** (-2 / 3), re, sg.crossflow_area_m2, f, g


# ---- Correction factors; each takes plain numbers so it can be tested alone ----

def j_c_baffle_cut(fc: float) -> float:
    """Jc = 0.55 + 0.72 Fc, Fc = fraction of tubes in pure crossflow."""
    if not 0 <= fc <= 1:
        raise ValueError("fc must be in [0, 1]")
    return 0.55 + 0.72 * fc


def j_l_leakage(ssb: float, stb: float, sm: float) -> float:
    """Jl = 0.44(1-rs) + [1-0.44(1-rs)] exp(-2.2 rlm); rs = Ssb/(Ssb+Stb), rlm = (Ssb+Stb)/Sm."""
    if ssb < 0 or stb < 0 or sm <= 0:
        raise ValueError("leakage areas cannot be negative and Sm must be positive")
    if ssb + stb == 0:
        return 1.0
    rs = ssb / (ssb + stb)
    rlm = (ssb + stb) / sm
    return 0.44 * (1 - rs) + (1 - 0.44 * (1 - rs)) * math.exp(-2.2 * rlm)


def j_b_bundle_bypass(sb: float, sm: float, seal_pairs: int, rows_crossflow: float, reynolds: float) -> float:
    """Jb = exp[-Cbh (Sb/Sm) (1-(2 Nss/Nc)^(1/3))], 1 if Nss/Nc >= 0.5; Cbh = 1.35 (Re<=100) else 1.25."""
    if sb < 0 or sm <= 0 or seal_pairs < 0 or rows_crossflow <= 0:
        raise ValueError("invalid bypass inputs")
    rss = seal_pairs / rows_crossflow
    if rss >= 0.5:
        return 1.0
    cbh = 1.35 if reynolds <= RE_LAMINAR_LIMIT else 1.25
    return math.exp(-cbh * (sb / sm) * (1 - (2 * rss) ** (1 / 3)))


def j_s_unequal_spacing(number_of_baffles: int, inlet_ratio: float, outlet_ratio: float, reynolds: float) -> float:
    """Js = [(Nb-1) + Li*^(1-n) + Lo*^(1-n)] / [(Nb-1) + Li* + Lo*]; n = 0.6 (Re>100) else 1/3."""
    if number_of_baffles < 1 or inlet_ratio <= 0 or outlet_ratio <= 0:
        raise ValueError("invalid spacing inputs")
    n = 0.6 if reynolds > RE_LAMINAR_LIMIT else 1 / 3
    return ((number_of_baffles - 1) + inlet_ratio ** (1 - n) + outlet_ratio ** (1 - n)) / (
        (number_of_baffles - 1) + inlet_ratio + outlet_ratio
    )


def j_r_laminar(reynolds: float, total_rows_crossed: float) -> float:
    """Jr: 1 for Re >= 100; (10/Nc_total)^0.18 for Re <= 20; linear in between (capped at 1)."""
    if reynolds <= 0 or total_rows_crossed <= 0:
        raise ValueError("reynolds and rows must be positive")
    if reynolds >= RE_LAMINAR_LIMIT:
        return 1.0
    jr_low = min(1.0, (10 / total_rows_crossed) ** 0.18)
    if reynolds <= 20:
        return jr_low
    return jr_low + (1.0 - jr_low) * (reynolds - 20) / (RE_LAMINAR_LIMIT - 20)


def r_l_leakage(ssb: float, stb: float, sm: float) -> float:
    """Rl = exp[-1.33 (1+rs) rlm^p], p = -0.15 (1+rs) + 0.8."""
    if ssb + stb == 0:
        return 1.0
    rs = ssb / (ssb + stb)
    rlm = (ssb + stb) / sm
    return math.exp(-1.33 * (1 + rs) * rlm ** (-0.15 * (1 + rs) + 0.8))


def r_b_bypass(sb: float, sm: float, seal_pairs: int, rows_crossflow: float, reynolds: float) -> float:
    """Rb = exp[-Cbp (Sb/Sm) (1-(2 Nss/Nc)^(1/3))]; Cbp = 4.5 (Re<=100) else 3.7."""
    rss = seal_pairs / rows_crossflow
    if rss >= 0.5:
        return 1.0
    cbp = 4.5 if reynolds <= RE_LAMINAR_LIMIT else 3.7
    return math.exp(-cbp * (sb / sm) * (1 - (2 * rss) ** (1 / 3)))


def r_s_unequal_spacing(inlet_ratio: float, outlet_ratio: float, reynolds: float) -> float:
    """Rs = 0.5[(B/Lbi)^(2-n) + (B/Lbo)^(2-n)]; n = 0.2 (Re>100) else 1; ratios are Lbi/B, Lbo/B."""
    n = 0.2 if reynolds > RE_LAMINAR_LIMIT else 1.0
    return 0.5 * (inlet_ratio ** (n - 2) + outlet_ratio ** (n - 2))


def bell_delaware_shell_side(
    mass_flow_kg_s: float,
    fluid: FluidProperties,
    geometry: BundleGeometry,
    tube_length_m: float | None = None,
    number_of_baffles: int | None = None,
) -> BellDelawareResult:
    """Calculate ideal/corrected shell h and total shell pressure drop.

    Pressure drop is ``dP_crossflow + dP_window + dP_end`` with
    ``dP_c = (Nb-1) dP_i Rl Rb``, ``dP_w = Nb dP_wi Rl`` and
    ``dP_e = 2 dP_i (1+Ncw/Nc) Rb Rs`` where ``dP_i = 2 f Nc G²/rho`` (f from
    :func:`ideal_j_and_f`).  The window term uses ``(2+0.6 Ncw) m²/(2 rho Sm Sw)``
    for Re > 100 and the laminar form of Serth for Re <= 100 (CHECK: the
    laminar window term was transcribed from a secondary source).
    ``tube_length_m`` (or ``number_of_baffles``) is required.
    """
    if mass_flow_kg_s <= 0:
        raise ValueError("mass_flow_kg_s must be positive")
    if min(fluid.density_kg_m3, fluid.viscosity_pa_s, fluid.conductivity_w_m_k, fluid.specific_heat_j_kg_k) <= 0:
        raise ValueError("fluid properties must be positive")
    sg = shell_geometry(geometry, tube_length_m, number_of_baffles)
    ideal_h, re, sm, f, g = _ideal_h(mass_flow_kg_s, fluid, sg)
    total_rows = (sg.rows_crossflow + sg.rows_window) * (sg.number_of_baffles + 1)
    li, lo = sg.inlet_spacing_m / sg.baffle_spacing_m, sg.outlet_spacing_m / sg.baffle_spacing_m
    factors = {
        "Jc": j_c_baffle_cut(sg.fc),
        "Jl": j_l_leakage(sg.shell_baffle_leak_m2, sg.tube_baffle_leak_m2, sm),
        "Jb": j_b_bundle_bypass(sg.bypass_area_m2, sm, sg.seal_strip_pairs, sg.rows_crossflow, re),
        "Js": j_s_unequal_spacing(sg.number_of_baffles, li, lo, re),
        "Jr": j_r_laminar(re, total_rows),
    }
    rl = r_l_leakage(sg.shell_baffle_leak_m2, sg.tube_baffle_leak_m2, sm)
    rb = r_b_bypass(sg.bypass_area_m2, sm, sg.seal_strip_pairs, sg.rows_crossflow, re)
    rs = r_s_unequal_spacing(li, lo, re)
    rho = fluid.density_kg_m3
    dp_ideal = 2 * f * sg.rows_crossflow * g**2 / rho
    m = mass_flow_kg_s
    if re > RE_LAMINAR_LIMIT:
        dp_window_ideal = (2 + 0.6 * sg.rows_window) * m**2 / (2 * rho * sm * sg.window_area_m2)
    else:
        mu = fluid.viscosity_pa_s
        dp_window_ideal = 26 * mu * m / (rho * math.sqrt(sm * sg.window_area_m2)) * (
            sg.rows_window / (sg.tube_pitch_m - sg.tube_outer_diameter_m)
            + sg.baffle_spacing_m / sg.window_diameter_m**2
        ) + m**2 / (rho * sm * sg.window_area_m2)
    crossflow = (sg.number_of_baffles - 1) * dp_ideal * rl * rb
    window = sg.number_of_baffles * dp_window_ideal * rl
    end_zones = 2 * dp_ideal * (1 + sg.rows_window / sg.rows_crossflow) * rb * rs
    components = {"crossflow": crossflow, "window": window, "end_zones": end_zones}
    return BellDelawareResult(
        ideal_h,
        ideal_h * math.prod(factors.values()),
        crossflow + window + end_zones,
        factors,
        {"Rl": rl, "Rb": rb, "Rs": rs},
        components,
        sm,
        re,
        sg.number_of_baffles,
    )


def shell_side_pressure_drop(
    mass_flow_kg_s: float,
    fluid: FluidProperties,
    geometry: BundleGeometry,
    tube_length_m: float | None = None,
    number_of_baffles: int | None = None,
) -> float:
    """Return the Bell--Delaware shell-side pressure drop in Pa."""
    return bell_delaware_shell_side(mass_flow_kg_s, fluid, geometry, tube_length_m, number_of_baffles).pressure_drop_pa
