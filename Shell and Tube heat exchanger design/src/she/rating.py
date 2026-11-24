"""Heat-exchanger rating (effectiveness--NTU) and constrained preliminary design search."""

from dataclasses import dataclass, replace
import math

from scipy.optimize import minimize_scalar

from .bell_delaware import BellDelawareResult, bell_delaware_shell_side
from .cost import exchanger_cost_proxy
from .geometry import BundleGeometry, minimum_shell_diameter_m
from .hydraulics import PressureDropResult, sieder_tate_correction, tube_side_pressure_drop
from .mechanical import check_baffle_spacing
from .properties import FluidProperties
from .thermal import (
    correction_factor_1_2plus,
    effectiveness_ntu,
    lmtd_counterflow,
    overall_u,
    tube_side_h,
)

_FAILED = 1e12  # objective value assigned to infeasible candidates


@dataclass(frozen=True)
class RatingInput:
    """Fully specified stream, thermal, and mechanical rating inputs.

    Temperatures K, flows kg/s, ``target_duty_w`` W.  The four terminal
    temperatures are the process specification; the streams' ``m cp ΔT`` and
    the target duty must agree within ``energy_balance_tolerance`` (relative),
    otherwise the input is rejected.  ``hot_on_shell_side`` places the hot
    stream on the shell side (``shell_fluid``) or the tube side.
    """

    target_duty_w: float
    hot_inlet_k: float
    hot_outlet_k: float
    cold_inlet_k: float
    cold_outlet_k: float
    tube_fluid: FluidProperties
    shell_fluid: FluidProperties
    tube_mass_flow_kg_s: float
    shell_mass_flow_kg_s: float
    tube_length_m: float
    tube_inner_diameter_m: float
    wall_conductivity_w_m_k: float = 16.0
    fouling_inner_m2_k_w: float = 0.0002
    fouling_outer_m2_k_w: float = 0.0001
    tube_wall_viscosity_pa_s: float | None = None
    shell_wall_viscosity_pa_s: float | None = None
    hot_on_shell_side: bool = True
    energy_balance_tolerance: float = 0.02

    def __post_init__(self) -> None:
        for name in ("target_duty_w", "tube_mass_flow_kg_s", "shell_mass_flow_kg_s", "tube_length_m",
                     "tube_inner_diameter_m", "wall_conductivity_w_m_k", "energy_balance_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("fouling_inner_m2_k_w", "fouling_outer_m2_k_w"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("tube_wall_viscosity_pa_s", "shell_wall_viscosity_pa_s"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        temperatures = (self.hot_inlet_k, self.hot_outlet_k, self.cold_inlet_k, self.cold_outlet_k)
        if not all(math.isfinite(t) and t > 0 for t in temperatures):
            raise ValueError("terminal temperatures must be finite and in K")
        if not (self.hot_inlet_k > self.hot_outlet_k and self.cold_outlet_k > self.cold_inlet_k):
            raise ValueError("hot stream must cool (hot_inlet > hot_outlet) and cold stream must heat")
        if self.hot_inlet_k <= self.cold_outlet_k or self.hot_outlet_k <= self.cold_inlet_k:
            raise ValueError("terminal temperature differences must be positive (no temperature cross)")
        hot_q, cold_q = self.stream_duties_w
        tol = self.energy_balance_tolerance
        if abs(hot_q - cold_q) > tol * max(hot_q, cold_q):
            raise ValueError(
                f"energy balance fails: hot stream gives {hot_q:.4g} W but cold stream takes {cold_q:.4g} W "
                f"(tolerance {tol:.1%})"
            )
        if abs(self.target_duty_w - cold_q) > tol * max(self.target_duty_w, cold_q):
            raise ValueError(
                f"target_duty_w={self.target_duty_w:.4g} W disagrees with stream duty "
                f"{cold_q:.4g} W (tolerance {tol:.1%})"
            )

    @property
    def hot_fluid(self) -> FluidProperties:
        return self.shell_fluid if self.hot_on_shell_side else self.tube_fluid

    @property
    def cold_fluid(self) -> FluidProperties:
        return self.tube_fluid if self.hot_on_shell_side else self.shell_fluid

    @property
    def hot_mass_flow_kg_s(self) -> float:
        return self.shell_mass_flow_kg_s if self.hot_on_shell_side else self.tube_mass_flow_kg_s

    @property
    def cold_mass_flow_kg_s(self) -> float:
        return self.tube_mass_flow_kg_s if self.hot_on_shell_side else self.shell_mass_flow_kg_s

    @property
    def stream_duties_w(self) -> tuple[float, float]:
        """Return (hot-stream duty, cold-stream duty) in W from ``m cp ΔT``."""
        return (
            self.hot_mass_flow_kg_s * self.hot_fluid.specific_heat_j_kg_k * (self.hot_inlet_k - self.hot_outlet_k),
            self.cold_mass_flow_kg_s * self.cold_fluid.specific_heat_j_kg_k * (self.cold_outlet_k - self.cold_inlet_k),
        )


@dataclass(frozen=True)
class RatingResult:
    """Thermal/hydraulic result for a geometry.

    ``duty_w`` and the outlet temperatures come from the effectiveness--NTU
    solution at the given inlet temperatures.  ``lmtd_k`` and
    ``correction_factor`` use the SPECIFIED terminal temperatures and give
    ``required_area_m2 = target / (U F LMTD)``.
    """

    duty_w: float
    overall_u_w_m2_k: float
    area_m2: float
    lmtd_k: float
    correction_factor: float
    tube_h_w_m2_k: float
    shell_h_w_m2_k: float
    tube_pressure_drop_pa: float
    shell_pressure_drop_pa: float
    tube_pressure: PressureDropResult
    shell_velocity_m_s: float
    shell: BellDelawareResult
    hot_outlet_k: float
    cold_outlet_k: float
    required_area_m2: float
    ntu: float
    effectiveness: float
    shell_inner_diameter_m: float
    tube_length_m: float
    meets_duty: bool = False
    meets_outlets: bool = False
    meets_pressure_drop: bool = False
    meets_geometry: bool = False

    @property
    def feasible(self) -> bool:
        return self.meets_duty and self.meets_outlets and self.meets_pressure_drop and self.meets_geometry


@dataclass(frozen=True)
class DesignConstraints:
    """Design limits and search options; pressure limits are Pa.

    ``minimum_duty_fraction`` is the fraction of the target duty (and of the
    specified temperature changes) that must be achieved.
    """

    max_tube_pressure_drop_pa: float = 100_000.0
    max_shell_pressure_drop_pa: float = 50_000.0
    minimum_duty_fraction: float = 1.0
    objective: str = "area"
    max_iterations: int = 2000
    max_shell_diameter_m: float = 2.0


@dataclass(frozen=True)
class DesignSolution:
    """Converged geometry and its rating."""

    geometry: BundleGeometry
    tube_passes: int
    rating: RatingResult
    iterations: int


def rate_exchanger(
    case: RatingInput,
    geometry: BundleGeometry,
    tube_passes: int,
    constraints: DesignConstraints | None = None,
) -> RatingResult:
    """Rate a specified exchanger geometry with effectiveness--NTU.

    Temperatures are K, flows kg/s, dimensions m, duty W, coefficients
    W/(m²·K), pressure drops Pa.  Duty is ``ε Cmin (Th,in − Tc,in)`` with
    ε from :func:`she.thermal.effectiveness_ntu` (counter-current for one tube
    pass, one shell pass with even tube passes otherwise), so duty and outlet
    temperatures are consistent with the stream energy balances.  Raises
    ``ValueError`` when the bundle does not fit the shell.
    """
    if tube_passes < 1 or geometry.tube_count % tube_passes:
        raise ValueError("tube_passes must be positive and divide tube_count")
    if tube_passes > 1 and tube_passes % 2:
        raise ValueError("odd tube pass counts above 1 are not supported")
    constraints = constraints or DesignConstraints()
    tube_is_cold = case.hot_on_shell_side
    tube_fluid, shell_fluid = case.tube_fluid, case.shell_fluid
    hi = tube_side_h(
        case.tube_mass_flow_kg_s, tube_fluid.density_kg_m3, tube_fluid.viscosity_pa_s, tube_fluid.conductivity_w_m_k,
        tube_fluid.specific_heat_j_kg_k, case.tube_inner_diameter_m, geometry.tube_count, tube_passes,
        heating=tube_is_cold, wall_viscosity_pa_s=case.tube_wall_viscosity_pa_s,
    )
    shell = bell_delaware_shell_side(case.shell_mass_flow_kg_s, shell_fluid, geometry, tube_length_m=case.tube_length_m)
    ho = shell.corrected_h_w_m2_k
    if case.shell_wall_viscosity_pa_s is not None:
        ho = sieder_tate_correction(ho, shell_fluid.viscosity_pa_s, case.shell_wall_viscosity_pa_s)
    u = overall_u(hi, ho, case.tube_inner_diameter_m, geometry.tube_outer_diameter_m, case.wall_conductivity_w_m_k,
                  case.fouling_inner_m2_k_w, case.fouling_outer_m2_k_w)
    area = math.pi * geometry.tube_outer_diameter_m * case.tube_length_m * geometry.tube_count
    c_hot = case.hot_mass_flow_kg_s * case.hot_fluid.specific_heat_j_kg_k
    c_cold = case.cold_mass_flow_kg_s * case.cold_fluid.specific_heat_j_kg_k
    c_min, c_max = min(c_hot, c_cold), max(c_hot, c_cold)
    ntu = u * area / c_min
    eff = effectiveness_ntu(ntu, c_min / c_max, tube_passes)
    duty = eff * c_min * (case.hot_inlet_k - case.cold_inlet_k)
    hot_out = case.hot_inlet_k - duty / c_hot
    cold_out = case.cold_inlet_k + duty / c_cold

    lmtd = lmtd_counterflow(case.hot_inlet_k, case.hot_outlet_k, case.cold_inlet_k, case.cold_outlet_k)
    if tube_passes == 1:
        f_corr = 1.0
    else:
        p = (case.cold_outlet_k - case.cold_inlet_k) / (case.hot_inlet_k - case.cold_inlet_k)
        r = (case.hot_inlet_k - case.hot_outlet_k) / (case.cold_outlet_k - case.cold_inlet_k)
        try:
            f_corr = correction_factor_1_2plus(p, r)
        except ValueError:
            f_corr = math.nan  # specified duty unreachable in one shell pass
    required_area = case.target_duty_w / (u * f_corr * lmtd) if f_corr == f_corr else math.inf

    tube_dp = tube_side_pressure_drop(case.tube_mass_flow_kg_s, tube_fluid.density_kg_m3, tube_fluid.viscosity_pa_s,
                                      case.tube_length_m, case.tube_inner_diameter_m, geometry.tube_count, tube_passes)
    shell_velocity = case.shell_mass_flow_kg_s / (shell_fluid.density_kg_m3 * shell.crossflow_area_m2)
    fraction = constraints.minimum_duty_fraction
    meets_outlets = (
        cold_out - case.cold_inlet_k >= fraction * (case.cold_outlet_k - case.cold_inlet_k) * (1 - 1e-9)
        and case.hot_inlet_k - hot_out >= fraction * (case.hot_inlet_k - case.hot_outlet_k) * (1 - 1e-9)
    )
    return RatingResult(
        duty_w=duty, overall_u_w_m2_k=u, area_m2=area, lmtd_k=lmtd, correction_factor=f_corr,
        tube_h_w_m2_k=hi, shell_h_w_m2_k=ho, tube_pressure_drop_pa=tube_dp.total_pa,
        shell_pressure_drop_pa=shell.pressure_drop_pa, tube_pressure=tube_dp, shell_velocity_m_s=shell_velocity,
        shell=shell, hot_outlet_k=hot_out, cold_outlet_k=cold_out, required_area_m2=required_area, ntu=ntu,
        effectiveness=eff, shell_inner_diameter_m=geometry.shell_inner_diameter_m, tube_length_m=case.tube_length_m,
        meets_duty=duty >= case.target_duty_w * fraction * (1 - 1e-9),
        meets_outlets=meets_outlets,
        meets_pressure_drop=(tube_dp.total_pa <= constraints.max_tube_pressure_drop_pa
                             and shell.pressure_drop_pa <= constraints.max_shell_pressure_drop_pa),
        meets_geometry=check_baffle_spacing(geometry.shell_inner_diameter_m, geometry.baffle_spacing_m).passed,
    )


def constraint_cost(rating: RatingResult, constraints: DesignConstraints, objective: str | None = None) -> float:
    """Return the objective value, or a large penalty if any constraint fails.

    Objectives: ``area`` (m²), ``pressure_drop`` (Pa, tube + shell) and
    ``cost`` (:func:`she.cost.exchanger_cost_proxy`, dimensionless).
    """
    selected = objective or constraints.objective
    if selected == "area":
        base = rating.area_m2
    elif selected == "pressure_drop":
        base = rating.tube_pressure_drop_pa + rating.shell_pressure_drop_pa
    elif selected == "cost":
        base = exchanger_cost_proxy(rating.shell_inner_diameter_m, rating.tube_length_m, rating.area_m2)
    else:
        raise ValueError("objective must be area, pressure_drop, or cost")
    return base + (0.0 if rating.feasible else _FAILED)


def _violation(rating: RatingResult, case: RatingInput, constraints: DesignConstraints) -> float:
    """Relative constraint violation (0 when feasible), used to report the closest infeasible design."""
    shortfall = max(0.0, 1 - rating.duty_w / (case.target_duty_w * constraints.minimum_duty_fraction))
    excess_t = max(0.0, rating.tube_pressure_drop_pa / constraints.max_tube_pressure_drop_pa - 1)
    excess_s = max(0.0, rating.shell_pressure_drop_pa / constraints.max_shell_pressure_drop_pa - 1)
    return shortfall + excess_t + excess_s + (0.0 if rating.meets_geometry else 1.0)


def solve_design(
    case: RatingInput,
    initial_geometry: BundleGeometry,
    tube_passes: int = 2,
    constraints: DesignConstraints | None = None,
) -> DesignSolution:
    """Search tube count, pass arrangement and baffle spacing for the cheapest feasible design.

    ``initial_geometry`` supplies the fixed parameters (tube OD, pitch, layout,
    baffle cut, clearances, seal strips); its tube count and shell ID are only
    a starting point.  For each tube count (geometric grid) and pass count
    (``tube_passes``, 1, 2, 4) the shell ID is the smallest that holds the
    bundle with the minimum clearance, and baffle spacing is scanned between
    the TEMA limits.  A design is feasible only if ALL hold: duty and outlet
    temperatures meet the specification (within ``minimum_duty_fraction``),
    both pressure drops are within limits, the bundle fits the shell, and the
    baffle spacing is within TEMA limits.  The best feasible grid point is then
    reduced tube-by-tube and its spacing refined with
    ``scipy.optimize.minimize_scalar`` (minimising total pressure drop).
    ``RuntimeError`` reports the closest infeasible design if none is found
    within ``max_iterations`` rating evaluations.
    """
    constraints = constraints or DesignConstraints()
    base = initial_geometry
    if case.tube_inner_diameter_m >= base.tube_outer_diameter_m:
        raise ValueError("tube_inner_diameter_m must be less than the geometry tube_outer_diameter_m")
    passes_options = sorted({tube_passes, 1, 2, 4})
    iterations = 0
    best: tuple[tuple[float, float], BundleGeometry, int, RatingResult] | None = None
    closest: tuple[float, BundleGeometry, int, RatingResult] | None = None
    last_error = ""

    def evaluate(geometry: BundleGeometry, passes: int):
        nonlocal iterations, best, closest, last_error
        if iterations >= constraints.max_iterations:
            return None
        iterations += 1
        try:
            rating = rate_exchanger(case, geometry, passes, constraints)
        except ValueError as exc:
            last_error = str(exc)
            return None  # geometrically impossible candidate
        violation = _violation(rating, case, constraints)
        if closest is None or violation < closest[0]:
            closest = (violation, geometry, passes, rating)
        if rating.feasible:
            key = (round(constraint_cost(rating, constraints), 9), rating.tube_pressure_drop_pa + rating.shell_pressure_drop_pa)
            if best is None or key < best[0]:
                best = (key, geometry, passes, rating)
        return rating

    def candidate(count: int, passes: int, spacing_fraction: float) -> BundleGeometry | None:
        shell = math.ceil(minimum_shell_diameter_m(count, base.tube_outer_diameter_m, base.tube_pitch_m, base.tube_layout) * 100) / 100
        if shell > constraints.max_shell_diameter_m:
            return None
        spacing = max(0.0508, spacing_fraction * shell)
        return replace(base, shell_inner_diameter_m=shell, tube_count=count, baffle_spacing_m=spacing,
                       inlet_baffle_spacing_m=None, outlet_baffle_spacing_m=None)

    count_grid = sorted({int(round(10 * 1.25**k)) for k in range(0, 31)})
    for passes in passes_options:
        for raw in count_grid:
            count = max(passes, math.ceil(raw / passes) * passes)
            for fraction in (0.25, 0.35, 0.5, 0.75, 1.0):
                geometry = candidate(count, passes, fraction)
                if geometry is not None:
                    evaluate(geometry, passes)

    if best is not None:
        # Local descent on tube count: remove one tube-per-pass step while feasible.
        _, seed, passes, _ = best
        spacing_fraction = seed.baffle_spacing_m / seed.shell_inner_diameter_m
        while seed.tube_count - passes >= passes:
            smaller = candidate(seed.tube_count - passes, passes, spacing_fraction)
            if smaller is None:
                break
            evaluate(smaller, passes)
            if best[1] is not smaller:
                break
            seed = smaller
        # Refine baffle spacing at the final tube count to minimise pressure drop.
        _, seed, passes, _ = best
        low = max(0.2 * seed.shell_inner_diameter_m, 0.0508)
        high = min(seed.shell_inner_diameter_m, case.tube_length_m / 2)
        if high > low and iterations < constraints.max_iterations:
            def spacing_objective(spacing: float) -> float:
                rating = evaluate(replace(seed, baffle_spacing_m=float(spacing)), passes)
                if rating is None or not rating.feasible:
                    return _FAILED
                return rating.tube_pressure_drop_pa + rating.shell_pressure_drop_pa

            minimize_scalar(spacing_objective, bounds=(low, high), method="bounded", options={"maxiter": 20, "xatol": 1e-4})
    if best is None:
        detail = f"no candidate geometry could be rated (last error: {last_error or 'none'})"
        if closest is not None:
            r = closest[3]
            detail = (
                f"closest design has duty {r.duty_w:.4g} W (target {case.target_duty_w:.4g} W), "
                f"tube dP {r.tube_pressure_drop_pa:.3g} Pa, shell dP {r.shell_pressure_drop_pa:.3g} Pa, "
                f"{closest[1].tube_count} tubes, {closest[2]} pass(es), shell {closest[1].shell_inner_diameter_m:.2f} m"
            )
        raise RuntimeError(f"design solver failed after {iterations} rating evaluations: {detail}")
    _, geometry, passes, rating = best
    return DesignSolution(geometry, passes, rating, iterations)
