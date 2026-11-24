import math
from dataclasses import replace

import pytest

from she.geometry import BundleGeometry
from she.hydraulics import colebrook_white, sieder_tate_correction, tube_side_pressure_drop
from she.mechanical import check_baffle_spacing
from she.properties import FluidProperties
from she.rating import DesignConstraints, RatingInput, constraint_cost, rate_exchanger, solve_design
from she.thermal import effectiveness_ntu, tube_side_h

COLD = FluidProperties(323, 980, 4180, 0.0008, 0.6)
HOT = FluidProperties(398, 850, 2400, 0.0012, 0.12)


def make_case(duty_w=501_600.0, **overrides):
    """Energy-balanced case: cold water 293.15 -> 353.15 K, hot oil 423.15 -> 373.15 K."""
    cold_flow = duty_w / (4180.0 * 60.0)
    hot_flow = duty_w / (2400.0 * 50.0)
    args = dict(target_duty_w=duty_w, hot_inlet_k=423.15, hot_outlet_k=373.15, cold_inlet_k=293.15, cold_outlet_k=353.15,
                tube_fluid=COLD, shell_fluid=HOT, tube_mass_flow_kg_s=cold_flow, shell_mass_flow_kg_s=hot_flow,
                tube_length_m=5.0, tube_inner_diameter_m=0.016)
    args.update(overrides)
    return RatingInput(**args)


TRIAL = BundleGeometry(0.65, 0.01905, 0.0254, 240, baffle_spacing_m=0.2)
LIMITS = DesignConstraints(max_tube_pressure_drop_pa=200_000, max_shell_pressure_drop_pa=100_000)


# ---------------------------------------------------------------- hydraulics
def test_colebrook_moody_reference():
    # Moody chart: Re = 1e5, e/D = 0.0018 gives f ~ 0.0246 (Darcy).
    assert colebrook_white(100_000, 0.0018, 1.0) == pytest.approx(0.0246, abs=0.0005)


def test_friction_factor_is_darcy_laminar_and_smooth_turbulent():
    assert colebrook_white(1000, 0.0, 0.01) == pytest.approx(0.064)  # Darcy 64/Re, not Fanning 16/Re
    assert colebrook_white(1e5, 0.0, 0.01) == pytest.approx(0.316 * 1e5**-0.25, rel=0.02)  # Blasius, Darcy basis


def test_friction_factor_is_continuous_across_regimes():
    for boundary in (2300.0, 4000.0):
        assert colebrook_white(boundary + 1e-6, 1e-5, 0.01) == pytest.approx(colebrook_white(boundary - 1e-6, 1e-5, 0.01), rel=1e-6)
    values = [colebrook_white(re, 1e-5, 0.01) for re in range(2000, 5001, 100)]
    assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.006  # no step anywhere


def test_straight_tube_pressure_drop_hand_calculation():
    result = tube_side_pressure_drop(0.01, 1000, 0.001, 10, 0.01, 1, 1, roughness_m=0, nozzle_loss_coefficient=0)
    velocity = 0.01 / (1000 * math.pi * 0.01**2 / 4)
    reynolds = 1000 * velocity * 0.01 / 0.001
    expected = (64 / reynolds) * 10 / 0.01 * 1000 * velocity**2 / 2
    assert result.friction_pa == pytest.approx(expected)
    assert result.total_pa == pytest.approx(expected)


def test_return_and_nozzle_losses_use_dynamic_pressure():
    one = tube_side_pressure_drop(1.0, 1000, 0.001, 5, 0.02, 20, 1)
    four = tube_side_pressure_drop(1.0, 1000, 0.001, 5, 0.02, 20, 4)
    q = 1000 * one.velocity_m_s**2 / 2
    assert one.return_loss_pa == 0 and one.nozzle_loss_pa == pytest.approx(2 * 1.0 * q)
    assert four.return_loss_pa == pytest.approx(3 * 1.5 * (1000 * four.velocity_m_s**2 / 2))
    assert four.friction_pa > 4 * one.friction_pa  # higher velocity AND four passes of length


def test_sieder_tate_direction_and_value():
    assert sieder_tate_correction(1000, 0.001, 0.0005) == pytest.approx(1000 * 2**0.14)
    assert sieder_tate_correction(1000, 0.001, 0.002) < 1000


# ---------------------------------------------------------------- RatingInput validation
def test_rating_input_rejects_energy_imbalance_and_bad_target():
    with pytest.raises(ValueError, match="energy balance"):
        make_case(shell_mass_flow_kg_s=8.0)  # hot stream would carry 960 kW against 501.6 kW cold
    with pytest.raises(ValueError, match="target_duty_w"):
        make_case(target_duty_w=100_000.0, tube_mass_flow_kg_s=2.0, shell_mass_flow_kg_s=4.18)
    # inside the tolerance is accepted
    assert make_case(shell_mass_flow_kg_s=4.18 * 1.01).target_duty_w == 501_600.0


@pytest.mark.parametrize("override", [
    dict(target_duty_w=float("nan")),
    dict(tube_mass_flow_kg_s=float("inf")),
    dict(tube_mass_flow_kg_s=-1.0),
    dict(hot_outlet_k=430.0),
    dict(cold_outlet_k=293.15),
    dict(cold_outlet_k=380.0),
    dict(tube_length_m=0.0),
    dict(tube_wall_viscosity_pa_s=-1.0),
    dict(fouling_inner_m2_k_w=-1e-4),
])
def test_rating_input_rejects_invalid_values(override):
    with pytest.raises(ValueError):
        make_case(**override)


# ---------------------------------------------------------------- rating
def test_rating_satisfies_both_stream_energy_balances():
    case = make_case()
    r = rate_exchanger(case, TRIAL, 2)
    c_hot = case.hot_mass_flow_kg_s * HOT.specific_heat_j_kg_k
    c_cold = case.cold_mass_flow_kg_s * COLD.specific_heat_j_kg_k
    assert c_hot * (case.hot_inlet_k - r.hot_outlet_k) == pytest.approx(r.duty_w)
    assert c_cold * (r.cold_outlet_k - case.cold_inlet_k) == pytest.approx(r.duty_w)
    # independent effectiveness: duty = eps * Cmin * dT_in
    eps = effectiveness_ntu(r.ntu, min(c_hot, c_cold) / max(c_hot, c_cold), 2)
    assert r.duty_w == pytest.approx(eps * min(c_hot, c_cold) * (case.hot_inlet_k - case.cold_inlet_k))
    assert r.area_m2 == pytest.approx(math.pi * 0.01905 * 5 * 240)
    assert r.ntu == pytest.approx(r.overall_u_w_m2_k * r.area_m2 / min(c_hot, c_cold))


def test_rating_round_trip_required_area_reproduces_target():
    # F-factor route (specified temperatures) and effectiveness-NTU route (inlets only) must agree:
    # at the required area the effectiveness-NTU duty equals the target.
    case = make_case()
    r = rate_exchanger(case, TRIAL, 2)
    c_hot = case.hot_mass_flow_kg_s * HOT.specific_heat_j_kg_k
    c_cold = case.cold_mass_flow_kg_s * COLD.specific_heat_j_kg_k
    c_min = min(c_hot, c_cold)
    eps = effectiveness_ntu(r.overall_u_w_m2_k * r.required_area_m2 / c_min, c_min / max(c_hot, c_cold), 2)
    assert eps * c_min * (case.hot_inlet_k - case.cold_inlet_k) == pytest.approx(case.target_duty_w, rel=1e-6)


def test_single_tube_pass_is_counterflow_with_unit_f():
    r1 = rate_exchanger(make_case(), TRIAL, 1)
    r2 = rate_exchanger(make_case(), TRIAL, 2)
    assert r1.correction_factor == 1.0 and 0 < r2.correction_factor < 1.0
    assert r1.effectiveness > 0 and r1.tube_pressure_drop_pa < r2.tube_pressure_drop_pa


def test_tube_side_cooling_uses_exponent_0_3():
    # Hot stream on the tube side: the tube fluid is being cooled.
    case = make_case(hot_on_shell_side=False, tube_fluid=HOT, shell_fluid=COLD,
                     tube_mass_flow_kg_s=4.18 * 3, shell_mass_flow_kg_s=2.0 * 3, target_duty_w=501_600.0 * 3)
    geometry = BundleGeometry(0.65, 0.01905, 0.0254, 240, baffle_spacing_m=0.2)
    r = rate_exchanger(case, geometry, 2)
    expected = tube_side_h(case.tube_mass_flow_kg_s, HOT.density_kg_m3, HOT.viscosity_pa_s, HOT.conductivity_w_m_k,
                           HOT.specific_heat_j_kg_k, 0.016, 240, 2, heating=False)
    assert r.tube_h_w_m2_k == pytest.approx(expected)
    wrong = tube_side_h(case.tube_mass_flow_kg_s, HOT.density_kg_m3, HOT.viscosity_pa_s, HOT.conductivity_w_m_k,
                        HOT.specific_heat_j_kg_k, 0.016, 240, 2, heating=True)
    assert wrong != pytest.approx(expected)


def test_wall_viscosity_reaches_the_rating():
    base = rate_exchanger(make_case(), TRIAL, 2)
    corrected = rate_exchanger(make_case(tube_wall_viscosity_pa_s=0.0004, shell_wall_viscosity_pa_s=0.0006), TRIAL, 2)
    assert corrected.tube_h_w_m2_k == pytest.approx(base.tube_h_w_m2_k * 2**0.14)
    assert corrected.shell_h_w_m2_k == pytest.approx(base.shell_h_w_m2_k * 2**0.14)


def test_rating_shell_pressure_drop_depends_on_tube_length():
    short = rate_exchanger(make_case(tube_length_m=2.0), TRIAL, 2)
    long = rate_exchanger(make_case(tube_length_m=8.0), TRIAL, 2)
    assert long.shell.number_of_baffles > 3 * short.shell.number_of_baffles - 3
    assert long.shell_pressure_drop_pa > 2 * short.shell_pressure_drop_pa


def test_rating_rejects_impossible_geometry_and_pass_counts():
    with pytest.raises(ValueError, match="does not fit"):
        rate_exchanger(make_case(), BundleGeometry(0.3, 0.01905, 0.0254, 1000, baffle_spacing_m=0.1), 2)
    with pytest.raises(ValueError):
        rate_exchanger(make_case(), TRIAL, 7)  # 240 % 7 != 0
    with pytest.raises(ValueError):
        rate_exchanger(make_case(), replace(TRIAL, tube_count=243), 3)  # odd multi-pass unsupported


# ---------------------------------------------------------------- solver
@pytest.mark.parametrize("duty", [100_000.0, 250_000.0, 501_600.0, 900_000.0])
def test_solver_meets_target_and_round_trips(duty):
    case = make_case(duty)
    solution = solve_design(case, TRIAL, 2, LIMITS)
    rated = rate_exchanger(case, solution.geometry, solution.tube_passes, LIMITS)  # feed the output back into rating
    assert rated.duty_w == pytest.approx(solution.rating.duty_w)
    assert duty <= rated.duty_w <= 1.05 * duty  # meets the target and is not grossly oversized
    assert rated.cold_outlet_k >= 353.15 - 1e-6 and rated.hot_outlet_k <= 373.15 + 1e-6
    assert rated.tube_pressure_drop_pa <= LIMITS.max_tube_pressure_drop_pa
    assert rated.shell_pressure_drop_pa <= LIMITS.max_shell_pressure_drop_pa
    assert solution.geometry.bundle_fits()
    assert check_baffle_spacing(solution.geometry.shell_inner_diameter_m, solution.geometry.baffle_spacing_m).passed


def test_solver_area_grows_with_target_duty():
    areas = [solve_design(make_case(d), TRIAL, 2, LIMITS).rating.area_m2 for d in (100_000.0, 300_000.0, 900_000.0)]
    assert areas == sorted(areas) and areas[-1] > 2 * areas[0]


def test_solver_result_does_not_depend_on_seed_shell_size():
    # An under-sized seed shell must not leak into the answer as an impossible geometry.
    small_seed = BundleGeometry(0.30, 0.01905, 0.0254, 240, baffle_spacing_m=0.1)
    solution = solve_design(make_case(), small_seed, 2, LIMITS)
    assert solution.geometry.bundle_fits()
    assert solution.geometry.bundle_diameter_m < solution.geometry.shell_inner_diameter_m
    assert check_baffle_spacing(solution.geometry.shell_inner_diameter_m, solution.geometry.baffle_spacing_m).passed
    assert solution.rating.feasible


def test_solver_reports_infeasible_constraints():
    with pytest.raises(RuntimeError, match="failed.*closest design"):
        solve_design(make_case(), TRIAL, 2, DesignConstraints(max_tube_pressure_drop_pa=1, max_shell_pressure_drop_pa=1))
    with pytest.raises(RuntimeError, match="failed"):  # duty cannot be met inside a 0.2 m shell
        solve_design(make_case(5e6), TRIAL, 2, DesignConstraints(200_000, 100_000, max_shell_diameter_m=0.2))
    with pytest.raises(RuntimeError, match="failed"):
        solve_design(make_case(), TRIAL, 2, DesignConstraints(200_000, 100_000, max_iterations=3))


def test_solver_objectives_and_cost_function():
    area = solve_design(make_case(), TRIAL, 2, replace(LIMITS, objective="area"))
    dp = solve_design(make_case(), TRIAL, 2, replace(LIMITS, objective="pressure_drop"))
    cost = solve_design(make_case(), TRIAL, 2, replace(LIMITS, objective="cost"))
    total_dp = lambda s: s.rating.tube_pressure_drop_pa + s.rating.shell_pressure_drop_pa
    assert total_dp(dp) <= total_dp(area) + 1e-6
    assert area.rating.area_m2 <= dp.rating.area_m2 + 1e-9
    assert cost.rating.feasible
    assert constraint_cost(area.rating, LIMITS) == pytest.approx(area.rating.area_m2)
    with pytest.raises(ValueError, match="objective"):
        constraint_cost(area.rating, LIMITS, "bogus")


def test_infeasible_rating_is_penalised():
    # A bad design (too small) fails duty and receives the large penalty.
    rated = rate_exchanger(make_case(), replace(TRIAL, tube_count=40), 2, LIMITS)
    assert not rated.meets_duty and not rated.meets_outlets
    assert constraint_cost(rated, LIMITS) > 1e11
