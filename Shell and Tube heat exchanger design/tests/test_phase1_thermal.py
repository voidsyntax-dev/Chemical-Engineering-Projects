import math

import pytest

from she.properties import fluid_properties_at_temperature
from she.thermal import (
    correction_factor_1_2plus,
    effectiveness_ntu,
    heat_duty,
    lmtd_counterflow,
    overall_u,
    required_area_and_tubes,
    tube_flow_regime,
    tube_side_h,
    tube_side_nusselt,
)


# ---------------------------------------------------------------- LMTD
def test_counterflow_lmtd_hand_calculation():
    # Hot 150 -> 100 C, cold 20 -> 80 C: dT1 = 70, dT2 = 80.
    expected = (70 - 80) / math.log(70 / 80)
    assert lmtd_counterflow(423.15, 373.15, 293.15, 353.15) == pytest.approx(expected)


def test_lmtd_equal_terminal_differences_limit():
    assert lmtd_counterflow(400, 350, 300, 350) == pytest.approx(50)
    assert lmtd_counterflow(400, 350, 300, 350 + 1e-6) == pytest.approx(50, rel=1e-6)


# ---------------------------------------------------------------- F factor
def _eps_1_2(ntu, cr):
    """Independent one-shell/even-tube-pass effectiveness (Incropera Eq. 11.30 / Kays & London)."""
    q = math.sqrt(1 + cr * cr)
    e = math.exp(-ntu * q)
    return 2 / (1 + cr + q * (1 + e) / (1 - e))


def _f_from_effectiveness(ntu, c_hot, c_cold):
    """Derive (P, R, F) from the effectiveness-NTU solution, without the Bowman equation."""
    c_min, c_max = min(c_hot, c_cold), max(c_hot, c_cold)
    t_hi, t_ci = 400.0, 300.0
    q = _eps_1_2(ntu, c_min / c_max) * c_min * (t_hi - t_ci)
    t_ho, t_co = t_hi - q / c_hot, t_ci + q / c_cold
    p = (t_co - t_ci) / (t_hi - t_ci)
    r = (t_hi - t_ho) / (t_co - t_ci)
    f = q / (ntu * c_min * lmtd_counterflow(t_hi, t_ho, t_ci, t_co))
    return p, r, f


@pytest.mark.parametrize("ntu,c_hot,c_cold", [
    (0.5, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0),
    (3.0, 1.0, 0.3), (0.2, 5.0, 1.0), (4.0, 1.0, 3.0), (1.5, 1.0, 1.0 + 1e-7),
])
def test_f_factor_matches_independent_effectiveness_ntu_derivation(ntu, c_hot, c_cold):
    p, r, f_ref = _f_from_effectiveness(ntu, c_hot, c_cold)
    assert correction_factor_1_2plus(p, r) == pytest.approx(f_ref, rel=1e-6)


def test_f_factor_r_equals_one_closed_form_values():
    # R = 1 limit: F = sqrt2 P/(1-P) / ln{[2-P(2-sqrt2)]/[2-P(2+sqrt2)]}
    s = math.sqrt(2)
    for p in (0.1, 0.3, 0.5):
        expected = s * p / (1 - p) / math.log((2 - p * (2 - s)) / (2 - p * (2 + s)))
        assert correction_factor_1_2plus(p, 1.0) == pytest.approx(expected)
    assert correction_factor_1_2plus(0.5, 1.0) == pytest.approx(0.8023, abs=1e-4)
    assert correction_factor_1_2plus(0.3, 1.0) == pytest.approx(0.9686, abs=1e-4)  # was 1.58 before the fix


@pytest.mark.parametrize("p", [0.05, 0.3, 0.5])
def test_f_factor_continuous_through_r_equals_one(p):
    below, at, above = (correction_factor_1_2plus(p, r) for r in (1 - 1e-6, 1.0, 1 + 1e-6))
    assert below == pytest.approx(at, rel=1e-5) and above == pytest.approx(at, rel=1e-5)
    assert 0 < at <= 1


def test_f_factor_limits_and_infeasible_region():
    assert correction_factor_1_2plus(0, 0) == 1.0
    assert correction_factor_1_2plus(0.3, 0.0) == pytest.approx(1.0)  # isothermal shell stream
    with pytest.raises(ValueError):
        correction_factor_1_2plus(1, 1)  # P=R=1 is not a feasible 1-2 exchanger
    with pytest.raises(ValueError, match="maximum 0.5858"):
        correction_factor_1_2plus(0.6, 1.0)  # above P_max = 2/(2+sqrt2)
    with pytest.raises(ValueError):
        correction_factor_1_2plus(0.4, -1)
    with pytest.raises(ValueError):
        correction_factor_1_2plus(float("nan"), 1.0)


def test_f_factor_is_symmetric_in_fluid_roles():
    # Swapping which fluid is "tube side" maps (P, R) to (P R, 1/R) and leaves F unchanged.
    p, r = 0.4, 0.8
    assert correction_factor_1_2plus(p * r, 1 / r) == pytest.approx(correction_factor_1_2plus(p, r), rel=1e-9)


# ---------------------------------------------------------------- effectiveness
def test_effectiveness_closed_forms():
    assert effectiveness_ntu(2.0, 1.0, tube_passes=1) == pytest.approx(2 / 3)  # NTU/(1+NTU)
    assert effectiveness_ntu(1.3, 0.0, tube_passes=1) == pytest.approx(1 - math.exp(-1.3))
    assert effectiveness_ntu(1.3, 0.0, tube_passes=2) == pytest.approx(1 - math.exp(-1.3))
    assert effectiveness_ntu(0.0, 0.5) == 0.0
    assert effectiveness_ntu(1.0, 1.0, 2) < effectiveness_ntu(1.0, 1.0, 1)  # counter-current is the upper bound
    with pytest.raises(ValueError):
        effectiveness_ntu(1.0, 1.5)


# ---------------------------------------------------------------- tube-side h
def test_dittus_boelter_turbulent_value():
    # Re = 10,000 and Pr = 5 exactly (m = Re mu pi D / 4); Dittus-Boelter 0.023 Re^0.8 Pr^0.4 (Incropera Eq. 8.60).
    h = tube_side_h(0.0785398163, 1000.0, 1e-3, 0.2, 1000.0, 0.01, 1, heating=True)
    assert h == pytest.approx(0.023 * 10000**0.8 * 5**0.4 * 0.2 / 0.01, rel=1e-6)


def test_heating_uses_exponent_0_4_and_cooling_0_3():
    kwargs = dict(mass_flow_kg_s=0.7853981634, density_kg_m3=1000.0, viscosity_pa_s=1e-3,
                  conductivity_w_m_k=0.6, specific_heat_j_kg_k=4180.0, tube_inner_diameter_m=0.01, number_of_tubes=1)
    pr = 4180.0 * 1e-3 / 0.6
    heating = tube_side_h(**kwargs, heating=True)
    cooling = tube_side_h(**kwargs, heating=False)
    assert cooling / heating == pytest.approx(pr ** (0.3 - 0.4), rel=1e-9)
    assert cooling < heating  # Pr > 1: n = 0.3 gives the smaller coefficient


def test_laminar_nusselt_and_regime_labels():
    assert tube_side_nusselt(1000, 7.0) == 3.66
    assert tube_side_nusselt(2300, 7.0) == 3.66
    assert [tube_flow_regime(r) for r in (500, 2300, 5000, 10000, 50000)] == [
        "laminar", "laminar", "transitional", "turbulent", "turbulent"]


@pytest.mark.parametrize("pr,heating", [(5.0, True), (5.0, False), (0.7, True)])
def test_nusselt_is_continuous_and_monotonic_across_regimes(pr, heating):
    eps = 1e-6
    for boundary in (2300.0, 10000.0):
        low, high = tube_side_nusselt(boundary - eps, pr, heating), tube_side_nusselt(boundary + eps, pr, heating)
        assert high == pytest.approx(low, rel=1e-6)
    values = [tube_side_nusselt(re, pr, heating) for re in range(2000, 12001, 250)]
    assert values == sorted(values)  # no drop anywhere through transition
    # The old step jumped 3.66 -> 24 at Re = 2300; it may now change by only a few % over 10 Re units.
    assert tube_side_nusselt(2310, pr, heating) / tube_side_nusselt(2300, pr, heating) < 1.05


@pytest.mark.parametrize("mass_flow", [0.02, 0.2])  # laminar and turbulent in a 16 mm tube
def test_sieder_tate_wall_viscosity_correction_known_ratio(mass_flow):
    kwargs = dict(mass_flow_kg_s=mass_flow, density_kg_m3=1000.0, viscosity_pa_s=2e-3,
                  conductivity_w_m_k=0.6, specific_heat_j_kg_k=4180.0, tube_inner_diameter_m=0.016, number_of_tubes=1)
    base = tube_side_h(**kwargs)
    assert tube_side_h(**kwargs, wall_viscosity_pa_s=1e-3) == pytest.approx(base * 2**0.14)  # hot wall, mu_b/mu_w = 2
    assert tube_side_h(**kwargs, wall_viscosity_pa_s=4e-3) == pytest.approx(base * 0.5**0.14)
    assert tube_side_h(**kwargs, wall_viscosity_pa_s=2e-3) == pytest.approx(base)


# ---------------------------------------------------------------- overall U / area
def test_overall_u_outside_basis_hand_calculation():
    # 1/U = 1/800 + 1e-4 + (0.02/0.018)(1/1000 + 2e-4) + 0.02 ln(0.02/0.018)/(2*16)
    resistance = 1 / 800 + 1e-4 + (0.02 / 0.018) * (1 / 1000 + 2e-4) + 0.02 * math.log(0.02 / 0.018) / 32
    u = overall_u(1000, 800, 0.018, 0.02, 16, 2e-4, 1e-4)
    assert u == pytest.approx(363.74, abs=0.01)  # evaluated by hand
    assert u == pytest.approx(1 / resistance)


def test_clean_vs_dirty_u_differ_by_added_fouling_resistance():
    clean = overall_u(1000, 800, 0.018, 0.02, 16)
    dirty = overall_u(1000, 800, 0.018, 0.02, 16, 2e-4, 1e-4)
    added = 1e-4 + (0.02 / 0.018) * 2e-4
    assert 1 / dirty - 1 / clean == pytest.approx(added)


def test_duty_and_required_area_worked_example():
    duty = heat_duty(2.0, 4180.0, 20.0)
    assert duty == pytest.approx(167200)
    u = overall_u(1000, 800, 0.018, 0.02, 16, 0.0002, 0.0001)
    result = required_area_and_tubes(duty, u, 50, 5, 0.02, correction_factor=0.9)
    assert result.area_m2 == pytest.approx(167200 / (363.74 * 0.9 * 50), rel=1e-4)
    assert result.number_of_tubes == math.ceil(result.area_m2 / (math.pi * 0.02 * 5))


def test_properties_interpolate_and_invalid_inputs():
    props = fluid_properties_at_temperature(350, [300, 400], [1000, 900], [4000, 4200], [1e-3, 0.8e-3], [0.6, 0.7])
    assert props.density_kg_m3 == pytest.approx(950)
    with pytest.raises(ValueError, match="outside"):
        fluid_properties_at_temperature(250, [300, 400], [1, 1], [1, 1], [1, 1], [1, 1])


@pytest.mark.parametrize("call", [
    lambda: heat_duty(0, 1000, 10),
    lambda: heat_duty(-1, 1000, 10),
    lambda: heat_duty(1, 1000, float("nan")),
    lambda: lmtd_counterflow(300, 310, 300, 320),
    lambda: lmtd_counterflow(float("nan"), 310, 300, 320),
    lambda: tube_side_h(1, 1000, 0.001, 0.6, 4000, 0.01, 0),
    lambda: tube_side_h(1, 1000, 0.001, 0.6, 4000, 0.01, 5, 2),
    lambda: tube_side_h(1, 1000, 0.001, 0.6, 4000, -0.01, 4),
    lambda: tube_side_h(1, 1000, 0.001, 0.6, 4000, 0.01, 4, wall_viscosity_pa_s=0),
    lambda: overall_u(1000, 0, 0.01, 0.012, 16),
    lambda: overall_u(1000, 800, 0.02, 0.01, 16),
    lambda: overall_u(1000, 800, 0.01, 0.012, 16, -1e-4),
    lambda: required_area_and_tubes(1, 0, 1, 1, 1),
])
def test_invalid_inputs_raise_clear_exceptions(call):
    with pytest.raises(ValueError):
        call()
