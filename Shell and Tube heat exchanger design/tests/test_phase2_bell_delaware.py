"""Bell--Delaware tests.

Reference values: the factor equations are those of Taborek / Serth (see the
module docstring of ``she.bell_delaware``).  The book was not available, so each
factor is tested against the equation evaluated BY HAND with simple numbers
(worked in the comments), not against the code's own output.
"""

import math

import pytest

from she.bell_delaware import (
    bell_delaware_shell_side,
    ideal_j_and_f,
    j_b_bundle_bypass,
    j_c_baffle_cut,
    j_l_leakage,
    j_r_laminar,
    j_s_unequal_spacing,
    r_b_bypass,
    r_l_leakage,
    r_s_unequal_spacing,
    shell_geometry,
    shell_side_pressure_drop,
)
from she.geometry import BundleGeometry, tube_count_from_shell_id
from she.properties import FluidProperties
from she.thermal import kern_shell_side_h


@pytest.fixture
def fluid():
    return FluidProperties(350, 950, 2200, 0.0012, 0.13)


@pytest.fixture
def geometry():
    return BundleGeometry(0.6, 0.01905, 0.0254, 250, baffle_spacing_m=0.2)


# ---------------------------------------------------------------- derived geometry
def test_shell_geometry_hand_values(geometry):
    sg = shell_geometry(geometry, tube_length_m=5.0)
    assert sg.theta_ds == pytest.approx(2 * math.pi / 3)  # 2 acos(1 - 2*0.25) = 2 acos(0.5)
    swg = 0.6**2 / 8 * (2 * math.pi / 3 - math.sin(2 * math.pi / 3))
    assert sg.window_area_m2 + 250 * sg.fw * math.pi * 0.01905**2 / 4 == pytest.approx(swg)
    assert sg.rows_crossflow == pytest.approx(0.6 * 0.5 / (0.866 * 0.0254))
    assert sg.rows_window == pytest.approx(0.8 * 0.25 * 0.6 / (0.866 * 0.0254))
    dotl = 0.0254 * math.sqrt(250 / (math.pi / (2 * math.sqrt(3)))) + 0.01905
    assert sg.otl_diameter_m == pytest.approx(dotl)
    assert sg.crossflow_area_m2 == pytest.approx(0.2 * (0.6 - dotl + (dotl - 0.01905) * (0.0254 - 0.01905) / 0.0254))
    assert 0 < sg.fw < 0.5 and sg.fc == pytest.approx(1 - 2 * sg.fw)


def test_baffle_count_follows_tube_length_and_spacing(geometry):
    assert shell_geometry(geometry, tube_length_m=5.0).number_of_baffles == 24  # (5 - 0.2 - 0.2)/0.2 + 1
    assert shell_geometry(geometry, tube_length_m=10.0).number_of_baffles == 49
    assert shell_geometry(geometry, number_of_baffles=7).number_of_baffles == 7
    with pytest.raises(ValueError, match="tube_length_m"):
        shell_geometry(geometry)
    with pytest.raises(ValueError, match="too short"):
        shell_geometry(geometry, tube_length_m=0.1)


def test_bundle_that_does_not_fit_the_shell_is_rejected():
    crowded = BundleGeometry(0.3, 0.01905, 0.0254, 1000, baffle_spacing_m=0.1)
    assert not crowded.bundle_fits()
    with pytest.raises(ValueError, match="does not fit"):
        shell_geometry(crowded, tube_length_m=5.0)


def test_tube_count_estimate_is_labelled_maximum_and_fits(geometry):
    count = tube_count_from_shell_id(0.6, 0.01905, 0.0254)
    assert BundleGeometry(0.6, 0.01905, 0.0254, count).bundle_fits()
    assert not BundleGeometry(0.6, 0.01905, 0.0254, int(count * 1.1)).bundle_fits()


# ---------------------------------------------------------------- ideal bank j, f
def test_ideal_bank_coefficients_hand_values():
    # p/d = 1.33 makes the (1.33/(p/d))^a factor exactly 1.
    j, f = ideal_j_and_f("triangular", 5000.0, 1.33)
    assert j == pytest.approx(0.321 * 5000**-0.388)
    assert f == pytest.approx(0.486 * 5000**-0.152)
    j, f = ideal_j_and_f("square", 20000.0, 1.33)
    assert j == pytest.approx(0.370 * 20000**-0.395)
    assert f == pytest.approx(0.391 * 20000**-0.148)
    j, f = ideal_j_and_f("triangular", 50.0, 1.33)
    assert j == pytest.approx(1.36 * 50**-0.657)


def test_ideal_bank_pitch_ratio_correction():
    # a = 1.45/(1 + 0.14 Re^0.519); at Re = 5000 a ~ 0.235 so a larger pitch lowers j.
    j_tight, _ = ideal_j_and_f("triangular", 5000.0, 1.25)
    j_wide, _ = ideal_j_and_f("triangular", 5000.0, 1.5)
    assert j_tight > j_wide
    a = 1.45 / (1 + 0.14 * 5000**0.519)
    assert j_wide == pytest.approx(0.321 * (1.33 / 1.5) ** a * 5000**-0.388)


# ---------------------------------------------------------------- each J / R factor on its own
def test_jc_hand_values():
    assert j_c_baffle_cut(1.0) == pytest.approx(1.27)  # 0.55 + 0.72
    assert j_c_baffle_cut(0.0) == pytest.approx(0.55)
    assert j_c_baffle_cut(0.7) == pytest.approx(0.55 + 0.504)
    with pytest.raises(ValueError):
        j_c_baffle_cut(1.2)


def test_jl_hand_value():
    # Ssb = Stb = 2, Sm = 10: rs = 0.5, rlm = 0.4 -> 0.44*0.5 + (1 - 0.22) exp(-0.88) = 0.5435
    assert j_l_leakage(2.0, 2.0, 10.0) == pytest.approx(0.22 + 0.78 * math.exp(-0.88))
    assert j_l_leakage(0.0, 0.0, 10.0) == 1.0
    # shell-baffle leakage (rs -> 1) is worse than tube-baffle leakage for the same total area
    assert j_l_leakage(4.0, 0.0, 10.0) < j_l_leakage(0.0, 4.0, 10.0)


def test_jb_hand_values():
    # Sb/Sm = 0.5, no seal strips: exp(-1.25*0.5) turbulent, exp(-1.35*0.5) laminar
    assert j_b_bundle_bypass(0.5, 1.0, 0, 10.0, 5000.0) == pytest.approx(math.exp(-0.625))
    assert j_b_bundle_bypass(0.5, 1.0, 0, 10.0, 50.0) == pytest.approx(math.exp(-0.675))
    # Nss/Nc = 0.1: exp[-1.25*0.5*(1 - 0.2^(1/3))]
    assert j_b_bundle_bypass(0.5, 1.0, 1, 10.0, 5000.0) == pytest.approx(math.exp(-0.625 * (1 - 0.2 ** (1 / 3))))
    assert j_b_bundle_bypass(0.5, 1.0, 5, 10.0, 5000.0) == 1.0  # Nss/Nc >= 0.5
    assert j_b_bundle_bypass(0.0, 1.0, 0, 10.0, 5000.0) == 1.0


def test_js_hand_values():
    assert j_s_unequal_spacing(10, 1.0, 1.0, 5000.0) == pytest.approx(1.0)
    # Nb = 10, Li* = Lo* = 2, n = 0.6: (9 + 2*2^0.4)/(9 + 4) = 0.8953
    assert j_s_unequal_spacing(10, 2.0, 2.0, 5000.0) == pytest.approx((9 + 2 * 2**0.4) / 13)
    # laminar n = 1/3: exponent 2/3
    assert j_s_unequal_spacing(10, 2.0, 2.0, 50.0) == pytest.approx((9 + 2 * 2 ** (2 / 3)) / 13)
    assert j_s_unequal_spacing(10, 2.0, 2.0, 5000.0) < 1.0  # longer end spacing lowers h; it never raises it
    assert j_s_unequal_spacing(10, 0.5, 0.5, 5000.0) > 1.0  # shorter end spacing: higher velocity there


def test_jr_hand_values_and_interpolation():
    assert j_r_laminar(500.0, 40.0) == 1.0
    assert j_r_laminar(100.0, 40.0) == 1.0
    jr_low = (10 / 40.0) ** 0.18
    assert j_r_laminar(10.0, 40.0) == pytest.approx(jr_low)
    assert j_r_laminar(20.0, 40.0) == pytest.approx(jr_low)
    assert j_r_laminar(60.0, 40.0) == pytest.approx((jr_low + 1) / 2)  # linear midpoint of 20..100
    assert j_r_laminar(99.999, 40.0) == pytest.approx(1.0, abs=1e-4)  # no step at Re = 100


def test_pressure_correction_factors_hand_values():
    # Ssb = Stb = 2, Sm = 10: rs = 0.5, rlm = 0.4, p = -0.15*1.5 + 0.8 = 0.575
    assert r_l_leakage(2.0, 2.0, 10.0) == pytest.approx(math.exp(-1.33 * 1.5 * 0.4**0.575))
    assert r_b_bypass(0.5, 1.0, 0, 10.0, 5000.0) == pytest.approx(math.exp(-3.7 * 0.5))
    assert r_b_bypass(0.5, 1.0, 0, 10.0, 50.0) == pytest.approx(math.exp(-4.5 * 0.5))
    assert r_s_unequal_spacing(1.0, 1.0, 5000.0) == pytest.approx(1.0)
    assert r_s_unequal_spacing(2.0, 2.0, 5000.0) == pytest.approx(2.0 ** -1.8)  # (B/Lbi)^(2-0.2)


# ---------------------------------------------------------------- assembled result
def test_assembled_result_uses_each_factor_once(fluid, geometry):
    result = bell_delaware_shell_side(8.0, fluid, geometry, tube_length_m=5.0)
    assert result.corrected_h_w_m2_k == pytest.approx(result.ideal_h_w_m2_k * math.prod(result.factors.values()))
    sg = shell_geometry(geometry, tube_length_m=5.0)
    assert result.factors["Jc"] == pytest.approx(0.55 + 0.72 * sg.fc)
    assert result.factors["Jl"] == pytest.approx(j_l_leakage(sg.shell_baffle_leak_m2, sg.tube_baffle_leak_m2, sg.crossflow_area_m2))
    assert result.factors["Js"] == pytest.approx(1.0)  # equal spacing
    assert result.factors["Jr"] == 1.0  # turbulent shell flow
    # Pressure drop uses R factors, not the heat-transfer J factors.
    assert result.pressure_factors["Rl"] != pytest.approx(result.factors["Jl"])
    assert result.pressure_factors["Rb"] != pytest.approx(result.factors["Jb"])
    assert sum(result.pressure_components_pa.values()) == pytest.approx(result.pressure_drop_pa)


def test_shell_pressure_drop_scales_with_tube_length(fluid, geometry):
    short = bell_delaware_shell_side(8.0, fluid, geometry, tube_length_m=2.0)
    base = bell_delaware_shell_side(8.0, fluid, geometry, tube_length_m=5.0)
    long = bell_delaware_shell_side(8.0, fluid, geometry, tube_length_m=10.0)
    assert (short.number_of_baffles, base.number_of_baffles, long.number_of_baffles) == (9, 24, 49)
    assert short.pressure_drop_pa < base.pressure_drop_pa < long.pressure_drop_pa
    # window + crossflow are proportional to baffle count; the end zones are fixed
    per_baffle = (long.pressure_drop_pa - base.pressure_drop_pa) / (49 - 24)
    assert base.pressure_drop_pa - short.pressure_drop_pa == pytest.approx(per_baffle * (24 - 9), rel=0.05)
    assert shell_side_pressure_drop(8.0, fluid, geometry, tube_length_m=5.0) == pytest.approx(base.pressure_drop_pa)


def test_pressure_drop_grows_with_flow_and_corrected_h_below_ideal(fluid, geometry):
    low = bell_delaware_shell_side(4.0, fluid, geometry, tube_length_m=5.0)
    high = bell_delaware_shell_side(8.0, fluid, geometry, tube_length_m=5.0)
    assert high.pressure_drop_pa > 3 * low.pressure_drop_pa  # roughly ~ m^2
    assert high.corrected_h_w_m2_k < high.ideal_h_w_m2_k
    assert high.corrected_h_w_m2_k > low.corrected_h_w_m2_k


def test_laminar_shell_flow_branch_runs(geometry):
    viscous = FluidProperties(350, 900, 2000, 0.5, 0.12)
    result = bell_delaware_shell_side(0.3, viscous, geometry, tube_length_m=5.0)
    assert result.reynolds < 100
    assert result.factors["Jr"] <= 1.0 and result.pressure_drop_pa > 0


# ---------------------------------------------------------------- Kern comparison
def test_kern_hand_calculation(fluid):
    # Kern: h = 0.36 k/de Re^0.55 Pr^(1/3), de = (3.464 p^2 - pi do^2)/(pi do) for a triangular pitch.
    d, p, ds, b = 0.01905, 0.0254, 0.6, 0.2
    de = (4 * (math.sqrt(3) / 2 * p**2 - math.pi * d**2 / 4)) / (math.pi * d)
    area = ds * b * (p - d) / p
    re = 8.0 / area * de / fluid.viscosity_pa_s
    pr = fluid.specific_heat_j_kg_k * fluid.viscosity_pa_s / fluid.conductivity_w_m_k
    expected = 0.36 * fluid.conductivity_w_m_k / de * re**0.55 * pr ** (1 / 3)
    assert kern_shell_side_h(8, fluid.density_kg_m3, fluid.viscosity_pa_s, fluid.conductivity_w_m_k,
                             fluid.specific_heat_j_kg_k, ds, d, p, b) == pytest.approx(expected)


def test_bell_delaware_vs_kern_plausibility_band(fluid):
    # Plausibility only (not validation): for a bundle that fills the shell the two methods are the same
    # order of magnitude.  Kern is a cruder correlation, so +-50 % is the most that can be asserted.
    geometry = BundleGeometry(0.6, 0.01905, 0.0254, 380, baffle_spacing_m=0.2)
    bell = bell_delaware_shell_side(8, fluid, geometry, tube_length_m=5.0)
    kern = kern_shell_side_h(8, fluid.density_kg_m3, fluid.viscosity_pa_s, fluid.conductivity_w_m_k,
                             fluid.specific_heat_j_kg_k, 0.6, 0.01905, 0.0254, 0.2)
    assert abs(bell.corrected_h_w_m2_k - kern) / kern < 0.5


# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("call", [
    lambda: BundleGeometry(0.6, 0.01905, 0.0254, 250, baffle_spacing_m=0),
    lambda: BundleGeometry(0.6, 0.01905, 0.0254, 250, baffle_cut_fraction=0.6),
    lambda: BundleGeometry(0.6, 0.01905, 0.018, 250),
    lambda: BundleGeometry(-0.6, 0.01905, 0.0254, 250),
    lambda: BundleGeometry(0.6, 0.01905, 0.0254, 0),
    lambda: ideal_j_and_f("triangular", -1.0, 1.3),
    lambda: j_l_leakage(-1, 1, 1),
    lambda: j_s_unequal_spacing(0, 1, 1, 1000),
    lambda: j_r_laminar(0.0, 10),
])
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


def test_nonpositive_flow_or_properties_rejected(fluid, geometry):
    with pytest.raises(ValueError):
        bell_delaware_shell_side(0.0, fluid, geometry, tube_length_m=5.0)
    with pytest.raises(ValueError, match="viscosity_pa_s"):
        FluidProperties(350, 950, 2200, -1.0, 0.13)
    with pytest.raises(ValueError, match="density_kg_m3"):
        FluidProperties(350, float("nan"), 2200, 0.001, 0.13)
