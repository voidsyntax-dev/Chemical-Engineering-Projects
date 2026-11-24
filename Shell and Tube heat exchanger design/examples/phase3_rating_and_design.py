"""Rate and design a preliminary exchanger."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from any directory without PYTHONPATH

from she.geometry import BundleGeometry
from she.properties import FluidProperties
from she.rating import DesignConstraints, RatingInput, solve_design


def main() -> None:
    water = FluidProperties(323, 980, 4180, 0.0008, 0.6)
    process = FluidProperties(398, 850, 2400, 0.0012, 0.12)
    # 2 kg/s water 293.15->353.15 K (501.6 kW) against 4.18 kg/s process 423.15->373.15 K (501.6 kW).
    case = RatingInput(501_600, 423.15, 373.15, 293.15, 353.15, water, process, 2.0, 4.18, 5.0, 0.016)
    trial = BundleGeometry(0.65, 0.01905, 0.0254, 240, baffle_spacing_m=0.2)
    solution = solve_design(case, trial, 2, DesignConstraints(max_tube_pressure_drop_pa=200_000, max_shell_pressure_drop_pa=100_000))
    r = solution.rating
    print(f"Tubes: {solution.geometry.tube_count}, shell ID: {solution.geometry.shell_inner_diameter_m:.3f} m, passes: {solution.tube_passes}")
    print(f"Duty: {r.duty_w:,.0f} W (target {case.target_duty_w:,.0f}); U: {r.overall_u_w_m2_k:.1f} W/(m2 K)")
    print(f"Outlets: hot {r.hot_outlet_k:.2f} K, cold {r.cold_outlet_k:.2f} K")
    print(f"Tube dP: {r.tube_pressure_drop_pa:,.0f} Pa; shell dP: {r.shell_pressure_drop_pa:,.0f} Pa")


if __name__ == "__main__":
    main()
