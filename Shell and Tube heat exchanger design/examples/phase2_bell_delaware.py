"""Compare Kern and Bell--Delaware shell-side estimates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from any directory without PYTHONPATH

from she.bell_delaware import bell_delaware_shell_side
from she.geometry import BundleGeometry
from she.properties import FluidProperties
from she.thermal import kern_shell_side_h


def main() -> None:
    fluid = FluidProperties(350, 950, 2200, 0.0012, 0.13)
    geometry = BundleGeometry(0.6, 0.01905, 0.0254, 250, baffle_spacing_m=0.2)
    bell = bell_delaware_shell_side(8.0, fluid, geometry, tube_length_m=5.0)
    kern = kern_shell_side_h(8.0, fluid.density_kg_m3, fluid.viscosity_pa_s, fluid.conductivity_w_m_k, fluid.specific_heat_j_kg_k, 0.6, 0.01905, 0.0254, 0.2)
    print(f"Kern h: {kern:.1f} W/(m2 K)")
    print(f"Bell-Delaware ideal h: {bell.ideal_h_w_m2_k:.1f} W/(m2 K)")
    print(f"Bell-Delaware corrected h: {bell.corrected_h_w_m2_k:.1f} W/(m2 K)")
    print(f"J factors: {bell.factors}")
    print(f"Baffles: {bell.number_of_baffles}; shell Re: {bell.reynolds:.0f}")
    print(f"Shell dP: {bell.pressure_drop_pa:.0f} Pa {bell.pressure_components_pa}")


if __name__ == "__main__":
    main()
