"""Worked phase 1 thermal design example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run from any directory without PYTHONPATH

from she.thermal import (
    correction_factor_1_2plus,
    heat_duty,
    lmtd_counterflow,
    overall_u,
    required_area_and_tubes,
    tube_side_h,
)


def main() -> None:
    duty = heat_duty(2.0, 4180.0, 20.0)
    lmtd = lmtd_counterflow(423.15, 373.15, 293.15, 353.15)
    hi = tube_side_h(1.0, 998, 0.0009, 0.6, 4180, 0.018, 40, 2)
    u = overall_u(hi, 800, 0.018, 0.02, 16, 0.0002, 0.0001)
    f = correction_factor_1_2plus(0.25, 2.0)
    result = required_area_and_tubes(duty, u, lmtd, 5.0, 0.02, f)
    print(f"Duty: {duty:,.0f} W")
    print(f"LMTD: {lmtd:.2f} K, F: {f:.3f}")
    print(f"Tube h: {hi:.1f} W/(m² K), Uo: {u:.1f} W/(m² K)")
    print(f"Required area: {result.area_m2:.2f} m²; tubes: {result.number_of_tubes}")


if __name__ == "__main__":
    main()
