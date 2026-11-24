"""Simple preliminary cost proxy."""

import math


def exchanger_cost_proxy(
    shell_inner_diameter_m: float,
    shell_length_m: float,
    tube_area_m2: float,
    material_factor: float = 1.0,
    shell_factor: float = 1.0,
    tube_factor: float = 1.0,
) -> float:
    """Return a dimensionless installed-cost proxy.

    Dimensions and area are m and m²; factors are dimensionless.  The proxy is
    ``material_factor*(shell_factor*shell area + tube_factor*tube area)`` and
    is intended only for comparing candidate geometries, not pricing.
    """
    if min(shell_inner_diameter_m, shell_length_m, tube_area_m2, material_factor, shell_factor, tube_factor) <= 0:
        raise ValueError("cost dimensions and factors must be positive")
    shell_area = math.pi * shell_inner_diameter_m * shell_length_m
    return material_factor * (shell_factor * shell_area + tube_factor * tube_area_m2)
