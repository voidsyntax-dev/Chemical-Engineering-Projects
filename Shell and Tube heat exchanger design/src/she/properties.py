"""Fluid property data and interpolation helpers."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FluidProperties:
    """Thermophysical properties at one temperature.

    All properties use SI units: temperature is K, density is kg/m³,
    specific heat is J/(kg·K), viscosity is Pa·s, and conductivity is W/(m·K).
    """

    temperature_k: float
    density_kg_m3: float
    specific_heat_j_kg_k: float
    viscosity_pa_s: float
    conductivity_w_m_k: float

    def __post_init__(self) -> None:
        for name in ("temperature_k", "density_kg_m3", "specific_heat_j_kg_k", "viscosity_pa_s", "conductivity_w_m_k"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def fluid_properties_at_temperature(
    temperature_k: float,
    temperatures_k: Sequence[float],
    densities_kg_m3: Sequence[float],
    specific_heats_j_kg_k: Sequence[float],
    viscosities_pa_s: Sequence[float],
    conductivities_w_m_k: Sequence[float],
) -> FluidProperties:
    """Linearly interpolate supplied fluid properties at a mean temperature.

    Temperatures are in K; density is kg/m³; heat capacity is J/(kg·K);
    viscosity is Pa·s; and conductivity is W/(m·K).  Data must contain at
    least two strictly increasing temperature points and the requested
    temperature must lie within the supplied range.
    """
    arrays = [
        np.asarray(values, dtype=float)
        for values in (
            temperatures_k,
            densities_kg_m3,
            specific_heats_j_kg_k,
            viscosities_pa_s,
            conductivities_w_m_k,
        )
    ]
    temperatures, density, cp, viscosity, conductivity = arrays
    if len(temperatures) < 2:
        raise ValueError("at least two property temperature points are required")
    if any(len(values) != len(temperatures) for values in arrays[1:]):
        raise ValueError("all property arrays must have the same length")
    if not np.all(np.isfinite(np.concatenate(arrays))):
        raise ValueError("property data must contain only finite values")
    if np.any(np.diff(temperatures) <= 0):
        raise ValueError("temperatures must be strictly increasing")
    if not temperatures[0] <= temperature_k <= temperatures[-1]:
        raise ValueError("requested temperature is outside the property data range")
    values = [float(np.interp(temperature_k, temperatures, values)) for values in arrays[1:]]
    if any(value <= 0 for value in values):
        raise ValueError("interpolated fluid properties must be positive")
    return FluidProperties(float(temperature_k), *values)
