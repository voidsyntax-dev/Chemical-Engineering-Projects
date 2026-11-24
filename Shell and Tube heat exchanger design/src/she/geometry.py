"""Tube-bundle geometry calculations for shell-side methods."""

from dataclasses import dataclass
import math

PACKING_TRIANGULAR = math.pi / (2 * math.sqrt(3))  # 0.9069, tubes per (D_ctl/p)^2 on a 30 degree layout
PACKING_SQUARE = math.pi / 4  # 90 degree layout


def min_bundle_clearance_m(shell_inner_diameter_m: float) -> float:
    """Minimum diametral shell-to-bundle clearance, m (fixed tubesheet).

    ``L_bb = 12 mm + 0.005 D_s`` as used with the Bell--Delaware method in
    Serth, *Process Heat Transfer* (transcribed from secondary sources;
    check against the book before final design).
    """
    return 0.012 + 0.005 * shell_inner_diameter_m


@dataclass(frozen=True)
class BundleGeometry:
    """A validated shell-and-tube bundle geometry, all dimensions in metres."""

    shell_inner_diameter_m: float
    tube_outer_diameter_m: float
    tube_pitch_m: float
    tube_count: int
    baffle_cut_fraction: float = 0.25
    baffle_spacing_m: float = 0.25
    tube_layout: str = "triangular"
    tube_to_baffle_clearance_m: float = 0.0008
    shell_to_baffle_clearance_m: float = 0.003
    inlet_baffle_spacing_m: float | None = None
    outlet_baffle_spacing_m: float | None = None
    seal_strips: int = 0

    def __post_init__(self) -> None:
        if self.shell_inner_diameter_m <= 0 or self.tube_outer_diameter_m <= 0 or self.tube_pitch_m <= 0:
            raise ValueError("shell and tube dimensions must be positive")
        if self.tube_pitch_m <= self.tube_outer_diameter_m:
            raise ValueError("tube pitch must exceed tube outside diameter")
        if self.tube_count < 1:
            raise ValueError("tube_count must be positive")
        if not 0 < self.baffle_cut_fraction < 0.5:
            raise ValueError("baffle_cut_fraction must be between 0 and 0.5")
        if self.baffle_spacing_m <= 0:
            raise ValueError("baffle_spacing_m must be positive")
        if self.tube_layout not in {"triangular", "square"}:
            raise ValueError("tube_layout must be 'triangular' or 'square'")
        if self.tube_to_baffle_clearance_m < 0 or self.shell_to_baffle_clearance_m < 0:
            raise ValueError("clearances cannot be negative")
        if self.seal_strips < 0:
            raise ValueError("seal_strips cannot be negative")

    @property
    def bundle_diameter_m(self) -> float:
        """Outer tube limit diameter D_otl in m (tube centres circle plus one tube OD)."""
        return bundle_diameter_from_tube_count(self.tube_count, self.tube_pitch_m, self.tube_layout) + self.tube_outer_diameter_m

    @property
    def bundle_clearance_m(self) -> float:
        """Diametral gap L_bb between shell ID and outer tube limit, m (negative if it does not fit)."""
        return self.shell_inner_diameter_m - self.bundle_diameter_m

    def bundle_fits(self) -> bool:
        """True when the bundle leaves at least the minimum shell-to-bundle clearance."""
        return self.bundle_clearance_m >= min_bundle_clearance_m(self.shell_inner_diameter_m)


def _packing(tube_layout: str) -> float:
    if tube_layout not in {"triangular", "square"}:
        raise ValueError("tube_layout must be 'triangular' or 'square'")
    return PACKING_TRIANGULAR if tube_layout == "triangular" else PACKING_SQUARE


def tube_count_from_shell_id(
    shell_inner_diameter_m: float,
    tube_outer_diameter_m: float,
    tube_pitch_m: float,
    tube_layout: str = "triangular",
) -> int:
    """Estimate the maximum tube count that fits a shell (single tube pass, no pass lanes).

    All dimensions are m.  ``N = packing * (D_ctl/p)^2`` with packing
    pi/(2 sqrt 3) (triangular) or pi/4 (square) and ``D_ctl`` the tube-centre
    circle left after the minimum shell clearance and one tube OD.  This is an
    APPROXIMATION: it ignores partition lanes, tie rods and edge effects, and
    typically exceeds TEMA/Kern count tables by several percent.  Use a vendor
    or TEMA table for final designs.
    """
    if shell_inner_diameter_m <= 0 or tube_outer_diameter_m <= 0 or tube_pitch_m <= tube_outer_diameter_m:
        raise ValueError("shell diameter and tube dimensions are invalid")
    packing = _packing(tube_layout)
    centre_circle = shell_inner_diameter_m - min_bundle_clearance_m(shell_inner_diameter_m) - tube_outer_diameter_m
    count = math.floor(packing * (centre_circle / tube_pitch_m) ** 2) if centre_circle > 0 else 0
    if count < 1:
        raise ValueError("shell is too small for one tube at the requested pitch")
    return count


def bundle_diameter_from_tube_count(
    tube_count: int,
    tube_pitch_m: float,
    tube_layout: str = "triangular",
) -> float:
    """Return the tube-centre circle diameter D_ctl in m from tube count and pitch."""
    if tube_count < 1 or tube_pitch_m <= 0:
        raise ValueError("tube_count and tube_pitch_m must be positive")
    return tube_pitch_m * math.sqrt(tube_count / _packing(tube_layout))


def minimum_shell_diameter_m(
    tube_count: int,
    tube_outer_diameter_m: float,
    tube_pitch_m: float,
    tube_layout: str = "triangular",
) -> float:
    """Return the smallest shell ID in m that holds the bundle with minimum clearance."""
    otl = bundle_diameter_from_tube_count(tube_count, tube_pitch_m, tube_layout) + tube_outer_diameter_m
    return (otl + 0.012) / (1 - 0.005)
