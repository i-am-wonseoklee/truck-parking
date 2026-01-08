# src/truck_parking/core/obstacle.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Mapping, Any

from shapely.geometry import Polygon
from shapely.prepared import PreparedGeometry, prep
from shapely.validation import explain_validity

from truck_parking.common.guards import req_finite


@dataclass(frozen=True, slots=True)
class Obstacle:
    """Immutable polygon obstacle defined by 2D coordinates."""

    coords: Tuple[
        Tuple[float, float], ...
    ] = ()  # Coordinates of the obstacle polygon vertices.

    _poly: Polygon = field(init=False, repr=False, compare=False)  # Cached polygon.
    _poly_prep: PreparedGeometry = field(
        init=False, repr=False, compare=False
    )  # Cached prepared polygon.
    _poly_bounds: Tuple[float, float, float, float] = field(
        init=False, repr=False, compare=False
    )  # Cached polygon bounds.

    def __post_init__(self) -> None:
        coords = self._validate_coords(self.coords)

        poly = Polygon(coords) if coords else Polygon()
        if coords and not poly.is_valid:
            raise ValueError(f"Invalid obstacle polygon: {explain_validity(poly)}")

        object.__setattr__(self, "_poly", poly)
        object.__setattr__(self, "_poly_prep", prep(poly))
        object.__setattr__(self, "_poly_bounds", poly.bounds)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Obstacle:
        """Creates an Obstacle instance from a dictionary.

        Args:
            d (Mapping[str, Any]): Dictionary containing obstacle data.

        Returns:
            Obstacle: Created Obstacle instance.
        """
        coords = d.get("coords", ())
        return cls(coords=tuple(tuple(p) for p in coords))

    def to_dict(self) -> Mapping[str, Any]:
        """Converts the Obstacle instance to a dictionary.

        Returns:
            Mapping[str, Any]: Dictionary representation of the Obstacle.
        """
        return {
            "coords": [list(p) for p in self.coords],
        }

    def chk_overlap(self, poly: Polygon) -> bool:
        """Checks whether the obstacle overlaps with the given polygon.

        Args:
            poly (Polygon): Polygon to check overlap with.

        Returns:
            bool: True if the obstacle overlaps with the polygon, False otherwise.
        """
        if self._poly.is_empty or poly.is_empty:
            return False

        # TODO: Check whether this pruning is actually beneficial.
        minx1, miny1, maxx1, maxy1 = self._poly_bounds
        minx2, miny2, maxx2, maxy2 = poly.bounds
        if maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1:
            return False

        return self._poly_prep.intersects(poly)

    @property
    def poly(self) -> Polygon:
        """Polygon representing the obstacle.

        Returns:
            Polygon: Obstacle polygon.
        """
        return self._poly

    @staticmethod
    def _validate_coords(
        coords: Tuple[Tuple[float, float], ...],
    ) -> Tuple[Tuple[float, float], ...]:
        if not coords:
            return ()

        if len(coords) != 5:
            raise ValueError(
                f"coords must contain exactly 5 points. Got: {len(coords)}"
            )

        for i, p in enumerate(coords):
            if not (isinstance(p, tuple) and len(p) == 2):
                raise ValueError(f"coords[{i}] must be a pair (x, y). Got: {p}")

            x, y = p
            req_finite(f"coords[{i}].x", x)
            req_finite(f"coords[{i}].y", y)

        return coords
