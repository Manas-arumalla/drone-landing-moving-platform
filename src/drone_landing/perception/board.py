"""Shared multi-scale ArUco fiducial geometry for the deck.

The deck carries two co-centred fiducials so the platform stays observable across the *entire*
descent:

* a **grid board** (DICT_4X4_50) spanning ~0.6 m, with large separations leaving a clear centre gap
  — detected from the ~2 m start altitude;
* a **small centre marker** (DICT_5X5_50) in that gap — stays fully within the camera FOV down to
  touchdown, when the big board has long overflowed the frame.

Both are concentric on the deck centre, so each yields the same deck-centre position. The texture
generator and the detector both import this single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    dict_id: int = cv2.aruco.DICT_4X4_50
    markers_x: int = 4
    markers_y: int = 4
    marker_length: float = 0.08       # m
    marker_separation: float = 0.10   # m  large gap leaves room for the centre marker
    deck_size: float = 1.0            # deck top-face width the texture maps onto
    # small centre marker (different dictionary so ids never collide with the grid)
    center_dict_id: int = cv2.aruco.DICT_5X5_50
    center_id: int = 0
    center_length: float = 0.05       # m

    @property
    def span(self) -> float:
        return self.markers_x * self.marker_length + (self.markers_x - 1) * self.marker_separation

    @property
    def center_obj(self) -> np.ndarray:
        """Board centre in the grid board's object frame (origin at the bottom-left corner)."""
        return np.array([self.span / 2.0, self.span / 2.0, 0.0])


def build_board(spec: BoardSpec) -> "cv2.aruco.GridBoard":
    aruco_dict = cv2.aruco.getPredefinedDictionary(spec.dict_id)
    return cv2.aruco.GridBoard((spec.markers_x, spec.markers_y),
                               spec.marker_length, spec.marker_separation, aruco_dict)
