"""Multi-scale ArUco detection + 6-DOF deck-centre pose via solvePnP.

Detects the deck's grid board (DICT_4X4) for the approach and its nested centre marker (DICT_5X5)
for the final descent, and returns the **deck-centre** position in the camera frame from whichever
fiducial is best conditioned. The grid board is preferred when enough of it is visible (more
accurate); the centre marker takes over once the board overflows the frame near touchdown. As long
as one fiducial is visible the platform stays observable, eliminating the touchdown blind spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from drone_landing.perception.board import BoardSpec, build_board
from drone_landing.perception.camera import CameraModel


@dataclass(frozen=True)
class ArucoConfig:
    board: BoardSpec = field(default_factory=BoardSpec)


@dataclass
class ArucoDetection:
    found: bool
    tvec_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))   # deck centre in OpenCV cam frame
    rvec_cam: np.ndarray = field(default_factory=lambda: np.zeros(3))
    centroid_px: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_markers: int = 0
    reproj_error: float = 0.0
    source: str = ""          # "grid" or "center"
    corners: list | None = None


class ArucoDetector:
    def __init__(self, camera: CameraModel, config: ArucoConfig | None = None):
        self.camera = camera
        self.config = config or ArucoConfig()
        spec = self.config.board
        self.board = build_board(spec)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._grid = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(spec.dict_id), params)
        self._center = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(spec.center_dict_id), params)
        h = spec.center_length / 2.0
        self._center_objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def detect(self, image: np.ndarray) -> ArucoDetection:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)

        # --- grid board (approach) ---
        corners, ids, _ = self._grid.detectMarkers(gray)
        if ids is not None and len(ids) >= 2:
            obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
            if obj_pts is not None and len(obj_pts) >= 4:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.camera.K, self.camera.dist,
                                              flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    center = (R @ self.config.board.center_obj.reshape(3, 1) + tvec).flatten()
                    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.camera.K, self.camera.dist)
                    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
                    centroid = np.concatenate(corners).reshape(-1, 2).mean(axis=0)
                    return ArucoDetection(True, center, rvec.flatten(), centroid, int(len(ids)),
                                          reproj, "grid", list(corners))

        # --- nested centre marker (final descent) ---
        c_corners, c_ids, _ = self._center.detectMarkers(gray)
        if c_ids is not None:
            ids_flat = c_ids.flatten()
            match = np.where(ids_flat == self.config.board.center_id)[0]
            if match.size:
                img_pts = c_corners[int(match[0])].reshape(4, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(self._center_objp, img_pts, self.camera.K,
                                              self.camera.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    proj, _ = cv2.projectPoints(self._center_objp, rvec, tvec, self.camera.K, self.camera.dist)
                    reproj = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_pts, axis=1)))
                    return ArucoDetection(True, tvec.flatten(), rvec.flatten(), img_pts.mean(axis=0),
                                          1, reproj, "center", [c_corners[int(match[0])]])

        return ArucoDetection(found=False)
