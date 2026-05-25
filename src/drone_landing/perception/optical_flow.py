"""Downward optical-flow velocity sensor (deck-masked dense flow).

Measures the drone's horizontal velocity relative to the deck, the way real GPS-denied
precision-landing drones do (flow sensor + rangefinder). Dense optical flow is computed over the
whole frame but averaged only within the fiducial's bounding box — guaranteed to lie on the deck, so
it is not contaminated by the static ground around the deck at altitude. Averaging flow *vectors*
(not differencing a centroid) is stable even as individual markers enter/leave the frame. The gyro
removes the rotation-induced component; the rangefinder gives the metric scale. Fused into the
Kalman filter as a direct relative-velocity measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drone_landing.perception.camera import OPENCV_TO_MUJOCO_CAM, CameraModel


@dataclass
class FlowConfig:
    min_range: float = 0.05
    max_range: float = 4.0
    gyro_comp: float = 1.0
    max_speed: float = 4.0
    min_box_px: int = 24      # minimum bounding-box size to trust the flow


class OpticalFlowVelocity:
    def __init__(self, camera: CameraModel, config: FlowConfig | None = None):
        self.cam = camera
        self.cfg = config or FlowConfig()

    def estimate(self, prev_gray, curr_gray, bbox, range_m: float,
                 gyro_body: np.ndarray, R_cam: np.ndarray, dt: float):
        """Return (rel_vel_xy_world, valid). ``bbox`` = (x0, y0, x1, y1) of the fiducial."""
        if not (self.cfg.min_range < range_m < self.cfg.max_range) or dt <= 0 or bbox is None:
            return np.zeros(2), False
        h, w = prev_gray.shape
        x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
        x1 = min(w, int(bbox[2])); y1 = min(h, int(bbox[3]))
        if (x1 - x0) < self.cfg.min_box_px or (y1 - y0) < self.cfg.min_box_px:
            return np.zeros(2), False

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        region = flow[y0:y1, x0:x1].reshape(-1, 2)
        du, dv = float(np.median(region[:, 0])), float(np.median(region[:, 1]))  # px over dt

        # remove rotation-induced flow (pitch rate -> u, roll rate -> v); signs set by validation
        g = self.cfg.gyro_comp
        du -= g * self.cam.fx * (gyro_body[1]) * dt
        dv -= g * self.cam.fy * (gyro_body[0]) * dt

        # the deck moves opposite to the drone in the image; metric scale via range
        vx = -du * range_m / self.cam.fx / dt
        vy = -dv * range_m / self.cam.fy / dt
        v_drone_rel_deck_cam = np.array([vx, vy, 0.0])
        v_world = R_cam @ (OPENCV_TO_MUJOCO_CAM @ v_drone_rel_deck_cam)
        rel_vel = -v_world[:2]   # platform - drone
        if np.linalg.norm(rel_vel) > self.cfg.max_speed:
            return np.zeros(2), False
        return rel_vel, True

    @staticmethod
    def bbox_from_corners(corners) -> np.ndarray | None:
        """Bounding box (x0, y0, x1, y1) enclosing all detected marker corners."""
        if not corners:
            return None
        pts = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0)
        return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
