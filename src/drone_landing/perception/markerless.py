"""Markerless deck tracker — a fallback when the ArUco code can't be decoded.

ArUco decoding fails when the markers are motion-blurred, partially out of frame, or too small/large,
even while the deck's bright fiducial pad is still clearly visible. This tracker recovers a
relative-position fix in that regime **without decoding any marker**: it segments the bright pad region
in the (nadir, gimbal-stabilized) downward image, takes its centroid, and back-projects that pixel
through the pinhole model + the rangefinder altitude to a deck-centre bearing → relative XY. Fused into
the EKF with a larger measurement noise than ArUco, it keeps the platform observable through brief
marker loss (the failure mode that previously caused go-arounds / fly-offs).

This is the dependency-light classical fallback (segmentation + back-projection); a learned (CNN)
detector trained on rendered decks is a natural future upgrade for cluttered real imagery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.perception.camera import CameraModel


@dataclass(frozen=True)
class MarkerlessConfig:
    bright_thresh: int = 235      # tight threshold isolates the saturated-white pad (its centroid is the
                                  # deck centre); a looser threshold dilutes it with the gray deck -> nadir bias
    min_area_frac: float = 0.002  # reject blobs smaller than this fraction of the image
    max_area_frac: float = 0.6    # reject huge blobs (pad fills the view -> likely clipped)
    border_margin: int = 4        # reject a pad blob touching the image edge (clipped -> biased centroid)


@dataclass
class MarkerlessDetection:
    found: bool
    rel_xy: np.ndarray            # deck-centre minus drone, world frame (nadir cam) [m]
    area_frac: float


class MarkerlessDeckTracker:
    """Segment the bright deck/pad blob and back-project its centroid to a relative-XY fix."""

    def __init__(self, camera: CameraModel, config: MarkerlessConfig | None = None):
        self.cam = camera
        self.cfg = config or MarkerlessConfig()

    def detect(self, image, range_m: float) -> MarkerlessDetection:
        import cv2

        c = self.cfg
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, c.bright_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return MarkerlessDetection(False, np.zeros(2), 0.0)
        biggest = max(contours, key=cv2.contourArea)
        area_frac = float(cv2.contourArea(biggest)) / (self.cam.width * self.cam.height)
        if not (c.min_area_frac <= area_frac <= c.max_area_frac):
            return MarkerlessDetection(False, np.zeros(2), area_frac)
        # Reject if the pad blob touches the image border: then it is clipped and its centroid is biased
        # (no longer the deck centre), which would corrupt the fix. Only fuse a fully-in-frame pad.
        bx, by, bw, bh = cv2.boundingRect(biggest)
        margin = c.border_margin
        if (bx <= margin or by <= margin
                or bx + bw >= self.cam.width - margin or by + bh >= self.cam.height - margin):
            return MarkerlessDetection(False, np.zeros(2), area_frac)
        m = cv2.moments(biggest)
        if m["m00"] <= 0:
            return MarkerlessDetection(False, np.zeros(2), area_frac)
        px, py = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # back-project the centroid through the nadir pinhole at altitude `range_m` (R_cam = identity):
        # world dir = OPENCV_TO_MUJOCO_CAM @ [(px-cx)/fx, (py-cy)/fy, 1]; the deck sits range_m below.
        rel_xy = np.array([range_m * (px - self.cam.cx) / self.cam.fx,
                           -range_m * (py - self.cam.cy) / self.cam.fy])
        return MarkerlessDetection(True, rel_xy, area_frac)
