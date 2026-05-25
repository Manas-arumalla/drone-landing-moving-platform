"""Pinhole camera model matching a MuJoCo fixed camera.

MuJoCo renders an ideal pinhole image whose vertical field of view is the camera's ``fovy``. This
module derives the OpenCV-style intrinsics (K, zero distortion) and provides the conversion between
the OpenCV camera frame (x-right, y-down, z-forward) used by ``solvePnP`` and the MuJoCo camera
frame (x-right, y-up, z-backward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Maps a vector from the OpenCV camera frame to the MuJoCo camera frame (flip y and z).
OPENCV_TO_MUJOCO_CAM = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fovy_deg: float

    @property
    def fy(self) -> float:
        return (self.height / 2.0) / np.tan(np.deg2rad(self.fovy_deg) / 2.0)

    @property
    def fx(self) -> float:
        # square pixels; MuJoCo's horizontal fov follows from the aspect ratio
        return self.fy

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        return np.zeros(5)

    def opencv_to_world(self, vec_cam_opencv: np.ndarray, R_cam_world: np.ndarray) -> np.ndarray:
        """Rotate a vector from the OpenCV camera frame into the world frame.

        ``R_cam_world`` is the MuJoCo camera-to-world rotation (from the onboard attitude estimate in
        the deployed pipeline, or ``data.cam_xmat`` when validating against truth).
        """
        return R_cam_world @ (OPENCV_TO_MUJOCO_CAM @ vec_cam_opencv)
