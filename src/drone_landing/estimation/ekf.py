"""Relative-state Kalman filter for vision-based landing.

State (world frame):  x = [r, v_rel]  where
  r     = platform_pos - drone_pos      (relative position)
  v_rel = platform_vel - drone_vel      (relative velocity)

The drone's own acceleration ``a_d`` (from the IMU + AHRS attitude) is a known input; the platform's
acceleration is unknown and modelled as process noise (constant-velocity platform). Measurements:

  * ArUco  -> r            (relative position rotated into the world frame)
  * range  -> r_z = -range (downward distance to the deck)

Because every quantity is *relative*, the absolute drone position never appears, so no GPS is
needed and the state is fully observable from the downward camera. This mirrors a real precision-
landing target estimator. The dynamics are linear given ``a_d``, so this is a linear KF; the EKF name
anticipates the nonlinear (ship-motion) platform model added later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (body->world) from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def accel_world(accel_body: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Drone inertial acceleration in the world frame from the accelerometer specific force.

    The accelerometer reads specific force f = R^T (a - g); hence a = R f + g.
    """
    return quat_to_rotmat(quat) @ np.asarray(accel_body) + GRAVITY


@dataclass(frozen=True)
class EKFConfig:
    sigma_a_platform: float = 2.0   # m/s^2  unmodelled platform acceleration (process noise)
    sigma_a_imu: float = 0.5        # m/s^2  drone-acceleration input noise
    r_aruco: float = 0.05           # m      ArUco relative-position measurement std
    r_range: float = 0.02           # m      rangefinder std
    r_flow: float = 0.30            # m/s    optical-flow relative-velocity std
    flow_gate: float = 1.0          # m/s    reject flow velocity updates beyond this innovation
    r_markerless: float = 0.15      # m      markerless deck-centroid horizontal-position std (coarser than ArUco)
    markerless_gate_m: float = 0.8  # m      reject markerless updates beyond this innovation
    aruco_gate_m: float = 0.5       # reject ArUco updates whose innovation exceeds this (catches
                                    # solvePnP planar pose-ambiguity flips) [m]
    init_pos_std: float = 1.5       # m
    init_vel_std: float = 0.4       # m/s  (low -> smooth velocity convergence, avoids init overshoot)


class RelativeStateEKF:
    def __init__(self, config: EKFConfig | None = None):
        self.config = config or EKFConfig()
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False
        self.reset()

    def reset(self, r0: np.ndarray | None = None, v0: np.ndarray | None = None) -> None:
        self.x = np.zeros(6)
        if r0 is not None:
            self.x[:3] = r0
        if v0 is not None:
            self.x[3:] = v0
        c = self.config
        self.P = np.diag([c.init_pos_std**2] * 3 + [c.init_vel_std**2] * 3)
        self.initialized = r0 is not None

    # ------------------------------------------------------------------ predict
    def predict(self, dt: float, a_d_world: np.ndarray | None = None) -> None:
        """Propagate by ``dt``. Constant-velocity by default; pass ``a_d_world`` to feed the
        IMU-measured drone acceleration as a known input (v_rel_dot = a_platform - a_drone)."""
        I = np.eye(3)
        F = np.block([[I, dt * I], [np.zeros((3, 3)), I]])
        B = np.vstack([0.5 * dt**2 * I, dt * I])
        if a_d_world is None:
            sigma_a2 = self.config.sigma_a_platform**2
            self.x = F @ self.x
        else:
            sigma_a2 = self.config.sigma_a_platform**2 + self.config.sigma_a_imu**2
            self.x = F @ self.x + (B @ (-np.asarray(a_d_world)))
        Q = (B @ B.T) * sigma_a2
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------ updates
    def update_aruco(self, r_meas: np.ndarray) -> bool:
        """Fuse an ArUco relative-position measurement. Returns False if gated as an outlier."""
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        R = np.eye(3) * self.config.r_aruco**2
        return self._update(np.asarray(r_meas), H, R, gate_m=self.config.aruco_gate_m)

    def update_range(self, range_m: float) -> bool:
        H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        R = np.array([[self.config.r_range**2]])
        return self._update(np.array([-range_m]), H, R)  # r_z = -range

    def update_markerless(self, rel_xy: np.ndarray) -> bool:
        """Fuse a markerless deck-centroid horizontal-position fix (coarser than ArUco). Returns False
        if gated. Keeps the platform observable when the ArUco code cannot be decoded but the pad is
        still visible."""
        H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        R = np.eye(2) * self.config.r_markerless**2
        return self._update(np.asarray(rel_xy, dtype=float), H, R, gate_m=self.config.markerless_gate_m)

    def update_velocity_xy(self, v_xy: np.ndarray) -> bool:
        """Fuse a direct horizontal relative-velocity measurement (e.g. from optical flow)."""
        H = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        R = np.eye(2) * self.config.r_flow**2
        return self._update(np.asarray(v_xy, dtype=float), H, R, gate_m=self.config.flow_gate)

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, gate_m: float | None = None) -> bool:
        y = z - H @ self.x
        if gate_m is not None and self.initialized and float(np.linalg.norm(y)) > gate_m:
            return False  # innovation too large -> reject (e.g. solvePnP pose-ambiguity flip)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return True

    # ----------------------------------------------------------------- outputs
    @property
    def rel_pos(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def rel_vel(self) -> np.ndarray:
        return self.x[3:].copy()
