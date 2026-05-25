"""Image-based visual servoing (IBVS) guidance for platform tracking.

Real moving-platform landing systems use IBVS because it is robust to depth/velocity-estimation
error — it acts on what the camera *measures*, not on a fragile fused 3-D velocity. Here, with the
stabilized nadir gimbal, the robust image-derived measurements are:

* relative position from the ArUco fiducial (position is well-anchored by vision), and
* relative velocity from the fiducial's **optical flow** (a direct image-plane measurement,
  validated at ~0.2 m/s vs truth).

The guidance law commands a horizontal acceleration

    a_xy = kp * r_xy + kd * v_flow_xy        (r = platform - drone, v_flow = platform - drone vel)

which feeds the geometric attitude inner loop (via ``GeometricController.a_xy_override``). Crucially,
the velocity term comes from optical flow — NOT the EKF's differentiated velocity, whose spikes were
the dominant cause of the fly-off divergences. This is the estimation-bottleneck fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IBVSGains:
    kp: float = 1.4       # image-position proportional gain
    kd: float = 2.0       # optical-flow (velocity) damping gain
    a_xy_max: float = 2.5  # m/s^2  horizontal accel cap (tilt limit, keeps marker in FOV)


class IBVSGuidance:
    def __init__(self, gains: IBVSGains | None = None):
        self.g = gains or IBVSGains()

    def horizontal_accel(self, rel_pos_xy: np.ndarray, v_flow_xy: np.ndarray) -> np.ndarray:
        """Return the horizontal acceleration command from image-derived position + optical flow."""
        a = self.g.kp * np.asarray(rel_pos_xy)[:2] + self.g.kd * np.asarray(v_flow_xy)[:2]
        n = float(np.linalg.norm(a))
        if n > self.g.a_xy_max:
            a *= self.g.a_xy_max / n
        return a
