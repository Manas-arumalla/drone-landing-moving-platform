"""Disturbance observer (DOB) for wind-aware control.

Estimates the lumped external acceleration acting on the drone (wind gusts + unmodeled aero) by
comparing the acceleration the commanded thrust *should* produce with the acceleration the IMU actually
measures. The estimate is low-pass filtered and fed forward in the controller to cancel the disturbance:

    a_expected = (thrust_cmd / m) · b_z + g          (what we commanded)
    d_hat     <- lowpass( a_measured - a_expected )   (lumped external accel, world frame)

Feeding ``-m·d_hat`` into the desired force makes the net acceleration track the command even under a
steady breeze or a gust — and it reacts faster than the controller's slow integral term (which mainly
cancels constant bias). This is the standard acceleration-residual DOB; it needs only the IMU and the
known commanded thrust, so it adds no new sensor.
"""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


class DisturbanceObserver:
    def __init__(self, mass: float, tau: float = 0.25, control_dt: float = 0.01):
        self.m = float(mass)
        self.tau = float(tau)
        self.dt = float(control_dt)
        self.reset()

    def reset(self) -> None:
        self.d_hat = np.zeros(3)        # estimated external acceleration (world frame)

    def update(self, accel_world: np.ndarray, thrust_cmd: float, body_z: np.ndarray) -> np.ndarray:
        """Update and return the disturbance-acceleration estimate (world frame).

        ``accel_world`` is the IMU specific force mapped to the world inertial frame (a = R f + g);
        ``thrust_cmd`` is the last total commanded thrust [N]; ``body_z`` is the drone's body-z axis.
        """
        a_expected = (thrust_cmd / self.m) * np.asarray(body_z) + GRAVITY
        residual = np.asarray(accel_world) - a_expected
        alpha = self.dt / (self.tau + self.dt)
        self.d_hat = self.d_hat + alpha * (residual - self.d_hat)
        return self.d_hat
