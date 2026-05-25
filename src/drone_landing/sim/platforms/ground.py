"""Randomized ground-rover motion: a smooth, jerk-limited random trajectory.

The model drives a desired velocity with an Ornstein-Uhlenbeck process, then tracks it with
acceleration- and jerk-limited dynamics so the path looks like a real wheeled vehicle wandering
within bounds (no teleporting, no instantaneous velocity changes). Yaw optionally aligns with the
direction of travel, rate-limited like a steered vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState, yaw_to_quat


@dataclass(frozen=True)
class GroundMotionConfig:
    # Default = a realistic ground vehicle cruising for a landing (smooth, predictable). Real landing
    # targets (cars, boats) move this way; the autopilot lands reliably on it. The adversarial fast
    # random rover (v_max 1.5, a_max 0.8, jerk 2.0) remains available as a stress test via overrides.
    v_max: float = 0.7          # m/s   top speed
    a_max: float = 0.3          # m/s^2 max acceleration magnitude
    jerk_max: float = 1.0       # m/s^3 max rate-of-change of acceleration
    vel_tau: float = 1.0        # s     how quickly accel drives v toward v_des
    vdes_tau: float = 3.5       # s     OU correlation time of desired velocity
    vdes_std_frac: float = 0.55  # desired-velocity std as a fraction of v_max
    bounds: float = 5.0         # m     keep within +/- bounds in x and y
    boundary_margin: float = 0.7  # fraction of bounds where inward bias kicks in
    deck_z: float = 0.20        # m     deck body origin height (constant for ground)
    align_yaw_to_heading: bool = True
    yaw_rate_max: float = 0.9   # rad/s
    yaw_speed_threshold: float = 0.2  # m/s below which heading is not updated


class RandomGroundMotion(PlatformMotion):
    """Smooth bounded random-walk motion for a ground platform."""

    def __init__(self, config: GroundMotionConfig | None = None):
        self.config = config or GroundMotionConfig()
        self._rng = np.random.default_rng()
        self._p = np.zeros(2)
        self._v = np.zeros(2)
        self._a = np.zeros(2)
        self._v_des = np.zeros(2)
        self._yaw = 0.0

    def reset(self, rng: np.random.Generator) -> PlatformState:
        self._rng = rng
        self._p = np.zeros(2)
        self._v = rng.uniform(-0.2, 0.2, size=2)
        self._a = np.zeros(2)
        # Start the desired velocity somewhere reasonable within the speed envelope.
        self._v_des = rng.uniform(-1.0, 1.0, size=2) * (0.5 * self.config.v_max)
        self._yaw = float(rng.uniform(-np.pi, np.pi))
        return self._state()

    def step(self, dt: float) -> PlatformState:
        c = self.config

        # --- Ornstein-Uhlenbeck desired velocity ---
        std = c.vdes_std_frac * c.v_max
        sigma = std * np.sqrt(2.0 / c.vdes_tau)
        self._v_des += (-self._v_des / c.vdes_tau) * dt + sigma * np.sqrt(dt) * self._rng.standard_normal(2)

        # Steer back toward the center when approaching the boundary.
        edge = c.boundary_margin * c.bounds
        for i in range(2):
            if abs(self._p[i]) > edge:
                self._v_des[i] -= np.sign(self._p[i]) * c.v_max * dt / max(c.vel_tau, 1e-3)
        self._v_des = self._clip_norm(self._v_des, c.v_max)

        # --- jerk-limited acceleration tracking the desired velocity ---
        a_cmd = self._clip_norm((self._v_des - self._v) / c.vel_tau, c.a_max)
        da = a_cmd - self._a
        max_da = c.jerk_max * dt
        da_norm = np.linalg.norm(da)
        if da_norm > max_da:
            da *= max_da / da_norm
        self._a = self._clip_norm(self._a + da, c.a_max)

        # --- integrate, with a hard wall reflection as a safety net ---
        self._v = self._clip_norm(self._v + self._a * dt, c.v_max)
        self._p = self._p + self._v * dt
        for i in range(2):
            if abs(self._p[i]) > c.bounds:
                self._p[i] = np.sign(self._p[i]) * c.bounds
                self._v[i] = -0.5 * self._v[i]
                self._a[i] = 0.0

        # --- yaw aligned to heading (rate limited) ---
        yaw_rate = 0.0
        speed = float(np.linalg.norm(self._v))
        if c.align_yaw_to_heading and speed > c.yaw_speed_threshold:
            target_yaw = float(np.arctan2(self._v[1], self._v[0]))
            err = (target_yaw - self._yaw + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = float(np.clip(err / dt, -c.yaw_rate_max, c.yaw_rate_max))
            self._yaw += yaw_rate * dt

        return self._state(yaw_rate)

    def _state(self, yaw_rate: float = 0.0) -> PlatformState:
        return PlatformState(
            pos=np.array([self._p[0], self._p[1], self.config.deck_z]),
            quat=yaw_to_quat(self._yaw),
            lin_vel=np.array([self._v[0], self._v[1], 0.0]),
            ang_vel=np.array([0.0, 0.0, yaw_rate]),
        )

    @staticmethod
    def _clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm > limit and norm > 0.0:
            return vec * (limit / norm)
        return vec
