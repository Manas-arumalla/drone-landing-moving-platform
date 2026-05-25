"""Moving-truck motion: a road vehicle cruising a smooth loop (the classic mobile-landing target).

Landing on a moving truck/car bed is a canonical autonomous-landing scenario (delivery, recovery, mobile
launch). Unlike the wandering rover (``RandomGroundMotion``), a truck on a road is **smooth and
predictable**: steady cruise speed, gentle continuous steering, level. We drive it along a smooth closed
**loop** (a large oval, like a truck circling a yard) so the path stays bounded *by construction* — no hard
wall reflections, which would snap the heading and fling a tracking drone off the target. Planar, constant
height — reuses the validated ground (3-DOF) world; yaw follows the heading like a steered vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState, yaw_to_quat


@dataclass(frozen=True)
class TruckMotionConfig:
    deck_z: float = 0.20            # m   flat bed height (constant)
    cruise_speed: float = 0.45      # m/s steady forward speed (a recovering truck slows; trackable)
    loop_radius: float = 2.6        # m   radius of the loop -> gentler curvature (v/radius) at this speed
    radius_ratio: float = 0.8       # minor/major axis ratio -> a gentle oval rather than a pure circle
    speed_jitter: float = 0.06      # m/s slow cruise-speed variation (traffic), as a fraction wobble


class TruckMotion(PlatformMotion):
    """Smooth road-cruising vehicle on a bounded oval loop: steady speed, gentle steering, level & planar."""

    def __init__(self, config: TruckMotionConfig | None = None):
        self.config = config or TruckMotionConfig()
        self._rng = np.random.default_rng()
        self._t = 0.0
        self._phase = 0.0
        self._dir = 1.0

    def reset(self, rng: np.random.Generator) -> PlatformState:
        self._rng = rng
        self._t = 0.0
        self._phase = float(rng.uniform(0, 2 * np.pi))
        self._dir = float(rng.choice([-1.0, 1.0]))         # clockwise or counter-clockwise loop
        return self.step(0.0)

    def step(self, dt: float) -> PlatformState:
        c = self.config
        self._t += dt
        a, b = c.loop_radius, c.loop_radius * c.radius_ratio
        # angular rate set so the tangential speed ~ cruise_speed (slow speed wobble for realism)
        speed = c.cruise_speed * (1.0 + c.speed_jitter * np.sin(0.37 * self._t))
        # mean ellipse circumference radius for the angular-rate conversion
        w = self._dir * speed / (0.5 * (a + b))
        th = self._phase + w * self._t
        # shift the loop so it STARTS at the origin (under the drone's spawn) -> immediate marker acquisition
        x = a * (np.cos(th) - np.cos(self._phase))
        y = b * (np.sin(th) - np.sin(self._phase))
        # analytic velocity along the ellipse (the constant shift drops out)
        vx, vy = -a * np.sin(th) * w, b * np.cos(th) * w
        sp = float(np.hypot(vx, vy))
        yaw = float(np.arctan2(vy, vx))
        # yaw rate ~ d/dt atan2(vy,vx); for a near-ellipse use the loop rate as a smooth proxy
        return PlatformState(
            pos=np.array([x, y, c.deck_z]),
            quat=yaw_to_quat(yaw),
            lin_vel=np.array([vx, vy, 0.0]),
            ang_vel=np.array([0.0, 0.0, w]),
        )
