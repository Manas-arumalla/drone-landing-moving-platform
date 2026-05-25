"""USV (unmanned surface vehicle) motion: a small, agile surface craft on a seaway.

Between a ship and a rover: a USV **maneuvers in-plane** (it turns far more than a big ship) *and*
**responds to waves** (a small hull rolls/pitches/heaves more, and at higher frequency, than a large deck).
Landing on one is harder than a ship deck — the target both translates *and* rocks. We drive the planar
base along a smooth bounded **loop** (no hard wall reflections, which would snap the heading and fling a
tracking drone off-target) and add a lively short-period seaway response on top. Reuses the validated 6-DOF
ship world (slide + roll/pitch/heave servos).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState
from drone_landing.sim.platforms.ship import WaveComponent, rpy_to_quat


@dataclass(frozen=True)
class USVMotionConfig:
    deck_z: float = 0.30
    cruise_speed: float = 0.4       # m/s nominal forward speed (agile, but trackable to touchdown)
    loop_radius: float = 2.4        # m   radius of the maneuvering loop -> gentler curvature
    radius_ratio: float = 0.75      # minor/major axis ratio -> an oval, not a pure circle
    # short-period small-hull seaway response (higher frequency + bigger roll than a ship deck)
    heave: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(0.10, 2.8), WaveComponent(0.05, 1.7)))
    roll: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(np.deg2rad(8.0), 3.0), WaveComponent(np.deg2rad(3.0), 1.9)))
    pitch: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(np.deg2rad(5.0), 2.6),))


class USVMotion(PlatformMotion):
    """Planar loop maneuvering + lively short-period wave response. Yaw follows the heading of travel."""

    def __init__(self, config: USVMotionConfig | None = None):
        self.config = config or USVMotionConfig()
        self._rng = np.random.default_rng()
        self._t = 0.0
        self._phase = 0.0
        self._dir = 1.0
        self._wphase: dict = {}

    def reset(self, rng: np.random.Generator) -> PlatformState:
        c = self.config
        self._rng = rng
        self._t = 0.0
        self._phase = float(rng.uniform(0, 2 * np.pi))
        self._dir = float(rng.choice([-1.0, 1.0]))
        self._wphase = {dof: rng.uniform(0, 2 * np.pi, size=len(comps))
                        for dof, comps in (("heave", c.heave), ("roll", c.roll), ("pitch", c.pitch))}
        return self.step(0.0)

    def _sum(self, dof: str, comps, deriv: bool = False) -> float:
        total = 0.0
        for i, comp in enumerate(comps):
            w = 2 * np.pi / comp.period
            ph = self._wphase[dof][i]
            total += (comp.amplitude * w * np.cos(w * self._t + ph) if deriv
                      else comp.amplitude * np.sin(w * self._t + ph))
        return float(total)

    def step(self, dt: float) -> PlatformState:
        c = self.config
        self._t += dt
        a, b = c.loop_radius, c.loop_radius * c.radius_ratio
        w = self._dir * c.cruise_speed / (0.5 * (a + b))
        th = self._phase + w * self._t
        # shift the loop so it STARTS at the origin (under the drone's spawn) -> immediate acquisition
        x = a * (np.cos(th) - np.cos(self._phase))
        y = b * (np.sin(th) - np.sin(self._phase))
        vx, vy = -a * np.sin(th) * w, b * np.cos(th) * w
        yaw = float(np.arctan2(vy, vx))

        z = c.deck_z + self._sum("heave", c.heave)
        vz = self._sum("heave", c.heave, deriv=True)
        roll = self._sum("roll", c.roll)
        pitch = self._sum("pitch", c.pitch)
        return PlatformState(
            pos=np.array([x, y, z]),
            quat=rpy_to_quat(roll, pitch, yaw),
            lin_vel=np.array([vx, vy, vz]),
            ang_vel=np.array([self._sum("roll", c.roll, deriv=True),
                              self._sum("pitch", c.pitch, deriv=True), w]),
        )
