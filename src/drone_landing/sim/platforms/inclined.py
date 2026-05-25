"""Inclined-deck motion: a persistently tilted landing surface (barge ramp / listing vessel / sloped pad).

Most landing targets are level on average; an **inclined deck** is not — a loaded barge ramp, a vessel
listing in a beam sea, or a sloped helipad presents a landing surface whose normal is tilted several
degrees off vertical. That is a genuinely different control problem: the geometric controller lands
*level*, so on a tilted deck it must still plant all four feet and not slide or tip. It also pairs
naturally with the reachability shield (`--shield`) — the safe touchdown envelope tightens on a slope.

We model a constant **mean tilt** (roll and/or pitch bias) plus an optional gentle seaway oscillation on
top, reusing the validated 6-DOF ship world (whose roll/pitch servos already hold the deck tilted). The
mean tilt is the new ingredient; the small oscillation keeps it from being a perfectly static target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState
from drone_landing.sim.platforms.ship import WaveComponent, rpy_to_quat


@dataclass(frozen=True)
class InclinedDeckConfig:
    deck_z: float = 0.30               # m   mean deck height
    incline_deg: float = 12.0          # deg total tilt magnitude of the surface normal off vertical
    tilt_heading: float = 0.0          # rad direction the deck slopes down toward (0 = +x downhill)
    drift_speed: float = 0.0           # m/s optional slow translation (a moving ramp / barge under tow)
    drift_heading: float = 0.0         # rad direction of the drift
    # gentle residual motion so the deck is not perfectly static (small relative to the mean tilt)
    heave: tuple[WaveComponent, ...] = field(default_factory=lambda: (WaveComponent(0.03, 5.0),))
    wobble: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(np.deg2rad(1.5), 6.0),))


def incline_preset(name: str = "moderate") -> InclinedDeckConfig:
    """Named inclines: ``gentle`` ~6 deg, ``moderate`` ~12 deg, ``steep`` ~18 deg (a hard, sliding deck)."""
    name = name.lower()
    if name == "gentle":
        return InclinedDeckConfig(incline_deg=6.0)
    if name == "moderate":
        return InclinedDeckConfig(incline_deg=12.0)
    if name == "steep":
        return InclinedDeckConfig(incline_deg=18.0, heave=(WaveComponent(0.04, 5.0),))
    raise ValueError(f"unknown incline '{name}' (use gentle | moderate | steep)")


class InclinedDeckMotion(PlatformMotion):
    """A deck with a persistent tilt (surface normal off vertical) plus a gentle residual oscillation."""

    def __init__(self, config: InclinedDeckConfig | None = None):
        self.config = config or InclinedDeckConfig()
        self._rng = np.random.default_rng()
        self._t = 0.0
        self._x = 0.0
        self._y = 0.0
        self._phase_h = np.zeros(0)
        self._phase_w = np.zeros(0)

    def reset(self, rng: np.random.Generator) -> PlatformState:
        c = self.config
        self._rng = rng
        self._t = 0.0
        self._x = 0.0
        self._y = 0.0
        self._phase_h = rng.uniform(0, 2 * np.pi, size=len(c.heave))
        self._phase_w = rng.uniform(0, 2 * np.pi, size=len(c.wobble))
        return self.step(0.0)

    @staticmethod
    def _sum(comps, phase, t: float, deriv: bool = False) -> float:
        total = 0.0
        for i, comp in enumerate(comps):
            w = 2 * np.pi / comp.period
            total += (comp.amplitude * w * np.cos(w * t + phase[i]) if deriv
                      else comp.amplitude * np.sin(w * t + phase[i]))
        return float(total)

    def step(self, dt: float) -> PlatformState:
        c = self.config
        self._t += dt
        t = self._t
        self._x += c.drift_speed * np.cos(c.drift_heading) * dt
        self._y += c.drift_speed * np.sin(c.drift_heading) * dt

        incline = np.deg2rad(c.incline_deg)
        # split the mean tilt into roll/pitch so the surface normal slopes downhill toward tilt_heading
        # (small-angle: normal horizontal component ~ (pitch, -roll)); a gentle wobble rocks the pitch axis
        wob = self._sum(c.wobble, self._phase_w, t)
        roll = incline * (-np.sin(c.tilt_heading))
        pitch = incline * np.cos(c.tilt_heading) + wob
        z = c.deck_z + self._sum(c.heave, self._phase_h, t)
        vz = self._sum(c.heave, self._phase_h, t, deriv=True)

        return PlatformState(
            pos=np.array([self._x, self._y, z]),
            quat=rpy_to_quat(roll, pitch, 0.0),
            lin_vel=np.array([c.drift_speed * np.cos(c.drift_heading),
                              c.drift_speed * np.sin(c.drift_heading), vz]),
            ang_vel=np.zeros(3),
        )
