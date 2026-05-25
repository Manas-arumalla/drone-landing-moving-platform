"""Ship-deck seakeeping motion: a wave-driven 6-DOF deck.

A ship on a seaway responds to waves with oscillatory heave (vertical), roll, and pitch, on top of
its forward surge along a heading. We model each oscillatory DOF as a sum of sinusoidal components
(amplitudes/periods set by a sea state) — the standard time-domain representation of wave-induced
ship motion via response-amplitude operators over a wave spectrum. This keeps the deck a true
contact surface whose *prescribed* motion is physically grounded (the ship is far heavier than the
drone), per the Realism Charter.

The dominant landing challenge this creates: the deck heaves and tilts, so a robust system must time
its touchdown to a low-motion ("green-deck") window — handled by the planner/MPC in a later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) from roll-pitch-yaw (ZYX) Euler angles."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


@dataclass(frozen=True)
class WaveComponent:
    amplitude: float   # m (heave) or rad (roll/pitch)
    period: float      # s


@dataclass(frozen=True)
class ShipMotionConfig:
    deck_z: float = 0.30            # m   mean deck height
    forward_speed: float = 0.4      # m/s ship surge along its heading
    heading: float = 0.0            # rad ship heading
    # sea-state response: a few components per DOF (sum of sinusoids). Defaults ~ moderate sea.
    heave: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(0.12, 5.0), WaveComponent(0.05, 3.1)))
    roll: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(np.deg2rad(6.0), 6.5), WaveComponent(np.deg2rad(2.0), 4.0)))
    pitch: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(np.deg2rad(4.0), 5.5), WaveComponent(np.deg2rad(1.5), 3.5)))
    sway: tuple[WaveComponent, ...] = field(default_factory=lambda: (
        WaveComponent(0.10, 7.0),))


def sea_state(name: str = "moderate", spectral: bool = False) -> ShipMotionConfig:
    """Named sea-state presets. ``moderate`` is the default seaway; ``rough`` roughly doubles the
    heave/roll/pitch (peak heave rate ~0.4 m/s, roll ±12°) so the green-deck timing matters more;
    ``calm`` is near-flat water (a sanity baseline where timing should be a no-op).

    ``spectral=True`` returns the **B1 spectral model** (JONSWAP/PM wave spectrum + RAOs), calibrated to
    the same RMS motion as these hand-tuned presets but with physically-correct frequency content."""
    if spectral:
        from drone_landing.sim.platforms.wave_spectrum import spectral_sea_state
        return spectral_sea_state(name)
    name = name.lower()
    if name == "moderate":
        return ShipMotionConfig()
    if name == "calm":
        return ShipMotionConfig(
            heave=(WaveComponent(0.02, 5.0),),
            roll=(WaveComponent(np.deg2rad(1.0), 6.5),),
            pitch=(WaveComponent(np.deg2rad(0.7), 5.5),),
            sway=(WaveComponent(0.02, 7.0),),
        )
    if name == "rough":
        return ShipMotionConfig(
            heave=(WaveComponent(0.22, 6.0), WaveComponent(0.10, 3.6)),
            roll=(WaveComponent(np.deg2rad(12.0), 7.0), WaveComponent(np.deg2rad(4.0), 4.2)),
            pitch=(WaveComponent(np.deg2rad(7.0), 6.0), WaveComponent(np.deg2rad(2.5), 3.8)),
            sway=(WaveComponent(0.18, 8.0),),
        )
    raise ValueError(f"unknown sea state '{name}' (use calm | moderate | rough)")


class ShipDeckMotion(PlatformMotion):
    """Wave-driven 6-DOF ship deck. Deterministic given the rng (random component phases)."""

    def __init__(self, config: ShipMotionConfig | None = None):
        self.config = config or ShipMotionConfig()
        self._rng = np.random.default_rng()
        self._t = 0.0
        self._x = 0.0
        self._y = 0.0
        self._phase: dict = {}

    def reset(self, rng: np.random.Generator) -> PlatformState:
        self._rng = rng
        self._t = 0.0
        self._x = 0.0
        self._y = 0.0
        c = self.config
        self._phase = {dof: rng.uniform(0, 2 * np.pi, size=len(comps))
                       for dof, comps in (("heave", c.heave), ("roll", c.roll),
                                          ("pitch", c.pitch), ("sway", c.sway))}
        return self.step(0.0)

    def _sum(self, dof: str, comps, t: float, deriv: bool = False) -> float:
        total = 0.0
        for i, comp in enumerate(comps):
            w = 2 * np.pi / comp.period
            ph = self._phase[dof][i]
            total += (comp.amplitude * w * np.cos(w * t + ph) if deriv
                      else comp.amplitude * np.sin(w * t + ph))
        return float(total)

    def step(self, dt: float) -> PlatformState:
        c = self.config
        self._t += dt
        t = self._t
        # forward surge along heading + wave-induced sway (lateral)
        self._x += c.forward_speed * np.cos(c.heading) * dt
        self._y += c.forward_speed * np.sin(c.heading) * dt
        sway = self._sum("sway", c.sway, t)
        sway_rate = self._sum("sway", c.sway, t, deriv=True)
        # lateral offset perpendicular to heading
        nx, ny = -np.sin(c.heading), np.cos(c.heading)
        x = self._x + sway * nx
        y = self._y + sway * ny
        z = c.deck_z + self._sum("heave", c.heave, t)
        roll = self._sum("roll", c.roll, t)
        pitch = self._sum("pitch", c.pitch, t)

        vx = c.forward_speed * np.cos(c.heading) + sway_rate * nx
        vy = c.forward_speed * np.sin(c.heading) + sway_rate * ny
        vz = self._sum("heave", c.heave, t, deriv=True)
        return PlatformState(
            pos=np.array([x, y, z]),
            quat=rpy_to_quat(roll, pitch, c.heading),
            lin_vel=np.array([vx, vy, vz]),
            ang_vel=np.array([self._sum("roll", c.roll, t, deriv=True),
                              self._sum("pitch", c.pitch, t, deriv=True), 0.0]),
        )

    def deck_motion_energy(self, horizon: float = 3.0, dt: float = 0.1) -> np.ndarray:
        """Forecast deck vertical speed magnitude over a short horizon (for green-deck timing).

        Returns sampled |heave rate| at times t..t+horizon without advancing the model.
        """
        c = self.config
        ts = self._t + np.arange(0.0, horizon, dt)
        out = []
        for tt in ts:
            v = 0.0
            for i, comp in enumerate(c.heave):
                w = 2 * np.pi / comp.period
                v += comp.amplitude * w * np.cos(w * tt + self._phase["heave"][i])
            out.append(abs(v))
        return np.array(out)
