"""Data-driven deck motion: replay a recorded 6-DOF ship-motion time series (P1.2).

Where ``ShipDeckMotion`` *synthesizes* deck motion from a wave spectrum, this model *replays* a measured
or recorded 6-DOF trajectory — so the deck can be driven by **real seakeeping data** (a sea-trial / motion-
reference-unit log, or NDBC-wave-derived heave) instead of a model. The CSV has columns
``t, x, y, z, roll, pitch, yaw`` (SI; angles in rad); the model linearly interpolates pose and finite-
differences velocities, looping when the log ends. It conforms to :class:`PlatformMotion`, so it is a
drop-in for any ship/offshore scenario.

Honesty: a *real* sea-trial CSV plugs straight in. The bundled reference produced by
``scripts/gen_seakeeping_data.py`` is **high-fidelity spectrum-generated** (JONSWAP + RAOs, many
components, fine dt) — a realistic stand-in, clearly labelled as such, not measured data. The point of
P1.2 is the *capability* to drive the deck from recorded data + the validation that the motion matches a
named sea state (P1.3).
"""

from __future__ import annotations

import numpy as np

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState
from drone_landing.sim.platforms.ship import rpy_to_quat

COLUMNS = ("t", "x", "y", "z", "roll", "pitch", "yaw")


class DataDrivenDeckMotion(PlatformMotion):
    """Replays a recorded 6-DOF deck trajectory (looping). Deterministic; ``rng`` only sets the start phase."""

    def __init__(self, times: np.ndarray, motion: np.ndarray, loop: bool = True,
                 random_start: bool = True, relative_xy: bool = True):
        self.t_data = np.asarray(times, dtype=float)
        self.motion = np.asarray(motion, dtype=float)        # (N, 6): x,y,z,roll,pitch,yaw
        if self.motion.shape[0] != self.t_data.shape[0] or self.motion.shape[1] != 6:
            raise ValueError("motion must be (N,6) aligned with times (N,)")
        self.duration = float(self.t_data[-1] - self.t_data[0])
        self.loop = loop
        self.random_start = random_start
        self.relative_xy = relative_xy   # start the deck at the world-origin XY (strip accumulated surge)
        self._t = 0.0
        self._t0 = float(self.t_data[0])
        self._xy_off = np.zeros(2)

    @classmethod
    def from_csv(cls, path: str, **kw) -> "DataDrivenDeckMotion":
        """Load ``t,x,y,z,roll,pitch,yaw`` (header optional). Real trial logs or generated references."""
        raw = np.genfromtxt(path, delimiter=",", names=None, comments="#")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        # tolerate a header row (non-numeric -> NaN) by dropping all-NaN rows
        raw = raw[~np.isnan(raw).any(axis=1)]
        return cls(raw[:, 0], raw[:, 1:7], **kw)

    def _sample(self, t: float) -> np.ndarray:
        """Interpolated [x,y,z,roll,pitch,yaw] at absolute log-time ``t`` (looped)."""
        if self.loop and self.duration > 0:
            t = self._t0 + (t - self._t0) % self.duration
        return np.array([np.interp(t, self.t_data, self.motion[:, k]) for k in range(6)])

    def _state(self, t: float) -> PlatformState:
        m = self._sample(t)
        m[:2] -= self._xy_off                                          # start at world-origin XY
        h = 1e-3
        dm = (self._sample(t + h) - self._sample(t - h)) / (2 * h)     # finite-difference velocities
        return PlatformState(pos=m[:3], quat=rpy_to_quat(m[3], m[4], m[5]),
                             lin_vel=dm[:3], ang_vel=dm[3:6])

    def reset(self, rng: np.random.Generator) -> PlatformState:
        self._t = self._t0 + (float(rng.uniform(0, self.duration)) if self.random_start else 0.0)
        self._xy_off = self._sample(self._t)[:2] if self.relative_xy else np.zeros(2)
        return self._state(self._t)

    def step(self, dt: float) -> PlatformState:
        self._t += dt
        return self._state(self._t)
