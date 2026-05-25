"""Base abstractions shared by all platform motion models (ground rover, ship deck, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlatformState:
    """Full 6-DOF kinematic state of the platform deck in the world frame.

    ``pos`` is the deck body origin, ``quat`` is body orientation (w, x, y, z), ``lin_vel`` is the
    world-frame linear velocity of the origin, and ``ang_vel`` is the world-frame angular velocity.
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "PlatformState":
        return PlatformState(
            pos=self.pos.copy(),
            quat=self.quat.copy(),
            lin_vel=self.lin_vel.copy(),
            ang_vel=self.ang_vel.copy(),
        )


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` radians about the world +Z axis."""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class PlatformMotion(ABC):
    """A prescribed, time-stepped platform motion model.

    Implementations must be deterministic given their seed so episodes are reproducible.
    """

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> PlatformState:
        """Reset internal state using ``rng`` and return the initial :class:`PlatformState`."""

    @abstractmethod
    def step(self, dt: float) -> PlatformState:
        """Advance the model by ``dt`` seconds and return the new :class:`PlatformState`."""
