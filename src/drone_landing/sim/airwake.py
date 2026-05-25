"""Ship air-wake ("burble"): the turbulent air behind the superstructure (B1).

The hardest part of a real shipboard landing is not the deck motion but the **air-wake** — the
separated, turbulent flow shed by the ship's superstructure that sits right over the landing spot. An
aircraft on short final flies into a region of mean downdraft + horizontal velocity deficit and sharp
turbulence ("the burble"), which is the dominant disturbance close to the deck.

We model it as a **position-dependent wind force** on the drone, switched on only for the ship scenario:

* a spatial **envelope** ``g(pos)`` that is strongest just above the deck and decays with height and with
  horizontal distance from the landing spot (the wake is local to the ship);
* a **mean burble** inside the envelope — a downdraft plus a horizontal deficit along the ship heading;
* **turbulence** — a per-axis Ornstein-Uhlenbeck process whose intensity is modulated by ``g`` (so it
  ramps up exactly where the drone is most vulnerable, on touchdown final).

This is a low-order surrogate (not CFD), but it captures the right phenomenology — disturbance that grows
as the drone commits to the deck — and it stresses the controller where it matters. It composes with the
ambient wind already in the world and is countered measurably by the disturbance-observer (`--dob`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirwakeConfig:
    z_wake: float = 1.2          # m   height above the deck over which the wake decays
    r_wake: float = 1.4         # m   horizontal radius of the wake region around the landing spot
    # Calibrated so the burble measurably stresses the approach (~92% -> ~58% on the moderate-sea
    # baseline) while staying recoverable by the disturbance observer (--dob). Scale up for a severe wake.
    burble_down: float = 1.8    # N   mean downdraft force at the deck (drone weight ~13.5 N)
    deficit: float = 1.2        # N   mean horizontal velocity-deficit force along the ship heading
    turb_std: float = 1.8       # N   peak turbulence std (at the deck), per axis
    turb_tau: float = 0.5       # s   turbulence correlation time (shorter than ambient gusts -> sharper)


class ShipAirwake:
    """Position-dependent air-wake disturbance force [N], with its own OU turbulence state."""

    def __init__(self, config: AirwakeConfig | None = None):
        self.cfg = config or AirwakeConfig()
        self._turb = np.zeros(3)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._turb = np.zeros(3)

    def envelope(self, drone_pos: np.ndarray, deck_pos: np.ndarray) -> float:
        """Wake strength in [0, 1]: strongest just above the deck, decaying with height and radius."""
        c = self.cfg
        rel = np.asarray(drone_pos, float) - np.asarray(deck_pos, float)
        zr = max(rel[2], 0.0)
        rho = float(np.hypot(rel[0], rel[1]))
        return float(np.exp(-zr / c.z_wake) * np.exp(-(rho / c.r_wake) ** 2))

    def force(self, drone_pos: np.ndarray, deck_pos: np.ndarray, heading: float,
              rng: np.random.Generator, dt: float) -> np.ndarray:
        """Air-wake force [N] on the drone at ``drone_pos`` given the deck pose, advanced ``dt``."""
        c = self.cfg
        g = self.envelope(drone_pos, deck_pos)
        # mean burble: downdraft + horizontal deficit along the ship heading
        mean = np.array([-c.deficit * np.cos(heading) * g,
                         -c.deficit * np.sin(heading) * g,
                         -c.burble_down * g])
        # turbulence: OU process, intensity modulated by the envelope
        tau = max(c.turb_tau, 1e-3)
        a = np.exp(-dt / tau)
        sigma = (c.turb_std * g) * np.sqrt(1.0 - a * a)
        self._turb = a * self._turb + sigma * rng.standard_normal(3)
        return mean + self._turb
