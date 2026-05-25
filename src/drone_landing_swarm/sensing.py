"""Onboard sensing + communication model for a realistic, no-cheats swarm.

The coordinator must not read ground truth. This layer turns the simulator's true states into the
*onboard view* each drone actually has:

* **own state** — a noisy estimate (the drone's own EKF, modeled as truth + calibrated noise);
* **deck pose** — a per-drone noisy estimate (its own camera→EKF of the deck; fused across drones by the
  consensus layer in A2);
* **neighbours** — only those within **comms range**, received as **broadcast estimates** subject to
  **latency** (one step stale) and **dropout** (lost packets), plus relative-sensing noise.

The drones' *true* motion is still pure physics; only the *decisions* (scheduling, CBF, holding) use
these estimates — so the coordination loop is audit-clean (no truth). A ``perfect`` preset (zero noise,
full comms) reproduces the old truth baseline for honest A/B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensingConfig:
    pos_noise: float = 0.05       # m     own position-estimate std (EKF-grade)
    vel_noise: float = 0.10       # m/s   own velocity-estimate std
    deck_noise: float = 0.06      # m     per-drone deck-position estimate std (base, at close range)
    rel_noise: float = 0.08       # m     inter-drone relative-position sensing std (UWB + vision)
    latency_steps: int = 1        # neighbour broadcasts arrive this many steps stale
    dropout_p: float = 0.05       # per-step probability a neighbour broadcast is lost
    # Heterogeneous deck observation (used by the A2 consensus estimator; the flat `deck` channel that
    # A1 uses is unaffected). A far/high drone sees the deck marker worse; past `deck_vis_range` it gets
    # no direct fix at all and must rely on consensus from better-placed neighbours.
    deck_range_gain: float = 0.05  # deck-measurement std grows this much (m) per metre of range
    deck_vis_range: float = 6.0    # m   beyond this the drone cannot resolve the deck (no measurement)

    @staticmethod
    def perfect() -> "SensingConfig":
        return SensingConfig(0.0, 0.0, 0.0, 0.0, 0, 0.0, deck_range_gain=0.0,
                             deck_vis_range=float("inf"))


class SwarmSensing:
    """Produces each drone's onboard view (estimates + delayed/dropped neighbour broadcasts)."""

    def __init__(self, n_drones: int, config: SensingConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.n = n_drones
        self.cfg = config or SensingConfig()
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        # ring buffer of past broadcasts (each entry: dict j -> (pos_est, vel_est)) for latency
        self._buffer: list[dict] = []

    def sense(self, true_pos: dict, true_vel: dict, deck_pos: np.ndarray, deck_vel: np.ndarray,
              landed: set, comms_range: float) -> dict:
        """Return the onboard view: own estimates, per-drone deck estimate, and neighbour observations.

        ``{ 'own_pos','own_vel','deck' : {i: ndarray}, 'neighbors': {i: {j: (pos,vel)}} }`` — all
        estimates (no truth). Neighbour entries come from delayed, possibly-dropped broadcasts in range.
        """
        c = self.cfg
        rng = self.rng
        active = [i for i in range(self.n) if i not in landed]

        own_pos = {i: true_pos[i] + rng.normal(0, c.pos_noise, 3) for i in active}
        own_vel = {i: true_vel[i] + rng.normal(0, c.vel_noise, 3) for i in active}
        deck_est = {i: deck_pos + rng.normal(0, c.deck_noise, 3) for i in active}
        deck_vel_est = {i: deck_vel + rng.normal(0, c.vel_noise, 3) for i in active}

        # Heterogeneous deck measurements for the consensus layer (A2): std grows with range; out of
        # visibility range -> no fix (None). The noise magnitude depends on geometry (physical), and the
        # drone knows its own std from its own range estimate -> still no truth leak into a decision.
        deck_meas: dict[int, np.ndarray | None] = {}
        deck_std: dict[int, float] = {}
        for i in active:
            rng_i = float(np.linalg.norm(true_pos[i] - deck_pos))
            std_i = c.deck_noise + c.deck_range_gain * rng_i
            deck_std[i] = std_i
            if rng_i > c.deck_vis_range:
                deck_meas[i] = None                          # marker unresolved at this range
            else:
                deck_meas[i] = deck_pos + rng.normal(0, max(std_i, 1e-9), 3)

        # current broadcasts = each drone's own estimate; neighbours receive the *stale* buffered one
        current = {i: (own_pos[i], own_vel[i]) for i in active}
        src = self._buffer[0] if (c.latency_steps > 0 and len(self._buffer) >= c.latency_steps) else current

        neighbors: dict[int, dict] = {}
        for i in active:
            seen = {}
            for j in active:
                if j == i or j not in src:
                    continue
                if np.linalg.norm(true_pos[j] - true_pos[i]) > comms_range:
                    continue                                  # out of comms/sensing range
                if rng.random() < c.dropout_p:
                    continue                                  # lost packet
                pj, vj = src[j]
                seen[j] = (pj + rng.normal(0, c.rel_noise, 3), vj)   # + relative-sensing noise
            neighbors[i] = seen

        # advance the latency buffer
        self._buffer.append(current)
        if len(self._buffer) > max(c.latency_steps, 0):
            self._buffer.pop(0)
        return {"own_pos": own_pos, "own_vel": own_vel, "deck": deck_est, "deck_vel": deck_vel_est,
                "neighbors": neighbors, "deck_meas": deck_meas, "deck_std": deck_std}
