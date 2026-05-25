"""Cooperative consensus deck estimation (A2): a distributed Kalman-Consensus Filter.

Each drone holds its *own* estimate of the moving deck's pose. Under the no-cheats sensing model (A1)
that estimate is noisy, and its quality degrades with range — a drone far out in the holding stack barely
resolves the deck marker, and past ``deck_vis_range`` it gets **no** direct fix at all. A drone hovering
right over the deck, by contrast, has a sharp lock.

A **Kalman-Consensus Filter** (Olfati-Saber, CDC 2007) lets the drones *cooperate*: over the comms graph
each drone fuses its own measurement with its neighbours' estimates, so the well-placed observers pull the
blind ones onto the true deck. Per drone, per step:

1. **Predict** a constant-velocity deck model ``x = [p(3), v(3)]``: ``x⁻ = F x``, ``P⁻ = F P Fᵀ + Q``.
2. **Measurement update** (only if the drone currently sees the deck) with its own range-dependent noise
   ``R = std² I`` — a standard KF position update.
3. **Consensus update** — pull toward the (stale, one-step-delayed) estimates broadcast by in-range
   neighbours: ``x ← x + γ · mean_{j∈N_i}(x_j − x_i)``. The averaging form is stable for ``0 ≤ γ ≤ 1``.

The result is a per-drone fused deck estimate that is **better than any single drone's raw measurement**
(and finite even for drones with no direct view). It is fully decentralized — only neighbour exchange,
no central node — and it directly counters the comms/partial-observability limit that capped the swarm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConsensusConfig:
    process_var: float = 0.6      # deck acceleration uncertainty (drives Q in the CV model)
    consensus_gain: float = 0.45  # gamma: weight on the neighbour-disagreement pull (0..1, averaging form)
    init_pos_var: float = 1.0     # initial position covariance
    init_vel_var: float = 1.0     # initial velocity covariance


class ConsensusDeckEstimator:
    """Per-drone distributed Kalman-Consensus Filter tracking the deck pose (position + velocity)."""

    def __init__(self, n_drones: int, dt: float, config: ConsensusConfig | None = None):
        self.n = n_drones
        self.dt = float(dt)
        self.cfg = config or ConsensusConfig()
        # constant-velocity model x=[px,py,pz,vx,vy,vz]
        self.F = np.eye(6)
        self.F[:3, 3:] = self.dt * np.eye(3)
        self.H = np.zeros((3, 6))
        self.H[:, :3] = np.eye(3)
        self.Q = self._cv_process_noise(self.dt, self.cfg.process_var)
        self.reset()

    @staticmethod
    def _cv_process_noise(dt: float, q: float) -> np.ndarray:
        # standard continuous-white-noise-acceleration discretization
        Q = np.zeros((6, 6))
        I = np.eye(3)
        Q[:3, :3] = (dt**4 / 4) * I
        Q[:3, 3:] = (dt**3 / 2) * I
        Q[3:, :3] = (dt**3 / 2) * I
        Q[3:, 3:] = (dt**2) * I
        return q * Q

    def reset(self, deck0: np.ndarray | None = None) -> None:
        c = self.cfg
        p0 = np.zeros(3) if deck0 is None else np.asarray(deck0, float)
        self.x = {i: np.concatenate([p0, np.zeros(3)]) for i in range(self.n)}
        P0 = np.diag([c.init_pos_var] * 3 + [c.init_vel_var] * 3)
        self.P = {i: P0.copy() for i in range(self.n)}
        self._prev_x = {i: self.x[i].copy() for i in range(self.n)}

    def step(self, measurements: dict[int, np.ndarray | None], stds: dict[int, float],
             neighbors: dict[int, list[int]]) -> dict[int, np.ndarray]:
        """Advance one step. Returns the fused deck-state estimate per active drone.

        ``measurements[i]`` is the drone's own deck-position fix (or ``None`` if it has no view this step);
        ``stds[i]`` its measurement std; ``neighbors[i]`` the in-range neighbour ids whose (one-step-stale)
        estimates drone ``i`` may fuse. Drones not present in ``measurements`` are treated as inactive.
        """
        active = list(measurements.keys())
        prev = self._prev_x                       # neighbours broadcast last step's fused estimate (stale)
        new_x, new_P = {}, {}
        for i in active:
            # 1. predict
            x_pred = self.F @ self.x[i]
            P_pred = self.F @ self.P[i] @ self.F.T + self.Q
            # 2. measurement update (only if this drone sees the deck)
            z = measurements[i]
            if z is not None:
                R = (max(stds.get(i, 0.06), 1e-6) ** 2) * np.eye(3)
                S = self.H @ P_pred @ self.H.T + R
                K = P_pred @ self.H.T @ np.linalg.inv(S)
                x_pred = x_pred + K @ (np.asarray(z, float) - self.H @ x_pred)
                P_pred = (np.eye(6) - K @ self.H) @ P_pred
            # 3. consensus update: pull toward neighbours' (stale) estimates
            nbrs = neighbors.get(i, [])
            if nbrs:
                disagreement = np.mean([prev[j] - x_pred for j in nbrs if j in prev], axis=0)
                x_pred = x_pred + self.cfg.consensus_gain * disagreement
            new_x[i], new_P[i] = x_pred, P_pred
        # commit
        for i in active:
            self.x[i], self.P[i] = new_x[i], new_P[i]
        self._prev_x = {i: self.x[i].copy() for i in active}
        return {i: self.x[i].copy() for i in active}

    def deck_pos(self, i: int) -> np.ndarray:
        return self.x[i][:3].copy()

    def deck_vel(self, i: int) -> np.ndarray:
        return self.x[i][3:].copy()
