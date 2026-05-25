"""Decentralized collision avoidance for the swarm — a control-barrier-function (CBF) safety filter.

Each drone has a *desired* velocity (from holding guidance or the landing autopilot). Before applying
it, this filter minimally modifies it to guarantee pairwise separation: for any two drones the squared
distance must stay above ``d_min²``. With the barrier ``h = ||p_i - p_j||² - d_min²`` and a
single-integrator velocity model, the CBF condition ``ḣ + α·h ≥ 0`` becomes a linear inequality in the
drone's velocity:

    2 (p_i - p_j)·(v_i - v_j) + α (‖p_i - p_j‖² - d_min²) ≥ 0

We solve the **exact** minimal-intervention QP

    min_v ½‖v − v_des‖²   s.t.   for every neighbour j:  -grad_j·v ≤ -b_j

(where ``grad_j = 2(p_i − p_j)`` and ``b_j = grad_j·v_j − α(‖p_i−p_j‖² − d_min²)``) via **Hildreth's
dual algorithm** — the classic exact method for small inequality-constrained QPs, no external solver.
This returns the velocity *closest* to the desired one that is still safe (a loose projection would
not be minimal-intervention). When drones are well separated every constraint is slack and the desired
velocity passes through unchanged; the filter only bends it near conflicts. Decentralized: each drone
needs only neighbours' broadcast positions/velocities. The same math gives the deck **keep-out**
(treat the deck as a virtual obstacle for any drone not cleared to land).
"""

from __future__ import annotations

import numpy as np


def cbf_safe_velocity(
    p_self: np.ndarray,
    v_des: np.ndarray,
    neighbors: list[tuple[np.ndarray, np.ndarray]],
    d_min: float,
    alpha: float = 2.0,
    v_max: float | None = None,
    iters: int = 60,
) -> np.ndarray:
    """Return the safe velocity closest to ``v_des`` satisfying the pairwise CBF constraints (exact QP).

    ``neighbors`` is a list of ``(position, velocity)`` for nearby agents/obstacles (same dimension as
    ``p_self``). ``alpha`` sets how assertively the barrier is enforced; ``v_max`` optionally caps speed.
    """
    p_self = np.asarray(p_self, dtype=float)
    v_des = np.asarray(v_des, dtype=float)
    if not neighbors:
        return _clip(v_des.copy(), v_max)

    # Build the constraints  G v <= h  (one row per neighbour) from the CBF condition.
    G, h = [], []
    for p_j, v_j in neighbors:
        d = p_self - np.asarray(p_j, dtype=float)
        grad = 2.0 * d
        b = float(grad @ np.asarray(v_j, dtype=float)) - alpha * (float(d @ d) - d_min**2)
        G.append(-grad)          # -grad·v <= -b
        h.append(-b)
    G = np.asarray(G)
    h = np.asarray(h)

    # Exact QP via Hildreth: min ½‖v-v_des‖² s.t. Gv<=h. E=I, F=-v_des -> P=GGᵀ, K=h-G v_des.
    P = G @ G.T
    K = h - G @ v_des
    lam = np.zeros(len(h))
    diag = np.clip(np.diag(P), 1e-12, None)
    for _ in range(iters):
        for i in range(len(h)):
            w = -(K[i] + P[i] @ lam - P[i, i] * lam[i]) / diag[i]
            lam[i] = max(0.0, w)
    v = v_des - G.T @ lam
    return _clip(v, v_max)


def _clip(v: np.ndarray, v_max: float | None) -> np.ndarray:
    if v_max is None:
        return v
    s = float(np.linalg.norm(v))
    return v * (v_max / s) if s > v_max else v


def min_pairwise_distance(positions: list[np.ndarray]) -> float:
    """Smallest pairwise distance among the given positions (a safety/metric helper)."""
    n = len(positions)
    if n < 2:
        return float("inf")
    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, float(np.linalg.norm(np.asarray(positions[i]) - np.asarray(positions[j]))))
    return best
