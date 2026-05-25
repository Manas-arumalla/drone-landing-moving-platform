"""Higher-order CBF sense-and-avoid for an acceleration-controlled drone (P3.3 / P3.4).

The swarm's inter-drone filter (``drone_landing_swarm.avoidance``) is a *single-integrator* CBF: it bends
a desired **velocity**. A real quadrotor is acceleration-controlled and cannot change velocity
instantaneously, so a static obstacle barrier ``h(p) = ||p - p_o||^2 - R^2`` has **relative degree 2**
w.r.t. the acceleration command — a velocity-level filter would act one integrator too late and clip the
obstacle. This module implements the standard **higher-order CBF (HOCBF)** for relative degree 2
(Xiao & Belta 2019):

    psi0 = h
    psi1 = h_dot + a1 * h
    require   psi1_dot + a2 * psi1 >= 0

For a double integrator (``p_ddot = a``) and a *static* obstacle this expands to a linear inequality in
the acceleration command ``a``::

    g . a  >=  -2||v||^2 - 2 a1 (p-p_o).v - a2 ( 2(p-p_o).v + a1 h ),      g = 2 (p - p_o)

We solve the minimal-intervention QP ``min ||a - a_des||^2  s.t.  g_i.a >= rhs_i  for every obstacle`` (and
an acceleration-magnitude cap) with **Hildreth's dual** — the same exact small-QP method used by the swarm
filter, no external solver.

**Actuation latency (P3.4).** A command takes ``latency`` seconds to bite. We robustify by evaluating the
barrier at the **look-ahead** state the drone will be in when the command takes effect (``p + v*latency``),
and inflate the safe radius ``R`` by the drone radius + a margin. This is a sound, cheap robustification:
the constraint is enforced where the vehicle *will* be, not where it *is*.

Inputs are whatever the drone can actually observe — surface points from the onboard
:class:`~drone_landing.safety.obstacles.RangeSensor`, never obstacle identities (no-cheats)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AvoidConfig:
    alpha1: float = 2.0        # HOCBF gain on h        (outer class-K)
    alpha2: float = 4.0        # HOCBF gain on psi1      (inner class-K)
    a_max: float = 4.0         # m/s^2  acceleration-command magnitude cap (matches geometric a_xy_max)
    drone_radius: float = 0.25 # m  half-span of the X2 (rotor tip) — inflates every obstacle
    margin: float = 0.15       # m  extra robustness margin (sensing/tracking error)
    latency: float = 0.05      # s  actuation latency -> look-ahead robustification


@dataclass
class AvoidReport:
    steps: int = 0
    activations: int = 0
    min_barrier: float = float("inf")      # smallest pre-step h seen (negative => inside the unsafe set)
    max_correction: float = 0.0            # largest ||a_safe - a_des||

    def as_dict(self) -> dict:
        return {"steps": self.steps, "activations": self.activations,
                "min_barrier": round(self.min_barrier, 4),
                "max_correction": round(self.max_correction, 4),
                "clear": bool(self.min_barrier >= -1e-6)}


class HOCBFAvoider:
    """Acceleration-level higher-order CBF filter against sensed static obstacle points."""

    def __init__(self, config: AvoidConfig | None = None):
        self.cfg = config or AvoidConfig()
        self.report = AvoidReport()

    def reset(self) -> None:
        self.report = AvoidReport()

    def filter(self, p: np.ndarray, v: np.ndarray, a_des: np.ndarray,
               obstacle_points: list[np.ndarray], obstacle_radius: float = 0.0) -> np.ndarray:
        """Return the minimal-intervention safe acceleration closest to ``a_des``.

        ``obstacle_points`` are world-XY surface returns (each treated as a point obstacle); ``R`` is the
        point's effective keep-out = ``obstacle_radius + drone_radius + margin``."""
        c = self.cfg
        p = np.asarray(p, dtype=float)[:2]
        v = np.asarray(v, dtype=float)[:2]
        a_des = np.asarray(a_des, dtype=float)[:2]
        R = obstacle_radius + c.drone_radius + c.margin
        p_look = p + v * c.latency                                     # latency look-ahead state

        rows_g, rows_rhs = [], []
        min_h = float("inf")
        for q in obstacle_points:
            q = np.asarray(q, dtype=float)[:2]
            rel = p_look - q
            h = float(rel @ rel) - R * R
            min_h = min(min_h, h)
            hdot = 2.0 * float(rel @ v)
            g = 2.0 * rel
            # psi1_dot + a2*psi1 >= 0  =>  g.a >= -2||v||^2 - a1*hdot - a2*(hdot + a1*h)
            rhs = -2.0 * float(v @ v) - c.alpha1 * hdot - c.alpha2 * (hdot + c.alpha1 * h)
            rows_g.append(g)
            rows_rhs.append(rhs)

        a_safe = self._solve(a_des, rows_g, rows_rhs)
        self._record(a_des, a_safe, min_h)
        return a_safe

    def _solve(self, a_des, rows_g, rows_rhs):
        """min 1/2||a-a_des||^2 s.t. g_i.a >= rhs_i (i.e. -g_i.a <= -rhs_i) + ||a||<=a_max, Hildreth dual."""
        c = self.cfg
        if not rows_g:
            return _cap(a_des.copy(), c.a_max)
        G = -np.asarray(rows_g)                      # -g.a <= -rhs
        hvec = -np.asarray(rows_rhs)
        P = G @ G.T
        K = hvec - G @ a_des
        lam = np.zeros(len(hvec))
        diag = np.clip(np.diag(P), 1e-12, None)
        for _ in range(80):
            for i in range(len(hvec)):
                w = -(K[i] + P[i] @ lam - P[i, i] * lam[i]) / diag[i]
                lam[i] = max(0.0, w)
        a = a_des - G.T @ lam
        return _cap(a, c.a_max)

    def _record(self, a_des, a_safe, min_h) -> None:
        r = self.report
        r.steps += 1
        corr = float(np.linalg.norm(a_safe - a_des))
        if corr > 1e-9:
            r.activations += 1
        r.max_correction = max(r.max_correction, corr)
        r.min_barrier = min(r.min_barrier, min_h)


def _cap(a: np.ndarray, a_max: float) -> np.ndarray:
    s = float(np.linalg.norm(a))
    return a * (a_max / s) if s > a_max else a


def cluster_returns(points: list[np.ndarray], tol: float = 0.4) -> list[np.ndarray]:
    """Greedy spatial clustering of range returns -> one representative (nearest-to-mean) point per cluster.

    A scan paints many points on one surface; collapsing them keeps the QP small and avoids double-counting
    the same wall. Cheap single-link clustering (O(n^2), fine for tens of beams)."""
    pts = [np.asarray(p, dtype=float)[:2] for p in points]
    clusters: list[list[np.ndarray]] = []
    for p in pts:
        placed = False
        for cl in clusters:
            if float(np.linalg.norm(p - cl[0])) <= tol:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    reps = []
    for cl in clusters:
        arr = np.array(cl)
        reps.append(arr[np.argmin(np.linalg.norm(arr - arr.mean(0), axis=1))])
    return reps
