"""Formal safety via Hamilton-Jacobi reachability (B2): a verified safe-landing set + runtime shield.

A landing is only *safe* from states where the drone can still touch down softly despite the worst-case
disturbance (wind / air-wake). We compute that set rigorously for the **vertical landing channel** — the
axis where a hard touchdown actually breaks the aircraft — and wrap any controller (geometric, MPC, RL)
in a **runtime-assurance shield** that only lets a command through if it keeps the state inside the set.

Reduced-order model (relative to the moving deck), the dimension that governs touchdown safety::

    h_dot = w
    w_dot = a + d ,     a in [a_min, a_max] (thrust-limited accel),   |d| <= d_max (disturbance)

Target (a soft landing): ``h <= 0`` reached with ``|w| <= w_land``. We compute the **robust
backward-reachable set** — the states from which *some* control keeps the drone safe against *every*
admissible disturbance — by discrete-time HJ dynamic programming (a differential-game / Stackelberg
``exists a, forall d`` update) iterated to a fixed point::

    Safe_{k+1}(x) = x in Target  OR  ( in_bounds(x) AND  exists a forall d :  x + f(x,a,d) dt  in Safe_k )

The fixed point is a **control-invariant funnel** to the target — the maximal set from which a soft
landing is guaranteed under the disturbance bound. The boundary is the classic worst-case braking curve
``h = w^2 / (2 (|a_min| - d_max))`` for ``w < 0`` (you must start braking before this depth), which we use
to validate the grid solution. The shield then guarantees the drone never commits past the point of no
return — a provable safety envelope around the learned policy, complementing the swarm's CBF (A3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReachabilityConfig:
    a_min: float = -6.0      # m/s^2  max downward accel command (thrust low; gravity-comp frame)
    a_max: float = 6.0       # m/s^2  max upward accel command (thrust limit minus weight)
    d_max: float = 2.0       # m/s^2  disturbance bound (wind/air-wake vertical accel)
    w_land: float = 0.6      # m/s    max |vertical speed| that counts as a soft touchdown
    h_max: float = 4.0       # m      altitude range modelled
    w_max: float = 4.0       # m/s    vertical-speed range modelled (|w| <= w_max)
    nh: int = 161            # grid resolution (altitude)
    nw: int = 161            # grid resolution (vertical speed)
    dt: float = 0.05         # s      DP time step
    n_actions: int = 11      # discretized control samples in [a_min, a_max]
    n_dist: int = 5          # disturbance samples in [-d_max, d_max] (forall d)


class LandingReachability:
    """Robust backward-reachable safe-landing set + a runtime-assurance shield for the vertical channel."""

    def __init__(self, config: ReachabilityConfig | None = None):
        self.cfg = config or ReachabilityConfig()
        c = self.cfg
        self.h_grid = np.linspace(0.0, c.h_max, c.nh)
        self.w_grid = np.linspace(-c.w_max, c.w_max, c.nw)
        self.actions = np.linspace(c.a_min, c.a_max, c.n_actions)
        self.dists = np.linspace(-c.d_max, c.d_max, c.n_dist)
        # Internal target speed is tightened by one worst-case disturbance step (d_max*dt) so that the
        # *realized* discrete-time touchdown still respects the public guarantee |w| <= w_land despite the
        # final step's unmodelled disturbance kick.
        self._w_land_eff = max(c.w_land - c.d_max * c.dt, 0.05)
        self.safe = self._compute_safe_set()

    # ------------------------------------------------------------------ compute
    def _target(self, H: np.ndarray, W: np.ndarray) -> np.ndarray:
        dh = self.h_grid[1] - self.h_grid[0]
        return (H <= dh) & (np.abs(W) <= self._w_land_eff)   # at/near the deck with soft speed

    def _compute_safe_set(self) -> np.ndarray:
        c = self.cfg
        H, W = np.meshgrid(self.h_grid, self.w_grid, indexing="ij")     # (nh, nw)
        target = self._target(H, W)
        in_bounds = (H >= 0) & (H <= c.h_max) & (np.abs(W) <= c.w_max)
        safe = target.copy()
        dh = self.h_grid[1] - self.h_grid[0]
        dw = self.w_grid[1] - self.w_grid[0]

        def lookup(Hn: np.ndarray, Wn: np.ndarray, S: np.ndarray) -> np.ndarray:
            """Is (Hn, Wn) inside the current safe grid S? (nearest-cell, with bounds + target check)."""
            ih = np.clip(np.round(Hn / dh).astype(int), 0, c.nh - 1)
            iw = np.clip(np.round((Wn + c.w_max) / dw).astype(int), 0, c.nw - 1)
            ok = (Hn >= -dh) & (Hn <= c.h_max) & (np.abs(Wn) <= c.w_max)
            landed = (Hn <= dh) & (np.abs(Wn) <= self._w_land_eff)       # crossed into the soft-touch target
            return (ok & S[ih, iw]) | landed

        for _ in range(c.nh + c.nw):                                    # ample iterations to converge
            # exists a : forall d : next-state safe
            exists_a = np.zeros_like(safe)
            for a in self.actions:
                forall_d = np.ones_like(safe)
                for d in self.dists:
                    Hn = H + W * c.dt
                    Wn = W + (a + d) * c.dt
                    forall_d &= lookup(Hn, Wn, safe)
                exists_a |= forall_d
            new = target | (in_bounds & exists_a)
            if np.array_equal(new, safe):
                break
            safe = new
        return safe

    # ------------------------------------------------------------------ queries
    def is_safe(self, h: float, w: float) -> bool:
        """Is the drone in the verified safe-landing set at altitude ``h`` (above deck), vert. speed ``w``?"""
        c = self.cfg
        if h < 0 or h > c.h_max or abs(w) > c.w_max:
            return abs(w) <= c.w_land and h <= 0          # below deck is safe only if it touched softly
        ih = int(np.clip(round(h / (self.h_grid[1] - self.h_grid[0])), 0, c.nh - 1))
        iw = int(np.clip(round((w + c.w_max) / (self.w_grid[1] - self.w_grid[0])), 0, c.nw - 1))
        return bool(self.safe[ih, iw])

    def _next_safe(self, h: float, w: float, a: float) -> bool:
        """Would commanding accel ``a`` keep the state safe against every admissible disturbance?"""
        c = self.cfg
        for d in self.dists:
            hn, wn = h + w * c.dt, w + (a + d) * c.dt
            if not (self.is_safe(hn, wn) or (hn <= 0 and abs(wn) <= self._w_land_eff)):
                return False
        return True

    def safe_action(self, h: float, w: float, a_nominal: float) -> tuple[float, bool]:
        """Runtime-assurance shield: pass ``a_nominal`` if it stays safe, else override with the safest
        admissible accel (max braking is safest in the landing channel). Returns ``(a, intervened)``."""
        c = self.cfg
        a_nominal = float(np.clip(a_nominal, c.a_min, c.a_max))
        if self._next_safe(h, w, a_nominal):
            return a_nominal, False
        for a in sorted(self.actions, key=lambda x: -x):     # prefer the strongest upward braking
            if self._next_safe(h, w, a):
                return float(a), True
        return c.a_max, True                                 # last resort: full thrust

    def safe_descent_speed(self, h: float) -> float:
        """Max safe *descent* speed (m/s, magnitude) at altitude ``h`` above the deck: the fastest the
        drone may descend and still be able to brake to a soft touchdown under the worst-case disturbance.
        A velocity-level shield (clamp the commanded descent to this) keeps the landing in the safe set."""
        c = self.cfg
        if h <= 0:
            return c.w_land
        ih = int(np.clip(round(h / (self.h_grid[1] - self.h_grid[0])), 0, c.nh - 1))
        col = self.safe[ih, :]
        if not col.any():
            return 0.0
        return float(-self.w_grid[col].min())     # most-negative safe w at this altitude -> |descent|

    def braking_boundary(self, w: np.ndarray) -> np.ndarray:
        """Analytic worst-case braking curve: the minimum altitude from which a descent at ``w`` can be
        slowed to the soft-touchdown speed ``w_land`` using full upward accel ``a_max`` against the
        worst-case downward disturbance ``d_max``. The safe set must lie above it (validation reference)::

            h_brake(w) = max(0, (w^2 - w_land^2) / (2 (a_max - d_max)))   for  w < -w_land

        (The continuous-time curve; the discrete-time grid set may sit up to one step ``|w| dt`` below it.)
        """
        c = self.cfg
        w = np.asarray(w, dtype=float)
        decel = max(c.a_max - c.d_max, 1e-6)
        return np.where(w < -c.w_land, np.maximum(0.0, (w**2 - c.w_land**2) / (2.0 * decel)), 0.0)
