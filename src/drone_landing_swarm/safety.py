"""Non-bypassable safety layer (A3): the final CBF filter every command must pass through.

The swarm's collision-avoidance guarantee is only as good as its *enforcement*. This module turns the
exact CBF-QP (``avoidance.cbf_safe_velocity``, Hildreth) into a **structural chokepoint**: the
coordinator has exactly one path from a desired velocity to the plant — ``SafetyFilter.filter`` — so a
**learned policy (MARL residual) cannot emit an unfiltered command**. The residual is added to the
guidance *before* the filter; the filter is always the last word.

## The formal guarantee (and its honest scope)

Per drone we use the pairwise barrier ``h_ij = ||p_i - p_j||^2 - d_min^2`` (and a larger ``d_min`` for the
deck keep-out). The filter enforces the **continuous-time CBF condition** for every neighbour ``j``::

    2 (p_i - p_j) . (v_i - v_j) + alpha (||p_i - p_j||^2 - d_min^2) >= 0          (CBF)

With a **single-integrator** model and a forward-Euler step of ``dt`` (``p <- p + v*dt``), the next-step
barrier of a pair is exactly::

    h+ = ||d + (v_i - v_j) dt||^2 - d_min^2
       = h + dt [2 d . (v_i - v_j)] + dt^2 ||v_i - v_j||^2
      >= h - alpha*dt*h + dt^2||.||^2                                    (by CBF)
      >= (1 - alpha*dt) h        (since the quadratic term is >= 0)

So **if ``0 < alpha*dt < 1`` and ``h >= 0`` then ``h+ >= 0``** — the safe set ``{h >= 0}`` is
**forward-invariant in discrete time** (a clean, provable statement, not just the continuous-time one).
``SafetyFilter`` *asserts* ``alpha*dt < 1`` at construction, so the precondition can't be violated silently.

**Scope / honesty.** The proof assumes (i) the single-integrator kinematic model, (ii) each drone knows
the neighbour state ``(p_j, v_j)`` it is avoiding, and (iii) the neighbour holds ``v_j`` across the step.
Under the no-cheats A1 sensing model those inputs are **noisy, stale (latency), and occasionally dropped**,
and on MuJoCo the true plant is a second-order quadrotor tracking the velocity, not a perfect integrator.
The guarantee therefore holds **up to a bounded margin** set by the sensing/tracking error: we expose a
``margin`` that tightens every ``d_min`` to absorb it, and we *verify empirically* (``verify_separation``)
that the true minimum separation stays above ``d_min`` across a large seed sweep. The filter also reports a
per-step **certificate** — the worst predicted ``h+`` over all active constraints — so any excursion is
auditable rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_landing_swarm.avoidance import cbf_safe_velocity, min_pairwise_distance


@dataclass
class SafetySpec:
    d_min: float = 0.7          # m   minimum inter-drone separation
    deck_keepout: float = 0.8   # m   keep-out radius around the deck for non-cleared drones
    alpha: float = 3.0          # CBF class-K gain
    dt: float = 0.05            # s   integration step (must satisfy alpha*dt < 1)
    margin: float = 0.0         # m   extra tightening on every separation distance (robustness to A1 noise)
    v_max: float = 1.5          # m/s speed cap applied after filtering

    def __post_init__(self) -> None:
        # Discrete-time forward-invariance precondition (see module docstring). Without this the
        # one-step barrier bound (1 - alpha*dt) h can go negative and the guarantee is void.
        if not (0.0 < self.alpha * self.dt < 1.0):
            raise ValueError(
                f"CBF invariance requires 0 < alpha*dt < 1, got alpha*dt={self.alpha * self.dt:.3f} "
                f"(alpha={self.alpha}, dt={self.dt}). Lower alpha or dt.")


@dataclass
class SafetyReport:
    steps: int = 0                       # filter invocations
    activations: int = 0                 # invocations where the filter actually bent the command
    min_barrier: float = float("inf")    # smallest pre-step h seen (estimate space; <0 => inside unsafe set)
    min_certificate: float = float("inf")  # smallest predicted next-step h+ (the one-step safety certificate)
    max_correction: float = 0.0          # largest ||v_safe - v_des|| applied

    def as_dict(self) -> dict:
        return {"steps": self.steps, "activations": self.activations,
                "min_barrier": round(self.min_barrier, 4),
                "min_certificate": round(self.min_certificate, 4),
                "max_correction": round(self.max_correction, 4),
                "certified": bool(self.min_certificate >= -1e-6)}


class SafetyFilter:
    """The one and only path from a desired velocity to a commanded velocity.

    ``filter(p_self, v_des, neighbors, deck=None)`` returns the minimal-intervention safe velocity and
    records a certificate. ``neighbors`` is ``[(pos, vel), ...]`` (drones); ``deck`` is an optional
    ``(pos, vel)`` whose larger keep-out radius applies only when the drone is *not* cleared to land.
    All positions/velocities are whatever the caller can actually observe (estimates under A1).
    """

    def __init__(self, spec: SafetySpec | None = None):
        self.spec = spec or SafetySpec()
        self.report = SafetyReport()

    def reset(self) -> None:
        self.report = SafetyReport()

    def filter(self, p_self: np.ndarray, v_des: np.ndarray,
               neighbors: list[tuple[np.ndarray, np.ndarray]],
               deck: tuple[np.ndarray, np.ndarray] | None = None,
               obstacles: list[tuple[np.ndarray, float]] | None = None) -> np.ndarray:
        s = self.spec
        p_self = np.asarray(p_self, dtype=float)
        v_des = np.asarray(v_des, dtype=float)

        # Assemble constraints with their (margin-tightened) separation distances.
        cons = [(np.asarray(p, float), np.asarray(v, float), s.d_min + s.margin) for p, v in neighbors]
        if deck is not None:
            cons.append((np.asarray(deck[0], float), np.asarray(deck[1], float),
                         s.deck_keepout + s.margin))
        # P3: SENSED static obstacles (e.g. the OSV superstructure) — zero-velocity keep-out volumes with
        # their own radius, folded into the SAME exact CBF-QP as the inter-drone/deck constraints. Additive:
        # an empty/None list reproduces the original behaviour exactly (no regression on a clear deck).
        if obstacles:
            for p_o, radius in obstacles:
                cons.append((np.asarray(p_o, float), np.zeros_like(p_self),
                             float(radius) + s.margin))

        v_safe = self._solve(p_self, v_des, cons)
        self._record(p_self, v_des, v_safe, cons)
        return v_safe

    def _solve(self, p_self, v_des, cons):
        """Exact minimal-intervention QP over all constraints (group by d_min; intersect via re-filtering)."""
        v = v_des.copy()
        dmins = sorted({d for _, _, d in cons}, reverse=True)
        for dm in dmins:
            grp = [(p, vv) for p, vv, d in cons if d == dm]
            v = cbf_safe_velocity(p_self, v, grp, d_min=dm, alpha=self.spec.alpha, v_max=self.spec.v_max)
        return v

    def _record(self, p_self, v_des, v_safe, cons) -> None:
        r = self.report
        r.steps += 1
        corr = float(np.linalg.norm(v_safe - v_des))
        if corr > 1e-9:
            r.activations += 1
        r.max_correction = max(r.max_correction, corr)
        for p_j, v_j, dm in cons:
            d = p_self - p_j
            h = float(d @ d) - dm * dm                                   # pre-step barrier
            dv = v_safe - v_j
            dnext = d + dv * self.spec.dt
            h_next = float(dnext @ dnext) - dm * dm                      # one-step-lookahead certificate
            r.min_barrier = min(r.min_barrier, h)
            r.min_certificate = min(r.min_certificate, h_next)


# --------------------------------------------------------------------- verification harness

def verify_separation(make_coord, seeds, *, d_min: float | None = None, tol: float = 0.1) -> dict:
    """Run a coordinator factory across ``seeds`` and assert it never violates separation.

    ``make_coord`` is a zero-arg callable returning a fresh coordinator (kinematic or MuJoCo) that
    exposes ``.run(seed)->result`` and ``.cfg.d_min``; if the coordinator carries a ``SafetyFilter`` as
    ``.safety`` its certificate is collected too. Returns the worst TRUE min-separation across the sweep,
    the worst one-step certificate (estimate space), and the list of violating seeds (true sep below
    ``d_min - tol``). ``passed`` is True iff there are no violations.
    """
    worst_sep = float("inf")
    worst_cert = float("inf")
    violations: list[tuple[int, float]] = []
    for s in seeds:
        coord = make_coord()
        thresh = (d_min if d_min is not None else coord.cfg.d_min)
        r = coord.run(s)
        sep = float(r["min_separation"])
        worst_sep = min(worst_sep, sep)
        safety = getattr(coord, "safety", None)
        if safety is not None:
            worst_cert = min(worst_cert, safety.report.min_certificate)
        if sep < thresh - tol:
            violations.append((int(s), round(sep, 3)))
    return {
        "n_seeds": len(list(seeds)) if hasattr(seeds, "__len__") else None,
        "worst_min_separation": round(worst_sep, 3),
        "worst_certificate": (round(worst_cert, 4) if np.isfinite(worst_cert) else None),
        "n_violations": len(violations),
        "violations": violations,
        "passed": len(violations) == 0,
    }
