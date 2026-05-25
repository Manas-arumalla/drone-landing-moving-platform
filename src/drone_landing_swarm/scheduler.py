"""Decentralized landing-slot scheduling for the swarm.

One moving deck has a small number of usable landing slots (``n_slots``, typically 1). The scheduler
decides which drones are *cleared* to begin their final descent; the rest hold. Design choices for the
best result:

* **Optimal selection.** For ``K`` *identical* slots on one deck, clearing the ``K`` lowest-cost
  (readiest) drones is provably optimal, so we select the K-minimum by cost. For the general
  ``M`` drones × ``K`` *distinct* platforms problem we expose :func:`optimal_assignment`, the
  **Hungarian algorithm** (``scipy.optimize.linear_sum_assignment``) — globally optimal min-cost
  assignment — ready for the multi-platform extension.
* **Anti-starvation.** A waiting-time bonus reduces a drone's effective cost the longer it holds, so
  no drone is indefinitely passed over.
* **Hysteresis.** Once cleared, a drone stays cleared until it lands or aborts — clearance never
  chatters between drones mid-approach.

Decentralized in spirit: the cost each drone needs is computable from broadcast states, so every drone
can run the same selection and reach the same clearance set; here one object computes it for the sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def optimal_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Globally-optimal min-cost assignment (Hungarian) of rows (drones) to columns (platforms/slots).

    Returns a list of ``(row, col)`` pairs. Used for the M-drones × K-distinct-platforms generalization.
    """
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(np.asarray(cost_matrix, dtype=float))
    return list(zip(r.tolist(), c.tolist()))


@dataclass(frozen=True)
class SchedulerConfig:
    n_slots: int = 1                 # drones allowed on final approach at once
    starvation_weight: float = 0.05  # per-second reduction of effective cost while holding


class LandingScheduler:
    """Assigns the (identical) landing slots of one deck to the readiest drones, with hysteresis."""

    def __init__(self, n_drones: int, config: SchedulerConfig | None = None):
        self.n = n_drones
        self.cfg = config or SchedulerConfig()
        self.reset()

    def reset(self) -> None:
        self.cleared: set[int] = set()
        self.done: set[int] = set()
        self._wait = np.zeros(self.n)

    def mark_done(self, i: int) -> None:
        """Mark a drone as landed/secured or permanently removed; frees its slot."""
        self.done.add(i)
        self.cleared.discard(i)

    def update(self, costs: np.ndarray, active: np.ndarray, dt: float) -> set[int]:
        """Return the set of drones cleared to land.

        ``costs[i]`` = readiness cost (lower = readier, e.g. horizontal distance to the deck centre).
        ``active[i]`` = drone i still wants to land (not landed/aborted).
        """
        costs = np.asarray(costs, dtype=float)
        active = np.asarray(active, dtype=bool)
        self.cleared -= self.done
        # accumulate waiting time for holding (active, not-yet-cleared, not-done) drones
        for i in range(self.n):
            if active[i] and i not in self.cleared and i not in self.done:
                self._wait[i] += dt
        # fill free slots with the lowest effective-cost candidates (optimal for identical slots)
        while len(self.cleared) < self.cfg.n_slots:
            cands = [i for i in range(self.n)
                     if active[i] and i not in self.cleared and i not in self.done]
            if not cands:
                break
            eff = {i: costs[i] - self.cfg.starvation_weight * self._wait[i] for i in cands}
            best = min(eff, key=eff.get)
            self.cleared.add(best)
            self._wait[best] = 0.0
        return set(self.cleared)
