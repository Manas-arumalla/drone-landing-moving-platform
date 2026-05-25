"""K moving platforms + dynamic re-tasking (A5): M drones recovering onto K moving decks.

Generalizes the single-deck :class:`SwarmCoordinator` to a small fleet of decks. The hard new problem is
**assignment**: which drone goes to which deck, re-solved online as the decks move apart and drones land.

* **Initial allocation** — a balanced **Hungarian** assignment (``optimal_assignment``) of M drones to K
  decks (each deck replicated to ``ceil(M/K)`` virtual slots) gives the globally min-distance balanced
  start.
* **Online re-tasking** — every ``reassign_dt`` a **decentralized auction**: each not-yet-committed drone
  picks the deck minimizing ``distance + congestion·load`` (its own onboard estimates), switching only if
  the improvement beats a **hysteresis** margin (no thrashing). As decks separate and drones land,
  freed capacity pulls waiting drones over — emergent multi-deck traffic flow.
* A drone **committed** to a deck (cleared for final descent) is locked to it until it lands.

Everything else is reused unchanged: per-deck :class:`LandingScheduler` + :class:`HoldingStack`, the
no-cheats :class:`SwarmSensing` onboard view, and the non-bypassable global :class:`SafetyFilter` (every
drone avoids every other drone; deck keep-out is around its *assigned* deck). No ground truth in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import numpy as np

from drone_landing.sim.platforms import RandomGroundMotion, ShipDeckMotion, sea_state
from drone_landing_swarm.avoidance import min_pairwise_distance
from drone_landing_swarm.holding import HoldingConfig, HoldingStack
from drone_landing_swarm.safety import SafetyFilter, SafetySpec
from drone_landing_swarm.scheduler import LandingScheduler, SchedulerConfig, optimal_assignment
from drone_landing_swarm.sensing import SensingConfig, SwarmSensing


@dataclass
class MultiDeckConfig:
    n_drones: int = 8
    n_decks: int = 3
    scenario: str = "ship"
    sea: str = "moderate"
    n_slots: int = 1                  # landing slots per deck
    v_max: float = 1.5
    d_min: float = 0.7
    deck_keepout: float = 0.8
    land_radius: float = 0.35
    land_speed: float = 0.5
    touchdown_alt: float = 0.12
    cbf_alpha: float = 3.0
    safety_margin: float = 0.0
    comms_range: float = float("inf")
    reassign_dt: float = 2.0          # re-solve the auction this often (s)
    congestion_weight: float = 1.2    # m of effective distance added per drone already on a deck
    reassign_hysteresis: float = 1.0  # switch deck only if effective cost improves by this (m)
    deck_spacing: float = 4.0         # initial separation between decks (m)
    # "Fouled deck" event: this deck closes at offline_time and its drones must re-task to the others
    # (the scenario dynamic re-tasking exists for). None = all decks stay available.
    offline_deck: int | None = None
    offline_time: float = 4.0
    dt: float = 0.05
    max_time: float = 150.0
    spawn_radius: float = 3.0
    spawn_alt: float = 2.0
    seed: int | None = None
    holding: HoldingConfig = field(default_factory=HoldingConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    # P3 sense-and-avoid: per-vessel static obstacles (the OSV superstructure) as deck-relative
    # (dx, dy, keep-out radius) offsets. Each drone places them at ITS ASSIGNED deck's estimate + offset
    # (no ground truth) and the non-bypassable SafetyFilter keeps clear. () = clear decks (no change).
    obstacles: tuple[tuple[float, float, float], ...] = ()


class MultiDeckCoordinator:
    def __init__(self, config: MultiDeckConfig | None = None):
        self.cfg = config or MultiDeckConfig()
        c = self.cfg
        self.decks = [self._make_deck(k) for k in range(c.n_decks)]
        self.scheduler = [LandingScheduler(c.n_drones, SchedulerConfig(n_slots=c.n_slots))
                          for _ in range(c.n_decks)]
        self.holding = [HoldingStack(c.holding) for _ in range(c.n_decks)]
        self.sensing = SwarmSensing(c.n_drones, c.sensing)
        self.safety = SafetyFilter(SafetySpec(
            d_min=c.d_min, deck_keepout=c.deck_keepout, alpha=c.cbf_alpha, dt=c.dt,
            margin=c.safety_margin, v_max=c.v_max))
        # decks spread out on a ring so they start well separated
        self.deck_base = [c.deck_spacing * np.array([np.cos(2 * np.pi * k / c.n_decks),
                                                     np.sin(2 * np.pi * k / c.n_decks), 0.0])
                          for k in range(c.n_decks)]
        self.reset(c.seed)

    def _make_deck(self, k: int = 0):
        if self.cfg.scenario == "ship":
            # each deck sails on a distinct heading so the fleet spreads out over time -> the assignment
            # geometry changes and dynamic re-tasking actually earns its keep.
            from dataclasses import replace as _replace
            base = sea_state(self.cfg.sea)
            heading = 2 * np.pi * k / max(self.cfg.n_decks, 1)
            return ShipDeckMotion(_replace(base, heading=heading))
        return RandomGroundMotion()

    def reset(self, seed: int | None = None):
        c = self.cfg
        self.rng = np.random.default_rng(seed if seed is not None else c.seed)
        self.deck_pos, self.deck_vel = [], []
        for k, deck in enumerate(self.decks):
            ps = deck.reset(self.rng)
            self.deck_pos.append(ps.pos.copy() + self.deck_base[k])
            self.deck_vel.append(ps.lin_vel.copy())
        for s in self.scheduler:
            s.reset()
        self.sensing.reset(self.rng)
        self.safety.reset()
        centroid = np.mean(self.deck_pos, axis=0)
        self.pos = {}
        for i in range(c.n_drones):
            ang = 2 * np.pi * i / c.n_drones + self.rng.uniform(-0.2, 0.2)
            r = c.spawn_radius * (0.8 + 0.4 * self.rng.random())
            self.pos[i] = centroid + np.array([r * np.cos(ang), r * np.sin(ang),
                                               c.spawn_alt + 0.3 * self.rng.random()])
        self.vel = {i: np.zeros(3) for i in range(c.n_drones)}
        self.landed: set[int] = set()
        self.t = 0.0
        self.min_sep = float("inf")
        self.min_obstacle_clear = float("inf")             # P3: true min clearance to any vessel obstacle
        self.land_time: dict[int, float] = {}
        self.land_deck: dict[int, int] = {}
        self.n_reassign = 0
        self._last_assign_t = -1e9
        self.available = [True] * c.n_decks
        self.assign = self._initial_assignment()
        return self._state()

    # --------------------------------------------------------------- assignment
    def _deck_estimates(self) -> list[np.ndarray]:
        """A shared (consensus-proxy) estimate of each deck: truth + small noise (no exact truth read)."""
        s = self.cfg.sensing.deck_noise
        return [self.deck_pos[k] + self.rng.normal(0, s, 3) for k in range(self.cfg.n_decks)]

    def _initial_assignment(self) -> dict[int, int]:
        """Balanced Hungarian: M drones -> K decks (each replicated to ceil(M/K) slots)."""
        c = self.cfg
        edeck = self._deck_estimates()
        avail = [k for k in range(c.n_decks) if self.available[k]]
        cap = ceil(c.n_drones / max(len(avail), 1))
        cost = np.zeros((c.n_drones, len(avail) * cap))
        for i in range(c.n_drones):
            for ci, k in enumerate(avail):
                d = float(np.linalg.norm(self.pos[i][:2] - edeck[k][:2]))
                cost[i, ci * cap:(ci + 1) * cap] = d
        assign = {}
        for r, col in optimal_assignment(cost):
            assign[r] = avail[col // cap]
        return assign

    def _retask(self, epos: dict, edeck: list[np.ndarray], cleared: set[int]) -> None:
        """Decentralized auction: free drones switch decks if cost (distance + congestion) improves.

        Drones on a now-unavailable (fouled) deck are *forced* to re-task even if committed.
        """
        c = self.cfg
        active = [i for i in range(c.n_drones) if i not in self.landed]
        avail = [k for k in range(c.n_decks) if self.available[k]]

        def load(k: int, exclude: int | None = None) -> int:
            return sum(1 for j in active if self.assign[j] == k and j != exclude)

        def eff_cost(i: int, k: int) -> float:
            return float(np.linalg.norm(epos[i][:2] - edeck[k][:2])) + c.congestion_weight * load(k, i)

        for i in active:
            cur = self.assign[i]
            forced = not self.available[cur]            # this drone's deck was fouled -> must move
            if i in cleared and not forced:             # committed -> locked unless its deck is gone
                continue
            best = min(avail, key=lambda k: eff_cost(i, k))
            if best != cur and (forced or eff_cost(i, cur) - eff_cost(i, best) > c.reassign_hysteresis):
                self.assign[i] = best
                self.n_reassign += 1

    def _state(self) -> dict:
        return {"t": self.t, "deck_pos": [p.copy() for p in self.deck_pos],
                "pos": {i: p.copy() for i, p in self.pos.items()}, "assign": dict(self.assign),
                "landed": set(self.landed), "min_sep": self.min_sep}

    # --------------------------------------------------------------------- step
    def step(self) -> dict:
        c = self.cfg
        for k, deck in enumerate(self.decks):
            ps = deck.step(c.dt)
            self.deck_pos[k] = ps.pos.copy() + self.deck_base[k]
            self.deck_vel[k] = ps.lin_vel.copy()

        # onboard view: own/neighbour estimates (deck 0 placeholder; per-deck estimates below)
        view = self.sensing.sense(self.pos, self.vel, self.deck_pos[0], self.deck_vel[0],
                                  self.landed, c.comms_range)
        epos, evel, neigh = view["own_pos"], view["own_vel"], view["neighbors"]
        edeck = self._deck_estimates()
        self.deck_est = edeck

        # "fouled deck" event: a deck closes mid-recovery -> its drones must re-task to the others
        if c.offline_deck is not None and self.available[c.offline_deck] and self.t >= c.offline_time:
            self.available[c.offline_deck] = False
            self.scheduler[c.offline_deck].cleared.clear()   # drop stale clearances on the closed deck

        # per-deck clearance among that deck's assigned drones (a closed deck clears no one)
        cleared_all: set[int] = set()
        cleared_by_deck: list[set[int]] = []
        for k in range(c.n_decks):
            assigned = np.array([(i not in self.landed) and (self.assign[i] == k) and self.available[k]
                                 for i in range(c.n_drones)])
            costs = np.array([float(np.linalg.norm(epos[i][:2] - edeck[k][:2])) if assigned[i] else 1e9
                              for i in range(c.n_drones)])
            ck = self.scheduler[k].update(costs, assigned, c.dt)
            cleared_by_deck.append(ck)
            cleared_all |= ck

        # re-task free drones periodically (after we know who is committed)
        if self.t - self._last_assign_t >= c.reassign_dt:
            self._retask(epos, edeck, cleared_all)
            self._last_assign_t = self.t

        # holding-slot assignment per deck (holders = assigned, active, not cleared)
        slot_of: dict[int, int] = {}
        for k in range(c.n_decks):
            holders = [i for i in range(c.n_drones)
                       if i not in self.landed and self.assign[i] == k and i not in cleared_by_deck[k]]
            slot_of.update(self.holding[k].assign(holders, {i: epos[i] for i in holders}, edeck[k]))

        new_vel = {}
        for i in range(c.n_drones):
            if i in self.landed:
                continue
            k = self.assign[i]
            if i in cleared_by_deck[k]:
                v_des = self._landing_velocity(epos[i], edeck[k], self.deck_vel[k])
            else:
                v_des = self.holding[k].hold_velocity(epos[i], slot_of.get(i, 0), edeck[k],
                                                      self.deck_vel[k], c.v_max)
            nbrs = list(neigh.get(i, {}).values())
            deck = (edeck[k], self.deck_vel[k]) if i not in cleared_all else None
            # P3: static obstacles around the drone's ASSIGNED vessel, at that deck's estimate + offset
            obstacles = None
            if c.obstacles:
                d = edeck[k]
                obstacles = [(np.array([d[0] + dx, d[1] + dy, d[2]]), r) for (dx, dy, r) in c.obstacles]
            new_vel[i] = self.safety.filter(epos[i], v_des, nbrs, deck=deck, obstacles=obstacles)

        for i, v in new_vel.items():
            self.vel[i] = v
            self.pos[i] = self.pos[i] + v * c.dt

        # true (metric-only) min clearance of any airborne drone to its assigned vessel's obstacles
        if c.obstacles:
            for i in range(c.n_drones):
                if i in self.landed:
                    continue
                dp = self.deck_pos[self.assign[i]]
                for dx, dy, r in c.obstacles:
                    p_o = np.array([dp[0] + dx, dp[1] + dy, dp[2]])
                    self.min_obstacle_clear = min(self.min_obstacle_clear,
                                                  float(np.linalg.norm(self.pos[i] - p_o)) - r)

        # touchdown on the drone's assigned deck
        for i in range(c.n_drones):
            if i in self.landed or i not in cleared_all:
                continue
            k = self.assign[i]
            rel = self.pos[i] - self.deck_pos[k]
            if np.linalg.norm(rel[:2]) <= c.land_radius and rel[2] <= c.touchdown_alt:
                self.landed.add(i)
                self.scheduler[k].mark_done(i)
                self.land_time[i] = self.t
                self.land_deck[i] = k

        alive = [self.pos[i] for i in range(c.n_drones) if i not in self.landed]
        self.min_sep = min(self.min_sep, min_pairwise_distance(alive))
        self.t += c.dt
        return self._state()

    def _landing_velocity(self, own_pos, deck_pos, deck_vel) -> np.ndarray:
        c = self.cfg
        rel = own_pos - deck_pos
        v_xy = 1.5 * (deck_pos[:2] - own_pos[:2]) + deck_vel[:2]
        centred = float(np.linalg.norm(rel[:2])) < 0.5
        v_z = -c.land_speed if centred else 0.3 * (1.0 - own_pos[2] + deck_pos[2])
        v = np.array([v_xy[0], v_xy[1], v_z])
        s = float(np.linalg.norm(v))
        return v * (c.v_max / s) if s > c.v_max else v

    def run(self, seed: int | None = None) -> dict:
        self.reset(seed)
        for _ in range(int(self.cfg.max_time / self.cfg.dt)):
            self.step()
            if len(self.landed) == self.cfg.n_drones:
                break
        per_deck = {k: sum(1 for v in self.land_deck.values() if v == k) for k in range(self.cfg.n_decks)}
        return {
            "all_landed": len(self.landed) == self.cfg.n_drones,
            "n_landed": len(self.landed), "n_drones": self.cfg.n_drones, "n_decks": self.cfg.n_decks,
            "time": round(self.t, 2), "min_separation": round(float(self.min_sep), 3),
            "separation_ok": bool(self.min_sep >= self.cfg.d_min - 0.15),
            "min_obstacle_clearance": (None if not np.isfinite(self.min_obstacle_clear)
                                       else round(float(self.min_obstacle_clear), 3)),
            "obstacle_ok": bool(self.min_obstacle_clear >= -0.15),
            "per_deck_landed": per_deck, "n_reassign": self.n_reassign,
            "safety": self.safety.report.as_dict(),
        }
