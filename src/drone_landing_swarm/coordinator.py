"""Swarm coordinator: N drones recovering onto one moving deck.

Ties the pieces together — deck motion (reusing the validated single-drone platform models), optimal
**slot scheduling**, the **holding stack**, the **CBF collision-avoidance** safety filter, and a
sequential landing — into one closed loop. Each step:

1. advance the moving deck (``ShipDeckMotion`` / ``RandomGroundMotion``);
2. the scheduler clears the K readiest drones to land; the rest hold;
3. the cleared drone(s) run a landing guidance toward the deck centre; holding drones track their
   assigned stack slot (which rides with the deck);
4. **every** drone's desired velocity passes through the CBF filter (pairwise separation + a deck
   keep-out column for drones not cleared), guaranteeing collision-free motion;
5. integrate, then check touchdowns and free slots for the next drones.

The drones are modelled here as velocity-controlled agents (the coordination layer is what this module
validates — scheduling, deconfliction, holding). Swapping the per-drone inner loop for the full
single-drone autopilot + MuJoCo physics is the next integration step (see docs/SWARM.md); the
coordinator API is built to allow that drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_landing.sim.platforms import RandomGroundMotion, ShipDeckMotion, sea_state
from drone_landing_swarm.avoidance import min_pairwise_distance
from drone_landing_swarm.consensus import ConsensusConfig, ConsensusDeckEstimator
from drone_landing_swarm.holding import HoldingConfig, HoldingStack
from drone_landing_swarm.safety import SafetyFilter, SafetySpec
from drone_landing_swarm.scheduler import LandingScheduler, SchedulerConfig
from drone_landing_swarm.sensing import SensingConfig, SwarmSensing


@dataclass
class SwarmConfig:
    n_drones: int = 4
    scenario: str = "ship"            # deck motion: ship | ground
    sea: str = "moderate"             # ship sea state
    n_slots: int = 1                  # drones allowed to land at once
    v_max: float = 1.5                # m/s   per-drone speed limit
    d_min: float = 0.7                # m     minimum inter-drone separation
    deck_keepout: float = 0.8         # m     keep-out radius around the deck for non-cleared drones
    land_radius: float = 0.35         # m     horizontal tolerance to count a touchdown
    land_speed: float = 0.5           # m/s   descent rate of a cleared drone
    touchdown_alt: float = 0.12       # m     height above deck counting as landed
    cbf_alpha: float = 3.0
    safety_margin: float = 0.0        # m  extra tightening on every CBF separation (robustness to A1 noise)
    comms_range: float = float("inf")  # a drone only sees neighbours within this range (m). Finite ->
                                       # partial observability: the global scheduler/CBF assumptions break
                                       # and collisions appear -> the hard regime where MARL can win.
    dt: float = 0.05
    max_time: float = 120.0
    spawn_radius: float = 3.0         # initial scatter radius
    spawn_alt: float = 2.0
    seed: int | None = None
    holding: HoldingConfig = field(default_factory=HoldingConfig)
    # Onboard sensing + comms model. Default = realistic (no-cheats); use SensingConfig.perfect() to
    # reproduce the old ground-truth baseline for A/B comparison.
    sensing: SensingConfig = field(default_factory=SensingConfig)
    # Cooperative consensus deck estimation (A2). Off by default = each drone uses its own flat deck
    # estimate (A1). On = a distributed Kalman-Consensus Filter fuses range-dependent/partial deck
    # measurements over the comms graph; well-placed observers stabilize the blind drones.
    consensus: bool = False
    consensus_cfg: ConsensusConfig = field(default_factory=ConsensusConfig)
    offshore: bool = False            # cosmetic: orange OSV-vessel look in the MuJoCo swarm world (visual only)
    # Cooperative perception (CP): each drone runs REAL onboard vision on its rendered camera (MuJoCo
    # engine only) to detect the pad; fixes are fused by the consensus filter, and a drone whose camera
    # can't see the pad lands on neighbours' shared fixes. Implies consensus fusion of visual detections.
    vision: bool = False
    # P2.4 heterogeneous fleet: only these drone indices carry a camera (None = all). Camera-less drones
    # have no visual fix and rely entirely on the camera drones' shared estimates (cooperative perception).
    camera_drones: tuple[int, ...] | None = None
    # P3 sense-and-avoid: DECK-RELATIVE static obstacles (the OSV superstructure) as (dx, dy, keep-out
    # radius) offsets from the deck origin. Each drone places them at *its own deck estimate* + offset (so
    # the obstacle positions inherit the drone's deck-estimate noise — no ground truth) and the
    # non-bypassable SafetyFilter keeps clear of them. Empty () = clear deck (no change). `swarm ... --avoid`
    # auto-populates this with the offshore superstructure.
    obstacles: tuple[tuple[float, float, float], ...] = ()


class SwarmCoordinator:
    def __init__(self, config: SwarmConfig | None = None):
        self.cfg = config or SwarmConfig()
        self.scheduler = LandingScheduler(self.cfg.n_drones, SchedulerConfig(n_slots=self.cfg.n_slots))
        self.sensing = SwarmSensing(self.cfg.n_drones, self.cfg.sensing)
        self.safety = SafetyFilter(SafetySpec(
            d_min=self.cfg.d_min, deck_keepout=self.cfg.deck_keepout, alpha=self.cfg.cbf_alpha,
            dt=self.cfg.dt, margin=self.cfg.safety_margin, v_max=self.cfg.v_max))
        self.deck_filter = ConsensusDeckEstimator(self.cfg.n_drones, self.cfg.dt, self.cfg.consensus_cfg)
        self.holding = HoldingStack(self.cfg.holding)
        self._deck = self._make_deck()
        self.reset(self.cfg.seed)

    def _make_deck(self):
        if self.cfg.scenario == "ship":
            return ShipDeckMotion(sea_state(self.cfg.sea))
        return RandomGroundMotion()

    def reset(self, seed: int | None = None):
        c = self.cfg
        self.rng = np.random.default_rng(seed if seed is not None else c.seed)
        ps = self._deck.reset(self.rng)
        self.deck_pos = ps.pos.copy()
        self.deck_vel = ps.lin_vel.copy()
        self.scheduler.reset()
        # scatter drones around the deck at altitude, on a ring so they start separated
        self.pos = {}
        for i in range(c.n_drones):
            ang = 2 * np.pi * i / c.n_drones + self.rng.uniform(-0.2, 0.2)
            r = c.spawn_radius * (0.7 + 0.3 * self.rng.random())
            self.pos[i] = self.deck_pos + np.array([r * np.cos(ang), r * np.sin(ang),
                                                    c.spawn_alt + 0.3 * self.rng.random()])
        self.vel = {i: np.zeros(3) for i in range(c.n_drones)}
        self.landed: set[int] = set()
        self.t = 0.0
        self.min_sep = float("inf")
        self.min_obstacle_clear = float("inf")             # P3: true min clearance to any static obstacle
        self.land_time: dict[int, float] = {}
        self.policy_residual: dict[int, np.ndarray] = {}   # per-drone horizontal velocity residual (MARL)
        self.sensing.reset(self.rng)
        self.safety.reset()
        self.deck_filter.reset()
        return self._state()

    # ------------------------------------------------------------ MARL interface
    def neighbors_in_range(self, i: int) -> list[int]:
        """Active neighbours of drone i within comms range (sorted nearest-first)."""
        c = self.cfg
        js = [j for j in range(c.n_drones) if j != i and j not in self.landed
              and np.linalg.norm(self.pos[j] - self.pos[i]) <= c.comms_range]
        return sorted(js, key=lambda j: np.linalg.norm(self.pos[j] - self.pos[i]))

    def local_obs(self, i: int, k_neighbors: int = 3) -> np.ndarray:
        """Decentralized observation for drone i: own state rel. deck + nearest-K neighbour rel. states.

        Uses only information available within the comms range (partial observability).
        """
        rel_deck = self.deck_pos - self.pos[i]
        own = np.concatenate([rel_deck, self.vel[i]])                  # 6
        feats = [own, np.array([1.0 if i in self.scheduler.cleared else 0.0])]
        nbrs = self.neighbors_in_range(i)[:k_neighbors]
        for j in nbrs:
            feats.append(np.concatenate([self.pos[j] - self.pos[i], self.vel[j] - self.vel[i]]))  # 6
        for _ in range(k_neighbors - len(nbrs)):
            feats.append(np.zeros(6))                                   # zero-pad missing neighbours
        return np.concatenate(feats).astype(np.float32)                # 6 + 1 + 6K

    def local_obs_graph(self, i: int, max_neighbors: int = 8) -> np.ndarray:
        """Graph observation for the GNN policy (A4): ego features + a *variable* set of in-range
        neighbour features, zero-padded to ``max_neighbors`` with a validity mask.

        Layout (flat, for a Box space): ``[ego(7)] + [neighbour feats (max_neighbors x 6)] + [mask(max_n)]``.
        The set + mask is permutation-invariant and size-agnostic, so one policy runs at any swarm size.
        """
        rel_deck = self.deck_pos - self.pos[i]
        cleared = 1.0 if i in self.scheduler.cleared else 0.0
        ego = np.concatenate([rel_deck, self.vel[i], [cleared]])           # 7
        feats = np.zeros((max_neighbors, 6), dtype=np.float32)
        mask = np.zeros(max_neighbors, dtype=np.float32)
        for idx, j in enumerate(self.neighbors_in_range(i)[:max_neighbors]):
            feats[idx] = np.concatenate([self.pos[j] - self.pos[i], self.vel[j] - self.vel[i]])
            mask[idx] = 1.0
        return np.concatenate([ego, feats.ravel(), mask]).astype(np.float32)  # 7 + 6*M + M

    def _state(self) -> dict:
        return {"t": self.t, "deck_pos": self.deck_pos.copy(), "pos": {i: p.copy() for i, p in self.pos.items()},
                "landed": set(self.landed), "cleared": set(self.scheduler.cleared), "min_sep": self.min_sep}

    def step(self) -> dict:
        c = self.cfg
        ps = self._deck.step(c.dt)
        self.deck_pos, self.deck_vel = ps.pos.copy(), ps.lin_vel.copy()

        # ---- ONBOARD VIEW (no truth in the decision loop): each drone's own noisy estimate of its
        # state + deck, and delayed/dropped neighbour broadcasts within comms range.
        view = self.sensing.sense(self.pos, self.vel, self.deck_pos, self.deck_vel,
                                  self.landed, c.comms_range)
        epos, evel, edeck, edeck_vel, neigh = (view["own_pos"], view["own_vel"], view["deck"],
                                               view["deck_vel"], view["neighbors"])
        # ---- A2 cooperative consensus: fuse range-dependent/partial deck measurements over the comms
        # graph so the well-placed observers stabilize the blind drones. Off -> each drone's flat estimate.
        if c.consensus:
            nbr_ids = {i: list(neigh.get(i, {}).keys()) for i in view["deck_meas"]}
            fused = self.deck_filter.step(view["deck_meas"], view["deck_std"], nbr_ids)
            edeck = {i: fused[i][:3] for i in fused}
            edeck_vel = {i: fused[i][3:] for i in fused}
        self.deck_est = edeck   # exposed for scoring (fused if consensus on, else flat per-drone)

        active = np.array([i not in self.landed for i in range(c.n_drones)])
        # readiness cost = horizontal distance from the drone's *estimate* of itself to its deck estimate
        costs = np.array([float(np.linalg.norm(epos[i][:2] - edeck[i][:2]))
                          if active[i] else 1e9 for i in range(c.n_drones)])
        cleared = self.scheduler.update(costs, active, c.dt)

        holders = [i for i in range(c.n_drones) if active[i] and i not in cleared]
        # slot assignment shares a deck reference (a consensus proxy = mean of the deck estimates)
        deck_ref = np.mean([edeck[i] for i in epos], axis=0) if epos else self.deck_pos
        slot_of = self.holding.assign(holders, {i: epos[i] for i in holders}, deck_ref)

        new_vel = {}
        for i in range(c.n_drones):
            if i in self.landed:
                continue
            if i in cleared:
                v_des = self._landing_velocity(epos[i], edeck[i], edeck_vel[i])
            else:
                v_des = self.holding.hold_velocity(epos[i], slot_of[i], edeck[i],
                                                   edeck_vel[i], c.v_max)
            # MARL residual (horizontal): a learned correction on the classical guidance (0 = classical).
            # It modifies the *desired* velocity only — the non-bypassable safety filter below is the
            # last word, so no learned command ever reaches the plant unfiltered.
            res = self.policy_residual.get(i)
            if res is not None:
                v_des = v_des + np.array([res[0], res[1], 0.0])
            # neighbours = delayed/dropped broadcast estimates received within comms range (no truth).
            nbrs = list(neigh.get(i, {}).values())
            deck = (edeck[i], edeck_vel[i]) if i not in cleared else None   # deck keep-out if not cleared
            new_vel[i] = self.safety.filter(epos[i], v_des, nbrs, deck=deck,
                                            obstacles=self._obstacles_for(edeck[i]))

        for i, v in new_vel.items():
            self.vel[i] = v
            self.pos[i] = self.pos[i] + v * c.dt

        # touchdowns
        for i in list(cleared):
            if i in self.landed:
                continue
            rel = self.pos[i] - self.deck_pos
            if np.linalg.norm(rel[:2]) <= c.land_radius and rel[2] <= c.touchdown_alt:
                self.landed.add(i)
                self.scheduler.mark_done(i)
                self.land_time[i] = self.t

        # safety metric
        alive = [self.pos[i] for i in range(c.n_drones) if i not in self.landed]
        self.min_sep = min(self.min_sep, min_pairwise_distance(alive))
        self._update_obstacle_clearance()
        self.t += c.dt
        return self._state()

    def _obstacles_for(self, edeck_i: np.ndarray):
        """P3: per-drone static-obstacle keep-outs at THIS drone's deck estimate + the configured offsets
        (so the obstacle positions carry the drone's own deck-estimate noise — no ground truth). Returns
        ``[(world_pos, radius), ...]`` for the SafetyFilter, or None when no obstacles are configured."""
        if not self.cfg.obstacles:
            return None
        d = np.asarray(edeck_i, dtype=float)
        return [(np.array([d[0] + dx, d[1] + dy, d[2]]), r) for (dx, dy, r) in self.cfg.obstacles]

    def _update_obstacle_clearance(self) -> None:
        """True (metric-only) min clearance of any *airborne* drone to any obstacle surface."""
        if not self.cfg.obstacles:
            return
        dp = self.deck_pos
        obs = [(np.array([dp[0] + dx, dp[1] + dy, dp[2]]), r) for (dx, dy, r) in self.cfg.obstacles]
        for i in range(self.cfg.n_drones):
            if i in self.landed:
                continue
            for p_o, r in obs:
                self.min_obstacle_clear = min(self.min_obstacle_clear,
                                              float(np.linalg.norm(self.pos[i] - p_o)) - r)

    def _landing_velocity(self, own_pos: np.ndarray, deck_pos: np.ndarray,
                          deck_vel: np.ndarray) -> np.ndarray:
        """Cleared-drone guidance (on the drone's own *estimates*): track the deck, descend when centred."""
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
        steps = int(self.cfg.max_time / self.cfg.dt)
        for _ in range(steps):
            self.step()
            if len(self.landed) == self.cfg.n_drones:
                break
        return {
            "all_landed": len(self.landed) == self.cfg.n_drones,
            "n_landed": len(self.landed),
            "n_drones": self.cfg.n_drones,
            "time": round(self.t, 2),
            "min_separation": round(float(self.min_sep), 3),
            "separation_ok": bool(self.min_sep >= self.cfg.d_min - 0.15),
            "min_obstacle_clearance": (None if not np.isfinite(self.min_obstacle_clear)
                                       else round(float(self.min_obstacle_clear), 3)),
            "obstacle_ok": bool(self.min_obstacle_clear >= -0.15),
            "land_times": {i: round(t, 2) for i, t in sorted(self.land_time.items())},
            "safety": self.safety.report.as_dict(),
        }
