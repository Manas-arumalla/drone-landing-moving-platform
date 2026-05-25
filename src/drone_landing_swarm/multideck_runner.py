"""MuJoCo multi-deck coordinator (A5 on true physics): M real X2 drones -> K moving decks.

The physics counterpart of the kinematic :class:`~drone_landing_swarm.multi_deck.MultiDeckCoordinator`.
Reuses that coordinator's *decision* layer unchanged — balanced **Hungarian** initial allocation +
decentralized **auction** re-tasking, per-deck :class:`LandingScheduler` + :class:`HoldingStack`, the
no-cheats :class:`SwarmSensing` onboard view, and the non-bypassable :class:`SafetyFilter` — but each drone
is a real MuJoCo X2 flown by the validated geometric controller onto a 6-DOF servo deck, and a drone is
"landed" only when its gear is **actually planted** on its assigned deck (true contact). This removes the
kinematic point-mass shortcut from the multi-deck scenario.

No ground truth in the decision loop: drones act on their own ``SwarmSensing`` estimate + a noisy per-deck
estimate (the A5 shared-observer convention); true state is used only for physics, contact, and the
separation metric.
"""

from __future__ import annotations

import numpy as np

from drone_landing.control import GeometricController
from drone_landing.estimation import quat_to_rotmat
from drone_landing.sim.platforms import ShipDeckMotion, sea_state
from drone_landing_swarm.avoidance import min_pairwise_distance
from drone_landing_swarm.holding import HoldingStack
from drone_landing_swarm.multi_deck import MultiDeckConfig
from drone_landing_swarm.multideck_world import MultiDeckMujocoWorld
from drone_landing_swarm.safety import SafetyFilter, SafetySpec
from drone_landing_swarm.scheduler import LandingScheduler, SchedulerConfig, optimal_assignment
from drone_landing_swarm.sensing import SwarmSensing


class MujocoMultiDeckCoordinator:
    """M real X2 drones recovering onto K moving decks on true MuJoCo physics + contact."""

    def __init__(self, config: MultiDeckConfig | None = None):
        self.cfg = config or MultiDeckConfig()
        c = self.cfg
        # decks spread on a ring (same geometry as the kinematic A5 deck_base offsets)
        self.deck_bases = [(c.deck_spacing * np.cos(2 * np.pi * k / c.n_decks),
                            c.deck_spacing * np.sin(2 * np.pi * k / c.n_decks))
                           for k in range(c.n_decks)]
        self.world = MultiDeckMujocoWorld(c.n_drones, self.deck_bases,
                                          spawn_radius=c.spawn_radius, spawn_alt=c.spawn_alt)
        self.decks = [self._make_deck(k) for k in range(c.n_decks)]
        self.scheduler = [LandingScheduler(c.n_drones, SchedulerConfig(n_slots=c.n_slots))
                          for _ in range(c.n_decks)]
        self.holding = [HoldingStack(c.holding) for _ in range(c.n_decks)]
        self.sensing = SwarmSensing(c.n_drones, c.sensing)
        self.safety = SafetyFilter(SafetySpec(
            d_min=c.d_min, deck_keepout=c.deck_keepout, alpha=c.cbf_alpha,
            dt=self.world.control_dt, margin=c.safety_margin, v_max=c.v_max))
        self.controllers = [GeometricController(self.world.mass, self.world.inertia,
                                                control_dt=self.world.control_dt)
                            for _ in range(c.n_drones)]
        self.kv = 2.5
        # distinct on-deck spots (a ring) so a second drone recovering onto the same deck lands BESIDE an
        # already-landed drone instead of on top of it (like the numbered spots on a real flight deck)
        self.spots = [0.45 * np.array([np.cos(2 * np.pi * i / c.n_drones),
                                       np.sin(2 * np.pi * i / c.n_drones)])
                      for i in range(c.n_drones)]
        self.reset(c.seed)

    def _make_deck(self, k: int):
        from dataclasses import replace as _replace
        if self.cfg.scenario == "ship":
            base = sea_state(self.cfg.sea)
            return ShipDeckMotion(_replace(base, heading=2 * np.pi * k / max(self.cfg.n_decks, 1)))
        from drone_landing.sim.platforms import RandomGroundMotion
        return RandomGroundMotion()

    def reset(self, seed: int | None = None):
        c = self.cfg
        self.rng = np.random.default_rng(seed if seed is not None else c.seed)
        d0 = [deck.reset(self.rng) for deck in self.decks]
        # shift each deck's motion to sit on its ring base in the world
        self.world.reset(d0, self.rng)
        for s in self.scheduler:
            s.reset()
        self.sensing.reset(self.rng)
        self.safety.reset()
        for ctl in self.controllers:
            ctl.reset()
        self.deck_pos = [self.world.deck_state(k)[0] for k in range(c.n_decks)]
        self.deck_vel = [self.world.deck_state(k)[1] for k in range(c.n_decks)]
        self.landed: set[int] = set()
        self.land_deck: dict[int, int] = {}
        self.land_time: dict[int, float] = {}
        self._settle = np.zeros(c.n_drones)
        self.available = [True] * c.n_decks
        self.t = 0.0
        self.min_sep = float("inf")
        self.n_reassign = 0
        self._last_assign_t = -1e9
        self.assign = self._initial_assignment()
        return self._state()

    # ----------------------------------------------------------- estimates / assignment
    def _deck_estimates(self) -> list[np.ndarray]:
        s = self.cfg.sensing.deck_noise
        return [self.deck_pos[k] + self.rng.normal(0, s, 3) for k in range(self.cfg.n_decks)]

    def _drone_xy_est(self) -> dict[int, np.ndarray]:
        return {i: self.world.drone_pos(i) for i in range(self.cfg.n_drones)}

    def _initial_assignment(self) -> dict[int, int]:
        from math import ceil
        c = self.cfg
        edeck = self._deck_estimates()
        pos = self._drone_xy_est()
        avail = [k for k in range(c.n_decks) if self.available[k]]
        cap = ceil(c.n_drones / max(len(avail), 1))
        cost = np.zeros((c.n_drones, len(avail) * cap))
        for i in range(c.n_drones):
            for ci, k in enumerate(avail):
                cost[i, ci * cap:(ci + 1) * cap] = float(np.linalg.norm(pos[i][:2] - edeck[k][:2]))
        assign = {}
        for r, col in optimal_assignment(cost):
            assign[r] = avail[col // cap]
        return assign

    def _retask(self, epos, edeck, cleared: set[int]) -> None:
        c = self.cfg
        active = [i for i in range(c.n_drones) if i not in self.landed]
        avail = [k for k in range(c.n_decks) if self.available[k]]

        def load(k, exclude=None):
            return sum(1 for j in active if self.assign[j] == k and j != exclude)

        def eff_cost(i, k):
            return float(np.linalg.norm(epos[i][:2] - edeck[k][:2])) + c.congestion_weight * load(k, i)

        for i in active:
            cur = self.assign[i]
            if i in cleared and self.available[cur]:
                continue
            best = min(avail, key=lambda k: eff_cost(i, k))
            if best != cur and (not self.available[cur]
                                or eff_cost(i, cur) - eff_cost(i, best) > c.reassign_hysteresis):
                self.assign[i] = best
                self.n_reassign += 1

    def _state(self) -> dict:
        return {"t": self.t, "deck_pos": [p.copy() for p in self.deck_pos],
                "pos": {i: self.world.drone_pos(i) for i in range(self.cfg.n_drones)},
                "assign": dict(self.assign), "landed": set(self.landed), "min_sep": self.min_sep}

    # --------------------------------------------------------------------- step
    def step(self) -> dict:
        c = self.cfg
        ds = [deck.step(self.world.control_dt) for deck in self.decks]
        self.deck_pos = [self.world.deck_state(k)[0] for k in range(c.n_decks)]
        self.deck_vel = [self.world.deck_state(k)[1] for k in range(c.n_decks)]

        pos = {i: self.world.drone_pos(i) for i in range(c.n_drones)}
        vel = {i: self.world.drone_vel(i) for i in range(c.n_drones)}
        view = self.sensing.sense(pos, vel, self.deck_pos[0], self.deck_vel[0], self.landed, c.comms_range)
        epos, evel, neigh = view["own_pos"], view["own_vel"], view["neighbors"]
        edeck = self._deck_estimates()

        # "fouled deck" event: a deck closes mid-recovery -> its drones must re-task to the others
        if c.offline_deck is not None and self.available[c.offline_deck] and self.t >= c.offline_time:
            self.available[c.offline_deck] = False
            self.scheduler[c.offline_deck].cleared.clear()

        # per-deck clearance among that deck's assigned, active drones
        cleared_all: set[int] = set()
        cleared_by_deck: list[set[int]] = []
        for k in range(c.n_decks):
            assigned = np.array([(i not in self.landed) and (self.assign[i] == k) and self.available[k]
                                 for i in range(c.n_drones)])
            costs = np.array([float(np.linalg.norm(epos[i][:2] - edeck[k][:2])) if assigned[i] else 1e9
                              for i in range(c.n_drones)])
            ck = self.scheduler[k].update(costs, assigned, self.world.control_dt)
            cleared_by_deck.append(ck)
            cleared_all |= ck

        if self.t - self._last_assign_t >= c.reassign_dt:
            self._retask(epos, edeck, cleared_all)
            self._last_assign_t = self.t

        # holding-slot assignment per deck
        slot_of: dict[int, int] = {}
        for k in range(c.n_decks):
            holders = [i for i in range(c.n_drones)
                       if i not in self.landed and self.assign[i] == k and i not in cleared_by_deck[k]]
            slot_of.update(self.holding[k].assign(holders, {i: epos[i] for i in holders}, edeck[k]))

        thrusts = {}
        for i in range(c.n_drones):
            if i in self.landed:
                thrusts[i] = np.zeros(4)
                continue
            k = self.assign[i]
            if i in cleared_by_deck[k]:
                v_des = self._landing_velocity(i, epos[i], edeck[k], self.deck_vel[k])
            else:
                v_des = self.holding[k].hold_velocity(epos[i], slot_of.get(i, 0), edeck[k],
                                                      self.deck_vel[k], c.v_max)
            nbrs = list(neigh.get(i, {}).values())
            deck = (edeck[k], self.deck_vel[k]) if i not in cleared_all else None
            obstacles = None
            if c.obstacles:                                # P3: keep clear of the assigned vessel's structure
                d = edeck[k]
                obstacles = [(np.array([d[0] + dx, d[1] + dy, d[2]]), r) for (dx, dy, r) in c.obstacles]
            v_safe = self.safety.filter(epos[i], v_des, nbrs, deck=deck, obstacles=obstacles)
            thrusts[i] = self._velocity_to_thrust(i, v_safe, evel[i])

        self.world.step(thrusts, ds)

        # touchdown: gear planted on the ASSIGNED deck for a few steps
        for i in list(cleared_all):
            if i in self.landed:
                continue
            if self.world.support_feet(i, self.assign[i]) >= 3:
                self._settle[i] += 1
                if self._settle[i] >= 5:
                    self.landed.add(i)
                    self.scheduler[self.assign[i]].mark_done(i)
                    self.land_time[i] = self.t
                    self.land_deck[i] = self.assign[i]
            else:
                self._settle[i] = 0

        alive = [self.world.drone_pos(i) for i in range(c.n_drones) if i not in self.landed]
        self.min_sep = min(self.min_sep, min_pairwise_distance(alive))
        self.t += self.world.control_dt
        return self._state()

    def _landing_velocity(self, i, own_pos, deck_pos, deck_vel) -> np.ndarray:
        c = self.cfg
        target_xy = deck_pos[:2] + self.spots[i]          # this drone's assigned spot on the deck
        rel_xy = target_xy - own_pos[:2]
        v_xy = 1.5 * rel_xy + deck_vel[:2]
        centred = float(np.linalg.norm(rel_xy)) < 0.4
        v_z = -c.land_speed if centred else 0.15
        v = np.array([v_xy[0], v_xy[1], v_z])
        s = float(np.linalg.norm(v))
        return v * (c.v_max / s) if s > c.v_max else v

    def _velocity_to_thrust(self, i, v_des, v_actual) -> np.ndarray:
        a_xy = self.kv * (v_des[:2] - v_actual[:2])
        R = quat_to_rotmat(self.world.drone_quat(i))
        gyro = self.world.drone_gyro(i)
        rel_vel_ctrl = np.array([0.0, 0.0, -float(v_actual[2])])
        return self.controllers[i].compute(np.zeros(3), rel_vel_ctrl, R, gyro,
                                           vz_des=float(v_des[2]), a_xy_override=a_xy)

    def run(self, seed: int | None = None) -> dict:
        self.reset(seed)
        steps = int(self.cfg.max_time / self.world.control_dt)
        for _ in range(steps):
            self.step()
            if len(self.landed) == self.cfg.n_drones:
                break
        per_deck = {k: sum(1 for v in self.land_deck.values() if v == k) for k in range(self.cfg.n_decks)}
        return {
            "all_landed": len(self.landed) == self.cfg.n_drones,
            "n_landed": len(self.landed), "n_drones": self.cfg.n_drones, "n_decks": self.cfg.n_decks,
            "time": round(self.t, 2), "min_separation": round(float(self.min_sep), 3),
            "separation_ok": bool(self.min_sep >= self.cfg.d_min - 0.2),
            "per_deck_landed": per_deck, "n_reassign": self.n_reassign,
            "safety": self.safety.report.as_dict(),
        }
