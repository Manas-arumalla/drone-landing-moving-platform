"""MuJoCo swarm runner: the coordination layer driving N real X2 drones on true physics.

Reuses the same coordination as the kinematic `SwarmCoordinator` — optimal slot **scheduling**, the
**CBF-QP** collision-avoidance safety filter, and the **holding stack** — but each drone is now a real
MuJoCo quadrotor flown by the validated **geometric controller** (desired velocity -> accel -> attitude
-> 4 motor thrusts), landing on the moving deck through **true contact**. A drone is "landed" when its
gear is planted (support feet) and its motors are then cut, freeing its slot for the next drone.

**No-cheats (A1):** the coordination loop reads *no* ground truth. Each drone acts on its own
``SwarmSensing`` onboard view — a noisy estimate of its own state + deck pose, and delayed/dropped
neighbour broadcasts within comms range. True ``world`` state is used only for the physics, true
contact (touchdown), and the separation *metric* — never for a decision. Use
``SensingConfig.perfect()`` to reproduce the old truth baseline for honest A/B comparison.
"""

from __future__ import annotations

import numpy as np

from drone_landing.control import GeometricController
from drone_landing.estimation import quat_to_rotmat
from drone_landing.sim.platforms import RandomGroundMotion, ShipDeckMotion, sea_state
from drone_landing_swarm.avoidance import min_pairwise_distance
from drone_landing_swarm.consensus import ConsensusDeckEstimator
from drone_landing_swarm.coordinator import SwarmConfig
from drone_landing_swarm.holding import HoldingStack
from drone_landing_swarm.safety import SafetyFilter, SafetySpec
from drone_landing_swarm.scheduler import LandingScheduler, SchedulerConfig
from drone_landing_swarm.sensing import SwarmSensing
from drone_landing_swarm.vision import SwarmVision
from drone_landing_swarm.world import SwarmMujocoWorld


class MujocoSwarmCoordinator:
    """Swarm flight-deck recovery on real MuJoCo physics (N drones, one moving deck)."""

    def __init__(self, config: SwarmConfig | None = None):
        self.cfg = config or SwarmConfig()
        c = self.cfg
        self.world = SwarmMujocoWorld(c.n_drones, spawn_radius=c.spawn_radius, spawn_alt=c.spawn_alt,
                                      offshore=c.offshore)
        self.deck = (ShipDeckMotion(sea_state(c.sea)) if c.scenario == "ship"
                     else RandomGroundMotion())
        self.scheduler = LandingScheduler(c.n_drones, SchedulerConfig(n_slots=c.n_slots))
        self.sensing = SwarmSensing(c.n_drones, c.sensing)
        self.safety = SafetyFilter(SafetySpec(
            d_min=c.d_min, deck_keepout=c.deck_keepout, alpha=c.cbf_alpha,
            dt=self.world.control_dt, margin=c.safety_margin, v_max=c.v_max))
        self.deck_filter = ConsensusDeckEstimator(c.n_drones, self.world.control_dt, c.consensus_cfg)
        self.vision = (SwarmVision(self.world, camera_ids=c.camera_drones)   # CP: real per-drone vision
                       if c.vision else None)                                # camera_drones=None -> all
        self.holding = HoldingStack(c.holding)
        self.controllers = [GeometricController(self.world.mass, self.world.inertia,
                                                control_dt=self.world.control_dt)
                            for _ in range(c.n_drones)]
        self.kv = 2.5   # velocity-tracking gain: desired velocity -> horizontal accel command
        # Distinct landing spots on the deck (a ring) so drones don't stack on the centre and collide
        # with already-landed drones — like the numbered spots on a real flight deck.
        self.spots = [0.6 * np.array([np.cos(2 * np.pi * i / c.n_drones),
                                      np.sin(2 * np.pi * i / c.n_drones)])
                      for i in range(c.n_drones)] if c.n_drones > 1 else [np.zeros(2)]
        self.reset(c.seed)

    def reset(self, seed: int | None = None):
        c = self.cfg
        self.rng = np.random.default_rng(seed if seed is not None else c.seed)
        d0 = self.deck.reset(self.rng)
        self.world.reset(d0, self.rng)
        self.scheduler.reset()
        self.sensing.reset(self.rng)
        self.safety.reset()
        self.deck_filter.reset()
        if self.vision is not None:
            self.vision._k = 0
        for ctl in self.controllers:
            ctl.reset()
        self.deck_pos, self.deck_vel = d0.pos.copy(), d0.lin_vel.copy()
        self.landed: set[int] = set()
        self._settle = np.zeros(c.n_drones)
        self._seen: dict[int, bool] = {}
        self.t = 0.0
        self.min_sep = float("inf")
        self.land_time: dict[int, float] = {}
        return self._state()

    def _state(self) -> dict:
        return {"t": self.t, "deck_pos": self.deck_pos.copy(),
                "pos": {i: self.world.drone_pos(i) for i in range(self.cfg.n_drones)},
                "landed": set(self.landed), "cleared": set(self.scheduler.cleared),
                "min_sep": self.min_sep}

    def step(self) -> dict:
        c = self.cfg
        ds = self.deck.step(self.world.control_dt)
        self.deck_pos, self.deck_vel = ds.pos.copy(), ds.lin_vel.copy()

        # TRUE state (physics only — never feeds a decision below)
        pos = {i: self.world.drone_pos(i) for i in range(c.n_drones)}
        vel = {i: self.world.drone_vel(i) for i in range(c.n_drones)}

        # ---- ONBOARD VIEW (no truth in the decision loop): each drone's own noisy estimate of its
        # state + deck, and delayed/dropped neighbour broadcasts within comms range.
        view = self.sensing.sense(pos, vel, self.deck_pos, self.deck_vel, self.landed, c.comms_range)
        epos, evel, edeck, edeck_vel, neigh = (view["own_pos"], view["own_vel"], view["deck"],
                                               view["deck_vel"], view["neighbors"])
        if c.vision:
            # COOPERATIVE PERCEPTION: each drone runs REAL onboard vision; visual fixes are fused by the
            # consensus filter, and a blind drone (camera can't see the pad) lands on neighbours' shared fix.
            vis_fix = self.vision.sense(epos, float(self.deck_pos[2]), self.landed)
            self._seen = {i: vis_fix.get(i) is not None for i in vis_fix}
            stds = {i: 0.05 if vis_fix.get(i) is not None else 1.0 for i in vis_fix}
            nbr_ids = {i: list(neigh.get(i, {}).keys()) for i in vis_fix}
            fused = self.deck_filter.step(vis_fix, stds, nbr_ids)
            edeck = {i: fused[i][:3] for i in fused}
            edeck_vel = {i: fused[i][3:] for i in fused}
        elif c.consensus:
            # A2: consensus over the *modeled* deck estimate (no images).
            nbr_ids = {i: list(neigh.get(i, {}).keys()) for i in view["deck_meas"]}
            fused = self.deck_filter.step(view["deck_meas"], view["deck_std"], nbr_ids)
            edeck = {i: fused[i][:3] for i in fused}
            edeck_vel = {i: fused[i][3:] for i in fused}
        self.deck_est = edeck   # exposed for scoring (fused if consensus/vision on, else flat per-drone)

        active = np.array([i not in self.landed for i in range(c.n_drones)])
        # readiness cost = horizontal distance from the drone's *estimate* of itself to its deck estimate
        costs = np.array([float(np.linalg.norm(epos[i][:2] - edeck[i][:2])) if active[i] else 1e9
                          for i in range(c.n_drones)])
        cleared = self.scheduler.update(costs, active, self.world.control_dt)

        holders = [i for i in range(c.n_drones) if active[i] and i not in cleared]
        # slot assignment shares a deck reference (a consensus proxy = mean of the deck estimates)
        deck_ref = np.mean([edeck[i] for i in epos], axis=0) if epos else self.deck_pos
        slot_of = self.holding.assign(holders, {i: epos[i] for i in holders}, deck_ref)

        thrusts = {}
        for i in range(c.n_drones):
            if i in self.landed:
                thrusts[i] = np.zeros(4)
                continue
            if i in cleared:
                v_des = self._landing_velocity(i, epos[i], edeck[i], edeck_vel[i])
            else:
                v_des = self.holding.hold_velocity(epos[i], slot_of[i], edeck[i],
                                                   edeck_vel[i], c.v_max)
            # neighbours = delayed/dropped broadcast estimates received within comms range (no truth).
            # The non-bypassable safety filter is the last word before the command becomes thrust.
            nbrs = list(neigh.get(i, {}).values())
            deck = (edeck[i], edeck_vel[i]) if i not in cleared else None   # deck keep-out if not cleared
            obstacles = self._obstacles_for(edeck[i])      # P3: sensed static superstructure keep-outs
            v_safe = self.safety.filter(epos[i], v_des, nbrs, deck=deck, obstacles=obstacles)
            thrusts[i] = self._velocity_to_thrust(i, v_safe, evel[i])

        self.world.step(thrusts, ds)

        # touchdown: gear planted for a few steps -> landed, cut motors, free the slot
        for i in list(cleared):
            if i in self.landed:
                continue
            if self.world.support_feet(i) >= 3:
                self._settle[i] += 1
                if self._settle[i] >= 5:
                    self.landed.add(i)
                    self.scheduler.mark_done(i)
                    self.land_time[i] = self.t
            else:
                self._settle[i] = 0

        alive = [self.world.drone_pos(i) for i in range(c.n_drones) if i not in self.landed]
        self.min_sep = min(self.min_sep, min_pairwise_distance(alive))
        self.t += self.world.control_dt
        return self._state()

    def _obstacles_for(self, edeck_i: np.ndarray):
        """P3: static-obstacle keep-outs at this drone's deck estimate + the configured deck-relative
        offsets (no ground truth). ``[(world_pos, radius), ...]`` for the SafetyFilter, or None if unset."""
        if not self.cfg.obstacles:
            return None
        d = np.asarray(edeck_i, dtype=float)
        return [(np.array([d[0] + dx, d[1] + dy, d[2]]), r) for (dx, dy, r) in self.cfg.obstacles]

    def _landing_velocity(self, i: int, own_pos: np.ndarray, deck_pos: np.ndarray,
                          deck_vel: np.ndarray) -> np.ndarray:
        """Cleared-drone guidance (on the drone's own *estimates*): track its deck spot, then descend."""
        c = self.cfg
        target_xy = deck_pos[:2] + self.spots[i]           # this drone's assigned deck spot
        rel_xy = target_xy - own_pos[:2]
        v_xy = 1.5 * rel_xy + deck_vel[:2]
        centred = float(np.linalg.norm(rel_xy)) < 0.4
        v_z = -c.land_speed if centred else 0.15
        v = np.array([v_xy[0], v_xy[1], v_z])
        s = float(np.linalg.norm(v))
        return v * (c.v_max / s) if s > c.v_max else v

    def _velocity_to_thrust(self, i: int, v_des: np.ndarray, v_actual: np.ndarray) -> np.ndarray:
        """Desired world velocity -> 4 motor thrusts via the geometric controller (vel-tracking accel)."""
        a_xy = self.kv * (v_des[:2] - v_actual[:2])
        R = quat_to_rotmat(self.world.drone_quat(i))
        gyro = self.world.drone_gyro(i)
        rel_vel_ctrl = np.array([0.0, 0.0, -float(v_actual[2])])   # so controller sees drone_vz correctly
        return self.controllers[i].compute(np.zeros(3), rel_vel_ctrl, R, gyro,
                                           vz_des=float(v_des[2]), a_xy_override=a_xy)

    def run(self, seed: int | None = None) -> dict:
        self.reset(seed)
        steps = int(self.cfg.max_time / self.world.control_dt)
        for _ in range(steps):
            self.step()
            if len(self.landed) == self.cfg.n_drones:
                break
        return {
            "all_landed": len(self.landed) == self.cfg.n_drones,
            "n_landed": len(self.landed), "n_drones": self.cfg.n_drones,
            "time": round(self.t, 2), "min_separation": round(float(self.min_sep), 3),
            "separation_ok": bool(self.min_sep >= self.cfg.d_min - 0.2),
            "land_times": {i: round(t, 2) for i, t in sorted(self.land_time.items())},
            "safety": self.safety.report.as_dict(),
        }
