"""High-fidelity MuJoCo landing world.

Wraps the X2 landing scene with a moving platform, realistic sensors, wind, and **strict
contact-based** touchdown detection (no weld/teleport lock). It exposes two views of the world:

* :meth:`LandingWorld.observe_truth` — ground-truth privileged state, for evaluation metrics and the
  RL critic only.
* the :class:`~drone_landing.sim.sensors.SensorReading` returned by :meth:`step` — what a real
  onboard controller/estimator is allowed to use.

The platform is a *dynamic* body driven by position servos (not a kinematic mocap body), so contact
friction correctly carries a landed drone along with the moving deck. Per docs/REALISM_CHARTER.md,
controllers must fly on sensor-derived state, never on truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from drone_landing.sim.airwake import AirwakeConfig, ShipAirwake
from drone_landing.sim.mjcf import load_model
from drone_landing.sim.platforms import (
    GroundMotionConfig,
    PlatformMotion,
    RandomGroundMotion,
    ShipDeckMotion,
    ShipMotionConfig,
)
from drone_landing.sim.sensors import SensorReading, SensorSuite, SensorSuiteConfig

# Platform joints in qpos order. Ground worlds expose the first three; ship worlds expose all six.
PLATFORM_JOINTS = ("plat_x", "plat_y", "plat_z", "plat_yaw", "plat_pitch", "plat_roll")

Termination = Literal["running", "success", "crash", "off_platform", "out_of_bounds", "timeout",
                      "hit_structure"]


@dataclass(frozen=True)
class LandingWorldConfig:
    world: str = "x2_landing_ground"
    control_hz: float = 100.0
    max_time: float = 25.0
    init_alt: float = 2.5
    init_alt_spread: float = 0.4
    init_xy_spread: float = 1.5
    bounds_xy: float = 6.0
    max_alt: float = 6.0
    # touchdown success (sustained for settle_time). 0.40 m matches the larger 1.8 m deck: a touchdown
    # anywhere on the central fiducial pad counts (the supervisor's align_radius still aims for centre).
    success_radius: float = 0.40
    success_max_vspeed: float = 0.6      # |v_z(drone) - v_z(deck)|
    success_max_relspeed: float = 0.4    # horizontal speed relative to deck
    success_max_tilt_deg: float = 18.0
    settle_time: float = 0.5         # s   sustained stable contact confirming a stable touchdown
    min_support_feet: int = 3
    # crash conditions
    tipover_deg: float = 50.0
    max_impact_vspeed: float = 2.0
    # Wind as a body force [N] (drone weight ~13.5 N): a steady breeze + correlated (Ornstein-
    # Uhlenbeck) gusts, the standard low-order turbulence surrogate. Modest by default so the world is
    # realistically disturbed without dominating; a full Dryden/ship air-wake model arrives later.
    wind_mean: tuple[float, float, float] = (0.2, 0.12, 0.0)
    wind_gust_std: float = 0.2          # N   gust magnitude (per-axis std of the OU process)
    wind_gust_tau: float = 1.2          # s   gust correlation time (OU relaxation)
    # Ship air-wake ("burble"): position-dependent turbulence behind the superstructure, over the deck
    # (ship scenario only). Off by default; the dominant real shipboard-landing disturbance.
    airwake: bool = False
    airwake_cfg: AirwakeConfig = field(default_factory=AirwakeConfig)
    # Rotor failure injection (fault-tolerance demo): physically kill rotor `failed_rotor` at `fail_time`.
    failed_rotor: int | None = None
    fail_time: float = 3.0              # s
    seed: int | None = None
    ground: GroundMotionConfig = field(default_factory=GroundMotionConfig)
    ship: ShipMotionConfig = field(default_factory=ShipMotionConfig)
    sensors: SensorSuiteConfig = field(default_factory=SensorSuiteConfig)


class LandingWorld:
    def __init__(
        self,
        config: LandingWorldConfig | None = None,
        platform: PlatformMotion | None = None,
    ):
        self.config = config or LandingWorldConfig()
        self.model = load_model(self.config.world)
        self.data = mujoco.MjData(self.model)
        self.platform = platform or self._default_platform()
        self.sensors = SensorSuite(self.config.sensors)
        self.rng = np.random.default_rng(self.config.seed)
        # ship air-wake (burble) — only meaningful for the ship scenario
        self._is_ship = "ship" in self.config.world
        self.airwake = ShipAirwake(self.config.airwake_cfg) if (self.config.airwake and self._is_ship) else None
        self._deck_pos = np.zeros(3)

        self.timestep = float(self.model.opt.timestep)
        self.control_dt = 1.0 / self.config.control_hz
        self.n_substeps = max(1, round(self.control_dt / self.timestep))
        self.n_rotors = 4

        self._cache_ids()
        self.thrust_max = float(self.model.actuator_ctrlrange[self.thrust_act[0], 1])
        self.steps = 0
        self.sim_time = 0.0
        self._settle_steps_needed = round(self.config.settle_time / self.control_dt)
        self._settle_counter = 0
        self._first_contact_logged = False
        self.reset(self.config.seed)

    def _default_platform(self) -> PlatformMotion:
        """Pick the motion model that matches the loaded world (6-DOF seakeeping vs. ground rover)."""
        w = self.config.world
        if "ship" in w or "offshore" in w:        # both are wave-driven 6-DOF decks
            return ShipDeckMotion(self.config.ship)
        return RandomGroundMotion(self.config.ground)

    @property
    def _is_seakeeping(self) -> bool:
        return "ship" in self.config.world or "offshore" in self.config.world

    # ------------------------------------------------------------------ setup
    def _cache_ids(self) -> None:
        m, M = self.model, mujoco.mjtObj
        self.drone_bid = mujoco.mj_name2id(m, M.mjOBJ_BODY, "x2")
        jid = mujoco.mj_name2id(m, M.mjOBJ_JOINT, "root")
        self.qadr = int(m.jnt_qposadr[jid])
        self.vadr = int(m.jnt_dofadr[jid])

        # Platform slide/hinge joints actually present in this world (3 for ground, 6 for ship).
        self.plat_joints = [n for n in PLATFORM_JOINTS
                            if mujoco.mj_name2id(m, M.mjOBJ_JOINT, n) != -1]
        self.plat_qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, M.mjOBJ_JOINT, n)])
                          for n in self.plat_joints}
        self.plat_vadr = {n: int(m.jnt_dofadr[mujoco.mj_name2id(m, M.mjOBJ_JOINT, n)])
                          for n in self.plat_joints}

        def act(name: str) -> int:
            return mujoco.mj_name2id(m, M.mjOBJ_ACTUATOR, name)

        self.thrust_act = [act(n) for n in ("thrust1", "thrust2", "thrust3", "thrust4")]
        self.plat_act = {n: act(f"{n}_srv") for n in self.plat_joints}

        self.gimbal_bid = mujoco.mj_name2id(m, M.mjOBJ_BODY, "gimbal")
        self.gimbal_mocap = int(m.body_mocapid[self.gimbal_bid])
        self.cam_mount_z = -0.09  # camera position in the drone body frame

        self.deck_gid = mujoco.mj_name2id(m, M.mjOBJ_GEOM, "deck")
        self.floor_gid = mujoco.mj_name2id(m, M.mjOBJ_GEOM, "floor")
        self.foot_gids = {mujoco.mj_name2id(m, M.mjOBJ_GEOM, n)
                          for n in ("foot_fl", "foot_fr", "foot_rl", "foot_rr")}
        self.leg_gids = {mujoco.mj_name2id(m, M.mjOBJ_GEOM, n)
                         for n in ("leg_fl", "leg_fr", "leg_rl", "leg_rr")}
        self.gear_gids = self.foot_gids | self.leg_gids
        self.body_hard_gids = {
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] == self.drone_bid and m.geom_contype[g] != 0 and g not in self.gear_gids
        }
        # Collidable static superstructure (offshore only): a drone touching any of these is a crash into
        # the structure -> the `--avoid` sense-and-avoid guard exists to prevent exactly this. Absent on the
        # clear-deck ground/ship worlds (the set is then empty -> no effect).
        self.obstacle_gids = set()
        for n in ("wheelhouse", "wheelhouse_base", "mast", "bow"):
            gid = mujoco.mj_name2id(m, M.mjOBJ_GEOM, n)
            if gid >= 0:
                self.obstacle_gids.add(gid)
        self.deck_top_z = float(m.geom_pos[self.deck_gid][2] + m.geom_size[self.deck_gid][2])

        def sensor(name: str) -> int:
            return int(m.sensor_adr[mujoco.mj_name2id(m, M.mjOBJ_SENSOR, name)])

        self.s_gyro = sensor("body_gyro")
        self.s_accel = sensor("body_linacc")
        self.s_quat = sensor("body_quat")
        self.s_range = sensor("altitude")

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> SensorReading:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.sim_time = 0.0
        self._settle_counter = 0
        self._first_contact_logged = False
        self._gust = np.zeros(3)   # OU wind-gust state
        if self.airwake is not None:
            self.airwake.reset(self.rng)

        p0 = self.platform.reset(self.rng)
        self._deck_pos = np.asarray(p0.pos, dtype=float).copy()
        targets = self._platform_targets(p0)
        for n in self.plat_joints:
            self.data.qpos[self.plat_qadr[n]] = targets[n]
        self._drive_platform(p0)

        c = self.config
        self.data.qpos[self.qadr : self.qadr + 3] = [
            p0.pos[0] + self.rng.uniform(-c.init_xy_spread, c.init_xy_spread),
            p0.pos[1] + self.rng.uniform(-c.init_xy_spread, c.init_xy_spread),
            c.init_alt + self.rng.uniform(-c.init_alt_spread, c.init_alt_spread),
        ]
        self.data.qpos[self.qadr + 3 : self.qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.sensors.reset(self.rng)
        self._drive_gimbal()
        mujoco.mj_forward(self.model, self.data)
        return self._read_sensors(self.control_dt)

    # ------------------------------------------------------------------- step
    def step(self, ctrl) -> "WorldStep":
        ctrl = np.clip(np.asarray(ctrl, dtype=float), 0.0, self.thrust_max)
        if ctrl.shape != (self.n_rotors,):
            raise ValueError(f"ctrl must have shape ({self.n_rotors},), got {ctrl.shape}")

        # Rotor failure injection: once failed, that motor produces no thrust regardless of command.
        c = self.config
        if c.failed_rotor is not None and self.sim_time >= c.fail_time:
            ctrl = ctrl.copy()
            ctrl[c.failed_rotor] = 0.0

        self.steps += 1
        for _ in range(self.n_substeps):
            ps = self.platform.step(self.timestep)
            self._deck_pos = np.asarray(ps.pos, dtype=float)
            self._drive_platform(ps)
            self._drive_gimbal()
            for k, aid in enumerate(self.thrust_act):
                self.data.ctrl[aid] = ctrl[k]
            self._apply_wind()
            mujoco.mj_step(self.model, self.data)
        self.sim_time = self.steps * self.control_dt

        sensors = self._read_sensors(self.control_dt)
        termination = self._termination()
        truth = self.observe_truth()
        terminated = termination in {"success", "crash", "off_platform", "out_of_bounds", "hit_structure"}
        truncated = termination == "timeout"
        info = {"termination": termination, **truth}
        return WorldStep(sensors=sensors, truth=truth, terminated=terminated,
                         truncated=truncated, info=info)

    # ----------------------------------------------------------- observations
    def platform_truth(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (pos[3], vel[3], yaw) of the deck from the actual physics state.

        ``pos[2]`` is the world deck-top height: the local deck-top offset plus the heave joint
        (``plat_z``) when the world has one (ship); a ground rover has no vertical DOF so it stays
        at the fixed deck height.
        """
        px = self.data.qpos[self.plat_qadr["plat_x"]]
        py = self.data.qpos[self.plat_qadr["plat_y"]]
        heave = self.data.qpos[self.plat_qadr["plat_z"]] if "plat_z" in self.plat_qadr else 0.0
        vz = self.data.qvel[self.plat_vadr["plat_z"]] if "plat_z" in self.plat_vadr else 0.0
        yaw = self.data.qpos[self.plat_qadr["plat_yaw"]]
        vx = self.data.qvel[self.plat_vadr["plat_x"]]
        vy = self.data.qvel[self.plat_vadr["plat_y"]]
        return (np.array([px, py, self.deck_top_z + heave]),
                np.array([vx, vy, vz]), float(yaw))

    def observe_truth(self) -> dict:
        pos = self.data.qpos[self.qadr : self.qadr + 3].copy()
        quat = self.data.qpos[self.qadr + 3 : self.qadr + 7].copy()
        lin = self.data.qvel[self.vadr : self.vadr + 3].copy()
        ang = self.data.qvel[self.vadr + 3 : self.vadr + 6].copy()
        ppos, pvel, pyaw = self.platform_truth()
        rel = pos - ppos
        return {
            "drone_pos": pos, "drone_quat": quat, "drone_vel": lin, "drone_angvel": ang,
            "platform_pos": ppos, "platform_vel": pvel, "platform_yaw": pyaw,
            "rel_pos": rel, "horizontal_error": float(np.hypot(rel[0], rel[1])),
            "relative_horizontal_speed": float(np.hypot(lin[0] - pvel[0], lin[1] - pvel[1])),
            "vertical_speed": float(abs(lin[2] - pvel[2])),
            "tilt_deg": self._tilt_deg(quat), "support_feet": self._support_feet(),
            "time": self.sim_time,
        }

    def _read_sensors(self, dt: float) -> SensorReading:
        sd = self.data.sensordata
        gyro = sd[self.s_gyro : self.s_gyro + 3].copy()
        accel = sd[self.s_accel : self.s_accel + 3].copy()
        quat = sd[self.s_quat : self.s_quat + 4].copy()
        rng = float(sd[self.s_range])
        pos = self.data.qpos[self.qadr : self.qadr + 3]
        # downward optical-flow sees velocity relative to the surface below (the deck): drone vel - deck vel
        lin = self.data.qvel[self.vadr : self.vadr + 3]
        _, pvel, _ = self.platform_truth()
        rel_vel_xy = np.asarray(lin[:2], float) - np.asarray(pvel[:2], float)
        return self.sensors.update(dt, pos, quat, gyro, accel, rng, raw_rel_vel_xy=rel_vel_xy)

    # ----------------------------------------------------------- termination
    def _termination(self) -> Termination:
        pos = self.data.qpos[self.qadr : self.qadr + 3]
        if abs(pos[0]) > self.config.bounds_xy or abs(pos[1]) > self.config.bounds_xy:
            return "out_of_bounds"
        if pos[2] > self.config.max_alt:
            return "out_of_bounds"

        contacts = self._contact_sets()
        if contacts["hit_structure"]:      # flew into the (now-collidable) superstructure
            return "hit_structure"
        if contacts["body_deck"]:
            return "crash"
        if contacts["drone_floor"]:
            return "off_platform"

        quat = self.data.qpos[self.qadr + 3 : self.qadr + 7]
        on_deck = contacts["foot_deck"] or contacts["leg_deck"]
        if self._tilt_deg(quat) > self.config.tipover_deg and on_deck:
            return "crash"
        if on_deck and not self._first_contact_logged:
            self._first_contact_logged = True
            if self.observe_truth()["vertical_speed"] > self.config.max_impact_vspeed:
                return "crash"

        if self.steps * self.control_dt >= self.config.max_time:
            return "timeout"

        if self._is_stable_touchdown():
            self._settle_counter += 1
            if self._settle_counter >= self._settle_steps_needed:
                return "success"
        else:
            self._settle_counter = 0
        return "running"

    def _is_stable_touchdown(self) -> bool:
        t = self.observe_truth()
        return (
            t["support_feet"] >= self.config.min_support_feet
            and t["horizontal_error"] <= self.config.success_radius
            and t["relative_horizontal_speed"] <= self.config.success_max_relspeed
            and t["vertical_speed"] <= self.config.success_max_vspeed
            and t["tilt_deg"] <= self.config.success_max_tilt_deg
        )

    # ------------------------------------------------------------- internals
    def _platform_targets(self, state) -> dict:
        """Map a :class:`PlatformState` to per-joint servo targets.

        ``plat_z`` tracks the heave *deviation* about the mean deck height: the state's world
        deck-top height minus the local deck-top offset. Roll/pitch/yaw come from the deck quaternion.
        """
        roll, pitch, yaw = self._rpy_of(state.quat)
        return {
            "plat_x": float(state.pos[0]),
            "plat_y": float(state.pos[1]),
            "plat_z": float(state.pos[2] - self.deck_top_z),
            "plat_yaw": yaw,
            "plat_pitch": pitch,
            "plat_roll": roll,
        }

    def _drive_platform(self, state) -> None:
        self._platform_state = state
        targets = self._platform_targets(state)
        for n in self.plat_joints:
            self.data.ctrl[self.plat_act[n]] = targets[n]

    def _drive_gimbal(self) -> None:
        """Hold the gimbal camera at the drone's belly but always world-level (nadir), so the
        downward camera looks straight down regardless of drone tilt."""
        pos = self.data.qpos[self.qadr : self.qadr + 3]
        quat = self.data.qpos[self.qadr + 3 : self.qadr + 7]
        cam_world = pos + self.cam_mount_z * self._body_z(quat)  # camera mount in world
        self.data.mocap_pos[self.gimbal_mocap] = cam_world
        self.data.mocap_quat[self.gimbal_mocap] = [1.0, 0.0, 0.0, 0.0]  # world-level -> nadir

    def _apply_wind(self) -> None:
        c = self.config
        wind = np.asarray(c.wind_mean, dtype=float)
        if c.wind_gust_std > 0.0:
            # Ornstein-Uhlenbeck gust: temporally correlated (unlike white noise), the standard
            # low-order surrogate for turbulence. Updated at the physics timestep.
            dt = self.timestep
            tau = max(c.wind_gust_tau, 1e-3)
            a = np.exp(-dt / tau)
            sigma = c.wind_gust_std * np.sqrt(1.0 - a * a)
            self._gust = a * self._gust + sigma * self.rng.standard_normal(3)
            wind = wind + self._gust
        if self.airwake is not None:
            drone_pos = self.data.qpos[self.qadr : self.qadr + 3]
            heading = self.config.ship.heading
            wind = wind + self.airwake.force(drone_pos, self._deck_pos, heading, self.rng, self.timestep)
        self.data.xfrc_applied[self.drone_bid, :3] = wind

    def _contact_sets(self) -> dict:
        foot_deck = leg_deck = body_deck = drone_floor = hit_structure = False
        drone_gids = self.body_hard_gids | self.gear_gids
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            pair = {g1, g2}
            if self.deck_gid in pair:
                other = g2 if g1 == self.deck_gid else g1
                if other in self.foot_gids:
                    foot_deck = True
                elif other in self.leg_gids:
                    leg_deck = True
                elif other in self.body_hard_gids:
                    body_deck = True
            if self.floor_gid in pair:
                other = g2 if g1 == self.floor_gid else g1
                if self.model.geom_bodyid[other] == self.drone_bid:
                    drone_floor = True
            # any part of the drone touching the static superstructure = a crash into the structure
            if self.obstacle_gids and pair & self.obstacle_gids and (pair & drone_gids):
                hit_structure = True
        return {"foot_deck": foot_deck, "leg_deck": leg_deck, "body_deck": body_deck,
                "drone_floor": drone_floor, "hit_structure": hit_structure}

    def _support_feet(self) -> int:
        touched = set()
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if self.deck_gid in pair:
                other = con.geom2 if con.geom1 == self.deck_gid else con.geom1
                if other in self.foot_gids:
                    touched.add(other)
        return len(touched)

    def _tilt_deg(self, quat) -> float:
        bz = self._body_z(quat)
        return float(np.degrees(np.arccos(np.clip(bz[2], -1.0, 1.0))))

    @staticmethod
    def _body_z(quat) -> np.ndarray:
        w, x, y, z = quat
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    @staticmethod
    def _yaw_of(quat) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @staticmethod
    def _rpy_of(quat) -> tuple[float, float, float]:
        """Roll, pitch, yaw (ZYX Euler, rad) from a (w, x, y, z) quaternion."""
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    def render(self, camera: str = "track", width: int = 960, height: int = 540) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        img = renderer.render()
        renderer.close()
        return img


@dataclass
class WorldStep:
    sensors: SensorReading
    truth: dict
    terminated: bool
    truncated: bool
    info: dict
