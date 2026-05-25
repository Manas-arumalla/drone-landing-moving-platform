"""MuJoCo world for multi-deck recovery: N real X2 drones + K servo-driven moving decks (true physics).

The kinematic :class:`~drone_landing_swarm.multi_deck.MultiDeckCoordinator` (A5) flies point-mass drones
onto K *modeled* decks. This world removes that shortcut: it stamps **K independent 6-DOF servo decks**
(each the validated single-deck platform, on its own ring base offset) plus N real X2 quadrotors into one
MuJoCo model, so multi-deck recovery runs on **true physics + true contact** — a drone is "landed" only
when its gear is actually planted on its assigned deck. Reuses the validated drone body / deck servo
pattern from :mod:`drone_landing_swarm.world`; the only new structure is replicating the deck K times and
addressing each deck's joints/geoms by index.
"""

from __future__ import annotations

import numpy as np

from drone_landing_swarm.world import (
    DECK_TOP_Z,
    _OBJ,
    _TEX,
    _drone_actuators,
    _drone_body,
    _gimbal_bodies,
)

_PLAT_DOF = ("x", "y", "z", "yaw", "pitch", "roll")


def _deck_body(k: int, base: tuple[float, float]) -> str:
    """MJCF for deck k: a 6-servo moving platform (chassis + deck + bright pad), suffixed with index k."""
    bx, by = base
    return f"""
    <body name="platform_{k}" pos="{bx:.3f} {by:.3f} 0" gravcomp="1">
      <joint name="plat_x_{k}" type="slide" axis="1 0 0" damping="2000"/>
      <joint name="plat_y_{k}" type="slide" axis="0 1 0" damping="2000"/>
      <joint name="plat_z_{k}" type="slide" axis="0 0 1" damping="2000"/>
      <joint name="plat_yaw_{k}" type="hinge" axis="0 0 1" damping="400"/>
      <joint name="plat_pitch_{k}" type="hinge" axis="0 1 0" damping="400"/>
      <joint name="plat_roll_{k}" type="hinge" axis="1 0 0" damping="400"/>
      <geom name="chassis_{k}" type="box" size="1.0 1.0 0.125" pos="0 0 0.125" mass="180"
            material="hull" contype="0" conaffinity="0"/>
      <geom name="deck_{k}" type="box" size="1.0 1.0 0.025" pos="0 0 0.275" mass="20" material="deck"
            contype="1" conaffinity="1" condim="4" friction="3.0 0.1 0.002"
            solref="0.02 1.5" solimp="0.96 0.99 0.001"/>
      <geom name="pad_{k}" type="box" size="0.5 0.5 0.004" pos="0 0 0.302" material="pad"
            contype="0" conaffinity="0"/>
    </body>"""


def _deck_actuators(k: int) -> str:
    return "".join(f"""
    <position name="plat_{d}_{k}_srv" joint="plat_{d}_{k}" kp="{kp}"/>"""
                   for d, kp in zip(_PLAT_DOF, (40000, 40000, 40000, 8000, 8000, 8000)))


def build_multideck_xml(n_drones: int, spawn: list[tuple[float, float, float]],
                        deck_bases: list[tuple[float, float]]) -> str:
    """MJCF for N drones + K decks (deck k placed at ``deck_bases[k]``)."""
    bodies = "".join(_drone_body(i, *spawn[i]) for i in range(n_drones))
    gimbals = _gimbal_bodies(n_drones)
    drone_acts = "".join(_drone_actuators(i) for i in range(n_drones))
    decks = "".join(_deck_body(k, base) for k, base in enumerate(deck_bases))
    deck_acts = "".join(_deck_actuators(k) for k in range(len(deck_bases)))
    return f"""<mujoco model="x2_multideck">
  <compiler autolimits="true"/>
  <option timestep="0.002" density="1.225" integrator="implicitfast"/>
  <default>
    <default class="x2">
      <geom mass="0"/>
      <motor ctrlrange="0 13"/>
      <mesh scale="0.01 0.01 0.01"/>
      <default class="visual"><geom group="2" type="mesh" contype="0" conaffinity="0"/></default>
      <default class="collision"><geom group="3" type="box"/>
        <default class="rotor"><geom type="ellipsoid" size=".13 .13 .01"/></default>
      </default>
      <site group="5"/>
    </default>
    <default class="leg"><geom type="capsule" size="0.006" mass="0.008" rgba="0.05 0.05 0.05 1"
        contype="1" conaffinity="1" condim="4" friction="1.2 0.05 0.001" solref="0.02 1.5" solimp="0.95 0.99 0.001"/></default>
    <default class="foot"><geom type="sphere" size="0.013" mass="0.004" rgba="0.05 0.05 0.05 1"
        contype="1" conaffinity="1" condim="4" friction="3.0 0.15 0.003" solref="0.025 1.8" solimp="0.96 0.99 0.001"/></default>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.6 0.78" rgb2="0.05 0.1 0.2" width="512" height="3072"/>
    <texture type="2d" name="sea" builtin="checker" mark="none" rgb1="0.08 0.18 0.30" rgb2="0.05 0.12 0.22" width="300" height="300"/>
    <material name="sea" texture="sea" texuniform="true" texrepeat="8 8" reflectance="0.3"/>
    <texture type="2d" name="x2tex" file="{_TEX.as_posix()}"/>
    <material name="phong3SG" texture="x2tex"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <material name="deck" rgba="0.30 0.32 0.36 1"/>
    <material name="pad" rgba="1 1 1 1"/>
    <material name="hull" rgba="0.20 0.22 0.26 1"/>
    <mesh class="x2" name="X2_lowpoly" file="{_OBJ.as_posix()}"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="sea"/>{decks}{bodies}{gimbals}
  </worldbody>
  <actuator>{deck_acts}{drone_acts}
  </actuator>
</mujoco>"""


class MultiDeckMujocoWorld:
    """N real X2 drones + K servo-driven moving decks (true physics + contact)."""

    def __init__(self, n_drones: int, deck_bases: list[tuple[float, float]],
                 spawn_radius: float = 3.0, spawn_alt: float = 2.0, control_hz: float = 100.0):
        import mujoco

        self.n = n_drones
        self.k = len(deck_bases)
        self.deck_bases = [np.array([b[0], b[1], 0.0]) for b in deck_bases]
        centroid = np.mean([np.array(b) for b in deck_bases], axis=0)
        ring = []
        for i in range(n_drones):
            ang = 2 * np.pi * i / max(1, n_drones)
            ring.append((centroid[0] + spawn_radius * np.cos(ang),
                         centroid[1] + spawn_radius * np.sin(ang), spawn_alt))
        self.model = mujoco.MjModel.from_xml_string(build_multideck_xml(n_drones, ring, deck_bases))
        self.data = mujoco.MjData(self.model)
        self.timestep = float(self.model.opt.timestep)
        self.control_dt = 1.0 / control_hz
        self.n_substeps = max(1, round(self.control_dt / self.timestep))
        self._spawn = ring
        self.deck_top_z = DECK_TOP_Z
        self._cache(mujoco)

    def _cache(self, mujoco) -> None:
        M = mujoco.mjtObj
        nid = lambda t, n: mujoco.mj_name2id(self.model, t, n)  # noqa: E731
        m = self.model
        self.qadr = [int(m.jnt_qposadr[nid(M.mjOBJ_JOINT, f"root_{i}")]) for i in range(self.n)]
        self.vadr = [int(m.jnt_dofadr[nid(M.mjOBJ_JOINT, f"root_{i}")]) for i in range(self.n)]
        self.thrust_act = [[nid(M.mjOBJ_ACTUATOR, f"thrust{c}_{i}") for c in (1, 2, 3, 4)]
                           for i in range(self.n)]
        # per-deck joint/actuator/geom addressing
        self.plat_qadr = [{d: int(m.jnt_qposadr[nid(M.mjOBJ_JOINT, f"plat_{d}_{k}")]) for d in _PLAT_DOF}
                          for k in range(self.k)]
        self.plat_vadr = [{d: int(m.jnt_dofadr[nid(M.mjOBJ_JOINT, f"plat_{d}_{k}")]) for d in _PLAT_DOF}
                          for k in range(self.k)]
        self.plat_act = [{d: nid(M.mjOBJ_ACTUATOR, f"plat_{d}_{k}_srv") for d in _PLAT_DOF}
                         for k in range(self.k)]
        self.deck_gid = [nid(M.mjOBJ_GEOM, f"deck_{k}") for k in range(self.k)]
        self.foot_gids = {i: {nid(M.mjOBJ_GEOM, f"foot_{p}_{i}") for p in ("fl", "fr", "rl", "rr")}
                          for i in range(self.n)}
        self.drone_bid = [nid(M.mjOBJ_BODY, f"x2_{i}") for i in range(self.n)]
        self.gimbal_mocap = [int(m.body_mocapid[nid(M.mjOBJ_BODY, f"gimbal_{i}")]) for i in range(self.n)]
        self.cam_mount_z = -0.06
        self.mass = float(m.body_mass[self.drone_bid[0]] + sum(
            m.body_mass[g] for g in range(m.nbody) if m.body_rootid[g] == self.drone_bid[0]
            and g != self.drone_bid[0]))
        self.inertia = m.body_inertia[self.drone_bid[0]].copy()

    @staticmethod
    def _rpy(quat):
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    def _drive_deck(self, k: int, ps) -> None:
        """Servo deck k to its motion state. Joints are body-local and the body already sits at its ring
        base, so the slide targets are the motion model's displacement ``ps.pos`` directly (world =
        base + joint)."""
        roll, pitch, yaw = self._rpy(ps.quat)
        tgt = {"x": ps.pos[0], "y": ps.pos[1], "z": ps.pos[2] - self.deck_top_z,
               "yaw": yaw, "pitch": pitch, "roll": roll}
        for d in _PLAT_DOF:
            self.data.ctrl[self.plat_act[k][d]] = tgt[d]

    def drive_gimbals(self) -> None:
        for i in range(self.n):
            p = self.data.qpos[self.qadr[i]:self.qadr[i] + 3]
            self.data.mocap_pos[self.gimbal_mocap[i]] = [p[0], p[1], p[2] + self.cam_mount_z]
            self.data.mocap_quat[self.gimbal_mocap[i]] = [1.0, 0.0, 0.0, 0.0]

    def deck_state(self, k: int):
        """World-frame (pos, lin_vel) of deck k's pad surface centre."""
        q, v = self.plat_qadr[k], self.plat_vadr[k]
        base = self.deck_bases[k]
        px = base[0] + self.data.qpos[q["x"]]
        py = base[1] + self.data.qpos[q["y"]]
        pz = self.deck_top_z + self.data.qpos[q["z"]]
        vel = np.array([self.data.qvel[v["x"]], self.data.qvel[v["y"]], self.data.qvel[v["z"]]])
        return np.array([px, py, pz]), vel

    def reset(self, deck0_list, rng=None):
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        for k in range(self.k):
            d0 = deck0_list[k]
            roll, pitch, yaw = self._rpy(d0.quat)
            q = self.plat_qadr[k]
            self.data.qpos[q["x"]] = d0.pos[0]
            self.data.qpos[q["y"]] = d0.pos[1]
            self.data.qpos[q["z"]] = d0.pos[2] - self.deck_top_z
            self.data.qpos[q["yaw"]] = yaw
            self.data.qpos[q["pitch"]] = pitch
            self.data.qpos[q["roll"]] = roll
            self._drive_deck(k, d0)
        for i in range(self.n):
            sx, sy, sz = self._spawn[i]
            jitter = rng.uniform(-0.3, 0.3, 2) if rng is not None else np.zeros(2)
            self.data.qpos[self.qadr[i]:self.qadr[i] + 3] = [sx + jitter[0], sy + jitter[1], sz]
            self.data.qpos[self.qadr[i] + 3:self.qadr[i] + 7] = [1, 0, 0, 0]
        self.drive_gimbals()
        mujoco.mj_forward(self.model, self.data)

    def step(self, thrusts: dict[int, np.ndarray], deck_states: list) -> None:
        import mujoco
        tmax = float(self.model.actuator_ctrlrange[self.thrust_act[0][0], 1])
        for _ in range(self.n_substeps):
            for k in range(self.k):
                self._drive_deck(k, deck_states[k])
            for i in range(self.n):
                u = np.clip(thrusts.get(i, np.zeros(4)), 0.0, tmax)
                for c, aid in enumerate(self.thrust_act[i]):
                    self.data.ctrl[aid] = u[c]
            mujoco.mj_step(self.model, self.data)
        self.drive_gimbals()

    def drone_pos(self, i):
        return self.data.qpos[self.qadr[i]:self.qadr[i] + 3].copy()

    def drone_quat(self, i):
        return self.data.qpos[self.qadr[i] + 3:self.qadr[i] + 7].copy()

    def drone_vel(self, i):
        return self.data.qvel[self.vadr[i]:self.vadr[i] + 3].copy()

    def drone_gyro(self, i):
        return self.data.qvel[self.vadr[i] + 3:self.vadr[i] + 6].copy()

    def support_feet(self, i, k: int) -> int:
        """Number of drone i's feet planted on deck k (for touchdown on the assigned deck)."""
        touched = set()
        deck_g = self.deck_gid[k]
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            pair = {con.geom1, con.geom2}
            if deck_g in pair:
                other = con.geom2 if con.geom1 == deck_g else con.geom1
                if other in self.foot_gids[i]:
                    touched.add(other)
        return len(touched)
