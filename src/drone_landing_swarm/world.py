"""MuJoCo multi-drone world for the swarm: N real Skydio-X2 quadrotors + one servo-driven moving deck.

Generated programmatically (one validated X2 body stamped N times, each with its own freejoint and four
thrust motors) so the swarm runs on **true MuJoCo physics + true contact** — not a kinematic model.
The deck is a dynamic body on 6 servo'd joints (slide x/y/z + hinge yaw/pitch/roll, buoyancy via
gravcomp) driven each step by the validated single-drone motion models (``ShipDeckMotion`` /
``RandomGroundMotion``). Drones are solid and collide with each other and the deck, so a coordination
failure is a real crash (the CBF filter must prevent it).

This module only provides the *physics* (build / reset / step / per-drone state). The coordination
(scheduling + CBF avoidance + holding) and the per-drone geometric controller live in the runner.
"""

from __future__ import annotations

import numpy as np

from drone_landing.sim.mjcf import repo_root

_MESH = repo_root() / "assets" / "mujoco" / "meshes" / "skydio_x2"
_TEX = _MESH / "X2_lowpoly_texture_SpinningProps_1024.png"
_OBJ = _MESH / "X2_lowpoly.obj"

DECK_TOP_Z = 0.30


def _drone_body(i: int, x: float, y: float, z: float) -> str:
    """MJCF for one X2 drone (validated geoms/rotors/gear), suffixed with index ``i``."""
    return f"""
    <body name="x2_{i}" pos="{x:.3f} {y:.3f} {z:.3f}" childclass="x2">
      <freejoint name="root_{i}"/>
      <site name="imu_{i}" pos="0 0 .02"/>
      <geom material="phong3SG" mesh="X2_lowpoly" class="visual" quat="0 0 1 1"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .02"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .06"/>
      <geom class="collision" size=".05 .027 .02" pos="-.07 0 .065"/>
      <geom name="rotor1_{i}" class="rotor" pos="-.14 -.18 .05" mass=".25"/>
      <geom name="rotor2_{i}" class="rotor" pos="-.14 .18 .05" mass=".25"/>
      <geom name="rotor3_{i}" class="rotor" pos=".14 .18 .08" mass=".25"/>
      <geom name="rotor4_{i}" class="rotor" pos=".14 -.18 .08" mass=".25"/>
      <geom size=".16 .04 .02" pos="0 0 0.02" type="ellipsoid" mass=".325" class="visual" material="invisible"/>
      <site name="thrust1_{i}" pos="-.14 -.18 .05"/>
      <site name="thrust2_{i}" pos="-.14 .18 .05"/>
      <site name="thrust3_{i}" pos=".14 .18 .08"/>
      <site name="thrust4_{i}" pos=".14 -.18 .08"/>
      <geom class="leg" name="leg_fl_{i}" fromto=".06 .06 -.01 .10 .12 -.13"/>
      <geom class="leg" name="leg_fr_{i}" fromto=".06 -.06 -.01 .10 -.12 -.13"/>
      <geom class="leg" name="leg_rl_{i}" fromto="-.06 .06 -.01 -.10 .12 -.13"/>
      <geom class="leg" name="leg_rr_{i}" fromto="-.06 -.06 -.01 -.10 -.12 -.13"/>
      <geom class="foot" name="foot_fl_{i}" pos=".10 .12 -.13"/>
      <geom class="foot" name="foot_fr_{i}" pos=".10 -.12 -.13"/>
      <geom class="foot" name="foot_rl_{i}" pos="-.10 .12 -.13"/>
      <geom class="foot" name="foot_rr_{i}" pos="-.10 -.12 -.13"/>
    </body>"""


def _drone_actuators(i: int) -> str:
    return f"""
    <motor class="x2" name="thrust1_{i}" site="thrust1_{i}" gear="0 0 1 0 0  .0201"/>
    <motor class="x2" name="thrust2_{i}" site="thrust2_{i}" gear="0 0 1 0 0 -.0201"/>
    <motor class="x2" name="thrust3_{i}" site="thrust3_{i}" gear="0 0 1 0 0  .0201"/>
    <motor class="x2" name="thrust4_{i}" site="thrust4_{i}" gear="0 0 1 0 0 -.0201"/>"""


def _osv_geoms() -> str:
    """Light orange OSV hull/bow/wheelhouse (visual only) around the swarm deck — the offshore look.
    The existing heavy chassis stays the mass/inertia ballast; these add no physics (contype=0)."""
    return """
      <geom name="osv_hull" type="box" size="2.0 1.45 0.13" pos="0.7 0 0.135" mass="1"
            material="osv_orange" contype="0" conaffinity="0"/>
      <geom name="osv_belt" type="box" size="2.02 1.47 0.03" pos="0.7 0 0.245" mass="0.2"
            material="osv_dark" contype="0" conaffinity="0"/>
      <geom name="osv_bow" type="box" size="0.4 0.95 0.12" pos="2.75 0 0.235" mass="0.3"
            material="osv_orange" contype="0" conaffinity="0"/>
      <geom name="osv_house" type="box" size="0.5 0.8 0.34" pos="2.1 0 0.66" mass="0.5"
            material="osv_white" contype="0" conaffinity="0"/>
      <geom name="osv_windows" type="box" size="0.42 0.82 0.09" pos="2.1 0 0.70" mass="0.05"
            material="osv_dark" contype="0" conaffinity="0"/>"""


def _gimbal_bodies(n_drones: int) -> str:
    """Per-drone nadir-stabilized gimbal cameras: kinematic mocap bodies held world-level at each drone's
    belly (driven each step by the world), so the downward camera looks straight down regardless of drone
    tilt -> the deck stays in view and the vision back-projection is exact (P2.1)."""
    return "".join(f"""
    <body name="gimbal_{i}" mocap="true" pos="0 0 1">
      <camera name="cam_{i}" pos="0 0 0" fovy="120" mode="fixed"/>
    </body>""" for i in range(n_drones))


def build_swarm_xml(n_drones: int, spawn: list[tuple[float, float, float]],
                    offshore: bool = False) -> str:
    """Return the MJCF for an ``n_drones`` swarm world (absolute asset paths -> from_xml_string-safe).
    ``offshore`` adds the orange OSV hull look (visual only; deck/pad/physics unchanged)."""
    bodies = "".join(_drone_body(i, *spawn[i]) for i in range(n_drones))
    gimbals = _gimbal_bodies(n_drones)
    acts = "".join(_drone_actuators(i) for i in range(n_drones))
    osv = _osv_geoms() if offshore else ""
    return f"""<mujoco model="x2_swarm">
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
    <material name="osv_orange" rgba="0.93 0.42 0.06 1" reflectance="0.1"/>
    <material name="osv_white" rgba="0.88 0.89 0.90 1"/>
    <material name="osv_dark" rgba="0.10 0.12 0.16 1"/>
    <mesh class="x2" name="X2_lowpoly" file="{_OBJ.as_posix()}"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="sea"/>
    <body name="platform" pos="0 0 0" gravcomp="1">
      <joint name="plat_x" type="slide" axis="1 0 0" damping="2000"/>
      <joint name="plat_y" type="slide" axis="0 1 0" damping="2000"/>
      <joint name="plat_z" type="slide" axis="0 0 1" damping="2000"/>
      <joint name="plat_yaw" type="hinge" axis="0 0 1" damping="400"/>
      <joint name="plat_pitch" type="hinge" axis="0 1 0" damping="400"/>
      <joint name="plat_roll" type="hinge" axis="1 0 0" damping="400"/>
      <geom name="chassis" type="box" size="1.3 1.3 0.125" pos="0 0 0.125" mass="200" material="hull" contype="0" conaffinity="0"/>
      <geom name="deck" type="box" size="1.3 1.3 0.025" pos="0 0 0.275" mass="20" material="deck"
            contype="1" conaffinity="1" condim="4" friction="3.0 0.1 0.002" solref="0.02 1.5" solimp="0.96 0.99 0.001"/>
      <geom name="pad" type="box" size="0.5 0.5 0.004" pos="0 0 0.302" material="pad"
            contype="0" conaffinity="0"/>{osv}
    </body>{bodies}{gimbals}
  </worldbody>
  <actuator>
    <position name="plat_x_srv" joint="plat_x" kp="40000"/>
    <position name="plat_y_srv" joint="plat_y" kp="40000"/>
    <position name="plat_z_srv" joint="plat_z" kp="40000"/>
    <position name="plat_yaw_srv" joint="plat_yaw" kp="8000"/>
    <position name="plat_pitch_srv" joint="plat_pitch" kp="8000"/>
    <position name="plat_roll_srv" joint="plat_roll" kp="8000"/>{acts}
  </actuator>
</mujoco>"""


class SwarmMujocoWorld:
    """N real X2 drones + a servo-driven moving deck (true physics). Spawns drones on a ring."""

    PLAT_JOINTS = ("plat_x", "plat_y", "plat_z", "plat_yaw", "plat_pitch", "plat_roll")

    def __init__(self, n_drones: int, deck_size: float = 1.3, spawn_radius: float = 3.0,
                 spawn_alt: float = 2.0, control_hz: float = 100.0, offshore: bool = False):
        import mujoco

        self.n = n_drones
        ring = []
        for i in range(n_drones):
            ang = 2 * np.pi * i / max(1, n_drones)
            ring.append((spawn_radius * np.cos(ang), spawn_radius * np.sin(ang), spawn_alt))
        self.model = mujoco.MjModel.from_xml_string(build_swarm_xml(n_drones, ring, offshore=offshore))
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
        self.thrust_act = [[nid(M.mjOBJ_ACTUATOR, f"thrust{k}_{i}") for k in (1, 2, 3, 4)]
                           for i in range(self.n)]
        self.plat_qadr = {n: int(m.jnt_qposadr[nid(M.mjOBJ_JOINT, n)]) for n in self.PLAT_JOINTS}
        self.plat_vadr = {n: int(m.jnt_dofadr[nid(M.mjOBJ_JOINT, n)]) for n in self.PLAT_JOINTS}
        self.plat_act = {n: nid(M.mjOBJ_ACTUATOR, f"{n}_srv") for n in self.PLAT_JOINTS}
        self.deck_gid = nid(M.mjOBJ_GEOM, "deck")
        self.foot_gids = {i: {nid(M.mjOBJ_GEOM, f"foot_{p}_{i}") for p in ("fl", "fr", "rl", "rr")}
                          for i in range(self.n)}
        self.drone_bid = [nid(M.mjOBJ_BODY, f"x2_{i}") for i in range(self.n)]
        self.gimbal_mocap = [int(m.body_mocapid[nid(M.mjOBJ_BODY, f"gimbal_{i}")]) for i in range(self.n)]
        self.cam_mount_z = -0.06   # camera at the drone belly
        self.mass = float(m.body_mass[self.drone_bid[0]] + sum(
            m.body_mass[g] for g in range(m.nbody) if m.body_rootid[g] == self.drone_bid[0]
            and g != self.drone_bid[0]))
        self.inertia = m.body_inertia[self.drone_bid[0]].copy()

    # ------------------------------------------------------------------ deck
    @staticmethod
    def _rpy(quat):
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    def _drive_deck(self, ps) -> None:
        roll, pitch, yaw = self._rpy(ps.quat)
        tgt = {"plat_x": ps.pos[0], "plat_y": ps.pos[1], "plat_z": ps.pos[2] - self.deck_top_z,
               "plat_yaw": yaw, "plat_pitch": pitch, "plat_roll": roll}
        for n in self.PLAT_JOINTS:
            self.data.ctrl[self.plat_act[n]] = tgt[n]

    def drive_gimbals(self) -> None:
        """Hold each drone's gimbal camera world-level (nadir) at its belly (P2.1 stabilization)."""
        for i in range(self.n):
            p = self.data.qpos[self.qadr[i]:self.qadr[i] + 3]
            self.data.mocap_pos[self.gimbal_mocap[i]] = [p[0], p[1], p[2] + self.cam_mount_z]
            self.data.mocap_quat[self.gimbal_mocap[i]] = [1.0, 0.0, 0.0, 0.0]

    def deck_state(self):
        px = self.data.qpos[self.plat_qadr["plat_x"]]
        py = self.data.qpos[self.plat_qadr["plat_y"]]
        pz = self.deck_top_z + self.data.qpos[self.plat_qadr["plat_z"]]
        v = [self.data.qvel[self.plat_vadr[n]] for n in ("plat_x", "plat_y", "plat_z")]
        return np.array([px, py, pz]), np.array(v)

    # --------------------------------------------------------------- drones
    def reset(self, deck0, rng=None):
        import mujoco
        mujoco.mj_resetData(self.model, self.data)
        roll, pitch, yaw = self._rpy(deck0.quat)
        tgt = {"plat_x": deck0.pos[0], "plat_y": deck0.pos[1], "plat_z": deck0.pos[2] - self.deck_top_z,
               "plat_yaw": yaw, "plat_pitch": pitch, "plat_roll": roll}
        for n in self.PLAT_JOINTS:
            self.data.qpos[self.plat_qadr[n]] = tgt[n]
        self._drive_deck(deck0)
        for i in range(self.n):
            sx, sy, sz = self._spawn[i]
            jitter = rng.uniform(-0.3, 0.3, 2) if rng is not None else np.zeros(2)
            self.data.qpos[self.qadr[i]:self.qadr[i] + 3] = [deck0.pos[0] + sx + jitter[0],
                                                             deck0.pos[1] + sy + jitter[1], sz]
            self.data.qpos[self.qadr[i] + 3:self.qadr[i] + 7] = [1, 0, 0, 0]
        self.drive_gimbals()
        mujoco.mj_forward(self.model, self.data)

    def step(self, thrusts: dict[int, np.ndarray], deck_state) -> None:
        import mujoco
        tmax = float(self.model.actuator_ctrlrange[self.thrust_act[0][0], 1])
        for _ in range(self.n_substeps):
            self._drive_deck(deck_state)
            for i in range(self.n):
                u = np.clip(thrusts.get(i, np.zeros(4)), 0.0, tmax)
                for k, aid in enumerate(self.thrust_act[i]):
                    self.data.ctrl[aid] = u[k]
            mujoco.mj_step(self.model, self.data)
        self.drive_gimbals()   # keep the nadir gimbal cameras at the drones' final poses

    def drone_pos(self, i):
        return self.data.qpos[self.qadr[i]:self.qadr[i] + 3].copy()

    def drone_quat(self, i):
        return self.data.qpos[self.qadr[i] + 3:self.qadr[i] + 7].copy()

    def drone_vel(self, i):
        return self.data.qvel[self.vadr[i]:self.vadr[i] + 3].copy()

    def drone_gyro(self, i):
        return self.data.qvel[self.vadr[i] + 3:self.vadr[i] + 6].copy()

    def support_feet(self, i) -> int:
        touched = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            pair = {con.geom1, con.geom2}
            if self.deck_gid in pair:
                other = con.geom2 if con.geom1 == self.deck_gid else con.geom1
                if other in self.foot_gids[i]:
                    touched.add(other)
        return len(touched)
