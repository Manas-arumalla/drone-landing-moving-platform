from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Literal

from drone_landing.envs.core import PlatformMode, StepResult


Termination = Literal["running", "success", "crash", "out_of_bounds", "timeout"]


@dataclass(frozen=True)
class MuJoCoLandingConfig:
    model_path: str = "assets/mujoco/landing_scene.xml"
    control_dt: float = 0.02
    max_steps: int = 900
    platform_mode: PlatformMode = "sinusoidal"
    platform_radius: float = 0.72
    platform_speed: float = 0.45
    platform_amplitude: float = 1.4
    platform_random_accel: float = 0.45
    initial_altitude: float = 3.0
    initial_xy_spread: float = 0.8
    bounds_xy: float = 8.0
    touchdown_height: float = 0.16
    success_radius: float = 0.35
    success_max_vertical_speed: float = 0.55
    success_max_horizontal_speed: float = 0.25
    success_max_tilt_rad: float = 0.22
    success_settle_steps: int = 25
    max_motor_rpm: float = 8500.0
    thrust_coefficient: float = 7.2e-8
    torque_coefficient: float = 2.5e-9
    motor_time_constant: float = 0.06
    arm_length: float = 0.32
    drag_coefficient_area: float = 0.08
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0
    wind_gust_std: float = 0.0
    lock_after_success: bool = False
    seed: int | None = None


@dataclass
class PlatformTrajectory:
    mode: PlatformMode
    speed: float
    amplitude: float
    random_accel: float
    rng: random.Random
    phase_x: float = 0.0
    phase_y: float = 0.0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    def reset(self) -> None:
        self.phase_x = self.rng.uniform(0.0, math.tau)
        self.phase_y = self.rng.uniform(0.0, math.tau)
        self.x = self.y = self.vx = self.vy = 0.0

    def step(self, t: float, dt: float) -> tuple[float, float, float, float]:
        previous_x = self.x
        previous_y = self.y
        if self.mode == "static":
            self.x = 0.0
            self.y = 0.0
        elif self.mode == "sinusoidal":
            self.x = self.amplitude * math.sin(self.speed * t + self.phase_x)
            self.y = 0.6 * self.amplitude * math.sin(0.7 * self.speed * t + self.phase_y)
            self.vx = self.amplitude * self.speed * math.cos(self.speed * t + self.phase_x)
            self.vy = 0.6 * self.amplitude * 0.7 * self.speed * math.cos(
                0.7 * self.speed * t + self.phase_y
            )
            return self.x, self.y, self.vx, self.vy
        elif self.mode == "random_walk":
            self.vx += self.rng.uniform(-1.0, 1.0) * self.random_accel * dt
            self.vy += self.rng.uniform(-1.0, 1.0) * self.random_accel * dt
            self.vx = max(-self.speed, min(self.speed, self.vx))
            self.vy = max(-self.speed, min(self.speed, self.vy))
            self.x += self.vx * dt
            self.y += self.vy * dt
            return self.x, self.y, self.vx, self.vy
        else:
            raise ValueError(f"Unsupported platform mode: {self.mode}")
        self.vx = (self.x - previous_x) / dt
        self.vy = (self.y - previous_y) / dt
        return self.x, self.y, self.vx, self.vy


class MuJoCoLandingEnv:
    """Research-level MuJoCo quadrotor landing environment.

    Action is four normalized motor commands in ``[-1, 1]``. The environment maps those
    commands to first-order motor thrust, applies each rotor force at its motor site,
    drives the platform as a kinematic mocap body, and uses MuJoCo contacts for touchdown.
    """

    def __init__(self, config: MuJoCoLandingConfig | None = None):
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "MuJoCoLandingEnv requires optional dependencies. Install with: "
                "python -m pip install -e .[mujoco]"
            ) from exc

        self.mujoco = mujoco
        self.np = np
        self.config = config or MuJoCoLandingConfig()
        model_path = Path(self.config.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.rng = random.Random(self.config.seed)
        self.trajectory = PlatformTrajectory(
            mode=self.config.platform_mode,
            speed=self.config.platform_speed,
            amplitude=self.config.platform_amplitude,
            random_accel=self.config.platform_random_accel,
            rng=self.rng,
        )
        self.steps = 0
        self.motor_rpm = np.zeros(4, dtype=float)
        self.motor_thrust = np.zeros(4, dtype=float)
        self.turn_off = False
        self.landed = False
        self.success_settle_counter = 0
        self.landed_offset_xy = np.zeros(2, dtype=float)
        self.landed_height = self.config.touchdown_height

        self.quad_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self.platform_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        self.platform_mocap_id = self.model.body_mocapid[self.platform_body_id]
        self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "quad_free")
        self.free_qpos_addr = self.model.jnt_qposadr[self.free_joint_id]
        self.free_qvel_addr = self.model.jnt_dofadr[self.free_joint_id]
        self.platform_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "platform_deck"
        )
        self.foot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "leg_fl",
                "leg_fr",
                "leg_rl",
                "leg_rr",
                "foot_fl",
                "foot_fr",
                "foot_rl",
                "foot_rr",
            )
        }
        self.skid_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("skid_left", "skid_right")
        }
        self.leg_geom_ids = self.foot_geom_ids | self.skid_geom_ids
        self.motor_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("motor_fl", "motor_fr", "motor_rl", "motor_rr")
        ]
        self.rotor_spin_directions = self.np.array([1.0, -1.0, -1.0, 1.0], dtype=float)
        self.reset()

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.turn_off = False
        self.landed = False
        self.success_settle_counter = 0
        self.landed_offset_xy[:] = 0.0
        self.landed_height = self.config.touchdown_height
        self.motor_rpm[:] = 0.0
        self.motor_thrust[:] = 0.0
        self.trajectory.reset()
        self._drive_platform(0.0)

        spread = self.config.initial_xy_spread
        platform_x, platform_y, _, _ = self.platform_state()
        self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3] = [
            platform_x + self.rng.uniform(-spread, spread),
            platform_y + self.rng.uniform(-spread, spread),
            self.config.initial_altitude,
        ]
        self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, action) -> StepResult:
        action = self.np.asarray(action, dtype=float)
        if action.shape != (4,):
            raise ValueError("MuJoCo action must have four motor commands.")

        self.steps += 1
        substeps = max(1, round(self.config.control_dt / self.model.opt.timestep))
        if self.landed:
            self.motor_thrust[:] = 0.0
            for _ in range(substeps):
                self._drive_platform(self.data.time)
                self._lock_drone_to_platform()
                self.mujoco.mj_forward(self.model, self.data)
                self.data.time += self.model.opt.timestep
            return self._step_result("success", reward=0.0)

        throttle = (self.np.clip(action, -1.0, 1.0) + 1.0) * 0.5
        target_rpm = throttle * self.config.max_motor_rpm
        if self.turn_off:
            target_rpm[:] = 0.0

        for _ in range(substeps):
            alpha = self.model.opt.timestep / max(self.config.motor_time_constant, self.model.opt.timestep)
            self.motor_rpm += alpha * (target_rpm - self.motor_rpm)
            self.motor_thrust = self.config.thrust_coefficient * self.motor_rpm**2
            self._drive_platform(self.data.time)
            self._apply_quadrotor_forces()
            self._apply_wind_disturbance()
            self.mujoco.mj_step(self.model, self.data)

        termination = self._termination()
        reward = self._reward(termination, action)
        if termination == "success":
            self.turn_off = True
            self.motor_rpm[:] = 0.0
            self.motor_thrust[:] = 0.0
            if self.config.lock_after_success:
                self._engage_landing_lock()

        return self._step_result(termination, reward)

    def observation(self) -> list[float]:
        qpos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7]
        qvel = self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 6]
        platform = self.platform_state()
        return [
            float(qpos[0] - platform[0]),
            float(qpos[1] - platform[1]),
            float(qpos[2] - self.config.touchdown_height),
            float(qpos[3]),
            float(qpos[4]),
            float(qpos[5]),
            float(qpos[6]),
            float(qvel[0] - platform[2]),
            float(qvel[1] - platform[3]),
            float(qvel[2]),
            float(qvel[3]),
            float(qvel[4]),
            float(qvel[5]),
            float(self.motor_thrust[0]),
            float(self.motor_thrust[1]),
            float(self.motor_thrust[2]),
            float(self.motor_thrust[3]),
        ]

    def platform_state(self) -> tuple[float, float, float, float]:
        return self.trajectory.x, self.trajectory.y, self.trajectory.vx, self.trajectory.vy

    def horizontal_error(self) -> float:
        qpos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3]
        px, py, _, _ = self.platform_state()
        return float(math.hypot(qpos[0] - px, qpos[1] - py))

    def relative_horizontal_speed(self) -> float:
        qvel = self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 3]
        _, _, pvx, pvy = self.platform_state()
        return float(math.hypot(qvel[0] - pvx, qvel[1] - pvy))

    def leg_contact_count(self) -> int:
        return len(self._contacting_landing_geoms())

    def _contacting_landing_geoms(self) -> set[int]:
        touched = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.platform_geom_id and contact.geom2 in self.leg_geom_ids:
                touched.add(contact.geom2)
            elif contact.geom2 == self.platform_geom_id and contact.geom1 in self.leg_geom_ids:
                touched.add(contact.geom1)
        return touched

    def stable_support_contact(self) -> bool:
        touched = self._contacting_landing_geoms()
        skid_contacts = len(touched & self.skid_geom_ids)
        foot_contacts = len(touched & self.foot_geom_ids)
        return skid_contacts == len(self.skid_geom_ids) or foot_contacts >= 4

    def render_rgb(self, width: int = 1280, height: int = 720, camera: str = "tracking"):
        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        pixels = renderer.render()
        renderer.close()
        return pixels

    def render_downward(self, width: int = 160, height: int = 120):
        return self.render_rgb(width=width, height=height, camera="downward")

    def _drive_platform(self, t: float) -> None:
        x, y, _, _ = self.trajectory.step(t, self.model.opt.timestep)
        self.data.mocap_pos[self.platform_mocap_id] = [x, y, 0.06]
        self.data.mocap_quat[self.platform_mocap_id] = [1.0, 0.0, 0.0, 0.0]

    def _engage_landing_lock(self) -> None:
        platform_x, platform_y, _, _ = self.platform_state()
        qpos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7]
        self.landed = True
        self.landed_offset_xy[:] = [qpos[0] - platform_x, qpos[1] - platform_y]
        self.landed_height = max(self.config.touchdown_height, min(float(qpos[2]), 0.28))
        self._lock_drone_to_platform()

    def _lock_drone_to_platform(self) -> None:
        platform_x, platform_y, platform_vx, platform_vy = self.platform_state()
        self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3] = [
            platform_x + self.landed_offset_xy[0],
            platform_y + self.landed_offset_xy[1],
            self.landed_height,
        ]
        self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 6] = [
            platform_vx,
            platform_vy,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qfrc_applied[:] = 0.0

    def _apply_quadrotor_forces(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        quat = self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7]
        body_z_world = self._rotate_vector(quat, self.np.array([0.0, 0.0, 1.0]))

        for rpm, thrust, spin_direction, site_id in zip(
            self.motor_rpm, self.motor_thrust, self.rotor_spin_directions, self.motor_site_ids
        ):
            force = body_z_world * float(thrust)
            yaw_torque = body_z_world * self.config.torque_coefficient * float(rpm**2) * spin_direction
            point = self.data.site_xpos[site_id]
            self.mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                yaw_torque,
                point,
                self.quad_body_id,
                self.data.qfrc_applied,
            )

    def _apply_wind_disturbance(self) -> None:
        wind = self.np.array([self.config.wind_x, self.config.wind_y, self.config.wind_z], dtype=float)
        if self.config.wind_gust_std > 0.0:
            wind += self.np.array(
                [
                    self.rng.gauss(0.0, self.config.wind_gust_std),
                    self.rng.gauss(0.0, self.config.wind_gust_std),
                    self.rng.gauss(0.0, 0.25 * self.config.wind_gust_std),
                ],
                dtype=float,
            )
        if not self.np.any(wind):
            return

        velocity = self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 3]
        relative_air_velocity = velocity - wind
        speed = float(self.np.linalg.norm(relative_air_velocity))
        if speed < 1e-6:
            return

        drag_force = -self.config.drag_coefficient_area * speed * relative_air_velocity
        self.mujoco.mj_applyFT(
            self.model,
            self.data,
            drag_force,
            self.np.zeros(3, dtype=float),
            self.data.xpos[self.quad_body_id],
            self.quad_body_id,
            self.data.qfrc_applied,
        )

    def _termination(self) -> Termination:
        pos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3]
        if abs(pos[0]) > self.config.bounds_xy or abs(pos[1]) > self.config.bounds_xy:
            return "out_of_bounds"
        if self.steps >= self.config.max_steps:
            return "timeout"
        if self.leg_contact_count() > 0 or pos[2] <= self.config.touchdown_height:
            if self._is_stable_touchdown():
                self.success_settle_counter += 1
                if self.success_settle_counter >= self.config.success_settle_steps:
                    return "success"
            else:
                self.success_settle_counter = 0
            if pos[2] < self.config.touchdown_height * 0.65 and self.leg_contact_count() == 0:
                return "crash"
        return "running"

    def _is_stable_touchdown(self) -> bool:
        qpos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7]
        vertical_speed = abs(float(self.data.qvel[self.free_qvel_addr + 2]))
        tilt = self._tilt_angle(qpos[3:7])
        return (
            self.stable_support_contact()
            and self.horizontal_error() <= self.config.success_radius
            and vertical_speed <= self.config.success_max_vertical_speed
            and self.relative_horizontal_speed() <= self.config.success_max_horizontal_speed
            and tilt <= self.config.success_max_tilt_rad
        )

    def _reward(self, termination: Termination, action) -> float:
        qpos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7]
        pos = qpos[:3]
        qvel = self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 6]
        horizontal = self.horizontal_error()
        vertical = max(0.0, float(pos[2]) - self.config.touchdown_height)
        rel_speed = self.relative_horizontal_speed()
        effort = float(self.np.mean(self.np.abs(action)))
        attitude_penalty = 1.0 - abs(float(qpos[3]))
        reward = (
            2.0
            - 2.4 * horizontal
            - 0.45 * vertical
            - 0.25 * rel_speed
            - 0.08 * abs(float(qvel[2]))
            - 0.03 * effort
            - 0.4 * attitude_penalty
        )
        if vertical < 0.8:
            reward += 1.2 * max(0.0, self.config.platform_radius - horizontal)
        if termination == "success":
            reward += 180.0
        elif termination == "crash":
            reward -= 120.0 + 15.0 * horizontal
        elif termination == "out_of_bounds":
            reward -= 150.0
        elif termination == "timeout":
            reward -= 30.0
        return float(reward)

    def _step_result(self, termination: Termination, reward: float) -> StepResult:
        return StepResult(
            observation=self.observation(),
            reward=reward,
            terminated=termination in {"success", "crash", "out_of_bounds"},
            truncated=termination == "timeout",
            info={
                "termination": termination,
                "success": termination == "success",
                "horizontal_error": self.horizontal_error(),
                "relative_horizontal_speed": self.relative_horizontal_speed(),
                "vertical_speed": abs(float(self.data.qvel[self.free_qvel_addr + 2])),
                "leg_contacts": self.leg_contact_count(),
                "stable_support": self.stable_support_contact(),
                "turn_off": self.turn_off,
                "landed": self.landed,
                "success_settle_counter": self.success_settle_counter,
                "tilt": self._tilt_angle(
                    self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7]
                ),
                "motor_rpm_fl": float(self.motor_rpm[0]),
                "motor_rpm_fr": float(self.motor_rpm[1]),
                "motor_rpm_rl": float(self.motor_rpm[2]),
                "motor_rpm_rr": float(self.motor_rpm[3]),
                "wind_x": self.config.wind_x,
                "wind_y": self.config.wind_y,
                "wind_z": self.config.wind_z,
            },
        )

    def _tilt_angle(self, quat) -> float:
        body_z = self._rotate_vector(quat, self.np.array([0.0, 0.0, 1.0]))
        cos_tilt = max(-1.0, min(1.0, float(body_z[2])))
        return float(math.acos(cos_tilt))

    def _rotate_vector(self, quat, vector):
        w, x, y, z = quat
        rotation = self.np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
        return rotation @ vector
