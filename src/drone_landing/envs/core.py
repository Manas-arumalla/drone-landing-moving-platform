from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Literal


PlatformMode = Literal["static", "sinusoidal", "random_walk"]
Termination = Literal["running", "success", "crash", "out_of_bounds", "timeout"]


@dataclass(frozen=True)
class LandingConfig:
    dt: float = 0.05
    max_steps: int = 600
    gravity: float = 9.81
    max_accel_xy: float = 5.0
    max_thrust_accel_z: float = 16.0
    linear_drag: float = 0.08
    platform_radius: float = 0.65
    platform_height: float = 0.0
    platform_mode: PlatformMode = "sinusoidal"
    platform_speed: float = 0.55
    platform_amplitude: float = 1.5
    platform_random_accel: float = 0.6
    initial_altitude: float = 4.0
    initial_xy_spread: float = 1.0
    bounds_xy: float = 8.0
    success_max_vertical_speed: float = 0.55
    success_max_horizontal_speed: float = 0.65
    seed: int | None = None


@dataclass
class DroneState:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class PlatformState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    phase_x: float = 0.0
    phase_y: float = 1.57079632679


@dataclass(frozen=True)
class StepResult:
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float | str | bool]


@dataclass
class LandingEnv:
    """Small, deterministic state-based landing environment.

    Action is ``[ax, ay, az]`` in normalized units from -1 to 1. The environment converts it
    to horizontal acceleration and upward thrust acceleration. Observation is relative drone
    state plus platform velocity:
    ``[dx, dy, dz, vx, vy, vz, platform_vx, platform_vy]``.
    """

    config: LandingConfig = field(default_factory=LandingConfig)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.steps = 0
        self.turn_off = False
        self.drone = DroneState(0.0, 0.0, self.config.initial_altitude)
        self.platform = PlatformState()
        self.reset()

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng.seed(seed)
        spread = self.config.initial_xy_spread
        self.steps = 0
        self.turn_off = False
        self.drone = DroneState(
            x=self.rng.uniform(-spread, spread),
            y=self.rng.uniform(-spread, spread),
            z=self.config.initial_altitude,
            vx=self.rng.uniform(-0.1, 0.1),
            vy=self.rng.uniform(-0.1, 0.1),
            vz=0.0,
        )
        self.platform = PlatformState(
            phase_x=self.rng.uniform(0.0, math.tau),
            phase_y=self.rng.uniform(0.0, math.tau),
        )
        self._update_platform()
        return self.observation()

    def step(self, action: list[float] | tuple[float, float, float]) -> StepResult:
        if len(action) != 3:
            raise ValueError("Action must have exactly three values: [ax, ay, az].")

        self.steps += 1
        self._update_platform()

        ax = self._clip(action[0]) * self.config.max_accel_xy
        ay = self._clip(action[1]) * self.config.max_accel_xy
        thrust_z = (self._clip(action[2]) + 1.0) * 0.5 * self.config.max_thrust_accel_z

        if self.turn_off:
            ax = ay = thrust_z = 0.0

        self._integrate_drone(ax, ay, thrust_z - self.config.gravity)
        termination = self._termination()
        reward = self._reward(termination, action)

        if termination == "success":
            self.turn_off = True
            self.drone.vx = self.drone.vy = self.drone.vz = 0.0
            self.drone.z = self.config.platform_height

        terminated = termination in {"success", "crash", "out_of_bounds"}
        truncated = termination == "timeout"
        return StepResult(
            observation=self.observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "termination": termination,
                "success": termination == "success",
                "horizontal_error": self.horizontal_error(),
                "relative_horizontal_speed": self.relative_horizontal_speed(),
                "vertical_speed": abs(self.drone.vz),
                "turn_off": self.turn_off,
            },
        )

    def observation(self) -> list[float]:
        return [
            self.drone.x - self.platform.x,
            self.drone.y - self.platform.y,
            self.drone.z - self.config.platform_height,
            self.drone.vx,
            self.drone.vy,
            self.drone.vz,
            self.platform.vx,
            self.platform.vy,
        ]

    def horizontal_error(self) -> float:
        dx = self.drone.x - self.platform.x
        dy = self.drone.y - self.platform.y
        return math.hypot(dx, dy)

    def relative_horizontal_speed(self) -> float:
        return math.hypot(self.drone.vx - self.platform.vx, self.drone.vy - self.platform.vy)

    def _integrate_drone(self, ax: float, ay: float, az: float) -> None:
        dt = self.config.dt
        drag = self.config.linear_drag
        self.drone.vx += (ax - drag * self.drone.vx) * dt
        self.drone.vy += (ay - drag * self.drone.vy) * dt
        self.drone.vz += (az - drag * self.drone.vz) * dt
        self.drone.x += self.drone.vx * dt
        self.drone.y += self.drone.vy * dt
        self.drone.z += self.drone.vz * dt

    def _update_platform(self) -> None:
        mode = self.config.platform_mode
        dt = self.config.dt
        previous_x = self.platform.x
        previous_y = self.platform.y

        if mode == "static":
            self.platform.x = 0.0
            self.platform.y = 0.0
        elif mode == "sinusoidal":
            t = self.steps * dt
            amp = self.config.platform_amplitude
            speed = self.config.platform_speed
            self.platform.x = amp * math.sin(speed * t + self.platform.phase_x)
            self.platform.y = 0.6 * amp * math.sin(0.7 * speed * t + self.platform.phase_y)
        elif mode == "random_walk":
            self.platform.vx += self.rng.uniform(-1.0, 1.0) * self.config.platform_random_accel * dt
            self.platform.vy += self.rng.uniform(-1.0, 1.0) * self.config.platform_random_accel * dt
            limit = self.config.platform_speed
            self.platform.vx = self._clip_abs(self.platform.vx, limit)
            self.platform.vy = self._clip_abs(self.platform.vy, limit)
            self.platform.x += self.platform.vx * dt
            self.platform.y += self.platform.vy * dt
        else:
            raise ValueError(f"Unsupported platform mode: {mode}")

        if mode != "random_walk":
            self.platform.vx = (self.platform.x - previous_x) / dt
            self.platform.vy = (self.platform.y - previous_y) / dt

    def _termination(self) -> Termination:
        if abs(self.drone.x) > self.config.bounds_xy or abs(self.drone.y) > self.config.bounds_xy:
            return "out_of_bounds"
        if self.steps >= self.config.max_steps:
            return "timeout"
        if self.drone.z <= self.config.platform_height:
            if (
                self.horizontal_error() <= self.config.platform_radius
                and abs(self.drone.vz) <= self.config.success_max_vertical_speed
                and self.relative_horizontal_speed() <= self.config.success_max_horizontal_speed
            ):
                return "success"
            return "crash"
        return "running"

    def _reward(self, termination: Termination, action: list[float] | tuple[float, float, float]) -> float:
        horizontal = self.horizontal_error()
        vertical = max(0.0, self.drone.z - self.config.platform_height)
        speed = math.sqrt(self.drone.vx**2 + self.drone.vy**2 + self.drone.vz**2)
        effort = sum(abs(self._clip(a)) for a in action)
        reward = 2.0 - 1.6 * horizontal - 0.35 * vertical - 0.08 * speed - 0.02 * effort

        if vertical < 0.8:
            reward += 0.8 * max(0.0, self.config.platform_radius - horizontal)
        if termination == "success":
            reward += 120.0
        elif termination == "crash":
            reward -= 80.0 + 10.0 * horizontal
        elif termination == "out_of_bounds":
            reward -= 100.0
        elif termination == "timeout":
            reward -= 20.0
        return reward

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    @staticmethod
    def _clip_abs(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))
