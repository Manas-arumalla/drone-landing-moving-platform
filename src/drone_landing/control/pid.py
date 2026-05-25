from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CascadedPIDController:
    """Simple baseline: track platform position while descending at a controlled rate."""

    kp_xy: float = 0.8
    kd_xy: float = 0.35
    kp_z: float = 0.55
    kd_z: float = 0.7
    target_descent_rate: float = -0.45
    hover_action_z: float = 0.22625

    def act(self, observation: list[float]) -> list[float]:
        dx, dy, dz, vx, vy, vz, platform_vx, platform_vy = observation
        target_vx = platform_vx - self.kp_xy * dx
        target_vy = platform_vy - self.kp_xy * dy

        ax_cmd = self.kd_xy * (target_vx - vx)
        ay_cmd = self.kd_xy * (target_vy - vy)

        desired_vz = self.target_descent_rate
        if dz < 1.2:
            desired_vz = -0.18
        if dz < 0.45:
            desired_vz = -0.08
        throttle_delta = self.kp_z * (desired_vz - vz) - 0.03 * dz
        az_cmd = self.hover_action_z + self.kd_z * throttle_delta

        return [self._clip(ax_cmd), self._clip(ay_cmd), self._clip(az_cmd)]

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, value))
