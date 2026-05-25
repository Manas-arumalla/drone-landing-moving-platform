from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class QuadrotorMotorPID:
    """Motor-level cascaded baseline for the MuJoCo quadrotor.

    Observation layout comes from ``MuJoCoLandingEnv``. Output is four normalized motor
    commands, ordered front-left, front-right, rear-left, rear-right.
    """

    hover_command: float = 0.49
    kp_xy: float = 0.22
    kd_xy: float = 0.11
    kp_z: float = 0.35
    kd_z: float = 0.18
    kp_att: float = 0.45
    kd_att: float = 0.08
    target_descent_rate: float = -0.28

    def act(self, observation: list[float]) -> list[float]:
        dx, dy, dz = observation[0], observation[1], observation[2]
        qw, qx, qy, qz = observation[3], observation[4], observation[5], observation[6]
        rvx, rvy, vz = observation[7], observation[8], observation[9]
        wx, wy, wz = observation[10], observation[11], observation[12]

        desired_roll = self._clip(self.kp_xy * dy + self.kd_xy * rvy, 0.25)
        desired_pitch = self._clip(-self.kp_xy * dx - self.kd_xy * rvx, 0.25)
        roll, pitch, _ = self._euler_from_quat(qw, qx, qy, qz)

        desired_vz = self.target_descent_rate
        if dz < 1.0:
            desired_vz = -0.14
        if dz < 0.45:
            desired_vz = -0.04

        collective = self.hover_command + self.kp_z * (desired_vz - vz) - 0.015 * dz
        roll_mix = self.kp_att * (desired_roll - roll) - self.kd_att * wx
        pitch_mix = self.kp_att * (desired_pitch - pitch) - self.kd_att * wy
        yaw_mix = -0.03 * wz

        fl = collective + roll_mix - pitch_mix + yaw_mix
        fr = collective - roll_mix - pitch_mix - yaw_mix
        rl = collective + roll_mix + pitch_mix - yaw_mix
        rr = collective - roll_mix + pitch_mix + yaw_mix
        return [self._clip(v, 1.0) for v in (fl, fr, rl, rr)]

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _euler_from_quat(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
