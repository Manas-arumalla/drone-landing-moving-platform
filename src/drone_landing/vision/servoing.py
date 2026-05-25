from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MarkerDetection:
    visible: bool
    error_x: float = 0.0
    error_y: float = 0.0
    area_fraction: float = 0.0
    confidence: float = 0.0


def detect_landing_marker(image) -> MarkerDetection:
    """Detect the landing marker in a MuJoCo downward RGB image.

    The detector is intentionally classical and dependency-light: it thresholds the yellow
    landing cross first, then falls back to the green platform deck. Returned errors are
    normalized image-plane errors where 0,0 is the image center.
    """

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Visual servoing requires numpy. Install with: python -m pip install -e .[mujoco]") from exc

    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape [height, width, channels].")

    rgb = pixels[:, :, :3].astype(np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    yellow_mask = (red > 145) & (green > 110) & (blue < 95) & ((red - blue) > 80)
    green_mask = (green > 95) & (red < 90) & (blue < 100) & ((green - red) > 35)
    mask = yellow_mask if int(yellow_mask.sum()) > 12 else green_mask

    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    if area < 12:
        return MarkerDetection(visible=False)

    height, width = pixels.shape[:2]
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    error_x = (center_x - (width - 1) * 0.5) / ((width - 1) * 0.5)
    error_y = (center_y - (height - 1) * 0.5) / ((height - 1) * 0.5)
    area_fraction = area / float(width * height)
    confidence = min(1.0, area_fraction * 80.0)
    return MarkerDetection(
        visible=True,
        error_x=float(error_x),
        error_y=float(error_y),
        area_fraction=float(area_fraction),
        confidence=float(confidence),
    )


@dataclass
class VisualServoController:
    """Hybrid IBVS controller for the MuJoCo quadrotor.

    Vision supplies lateral target error from the downward camera. The state vector supplies
    attitude, vertical speed, and body rates. The output is four independent normalized motor
    commands in front-left, front-right, rear-left, rear-right order.
    """

    hover_command: float = 0.49
    image_gain: float = 0.16
    velocity_damping: float = 0.06
    kp_z: float = 0.35
    kp_att: float = 0.45
    kd_att: float = 0.08
    target_descent_rate: float = -0.28
    search_roll_limit: float = 0.08

    def __post_init__(self) -> None:
        self.last_detection = MarkerDetection(visible=False)

    def act(self, observation: list[float], detection: MarkerDetection) -> list[float]:
        dz = observation[2]
        qw, qx, qy, qz = observation[3], observation[4], observation[5], observation[6]
        rvx, rvy, vz = observation[7], observation[8], observation[9]
        wx, wy, wz = observation[10], observation[11], observation[12]
        roll, pitch, _ = self._euler_from_quat(qw, qx, qy, qz)

        if detection.visible:
            self.last_detection = detection
            desired_roll = self._clip(self.image_gain * detection.error_y + self.velocity_damping * rvy, 0.25)
            desired_pitch = self._clip(self.image_gain * detection.error_x - self.velocity_damping * rvx, 0.25)
        elif self.last_detection.visible:
            desired_roll = self._clip(0.08 * self.last_detection.error_y, 0.16)
            desired_pitch = self._clip(0.08 * self.last_detection.error_x, 0.16)
        else:
            desired_roll = self.search_roll_limit * math.sin(0.05 * len(str(observation)))
            desired_pitch = 0.0

        desired_vz = self.target_descent_rate
        if not detection.visible:
            desired_vz = -0.05
        elif dz < 1.0:
            desired_vz = -0.16
        if dz < 0.45:
            desired_vz = -0.05

        visual_scale = 1.0
        if detection.visible:
            visual_scale = 1.0 - min(0.35, 3.5 * detection.area_fraction)
        collective = self.hover_command + self.kp_z * (desired_vz - vz) - 0.01 * dz * visual_scale
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
