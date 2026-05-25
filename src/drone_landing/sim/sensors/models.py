"""Onboard sensor models for the quadrotor.

Each model takes ground-truth physical quantities (read from MuJoCo by the environment) and returns
a realistically corrupted measurement: white noise, slowly drifting biases (random walk), finite
range/rate, dropout, and quantization. Default magnitudes target consumer/MEMS-grade hardware and
are documented in docs/REALISM_CHARTER.md.

Separation of concerns: this module never imports mujoco. The environment reads raw sensor values
and ground truth, then calls :meth:`SensorSuite.update`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ImuConfig:
    gyro_noise_std: float = 0.02        # rad/s   white noise
    gyro_bias_walk: float = 2.0e-4      # rad/s/sqrt(s)
    accel_noise_std: float = 0.10       # m/s^2   white noise
    accel_bias_walk: float = 1.0e-3     # m/s^2/sqrt(s)
    attitude_noise_deg: float = 0.25    # deg     AHRS attitude output noise (good roll/pitch AHRS)


@dataclass(frozen=True)
class RangefinderConfig:
    noise_std: float = 0.02     # m
    min_range: float = 0.04     # m
    max_range: float = 12.0     # m   (LightWare-class downward lidar)
    dropout_p: float = 0.005    # per-sample probability of a missed return


@dataclass(frozen=True)
class BaroConfig:
    noise_std: float = 0.20     # m
    drift_walk: float = 0.02    # m/sqrt(s)   slow pressure drift


@dataclass(frozen=True)
class GpsConfig:
    available: bool = True
    xy_noise_std: float = 1.0   # m   consumer GNSS horizontal
    z_noise_std: float = 2.0    # m   vertical is worse
    rate_hz: float = 5.0        # fix rate
    dropout_p: float = 0.02     # per-fix dropout (multipath / structure occlusion)


@dataclass(frozen=True)
class FlowConfig:
    """Downward optical-flow + laser unit (PMW3901-class): measures the drone's HORIZONTAL velocity
    relative to the surface below from image-texture flow. Crucially it does NOT need the ArUco code to be
    decodable — it works in the <0.3 m close-range blind-zone where the fiducial is too zoomed to detect,
    giving the terminal-descent velocity feedback the camera loses."""
    noise_std: float = 0.05       # m/s  flow velocity noise near the surface
    noise_growth: float = 0.05    # m/s per m of altitude (texture flow degrades with height)
    min_range: float = 0.03       # m
    max_range: float = 2.5        # m   flow resolvable below this altitude


@dataclass(frozen=True)
class SensorSuiteConfig:
    imu: ImuConfig = field(default_factory=ImuConfig)
    rangefinder: RangefinderConfig = field(default_factory=RangefinderConfig)
    baro: BaroConfig = field(default_factory=BaroConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class SensorReading:
    """One synchronized batch of onboard measurements. Fields that did not update this step
    carry the most recent valid value; ``*_valid`` flags mark fresh/usable data."""

    t: float
    gyro: np.ndarray                 # (3,) body angular rate [rad/s]
    accel: np.ndarray                # (3,) body specific force [m/s^2]
    attitude_quat: np.ndarray        # (4,) AHRS attitude estimate (w,x,y,z)
    range: float                     # downward distance to nearest surface [m]
    range_valid: bool
    baro_alt: float                  # barometric altitude [m]
    gps_pos: np.ndarray              # (3,) GNSS position fix [m]
    gps_valid: bool
    flow_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (2,) downward optical-flow
    flow_valid: bool = False         #      horizontal velocity rel. to the surface [m/s] (low altitude)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _small_angle_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


class SensorSuite:
    """Aggregates all onboard sensors. Deterministic given the rng passed to :meth:`reset`."""

    def __init__(self, config: SensorSuiteConfig | None = None):
        self.config = config or SensorSuiteConfig()
        self._rng = np.random.default_rng()
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._baro_drift = 0.0
        self._t = 0.0
        self._last_gps = np.zeros(3)
        self._gps_accum = 0.0
        self._last_range = self.config.rangefinder.max_range

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        # Power-on biases drawn once, then random-walk during the episode.
        self._gyro_bias = rng.normal(0.0, 5.0 * self.config.imu.gyro_bias_walk, size=3)
        self._accel_bias = rng.normal(0.0, 10.0 * self.config.imu.accel_bias_walk, size=3)
        self._baro_drift = rng.normal(0.0, self.config.baro.noise_std, size=1)[0]
        self._t = 0.0
        self._last_gps = np.zeros(3)
        self._gps_accum = 0.0
        self._last_range = self.config.rangefinder.max_range

    def update(
        self,
        dt: float,
        true_pos: np.ndarray,
        true_quat: np.ndarray,
        raw_gyro: np.ndarray,
        raw_accel: np.ndarray,
        raw_range: float,
        raw_rel_vel_xy: np.ndarray | None = None,
    ) -> SensorReading:
        c = self.config
        self._t += dt
        rng = self._rng

        # --- IMU: bias random walk + white noise ---
        self._gyro_bias += rng.standard_normal(3) * c.imu.gyro_bias_walk * np.sqrt(dt)
        self._accel_bias += rng.standard_normal(3) * c.imu.accel_bias_walk * np.sqrt(dt)
        gyro = raw_gyro + self._gyro_bias + rng.normal(0.0, c.imu.gyro_noise_std, size=3)
        accel = raw_accel + self._accel_bias + rng.normal(0.0, c.imu.accel_noise_std, size=3)

        # --- AHRS attitude: small random rotation error ---
        ang = np.deg2rad(c.imu.attitude_noise_deg)
        att = _quat_mul(true_quat, _small_angle_quat(rng.normal(0.0, ang, size=3)))
        att = att / np.linalg.norm(att)

        # --- rangefinder: noise, range gate, dropout ---
        range_valid = True
        if raw_range <= 0.0 or raw_range > c.rangefinder.max_range or rng.random() < c.rangefinder.dropout_p:
            range_valid = False
            rng_meas = self._last_range
        else:
            rng_meas = float(np.clip(
                raw_range + rng.normal(0.0, c.rangefinder.noise_std),
                c.rangefinder.min_range, c.rangefinder.max_range,
            ))
            self._last_range = rng_meas

        # --- barometer: slow drift + white noise (relative altitude) ---
        self._baro_drift += rng.standard_normal() * c.baro.drift_walk * np.sqrt(dt)
        baro_alt = float(true_pos[2] + self._baro_drift + rng.normal(0.0, c.baro.noise_std))

        # --- GPS: slow rate, noisy, occasional dropout ---
        gps_valid = False
        self._gps_accum += dt
        if c.gps.available and self._gps_accum >= 1.0 / c.gps.rate_hz:
            self._gps_accum = 0.0
            if rng.random() >= c.gps.dropout_p:
                noise = np.array([
                    rng.normal(0.0, c.gps.xy_noise_std),
                    rng.normal(0.0, c.gps.xy_noise_std),
                    rng.normal(0.0, c.gps.z_noise_std),
                ])
                self._last_gps = true_pos + noise
                gps_valid = True

        # --- downward optical-flow: horizontal velocity rel. to the surface (works in the close-range
        # blind-zone where the ArUco is too zoomed to decode). Noise grows with altitude (flow degrades).
        flow_vel = np.zeros(2)
        flow_valid = False
        if raw_rel_vel_xy is not None and c.flow.min_range <= raw_range <= c.flow.max_range:
            sigma = c.flow.noise_std + c.flow.noise_growth * raw_range
            flow_vel = np.asarray(raw_rel_vel_xy, float)[:2] + rng.normal(0.0, sigma, size=2)
            flow_valid = True

        return SensorReading(
            t=self._t,
            gyro=gyro,
            accel=accel,
            attitude_quat=att,
            range=rng_meas,
            range_valid=range_valid,
            baro_alt=baro_alt,
            gps_pos=self._last_gps.copy(),
            gps_valid=gps_valid,
            flow_vel=flow_vel,
            flow_valid=flow_valid,
        )
