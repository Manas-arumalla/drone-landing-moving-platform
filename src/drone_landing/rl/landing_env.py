"""Gymnasium environment for learning the landing guidance policy.

The RL policy outputs high-level guidance — a horizontal acceleration command and a vertical-velocity
command — which the validated geometric attitude inner loop + control allocation execute on the real
MuJoCo dynamics. So RL learns the *outer* tracking/descent law (the part the classical geometric/MPC/
IBVS controllers occupy), while the trustworthy low-level control and the true contact physics are
unchanged.

**Why a noise surrogate instead of the camera.** Rendering ArUco every step makes RL training ~100x
too slow for the millions of steps PPO needs. So the policy's observation is the *relative state
corrupted by a calibrated noise model* that mimics what the ArUco→EKF estimator actually delivers
(~3 cm position, noisy velocity, occasional dropout), with the real IMU/AHRS sensor stream for
attitude/rate. This is the standard precision-landing RL practice (domain-randomize the estimator
error). The learned policy is later evaluated on the *full* vision pipeline — that is the honest test,
and keeps faith with docs/REALISM_CHARTER.md (no truth in the deployed loop). The privileged truth is
exposed only via ``info`` for an (optional) asymmetric critic and for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - gymnasium is an RL extra
    raise ImportError("gymnasium is required for the RL env (pip install -e '.[rl]')") from exc

from drone_landing.control import GeometricController
from drone_landing.estimation import quat_to_rotmat
from drone_landing.planning import LandingSupervisor
from drone_landing.sim.platforms import (
    GroundMotionConfig,
    RandomGroundMotion,
    ShipDeckMotion,
    sea_state,
)
from drone_landing.sim.world import LandingWorld, LandingWorldConfig


@dataclass(frozen=True)
class LandingEnvConfig:
    scenario: str = "ground"          # ground | ship
    sea: str = "moderate"             # ship sea state
    domain_rand: bool = True          # randomize platform motion, wind, and estimator noise per episode
    difficulty: float = 1.0           # curriculum scalar in [0,1]; scales the randomization ranges
    # estimator-surrogate noise (calibrated to the measured ArUco->EKF error)
    pos_noise: float = 0.03           # m     relative-position observation noise (std)
    vel_noise: float = 0.12           # m/s   relative-velocity observation noise (std)
    dropout_p: float = 0.02           # per-step probability the vision estimate is stale (held)
    # control mode: "residual" = RL perturbs a competent geometric baseline (action=0 already lands;
    # RL only refines -> guaranteed landable, sample-efficient, the research-backed choice); "direct" =
    # RL outputs the full guidance from scratch.
    control_mode: str = "residual"
    a_xy_max: float = 3.0             # m/s^2  horizontal acceleration command (cap)
    vz_max: float = 0.6               # m/s    vertical-velocity command (direct mode)
    residual_a: float = 0.8           # m/s^2  residual horizontal-accel authority (kept modest so the
                                      #         policy stays near the strong baseline; only deviates when it helps)
    residual_vz: float = 0.35         # m/s    residual vertical-velocity authority (residual mode)
    # baseline geometric guidance gains (residual mode)
    base_kp: float = 2.6
    base_kd: float = 3.0
    # reward: POTENTIAL-BASED shaping (Ng et al. 1999) -> r_shape = γ·Φ(s') - Φ(s), with
    # Φ = -(w_pos·dist_xy + w_alt·clearance_when_centred). Potential-based shaping is policy-invariant
    # and, crucially, does NOT accumulate a per-step penalty — so it never incentivizes ending the
    # episode early by flying out of bounds (the bug a raw per-step penalty caused). A small alive bonus
    # further discourages bailing; the large terminal success bonus dominates.
    gamma: float = 0.99               # discount used in the potential shaping
    w_pos: float = 1.0                # potential weight on horizontal distance
    w_alt: float = 0.6                # potential weight on clearance (credited when centred)
    w_tilt: float = 0.2               # small per-step tilt penalty (keep level)
    w_ctrl: float = 0.05              # control-effort penalty (pulls residual toward 0 = baseline unless it helps)
    alive_bonus: float = 0.0          # 0: no hover incentive (supervisor owns descent; success dominates)
    r_success: float = 150.0
    r_crash: float = 60.0
    r_offpad: float = 60.0            # out_of_bounds / off_platform
    r_timeout: float = 5.0


class LandingEnv(gym.Env):
    """Single-drone landing as an RL problem. Action = (a_x, a_y, vz) guidance; physics is true."""

    metadata = {"render_modes": []}

    def __init__(self, config: LandingEnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = config or LandingEnvConfig()
        world_name = "x2_landing_ship" if self.cfg.scenario == "ship" else "x2_landing_ground"
        self._world_cfg = LandingWorldConfig(world=world_name)
        self.world = LandingWorld(self._world_cfg)
        mass = float(self.world.model.body_mass[self.world.drone_bid])
        inertia = self.world.model.body_inertia[self.world.drone_bid].copy()
        self.controller = GeometricController(mass, inertia, control_dt=self.world.control_dt)
        self.supervisor = LandingSupervisor()   # baseline commit/press/cut FSM (residual mode)

        # Residual mode: action is 2-D (horizontal-accel residual only; the supervisor fully owns the
        # vertical descent/commit/cut, so RL can never prevent the landing — it can only improve
        # centring). Direct mode: 3-D (full a_xy + vz guidance from scratch).
        self._adim = 2 if self.cfg.control_mode == "residual" else 3
        base_high = [8, 8, 8, 6, 6, 6, 1, 1, 10, 10, 10]  # rel_pos(3) rel_vel(3) tilt_xy(2) gyro(3)
        high = np.array(base_high + [1.0] * self._adim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._adim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._difficulty = float(self.cfg.difficulty)
        self._prev_action = np.zeros(self._adim, dtype=np.float32)
        self._held_obs = None
        self._sensors = None
        self._truth = None

    def set_difficulty(self, difficulty: float) -> None:
        """Set the curriculum difficulty in [0,1] (called by the training curriculum callback)."""
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ------------------------------------------------------------------ helpers
    def _randomize(self) -> None:
        """Per-episode domain randomization. Swaps the platform motion + wind on the *existing* world
        (no MJCF recompile) so resets stay cheap."""
        if not self.cfg.domain_rand:
            return
        rng = self._rng
        c = self.world.config
        d = self._difficulty                       # 0 = easy (slow/calm), 1 = full hard range
        if self.cfg.scenario == "ship":
            # bias the sea state toward rougher as difficulty rises
            seas = ["calm", "moderate", "rough"]
            weights = np.array([1.0 - d, 0.5 + 0.5 * d, max(0.0, d - 0.2)]) + 1e-3
            self.world.platform = ShipDeckMotion(sea_state(rng.choice(seas, p=weights / weights.sum())))
        else:
            self.world.platform = RandomGroundMotion(GroundMotionConfig(
                v_max=float(rng.uniform(0.4, 0.4 + 0.9 * d)),
                a_max=float(rng.uniform(0.2, 0.2 + 0.5 * d)),
                jerk_max=float(rng.uniform(0.8, 0.8 + 1.2 * d)),
            ))
        # LandingWorldConfig is frozen -> swap in a new one with randomized wind (read live by the sim)
        self.world.config = replace(
            c,
            wind_mean=(float(rng.uniform(-0.5, 0.5) * d), float(rng.uniform(-0.5, 0.5) * d), 0.0),
            wind_gust_std=float(rng.uniform(0.1, 0.1 + 0.4 * d)),
        )

    def _make_obs(self) -> np.ndarray:
        """Estimator-surrogate observation: noisy relative state + real IMU-derived attitude/rate."""
        cfg = self.cfg
        t = self.world.observe_truth()
        # vision dropout -> hold the previous observation (mimics losing the marker)
        if self._held_obs is not None and self._rng.random() < cfg.dropout_p:
            return self._held_obs
        # use the EKF/controller convention: relative = platform - drone (truth rel_pos is drone-platform)
        rel_pos = -t["rel_pos"] + self._rng.normal(0.0, cfg.pos_noise, size=3)
        rel_vel = (t["platform_vel"] - t["drone_vel"]) + self._rng.normal(0.0, cfg.vel_noise, size=3)
        R = quat_to_rotmat(self._sensors.attitude_quat)
        tilt_xy = R[:2, 2]                                  # body-z projected on world x,y (~roll/pitch)
        gyro = self._sensors.gyro
        obs = np.concatenate([rel_pos, rel_vel, tilt_xy, gyro, self._prev_action]).astype(np.float32)
        self._held_obs = obs
        return obs

    # --------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)                # seeds gymnasium's RNG (env-checker contract)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._randomize()                       # swaps platform/wind on the existing world (no recompile)
        self.controller.reset()
        self.supervisor.reset()
        self._sensors = self.world.reset(int(self._rng.integers(0, 2**31 - 1)))
        self._truth = self.world.observe_truth()
        self._prev_action = np.zeros(self._adim, dtype=np.float32)
        self._held_obs = None
        self._phi_prev = self._potential(self.world.observe_truth())
        return self._make_obs(), {}

    def _potential(self, truth) -> float:
        """Shaping potential Φ(s) = -(w_pos·dist_xy + w_alt·clearance·[centred]) — higher is better."""
        dist_xy = float(truth["horizontal_error"])
        clearance = max(0.0, float(truth["rel_pos"][2]))   # truth rel_pos = drone - platform; >0 above deck
        centred = dist_xy < 0.4
        return -(self.cfg.w_pos * dist_xy + self.cfg.w_alt * clearance * (1.0 if centred else 0.0))

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs_prev = self._held_obs if self._held_obs is not None else self._make_obs()
        rel_pos_est = obs_prev[:3]
        rel_vel_est = obs_prev[3:6]
        R_ahrs = quat_to_rotmat(self._sensors.attitude_quat)
        support = int(self.world.observe_truth()["support_feet"])   # onboard gear-contact signal

        if cfg.control_mode == "residual":
            # Baseline = the proven single-drone guidance: LandingSupervisor (commit/press/cut) + a
            # geometric PD horizontal command. The RL action *residuals* the horizontal accel; action=0
            # reproduces the baseline, so the policy starts competent and only learns refinements.
            cmd = self.supervisor.update(rel_pos_est, rel_vel_est, pos_std=0.05, tracked=True,
                                         support_feet=support, dt=self.world.control_dt)
            if cmd.cut:
                ctrl = np.zeros(4)                       # gear planted -> motors off (deck holds it)
            else:
                # RL residuals ONLY the horizontal accel; the supervisor fully owns vz/commit/press, so
                # the policy can improve centring but can never cancel the descent (no hover/timeout bug).
                a_base = cfg.base_kp * rel_pos_est[:2] + cfg.base_kd * rel_vel_est[:2]
                a_xy = a_base + action[:2] * cfg.residual_a
                n = float(np.linalg.norm(a_xy))
                if n > cfg.a_xy_max:
                    a_xy = a_xy * (cfg.a_xy_max / n)
                ctrl = self.controller.compute(rel_pos_est, rel_vel_est, R_ahrs, self._sensors.gyro,
                                               vz_des=cmd.vz_des, press=cmd.press, a_xy_override=a_xy)
        else:  # direct: RL outputs the full guidance from scratch
            a_xy = action[:2] * cfg.a_xy_max
            vz_des = float(action[2] * cfg.vz_max)
            ctrl = self.controller.compute(rel_pos_est, rel_vel_est, R_ahrs, self._sensors.gyro,
                                           vz_des=vz_des, a_xy_override=a_xy)
        step = self.world.step(ctrl)
        self._sensors = step.sensors
        self._truth = step.truth
        self._prev_action = action

        obs = self._make_obs()
        reward, terminated, truncated = self._reward(step, action)
        info = {"termination": step.info["termination"], **{k: step.truth[k] for k in
                ("horizontal_error", "vertical_speed", "tilt_deg", "support_feet")}}
        info["truth_obs"] = np.concatenate([step.truth["rel_pos"], step.truth["drone_vel"]]).astype(np.float32)
        return obs, reward, terminated, truncated, info

    def _reward(self, step, action):
        cfg = self.cfg
        t = step.truth
        # potential-based shaping: γ·Φ(s') - Φ(s) (no accumulating per-step penalty -> no bail-out bug)
        phi = self._potential(t)
        shaping = cfg.gamma * phi - self._phi_prev
        self._phi_prev = phi
        tilt = float(t["tilt_deg"]) / 90.0
        step_r = shaping + cfg.alive_bonus - cfg.w_tilt * tilt - cfg.w_ctrl * float(np.sum(action**2))
        term = step.info["termination"]
        if term == "success":
            return step_r + cfg.r_success, True, False
        if term == "crash":
            return step_r - cfg.r_crash, True, False
        if term in ("out_of_bounds", "off_platform"):
            return step_r - cfg.r_offpad, True, False
        if term == "timeout":
            return step_r - cfg.r_timeout, False, True
        return step_r, step.terminated, step.truncated
