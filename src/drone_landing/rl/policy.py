"""Deploy a trained residual-RL policy inside the real autopilot (full vision pipeline).

The policy was trained in ``LandingEnv`` (residual mode) on a calibrated estimator-noise surrogate; here
we run it on the *real* ArUco -> EKF estimate. The observation and the residual are reproduced exactly
as in training: obs = [rel_pos(3), rel_vel(3), tilt_xy(2), gyro(3), prev_action(2)] in the
platform-minus-drone convention, and the horizontal accel = geometric PD baseline + action·residual_a.
The supervisor still owns the vertical (descent/commit/cut), so the learned policy only refines
horizontal tracking — exactly the train-time control authority. This is the honest full-pipeline test
of the policy (no truth in the loop)."""

from __future__ import annotations

import numpy as np


class ResidualPolicy:
    """Loads a trained SB3 policy and turns the autopilot's state into a horizontal-accel command."""

    def __init__(self, model_path: str, algo: str = "ppo", base_kp: float = 2.6, base_kd: float = 3.0,
                 residual_a: float = 1.5, a_xy_max: float = 3.0):
        if algo == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(model_path, device="cpu")
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path, device="cpu")
        from drone_landing.rl.train import load_obs_normalizer
        self._norm = load_obs_normalizer(model_path)   # re-apply train-time obs normalization (VecNormalize)
        self.base_kp, self.base_kd = base_kp, base_kd
        self.residual_a, self.a_xy_max = residual_a, a_xy_max
        self.reset()

    def reset(self) -> None:
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._lstm_state = None
        self._episode_start = True

    def horizontal_accel(self, rel_pos: np.ndarray, rel_vel: np.ndarray, R_ahrs: np.ndarray,
                         gyro: np.ndarray) -> np.ndarray:
        """Return the residual-augmented horizontal acceleration command (world frame, 2-D)."""
        obs = np.concatenate([
            rel_pos, rel_vel, R_ahrs[:2, 2], gyro, self._prev_action,
        ]).astype(np.float32)
        obs = np.asarray(self._norm(obs), dtype=np.float32)   # VecNormalize obs stats (identity if none)
        action, self._lstm_state = self.model.predict(
            obs, state=self._lstm_state, episode_start=np.array([self._episode_start]),
            deterministic=True)
        self._episode_start = False
        action = np.clip(np.asarray(action, dtype=np.float32).ravel()[:2], -1.0, 1.0)
        self._prev_action = action
        a_base = self.base_kp * rel_pos[:2] + self.base_kd * rel_vel[:2]
        a_xy = a_base + action * self.residual_a
        n = float(np.linalg.norm(a_xy))
        return a_xy * (self.a_xy_max / n) if n > self.a_xy_max else a_xy
