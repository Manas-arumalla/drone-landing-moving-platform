from __future__ import annotations

from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv


class GymMuJoCoLandingEnv:
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: MuJoCoLandingConfig | None = None, render_mode: str | None = None):
        try:
            import gymnasium as gym
            from gymnasium import spaces
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "GymMuJoCoLandingEnv requires optional dependencies. Install with: "
                "python -m pip install -e .[rl,mujoco]"
            ) from exc

        self._gym = gym
        self._np = np
        self.render_mode = render_mode
        self.env = MuJoCoLandingEnv(config or MuJoCoLandingConfig())
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(17,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        obs = self.env.reset(seed=seed)
        return self._np.array(obs, dtype=self._np.float32), {}

    def step(self, action):
        result = self.env.step(action.tolist() if hasattr(action, "tolist") else list(action))
        return (
            self._np.array(result.observation, dtype=self._np.float32),
            float(result.reward),
            result.terminated,
            result.truncated,
            result.info,
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.render_rgb()
        return None

