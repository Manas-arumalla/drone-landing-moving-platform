from __future__ import annotations

from drone_landing.envs.core import LandingConfig, LandingEnv


class GymLandingEnv:
    """Gymnasium adapter loaded lazily so the core project works without Gym installed."""

    metadata = {"render_modes": []}

    def __init__(self, config: LandingConfig | None = None):
        try:
            import gymnasium as gym
            from gymnasium import spaces
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "GymLandingEnv requires optional dependencies. Install with: "
                "python -m pip install -e .[rl]"
            ) from exc

        self._gym = gym
        self._np = np
        self.env = LandingEnv(config or LandingConfig())
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(8,), dtype=np.float32)

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
        return None

