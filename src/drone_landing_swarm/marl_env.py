"""MARL environment for swarm collision avoidance under limited communication.

In the hard regime (many drones, short comms range) the reactive CBF can't keep separation — drones
don't see neighbours until they're already close. We learn a **decentralized residual policy**: from a
drone's *local* observation (own state + nearest-K neighbours within comms) it outputs a small velocity
residual on the classical holding/landing guidance, trained to keep separation. ``action = 0`` reproduces
the classical coordinator, so the policy starts safe and only learns to anticipate conflicts the reactive
CBF misses.

Training uses **parameter sharing with an ego agent**: one ego drone is driven by the policy while the
others run the classical coordinator; the ego is randomized each episode so the single shared policy
generalizes, and at deployment it is run on *every* drone (CTDE-style: decentralized execution from local
obs). The reward is dominated by a separation penalty (the metric the classical baseline fails on).
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError("gymnasium is required (pip install -e '.[rl]')") from exc

from drone_landing_swarm.avoidance import min_pairwise_distance
from drone_landing_swarm.coordinator import SwarmConfig, SwarmCoordinator

K_NEIGHBORS = 3
RESIDUAL_SCALE = 0.8        # m/s  residual velocity authority


class SwarmMARLEnv(gym.Env):
    """Single-ego view of the swarm; the shared policy is later deployed on all drones."""

    metadata = {"render_modes": []}

    def __init__(self, config: SwarmConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = config or SwarmConfig(n_drones=14, scenario="ship", comms_range=1.0, spawn_radius=2.5)
        self.coord = SwarmCoordinator(self.cfg)
        dim = 6 + 1 + 6 * K_NEIGHBORS
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.ego = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.coord.reset(int(self._rng.integers(0, 2**31 - 1)))
        self.ego = int(self._rng.integers(0, self.cfg.n_drones))
        return self.coord.local_obs(self.ego, K_NEIGHBORS), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # ego uses the policy residual; all other drones run the classical coordinator (residual 0)
        self.coord.policy_residual = {self.ego: action * RESIDUAL_SCALE}
        self.coord.step()

        ego_landed = self.ego in self.coord.landed
        # separation: closest active neighbour to the ego (true distance, for the reward signal)
        others = [self.coord.pos[j] for j in range(self.cfg.n_drones)
                  if j != self.ego and j not in self.coord.landed]
        min_d = min((float(np.linalg.norm(self.coord.pos[self.ego] - p)) for p in others), default=9.9)
        sep_pen = max(0.0, self.cfg.d_min - min_d)             # >0 when too close
        dist_deck = float(np.linalg.norm((self.coord.pos[self.ego] - self.coord.deck_pos)[:2]))

        reward = (-4.0 * sep_pen                               # keep separation (the hard objective)
                  - 0.02 * dist_deck                          # mild pull toward the deck (still progress)
                  - 0.05 * float(np.sum(action**2))           # small control effort
                  - 0.01)                                     # time
        terminated = False
        truncated = False
        if ego_landed:
            reward += 20.0
            terminated = True
        elif self.coord.t >= self.cfg.max_time or self.ego not in self.coord.pos:
            truncated = True
        # hard collision (well inside the safety distance): strong penalty
        if min_d < 0.5 * self.cfg.d_min:
            reward -= 5.0
        obs = self.coord.local_obs(self.ego, K_NEIGHBORS)
        info = {"min_d": min_d, "ego_landed": ego_landed}
        return obs, reward, terminated, truncated, info
