"""GNN-based MARL policy (A4): a permutation-invariant, size-agnostic collision-avoidance policy.

The fixed-K MLP policy (``marl_env``) bakes the neighbour count into the observation, so it is tied to one
swarm size and to an arbitrary neighbour ordering. A4 replaces it with a **graph neural network** over the
ego's neighbour graph: each in-range neighbour is encoded by a shared MLP, the messages are
**permutation-invariantly aggregated** (masked mean + max), and combined with the ego encoding. This is a
one-layer message-passing GNN over the star graph (ego <- neighbours) — the SOTA representation for swarm
policies — and it is **size-agnostic**: a single trained policy runs at *any* number of drones.

Trained with parameter-sharing PPO on a randomized-N ego env (so it sees a range of swarm sizes), then
deployed on every drone. Honest expectation (consistent with the project's MARL finding): in the
comms-physics-limited hard regime it **ties** the classical coordinator on separation — the GNN's win is
**generalization across N** (train small, hold at larger N with the *same* weights), the A4 success metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError("gymnasium is required (pip install -e '.[rl]')") from exc

from drone_landing_swarm.coordinator import SwarmConfig, SwarmCoordinator

MAX_NEIGHBORS = 8
RESIDUAL_SCALE = 0.8
EGO_DIM = 7
NB_DIM = 6


def graph_obs_dim(max_neighbors: int = MAX_NEIGHBORS) -> int:
    return EGO_DIM + NB_DIM * max_neighbors + max_neighbors


# --------------------------------------------------------------------- GNN features extractor

def build_extractor_class():
    """Lazily build the torch features-extractor class (so importing this module needs no torch)."""
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class GNNFeaturesExtractor(BaseFeaturesExtractor):
        """Deep-Sets / message-passing GNN over the ego's neighbour set (permutation-invariant)."""

        def __init__(self, observation_space, features_dim: int = 128,
                     max_neighbors: int = MAX_NEIGHBORS, hidden: int = 64):
            super().__init__(observation_space, features_dim)
            self.m = max_neighbors
            self.phi = nn.Sequential(nn.Linear(NB_DIM, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())   # per-neighbour encoder
            self.ego_enc = nn.Sequential(nn.Linear(EGO_DIM, hidden), nn.ReLU())
            # ego encoding + (mean,max) aggregated messages -> features
            self.head = nn.Sequential(nn.Linear(hidden + 2 * hidden, features_dim), nn.ReLU())

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            b = obs.shape[0]
            ego = obs[:, :EGO_DIM]
            feats = obs[:, EGO_DIM:EGO_DIM + NB_DIM * self.m].reshape(b, self.m, NB_DIM)
            mask = obs[:, EGO_DIM + NB_DIM * self.m:].reshape(b, self.m, 1)        # (b, m, 1)
            h = self.phi(feats) * mask                                            # mask out padded neighbours
            n = mask.sum(dim=1).clamp(min=1.0)                                    # valid-neighbour count
            mean = h.sum(dim=1) / n                                               # masked mean
            very_neg = torch.finfo(h.dtype).min
            mx = torch.where(mask.bool(), h, torch.full_like(h, very_neg)).max(dim=1).values
            mx = torch.where(torch.isinf(mx) | (mx == very_neg), torch.zeros_like(mx), mx)
            agg = torch.cat([mean, mx], dim=1)                                    # permutation-invariant
            return self.head(torch.cat([self.ego_enc(ego), agg], dim=1))

    return GNNFeaturesExtractor


# --------------------------------------------------------------------- env

class SwarmMARLGraphEnv(gym.Env):
    """Ego-view swarm env emitting graph observations; ``n_drones`` randomized per reset for N-diversity."""

    metadata = {"render_modes": []}

    def __init__(self, base: SwarmConfig | None = None, n_range: tuple[int, int] = (6, 10),
                 seed: int | None = None, max_neighbors: int = MAX_NEIGHBORS):
        super().__init__()
        self.base = base or SwarmConfig(scenario="ship", sea="moderate", comms_range=1.0, spawn_radius=2.5)
        self.n_range = n_range
        self.m = max_neighbors
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(graph_obs_dim(max_neighbors),),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._build(int(self._rng.integers(*self.n_range)))

    def _build(self, n: int) -> None:
        from dataclasses import replace
        self.cfg = replace(self.base, n_drones=n)
        self.coord = SwarmCoordinator(self.cfg)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._build(int(self._rng.integers(self.n_range[0], self.n_range[1] + 1)))   # randomize swarm size
        self.coord.reset(int(self._rng.integers(0, 2**31 - 1)))
        self.ego = int(self._rng.integers(0, self.cfg.n_drones))
        return self.coord.local_obs_graph(self.ego, self.m), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.coord.policy_residual = {self.ego: action * RESIDUAL_SCALE}
        self.coord.step()
        ego_landed = self.ego in self.coord.landed
        others = [self.coord.pos[j] for j in range(self.cfg.n_drones)
                  if j != self.ego and j not in self.coord.landed]
        min_d = min((float(np.linalg.norm(self.coord.pos[self.ego] - p)) for p in others), default=9.9)
        sep_pen = max(0.0, self.cfg.d_min - min_d)
        dist_deck = float(np.linalg.norm((self.coord.pos[self.ego] - self.coord.deck_pos)[:2]))
        reward = -4.0 * sep_pen - 0.02 * dist_deck - 0.05 * float(np.sum(action**2)) - 0.01
        terminated = truncated = False
        if ego_landed:
            reward += 20.0
            terminated = True
        elif self.coord.t >= self.cfg.max_time or self.ego not in self.coord.pos:
            truncated = True
        if min_d < 0.5 * self.cfg.d_min:
            reward -= 5.0
        return self.coord.local_obs_graph(self.ego, self.m), reward, terminated, truncated, {"min_d": min_d}


# --------------------------------------------------------------------- train / eval

def _repo_runs() -> Path:
    from drone_landing.sim import repo_root
    return repo_root() / "runs" / "marl_gnn"


def train(timesteps: int = 600_000, n_envs: int = 8, device: str = "cpu", seed: int = 0,
          save_dir: str | None = None, n_range: tuple[int, int] = (6, 10)) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    out = Path(save_dir) if save_dir else _repo_runs()
    out.mkdir(parents=True, exist_ok=True)
    # DummyVecEnv (single process): the kinematic swarm env is fast pure-numpy, so in-process stepping is
    # efficient and avoids the Windows 'spawn' SubprocVecEnv fragility (worker import races -> EOFError).
    venv = DummyVecEnv([(lambda s=seed + i: SwarmMARLGraphEnv(n_range=n_range, seed=s))
                        for i in range(n_envs)])
    venv = VecMonitor(venv, filename=str(out / "monitor.csv"))
    print(f"[marl-gnn] timesteps={timesteps} n_envs={n_envs} device={device} n_range={n_range}")
    model = PPO("MlpPolicy", venv, device=device, seed=seed, verbose=1, n_steps=1024, batch_size=4096,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4, n_epochs=10,
                policy_kwargs=dict(features_extractor_class=build_extractor_class(),
                                   features_extractor_kwargs=dict(features_dim=128),
                                   net_arch=[128, 128]),
                tensorboard_log=str(out / "tb"))
    model.learn(total_timesteps=timesteps,
                callback=CheckpointCallback(save_freq=max(100_000 // n_envs, 1),
                                            save_path=str(out / "ckpt"), name_prefix="gnn"))
    final = out / "gnn_final.zip"
    model.save(str(final))
    venv.close()
    print(f"[marl-gnn] saved {final}")
    return final


def _rollout(coord: SwarmCoordinator, model, seed: int, max_neighbors: int = MAX_NEIGHBORS) -> dict:
    coord.reset(seed)
    for _ in range(int(coord.cfg.max_time / coord.cfg.dt)):
        if model is not None:
            res = {}
            for i in range(coord.cfg.n_drones):
                if i not in coord.landed:
                    a, _ = model.predict(coord.local_obs_graph(i, max_neighbors), deterministic=True)
                    res[i] = np.asarray(a, dtype=float) * RESIDUAL_SCALE
            coord.policy_residual = res
        coord.step()
        if len(coord.landed) == coord.cfg.n_drones:
            break
    return {"all_landed": len(coord.landed) == coord.cfg.n_drones,
            "sep_ok": bool(coord.min_sep >= coord.cfg.d_min - 0.15),
            "min_sep": float(coord.min_sep)}


def evaluate(model_path: str, sizes=(10, 14, 18), comms: float = 1.0, episodes: int = 15) -> dict:
    """Deploy ONE GNN policy at several swarm sizes (generalization across N) vs the classical baseline."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    out = {}
    for n in sizes:
        cfg = SwarmConfig(n_drones=n, scenario="ship", sea="moderate", comms_range=comms, spawn_radius=2.5)
        row = {}
        for name, mdl in (("classical", None), ("gnn", model)):
            rs = [_rollout(SwarmCoordinator(cfg), mdl, seed=2000 + e) for e in range(episodes)]
            row[name] = dict(sep_kept_pct=round(100 * sum(r["sep_ok"] for r in rs) / episodes, 1),
                             mean_min_sep=round(float(np.mean([r["min_sep"] for r in rs])), 3))
        out[f"N={n}"] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="GNN-MARL swarm policy (A4) train/eval.")
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--eval", default=None, help="evaluate this checkpoint across swarm sizes")
    args = p.parse_args()
    if args.eval:
        import json
        print(json.dumps(evaluate(args.eval), indent=2))
    else:
        train(args.timesteps, args.n_envs, args.device, args.seed, args.save_dir)


if __name__ == "__main__":
    main()
