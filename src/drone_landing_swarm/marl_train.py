"""Train + evaluate the decentralized swarm collision-avoidance policy (parameter-sharing PPO).

    python -m drone_landing_swarm.marl_train --timesteps 1000000
    python -m drone_landing_swarm.marl_train --eval runs/marl/ppo_final.zip

Training: PPO on the ego-view ``SwarmMARLEnv`` (ego randomized per episode; others classical).
Evaluation: deploy the shared policy on **every** drone (decentralized execution from local obs) and
compare against the classical coordinator on the hard regime (many drones, short comms) — the metric is
separation kept (the classical baseline's failure) at equal/again-100% landings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drone_landing_swarm.avoidance import min_pairwise_distance
from drone_landing_swarm.coordinator import SwarmConfig, SwarmCoordinator
from drone_landing_swarm.marl_env import K_NEIGHBORS, RESIDUAL_SCALE, SwarmMARLEnv


def hard_config(n_drones: int = 14, comms: float = 1.0) -> SwarmConfig:
    return SwarmConfig(n_drones=n_drones, scenario="ship", sea="moderate",
                       comms_range=comms, spawn_radius=2.5)


def _repo_runs() -> Path:
    from drone_landing.sim import repo_root
    return repo_root() / "runs" / "marl"


def train(timesteps: int = 1_000_000, n_envs: int = 8, device: str = "auto",
          seed: int = 0, save_dir: str | None = None) -> Path:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    out = Path(save_dir) if save_dir else _repo_runs()
    out.mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([(lambda s=seed + i: SwarmMARLEnv(hard_config(), seed=s)) for i in range(n_envs)])
    venv = VecMonitor(venv, filename=str(out / "monitor.csv"))
    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else (device if device != "auto" else "cpu")
    print(f"[marl] timesteps={timesteps} n_envs={n_envs} device={dev}")
    model = PPO("MlpPolicy", venv, device=dev, seed=seed, verbose=1, n_steps=1024, batch_size=4096,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4, n_epochs=10,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log=str(out / "tb"))
    model.learn(total_timesteps=timesteps,
                callback=CheckpointCallback(save_freq=max(50_000 // n_envs, 1),
                                            save_path=str(out / "ckpt"), name_prefix="ppo"))
    final = out / "ppo_final.zip"
    model.save(str(final))
    venv.close()
    print(f"[marl] saved {final}")
    return final


def _rollout(coord: SwarmCoordinator, model, seed: int) -> dict:
    """One episode; if ``model`` is given, every drone runs the shared policy on its local obs."""
    coord.reset(seed)
    steps = int(coord.cfg.max_time / coord.cfg.dt)
    for _ in range(steps):
        if model is not None:
            res = {}
            for i in range(coord.cfg.n_drones):
                if i not in coord.landed:
                    a, _ = model.predict(coord.local_obs(i, K_NEIGHBORS), deterministic=True)
                    res[i] = np.asarray(a, dtype=float) * RESIDUAL_SCALE
            coord.policy_residual = res
        coord.step()
        if len(coord.landed) == coord.cfg.n_drones:
            break
    return {"all_landed": len(coord.landed) == coord.cfg.n_drones,
            "n_landed": len(coord.landed), "min_sep": float(coord.min_sep),
            "sep_ok": bool(coord.min_sep >= coord.cfg.d_min - 0.15)}


def evaluate(model_path: str, n_drones: int = 14, comms: float = 1.0, episodes: int = 20) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    cfg = hard_config(n_drones, comms)
    out = {}
    for name, mdl in (("classical", None), ("marl", model)):
        rows = [_rollout(SwarmCoordinator(cfg), mdl, seed=1000 + e) for e in range(episodes)]
        out[name] = {
            "all_landed_pct": round(100.0 * sum(r["all_landed"] for r in rows) / episodes, 1),
            "sep_kept_pct": round(100.0 * sum(r["sep_ok"] for r in rows) / episodes, 1),
            "mean_min_sep": round(float(np.mean([r["min_sep"] for r in rows])), 3),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Swarm MARL collision-avoidance training/eval.")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--eval", default=None, help="evaluate this checkpoint instead of training")
    p.add_argument("--drones", type=int, default=14)
    p.add_argument("--comms", type=float, default=1.0)
    args = p.parse_args()
    if args.eval:
        print(evaluate(args.eval, n_drones=args.drones, comms=args.comms))
    else:
        train(args.timesteps, args.n_envs, args.device, args.seed, args.save_dir)


if __name__ == "__main__":
    main()
