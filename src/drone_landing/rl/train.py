"""PPO training for the landing guidance policy (Phase 6).

Trains on the fast estimator-noise surrogate env (``LandingEnv``) with domain randomization, so the
policy sees a wide spread of platform motions, winds, and estimator-error realizations. Vectorized
across processes for CPU throughput; uses the GPU automatically if a CUDA build of torch is present.

    drone train --scenario ground --timesteps 2_000_000 --n-envs 8
    # or: python -m drone_landing.rl.train --scenario ship --timesteps 3_000_000

Checkpoints + tensorboard logs land in ``runs/rl/<scenario>/``. The trained policy is later wired into
the autopilot/CLI and evaluated on the *full* vision pipeline (the honest test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drone_landing.rl.landing_env import LandingEnv, LandingEnvConfig
from drone_landing.sim import repo_root


def make_env(scenario: str, domain_rand: bool, seed: int):
    def _factory():
        env = LandingEnv(LandingEnvConfig(scenario=scenario, domain_rand=domain_rand), seed=seed)
        return env
    return _factory


def _make_curriculum_callback(timesteps: int, start: float, ramp_frac: float):
    """Callback that ramps the env difficulty from `start` to 1.0 over the first `ramp_frac` of
    training (research-backed curriculum: learn the easy regime first, then widen to the hard one)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Curriculum(BaseCallback):
        def _on_step(self) -> bool:
            frac = self.num_timesteps / max(1, int(ramp_frac * timesteps))
            d = min(1.0, start + (1.0 - start) * frac)
            self.training_env.env_method("set_difficulty", d)
            return True

    return Curriculum()


def train(scenario: str = "ground", timesteps: int = 2_000_000, n_envs: int = 8,
          device: str = "auto", seed: int = 0, save_dir: str | None = None,
          resume: str | None = None, algo: str = "ppo", curriculum: bool = True,
          difficulty_start: float = 0.2, normalize: bool = False, anneal_lr: bool = False) -> Path:
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    out = Path(save_dir) if save_dir else (repo_root() / "runs" / "rl" / scenario)
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(scenario, True, seed + i) for i in range(n_envs)])
    venv = VecMonitor(venv, filename=str(out / "monitor.csv"))
    # VecNormalize (obs+reward normalization) is the usual top PPO tweak, but here it HURT — its reward
    # normalization dilutes our carefully-shaped, success-bonus-dominated reward (measured: 36% vs 94%).
    # So it is OFF by default (default reproduces the 94% policy); enable with normalize=True to revisit.
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    dev = device
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    lr = (lambda p: 3e-4 * p) if anneal_lr else 3e-4   # optional linear LR annealing

    # Algorithm: PPO (robust baseline) or recurrent PPO (LSTM) for the partial-observability POMDP
    # (noisy/dropping vision estimate) — the 2024-25 SOTA for vision-based ship landing.
    common = dict(device=dev, seed=seed, verbose=1, n_steps=1024, gae_lambda=0.95, gamma=0.995,
                  ent_coef=0.0, learning_rate=lr, n_epochs=10, clip_range=0.2,
                  tensorboard_log=str(out / "tb"))
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        cls, policy = RecurrentPPO, "MlpLstmPolicy"
        # Memory-frugal recurrent config: the LSTM backprop-through-time graph scales with n_steps x
        # n_envs, and the variable-length episode sequences fragment the CUDA allocator over a long run
        # (OOM at ~250k steps on an 8 GB GPU even though the average fits). Shorter rollouts + smaller
        # minibatch keep the peak/fragmented footprint bounded; pair with PYTORCH_CUDA_ALLOC_CONF=
        # expandable_segments:True (set in the launcher) which defragments the caching allocator.
        common.update(n_steps=256, batch_size=256,
                      policy_kwargs=dict(net_arch=[128, 128], lstm_hidden_size=128))
    else:
        from stable_baselines3 import PPO
        cls, policy = PPO, "MlpPolicy"
        common.update(batch_size=4096, policy_kwargs=dict(net_arch=[256, 256]))

    print(f"[train] scenario={scenario} algo={algo} timesteps={timesteps} n_envs={n_envs} device={dev}")
    print(f"[train] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"curriculum={curriculum}")

    model = cls.load(resume, env=venv, device=dev) if resume else cls(policy, venv, **common)
    cbs = [CheckpointCallback(save_freq=max(50_000 // n_envs, 1), save_path=str(out / "ckpt"),
                              name_prefix=algo)]
    if curriculum:
        cbs.append(_make_curriculum_callback(timesteps, difficulty_start, ramp_frac=0.6))
    model.learn(total_timesteps=timesteps, callback=CallbackList(cbs), progress_bar=False)
    final = out / f"{algo}_final.zip"
    model.save(str(final))
    # Save the observation-normalization stats (only when normalizing) so the policy can be deployed.
    if normalize:
        np.save(str(out / "obs_rms.npy"),
                {"mean": venv.obs_rms.mean, "var": venv.obs_rms.var, "clip": venv.clip_obs,
                 "eps": venv.epsilon}, allow_pickle=True)
    venv.close()
    print(f"[train] saved {final}")
    return final


def load_obs_normalizer(model_path: str):
    """Return a function obs->normalized using the saved VecNormalize obs stats (identity if absent)."""
    import numpy as _np
    stats_path = Path(model_path).parent / "obs_rms.npy"
    if not stats_path.exists():
        return lambda o: o
    s = _np.load(str(stats_path), allow_pickle=True).item()
    mean, var, clip, eps = s["mean"], s["var"], float(s["clip"]), float(s["eps"])
    return lambda o: _np.clip((_np.asarray(o) - mean) / _np.sqrt(var + eps), -clip, clip)


def evaluate(model_path: str, scenario: str = "ground", episodes: int = 50, seed: int = 1000,
             hard: bool = True, algo: str = "ppo") -> dict:
    """Compare the trained residual policy vs the zero-action baseline (the proven supervisor+geometric
    guidance) on the SAME hard, domain-randomized episodes. RL's value is the *gap* on the hard regime
    (fast/random rover, wind); on easy episodes the baseline already lands. (Surrogate env; the full
    vision-pipeline eval is done separately via the autopilot integration.)"""
    import numpy as np
    from collections import Counter

    env = LandingEnv(LandingEnvConfig(scenario=scenario, domain_rand=hard))
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device="cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")

    def rollout(policy):
        succ, outc = 0, Counter()
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            if hard:
                env.set_difficulty(1.0)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy(obs))
                done = term or trunc
            outc[info["termination"]] += 1
            succ += info["termination"] == "success"
        return {"success_pct": round(100.0 * succ / episodes, 1), "outcomes": dict(outc)}

    norm = load_obs_normalizer(model_path)                    # apply train-time obs normalization
    pol = lambda o: model.predict(norm(o), deterministic=True)[0]            # noqa: E731
    base = lambda o: np.zeros(env.action_space.shape, dtype=np.float32)      # noqa: E731
    return {"hard": hard, "rl_residual": rollout(pol), "baseline": rollout(base)}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the landing guidance policy with PPO.")
    p.add_argument("--scenario", default="ground", choices=["ground", "ship"])
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None, help="checkpoint .zip to resume from")
    p.add_argument("--algo", default="ppo", choices=["ppo", "recurrent_ppo"])
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    p.add_argument("--normalize", action="store_true",
                   help="enable VecNormalize (obs+reward norm); OFF by default (it hurt this reward)")
    p.add_argument("--anneal-lr", dest="anneal_lr", action="store_true", help="linear LR annealing")
    p.add_argument("--eval", default=None, help="evaluate this checkpoint instead of training")
    args = p.parse_args()
    if args.eval:
        print(evaluate(args.eval, scenario=args.scenario, algo=args.algo))
    else:
        train(args.scenario, args.timesteps, args.n_envs, args.device, args.seed,
              args.save_dir, args.resume, algo=args.algo, curriculum=args.curriculum,
              normalize=args.normalize, anneal_lr=args.anneal_lr)


if __name__ == "__main__":
    main()
