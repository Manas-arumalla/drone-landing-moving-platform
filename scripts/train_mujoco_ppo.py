from __future__ import annotations

import argparse

from drone_landing.envs.mujoco_env import MuJoCoLandingConfig
from drone_landing.envs.mujoco_gym_env import GymMuJoCoLandingEnv


STAGES = {
    "hover": MuJoCoLandingConfig(platform_mode="static", initial_xy_spread=0.15, platform_amplitude=0.0),
    "static": MuJoCoLandingConfig(platform_mode="static", initial_xy_spread=0.7),
    "slow": MuJoCoLandingConfig(platform_mode="sinusoidal", platform_speed=0.20, platform_amplitude=0.7),
    "fast": MuJoCoLandingConfig(platform_mode="sinusoidal", platform_speed=0.55, platform_amplitude=1.5),
    "random": MuJoCoLandingConfig(platform_mode="random_walk", platform_speed=0.65),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, default="hover")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-path", default="runs/ppo_mujoco_landing.zip")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install -e .[rl,mujoco]") from exc

    env = Monitor(GymMuJoCoLandingEnv(STAGES[args.stage]))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="runs/tensorboard",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.save_path)
    print(f"saved={args.save_path}")


if __name__ == "__main__":
    main()

