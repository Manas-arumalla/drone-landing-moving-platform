from __future__ import annotations

import argparse

from drone_landing.envs.core import LandingConfig
from drone_landing.envs.gym_env import GymLandingEnv


STAGES = {
    "hover": LandingConfig(platform_mode="static", initial_xy_spread=0.25, platform_amplitude=0.0),
    "static": LandingConfig(platform_mode="static", initial_xy_spread=1.0),
    "slow": LandingConfig(platform_mode="sinusoidal", platform_speed=0.25, platform_amplitude=0.8),
    "fast": LandingConfig(platform_mode="sinusoidal", platform_speed=0.65, platform_amplitude=1.6),
    "random": LandingConfig(platform_mode="random_walk", platform_speed=0.75),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, default="hover")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save-path", default="runs/ppo_landing.zip")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: python -m pip install -e .[rl]") from exc

    env = GymLandingEnv(STAGES[args.stage])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/tensorboard")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.save_path)
    print(f"saved={args.save_path}")


if __name__ == "__main__":
    main()

