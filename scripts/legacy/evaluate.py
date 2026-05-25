from __future__ import annotations

import argparse

from drone_landing.control import CascadedPIDController
from drone_landing.envs import LandingConfig, LandingEnv
from drone_landing.evaluate import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="sinusoidal")
    args = parser.parse_args()

    env = LandingEnv(LandingConfig(seed=11, platform_mode=args.mode))
    summary = evaluate_policy(env, CascadedPIDController(), episodes=args.episodes)
    print(summary)


if __name__ == "__main__":
    main()

