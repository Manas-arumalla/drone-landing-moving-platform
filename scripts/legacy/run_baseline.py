from __future__ import annotations

from drone_landing.control import CascadedPIDController
from drone_landing.envs import LandingConfig, LandingEnv


def main() -> None:
    env = LandingEnv(LandingConfig(seed=7, platform_mode="sinusoidal"))
    policy = CascadedPIDController()
    obs = env.reset()
    total_reward = 0.0

    while True:
        result = env.step(policy.act(obs))
        obs = result.observation
        total_reward += result.reward
        if result.terminated or result.truncated:
            print(f"termination={result.info['termination']}")
            print(f"success={result.info['success']}")
            print(f"steps={env.steps}")
            print(f"return={total_reward:.2f}")
            print(f"touchdown_error={result.info['horizontal_error']:.3f}")
            break


if __name__ == "__main__":
    main()

