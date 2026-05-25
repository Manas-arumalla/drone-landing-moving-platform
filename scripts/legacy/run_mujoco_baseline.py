from __future__ import annotations

import argparse

from drone_landing.control import QuadrotorMotorPID
from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="sinusoidal")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--lock-after-success", action="store_true")
    parser.add_argument("--post-landing-steps", type=int, default=0)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--wind-y", type=float, default=0.0)
    parser.add_argument("--wind-z", type=float, default=0.0)
    parser.add_argument("--wind-gust-std", type=float, default=0.0)
    args = parser.parse_args()

    config = MuJoCoLandingConfig(
        seed=args.seed,
        platform_mode=args.mode,
        lock_after_success=args.lock_after_success,
        wind_x=args.wind_x,
        wind_y=args.wind_y,
        wind_z=args.wind_z,
        wind_gust_std=args.wind_gust_std,
    )
    if args.speed is not None:
        config = MuJoCoLandingConfig(
            **{**config.__dict__, "platform_speed": args.speed}
        )
    if args.amplitude is not None:
        config = MuJoCoLandingConfig(
            **{**config.__dict__, "platform_amplitude": args.amplitude}
        )
    env = MuJoCoLandingEnv(config)
    policy = QuadrotorMotorPID()
    obs = env.reset(seed=args.seed)
    total_reward = 0.0

    while True:
        result = env.step(policy.act(obs))
        obs = result.observation
        total_reward += result.reward
        if result.terminated or result.truncated:
            for _ in range(args.post_landing_steps):
                result = env.step(policy.act(obs))
                obs = result.observation
            print(f"termination={result.info['termination']}")
            print(f"success={result.info['success']}")
            print(f"steps={env.steps}")
            print(f"return={total_reward:.2f}")
            print(f"touchdown_error={result.info['horizontal_error']:.3f}")
            print(f"relative_speed={result.info['relative_horizontal_speed']:.3f}")
            print(f"leg_contacts={result.info['leg_contacts']}")
            print(f"landed_lock={result.info['landed']}")
            print(
                "motor_rpm="
                f"({result.info['motor_rpm_fl']:.1f}, "
                f"{result.info['motor_rpm_fr']:.1f}, "
                f"{result.info['motor_rpm_rl']:.1f}, "
                f"{result.info['motor_rpm_rr']:.1f})"
            )
            break


if __name__ == "__main__":
    main()
