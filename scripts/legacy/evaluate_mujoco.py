from __future__ import annotations

import argparse
import json
from pathlib import Path

from drone_landing.control import QuadrotorMotorPID
from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv
from drone_landing.evaluation import evaluate_mujoco_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="sinusoidal")
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--post-landing-steps", type=int, default=300)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--wind-y", type=float, default=0.0)
    parser.add_argument("--wind-z", type=float, default=0.0)
    parser.add_argument("--wind-gust-std", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    def make_env(seed: int) -> MuJoCoLandingEnv:
        return MuJoCoLandingEnv(
            MuJoCoLandingConfig(
                seed=seed,
                platform_mode=args.mode,
                platform_speed=args.speed,
                platform_amplitude=args.amplitude,
                wind_x=args.wind_x,
                wind_y=args.wind_y,
                wind_z=args.wind_z,
                wind_gust_std=args.wind_gust_std,
            )
        )

    summary = evaluate_mujoco_policy(
        env_factory=make_env,
        policy_factory=QuadrotorMotorPID,
        seeds=seeds,
        post_landing_steps=args.post_landing_steps,
    )
    data = summary.to_dict()
    print(json.dumps({k: v for k, v in data.items() if k != "results"}, indent=2))
    for result in summary.results:
        print(
            f"seed={result.seed} success={result.success} termination={result.termination} "
            f"error={result.touchdown_error:.3f} rel_speed={result.relative_speed:.3f} "
            f"contacts={result.leg_contacts}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote={output.resolve()}")


if __name__ == "__main__":
    main()

