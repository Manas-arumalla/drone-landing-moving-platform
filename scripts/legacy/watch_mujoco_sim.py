from __future__ import annotations

import argparse
import time

from drone_landing.control import QuadrotorMotorPID
from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="sinusoidal")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--speed", type=float, default=0.18)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    try:
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit("Install MuJoCo first: python -m pip install -e .[mujoco]") from exc

    config = MuJoCoLandingConfig(
        seed=args.seed,
        platform_mode=args.mode,
        platform_speed=args.speed,
        platform_amplitude=args.amplitude,
    )
    env = MuJoCoLandingEnv(config)
    policy = QuadrotorMotorPID()
    obs = env.reset(seed=args.seed)
    total_reward = 0.0
    printed_result = False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.0, 0.0, 1.4]

        while viewer.is_running():
            loop_start = time.perf_counter()

            result = env.step(policy.act(obs))
            obs = result.observation
            total_reward += result.reward
            viewer.sync()

            if (result.terminated or result.truncated) and not printed_result:
                printed_result = True
                print(f"termination={result.info['termination']}")
                print(f"success={result.info['success']}")
                print(f"steps={env.steps}")
                print(f"return={total_reward:.2f}")
                print(f"touchdown_error={result.info['horizontal_error']:.3f}")
                print(f"relative_speed={result.info['relative_horizontal_speed']:.3f}")
                print(f"leg_contacts={result.info['leg_contacts']}")

            if args.realtime:
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, env.config.control_dt - elapsed))


if __name__ == "__main__":
    main()
