from __future__ import annotations

import argparse

from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv
from drone_landing.vision import VisualServoController, detect_landing_marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="static")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--speed", type=float, default=0.18)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--camera-width", type=int, default=160)
    parser.add_argument("--camera-height", type=int, default=120)
    args = parser.parse_args()

    env = MuJoCoLandingEnv(
        MuJoCoLandingConfig(
            seed=args.seed,
            platform_mode=args.mode,
            platform_speed=args.speed,
            platform_amplitude=args.amplitude,
        )
    )
    controller = VisualServoController()
    obs = env.reset(seed=args.seed)
    total_reward = 0.0
    visible_count = 0

    while True:
        image = env.render_downward(width=args.camera_width, height=args.camera_height)
        detection = detect_landing_marker(image)
        visible_count += int(detection.visible)
        result = env.step(controller.act(obs, detection))
        obs = result.observation
        total_reward += result.reward
        if result.terminated or result.truncated:
            print(f"termination={result.info['termination']}")
            print(f"success={result.info['success']}")
            print(f"steps={env.steps}")
            print(f"return={total_reward:.2f}")
            print(f"touchdown_error={result.info['horizontal_error']:.3f}")
            print(f"relative_speed={result.info['relative_horizontal_speed']:.3f}")
            print(f"leg_contacts={result.info['leg_contacts']}")
            print(f"marker_visible_frames={visible_count}")
            print(f"last_marker_visible={detection.visible}")
            print(f"last_marker_error=({detection.error_x:.3f}, {detection.error_y:.3f})")
            break


if __name__ == "__main__":
    main()

