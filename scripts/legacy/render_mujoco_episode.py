from __future__ import annotations

import argparse
from pathlib import Path

from drone_landing.control import QuadrotorMotorPID
from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="sinusoidal")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--output-dir", default="runs/mujoco_frames")
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Install Pillow to save frames: python -m pip install pillow") from exc

    config = MuJoCoLandingConfig(seed=args.seed, platform_mode=args.mode)
    if args.speed is not None:
        config = MuJoCoLandingConfig(**{**config.__dict__, "platform_speed": args.speed})
    if args.amplitude is not None:
        config = MuJoCoLandingConfig(**{**config.__dict__, "platform_amplitude": args.amplitude})
    env = MuJoCoLandingEnv(config)
    policy = QuadrotorMotorPID()
    obs = env.reset(seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(args.frames):
        result = env.step(policy.act(obs))
        obs = result.observation
        if frame_idx % 2 == 0:
            pixels = env.render_rgb(width=1280, height=720, camera="tracking")
            Image.fromarray(pixels).save(output_dir / f"frame_{frame_idx:04d}.png")
        if result.terminated or result.truncated:
            print(f"ended frame={frame_idx} termination={result.info['termination']}")
            break

    print(f"frames_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
