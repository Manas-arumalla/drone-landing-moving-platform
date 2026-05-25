"""Watch the drone land on the moving rover in an interactive MuJoCo viewer.

Opens a real-time window and loops landing episodes. Drag to orbit, scroll to zoom; press Tab to
cycle to the on-board cameras (track / down). Uses the truth-based validation controller (Phase 1).

Usage:
    $env:PYTHONPATH="src"; python scripts/watch_landing.py --seed 2
    $env:PYTHONPATH="src"; python scripts/watch_landing.py --seed 0 --episodes 10
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from drone_landing.sim.world import LandingWorld, LandingWorldConfig
from validate_landing import GeometricCascade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.0, help="real-time multiplier (1.0 = real time)")
    args = ap.parse_args()

    world = LandingWorld(LandingWorldConfig())
    ctl = GeometricCascade(world)

    seed = args.seed
    remaining = args.episodes
    world.reset(seed)
    ctl.reset()
    print(f"episode seed={seed} ...")

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        while viewer.is_running() and remaining > 0:
            t0 = time.perf_counter()
            truth = world.observe_truth()
            step = world.step(ctl.act(truth))
            viewer.sync()

            if step.terminated or step.truncated:
                tag = step.info["termination"].upper()
                print(f"  -> {tag}  t={step.truth['time']:.1f}s  "
                      f"horiz_err={step.truth['horizontal_error']:.3f} m  "
                      f"contact_v={step.truth['vertical_speed']:.3f} m/s")
                time.sleep(1.2)
                remaining -= 1
                seed += 1
                if remaining > 0:
                    world.reset(seed)
                    ctl.reset()
                    print(f"episode seed={seed} ...")

            dt = world.control_dt / max(args.speed, 1e-3) - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
