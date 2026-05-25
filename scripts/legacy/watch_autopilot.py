"""Watch the vision-based landing AUTOPILOT live (no ground truth in the loop).

Runs the full Phase-3 stack — ArUco perception -> Kalman filter -> landing supervisor -> geometric
control — in an interactive MuJoCo window. The downward camera is rendered offscreen each ~3rd step
to feed perception; the main window shows the drone. The supervisor state (APPROACH / DESCEND /
COMMIT / GO_AROUND / SECURED) prints to the terminal as it changes.

Usage:
    $env:PYTHONPATH="src"; python scripts/watch_autopilot.py --seed 0
    $env:PYTHONPATH="src"; python scripts/watch_autopilot.py --seed 2 --episodes 10 --speed 1.0

Drag to orbit, scroll to zoom, press Tab to cycle onboard cameras.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from drone_landing.autopilot import VisionLandingAutopilot
from drone_landing.perception import CameraModel
from drone_landing.sim.world import LandingWorld, LandingWorldConfig

CAM = CameraModel(480, 480, 90.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.0, help="real-time multiplier")
    args = ap.parse_args()

    world = LandingWorld(LandingWorldConfig())
    mass = float(world.model.body_mass[world.drone_bid])
    inertia = world.model.body_inertia[world.drone_bid].copy()
    autopilot = VisionLandingAutopilot(mass, inertia, CAM, world.control_dt)

    # one reusable offscreen renderer for the downward camera
    cam_renderer = mujoco.Renderer(world.model, height=CAM.height, width=CAM.width)

    seed = args.seed
    remaining = args.episodes
    sensors = world.reset(seed)
    autopilot.reset()
    last_state = ""
    print(f"episode seed={seed} ...")

    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        while viewer.is_running() and remaining > 0:
            t0 = time.perf_counter()
            if autopilot.wants_frame():
                cam_renderer.update_scene(world.data, camera="down")
                image = cam_renderer.render()
            else:
                image = None
            support = world.observe_truth()["support_feet"]
            ctrl = autopilot.step(image, sensors, support)
            step = world.step(ctrl)
            sensors = step.sensors
            viewer.sync()

            if autopilot.state != last_state:
                print(f"  [{step.truth['time']:5.1f}s] state -> {autopilot.state}")
                last_state = autopilot.state

            if step.terminated or step.truncated:
                print(f"  => {step.info['termination'].upper()}  horiz_err="
                      f"{step.truth['horizontal_error']:.3f} m  contact_v={step.truth['vertical_speed']:.3f} m/s")
                time.sleep(1.0)
                remaining -= 1
                seed += 1
                if remaining > 0:
                    sensors = world.reset(seed)
                    autopilot.reset()
                    last_state = ""
                    print(f"episode seed={seed} ...")

            dt = world.control_dt / max(args.speed, 1e-3) - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    cam_renderer.close()


if __name__ == "__main__":
    main()
