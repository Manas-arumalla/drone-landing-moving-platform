"""Phase-2 step 5: the first no-cheats closed loop.

The controller flies on the EKF's relative-state estimate, which is built only from the downward
camera (ArUco), the IMU/AHRS, and the rangefinder — never simulator truth. Ground truth is read
only to score the outcome.

Usage:
    $env:PYTHONPATH="src"; python scripts/run_closed_loop.py --episodes 10 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from drone_landing.estimation import RelativeStateEKF, accel_world, quat_to_rotmat
from drone_landing.perception import ArucoDetector, CameraModel
from drone_landing.sim.world import LandingWorld, LandingWorldConfig
from validate_landing import GeometricCascade

CAM = CameraModel(480, 480, 90.0)
CAM_PERIOD = 3  # ~33 Hz at 100 Hz control
CAM_OFFSET_BODY = np.array([0.0, 0.0, -0.09])  # downward camera position in the body frame
DESCENT_STALE_STEPS = 12  # hold altitude if the marker has not been tracked this many steps


def run_episode(world, ctl, detector, ekf, seed, mode="est", max_steps=2600):
    sensors = world.reset(seed)
    ctl.reset()
    ekf.reset()
    hover = np.full(4, 3.37)
    touchdown_v = None
    last_good_k = -10_000
    for k in range(max_steps):
        if ekf.initialized:
            ekf.predict(world.control_dt, accel_world(sensors.accel, sensors.attitude_quat))
        if k % CAM_PERIOD == 0:
            det = detector.detect(world.render(camera="down", width=CAM.width, height=CAM.height))
            # trust only well-conditioned detections
            good = det.found and det.reproj_error < 3.0 and det.n_markers >= 2
            if good:
                R_ahrs = quat_to_rotmat(sensors.attitude_quat)
                # marker relative to body origin = (marker rel camera) + (camera rel body)
                rel = CAM.opencv_to_world(det.tvec_cam, R_ahrs) + R_ahrs @ CAM_OFFSET_BODY
                last_good_k = k
                if not ekf.initialized and det.n_markers >= 4 and det.reproj_error < 2.0:
                    ekf.reset(r0=rel, v0=np.zeros(3))
                elif ekf.initialized:
                    ekf.update_aruco(rel)
        if ekf.initialized and sensors.range_valid and sensors.range < 3.0:
            ekf.update_range(sensors.range)

        truth = world.observe_truth()
        support = truth["support_feet"]  # gear contact sensor (not position truth)
        true_rp = truth["platform_pos"] - truth["drone_pos"]
        true_rv = truth["platform_vel"] - truth["drone_vel"]
        if mode == "truth":
            ctrl = ctl.act_core(true_rp, true_rv, quat_to_rotmat(truth["drone_quat"]),
                                truth["drone_angvel"], support)
        elif ekf.initialized:
            rp = ekf.rel_pos
            rv = true_rv if mode == "estpos" else ekf.rel_vel
            ctrl = ctl.act_core(rp, rv, quat_to_rotmat(sensors.attitude_quat), sensors.gyro, support)
        else:
            ctrl = hover

        step = world.step(ctrl)
        sensors = step.sensors
        if support >= 1 and touchdown_v is None:
            touchdown_v = step.truth["vertical_speed"]
        if step.terminated or step.truncated:
            return {"outcome": step.info["termination"],
                    "horiz_err": round(step.truth["horizontal_error"], 3),
                    "touchdown_v": None if touchdown_v is None else round(touchdown_v, 3),
                    "time": round(step.truth["time"], 1)}
    return {"outcome": "timeout", "horiz_err": None, "touchdown_v": None, "time": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["truth", "estpos", "est"], default="est",
                    help="truth=full truth; estpos=est position + true velocity; est=full estimate")
    args = ap.parse_args()

    world = LandingWorld(LandingWorldConfig())
    ctl = GeometricCascade(world)
    detector = ArucoDetector(CAM)
    ekf = RelativeStateEKF()

    results = []
    for i in range(args.episodes):
        r = run_episode(world, ctl, detector, ekf, seed=args.seed + i, mode=args.mode)
        results.append(r)
        print(f"ep {i:02d} seed={args.seed + i}: {r}")

    succ = [r for r in results if r["outcome"] == "success"]
    print(f"\nSUCCESS {len(succ)}/{len(results)} = {100 * len(succ) / len(results):.0f}%  (vision-only EKF, no truth in loop)")
    if succ:
        print(f"  mean touchdown horiz err = {np.mean([r['horiz_err'] for r in succ]):.3f} m")
        tv = [r['touchdown_v'] for r in succ if r['touchdown_v'] is not None]
        if tv:
            print(f"  mean contact vspeed      = {np.mean(tv):.3f} m/s")


if __name__ == "__main__":
    main()
