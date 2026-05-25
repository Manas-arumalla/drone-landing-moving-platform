"""Phase-3 test: robust closed-loop landing with the full vision autopilot
(perception -> Kalman filter -> landing supervisor -> geometric control), no truth in the loop.

Usage:
    $env:PYTHONPATH="src"; python scripts/run_autopilot.py --episodes 10 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from drone_landing.autopilot import AutopilotConfig, VisionLandingAutopilot
from drone_landing.perception import CameraModel
from drone_landing.sim.platforms import GroundMotionConfig
from drone_landing.sim.world import LandingWorld, LandingWorldConfig

CAM = CameraModel(480, 480, 90.0)


def run_episode(world, ap, seed):
    sensors = world.reset(seed)
    ap.reset()
    touchdown_v = None
    while True:
        image = world.render(camera="down", width=CAM.width, height=CAM.height) if ap.wants_frame() else None
        support = world.observe_truth()["support_feet"]   # gear-contact sensor
        ctrl = ap.step(image, sensors, support)
        step = world.step(ctrl)
        sensors = step.sensors
        if support >= 1 and touchdown_v is None:
            touchdown_v = step.truth["vertical_speed"]
        if step.terminated or step.truncated:
            return {"outcome": step.info["termination"],
                    "horiz_err": round(step.truth["horizontal_error"], 3),
                    "touchdown_v": None if touchdown_v is None else round(touchdown_v, 3),
                    "state": ap.state, "time": round(step.truth["time"], 1)}


def main() -> None:
    ap_args = argparse.ArgumentParser()
    ap_args.add_argument("--episodes", type=int, default=10)
    ap_args.add_argument("--seed", type=int, default=0)
    ap_args.add_argument("--vmax", type=float, default=None, help="platform top speed (m/s)")
    ap_args.add_argument("--amax", type=float, default=None, help="platform max accel (m/s^2)")
    ap_args.add_argument("--jerk", type=float, default=None, help="platform max jerk (m/s^3)")
    ap_args.add_argument("--mpc", action="store_true", help="use the predictive MPC horizontal tracker")
    ap_args.add_argument("--ibvs", action="store_true", help="use IBVS guidance (flow velocity)")
    ap_args.add_argument("--ship", action="store_true",
                         help="maritime scenario: 6-DOF seakeeping ship deck (vs. ground rover)")
    ap_args.add_argument("--green-deck", dest="green_deck", action="store_true",
                         help="maritime: time the commit to a low-motion (green) deck window")
    ap_args.add_argument("--sea", default="moderate", choices=["calm", "moderate", "rough"],
                         help="ship sea state (with --ship)")
    args = ap_args.parse_args()

    if args.ship:
        from drone_landing.sim.platforms import sea_state
        cfg = LandingWorldConfig(world="x2_landing_ship", ship=sea_state(args.sea))
    else:
        cfg = LandingWorldConfig()
        if args.vmax is not None or args.amax is not None or args.jerk is not None:
            d = GroundMotionConfig()
            cfg = LandingWorldConfig(ground=GroundMotionConfig(
                v_max=args.vmax if args.vmax is not None else d.v_max,
                a_max=args.amax if args.amax is not None else d.a_max,
                jerk_max=args.jerk if args.jerk is not None else d.jerk_max,
            ))
    world = LandingWorld(cfg)
    mass = float(world.model.body_mass[world.drone_bid])
    inertia = world.model.body_inertia[world.drone_bid].copy()
    autopilot = VisionLandingAutopilot(mass, inertia, CAM, world.control_dt,
                                       config=AutopilotConfig(use_mpc=args.mpc, use_ibvs=args.ibvs,
                                                              use_green_deck=args.green_deck))

    results = []
    for i in range(args.episodes):
        r = run_episode(world, autopilot, seed=args.seed + i)
        results.append(r)
        print(f"ep {i:02d} seed={args.seed + i}: {r}")

    succ = [r for r in results if r["outcome"] == "success"]
    n = len(results)
    print(f"\nSUCCESS {len(succ)}/{n} = {100 * len(succ) / n:.0f}%  (vision autopilot, no truth in loop)")
    if succ:
        print(f"  mean touchdown horiz err = {np.mean([r['horiz_err'] for r in succ]):.3f} m")
        tv = [r['touchdown_v'] for r in succ if r['touchdown_v'] is not None]
        if tv:
            print(f"  mean contact vspeed      = {np.mean(tv):.3f} m/s")
    from collections import Counter
    print("  outcomes:", dict(Counter(r["outcome"] for r in results)))


if __name__ == "__main__":
    main()
