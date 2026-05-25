"""Trace the autopilot's supervisor state + estimated/true quantities to see why it won't land."""

from __future__ import annotations

import argparse

import numpy as np

from drone_landing.autopilot import VisionLandingAutopilot
from drone_landing.perception import CameraModel
from drone_landing.sim.world import LandingWorld, LandingWorldConfig

CAM = CameraModel(480, 480, 90.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()
    world = LandingWorld(LandingWorldConfig())
    mass = float(world.model.body_mass[world.drone_bid])
    inertia = world.model.body_inertia[world.drone_bid].copy()
    autop = VisionLandingAutopilot(mass, inertia, CAM, world.control_dt)
    sensors = world.reset(args.seed)
    autop.reset()
    print(f"{'t':>5} {'state':>9} {'clr_e':>6} {'clr_t':>6} {'hor_e':>6} {'hor_t':>6} "
          f"{'rsp_e':>6} {'pstd':>5} {'trk':>3} {'ft':>2}")
    for k in range(2600):
        image = world.render(camera="down", width=CAM.width, height=CAM.height) if autop.wants_frame() else None
        truth = world.observe_truth()
        ctrl = autop.step(image, sensors, truth["support_feet"])
        step = world.step(ctrl)
        sensors = step.sensors
        if k % 25 == 0 or step.terminated or step.truncated:
            rp, rv = autop.ekf.rel_pos, autop.ekf.rel_vel
            clr_e = -rp[2]
            clr_t = truth["drone_pos"][2] - world.deck_top_z
            pstd = float(np.sqrt(np.mean(np.diag(autop.ekf.P)[:2])))
            trk = (autop.k - autop.last_good_k) <= 15
            print(f"{truth['time']:5.1f} {autop.state:>9} {clr_e:6.2f} {clr_t:6.2f} "
                  f"{np.linalg.norm(rp[:2]):6.3f} {truth['horizontal_error']:6.3f} "
                  f"{np.linalg.norm(rv[:2]):6.3f} {pstd:5.2f} {str(trk):>3} {truth['support_feet']:>2}")
        if step.terminated or step.truncated:
            print("OUTCOME:", step.info["termination"])
            break


if __name__ == "__main__":
    main()
