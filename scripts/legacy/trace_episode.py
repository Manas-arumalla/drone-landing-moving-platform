"""Trace altitude / thrust / contacts to diagnose early terminations."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from drone_landing.sim.world import LandingWorld, LandingWorldConfig
from validate_landing import GeometricCascade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    world = LandingWorld(LandingWorldConfig())
    ctl = GeometricCascade(world)
    world.reset(args.seed)
    ctl.reset()
    for k in range(2600):
        truth = world.observe_truth()
        ctrl = ctl.act(truth)
        step = world.step(ctrl)
        if k % 10 == 0 or step.terminated or step.truncated:
            t = step.truth
            print(f"t={t['time']:5.2f} z={t['drone_pos'][2]:6.3f} clr={t['drone_pos'][2]-world.deck_top_z:6.3f} "
                  f"Tsum={float(np.sum(ctrl)):6.2f} hErr={t['horizontal_error']:.3f} "
                  f"feet={t['support_feet']} ncon={world.data.ncon} commit={ctl.committed} "
                  f"pz={t['platform_pos'][2]:.3f}")
        if step.terminated or step.truncated:
            print("OUTCOME:", step.info["termination"])
            break


if __name__ == "__main__":
    main()
