"""Maritime green-deck ablation: does timing the commit to a low-motion deck window help?

Runs the vision autopilot on the 6-DOF ship deck with the green-deck timing OFF vs ON, across sea
states, and reports landing success and — the metric the timing targets — the relative vertical
*impact velocity* at touchdown. Everything is closed-loop on onboard sensors only (no truth in the
control path); truth is read only to score the outcome.

Usage:
    $env:PYTHONPATH="src"; python scripts/eval_green_deck.py --episodes 12 --seas moderate rough
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from drone_landing.autopilot import AutopilotConfig, VisionLandingAutopilot
from drone_landing.perception import CameraModel
from drone_landing.sim.platforms import sea_state
from drone_landing.sim.world import LandingWorld, LandingWorldConfig

CAM = CameraModel(480, 480, 90.0)


def run_episode(world, ap, seed):
    sensors = world.reset(seed)
    ap.reset()
    impact_v = None
    while True:
        image = world.render(camera="down", width=CAM.width, height=CAM.height) if ap.wants_frame() else None
        support = world.observe_truth()["support_feet"]
        ctrl = ap.step(image, sensors, support)
        step = world.step(ctrl)
        sensors = step.sensors
        if support >= 1 and impact_v is None:
            impact_v = step.truth["vertical_speed"]   # |v_z(drone) - v_z(deck)| at first contact
        if step.terminated or step.truncated:
            return {"outcome": step.info["termination"], "impact_v": impact_v,
                    "horiz_err": step.truth["horizontal_error"]}


def evaluate(world, ap, n, seed0):
    rows = [run_episode(world, ap, seed0 + i) for i in range(n)]
    succ = [r for r in rows if r["outcome"] == "success"]
    impacts = [r["impact_v"] for r in rows if r["impact_v"] is not None]
    return {
        "success_pct": 100.0 * len(succ) / n,
        "mean_impact": float(np.mean(impacts)) if impacts else float("nan"),
        "max_impact": float(np.max(impacts)) if impacts else float("nan"),
        "mean_impact_succ": float(np.mean([r["impact_v"] for r in succ if r["impact_v"] is not None]))
        if succ else float("nan"),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seas", nargs="+", default=["moderate", "rough"],
                   choices=["calm", "moderate", "rough"])
    args = p.parse_args()

    print(f"Green-deck ablation — {args.episodes} episodes/condition, vision autopilot (no truth in loop)\n")
    header = f"{'sea':<9} {'timing':<10} {'success':>8} {'mean|vz|':>9} {'max|vz|':>8} {'mean|vz|(succ)':>15}"
    print(header)
    print("-" * len(header))
    for sea in args.seas:
        world = LandingWorld(LandingWorldConfig(world="x2_landing_ship", ship=sea_state(sea)))
        mass = float(world.model.body_mass[world.drone_bid])
        inertia = world.model.body_inertia[world.drone_bid].copy()
        for label, green in (("baseline", False), ("green-deck", True)):
            ap = VisionLandingAutopilot(mass, inertia, CAM, world.control_dt,
                                        config=AutopilotConfig(use_green_deck=green))
            m = evaluate(world, ap, args.episodes, args.seed)
            print(f"{sea:<9} {label:<10} {m['success_pct']:>7.0f}% {m['mean_impact']:>9.3f} "
                  f"{m['max_impact']:>8.3f} {m['mean_impact_succ']:>15.3f}")
        print()


if __name__ == "__main__":
    main()
