"""P4 — landing across the new platform types (inclined deck, USV, moving truck).

Runs the validated vision autopilot (ArUco -> EKF -> supervisor -> geometric, no truth in the loop) on the
new platforms and reports the honest success gradient. The inclined deck shows how a *level*-attitude
touchdown degrades with slope; the truck/USV show the cost of a *continuously translating* target.

  python scripts/eval_platforms.py            # default episode count
  python scripts/eval_platforms.py --episodes 20
"""

from __future__ import annotations

import argparse

from drone_landing.cli import PRESETS, run_batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = ["inclined", "inclined-moderate", "inclined-steep", "truck", "usv"]
    print(f"P4 platform landing ({args.episodes} episodes each, vision autopilot, no truth in loop):\n")
    print(f"  {'platform':<20} success   horiz(succ)   notes")
    for name in names:
        s = run_batch(PRESETS[name], args.episodes, args.seed, verbose=False)
        others = {k: v for k, v in s["outcomes"].items() if k != "success"}
        he = "-" if s["mean_horiz_err_succ"] is None else f"{s['mean_horiz_err_succ']:.3f} m"
        print(f"  {name:<20} {s['success_pct']:5.0f}%   {he:>10}    {others}")
    print("\n  inclined: level-attitude touchdown lands a gentle slope (~6 deg) reliably and degrades on")
    print("  steeper tilt (the level press can't seat 3 feet) -> motivates attitude-matched touchdown.")
    print("  truck/usv: a continuously translating (and, for the USV, rocking) target is harder than a")
    print("  near-stationary rover -- the drone must ride the moving deck through the descent.")


if __name__ == "__main__":
    main()
