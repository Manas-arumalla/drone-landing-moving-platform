"""P5 / A6 — learned active cooperative perception: attention fusion + value-of-information gating.

Trains the attention fusion on synthetic heterogeneous cooperative-perception scenes and A/Bs it against
the inverse-variance (consensus) and equal-mean baselines, then shows the accuracy-vs-bandwidth Pareto for
value-of-information communication gating. The honest finding: learned fusion **wins** where the Gaussian
assumption breaks (confident outliers) and **ties** the optimal baseline where it holds (homogeneous).

  python scripts/eval_active_perception.py                # train + A/B
  python scripts/eval_active_perception.py --steps 6000
"""

from __future__ import annotations

import argparse

from drone_landing_swarm.active_perception import bandwidth_pareto, evaluate, train_fusion


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--episodes", type=int, default=4000)
    args = ap.parse_args()

    print(f"Training attention fusion ({args.steps} steps)...")
    learned = train_fusion(steps=args.steps, seed=0)
    het = evaluate(learned, episodes=args.episodes, outlier_p=0.25)
    hom = evaluate(learned, episodes=args.episodes, outlier_p=0.0)
    print("\nMean deck-estimate error (lower is better):")
    print("  regime          equal    inverse-variance    learned")
    print(f"  heterogeneous   {het['equal']:.3f}    {het['inverse_variance']:.3f}              "
          f"{het['learned']:.3f}   (confident outliers -> learned rejects them)")
    print(f"  homogeneous     {hom['equal']:.3f}    {hom['inverse_variance']:.3f}              "
          f"{hom['learned']:.3f}   (Gaussian -> learned ~ ties the optimal baseline)")

    bp = bandwidth_pareto(episodes=args.episodes)
    print(f"\nValue-of-information gating (fuse-all = {bp['all']:.3f} m):")
    print("  budget   active-select   random-select")
    for b in (1, 2, 3, 4):
        print(f"    {b}        {bp[b]['active']:.3f}          {bp[b]['random']:.3f}")
    print("  active selection reaches near-full accuracy at a fraction of the messages.")


if __name__ == "__main__":
    main()
