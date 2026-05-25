"""Quantify the A2 cooperative consensus deck estimator vs each drone's raw onboard fix.

Drones sit at varied ranges from a moving deck under the no-cheats sensing model (A1): deck-measurement
noise grows with range, and past ``deck_vis_range`` a drone gets NO direct fix at all. The distributed
Kalman-Consensus Filter (Olfati-Saber 2007) lets the well-placed observers stabilize the blind ones.

Run:  python scripts/eval_consensus.py
"""

from __future__ import annotations

import numpy as np

from drone_landing_swarm.consensus import ConsensusConfig, ConsensusDeckEstimator
from drone_landing_swarm.sensing import SensingConfig, SwarmSensing


def main(seed: int = 0) -> None:
    n, dt = 5, 0.05
    rng = np.random.default_rng(seed)
    sens = SwarmSensing(n, SensingConfig(), rng)
    filt = ConsensusDeckEstimator(n, dt, ConsensusConfig())
    filt.reset()

    # ring of drones at increasing range; drones 3,4 are beyond deck_vis_range (6 m) -> blind
    offs = {0: np.array([0.3, 0.2, 1.0]), 1: np.array([2.0, 0.0, 1.5]), 2: np.array([3.5, 1.0, 1.8]),
            3: np.array([7.5, 0.0, 2.0]), 4: np.array([0.0, 8.0, 2.2])}
    raw_err: dict[int, list] = {i: [] for i in range(n)}
    fused_err: dict[int, list] = {i: [] for i in range(n)}
    seen = {i: 0 for i in range(n)}

    for k in range(600):
        t = k * dt
        deck = np.array([1.0 * np.sin(0.3 * t), 0.6 * np.cos(0.2 * t), 0.15 * np.sin(0.8 * t)])
        dvel = np.array([0.3 * np.cos(0.3 * t), -0.12 * np.sin(0.2 * t), 0.12 * np.cos(0.8 * t)])
        pos = {i: deck + offs[i] for i in range(n)}
        view = sens.sense(pos, {i: np.zeros(3) for i in range(n)}, deck, dvel, set(), float("inf"))
        nbr = {i: list(view["neighbors"].get(i, {}).keys()) for i in range(n)}
        fused = filt.step(view["deck_meas"], view["deck_std"], nbr)
        if k > 50:                                            # let the filter warm up
            for i in range(n):
                if view["deck_meas"][i] is not None:
                    raw_err[i].append(float(np.linalg.norm(view["deck_meas"][i] - deck)))
                    seen[i] += 1
                fused_err[i].append(float(np.linalg.norm(fused[i][:3] - deck)))

    print("Per-drone deck-estimate error (mean over run), 5 drones, deck_vis_range=6 m:")
    print("  drone  range   sees_deck   RAW meas err        FUSED (consensus) err")
    for i in range(n):
        r = float(np.linalg.norm(offs[i]))
        raw = ("%.3f m" % np.mean(raw_err[i])) if raw_err[i] else "  --  (no view)"
        tag = "YES" if seen[i] > 0 else "BLIND"
        print("   %d    %5.1fm    %-6s    %-16s    %.3f m" % (i, r, tag, raw, np.mean(fused_err[i])))

    allraw = np.concatenate([raw_err[i] for i in range(n) if raw_err[i]])
    allfus = np.concatenate([fused_err[i] for i in range(n)])
    print("\nNETWORK: mean RAW (drones with a view) = %.3f m   mean FUSED (all drones) = %.3f m  "
          "-> %.0f%% lower" % (allraw.mean(), allfus.mean(), 100 * (1 - allfus.mean() / allraw.mean())))


if __name__ == "__main__":
    main()
