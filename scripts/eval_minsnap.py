"""Honest A/B for the flatness/minimum-snap tracker (planning/minsnap.py, --controller minsnap).

Two questions, answered on matched seeds with the full vision pipeline (no truth in the loop):

  1. Landing outcomes: minsnap vs geometric (and MPC for context) on ground + ship.
     Expectation (recorded before running): estimation is the documented bottleneck, so minsnap
     should be comparable to geometric, not better — the honest claim is smoothness, not success.
  2. Approach smoothness: per-episode RMS jerk and tilt (truth, metrics-only) during the airborne
     approach — the actual benefit a snap-optimal reference + feedforward should deliver.

  python scripts/eval_minsnap.py            # both parts (~15 min)
  python scripts/eval_minsnap.py --quick    # 4 episodes/cell smoke check
"""

from __future__ import annotations

import argparse

import numpy as np

EPISODES = 12
SMOOTH_SEEDS = list(range(8))


def outcomes(episodes: int) -> None:
    from drone_landing.cli import SimSpec, run_batch

    print(f"--- landing outcomes ({episodes} episodes/cell, seed 0) ---")
    rows = []
    for scenario, controller in [("ground", "geometric"), ("ground", "mpc"), ("ground", "minsnap"),
                                 ("ship", "geometric"), ("ship", "minsnap")]:
        s = run_batch(SimSpec(scenario, controller), episodes, 0, verbose=False)
        rows.append(s)
        print(f"  {s['sim']:22s} {s['success_pct']:5.1f}%  herr(succ)={s['mean_horiz_err_succ']}  "
              f"outcomes={s['outcomes']}", flush=True)


def smoothness(seeds: list[int]) -> None:
    """Matched-seed approach-phase smoothness: RMS jerk + tilt from truth (metrics only)."""
    import mujoco

    from drone_landing.cli import CAM_H, CAM_W, SimSpec, build
    from drone_landing.estimation import quat_to_rotmat

    print("\n--- approach smoothness (ground, matched seeds, airborne phase) ---")
    results: dict[str, dict[str, list[float]]] = {}
    for controller in ("geometric", "minsnap"):
        world, ap = build(SimSpec("ground", controller))
        renderer = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)
        jerks, tilts = [], []
        try:
            for seed in seeds:
                sensors = world.reset(seed)
                ap.reset()
                prev_v, prev_a = None, None
                ep_j, ep_t = [], []
                while True:
                    image = None
                    if ap.wants_frame():
                        renderer.update_scene(world.data, camera="down")
                        image = renderer.render()
                    support = world.observe_truth()["support_feet"]
                    ctrl = ap.step(image, sensors, support)
                    step = world.step(ctrl)
                    sensors = step.sensors
                    if support == 0:                       # airborne approach only
                        v = world.data.qvel[world.vadr:world.vadr + 3].copy()
                        if prev_v is not None:
                            a = (v - prev_v) / world.control_dt
                            if prev_a is not None:
                                ep_j.append(float(np.linalg.norm((a - prev_a) / world.control_dt)))
                            prev_a = a
                        prev_v = v
                        q = world.data.qpos[world.qadr + 3:world.qadr + 7]
                        R = quat_to_rotmat(q)
                        ep_t.append(float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))))
                    if step.terminated or step.truncated:
                        break
                if ep_j and ep_t:
                    jerks.append(float(np.sqrt(np.mean(np.square(ep_j)))))
                    tilts.append(float(np.max(ep_t)))
        finally:
            renderer.close()
        results[controller] = {"rms_jerk": jerks, "max_tilt": tilts}
        print(f"  {controller:10s} median RMS jerk = {np.median(jerks):7.1f} m/s^3   "
              f"median max tilt = {np.median(tilts):5.1f} deg   ({len(seeds)} seeds)", flush=True)
    g, m = results["geometric"], results["minsnap"]
    dj = 100.0 * (1.0 - np.median(m["rms_jerk"]) / np.median(g["rms_jerk"]))
    dt_ = 100.0 * (1.0 - np.median(m["max_tilt"]) / np.median(g["max_tilt"]))
    print(f"  -> minsnap vs geometric: RMS jerk {dj:+.0f}% lower, max tilt {dt_:+.0f}% lower "
          "(positive = smoother)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    eps = 4 if args.quick else EPISODES
    seeds = SMOOTH_SEEDS[:3] if args.quick else SMOOTH_SEEDS
    outcomes(eps)
    smoothness(seeds)


if __name__ == "__main__":
    main()
