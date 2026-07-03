"""P6 dissemination — the benchmark matrix: controller x scenario x disturbance, + swarm.

Runs a curated matrix and writes ``docs/BENCHMARK.md`` (markdown tables), flushing after every cell so
partial progress is saved. Every number is from the no-cheats pipeline (vision/estimate-driven, no truth
in the loop). Heavy (full vision rollouts) — run in the background:

  python scripts/benchmark.py --episodes 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from drone_landing.cli import SimSpec, run_batch

OUT = Path(__file__).resolve().parent.parent / "docs" / "BENCHMARK.md"

# (row-label, SimSpec) grouped into tables. Curated to cover the space without exploding the compute.
LANDING = [
    ("ground (cruising rover)", SimSpec("ground", "geometric")),
    ("ground-hard (fast/random)", SimSpec("ground", "geometric", vmax=1.5, amax=0.8, jerk=2.0)),
    ("ship / calm", SimSpec("ship", "geometric", sea="calm")),
    ("ship / moderate", SimSpec("ship", "geometric", sea="moderate")),
    ("ship / rough", SimSpec("ship", "geometric", sea="rough")),
    ("ship / moderate + green-deck", SimSpec("ship", "geometric", sea="moderate", green_deck=True)),
    ("offshore OSV", SimSpec("offshore", "geometric", sea="moderate")),
    ("inclined / gentle 6deg", SimSpec("inclined", "geometric", incline="gentle")),
    ("inclined / moderate 12deg", SimSpec("inclined", "geometric", incline="moderate")),
    ("USV (maneuver + rock)", SimSpec("usv", "geometric")),
    ("moving truck", SimSpec("truck", "geometric")),
]

CONTROLLERS = [
    ("ground / geometric", SimSpec("ground", "geometric")),
    ("ground / IBVS", SimSpec("ground", "ibvs")),
    ("ground / MPC", SimSpec("ground", "mpc")),
    ("ground / min-snap", SimSpec("ground", "minsnap")),
    ("ground / RL residual", SimSpec("ground", "rl")),
    ("ship / geometric", SimSpec("ship", "geometric")),
    ("ship / IBVS", SimSpec("ship", "ibvs")),
    ("ship / min-snap", SimSpec("ship", "minsnap")),
]

DISTURB = [
    ("ship (baseline)", SimSpec("ship", "geometric", sea="moderate")),
    ("ship + air-wake", SimSpec("ship", "geometric", sea="moderate", airwake=True)),
    ("ship + air-wake + DOB", SimSpec("ship", "geometric", sea="moderate", airwake=True, dob=True)),
    ("ship + spectral waves (B1)", SimSpec("ship", "geometric", sea="moderate", sea_model="spectral")),
    ("ship + shield (B2)", SimSpec("ship", "geometric", sea="moderate", shield=True)),
    ("offshore + sense-and-avoid", SimSpec("offshore", "geometric", sea="moderate", avoid=True)),
]


def _row(label: str, spec: SimSpec, episodes: int, seed: int) -> str:
    s = run_batch(spec, episodes, seed, verbose=False)
    others = {k: v for k, v in s["outcomes"].items() if k != "success"}
    he = "-" if s["mean_horiz_err_succ"] is None else f"{s['mean_horiz_err_succ']:.2f}"
    mi = "-" if s["mean_impact"] is None else f"{s['mean_impact']:.2f}"
    return (f"| {label} | {s['success_pct']:.0f}% | {he} | {mi} | "
            f"{others or '-'} |"), s["success_pct"]


def _table(f, title: str, rows: list, episodes: int, seed: int) -> None:
    f.write(f"\n## {title}\n\n")
    f.write("| configuration | success | horiz-err (succ, m) | mean impact (m/s) | non-success outcomes |\n")
    f.write("|---|---|---|---|---|\n")
    f.flush()
    for label, spec in rows:
        line, pct = _row(label, spec, episodes, seed)
        f.write(line + "\n"); f.flush()
        print(f"  {label:34s} {pct:5.0f}%", flush=True)


def _swarm_table(f, episodes: int) -> None:
    from drone_landing_swarm import SwarmConfig, SwarmCoordinator
    from drone_landing_swarm.multi_deck import MultiDeckConfig, MultiDeckCoordinator
    f.write("\n## Swarm (kinematic, no-cheats sensing) — all-landed % over episodes\n\n")
    f.write("| configuration | all-landed | mean recovered | min-sep (m) | obstacle clear (m) |\n")
    f.write("|---|---|---|---|---|\n"); f.flush()
    cfgs = [
        ("4 drones / ship", SwarmConfig(n_drones=4, scenario="ship", sea="moderate")),
        ("6 drones / ship", SwarmConfig(n_drones=6, scenario="ship", sea="moderate")),
        ("6 drones / consensus (A2)", SwarmConfig(n_drones=6, scenario="ship", consensus=True)),
        ("5 drones / offshore + avoid (P3)",
         SwarmConfig(n_drones=5, scenario="ship", obstacles=((2.1, 0.0, 0.9), (2.75, 0.0, 1.0)))),
    ]
    import numpy as np
    for label, cfg in cfgs:
        rows = [SwarmCoordinator(cfg).run(seed=s) for s in range(episodes)]
        full = 100 * np.mean([r["all_landed"] for r in rows])
        rec = 100 * np.mean([r["n_landed"] / r["n_drones"] for r in rows])
        sep = np.mean([r["min_separation"] for r in rows])
        clr = [r["min_obstacle_clearance"] for r in rows if r.get("min_obstacle_clearance") is not None]
        clrs = f"{min(clr):+.2f}" if clr else "-"
        f.write(f"| {label} | {full:.0f}% | {rec:.0f}% | {sep:.2f} | {clrs} |\n"); f.flush()
        print(f"  {label:34s} {full:5.0f}%", flush=True)
    # multi-deck
    md = MultiDeckConfig(n_drones=9, n_decks=3, scenario="ship", offline_deck=2, offline_time=4.0)
    rows = [MultiDeckCoordinator(md).run(seed=s) for s in range(episodes)]
    full = 100 * np.mean([r["all_landed"] for r in rows])
    f.write(f"| 9 drones -> 3 decks (deck 2 fouls) | {full:.0f}% | "
            f"{100 * np.mean([r['n_landed'] / r['n_drones'] for r in rows]):.0f}% | "
            f"{np.mean([r['min_separation'] for r in rows]):.2f} | - |\n"); f.flush()
    print(f"  {'9->3 decks (foul)':34s} {full:5.0f}%", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Benchmark matrix (P6)\n\n")
        f.write(f"Controller x scenario x disturbance, **{args.episodes} episodes/cell**, vision/estimate "
                "pipeline (no ground truth in the loop). Auto-generated by `scripts/benchmark.py`. "
                "Geometric is the default controller; alternatives are for reference.\n")
        print("Landing scenarios..."); _table(f, "Single-drone landing (geometric)", LANDING, args.episodes, args.seed)
        print("Controllers..."); _table(f, "Controllers (ground & ship)", CONTROLLERS, args.episodes, args.seed)
        print("Disturbance & safety..."); _table(f, "Disturbance rejection & safety (ship/offshore)", DISTURB, args.episodes, args.seed)
        print("Swarm..."); _swarm_table(f, args.episodes)
        f.write(f"\n_Generated in {time.time() - t0:.0f}s._\n")
    print(f"\nWrote {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
