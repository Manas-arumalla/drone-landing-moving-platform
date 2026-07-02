"""Seeded, reproducible evaluation for the rotor-out workshop paper (docs/paper/outline.md).

Re-runs the three headline experiments behind the paper's figures/tables with FIXED seeds and
exports per-episode CSVs, per-step traces, and publication-ready figures to ``runs/paper/``:

  A. clean-hover A/B (truth state, wind off): reduced-attitude PD vs fixed-point LQR vs the
     spin-averaged precession (Floquet) controller           -> Table: outcomes; Fig: tilt/drift traces
  B. full vision pipeline (ArUco -> EKF -> supervisor), mid-flight rotor failure, floquet mode,
     wind on and off                                          -> Fig: horizontal error vs altitude
  C. terminal ablation on the full pipeline: bounded spinning DESCENT (default contingency)
     vs FLOQUET averaged-precession landing, wind on/off      -> Table: outcome breakdown

Every number in the paper should regenerate from:  python scripts/eval_rotor_out_paper.py
Quick smoke check (2 seeds / 3 episodes):          python scripts/eval_rotor_out_paper.py --quick
One experiment:                                     python scripts/eval_rotor_out_paper.py --exp a
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "paper"

CLEAN_SEEDS = list(range(10))     # experiment A
PIPE_SEEDS = list(range(12))      # experiments B & C (12 episodes, matching BENCHMARK.md cells)
SIM_T_MAX = 16.0                  # clean-hover episode cap (s)
TRACE_EVERY = 5                   # log every 5th control step (100 Hz -> 20 Hz traces)


# ------------------------------------------------------------------ experiment A: clean hover
def _tilt_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def _make_controller(name: str, mass: float, J: np.ndarray, dt: float):
    from drone_landing.control.allocation import x2_allocator
    if name == "pd":
        from drone_landing.control.geometric import GeometricController
        ctl = GeometricController(mass, J, control_dt=dt)
        ctl.failed_rotor = 2
        return lambda rp, rv, R, om: ctl.compute_rotor_out(rp, rv, R, om, vz_des=-0.35)
    if name == "lqr":
        from drone_landing.control.rotor_out_lqr import RotorOutLQR
        ctl = RotorOutLQR(mass, J, x2_allocator(), failed_rotor=2)
        return lambda rp, rv, R, om: ctl.control(rp, rv, R, om, vz_des=-0.35)
    if name == "floquet":
        from drone_landing.control.rotor_out_floquet import RotorOutFloquet
        ctl = RotorOutFloquet(mass, J, x2_allocator(), failed_rotor=2, control_dt=dt)
        return lambda rp, rv, R, om: ctl.control(rp, rv, R, om, vz_des=-0.35)
    raise ValueError(name)


def exp_a(seeds: list[int]) -> None:
    """Clean-hover A/B: truth state, wind off, dead rotor from t=0 (world injects at fail_time=0)."""
    import mujoco

    from drone_landing.estimation import quat_to_rotmat
    from drone_landing.sim.world import LandingWorld, LandingWorldConfig

    rows, traces = [], []
    for name in ("pd", "lqr", "floquet"):
        for seed in seeds:
            w = LandingWorld(LandingWorldConfig(world="x2_landing_ground", failed_rotor=2,
                                                wind_mean=(0.0, 0.0, 0.0), wind_gust_std=0.0))
            w.reset(seed)
            mass = float(w.model.body_mass[w.drone_bid])
            J = w.model.body_inertia[w.drone_bid].copy()
            control = _make_controller(name, mass, J, w.control_dt)
            dp = w._deck_pos
            w.data.qpos[w.qadr:w.qadr + 3] = [dp[0] + 0.3, dp[1], dp[2] + 1.8]
            w.data.qpos[w.qadr + 3:w.qadr + 7] = [1, 0, 0, 0]
            w.data.qvel[w.vadr:w.vadr + 6] = 0
            mujoco.mj_forward(w.model, w.data)

            t, k, max_tilt, touch_vz, came_down = 0.0, 0, 0.0, None, False
            while t < SIM_T_MAX:
                p = w.data.qpos[w.qadr:w.qadr + 3].copy()
                q = w.data.qpos[w.qadr + 3:w.qadr + 7]
                R = quat_to_rotmat(q)
                om = w.data.qvel[w.vadr + 3:w.vadr + 6].copy()
                rel_pos = w._deck_pos - p
                rel_vel = -w.data.qvel[w.vadr:w.vadr + 3].copy()
                tilt = _tilt_deg(R)
                max_tilt = max(max_tilt, tilt)
                drift = float(np.hypot(rel_pos[0], rel_pos[1]))
                if k % TRACE_EVERY == 0:
                    traces.append({"controller": name, "seed": seed, "t": round(t, 2),
                                   "tilt_deg": round(tilt, 2), "drift_m": round(drift, 3),
                                   "alt_m": round(float(p[2] - w._deck_pos[2]), 3)})
                w.step(control(rel_pos, rel_vel, R, om))
                t += w.control_dt
                k += 1
                if w.data.qpos[w.qadr + 2] - w._deck_pos[2] < 0.15:
                    came_down = True
                    touch_vz = float(w.data.qvel[w.vadr + 2])
                    break
            p = w.data.qpos[w.qadr:w.qadr + 3]
            drift = float(np.hypot(p[0] - w._deck_pos[0], p[1] - w._deck_pos[1]))
            rows.append({"controller": name, "seed": seed, "came_down": came_down,
                         "final_drift_m": round(drift, 3), "max_tilt_deg": round(max_tilt, 1),
                         "on_deck": came_down and drift < 0.9, "on_pad": came_down and drift < 0.4,
                         "touch_vz_ms": None if touch_vz is None else round(touch_vz, 3),
                         "t_end_s": round(t, 1)})
            print(f"  A {name:8s} seed {seed}: down={came_down} drift={drift:6.2f} m "
                  f"max_tilt={max_tilt:5.1f} deg", flush=True)

    _write_csv(OUT / "clean_hover.csv", rows)
    _write_csv(OUT / "clean_hover_traces.csv", traces)


# --------------------------------------------------- experiments B & C: full vision pipeline
def _pipeline_batch(mode: str, wind: bool, seeds: list[int], want_traces: bool):
    """Seeded full-pipeline episodes (mid-flight failure at fail_time=3 s); returns rows, traces."""
    import mujoco

    from drone_landing.cli import CAM_H, CAM_W, SimSpec, build

    spec = SimSpec("ground", "geometric", fail_rotor=2, rotor_out_mode=mode, wind=wind)
    world, ap = build(spec)
    renderer = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)
    rows, traces = [], []
    try:
        for seed in seeds:
            sensors = world.reset(seed)
            ap.reset()
            k = 0
            while True:
                image = None
                if ap.wants_frame():
                    renderer.update_scene(world.data, camera="down")
                    image = renderer.render()
                support = world.observe_truth()["support_feet"]
                ctrl = ap.step(image, sensors, support)
                step = world.step(ctrl)
                sensors = step.sensors
                if want_traces and k % TRACE_EVERY == 0:
                    p = world.data.qpos[world.qadr:world.qadr + 3]
                    tracked = (ap.k - ap.last_good_k) <= ap.cfg.track_timeout_steps
                    traces.append({"mode": mode, "wind": wind, "seed": seed,
                                   "t": round(float(step.truth["time"]), 2),
                                   "herr_m": round(float(step.truth["horizontal_error"]), 3),
                                   "alt_m": round(float(p[2] - world._deck_pos[2]), 3),
                                   "tracked": int(tracked), "state": ap.state})
                k += 1
                if step.terminated or step.truncated:
                    rows.append({"mode": mode, "wind": wind, "seed": seed,
                                 "outcome": step.info["termination"],
                                 "herr_m": round(float(step.truth["horizontal_error"]), 3),
                                 "contact_vz_ms": round(float(step.truth["vertical_speed"]), 3),
                                 "t_s": round(float(step.truth["time"]), 1)})
                    print(f"  {mode:8s} wind={int(wind)} seed {seed}: "
                          f"{step.info['termination']:14s} herr={step.truth['horizontal_error']:.2f}",
                          flush=True)
                    break
    finally:
        renderer.close()
    return rows, traces


def exp_bc(seeds: list[int]) -> None:
    all_rows, all_traces = [], []
    for mode in ("floquet", "descent"):
        for wind in (False, True):
            rows, traces = _pipeline_batch(mode, wind, seeds, want_traces=(mode == "floquet"))
            all_rows += rows
            all_traces += traces
    _write_csv(OUT / "pipeline_episodes.csv", all_rows)
    _write_csv(OUT / "pipeline_traces.csv", all_traces)


# ------------------------------------------------------------------------------ figures
def make_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rd(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    # Fig 1: clean-hover tilt + drift traces (one representative seed per controller)
    traces = rd(OUT / "clean_hover_traces.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    colors = {"pd": "#9aa6b2", "lqr": "#d9534f", "floquet": "#1aa564"}
    seed0 = traces[0]["seed"]
    for name in ("pd", "lqr", "floquet"):
        tr = [r for r in traces if r["controller"] == name and r["seed"] == seed0]
        ts = [float(r["t"]) for r in tr]
        ax1.plot(ts, [float(r["tilt_deg"]) for r in tr], label=name, color=colors[name])
        ax2.plot(ts, [float(r["drift_m"]) for r in tr], label=name, color=colors[name])
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("tilt [deg]"); ax1.set_title("instantaneous tilt")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("horizontal drift [m]"); ax2.set_title("drift from deck")
    ax2.set_yscale("log"); ax1.legend(); ax2.legend()
    fig.suptitle(f"Clean-hover rotor-out (truth state, wind off, seed {seed0})")
    fig.tight_layout(); fig.savefig(OUT / "fig_clean_hover.png", bbox_inches="tight"); plt.close(fig)

    # Fig 2: full-pipeline horizontal error vs altitude (floquet), wind on vs off + blind zone
    tr = [r for r in rd(OUT / "pipeline_traces.csv") if r["mode"] == "floquet"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for wind, color, label in ((False, "#2f6df6", "wind off"), (True, "#e8a13a", "wind on")):
        pts = [r for r in tr if r["wind"] == str(wind)]
        alts = np.array([float(r["alt_m"]) for r in pts])
        herr = np.array([float(r["herr_m"]) for r in pts])
        bins = np.linspace(0.0, 2.0, 21)
        mid, med = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (alts >= lo) & (alts < hi)
            if m.sum() >= 5:
                mid.append((lo + hi) / 2)
                med.append(float(np.median(herr[m])))
        ax.plot(med, mid, "-o", ms=3, color=color, label=f"{label} (median)")
    ax.axhspan(0.0, 0.3, color="#d9534f", alpha=0.12)
    ax.text(ax.get_xlim()[1] * 0.98, 0.15, "ArUco blind zone (<0.3 m)", ha="right",
            va="center", fontsize=8, color="#d9534f")
    ax.set_xlabel("horizontal error [m]"); ax.set_ylabel("altitude above deck [m]")
    ax.set_title("Full-pipeline spinning descent (floquet), median by altitude")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig_herr_vs_alt.png", bbox_inches="tight"); plt.close(fig)

    # Fig 3: pipeline outcome breakdown by mode x wind
    rows = rd(OUT / "pipeline_episodes.csv")
    combos = [("descent", "False"), ("descent", "True"), ("floquet", "False"), ("floquet", "True")]
    outcomes = sorted({r["outcome"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    bottoms = np.zeros(len(combos))
    for oc in outcomes:
        vals = [sum(1 for r in rows if (r["mode"], r["wind"]) == c and r["outcome"] == oc)
                for c in combos]
        ax.bar([f"{m}\nwind={'on' if w == 'True' else 'off'}" for m, w in combos], vals,
               bottom=bottoms, label=oc)
        bottoms += np.array(vals, dtype=float)
    ax.set_ylabel("episodes"); ax.set_title("Full-pipeline rotor-out outcomes (12 seeds/cell)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "fig_outcomes.png", bbox_inches="tight"); plt.close(fig)
    print(f"  figures -> {OUT}", flush=True)


# ------------------------------------------------------------------------------ plumbing
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)


def write_manifest(seeds_a: list[int], seeds_bc: list[int]) -> None:
    import mujoco
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "clean_hover_seeds": seeds_a,
        "pipeline_seeds": seeds_bc,
        "fail_time_s": 3.0,
        "notes": "vz_des=-0.35 for all clean-hover controllers; ground scenario; "
                 "pipeline = full vision autopilot (ArUco->EKF->supervisor), no truth in loop.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {OUT / 'manifest.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", choices=["a", "bc", "figs", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="tiny counts for a smoke test")
    args = ap.parse_args()
    seeds_a = CLEAN_SEEDS[:2] if args.quick else CLEAN_SEEDS
    seeds_bc = PIPE_SEEDS[:3] if args.quick else PIPE_SEEDS
    OUT.mkdir(parents=True, exist_ok=True)
    if args.exp in ("a", "all"):
        print("experiment A: clean-hover A/B (pd / lqr / floquet) ...", flush=True)
        exp_a(seeds_a)
    if args.exp in ("bc", "all"):
        print("experiments B+C: full-pipeline (floquet + descent, wind on/off) ...", flush=True)
        exp_bc(seeds_bc)
    if args.exp in ("figs", "all"):
        print("figures ...", flush=True)
        make_figures()
    write_manifest(seeds_a, seeds_bc)


if __name__ == "__main__":
    main()
