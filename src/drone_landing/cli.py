"""Unified command-line interface for the drone-landing simulations.

One entry point to choose and launch any scenario/controller combination — individually or several in
parallel. Every run is the full closed-loop vision autopilot (ArUco -> EKF -> supervisor -> control)
on onboard sensors only; ground truth is read only to score outcomes.

Examples
--------
    drone list                                   # show scenarios, controllers, and named presets
    drone run ship --sea rough --green-deck      # one preset-ish run, flags override
    drone run --scenario ground --controller mpc --episodes 20
    drone watch ship --sea moderate              # live MuJoCo viewer (one window)
    drone eval --seas moderate rough             # green-deck ablation table
    drone parallel ground ship-rough ship-rough-green   # run several at once
    drone parallel --all --episodes 12

``drone`` is installed by ``pip install -e .`` (aliases: ``dl``, ``drone-landing``). Without an install,
use ``python -m drone_landing ...`` with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

# Offscreen downward-camera frame size fed to perception (matches the autopilot's CameraModel).
CAM_W = CAM_H = 480
CAM_FOVY = 90.0

SCENARIOS = ("ground", "ship", "offshore", "inclined", "usv", "truck")
CONTROLLERS = ("geometric", "mpc", "ibvs", "rl", "minsnap")
SEA_STATES = ("calm", "moderate", "rough")
INCLINES = ("gentle", "moderate", "steep")
# 6-DOF seakeeping worlds (wave-driven roll/pitch/heave): drive the ship world model
SEAKEEPING = ("ship", "offshore", "inclined", "usv")


def _require_file(path, what: str, hint: str) -> None:
    """Fail with a clear, actionable message if a required model/checkpoint file is missing."""
    from pathlib import Path
    if not Path(path).exists():
        raise SystemExit(f"error: {what} not found at {path}\n       {hint}")


@dataclass(frozen=True)
class SimSpec:
    """A fully-specified simulation configuration."""

    scenario: str = "ground"          # ground | ship
    controller: str = "geometric"     # geometric | mpc | ibvs
    sea: str = "moderate"             # calm | moderate | rough (ship only)
    sea_model: str = "sinusoid"       # sinusoid (validated) | spectral (B1 JONSWAP+RAO) (ship only)
    airwake: bool = False             # ship air-wake (burble) turbulence over the deck (ship only)
    green_deck: bool = False          # maritime green-deck commit timing (ship only)
    vmax: float | None = None         # ground-rover motion overrides (m/s, m/s^2, m/s^3)
    amax: float | None = None
    jerk: float | None = None
    wind: bool = True                 # correlated wind gusts (default on; --no-wind to ablate)
    rl_policy: str | None = None      # trained policy .zip for controller="rl" (default: runs/rl/<scn>/...)
    fail_rotor: int | None = None     # fault-tolerance demo: kill this rotor mid-flight (3-rotor landing)
    rotor_out_mode: str = "descent"   # descent = bounded spinning descent (default); floquet = averaged-precession landing (wind-off)
    dob: bool = False                 # wind-aware disturbance-observer feedforward
    markerless: bool = False          # markerless deck fallback when ArUco can't be decoded
    cnn_markerless: bool = False      # B3: learned CNN deck detector as the markerless fallback
    shield: bool = False              # B2: HJ reachability safe-descent shield (provable soft touchdown)
    motion_data: str | None = None    # P1.2: drive the ship/offshore deck from a recorded 6-DOF CSV
    incline: str = "moderate"         # P4: inclined-deck tilt level (gentle | moderate | steep)
    avoid: bool = False               # P3: in-loop sense-and-avoid (HOCBF) + obstacle-abort (offshore)

    def label(self) -> str:
        seakeeping = self.scenario in ("ship", "offshore")
        parts = [self.scenario]
        if seakeeping:
            parts.append(self.sea)
        if self.scenario == "inclined":
            parts.append(self.incline)
        parts.append(self.controller)
        if self.green_deck:
            parts.append("green")
        if seakeeping and self.sea_model == "spectral":
            parts.append("spectral")
        if seakeeping and self.airwake:
            parts.append("airwake")
        if self.scenario == "ground" and self.vmax is not None:
            parts.append(f"v{self.vmax}")
        if self.fail_rotor is not None:
            parts.append(f"fail{self.fail_rotor}")
        return "/".join(parts)


# Curated, human-friendly named simulations. `dl run <name>` or `dl parallel <name> ...`.
PRESETS: dict[str, SimSpec] = {
    "ground":           SimSpec("ground", "geometric"),
    "ground-mpc":       SimSpec("ground", "mpc"),
    "ground-ibvs":      SimSpec("ground", "ibvs"),
    "ground-minsnap":   SimSpec("ground", "minsnap"),
    "ground-hard":      SimSpec("ground", "geometric", vmax=1.5, amax=0.8, jerk=2.0),
    "ship-calm":        SimSpec("ship", "geometric", sea="calm"),
    "ship":             SimSpec("ship", "geometric", sea="moderate"),
    "ship-rough":       SimSpec("ship", "geometric", sea="rough"),
    "ship-green":       SimSpec("ship", "geometric", sea="moderate", green_deck=True),
    "ship-rough-green": SimSpec("ship", "geometric", sea="rough", green_deck=True),
    "ship-spectral":    SimSpec("ship", "geometric", sea="moderate", sea_model="spectral"),
    "ship-airwake":     SimSpec("ship", "geometric", sea="moderate", airwake=True),
    "offshore":         SimSpec("offshore", "geometric", sea="moderate"),
    "offshore-rough":   SimSpec("offshore", "geometric", sea="rough"),
    "offshore-green":   SimSpec("offshore", "geometric", sea="moderate", green_deck=True),
    "inclined":         SimSpec("inclined", "geometric", incline="gentle"),
    "inclined-moderate": SimSpec("inclined", "geometric", incline="moderate"),
    "inclined-steep":   SimSpec("inclined", "geometric", incline="steep"),
    "inclined-shield":  SimSpec("inclined", "geometric", incline="steep", shield=True),
    "usv":              SimSpec("usv", "geometric"),
    "truck":            SimSpec("truck", "geometric"),
}


# --------------------------------------------------------------------------- build & run
def build(spec: SimSpec):
    """Construct the (world, autopilot) pair for a spec. Imported lazily so `dl list`/`-h` are fast."""
    from drone_landing.autopilot import AutopilotConfig, VisionLandingAutopilot
    from drone_landing.perception import CameraModel
    from drone_landing.sim.mjcf import repo_root
    from drone_landing.sim.platforms import GroundMotionConfig, sea_state
    from drone_landing.sim.world import LandingWorld, LandingWorldConfig

    # inclined/usv reuse the validated 6-DOF ship world (roll/pitch/heave servos); truck reuses ground.
    platform = None
    if spec.scenario in SEAKEEPING:
        world_name = "x2_landing_offshore" if spec.scenario == "offshore" else "x2_landing_ship"
        cfg = LandingWorldConfig(world=world_name,
                                 ship=sea_state(spec.sea, spectral=spec.sea_model == "spectral"),
                                 airwake=spec.airwake)
        if spec.scenario == "inclined":    # P4: a persistently tilted deck (custom platform motion)
            from drone_landing.sim.platforms import InclinedDeckMotion, incline_preset
            platform = InclinedDeckMotion(incline_preset(spec.incline))
        elif spec.scenario == "usv":       # P4: agile surface craft (planar maneuvering + lively seaway)
            from drone_landing.sim.platforms import USVMotion
            platform = USVMotion()
    else:
        ground = GroundMotionConfig()
        if any(v is not None for v in (spec.vmax, spec.amax, spec.jerk)):
            ground = GroundMotionConfig(
                v_max=spec.vmax if spec.vmax is not None else ground.v_max,
                a_max=spec.amax if spec.amax is not None else ground.a_max,
                jerk_max=spec.jerk if spec.jerk is not None else ground.jerk_max,
            )
        cfg = LandingWorldConfig(ground=ground)
        if spec.scenario == "truck":       # P4: road-cruising vehicle (smooth, predictable, faster)
            from drone_landing.sim.platforms import TruckMotion
            platform = TruckMotion()

    from dataclasses import replace as _replace
    if spec.scenario in ("inclined", "usv", "truck"):
        # These moving/tilted decks are harder to ACQUIRE from a wide offset: a ~2 m offset at the 2.5 m
        # spawn altitude puts the marker grid at the 90deg-FOV edge, so the drone never gets the initial
        # 4-marker lock and drifts out (the seed-0 "no controller" failure). Deploy the drone closer to the
        # target (a realistic launch-roughly-overhead assumption) so acquisition is reliable.
        cfg = _replace(cfg, init_xy_spread=0.8)
    if not spec.wind:                      # ablate wind (default is correlated gusts on)
        cfg = _replace(cfg, wind_mean=(0.0, 0.0, 0.0), wind_gust_std=0.0)
    if spec.fail_rotor is not None:        # fault-tolerance demo: inject the rotor failure in the world
        cfg = _replace(cfg, failed_rotor=spec.fail_rotor)

    if spec.motion_data:                   # P1.2: drive the deck from a recorded 6-DOF seakeeping CSV
        _require_file(spec.motion_data, "seakeeping motion CSV",
                      "generate one with `python scripts/gen_seakeeping_data.py`")
        from drone_landing.sim.platforms.data_driven import DataDrivenDeckMotion
        platform = DataDrivenDeckMotion.from_csv(spec.motion_data)
    world = LandingWorld(cfg, platform=platform)
    mass = float(world.model.body_mass[world.drone_bid])
    inertia = world.model.body_inertia[world.drone_bid].copy()
    cam = CameraModel(CAM_W, CAM_H, CAM_FOVY)
    rl_path = None
    if spec.controller == "rl":
        rl_path = spec.rl_policy or str(repo_root() / "runs" / "rl" / spec.scenario / "ppo_final.zip")
        _require_file(rl_path, "trained RL policy",
                      f"train one with `drone train --scenario {spec.scenario}`, or pass --rl-policy <zip>")
    cnn_path = None
    if spec.cnn_markerless:
        cnn_path = str(repo_root() / "runs" / "cnn" / "deck_cnn.pt")
        _require_file(cnn_path, "trained CNN deck detector",
                      "generate + train it (see scripts / perception.cnn_detector.generate_dataset+train)")
    # P3 in-loop sense-and-avoid: the offshore scenario carries superstructure (wheelhouse/mast/bow) to
    # avoid — mirrored as a deck-relative ObstacleField. Other scenarios have a clear deck (no field).
    obstacle_field = None
    if spec.avoid:
        from drone_landing.safety import ObstacleField
        if spec.scenario == "offshore":
            obstacle_field = ObstacleField.offshore_osv(deck_xy=(0.0, 0.0), deck_z=0.0)
        else:
            obstacle_field = ObstacleField()      # no static structure on a clear deck -> passthrough
    ap_cfg = AutopilotConfig(
        use_mpc=spec.controller == "mpc",
        use_ibvs=spec.controller == "ibvs",
        use_minsnap=spec.controller == "minsnap",
        use_green_deck=spec.green_deck,
        rl_policy_path=rl_path,
        failed_rotor=spec.fail_rotor,
        rotor_out_mode=spec.rotor_out_mode,
        use_dob=spec.dob,
        use_markerless=spec.markerless,
        use_cnn_markerless=spec.cnn_markerless,
        cnn_weights_path=cnn_path,
        use_shield=spec.shield,
        use_avoid=spec.avoid,
    )
    autopilot = VisionLandingAutopilot(mass, inertia, cam, world.control_dt, config=ap_cfg,
                                       obstacle_field=obstacle_field)
    return world, autopilot


def run_episode(world, autopilot, renderer, seed: int) -> dict:
    """Run one closed-loop episode using a *reused* renderer (per-frame renderer creation is slow)."""
    sensors = world.reset(seed)
    autopilot.reset()
    impact_v = None
    while True:
        if autopilot.wants_frame():
            renderer.update_scene(world.data, camera="down")
            image = renderer.render()
        else:
            image = None
        support = world.observe_truth()["support_feet"]
        ctrl = autopilot.step(image, sensors, support)
        step = world.step(ctrl)
        sensors = step.sensors
        if support >= 1 and impact_v is None:
            impact_v = step.truth["vertical_speed"]
        if step.terminated or step.truncated:
            return {
                "outcome": step.info["termination"],
                "impact_v": None if impact_v is None else round(float(impact_v), 3),
                "horiz_err": round(float(step.truth["horizontal_error"]), 3),
                "time": round(float(step.truth["time"]), 1),
                "state": autopilot.state,
            }


def summarize(label: str, rows: list[dict]) -> dict:
    n = len(rows)
    succ = [r for r in rows if r["outcome"] == "success"]
    impacts = [r["impact_v"] for r in rows if r["impact_v"] is not None]
    return {
        "sim": label,
        "episodes": n,
        "success_pct": round(100.0 * len(succ) / n, 1) if n else 0.0,
        "mean_horiz_err_succ": round(float(np.mean([r["horiz_err"] for r in succ])), 3) if succ else None,
        "mean_impact": round(float(np.mean(impacts)), 3) if impacts else None,
        "max_impact": round(float(np.max(impacts)), 3) if impacts else None,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }


def run_batch(spec: SimSpec, episodes: int, seed0: int, verbose: bool = True) -> dict:
    """Run `episodes` episodes of a spec and return a summary dict (one reused renderer)."""
    import mujoco

    world, autopilot = build(spec)
    renderer = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)
    try:
        rows = []
        for i in range(episodes):
            r = run_episode(world, autopilot, renderer, seed0 + i)
            rows.append(r)
            if verbose:
                print(f"  ep {i:02d} seed={seed0 + i}: {r}", flush=True)
    finally:
        renderer.close()
    return summarize(spec.label(), rows)


# ----------------------------------------------------------------------- spec resolution
def resolve_spec(args) -> SimSpec:
    """Start from a preset (if named) and apply any explicitly-given flag overrides."""
    base = PRESETS[args.preset] if getattr(args, "preset", None) else SimSpec()
    return replace(
        base,
        scenario=args.scenario or base.scenario,
        controller=args.controller or base.controller,
        sea=args.sea or base.sea,
        incline=getattr(args, "incline", None) or base.incline,
        sea_model=getattr(args, "sea_model", None) or base.sea_model,
        airwake=base.airwake or getattr(args, "airwake", False),
        green_deck=base.green_deck or args.green_deck,
        vmax=args.vmax if args.vmax is not None else base.vmax,
        amax=args.amax if args.amax is not None else base.amax,
        jerk=args.jerk if args.jerk is not None else base.jerk,
        wind=base.wind and getattr(args, "wind", True),
        rl_policy=getattr(args, "rl_policy", None) or base.rl_policy,
        fail_rotor=getattr(args, "fail_rotor", None) if getattr(args, "fail_rotor", None) is not None else base.fail_rotor,
        rotor_out_mode=getattr(args, "rotor_out_mode", None) or base.rotor_out_mode,
        dob=base.dob or getattr(args, "dob", False),
        markerless=base.markerless or getattr(args, "markerless", False),
        cnn_markerless=base.cnn_markerless or getattr(args, "cnn_markerless", False),
        shield=base.shield or getattr(args, "shield", False),
        avoid=base.avoid or getattr(args, "avoid", False),
        motion_data=getattr(args, "motion_data", None) or base.motion_data,
    )


def _add_spec_flags(p: argparse.ArgumentParser, with_preset: bool = True) -> None:
    if with_preset:
        p.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="named preset to start from")
    p.add_argument("--scenario", choices=SCENARIOS, help="ground rover or ship deck")
    p.add_argument("--controller", choices=CONTROLLERS, help="horizontal control law")
    p.add_argument("--sea", choices=SEA_STATES, help="ship sea state (ship only)")
    p.add_argument("--incline", choices=INCLINES, help="inclined-deck tilt level (inclined scenario)")
    p.add_argument("--sea-model", dest="sea_model", choices=["sinusoid", "spectral"],
                   help="ship wave model: sinusoid (validated) or spectral (B1 JONSWAP+RAO)")
    p.add_argument("--airwake", action="store_true",
                   help="ship air-wake (burble) turbulence over the deck (ship only)")
    p.add_argument("--green-deck", dest="green_deck", action="store_true",
                   help="maritime: time the commit to a low-motion (green) deck window")
    p.add_argument("--vmax", type=float, help="ground rover top speed m/s")
    p.add_argument("--amax", type=float, help="ground rover max accel m/s^2")
    p.add_argument("--jerk", type=float, help="ground rover max jerk m/s^3")
    p.add_argument("--no-wind", dest="wind", action="store_false", help="disable wind gusts (ablation)")
    p.add_argument("--rl-policy", dest="rl_policy", default=None,
                   help="trained policy .zip for --controller rl (default: runs/rl/<scenario>/ppo_final.zip)")
    p.add_argument("--fail-rotor", dest="fail_rotor", type=int, choices=[0, 1, 2, 3], default=None,
                   help="fault-tolerance demo: kill this rotor mid-flight (3-rotor landing)")
    p.add_argument("--fail-rotor-mode", dest="rotor_out_mode", choices=["descent", "floquet"], default=None,
                   help="rotor-out handling: descent (bounded spinning descent) or floquet "
                        "(averaged-precession controller that lands the dead-rotor drone wind-off)")
    p.add_argument("--dob", action="store_true", help="wind-aware disturbance-observer feedforward")
    p.add_argument("--markerless", action="store_true",
                   help="markerless deck fallback when the ArUco code can't be decoded")
    p.add_argument("--cnn-markerless", dest="cnn_markerless", action="store_true",
                   help="B3: learned CNN deck detector as the markerless fallback (runs/cnn/deck_cnn.pt)")
    p.add_argument("--shield", action="store_true",
                   help="B2: reachability safe-descent shield (clamp descent so a soft touchdown stays reachable)")
    p.add_argument("--avoid", action="store_true",
                   help="P3: in-loop sense-and-avoid (higher-order CBF) + obstacle-abort around the offshore "
                        "superstructure (latent guard; engages only near a sensed obstacle)")
    p.add_argument("--motion-data", dest="motion_data", default=None,
                   help="P1.2: drive the ship/offshore deck from a recorded 6-DOF CSV (assets/seakeeping/<sea>.csv)")


# --------------------------------------------------------------------------- subcommands
def cmd_list(args) -> int:
    print("Scenarios :", ", ".join(SCENARIOS))
    print("Controllers:", ", ".join(CONTROLLERS))
    print("Sea states :", ", ".join(SEA_STATES), "(ship only)")
    print("\nPresets:")
    width = max(len(n) for n in PRESETS)
    for name, spec in PRESETS.items():
        print(f"  {name:<{width}}  {spec.label()}")
    print("\nFlags (run/watch): --green-deck (ship), --vmax/--amax/--jerk (ground motion), --no-wind,")
    print("                   --sea-model spectral (B1 JONSWAP+RAO waves), --airwake (ship burble),")
    print("                   --dob (wind-aware disturbance observer), --markerless / --cnn-markerless")
    print("                   (deck fallbacks; --cnn-markerless = learned B3 detector),")
    print("                   --shield (B2 reachability safe-descent guard), --fail-rotor K (rotor-out;")
    print("                   --fail-rotor-mode descent|floquet -- floquet lands the dead-rotor drone wind-off),")
    print("                   --incline gentle|moderate|steep (inclined-deck scenario, P4),")
    print("                   --avoid (P3 in-loop sense-and-avoid + obstacle-abort; offshore superstructure)")
    print("\nRun one:        drone run <preset>         (or: drone run --scenario ship --sea rough --green-deck)")
    print("Watch live:     drone watch <preset>")
    print("Run several:    drone parallel <preset> <preset> ...   |   drone parallel --all")
    print("Green-deck A/B: drone eval --seas moderate rough")
    print("Formal safety:  drone reachability   (HJ safe-landing set + runtime shield, B2)")
    print("Sense & avoid:  drone safety         (higher-order CBF obstacle avoidance + contingency FSM, P3)")
    print("Train RL:       drone train --scenario ground [--algo recurrent_ppo]")
    print("Eval RL:        drone train --eval runs/rl/ground/ppo_final.zip   (RL-residual vs baseline)")
    print("Environment:    drone info")
    print("Benchmark/demos: python scripts/benchmark.py (-> docs/BENCHMARK.md) | "
          "python scripts/make_demo_gifs.py (-> docs/media/) [P6]")
    print("Swarm (N drones, separate command):  swarm run --drones 6 --scenario ship --sea rough")
    return 0


def cmd_info(args) -> int:
    """Print the build/capability status: worlds, optional deps, compute device, RL checkpoints."""
    import importlib.util as iu

    from drone_landing.sim.mjcf import WORLDS_DIR, repo_root
    print("Drone moving-platform landing - environment status\n")
    worlds = sorted(p.stem for p in WORLDS_DIR.glob("*.xml"))
    print("Worlds      :", ", ".join(worlds) or "(none)")
    print("Scenarios   :", ", ".join(SCENARIOS), "| controllers:", ", ".join(CONTROLLERS))
    for mod, label in [("cv2", "opencv (ArUco)"), ("casadi", "casadi (MPC)"),
                       ("gymnasium", "gymnasium (RL)"), ("stable_baselines3", "SB3 (RL)")]:
        print(f"  {label:<18}: {'yes' if iu.find_spec(mod) else 'NO'}")
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        print(f"  torch             : {torch.__version__}  (device: {dev})")
    except Exception:
        print("  torch             : not installed")
    rl_dir = repo_root() / "runs" / "rl"
    ckpts = sorted(rl_dir.glob("*/*_final.zip")) if rl_dir.exists() else []
    print("RL policies :", ", ".join(p.parent.name for p in ckpts) or "(none trained yet)")
    print("Swarm       : separate `swarm` command (N-drone deck recovery) - see `swarm info`")
    return 0


def cmd_run(args) -> int:
    spec = resolve_spec(args)
    if not args.json:
        print(f"Running {spec.label()}  ({args.episodes} episodes, seed {args.seed}); "
              "vision autopilot, no truth in loop\n")
    summary = run_batch(spec, args.episodes, args.seed, verbose=not args.json and not args.quiet)
    if args.json:
        print(json.dumps(summary))
    else:
        _print_summary_table([summary])
    return 0


def cmd_watch(args) -> int:
    import time

    import mujoco
    import mujoco.viewer

    spec = resolve_spec(args)
    world, autopilot = build(spec)
    renderer = mujoco.Renderer(world.model, height=CAM_H, width=CAM_W)
    seed, remaining = args.seed, args.episodes
    sensors = world.reset(seed)
    autopilot.reset()
    last_state = ""
    print(f"Watching {spec.label()} — episode seed={seed}  (drag to orbit, scroll to zoom, Tab cycles cameras)")
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        while viewer.is_running() and remaining > 0:
            t0 = time.perf_counter()
            if autopilot.wants_frame():
                renderer.update_scene(world.data, camera="down")
                image = renderer.render()
            else:
                image = None
            support = world.observe_truth()["support_feet"]
            ctrl = autopilot.step(image, sensors, support)
            step = world.step(ctrl)
            sensors = step.sensors
            viewer.sync()
            if autopilot.state != last_state:
                print(f"  [{step.truth['time']:5.1f}s] -> {autopilot.state}")
                last_state = autopilot.state
            if step.terminated or step.truncated:
                print(f"  => {step.info['termination'].upper()}  horiz_err={step.truth['horizontal_error']:.3f} m "
                      f"contact_v={step.truth['vertical_speed']:.3f} m/s")
                time.sleep(1.0)
                remaining -= 1
                seed += 1
                if remaining > 0:
                    sensors = world.reset(seed)
                    autopilot.reset()
                    last_state = ""
                    print(f"episode seed={seed} ...")
            wait = world.control_dt / max(args.speed, 1e-3) - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
    renderer.close()
    return 0


def cmd_eval(args) -> int:
    """Green-deck ablation: baseline vs green-deck commit timing, per sea state."""
    print(f"Green-deck ablation: {args.episodes} episodes/condition, vision autopilot (no truth in loop)\n")
    rows = []
    for sea in args.seas:
        for green in (False, True):
            spec = SimSpec("ship", "geometric", sea=sea, green_deck=green)
            label = f"ship/{sea}/{'green-deck' if green else 'baseline'}"
            print(f"... {label}", flush=True)
            s = run_batch(spec, args.episodes, args.seed, verbose=False)
            s["sim"] = label
            rows.append(s)
    print()
    _print_summary_table(rows)
    return 0


def cmd_train(args) -> int:
    """Train (or, with --eval, evaluate) the RL landing guidance policy. Deps imported lazily."""
    if args.eval:
        _require_file(args.eval, "policy checkpoint", "pass a valid .zip saved by `drone train`")
        from drone_landing.rl.train import evaluate
        print(evaluate(args.eval, scenario=args.scenario, algo=args.algo))
        return 0
    from drone_landing.rl.train import train
    train(scenario=args.scenario, timesteps=args.timesteps, n_envs=args.n_envs,
          device=args.device, seed=args.seed, save_dir=args.save_dir, resume=args.resume,
          algo=args.algo, curriculum=args.curriculum, normalize=args.normalize,
          anneal_lr=args.anneal_lr)
    return 0


def cmd_reachability(args) -> int:
    """B2 formal safety: compute the HJ safe-landing set + show the runtime shield prevents hard landings."""
    import numpy as np

    from drone_landing.control.reachability import LandingReachability, ReachabilityConfig
    R = LandingReachability(ReachabilityConfig(d_max=args.d_max, w_land=args.w_land))
    print(f"HJ safe-landing set: grid {R.cfg.nh}x{R.cfg.nw}, {100 * R.safe.mean():.0f}% of states safe, "
          f"w_land={R.cfg.w_land} m/s, disturbance bound {R.cfg.d_max} m/s^2\n")
    # ASCII map of the safe set over (vertical speed, altitude)
    cols, rows = 56, 20
    ws = np.linspace(-R.cfg.w_max, R.cfg.w_max, cols)
    brake = R.braking_boundary(ws)
    print("Safe to commit ('#'), unsafe ('.'), analytic braking curve ('o'):")
    for h in np.linspace(R.cfg.h_max, 0.0, rows):
        line = "".join("o" if (w < 0 and abs(h - brake[j]) < R.cfg.h_max / rows / 2)
                       else ("#" if R.is_safe(float(h), float(w)) else ".")
                       for j, w in enumerate(ws))
        print("  " + line)
    print(f"  altitude {R.cfg.h_max:.0f}..0 m (top down); vertical speed {-R.cfg.w_max:.0f}..{R.cfg.w_max:.0f}"
          " m/s (left=descending)\n")
    # shield A/B: reckless dive controller under random wind
    rng = np.random.default_rng(0)
    print(f"Runtime shield vs a reckless dive controller under random wind ({args.trials} trials):")
    for shielded in (False, True):
        hard, tds = 0, []
        for _ in range(args.trials):
            h, w = 2.5 + 0.5 * rng.random(), 0.0
            for _ in range(4000):
                a_nom = R.cfg.a_min + 1.0
                a = R.safe_action(h, w, a_nom)[0] if shielded else float(np.clip(a_nom, R.cfg.a_min, R.cfg.a_max))
                w += (a + float(rng.uniform(-R.cfg.d_max, R.cfg.d_max))) * R.cfg.dt
                h += w * R.cfg.dt
                if h <= 0:
                    tds.append(abs(w)); hard += abs(w) > R.cfg.w_land
                    break
        tag = "SHIELDED  " if shielded else "unshielded"
        print(f"  {tag}: hard landings {hard:3d}/{args.trials} = {100 * hard / args.trials:3.0f}%   "
              f"mean |touchdown v| = {np.mean(tds):.2f} m/s")
    return 0


def cmd_safety(args) -> int:
    """P3 sense-and-avoid + contingency failsafes: obstacle avoidance demo and/or the failsafe FSM."""
    from drone_landing.safety.demo import avoid_demo, contingency_demo
    if args.demo in ("avoid", "all"):
        avoid_demo()
        if args.demo == "all":
            print()
    if args.demo in ("contingency", "all"):
        contingency_demo()
    return 0


def _parallel_worker(payload):
    name, episodes, seed = payload
    return run_batch(PRESETS[name], episodes, seed, verbose=False)


def cmd_parallel(args) -> int:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    names = sorted(PRESETS) if args.all else args.names
    if not names:
        print("nothing to run: pass preset names or --all", file=sys.stderr)
        return 2
    bad = [n for n in names if n not in PRESETS]
    if bad:
        print(f"unknown preset(s): {', '.join(bad)}\navailable: {', '.join(sorted(PRESETS))}", file=sys.stderr)
        return 2
    workers = args.workers or min(len(names), max(1, (os.cpu_count() or 4) // 2))
    print(f"Launching {len(names)} simulations on {workers} workers "
          f"({args.episodes} episodes each, seed {args.seed})...\n", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_parallel_worker, (n, args.episodes, args.seed)): n for n in names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                s = fut.result()
            except Exception as exc:  # noqa: BLE001 - surface any worker failure, keep others going
                print(f"  [FAIL] {name}: {exc}", flush=True)
                continue
            print(f"  [done] {name}: {s['success_pct']:.0f}% success, mean impact {s['mean_impact']} m/s",
                  flush=True)
            results.append(s)
    print()
    _print_summary_table(results)
    return 0


# --------------------------------------------------------------------------- output
def _print_summary_table(rows: list[dict]) -> None:
    if not rows:
        print("(no results)")
        return
    name_w = max(12, max(len(r["sim"]) for r in rows))
    header = f"{'sim':<{name_w}} {'eps':>4} {'success':>8} {'horiz(succ)':>12} {'mean|vz|':>9} {'max|vz|':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x["sim"]):
        he = "-" if r["mean_horiz_err_succ"] is None else f"{r['mean_horiz_err_succ']:.3f}"
        mi = "-" if r["mean_impact"] is None else f"{r['mean_impact']:.3f}"
        mx = "-" if r["max_impact"] is None else f"{r['max_impact']:.3f}"
        print(f"{r['sim']:<{name_w}} {r['episodes']:>4} {r['success_pct']:>7.0f}% {he:>12} {mi:>9} {mx:>8}")
    # show non-success outcome breakdown so failures are visible
    for r in sorted(rows, key=lambda x: x["sim"]):
        others = {k: v for k, v in r["outcomes"].items() if k != "success"}
        if others:
            print(f"  {r['sim']}: {others}")


# --------------------------------------------------------------------------- parser
_EPILOG = """\
examples:
  drone list                                  show scenarios, controllers, presets, flags
  drone run ship --sea rough --green-deck     one config, N episodes (headless metrics)
  drone run --controller rl                   run the trained RL policy (ground)
  drone run ship --airwake --dob              ship air-wake stressor + disturbance observer
  drone run ship --sea-model spectral         JONSWAP/PM spectral deck (B1)
  drone run ship --cnn-markerless             learned CNN deck detector fallback (B3)
  drone run inclined --incline steep          tilted-deck landing (P4); also `usv`, `truck`
  drone run offshore --avoid                  in-loop sense-and-avoid on the OSV superstructure (P3)
  drone watch ship --green-deck               live MuJoCo viewer
  drone eval --seas moderate rough            green-deck ablation table
  drone reachability                          HJ safe-landing set + shield (B2)
  drone safety                                sense-and-avoid (HOCBF) + contingency failsafes (P3)
  drone parallel --all                        run every preset concurrently
  drone train --scenario ground --algo recurrent_ppo     train RL (LSTM)
  drone train --eval runs/rl/ground/ppo_final.zip        eval a policy vs baseline
the multi-drone swarm is the separate `swarm` command (see `swarm info`).
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drone", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Drone moving-platform landing - simulation launcher.",
                                epilog=_EPILOG)
    p.set_defaults(func=None)
    sub = p.add_subparsers(dest="command",
                           metavar="{list,info,run,watch,eval,reachability,safety,train,parallel}")

    sub.add_parser("list", help="list scenarios, controllers, and presets").set_defaults(func=cmd_list)
    sub.add_parser("info", help="show build status: worlds, deps, compute device, RL policies").set_defaults(func=cmd_info)

    pr = sub.add_parser("run", help="run one configuration (batch of episodes)")
    _add_spec_flags(pr)
    pr.add_argument("--episodes", type=int, default=10)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--json", action="store_true", help="emit a single JSON summary line")
    pr.add_argument("--quiet", action="store_true", help="suppress per-episode lines")
    pr.set_defaults(func=cmd_run)

    pw = sub.add_parser("watch", help="watch one configuration live in a MuJoCo window")
    _add_spec_flags(pw)
    pw.add_argument("--episodes", type=int, default=10)
    pw.add_argument("--seed", type=int, default=0)
    pw.add_argument("--speed", type=float, default=1.0, help="real-time multiplier")
    pw.set_defaults(func=cmd_watch)

    pe = sub.add_parser("eval", help="green-deck ablation (baseline vs green-deck)")
    pe.add_argument("--seas", nargs="+", default=["moderate", "rough"], choices=SEA_STATES)
    pe.add_argument("--episodes", type=int, default=10)
    pe.add_argument("--seed", type=int, default=0)
    pe.set_defaults(func=cmd_eval)

    ptr = sub.add_parser("train",
                         help="train/evaluate the RL landing policy (residual PPO; --eval scores vs baseline)")
    ptr.add_argument("--scenario", default="ground", choices=["ground", "ship"])
    ptr.add_argument("--timesteps", type=int, default=2_000_000)
    ptr.add_argument("--n-envs", type=int, default=8)
    ptr.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ptr.add_argument("--seed", type=int, default=0)
    ptr.add_argument("--save-dir", default=None)
    ptr.add_argument("--resume", default=None, help="checkpoint .zip to resume from")
    ptr.add_argument("--algo", default="ppo", choices=["ppo", "recurrent_ppo"],
                     help="ppo (baseline) or recurrent_ppo (LSTM, for partial observability)")
    ptr.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                     help="disable the easy->hard difficulty curriculum")
    ptr.add_argument("--normalize", action="store_true",
                     help="enable VecNormalize (obs+reward norm); off by default (it hurt this reward)")
    ptr.add_argument("--anneal-lr", dest="anneal_lr", action="store_true", help="linear LR annealing")
    ptr.add_argument("--eval", default=None,
                     help="evaluate this checkpoint (RL-residual vs baseline on hard episodes) instead of training")
    ptr.set_defaults(func=cmd_train, curriculum=True)

    pp = sub.add_parser("parallel", help="run several presets concurrently")
    pp.add_argument("names", nargs="*", help="preset names (omit with --all)")
    pp.add_argument("--all", action="store_true", help="run every preset")
    pp.add_argument("--episodes", type=int, default=10)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--workers", type=int, default=None, help="max parallel processes")
    pp.set_defaults(func=cmd_parallel)

    pre = sub.add_parser("reachability",
                         help="B2: HJ safe-landing set + runtime-assurance shield (formal safety)")
    pre.add_argument("--d-max", dest="d_max", type=float, default=2.0,
                     help="disturbance bound m/s^2 (wind/air-wake)")
    pre.add_argument("--w-land", dest="w_land", type=float, default=0.6,
                     help="max soft-touchdown vertical speed m/s")
    pre.add_argument("--trials", type=int, default=200, help="shield A/B trials")
    pre.set_defaults(func=cmd_reachability)

    psa = sub.add_parser("safety",
                         help="P3: sense-and-avoid (higher-order CBF) + contingency failsafe FSM demos")
    psa.add_argument("--demo", choices=["avoid", "contingency", "all"], default="all",
                     help="avoid = obstacle sense-and-avoid; contingency = failsafe FSM; all = both")
    psa.set_defaults(func=cmd_safety)

    return p


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()  # safe no-op except on frozen Windows builds
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:           # bare `drone` with no subcommand -> show help
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
