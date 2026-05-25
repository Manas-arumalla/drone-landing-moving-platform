"""The `swarm` command — a SEPARATE launcher for the multi-drone flight-deck recovery application.

Kept distinct from the single-drone `drone` CLI on purpose (the swarm is its own module). Examples:

    swarm info
    swarm run --drones 5 --scenario ship --sea rough --episodes 10
    swarm watch --drones 4 --scenario ship          # matplotlib 3-D animation (needs viz extra)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from drone_landing_swarm.coordinator import SwarmConfig, SwarmCoordinator


# P3 sense-and-avoid: the OSV superstructure as DECK-RELATIVE (dx, dy, keep-out radius) obstacles, matching
# the offshore swarm world's wheelhouse + bow. `--avoid` folds these into the non-bypassable SafetyFilter.
OSV_OBSTACLES = ((2.1, 0.0, 0.9), (2.75, 0.0, 1.0))


def _cfg(args) -> SwarmConfig:
    vision = getattr(args, "vision", False)
    if vision and getattr(args, "engine", "mujoco") != "mujoco":
        raise SystemExit("error: --vision (cooperative perception) needs --engine mujoco "
                         "(per-drone cameras require rendering; kinematic has none)")
    # Heterogeneous fleet (P2.4): --cameras N => only the first N drones carry a camera; the rest are
    # camera-less and recover the deck purely from neighbours' shared fixes (cooperative perception).
    cams = getattr(args, "cameras", None)
    camera_drones = None
    if cams is not None:
        if not vision:
            raise SystemExit("error: --cameras N only applies with --vision (cooperative perception)")
        if not (0 < cams <= args.drones):
            raise SystemExit(f"error: --cameras must be in 1..{args.drones} (drones with a camera)")
        camera_drones = tuple(range(cams))
    obstacles = OSV_OBSTACLES if getattr(args, "avoid", False) else ()
    return SwarmConfig(n_drones=args.drones, scenario=args.scenario, sea=args.sea,
                       n_slots=args.slots, comms_range=getattr(args, "comms", float("inf")),
                       safety_margin=getattr(args, "margin", 0.0),
                       consensus=getattr(args, "consensus", False), vision=vision,
                       camera_drones=camera_drones, offshore=getattr(args, "offshore", False),
                       obstacles=obstacles)


def _make_coordinator(cfg: SwarmConfig, engine: str):
    """Build the coordinator for the chosen engine: mujoco (real physics) or kinematic (point-mass)."""
    if engine == "mujoco":
        from drone_landing_swarm.mujoco_runner import MujocoSwarmCoordinator
        return MujocoSwarmCoordinator(cfg)
    return SwarmCoordinator(cfg)


def cmd_info(args) -> int:
    print("Swarm flight-deck recovery - separate module (reuses single-drone components)\n")
    print("Coordination : optimal slot scheduling (Hungarian) + CBF-QP collision avoidance + holding stack")
    print("Sensing (A1) : no-cheats onboard view - per-drone noisy estimates + delayed/dropped neighbour")
    print("               broadcasts within comms range (zero ground truth in the decision loop)")
    print("Safety (A3)  : non-bypassable CBF filter (every command, classical or learned); discrete-time")
    print("               forward-invariance certificate (alpha*dt<1); 'swarm verify' sweeps separation")
    print("Sense&avoid  : --avoid folds the OSV superstructure (wheelhouse/bow) into the same CBF as static")
    print("               keep-outs (each drone uses its own deck estimate -> no truth); P3 swarm hook")
    print("Consensus(A2): distributed Kalman-Consensus deck estimation (--consensus) - well-placed")
    print("               observers stabilize the blind drones (57% lower deck-estimate error)")
    print("Coop-percep  : REAL per-drone onboard vision (--vision); heterogeneous fleet (--cameras N) -")
    print("               only N drones carry a camera, the camera-less rest recover the deck from shared")
    print("               fixes (camera-less deck error 4.24 m -> 0.34 m via the camera drones)")
    print("Multi-deck(A5): K moving decks + dynamic re-tasking (Hungarian start + decentralized auction);")
    print("               recovers drones stranded by a fouled deck ('swarm multi --foul-deck K').")
    print("               REAL physics multi-deck too ('swarm multi --engine mujoco'): N X2 drones land")
    print("               on K servo decks with distinct on-deck spots (P4, removes the kinematic shortcut)")
    print("GNN-MARL (A4): permutation-invariant graph policy, generalizes across swarm size ('swarm gnn')")
    print("Active CP(A6): learned attention fusion (rejects confident outliers) + value-of-information")
    print("              comms gating under a bandwidth budget ('swarm active'); wins where Gaussian breaks")
    print("Deck motion  : reuses ShipDeckMotion / RandomGroundMotion (validated single-drone models)")
    print("Engines      : mujoco = real X2 physics + contact (default); kinematic = fast point-mass")
    print("Perception   : coordination runs on a calibrated onboard-estimate model (truth + sensor noise),")
    print("               NOT per-drone vision yet (no ArUco/CNN render per drone) - validates coordination")
    print("Commands     : swarm run | watch | verify | multi | marl | gnn | active | info")
    print("Headless     : run (single-deck) | verify (safety sweep) | multi (K decks) | gnn/marl --eval")
    print("Visualize    : watch (single-deck, +--policy for GNN) | multi --watch (K decks) | watch --engine mujoco")
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    print(f"Swarm recovery [{args.engine}]: {cfg.n_drones} drones, {cfg.scenario}"
          f"{'/' + cfg.sea if cfg.scenario == 'ship' else ''}, {cfg.n_slots} slot(s); "
          f"{args.episodes} episodes\n")
    coord = _make_coordinator(cfg, args.engine)   # built once; reset per episode
    rows = []
    for ep in range(args.episodes):
        r = coord.run(seed=args.seed + ep)
        rows.append(r)
        print(f"  ep {ep:02d}: landed {r['n_landed']}/{r['n_drones']}  time {r['time']:5.1f}s  "
              f"min_sep {r['min_separation']:.2f} m  {'OK' if r['separation_ok'] else 'SEP-VIOLATION'}")
    n = len(rows)
    full = sum(r["all_landed"] for r in rows)
    safe = sum(r["separation_ok"] for r in rows)
    import numpy as np
    print(f"\nALL-LANDED {full}/{n} = {100 * full / n:.0f}%   |   separation kept {safe}/{n}")
    landed_frac = np.mean([r["n_landed"] / r["n_drones"] for r in rows])
    print(f"  mean drones recovered: {100 * landed_frac:.0f}%   mean min-separation: "
          f"{np.mean([r['min_separation'] for r in rows]):.2f} m")
    cert = min(r["safety"]["min_certificate"] for r in rows if "safety" in r)
    acts = sum(r["safety"]["activations"] for r in rows if "safety" in r)
    print(f"  safety filter: {acts} activations; worst one-step certificate {cert:+.3f} "
          f"(>= 0 => provably collision-free that step)")
    clr = [r["min_obstacle_clearance"] for r in rows if r.get("min_obstacle_clearance") is not None]
    if clr:
        obs_ok = sum(r.get("obstacle_ok", True) for r in rows)
        print(f"  sense-and-avoid (P3): worst obstacle clearance {min(clr):+.2f} m; "
              f"kept clear {obs_ok}/{n} (>= 0 => never entered a superstructure keep-out)")
    return 0


def cmd_verify(args) -> int:
    """Formal-safety verification harness (A3): sweep seeds, assert no separation violation."""
    from drone_landing_swarm.safety import verify_separation
    print(f"Safety verification [{args.engine}]: {args.drones} drones, {args.scenario}"
          f"{'/' + args.sea if args.scenario == 'ship' else ''}, comms {args.comms}, "
          f"margin {args.margin} m; {args.seeds} seeds\n")
    mk = lambda: _make_coordinator(_cfg(args), args.engine)
    rep = verify_separation(mk, range(args.seeds))
    for k in ("n_seeds", "worst_min_separation", "worst_certificate", "n_violations", "violations"):
        print(f"  {k:22s}= {rep[k]}")
    verdict = "PASS - no separation violations across the sweep" if rep["passed"] else \
              f"FAIL - {rep['n_violations']} violation(s): {rep['violations']}"
    print(f"\n{verdict}")
    return 0 if rep["passed"] else 1


def cmd_watch(args) -> int:
    cfg = _cfg(args)
    if args.engine == "mujoco":
        import time

        import mujoco
        import mujoco.viewer
        coord = _make_coordinator(cfg, "mujoco")
        coord.reset(args.seed)
        print(f"Watching MuJoCo swarm: {cfg.n_drones} drones, {cfg.scenario}/{cfg.sea} "
              f"(drag to orbit, scroll to zoom)")
        with mujoco.viewer.launch_passive(coord.world.model, coord.world.data) as viewer:
            while (viewer.is_running() and coord.t < cfg.max_time
                   and len(coord.landed) < cfg.n_drones):
                t0 = time.perf_counter()
                coord.step()
                viewer.sync()
                wait = coord.world.control_dt - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
        print(f"  landed {len(coord.landed)}/{cfg.n_drones}  min_sep {coord.min_sep:.2f} m")
        return 0

    # kinematic engine -> matplotlib 3-D animation
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed (pip install -e '.[viz]')", file=sys.stderr)
        return 2
    import numpy as np

    coord = SwarmCoordinator(cfg)
    coord.reset(args.seed)
    # Optional: deploy a trained GNN policy on every drone (watch the learned coordinated behaviour).
    policy = None
    if getattr(args, "policy", None):
        _require_file(args.policy, "GNN policy checkpoint", "train one with `swarm gnn`")
        from stable_baselines3 import PPO

        from drone_landing_swarm.marl_gnn import MAX_NEIGHBORS, RESIDUAL_SCALE
        policy = PPO.load(args.policy, device="cpu")
        print(f"  deploying GNN policy {args.policy} on all drones")
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def frame(_):
        ax.clear()
        if policy is not None:                       # learned residual on every active drone
            res = {}
            for i in range(cfg.n_drones):
                if i not in coord.landed:
                    a, _ = policy.predict(coord.local_obs_graph(i, MAX_NEIGHBORS), deterministic=True)
                    res[i] = np.asarray(a, dtype=float) * RESIDUAL_SCALE
            coord.policy_residual = res
        s = coord.step()
        dp = s["deck_pos"]
        ax.scatter(dp[0], dp[1], dp[2], c="gold", s=160, marker="s", label="deck")
        for i, p in s["pos"].items():
            if i in s["landed"]:
                col = "green"
            elif i in s["cleared"]:
                col = "red"
            else:
                col = "tab:blue"
            ax.scatter(p[0], p[1], p[2], c=col, s=40)
            ax.text(p[0], p[1], p[2], str(i))
        ax.set_xlim(dp[0] - 4, dp[0] + 4); ax.set_ylim(dp[1] - 4, dp[1] + 4); ax.set_zlim(0, 3)
        ax.set_title(f"t={s['t']:.1f}s  landed {len(s['landed'])}/{cfg.n_drones}  "
                     f"min_sep {s['min_sep']:.2f}m  (red=landing, blue=holding, green=down)")
        if len(s["landed"]) == cfg.n_drones:
            anim.event_source.stop()

    anim = animation.FuncAnimation(fig, frame, frames=int(cfg.max_time / cfg.dt), interval=30)
    plt.legend(); plt.show()
    return 0


_EPILOG = """\
examples:
  swarm info                                       module status + capabilities
  swarm run --drones 6 --scenario ship --sea rough        N drones, real MuJoCo physics
  swarm run --drones 6 --consensus                  A2 distributed deck estimation
  swarm run --drones 5 --vision --cameras 2         heterogeneous fleet: 2 cameras guide 3 camera-less
  swarm run --drones 5 --offshore --avoid           P3 sense-and-avoid: keep clear of the OSV superstructure
  swarm run --drones 8 --engine kinematic           fast point-mass (no physics)
  swarm verify --drones 6 --seeds 25                A3 formal-safety separation sweep
  swarm multi --drones 9 --decks 3 --foul-deck 2    A5 K decks + fouled-deck re-tasking
  swarm gnn --eval runs/marl_gnn/gnn_final.zip      A4 GNN policy across swarm sizes
  swarm active                                      A6 learned active cooperative perception (P5)
  swarm watch --drones 4 --scenario ship            live 3-D viewer
the single-drone simulator is the separate `drone` command (see `drone list`).
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swarm", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Multi-drone flight-deck recovery (separate from `drone`).",
                                epilog=_EPILOG)
    p.set_defaults(func=None)
    sub = p.add_subparsers(dest="command", metavar="{info,run,watch,verify,multi,marl,gnn,active}")
    sub.add_parser("info", help="show the swarm module status").set_defaults(func=cmd_info)

    def add_common(sp):
        sp.add_argument("--drones", type=int, default=4)
        sp.add_argument("--scenario", default="ship", choices=["ground", "ship"])
        sp.add_argument("--sea", default="moderate", choices=["calm", "moderate", "rough"])
        sp.add_argument("--slots", type=int, default=1, help="drones allowed to land at once")
        sp.add_argument("--engine", default="mujoco", choices=["mujoco", "kinematic"],
                        help="mujoco = real physics + contact (default); kinematic = fast point-mass")
        sp.add_argument("--comms", type=float, default=float("inf"),
                        help="comms/sensing range in m (finite = hard partial-observability regime)")
        sp.add_argument("--margin", type=float, default=0.0,
                        help="extra CBF separation margin in m (robustness to sensing noise)")
        sp.add_argument("--consensus", action="store_true",
                        help="A2: distributed Kalman-Consensus deck estimation (fuse deck fixes across drones)")
        sp.add_argument("--vision", action="store_true",
                        help="cooperative perception: REAL per-drone onboard vision (mujoco engine only); "
                             "blind drones land on neighbours' shared visual fixes")
        sp.add_argument("--cameras", type=int, default=None,
                        help="heterogeneous fleet: only the first N drones carry a camera (needs --vision); "
                             "the camera-less rest recover the deck from neighbours' shared fixes")
        sp.add_argument("--offshore", action="store_true",
                        help="orange OSV-vessel look for the MuJoCo deck (visual only; physics unchanged)")
        sp.add_argument("--avoid", action="store_true",
                        help="P3 sense-and-avoid: fold the OSV superstructure (wheelhouse/bow) into the "
                             "non-bypassable CBF as static keep-outs (each drone uses its own deck estimate)")
        sp.add_argument("--seed", type=int, default=0)

    pr = sub.add_parser("run", help="run swarm recovery episodes (headless metrics)")
    add_common(pr)
    pr.add_argument("--episodes", type=int, default=10)
    pr.set_defaults(func=cmd_run)

    pw = sub.add_parser("watch", help="watch one swarm recovery (MuJoCo 3-D / matplotlib)")
    add_common(pw)
    pw.add_argument("--policy", default=None,
                    help="deploy a trained GNN policy on every drone (kinematic engine) — watch it fly")
    pw.set_defaults(func=cmd_watch)

    pv = sub.add_parser("verify", help="formal safety: sweep seeds, assert no separation violation (A3)")
    add_common(pv)
    pv.add_argument("--seeds", type=int, default=25, help="number of seeds to sweep")
    pv.set_defaults(func=cmd_verify)

    pk = sub.add_parser("multi", help="A5: M drones -> K moving decks with dynamic re-tasking")
    pk.add_argument("--drones", type=int, default=9)
    pk.add_argument("--decks", type=int, default=3)
    pk.add_argument("--scenario", default="ship", choices=["ground", "ship"])
    pk.add_argument("--sea", default="moderate", choices=["calm", "moderate", "rough"])
    pk.add_argument("--slots", type=int, default=1, help="landing slots per deck")
    pk.add_argument("--comms", type=float, default=float("inf"))
    pk.add_argument("--margin", type=float, default=0.0, help="extra CBF separation margin (m)")
    pk.add_argument("--foul-deck", dest="foul_deck", type=int, default=None,
                    help="close this deck mid-recovery (forces re-tasking)")
    pk.add_argument("--foul-time", dest="foul_time", type=float, default=4.0, help="fouled-deck time (s)")
    pk.add_argument("--avoid", action="store_true",
                    help="P3 sense-and-avoid: keep clear of each vessel's superstructure (static CBF keep-outs)")
    pk.add_argument("--engine", default="kinematic", choices=["kinematic", "mujoco"],
                    help="kinematic = fast point-mass (default); mujoco = real X2 physics on K servo decks")
    pk.add_argument("--episodes", type=int, default=5)
    pk.add_argument("--watch", action="store_true", help="visualize one episode (matplotlib / MuJoCo)")
    pk.add_argument("--seed", type=int, default=0)
    pk.set_defaults(func=cmd_multi)

    pm = sub.add_parser("marl", help="train/eval the decentralized MARL collision-avoidance policy")
    pm.add_argument("--eval", default=None, help="evaluate this checkpoint (classical vs MARL) instead of training")
    pm.add_argument("--timesteps", type=int, default=1_000_000)
    pm.add_argument("--drones", type=int, default=14)
    pm.add_argument("--comms", type=float, default=1.0, help="comms range (the hard regime)")
    pm.add_argument("--device", default="auto")
    pm.set_defaults(func=cmd_marl)

    pg = sub.add_parser("gnn", help="A4: train/eval the GNN swarm policy (generalizes across swarm size)")
    pg.add_argument("--eval", default=None, help="evaluate this checkpoint across swarm sizes")
    pg.add_argument("--sizes", type=int, nargs="+", default=[10, 14, 18], help="swarm sizes to eval")
    pg.add_argument("--timesteps", type=int, default=600_000)
    pg.add_argument("--device", default="cpu")
    pg.set_defaults(func=cmd_gnn)

    pa = sub.add_parser("active",
                        help="A6/P5: learned active cooperative perception (attention fusion + VoI gating)")
    pa.add_argument("--eval", default=None, help="evaluate this fusion checkpoint instead of training")
    pa.add_argument("--steps", type=int, default=4000, help="supervised training steps")
    pa.add_argument("--episodes", type=int, default=4000, help="A/B evaluation episodes")
    pa.add_argument("--outlier-p", dest="outlier_p", type=float, default=0.25,
                    help="fraction of confident-outlier fixes in training scenes")
    pa.add_argument("--device", default="cpu")
    pa.set_defaults(func=cmd_active)
    return p


def cmd_multi(args) -> int:
    """A5: M drones recovering onto K moving decks with dynamic re-tasking (optional fouled-deck event)."""
    import numpy as np

    from drone_landing_swarm.multi_deck import MultiDeckConfig, MultiDeckCoordinator
    engine = getattr(args, "engine", "kinematic")
    cfg = MultiDeckConfig(n_drones=args.drones, n_decks=args.decks, scenario=args.scenario,
                          sea=args.sea, n_slots=args.slots, comms_range=args.comms,
                          safety_margin=args.margin,
                          offline_deck=args.foul_deck, offline_time=args.foul_time,
                          obstacles=(OSV_OBSTACLES if getattr(args, "avoid", False) else ()))
    foul = f", deck {args.foul_deck} fouls at {args.foul_time}s" if args.foul_deck is not None else ""
    if getattr(args, "watch", False):
        if engine == "mujoco":
            return _watch_multi_mujoco(cfg, args.seed, foul)
        return _watch_multi(cfg, args.seed, foul)
    print(f"Multi-deck recovery [{engine}]: {cfg.n_drones} drones -> {cfg.n_decks} moving decks, "
          f"{cfg.scenario}{'/' + cfg.sea if cfg.scenario == 'ship' else ''}{foul}; {args.episodes} episodes\n")
    if engine == "mujoco":
        from drone_landing_swarm.multideck_runner import MujocoMultiDeckCoordinator
        coord = MujocoMultiDeckCoordinator(cfg)
    else:
        coord = MultiDeckCoordinator(cfg)
    rows = [coord.run(seed=args.seed + ep) for ep in range(args.episodes)]
    for ep, r in enumerate(rows):
        print(f"  ep {ep:02d}: landed {r['n_landed']}/{r['n_drones']}  time {r['time']:5.1f}s  "
              f"min_sep {r['min_separation']:.2f}  per-deck {r['per_deck_landed']}  "
              f"reassigns {r['n_reassign']}")
    full = sum(r["all_landed"] for r in rows)
    print(f"\nALL-LANDED {full}/{len(rows)} = {100 * full / len(rows):.0f}%   "
          f"mean recovered {100 * np.mean([r['n_landed'] / r['n_drones'] for r in rows]):.0f}%   "
          f"mean reassigns {np.mean([r['n_reassign'] for r in rows]):.1f}")
    clr = [r["min_obstacle_clearance"] for r in rows if r.get("min_obstacle_clearance") is not None]
    if clr:
        obs_ok = sum(r.get("obstacle_ok", True) for r in rows)
        print(f"  sense-and-avoid (P3): worst obstacle clearance {min(clr):+.2f} m; "
              f"kept clear {obs_ok}/{len(rows)}")
    return 0


def _require_file(path: str, what: str, hint: str) -> None:
    """Fail with a clear, actionable message if a required checkpoint is missing."""
    import os
    if not os.path.exists(path):
        raise SystemExit(f"error: {what} not found at {path}\n       {hint}")


def _watch_multi_mujoco(cfg, seed: int, foul: str) -> int:
    """Watch A5 multi-deck recovery on REAL physics: M X2 drones -> K servo decks in the MuJoCo viewer."""
    import time

    import mujoco
    import mujoco.viewer

    from drone_landing_swarm.multideck_runner import MujocoMultiDeckCoordinator
    coord = MujocoMultiDeckCoordinator(cfg)
    coord.reset(seed)
    print(f"Watching MuJoCo multi-deck: {cfg.n_drones} drones -> {cfg.n_decks} decks{foul} "
          f"(drag to orbit, scroll to zoom)")
    with mujoco.viewer.launch_passive(coord.world.model, coord.world.data) as viewer:
        while (viewer.is_running() and coord.t < cfg.max_time
               and len(coord.landed) < cfg.n_drones):
            t0 = time.perf_counter()
            coord.step()
            viewer.sync()
            wait = coord.world.control_dt - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
    per_deck = {k: sum(1 for v in coord.land_deck.values() if v == k) for k in range(cfg.n_decks)}
    print(f"  landed {len(coord.landed)}/{cfg.n_drones}  per-deck {per_deck}  min_sep {coord.min_sep:.2f} m")
    return 0


def _watch_multi(cfg, seed: int, foul: str) -> int:
    """Visualize A5 multi-deck recovery: M drones -> K moving decks, coloured by assignment (matplotlib
    3-D, the kinematic multi-deck coordinator). Decks are squares; drones are dots in their assigned
    deck's colour; landed drones turn green; a fouled deck greys out."""
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed (pip install -e '.[viz]')", file=sys.stderr)
        return 2
    import numpy as np

    from drone_landing_swarm.multi_deck import MultiDeckCoordinator
    coord = MultiDeckCoordinator(cfg)
    coord.reset(seed)
    colors = plt.cm.tab10(np.linspace(0, 1, max(cfg.n_decks, 1)))
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    print(f"Watching multi-deck: {cfg.n_drones} drones -> {cfg.n_decks} decks{foul}  (close window to stop)")

    def frame(_):
        ax.clear()
        s = coord.step()
        decks = s["deck_pos"]
        for k, dp in enumerate(decks):
            avail = coord.available[k]
            ax.scatter(dp[0], dp[1], dp[2], c=[colors[k]], s=240, marker="s",
                       edgecolors="k" if avail else "red", alpha=1.0 if avail else 0.3,
                       label=f"deck {k}" + ("" if avail else " (fouled)"))
        for i, p in s["pos"].items():
            k = s["assign"][i]
            if i in s["landed"]:
                ax.scatter(p[0], p[1], p[2], c="green", s=55, marker="^")
            else:
                ax.scatter(p[0], p[1], p[2], c=[colors[k]], s=40)
                ax.text(p[0], p[1], p[2], str(i), fontsize=7)
        allp = np.array([p for p in s["pos"].values()] + [d for d in decks])
        c0 = allp.mean(axis=0)
        ax.set_xlim(c0[0] - 6, c0[0] + 6); ax.set_ylim(c0[1] - 6, c0[1] + 6); ax.set_zlim(0, 3)
        ax.set_title(f"t={s['t']:.1f}s  landed {len(s['landed'])}/{cfg.n_drones}  "
                     f"re-tasks {coord.n_reassign}  min_sep {s['min_sep']:.2f}m\n"
                     "(square=deck, dot=drone coloured by assigned deck, ^=landed, faded=fouled)")
        ax.legend(loc="upper right", fontsize=7)
        if len(s["landed"]) == cfg.n_drones:
            anim.event_source.stop()

    anim = animation.FuncAnimation(fig, frame, frames=int(cfg.max_time / cfg.dt), interval=40)
    plt.show()
    return 0


def cmd_marl(args) -> int:
    from drone_landing_swarm.marl_train import evaluate, train
    if args.eval:
        _require_file(args.eval, "MARL checkpoint", "pass a .zip saved by `swarm marl` (training)")
        print(evaluate(args.eval, n_drones=args.drones, comms=args.comms))
    else:
        train(timesteps=args.timesteps, device=args.device)
    return 0


def cmd_gnn(args) -> int:
    """A4: train / eval the GNN swarm policy (permutation-invariant, generalizes across swarm size)."""
    from drone_landing_swarm.marl_gnn import evaluate, train
    if args.eval:
        _require_file(args.eval, "GNN checkpoint", "pass a .zip saved by `swarm gnn` (training)")
        import json
        print(json.dumps(evaluate(args.eval, sizes=tuple(args.sizes)), indent=2))
    else:
        train(timesteps=args.timesteps, device=args.device)
    return 0


def cmd_active(args) -> int:
    """A6 / P5: learned active cooperative perception (attention fusion + value-of-information gating)."""
    import os

    from drone_landing_swarm.active_perception import (
        LearnedFusion,
        bandwidth_pareto,
        evaluate,
        train_fusion,
    )
    default_path = os.path.join("runs", "active", "fusion.pt")
    if args.eval:
        _require_file(args.eval, "fusion checkpoint", "train one with `swarm active` (no --eval)")
        learned = LearnedFusion.load(args.eval)
    else:
        print(f"Training attention fusion ({args.steps} steps, outlier_p={args.outlier_p})...")
        learned = train_fusion(steps=args.steps, outlier_p=args.outlier_p, device=args.device)
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        learned.save(default_path)
        print(f"  saved -> {default_path}")

    print("\nLearned fusion vs baselines (mean deck-estimate error):")
    het = evaluate(learned, episodes=args.episodes, outlier_p=0.25)
    hom = evaluate(learned, episodes=args.episodes, outlier_p=0.0)
    print("  regime          equal    inverse-variance    learned")
    print(f"  heterogeneous   {het['equal']:.3f}    {het['inverse_variance']:.3f}              "
          f"{het['learned']:.3f}   <- confident outliers: learned rejects them")
    print(f"  homogeneous     {hom['equal']:.3f}    {hom['inverse_variance']:.3f}              "
          f"{hom['learned']:.3f}   <- Gaussian: learned ~ ties the optimal baseline")
    bp = bandwidth_pareto(episodes=args.episodes)
    print(f"\nValue-of-information gating (fuse-all = {bp['all']:.3f} m):")
    print("  budget   active-select   random-select")
    for b in (1, 2, 3, 4):
        print(f"    {b}        {bp[b]['active']:.3f}          {bp[b]['random']:.3f}")
    print("  active selection reaches near-full accuracy at a fraction of the messages (and beats random).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
