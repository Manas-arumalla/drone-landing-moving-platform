"""Printable demonstrations of the Phase 3 safety layer (shared by the CLI and the eval scripts).

Keeping the demo logic here (one source of truth) lets ``drone safety`` and ``scripts/eval_*.py`` show the
same thing without duplicating it."""

from __future__ import annotations

import numpy as np

from drone_landing.safety.avoid import AvoidConfig, HOCBFAvoider, cluster_returns
from drone_landing.safety.contingency import (
    ContingencyConfig,
    ContingencySupervisor,
    GeofenceSpec,
    HealthStatus,
)
from drone_landing.safety.obstacles import ObstacleField, RangeSensor, RangeSensorConfig


def _run_transit(avoid: bool, field: ObstacleField, cfg: AvoidConfig, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sensor = RangeSensor(field, RangeSensorConfig(max_range=5.0, sigma=0.02, dropout=0.05, n_beams=48))
    av = HOCBFAvoider(cfg)
    p = np.array([4.0, 1.2]); v = np.zeros(2)     # offset approach: a clear side exists (deck sector)
    goal = np.array([0.0, 0.0])
    z = 0.7                                        # level transit altitude (within the wheelhouse span)
    dt = 0.02
    min_clear = float("inf")
    collided = reached = False
    for _ in range(2500):
        a_des = 2.2 * (goal - p) - 2.4 * v
        n = float(np.linalg.norm(a_des))
        if n > cfg.a_max:
            a_des *= cfg.a_max / n
        if avoid:
            hits = cluster_returns(sensor.scan(p, z, rng), tol=0.4)     # onboard sensing (noisy surfaces)
            a = av.filter(p, v, a_des, hits, obstacle_radius=0.0)
        else:
            a = a_des
        v = v + a * dt
        p = p + v * dt
        d = field.nearest_surface_distance(p, z)
        min_clear = min(min_clear, d)
        collided = collided or d < 0.0
        if float(np.linalg.norm(p - goal)) < 0.4:
            reached = True
            break
    return {"reached": reached, "collided": collided, "min_clearance": round(min_clear, 3),
            "activations": av.report.activations}


def avoid_demo() -> None:
    """Closed-loop sense-and-avoid past the OSV superstructure (no-avoid vs higher-order CBF)."""
    field = ObstacleField.offshore_osv(deck_xy=(0.0, 0.0), deck_z=0.0)
    cfg = AvoidConfig(a_max=4.0, drone_radius=0.25, margin=0.15, latency=0.05)
    print("Sense-and-avoid past the OSV superstructure (wheelhouse + bow + mast), level transit @0.7 m:\n")
    print(f"  obstacles: {', '.join(o.name for o in field.obstacles)}")
    print(f"  keep-out = obstacle + drone_radius {cfg.drone_radius} m + margin {cfg.margin} m,"
          f" latency look-ahead {cfg.latency}s\n")
    print("  mode          reached pad   collided   min clearance to surface")
    for avoid in (False, True):
        r = _run_transit(avoid, field, cfg)
        tag = "AVOID (CBF)" if avoid else "no-avoid   "
        extra = f"   ({r['activations']} filter activations)" if avoid else ""
        print(f"   {tag}    {str(r['reached']):5s}        {str(r['collided']):5s}      "
              f"{r['min_clearance']:+.2f} m{extra}")
    print("\n  no-avoid flies into the superstructure (negative clearance = inside a structure);")
    print("  the CBF keeps clearance >= 0 throughout and still reaches the pad.")


def contingency_demo() -> None:
    """Priority-ordered failsafe FSM over a scripted mission timeline with injected faults."""
    sup = ContingencySupervisor(
        ContingencyConfig(batt_reserve=0.2, comms_timeout=2.0, obstacle_abort=0.6, home=(0.0, 0.0)),
        GeofenceSpec(center=(0.0, 0.0), radius=10.0, z_max=15.0))
    timeline = [
        (0.0,  [1.0, 0.0, 5.0], 0.95, 0.0, 9.9, True),
        (2.0,  [3.0, 1.0, 5.0], 0.85, 0.0, 9.9, True),
        (4.0,  [12.0, 1.0, 5.0], 0.80, 0.0, 9.9, True),    # outside the 10 m geofence
        (6.0,  [6.0, 0.0, 5.0], 0.70, 3.5, 9.9, True),     # datalink stale -> lost comms
        (8.0,  [2.0, 0.0, 4.0], 0.60, 0.0, 0.45, True),    # obstacle ahead -> abort
        (10.0, [1.0, 0.0, 8.0], 0.55, 0.0, 2.0, True),     # cleared -> nominal
        (12.0, [1.0, 0.0, 6.0], 0.15, 0.0, 9.9, True),     # low battery -> RTL+land
        (14.0, [0.2, 0.0, 3.0], 0.12, 0.0, 9.9, False),    # rotor failure -> spinning descent
    ]
    print("Contingency / failsafe FSM (geofence 10 m, batt reserve 20%, comms timeout 2 s):\n")
    print("   t     batt  comms  obst    rotor   ->  STATE            action")
    for t, pos, batt, comms, obst, rotor in timeline:
        c = sup.assess(HealthStatus(pos=np.array(pos, float), battery=batt, comms_age=comms,
                                    nearest_obstacle=obst, rotor_ok=rotor), t)
        act = c.reason if c.override else "pass guidance through"
        print(f"  {t:4.0f}s   {batt:4.0%}  {comms:4.1f}s  {obst:4.1f}m  {'OK' if rotor else 'FAIL':4s}"
              f"   ->  {c.state:14s}   {act}")
    print("\n  Transitions:", " -> ".join(s for _, s in sup.events) or "(none)")
    print("  Priority: ROTOR_OUT > LOW_BATTERY > GEOFENCE > LOST_COMMS > OBSTACLE_ABORT > NOMINAL")
