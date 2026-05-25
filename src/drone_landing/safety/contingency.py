"""Contingency / failsafe supervisor (P3.5) — the layer real autopilots have and sims usually skip.

A field drone does not just track a setpoint; it runs a **failsafe state machine** that overrides guidance
when something goes wrong. PX4/ArduPilot ship exactly this (geofence breach, low battery, RC/datalink loss,
land/RTL). We implement the same idea as a small, priority-ordered FSM that wraps any guidance source and
emits an override when a fault is active.

States (highest priority first):

* ``ROTOR_OUT``     — a rotor has failed. A quad is underactuated here (see :doc:`rotor-out-decision`); we
  do **not** attempt a precision landing. The contingency is a *controlled spinning descent*: hold over the
  current spot and ride down at a bounded sink rate (graceful degradation, bounds the impact).
* ``LOW_BATTERY``   — energy below the return reserve -> return-to-launch, then land.
* ``GEOFENCE``      — outside the allowed cylinder -> steer back inside before resuming.
* ``LOST_COMMS``    — datalink stale beyond a timeout -> loiter (single) / RTL.
* ``OBSTACLE_ABORT``— a sensed obstacle is inside the abort radius -> climb-and-hold (break off approach).
* ``NOMINAL``       — pass guidance through unchanged.

The supervisor is **decision-only**: it consumes a :class:`HealthStatus` (battery, link age, position,
nearest-obstacle distance, rotor-ok) and returns a :class:`Contingency` action the caller applies. It reads
estimates, never ground truth (:doc:`no-cheats-realism`)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GeofenceSpec:
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 50.0          # m  allowed horizontal cylinder radius
    z_max: float = 40.0           # m  ceiling


@dataclass(frozen=True)
class ContingencyConfig:
    batt_reserve: float = 0.20    # fraction; below this -> RTL+land
    comms_timeout: float = 2.0    # s   datalink staleness before LOST_COMMS
    obstacle_abort: float = 0.6   # m   nearest-obstacle distance triggering an approach abort
    obstacle_clear: float = 1.2   # m   distance to clear the abort (hysteresis)
    abort_climb_alt: float = 8.0  # m   safe loiter altitude during an abort/RTL
    descent_speed: float = 0.4    # m/s nominal RTL/contingency descent
    rotor_out_sink: float = 0.8   # m/s bounded sink for the spinning-descent contingency
    home: tuple[float, float] = (0.0, 0.0)


@dataclass
class HealthStatus:
    """Onboard health snapshot (all estimated/measured, no ground truth)."""

    pos: np.ndarray                       # world position estimate (3,)
    battery: float = 1.0                  # remaining fraction [0,1]
    comms_age: float = 0.0                # s since last datalink contact
    nearest_obstacle: float = float("inf")  # m to nearest sensed obstacle surface
    rotor_ok: bool = True                 # False once a rotor failure is detected


@dataclass
class Contingency:
    """The supervisor's decision for this step."""

    state: str                            # NOMINAL | ROTOR_OUT | LOW_BATTERY | GEOFENCE | LOST_COMMS | OBSTACLE_ABORT
    override: bool                        # True if the caller must use target/vz instead of nominal guidance
    target_xy: np.ndarray | None = None   # world XY to steer toward (None = hold current XY)
    vz: float = 0.0                       # commanded vertical velocity (+up) when overriding
    reason: str = ""

    def as_dict(self) -> dict:
        return {"state": self.state, "override": self.override,
                "vz": round(self.vz, 3), "reason": self.reason}


class ContingencySupervisor:
    """Priority-ordered failsafe FSM. ``assess(status)`` -> :class:`Contingency` for the current step."""

    def __init__(self, config: ContingencyConfig | None = None, geofence: GeofenceSpec | None = None):
        self.cfg = config or ContingencyConfig()
        self.fence = geofence or GeofenceSpec()
        self.state = "NOMINAL"
        self._aborting = False                # hysteresis latch for OBSTACLE_ABORT
        self.events: list[tuple[float, str]] = []

    def reset(self) -> None:
        self.state = "NOMINAL"
        self._aborting = False
        self.events = []

    def assess(self, status: HealthStatus, t: float = 0.0) -> Contingency:
        c = self.cfg
        p = np.asarray(status.pos, dtype=float)
        home = np.asarray(c.home, dtype=float)

        # ---- priority 1: rotor failure -> controlled spinning descent (graceful degradation, not a landing)
        if not status.rotor_ok:
            return self._enter("ROTOR_OUT", t, Contingency(
                "ROTOR_OUT", override=True, target_xy=p[:2], vz=-c.rotor_out_sink,
                reason="rotor failure: bounded spinning descent over current spot"))

        # ---- priority 2: low battery -> RTL then land
        if status.battery <= c.batt_reserve:
            at_home = float(np.linalg.norm(p[:2] - home)) < 1.0
            return self._enter("LOW_BATTERY", t, Contingency(
                "LOW_BATTERY", override=True, target_xy=home,
                vz=(-c.descent_speed if at_home else 0.0),
                reason=f"battery {status.battery:.0%} <= reserve {c.batt_reserve:.0%}: RTL+land"))

        # ---- priority 3: geofence breach -> steer back inside / descend under ceiling
        d_fence = float(np.linalg.norm(p[:2] - np.asarray(self.fence.center, float)))
        if d_fence > self.fence.radius or p[2] > self.fence.z_max:
            vz = -c.descent_speed if p[2] > self.fence.z_max else 0.0
            return self._enter("GEOFENCE", t, Contingency(
                "GEOFENCE", override=True, target_xy=np.asarray(self.fence.center, float), vz=vz,
                reason="outside geofence: return inside"))

        # ---- priority 4: lost datalink -> loiter in place at a safe altitude
        if status.comms_age > c.comms_timeout:
            vz = self._climb_to(p[2], c.abort_climb_alt)
            return self._enter("LOST_COMMS", t, Contingency(
                "LOST_COMMS", override=True, target_xy=p[:2], vz=vz,
                reason=f"datalink stale {status.comms_age:.1f}s: loiter"))

        # ---- priority 5: obstacle abort -> climb-and-hold, with hysteresis so it doesn't chatter
        if status.nearest_obstacle < c.obstacle_abort:
            self._aborting = True
        elif status.nearest_obstacle > c.obstacle_clear:
            self._aborting = False
        if self._aborting:
            vz = self._climb_to(p[2], c.abort_climb_alt)
            return self._enter("OBSTACLE_ABORT", t, Contingency(
                "OBSTACLE_ABORT", override=True, target_xy=p[:2], vz=vz,
                reason=f"obstacle {status.nearest_obstacle:.2f} m < {c.obstacle_abort} m: break off"))

        # ---- nominal
        if self.state != "NOMINAL":
            self.events.append((round(t, 2), "NOMINAL"))
        self.state = "NOMINAL"
        return Contingency("NOMINAL", override=False, reason="nominal")

    def _climb_to(self, z: float, target_alt: float) -> float:
        c = self.cfg
        if z < target_alt - 0.3:
            return c.descent_speed                     # climb up to the safe altitude
        return 0.0                                     # then hold

    def _enter(self, state: str, t: float, action: Contingency) -> Contingency:
        if self.state != state:
            self.events.append((round(t, 2), state))
        self.state = state
        return action
