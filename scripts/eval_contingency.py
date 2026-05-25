"""P3.5 — contingency / failsafe state machine over a mission timeline.

Injects the faults a field autopilot must survive — a geofence excursion, a datalink dropout, a low-battery
event, an obstacle that forces an approach abort, and a rotor failure — and prints the failsafe state at
each event so you can see the priority-ordered FSM take over guidance.

  python scripts/eval_contingency.py
"""

from __future__ import annotations

from drone_landing.safety.demo import contingency_demo

if __name__ == "__main__":
    contingency_demo()
