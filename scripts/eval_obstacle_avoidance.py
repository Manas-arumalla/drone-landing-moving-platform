"""P3.3/P3.4 — closed-loop sense-and-avoid past the OSV superstructure.

A drone transits toward the helideck on a line that runs straight through the vessel's wheelhouse/bow. With
only a setpoint tracker it flies into the superstructure; with the onboard range sensor + higher-order CBF
it senses the surface returns and bends its path around them — never penetrating the keep-out — then still
reaches the pad. No ground truth: avoidance runs on noisy scan returns.

  python scripts/eval_obstacle_avoidance.py
"""

from __future__ import annotations

from drone_landing.safety.demo import avoid_demo

if __name__ == "__main__":
    avoid_demo()
