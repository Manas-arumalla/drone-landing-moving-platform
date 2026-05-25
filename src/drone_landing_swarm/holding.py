"""Holding-stack management for drones waiting their turn to land.

Drones not cleared to land must wait somewhere safe and deconflicted while the moving deck travels.
We arrange a set of **holding slots** on a ring around the deck, staggered into **altitude layers**
(like a real holding stack / carrier marshalling pattern). The slots are defined *relative to the
deck*, so the whole stack rides with the ship. Waiting drones are assigned to slots by the **Hungarian
algorithm** (globally minimal total repositioning), and a simple guidance law tracks the assigned slot
(with the deck's velocity fed forward so the drone holds station over the moving deck). The CBF safety
filter still runs on top, so the holding pattern is collision-free by construction *and* by reaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing_swarm.scheduler import optimal_assignment


@dataclass(frozen=True)
class HoldingConfig:
    radius: float = 2.0       # m   ring radius of the holding pattern around the deck centre
    base_alt: float = 1.2     # m   altitude of the lowest holding layer above the deck
    layer_gap: float = 0.7    # m   vertical spacing between stacked layers
    per_layer: int = 6        # slots per ring layer before stacking a new layer
    kp: float = 1.2           # position-tracking gain -> commanded velocity


class HoldingStack:
    """Generates deck-relative holding slots and assigns waiting drones to them optimally."""

    def __init__(self, config: HoldingConfig | None = None):
        self.cfg = config or HoldingConfig()

    def slot_offset(self, slot: int) -> np.ndarray:
        """Deck-relative (x, y, z) offset of holding ``slot`` (z is above the deck)."""
        c = self.cfg
        layer, idx = divmod(slot, c.per_layer)
        ang = 2.0 * np.pi * idx / c.per_layer + 0.5 * layer  # rotate each layer to avoid stacking
        return np.array([c.radius * np.cos(ang), c.radius * np.sin(ang),
                         c.base_alt + layer * c.layer_gap])

    def assign(self, drone_ids: list[int], positions: dict[int, np.ndarray],
               deck_pos: np.ndarray) -> dict[int, int]:
        """Assign each holding drone to a distinct slot, minimizing total repositioning (Hungarian)."""
        if not drone_ids:
            return {}
        n = len(drone_ids)
        slots = list(range(n))
        slot_world = [deck_pos + self.slot_offset(s) for s in slots]
        cost = np.array([[float(np.linalg.norm(positions[d] - slot_world[s])) for s in slots]
                         for d in drone_ids])
        pairs = optimal_assignment(cost)
        return {drone_ids[r]: slots[c] for r, c in pairs}

    def hold_velocity(self, p_self: np.ndarray, slot: int, deck_pos: np.ndarray,
                      deck_vel: np.ndarray, v_max: float) -> np.ndarray:
        """Velocity command to hold ``slot`` over the moving deck (track slot + deck-velocity feedforward)."""
        target = deck_pos + self.slot_offset(slot)
        v = self.cfg.kp * (target - p_self) + deck_vel
        s = float(np.linalg.norm(v))
        return v * (v_max / s) if s > v_max else v
