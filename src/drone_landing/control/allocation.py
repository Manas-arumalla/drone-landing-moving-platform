"""Control allocation (mixer): map a desired collective thrust + body torque to motor thrusts.

For a quad-X each rotor produces an up thrust at its arm position and a reaction (yaw) torque, so the
map from motor thrusts ``f`` to the wrench ``[T, tau_x, tau_y, tau_z]`` is

    T     = sum f_i
    tau_x = sum  y_i f_i
    tau_y = sum -x_i f_i
    tau_z = sum  spin_i * k_q * f_i

This is built once from the geometry and inverted. The X2 spin pattern is the corrected quad-X
(diagonal rotors share a spin) so the matrix is full rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlAllocator:
    A_inv: np.ndarray   # (4,4) maps [T, tau_x, tau_y, tau_z] -> motor thrusts
    thrust_min: float
    thrust_max: float
    A: np.ndarray | None = None   # (4,4) forward map (for fault-tolerant reallocation)

    def allocate(self, thrust: float, torque: np.ndarray, failed: int | None = None) -> np.ndarray:
        if failed is None:
            wrench = np.array([thrust, torque[0], torque[1], torque[2]])
            return np.clip(self.A_inv @ wrench, self.thrust_min, self.thrust_max)
        # Fault-tolerant (rotor-out): with one rotor dead, a quad loses an actuator, so yaw torque is
        # no longer independently controllable. The standard strategy is to **sacrifice yaw** and use
        # the three working rotors to hold collective thrust + roll + pitch (the attitude that aims the
        # thrust vector for position control); the airframe spins slowly about yaw. We solve the 3x3
        # system [T, tau_x, tau_y] = B f3 for the three working-rotor thrusts.
        working = [i for i in range(4) if i != failed]
        B = self.A[:3][:, working]                      # rows [T, tau_x, tau_y], working-rotor columns
        f3 = np.linalg.solve(B, np.array([thrust, torque[0], torque[1]]))
        f = np.zeros(4)
        f[working] = f3
        return np.clip(f, self.thrust_min, self.thrust_max)


def build_allocator(motor_xy: np.ndarray, spin_signs: np.ndarray, torque_coeff: float,
                    thrust_min: float, thrust_max: float) -> ControlAllocator:
    motor_xy = np.asarray(motor_xy, dtype=float)
    A = np.vstack([
        np.ones(4),                          # T
        motor_xy[:, 1],                      # tau_x = sum y_i f_i
        -motor_xy[:, 0],                     # tau_y = sum -x_i f_i
        np.asarray(spin_signs) * torque_coeff,  # tau_z
    ])
    return ControlAllocator(np.linalg.inv(A), thrust_min, thrust_max, A=A)


def x2_allocator() -> ControlAllocator:
    """Allocator matching the Skydio X2 landing world (corrected quad-X spin pattern)."""
    motor_xy = np.array([[-.14, -.18], [-.14, .18], [.14, .18], [.14, -.18]])
    spin_signs = np.array([1.0, -1.0, 1.0, -1.0])  # diagonal rotors share a spin
    return build_allocator(motor_xy, spin_signs, torque_coeff=0.0201, thrust_min=0.0, thrust_max=13.0)
