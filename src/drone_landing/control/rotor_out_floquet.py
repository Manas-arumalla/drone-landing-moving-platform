"""Averaged / precession control for rotor-out flight — steer the SPIN-AVERAGED thrust axis.

The single-rotor-out equilibrium is a **limit cycle**: the vehicle spins fast (~15-20 rad/s here) about a
near-vertical axis, with body-z coning around it. Position is driven by the **spin-averaged** thrust
direction (≈ the spin axis), *not* the instantaneous body-z — which is why a fixed-point attitude LQR fails
(it fights the cone). This controller works on the averaged axis directly:

1. **Estimate the spin axis** ``a`` by low-pass-filtering body-z over a few spin periods.
2. **Position loop** -> a desired (tilt-limited, near-vertical) averaged axis ``a_des``.
3. **Precession steering.** The angular momentum is ``L = J_z·Ω`` along ``a``. Since ``dL/dt = τ``, an
   inertial torque perpendicular to ``a`` precesses the axis: ``ȧ = τ_perp/|L|``. So to drive ``a -> a_des``
   we command the **inertial** torque ``τ_world = |L|·(k_a·(a_des−a)_perp − k_da·ȧ)`` and realize it with a
   **phase-mapped body torque** ``τ_body = Rᵀ·τ_world`` (roll/pitch only; yaw free). Because
   ``R·(Rᵀτ_world) = τ_world``, the spinning body produces exactly the intended inertial torque every
   instant — the step the instantaneous LQR/PD missed.
4. **PD fallback during spin-up** (``|Ω|`` below threshold, before the dead rotor has spun it up) blends in
   an instantaneous reduced-attitude law so it doesn't tumble before the averaging is valid.

Honest scope: a sim artifact is the **unbounded spin** (no rotor aero-drag), which slowly erodes precession
authority; and **wind** remains the hardest stressor. See the rotor-out memory."""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class RotorOutFloquet:
    """Spin-averaged precession controller for a single failed rotor (steer the averaged thrust axis)."""

    def __init__(self, mass: float, inertia: np.ndarray, allocator, failed_rotor: int,
                 control_dt: float = 0.01, tau_a: float = 0.5, k_a: float = 7.0, k_da: float = 1.0,
                 spin_lo: float = 3.0, spin_hi: float = 8.0, kp_xy: float = 1.2, kd_xy: float = 2.0,
                 tilt_max: float = 0.22, kp_vz: float = 4.0, vz_min: float = -0.35, ki_xy: float = 0.0,
                 commit_radius: float = 0.35, kpR_pd: float = 130.0, kdR_pd: float = 22.0):
        self.m, self.J, self.alloc, self.failed = float(mass), np.asarray(inertia, float), allocator, int(failed_rotor)
        self.dt = float(control_dt)
        self.tau_a, self.k_a, self.k_da = tau_a, k_a, k_da
        self.spin_lo, self.spin_hi = spin_lo, spin_hi
        self.kp_xy, self.kd_xy, self.ki_xy, self.tilt_max = kp_xy, kd_xy, ki_xy, tilt_max
        self.kp_vz, self.vz_min, self.commit_radius = kp_vz, vz_min, commit_radius
        self.kpR_pd, self.kdR_pd = kpR_pd, kdR_pd
        self.reset()

    def reset(self) -> None:
        self.a = np.array([0.0, 0.0, 1.0])      # averaged thrust-axis estimate (starts vertical)
        self.a_prev = self.a.copy()
        self._int_xy = np.zeros(2)              # integral of horizontal error -> reject steady wind bias

    def control(self, rel_pos: np.ndarray, rel_vel: np.ndarray, R: np.ndarray, omega: np.ndarray,
                vz_des: float = 0.0, hold_xy: bool = False, flow_vel: np.ndarray | None = None) -> np.ndarray:
        """``hold_xy`` (vision tracking lost near touchdown): stop chasing the diverging POSITION estimate.
        If a downward optical-flow measurement ``flow_vel`` (drone velocity rel. to the deck) is available it
        **damps that measured velocity** to arrest the blind-descent drift — this is the close-range sensor
        that works inside the <0.3 m ArUco blind-zone. Without flow, fall back to holding the axis vertical."""
        n = R[:, 2]                                          # instantaneous body-z (world)
        # 1. averaged thrust axis (low-pass over a few spin periods)
        alpha = min(1.0, self.dt / self.tau_a)
        self.a = _norm(self.a + alpha * (n - self.a))
        adot = (self.a - self.a_prev) / self.dt
        self.a_prev = self.a.copy()

        # 2. position loop -> desired averaged axis (tilt-limited, near vertical). A leaky, clamped integral
        # of the horizontal error biases the tilt to counter a steady wind force (the dead-rotor vehicle has
        # little authority, so the integral does the standing-offset rejection the proportional term can't).
        if hold_xy:
            # vision lost near touchdown: don't chase the diverging POSITION estimate. Damp the horizontal
            # velocity to arrest the blind-descent drift — preferring the optical-flow MEASUREMENT (which
            # works in the close-range blind-zone) over the EKF velocity (which diverges after marker loss).
            # flow_vel is the drone's velocity relative to the deck, i.e. the quantity to drive to zero.
            v_meas = np.asarray(flow_vel, float)[:2] if flow_vel is not None else -rel_vel[:2]
            a_cmd_xy = np.clip(-self.kd_xy * v_meas / GRAVITY, -self.tilt_max, self.tilt_max)
            a_des = _norm(np.array([a_cmd_xy[0], a_cmd_xy[1], 1.0]))
        else:
            e_xy, ev_xy = -rel_pos[:2], -rel_vel[:2]
            self._int_xy = np.clip(0.985 * self._int_xy + e_xy * self.dt, -0.8, 0.8)
            a_cmd_xy = np.clip((-self.kp_xy * e_xy - self.kd_xy * ev_xy - self.ki_xy * self._int_xy) / GRAVITY,
                               -self.tilt_max, self.tilt_max)
            a_des = _norm(np.array([a_cmd_xy[0], a_cmd_xy[1], 1.0]))

        Omega = float(omega[2])
        # 3. precession steering torque (inertial), realized by a phase-mapped body torque.
        # The angular momentum along the axis is L = J_z·Ω·a with Ω **signed** — here Ω < 0, so L points
        # OPPOSITE a. The precession ȧ = τ_perp/(J_z·Ω) therefore needs the SIGNED J_z·Ω (using |·| drove
        # the axis the wrong way and tumbled). Floor the magnitude to avoid div-by-zero at low spin.
        M = self.J[2] * Omega
        if abs(M) < 0.05:
            M = 0.05 if M >= 0 else -0.05
        a_err = a_des - float(self.a @ a_des) * self.a    # component of a_des perpendicular to a
        tau_world = M * (self.k_a * a_err - self.k_da * adot)
        tau_avg = R.T @ tau_world                          # phase map: R (Rᵀ τ_world) = τ_world

        # 4. PD fallback (instantaneous reduced attitude) — dominant during spin-up, faded out once spinning
        e_body = R.T @ np.cross(n, a_des)
        tau_pd = self.J * (self.kpR_pd * e_body - self.kdR_pd * omega)
        w = float(np.clip((abs(Omega) - self.spin_lo) / (self.spin_hi - self.spin_lo), 0.0, 1.0))
        tx = w * tau_avg[0] + (1 - w) * tau_pd[0]
        ty = w * tau_avg[1] + (1 - w) * tau_pd[1]

        # collective along body-z (descent-capped); zero past 90 deg. **Center-then-descend:** only sink
        # once the averaged axis has brought us over the pad (horizontal error within commit_radius);
        # otherwise hover and keep centring — so the slow precession steering finishes before touchdown.
        drone_vz = -rel_vel[2]
        centered = hold_xy or float(np.linalg.norm(rel_pos[:2])) < self.commit_radius
        vz_cmd = max(vz_des, self.vz_min) if centered else 0.0
        a_z = self.kp_vz * (vz_cmd - drone_vz)
        thrust = max(0.0, self.m * (a_z + GRAVITY) * float(n[2]))
        return self.alloc.allocate(thrust, np.array([tx, ty, 0.0]), failed=self.failed)
