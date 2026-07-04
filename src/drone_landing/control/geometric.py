"""Geometric SE(3) tracking controller for the quadrotor.

A position outer loop turns the platform-relative state into a desired thrust vector; a geometric
attitude inner loop (Lee et al.) tracks the corresponding orientation and produces a body torque;
the allocator maps collective thrust + torque to motor thrusts. Consumes only the relative state +
attitude/rate, so it runs identically on ground truth or on the EKF estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.control.allocation import ControlAllocator, x2_allocator

GRAVITY = 9.81


def vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


@dataclass(frozen=True)
class GeometricGains:
    kp_xy: float = 2.6
    kd_xy: float = 3.0
    ki_xy: float = 1.4         # integral action -> zero steady-state lag tracking a moving platform
    int_limit: float = 1.2     # m  anti-windup clamp on the integral state
    int_leak: float = 0.2      # 1/s  gentle integral leak -> bleeds stale bias when conditions change
    a_xy_max: float = 4.0      # m/s^2  caps commanded horizontal accel (-> tilt limit)
    kp_vz: float = 4.5
    kpR: float = 130.0         # attitude proportional (angular accel per rad)
    kdR: float = 22.0          # attitude derivative (per rad/s)
    # reduced-attitude (rotor-out) gains: track only the thrust-axis direction, yaw left free
    kpR_ro: float = 140.0      # fast thrust-axis tracking (must outpace the yaw spin)
    kdR_ro: float = 24.0
    kp_xy_ro: float = 1.1      # gentle horizontal gains while spinning (avoid over-tilting -> drift)
    kd_xy_ro: float = 2.0
    a_xy_max_ro: float = 1.2   # cap the thrust-axis tilt so the spinning frame holds position
    press_collective: float = 0.85  # fraction of weight at touchdown: gentle settle, no bounce


class GeometricController:
    def __init__(self, mass: float, inertia: np.ndarray, control_dt: float = 0.01,
                 allocator: ControlAllocator | None = None,
                 gains: GeometricGains | None = None):
        self.m = float(mass)
        self.J = np.asarray(inertia, dtype=float)
        self.dt = control_dt
        self.alloc = allocator or x2_allocator()
        self.g = gains or GeometricGains()
        self._int_xy = np.zeros(2)
        self.failed_rotor: int | None = None   # set to a rotor index for fault-tolerant allocation

    def reset(self) -> None:
        self._int_xy = np.zeros(2)

    def compute(self, rel_pos: np.ndarray, rel_vel: np.ndarray, R: np.ndarray, omega: np.ndarray,
                vz_des: float = 0.0, press: bool = False, hold_level: bool = False,
                velocity_hold: bool = False, a_xy_override: np.ndarray | None = None,
                dist_ff: np.ndarray | None = None,
                press_normal: np.ndarray | None = None) -> np.ndarray:
        """Return four motor thrusts.

        ``rel_pos`` / ``rel_vel`` are platform-minus-drone (world); ``R`` is body->world attitude;
        ``omega`` is the body angular rate; ``vz_des`` is the commanded drone vertical velocity.
        ``press`` engages weight-on-gear touchdown (sub-weight collective, still tracking laterally).
        ``hold_level`` flies straight up/down at level attitude, ignoring the (untrusted) horizontal
        estimate — used to climb and re-acquire when vision is lost mid-flight.
        ``press_normal`` (optional, unit world vector): attitude-matched touchdown on a tilted deck —
        the flatness insight that touchdown attitude is set by terminal acceleration. During the
        commit velocity-hold it adds the acceleration g·n_xy/n_z that tilts body-z toward the deck
        normal, and during the press it pushes along the normal instead of straight-down-level, so
        the feet meet the tilted surface together. ``None`` (every existing caller) is unchanged.
        """
        g = self.g
        if hold_level:
            drone_vz = -rel_vel[2]
            f_des = self.m * np.array([0.0, 0.0, g.kp_vz * (vz_des - drone_vz) + GRAVITY])
            return self._attitude_thrust(f_des, R, omega, level=True)  # integral frozen

        e_xy = -rel_pos[:2]            # drone - platform
        ev_xy = -rel_vel[:2]
        if a_xy_override is not None:
            # horizontal acceleration supplied by an outer controller (e.g. MPC predictive tracker)
            a_xy = np.asarray(a_xy_override, dtype=float)
        elif velocity_hold:
            # final descent: only match the platform velocity (no position chase). Keeps the relative
            # position frozen at the commit offset even when the marker has left the FOV, instead of
            # chasing a coasting/biased position estimate off the deck.
            a_xy = -g.kd_xy * ev_xy
            if press_normal is not None and -float(rel_pos[2]) <= 0.15:
                # attitude-matched touchdown: pre-tilt body-z toward the deck normal (terminal accel
                # a = g·n_xy/n_z aligns the thrust vector with the normal) — but ONLY over the last
                # ~15 cm. The tilt acceleration is a real horizontal force (g·tan 12° ≈ 2.1 m/s²);
                # holding it through the whole commit descent accelerates the vehicle down-slope and
                # carries it off the deck before contact, so the pre-tilt is a last-instant maneuver.
                nz = max(float(press_normal[2]), 0.9)          # cap the tilt authority (~25 deg)
                a_xy = a_xy + GRAVITY * np.asarray(press_normal[:2], dtype=float) / nz
        else:
            # integral of position error eliminates the PD lag when tracking an accelerating platform.
            # Leak bleeds a stale bias once the platform's acceleration changes; conditional integration
            # (below) is the anti-windup: we only accept the new integral if it does not *increase* an
            # already-saturated command, so the integrator never winds up against the accel/tilt limit.
            self._int_xy *= max(0.0, 1.0 - g.int_leak * self.dt)
            pd = -g.kp_xy * e_xy - g.kd_xy * ev_xy
            candidate = np.clip(self._int_xy + e_xy * self.dt, -g.int_limit, g.int_limit)
            a_prev = pd - g.ki_xy * self._int_xy
            a_cand = pd - g.ki_xy * candidate
            saturated = np.linalg.norm(a_prev) > g.a_xy_max
            if (not saturated) or (np.linalg.norm(a_cand) <= np.linalg.norm(a_prev)):
                self._int_xy = candidate
            a_xy = pd - g.ki_xy * self._int_xy
        if dist_ff is not None:                      # DOB feedforward: cancel the estimated wind accel
            a_xy = a_xy - np.asarray(dist_ff)[:2]
        n = np.linalg.norm(a_xy)
        if n > g.a_xy_max:
            a_xy *= g.a_xy_max / n

        if press:
            # weight-on-gear: push straight down at sub-weight, LEVEL. Near the deck the close-range
            # vision is noisy; tracking it tilts the drone and lands it on one foot (bounce). Level
            # press contacts all four feet evenly; the drone coasts horizontally with the platform.
            # With ``press_normal``: push along the DECK NORMAL at the same sub-weight collective and
            # let the attitude loop align body-z with it — all feet meet a *tilted* surface together
            # (a level press on a slope catches one edge and never seats: the inclined-deck failure).
            if press_normal is not None:
                n_hat = np.asarray(press_normal, dtype=float)
                n_hat = n_hat / max(float(np.linalg.norm(n_hat)), 1e-9)
                f_des = self.m * g.press_collective * GRAVITY * n_hat
                level = False
            else:
                f_des = self.m * np.array([0.0, 0.0, g.press_collective * GRAVITY])
                level = True
        else:
            drone_vz = -rel_vel[2]     # deck vertical velocity ~ 0 for ground
            a_z = g.kp_vz * (vz_des - drone_vz)
            f_des = self.m * np.array([a_xy[0], a_xy[1], a_z + GRAVITY])
            level = False
        return self._attitude_thrust(f_des, R, omega, level)

    def compute_rotor_out(self, rel_pos: np.ndarray, rel_vel: np.ndarray, R: np.ndarray,
                          omega: np.ndarray, vz_des: float = 0.0) -> np.ndarray:
        """Reduced-attitude control for a failed rotor (Mueller & D'Andrea 2014).

        A quad that loses a rotor is underactuated and cannot hold yaw — so it **spins** about its
        vertical and we control only the *direction of the primary thrust axis* (body-z) to point the
        net thrust where position control needs it. The 3-rotor allocator provides collective thrust +
        roll/pitch torque (yaw deliberately uncontrolled). Gains are gentler than nominal to avoid
        over-tilting the spinning frame. Requires ``self.failed_rotor`` set.
        """
        g = self.g
        # position outer loop -> desired thrust vector in the world frame (gentler gains while spinning)
        e_xy = -rel_pos[:2]
        ev_xy = -rel_vel[:2]
        a_xy = -g.kp_xy_ro * e_xy - g.kd_xy_ro * ev_xy
        n_a = float(np.linalg.norm(a_xy))
        if n_a > g.a_xy_max_ro:
            a_xy *= g.a_xy_max_ro / n_a
        drone_vz = -rel_vel[2]
        a_z = g.kp_vz * (vz_des - drone_vz)
        f_des = self.m * np.array([a_xy[0], a_xy[1], a_z + GRAVITY])
        fmag = float(np.linalg.norm(f_des))
        if fmag < 1e-6:
            return self.alloc.allocate(0.0, np.zeros(3), failed=self.failed_rotor)
        n_des = f_des / fmag                      # desired thrust-axis direction (world)
        n = R[:, 2]                               # current body-z (world)
        # reduced-attitude error: world-frame rotation axis taking n -> n_des, expressed in body frame
        e_body = R.T @ np.cross(n, n_des)
        # roll/pitch torque to align the thrust axis; yaw torque = 0 (spin is free)
        torque = np.array([
            self.J[0] * (g.kpR_ro * e_body[0] - g.kdR_ro * omega[0]),
            self.J[1] * (g.kpR_ro * e_body[1] - g.kdR_ro * omega[1]),
            0.0,
        ])
        thrust = max(0.0, float(f_des @ n))       # thrust along the current body-z
        return self.alloc.allocate(thrust, torque, failed=self.failed_rotor)

    def _attitude_thrust(self, f_des: np.ndarray, R: np.ndarray, omega: np.ndarray,
                         level: bool) -> np.ndarray:
        zb = R[:, 2]
        thrust = max(0.0, float(f_des @ zb))
        if level or np.linalg.norm(f_des) < 1e-6:
            R_des = np.eye(3)
        else:
            zb_des = f_des / np.linalg.norm(f_des)
            yb = np.cross(zb_des, np.array([1.0, 0.0, 0.0]))  # desired yaw = 0
            yb /= np.linalg.norm(yb)
            xb = np.cross(yb, zb_des)
            R_des = np.column_stack([xb, yb, zb_des])
        eR = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        torque = self.J * (-self.g.kpR * eR - self.g.kdR * omega) + np.cross(omega, self.J * omega)
        return self.alloc.allocate(thrust, torque, failed=self.failed_rotor)
