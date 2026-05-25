"""Mueller-style reduced-attitude **LQR** for rotor-out flight — a RESEARCH ARTIFACT (does not yet land).

⚠️ HONEST STATUS (2026-05-25): this controller **underperforms the existing reduced-attitude PD**
(``GeometricController.compute_rotor_out``) and is **NOT wired into the live system** — the production
rotor-out handling remains the Option-D bounded spinning-descent contingency. Kept as a documented,
tested attempt at the textbook fix, with the negative finding recorded so the next iteration starts here.

The idea (Mueller & D'Andrea 2014): a quad that loses a rotor is underactuated — it spins about its primary
(thrust) axis and can only control that axis's **direction**. We linearize the *reduced-attitude* dynamics
— the inertial tilt of the thrust axis ``n = (n_x, n_y)`` and the body roll/pitch rates ``(p, q)`` — about
the spin, keeping the **gyroscopic coupling** (parametrized by the measured spin rate ``r``):

    ṅ_x = q − r·n_y
    ṅ_y = −p + r·n_x
    ṗ   = τ_x/Jx − ((Jz−Jy)/Jx)·r·q
    q̇   = τ_y/Jy − ((Jx−Jz)/Jy)·r·p

and solve an LQR ``τ = −K(r)·[n−n_des; p; q]`` (Riccati, recomputed per spin rate, cached).

**Why it doesn't (yet) work — the decisive finding.** With heavy rate-damping it keeps the tilt *magnitude*
bounded (~26°, comparable to the PD), but it **drifts away (tens of metres) and climbs**: regulating the
*instantaneous* inertial tilt to a fixed point does **not** regulate the **period-averaged thrust
direction**, because the single-rotor-out equilibrium is a **limit cycle (a spin), not a fixed point**. A
correct treatment needs **averaged / Floquet control around the periodic orbit** (control the spin-averaged
thrust vector), plus a descent that holds enough thrust for authority and wind robustness — a substantially
deeper build. The simpler PD's body-frame ``cross(n, n_des)`` law happens to regulate the averaged direction
far better (0.5 m drift in a wind-off hover), which is why it remains the basis of the contingency."""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81


class RotorOutLQR:
    """Reduced-attitude LQR controller for a single failed rotor (spinning-equilibrium descent)."""

    def __init__(self, mass: float, inertia: np.ndarray, allocator, failed_rotor: int,
                 kp_xy: float = 0.7, kd_xy: float = 1.4, tilt_max: float = 0.18,
                 kp_vz: float = 4.0, vz_min: float = -0.6):
        self.m = float(mass)
        self.J = np.asarray(inertia, dtype=float)
        self.alloc = allocator
        self.failed = int(failed_rotor)
        self.kp_xy, self.kd_xy, self.tilt_max = kp_xy, kd_xy, tilt_max
        self.kp_vz, self.vz_min = kp_vz, vz_min
        # LQR weights: regulate the thrust-axis tilt but *damp the body rates strongly* (the spinning
        # equilibrium is a limit cycle — over-penalising the instantaneous tilt fights the natural cone and
        # diverges; heavy rate damping keeps the cone bounded, like the PD that holds an 18 deg hover).
        self.Q = np.diag([18.0, 18.0, 4.0, 4.0])
        self.R = np.diag([0.4, 0.4])
        self._cache: dict[int, np.ndarray] = {}

    def _gain(self, r: float) -> np.ndarray:
        """LQR gain K for spin rate r (cached on r rounded to 0.5 rad/s — avoids a Riccati solve each step)."""
        key = int(round(r * 2.0))
        K = self._cache.get(key)
        if K is None:
            from scipy.linalg import solve_continuous_are
            rr = key / 2.0
            Jx, Jy, Jz = self.J
            A = np.array([
                [0.0, -rr, 0.0, 1.0],
                [rr, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -(Jz - Jy) / Jx * rr],
                [0.0, 0.0, -(Jx - Jz) / Jy * rr, 0.0],
            ])
            B = np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / Jx, 0.0], [0.0, 1.0 / Jy]])
            P = solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.solve(self.R, B.T @ P)
            self._cache[key] = K
        return K

    def control(self, rel_pos: np.ndarray, rel_vel: np.ndarray, R: np.ndarray, omega: np.ndarray,
                vz_des: float = 0.0) -> np.ndarray:
        """Return four motor thrusts. ``rel_pos``/``rel_vel`` = platform-minus-drone (world)."""
        # ---- outer position loop -> desired thrust-axis tilt n_des (gentle, authority-limited)
        e_xy, ev_xy = -rel_pos[:2], -rel_vel[:2]
        a_xy = -self.kp_xy * e_xy - self.kd_xy * ev_xy          # desired horizontal accel
        n_des = np.clip(a_xy / GRAVITY, -self.tilt_max, self.tilt_max)   # small-tilt: n_xy ~ a_xy/g

        # ---- reduced-attitude state: thrust-axis tilt (inertial) + body roll/pitch rates
        n = R[:, 2]                                            # body-z in world
        r = float(omega[2])                                   # measured spin rate (yaw)
        x = np.array([n[0] - n_des[0], n[1] - n_des[1], float(omega[0]), float(omega[1])])
        tau_xy = -self._gain(r) @ x                            # LQR torque (roll, pitch); yaw free

        # ---- collective thrust for altitude (along the current body-z), descent-capped
        drone_vz = -rel_vel[2]
        vz_cmd = max(vz_des, self.vz_min)
        a_z = self.kp_vz * (vz_cmd - drone_vz)
        # thrust along the current body-z: f_des is vertical, so this is m*(a_z+g)*n_z. When the vehicle is
        # past 90 deg (n_z < 0) the projection goes negative -> thrust 0 (don't keep driving an inverted body).
        thrust = max(0.0, self.m * (a_z + GRAVITY) * float(n[2]))
        torque = np.array([tau_xy[0], tau_xy[1], 0.0])
        return self.alloc.allocate(thrust, torque, failed=self.failed)
