"""Differential-flatness minimum-snap approach planning (Mellinger & Kumar, ICRA 2011).

The quadrotor is differentially flat in (x, y, z, yaw): every state and input follows algebraically
from the flat outputs and their derivatives — acceleration fixes the thrust vector (and so attitude),
jerk fixes the body rates. Planning the flat outputs as polynomials that minimize the integral of
squared **snap** (the 4th derivative) yields smooth, dynamically-graceful trajectories whose
feedforward the attitude loop only has to *correct*, not generate.

Here the planner runs in the **platform-relative frame** the whole stack estimates in: it plans the
relative position ``r = platform − drone`` from the current EKF value to a rendezvous at ``r = 0``
with matched velocity (``ṙ = 0``), replanning receding-horizon style. With the platform near constant
velocity over the horizon, ``r̈ = −a`` where ``a`` is the drone's horizontal acceleration — so the
planned relative acceleration maps directly to the drone feedforward. The tracker outputs the same
horizontal-acceleration command as :class:`~drone_landing.control.mpc.HorizontalMPC`, feeding the
identical geometric attitude inner loop (drop-in comparable, supervisor commit logic untouched).

No privileged state anywhere: the planner consumes the EKF relative estimate and the optical-flow
relative velocity — the same inputs the MPC uses.

Math: a single segment of an order-9 polynomial per axis, minimizing ``∫ (d⁴p/dt⁴)² dt`` subject to
position/velocity/acceleration/jerk boundary conditions at both ends (8 equality constraints, 10
coefficients → a genuine 2-DOF-per-axis QP, solved exactly via its KKT system in normalized time).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

GRAVITY = 9.81
_N_COEF = 10          # order-9 polynomial
_N_DERIV = 4          # constrain pos / vel / acc / jerk at each end


def _snap_cost_matrix() -> np.ndarray:
    """Q with ∫₀¹ q⁗(τ)² dτ = cᵀ Q c for coefficients c of q(τ) = Σ c_i τ^i."""
    Q = np.zeros((_N_COEF, _N_COEF))
    for i in range(4, _N_COEF):
        for j in range(4, _N_COEF):
            ki = factorial(i) / factorial(i - 4)
            kj = factorial(j) / factorial(j - 4)
            Q[i, j] = ki * kj / (i + j - 7)
    return Q


def _boundary_matrix() -> np.ndarray:
    """A with A c = [q(0), q̇(0), q̈(0), q⃛(0), q(1), q̇(1), q̈(1), q⃛(1)] (normalized time)."""
    A = np.zeros((2 * _N_DERIV, _N_COEF))
    for k in range(_N_DERIV):
        A[k, k] = factorial(k)
        for j in range(k, _N_COEF):
            A[_N_DERIV + k, j] = factorial(j) / factorial(j - k)   # τ=1: Σ j!/(j−k)! c_j
    return A


_Q = _snap_cost_matrix()
_A = _boundary_matrix()
_KKT = np.block([[2.0 * _Q, _A.T], [_A, np.zeros((2 * _N_DERIV, 2 * _N_DERIV))]])


class MinSnapPlan:
    """One minimum-snap segment over ``d`` axes: evaluate the k-th derivative at time t ∈ [0, T]."""

    def __init__(self, T: float, b0: np.ndarray, bT: np.ndarray):
        """``b0``/``bT``: (4, d) boundary [pos; vel; acc; jerk] at t=0 and t=T (real time units)."""
        self.T = float(T)
        b0 = np.atleast_2d(np.asarray(b0, dtype=float))
        bT = np.atleast_2d(np.asarray(bT, dtype=float))
        d = b0.shape[1]
        scale = self.T ** np.arange(_N_DERIV)                     # to normalized time τ = t/T
        rhs = np.zeros((_N_COEF + 2 * _N_DERIV, d))
        rhs[_N_COEF:_N_COEF + _N_DERIV] = b0 * scale[:, None]
        rhs[_N_COEF + _N_DERIV:] = bT * scale[:, None]
        try:
            sol = np.linalg.solve(_KKT, rhs)
        except np.linalg.LinAlgError:                              # pragma: no cover
            sol = np.linalg.lstsq(_KKT, rhs, rcond=None)[0]
        self.coef = sol[:_N_COEF]                                  # (10, d), normalized time

    def eval(self, t: float, deriv: int = 0) -> np.ndarray:
        """k-th time derivative of the trajectory at t (clamped to [0, T]), shape (d,)."""
        tau = float(np.clip(t / self.T, 0.0, 1.0))
        out = np.zeros(self.coef.shape[1])
        for j in range(deriv, _N_COEF):
            out += factorial(j) / factorial(j - deriv) * (tau ** (j - deriv)) * self.coef[j]
        return out / self.T ** deriv

    def snap_cost(self) -> float:
        """∫₀ᵀ ‖snap‖² dt (for tests: the solution must beat any other constraint-satisfying poly)."""
        return float(sum(c @ _Q @ c for c in self.coef.T) / self.T ** 7)

    def peak_accel(self, samples: int = 60) -> float:
        ts = np.linspace(0.0, self.T, samples)
        return float(max(np.linalg.norm(self.eval(t, 2)) for t in ts))


def flatness_feedforward(mass: float, accel: np.ndarray, jerk: np.ndarray | None = None,
                         yaw: float = 0.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Flat outputs → inputs: desired accel fixes (thrust, body-z); jerk fixes the roll/pitch rates.

    Returns ``(thrust [N], z_b unit vector, omega_xy [rad/s])`` — the Mellinger & Kumar map. At hover
    (accel = jerk = 0): thrust = m·g, z_b = e3, rates = 0.
    """
    f = mass * (np.asarray(accel, dtype=float) + np.array([0.0, 0.0, GRAVITY]))
    thrust = float(np.linalg.norm(f))
    z_b = f / max(thrust, 1e-9)
    omega = np.zeros(2)
    if jerk is not None:
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= max(np.linalg.norm(y_b), 1e-9)
        x_b = np.cross(y_b, z_b)
        h = (mass / max(thrust, 1e-9)) * (np.asarray(jerk, float) - (z_b @ jerk) * z_b)
        omega = np.array([-h @ y_b, h @ x_b])                      # p, q
    return thrust, z_b, omega


@dataclass(frozen=True)
class MinSnapConfig:
    a_max: float = 3.0          # m/s^2 horizontal accel limit (same tilt budget as the MPC)
    kp: float = 2.6             # trajectory-tracking feedback (matched to the geometric gains)
    kd: float = 3.0
    v_cruise: float = 0.6       # m/s  sizes the segment time from the initial distance
    t_min: float = 2.0          # s    segment-duration bounds
    t_max: float = 6.0
    replan_every: float = 0.5   # s    receding-horizon replan cadence
    replan_dev: float = 0.6     # m    replan early if the estimate deviates this far from the plan
    stretch: float = 1.4        # time-stretch factor when the plan exceeds the accel budget
    max_stretch: int = 4
    # Boundary sanitization — the decisive robustness pieces on a NOISY estimate. Unlike the MPC
    # (which re-reads the state every step, so a spiky velocity hurts for 10 ms), a planner BAKES its
    # boundary conditions into seconds of feedforward. Raw optical-flow spikes as velocity boundaries
    # and "current accel = −last command" as the accel boundary both destabilize the loop (the latter
    # self-excites once a command saturates). So: plan from a low-passed, clipped velocity and a
    # tightly clipped accel boundary; the tracking FEEDBACK still uses the raw current estimate.
    v_bound_max: float = 1.2    # m/s  clip on the plan's velocity boundary (mirrors the EKF clamp)
    v_bound_tau: float = 0.3    # s    low-pass time constant for the boundary velocity
    a_bound_max: float = 1.0    # m/s^2 clip on the plan's initial-acceleration boundary


class MinSnapTracker:
    """Receding-horizon minimum-snap tracker: build once, call :meth:`compute` each control step.

    Same interface and frame conventions as :class:`HorizontalMPC`: input the platform-relative
    position/velocity estimate, output the drone's horizontal acceleration command.
    """

    def __init__(self, control_dt: float = 0.01, config: MinSnapConfig | None = None):
        self.dt = float(control_dt)
        self.cfg = config or MinSnapConfig()
        self.reset()

    def reset(self) -> None:
        self.plan: MinSnapPlan | None = None
        self._age = 0.0
        self._last_a = np.zeros(2)
        self._v_lpf = np.zeros(2)                                  # boundary-velocity low-pass state

    def _replan(self, r_xy: np.ndarray) -> None:
        c = self.cfg
        # Sanitized boundary state: position from the current estimate; velocity low-passed + clipped
        # (never a raw flow spike); initial accel from command continuity, tightly clipped so a
        # saturated command cannot self-excite the next plan.
        v0 = np.clip(self._v_lpf, -c.v_bound_max, c.v_bound_max)
        a0 = np.clip(-self._last_a, -c.a_bound_max, c.a_bound_max)
        b0 = np.stack([r_xy, v0, a0, np.zeros(2)])
        bT = np.zeros((4, 2))                                      # rendezvous: r = ṙ = r̈ = r⃛ = 0
        T = float(np.clip(np.linalg.norm(r_xy) / c.v_cruise, c.t_min, c.t_max))
        plan = MinSnapPlan(T, b0, bT)
        for _ in range(c.max_stretch):                             # stretch time until within budget
            if plan.peak_accel() <= 0.85 * c.a_max:
                break
            T *= c.stretch
            plan = MinSnapPlan(T, b0, bT)
        self.plan, self._age = plan, 0.0

    def compute(self, r_xy: np.ndarray, vr_xy: np.ndarray) -> np.ndarray:
        """Horizontal acceleration command [ax, ay] for the current relative estimate."""
        c = self.cfg
        r_xy = np.asarray(r_xy, dtype=float)[:2]
        vr_xy = np.asarray(vr_xy, dtype=float)[:2]
        alpha = min(1.0, self.dt / c.v_bound_tau)
        self._v_lpf = self._v_lpf + alpha * (vr_xy - self._v_lpf)
        if self.plan is not None:
            self._age += self.dt
        need = (self.plan is None or self._age >= c.replan_every
                or float(np.linalg.norm(r_xy - self.plan.eval(self._age, 0))) > c.replan_dev)
        if need:
            self._replan(r_xy)
        t = self._age
        r_ref = self.plan.eval(t, 0)
        v_ref = self.plan.eval(t, 1)
        a_ff = -self.plan.eval(t, 2)                               # r̈ = −a  →  drone accel feedforward
        a = a_ff + c.kp * (r_xy - r_ref) + c.kd * (vr_xy - v_ref)
        a = np.clip(a, -c.a_max, c.a_max)
        self._last_a = a
        return a
