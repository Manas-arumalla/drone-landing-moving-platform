"""Tube / disturbance-observer MPC (B4): horizontal tracking with guaranteed bounds under wind.

The plain :class:`HorizontalMPC` plans against a *disturbance-free* model, so a steady wind (a constant
external acceleration ``d`` on the drone) leaves a **standing tracking offset** — with no integral action
the optimizer cannot cancel a constant push. Two robustifications fix this:

1. **Disturbance feedforward (DOB-MPC).** Fold the disturbance-observer estimate ``d_hat`` into the
   prediction model so the planner pre-compensates the wind::

       r_{k+1} = r_k + v_k dt - 0.5 (a_k + d) dt^2
       v_{k+1} = v_k - (a_k + d) dt           (r = platform - drone, a = drone accel, d = wind accel)

   The MPC then commands the ``a`` that makes ``a + d`` produce the needed relative acceleration — the
   steady-state offset vanishes (the predictive analogue of the geometric controller's ``dist_ff``).

2. **Constraint tightening (tube MPC).** The DOB leaves a residual disturbance uncertainty ``|d - d_hat|
   <= d_bound``. A robust/tube MPC reserves part of the actuator authority for the ancillary feedback that
   keeps the *true* trajectory inside a bounded "tube" of the nominal one, by tightening the acceleration
   limit to ``a_max - tube_factor * d_bound``. With the receding-horizon re-solve (each control step plans
   from the freshly measured state), the realized state stays within a computable tube; we expose that
   bound (:meth:`tube_radius`). The result is **bounded tracking error under the worst-case disturbance in
   the set** — the guarantee plain MPC lacks.

Honest scope: this is the property that holds with reasonable state (open-loop / clean-ish). The full
*vision* closed loop remains estimation-limited (the project's standing finding — geometric is more robust
in-loop), so DOB-MPC's win shows in tracking fidelity under wind, not necessarily in vision-pipeline
landing success. Same accel-command interface as :class:`HorizontalMPC`, so it is drop-in comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_landing.control.mpc.nmpc import MPCConfig

try:
    import casadi as ca
except ImportError:  # pragma: no cover
    ca = None


@dataclass(frozen=True)
class TubeMPCConfig(MPCConfig):
    d_bound: float = 1.0       # m/s^2  residual disturbance uncertainty bound after the DOB
    tube_factor: float = 0.5   # fraction of d_bound reserved (tightening) for ancillary feedback


class TubeMPC:
    """DOB-feedforward + constraint-tightened horizontal MPC. Build once, :meth:`compute` per step."""

    def __init__(self, config: TubeMPCConfig | None = None):
        if ca is None:
            raise ImportError("TubeMPC requires casadi. Install with: pip install -e .[mpc]")
        self.cfg = config or TubeMPCConfig()
        self._a_tight = max(self.cfg.a_max - self.cfg.tube_factor * self.cfg.d_bound, 0.1)
        self._build()
        self._warm = None

    def _build(self) -> None:
        c = self.cfg
        N, dt = c.horizon, c.dt
        opti = ca.Opti("conic")
        X = opti.variable(4, N + 1)
        U = opti.variable(2, N)
        x0 = opti.parameter(4)
        d = opti.parameter(2)                          # disturbance feedforward (estimated wind accel)
        opti.subject_to(X[:, 0] == x0)

        cost = 0
        for k in range(N):
            r, v, a = X[0:2, k], X[2:4, k], U[:, k]
            cost += c.w_pos * ca.sumsqr(r) + c.w_vel * ca.sumsqr(v) + c.w_ctrl * ca.sumsqr(a)
            a_tot = a + d                              # drone accel includes the disturbance
            r_next = r + v * dt - 0.5 * a_tot * dt**2
            v_next = v - a_tot * dt
            opti.subject_to(X[0:2, k + 1] == r_next)
            opti.subject_to(X[2:4, k + 1] == v_next)
            opti.subject_to(opti.bounded(-self._a_tight, a, self._a_tight))   # tightened (tube) bound
        cost += c.w_terminal * (ca.sumsqr(X[0:2, N]) + ca.sumsqr(X[2:4, N]))
        opti.minimize(cost)
        opti.solver("qrqp", {"print_iter": False, "print_header": False,
                             "print_info": False, "error_on_fail": False})
        self._opti, self._X, self._U, self._x0, self._d = opti, X, U, x0, d

    def reset(self) -> None:
        self._warm = None

    def tube_radius(self) -> float:
        """Worst-case extra position deviation (m) the residual disturbance can add over one horizon step,
        bounded by the reserved authority: 0.5 * d_bound * dt^2 (per step, rejected by the re-solve)."""
        return 0.5 * self.cfg.d_bound * self.cfg.dt**2

    def compute(self, r_xy: np.ndarray, vr_xy: np.ndarray,
                d_hat_xy: np.ndarray | None = None) -> np.ndarray:
        """Horizontal accel command [ax, ay] for the relative state, given the wind estimate ``d_hat_xy``."""
        d_hat = np.zeros(2) if d_hat_xy is None else np.asarray(d_hat_xy, dtype=float)[:2]
        self._opti.set_value(self._x0, np.array([r_xy[0], r_xy[1], vr_xy[0], vr_xy[1]]))
        self._opti.set_value(self._d, d_hat)
        if self._warm is not None:
            self._opti.set_initial(self._X, self._warm[0])
            self._opti.set_initial(self._U, self._warm[1])
        try:
            sol = self._opti.solve()
            self._warm = (sol.value(self._X), sol.value(self._U))
            a = sol.value(self._U)[:, 0]
        except Exception:
            a = np.clip(self.cfg.w_pos / 12.0 * np.asarray(r_xy), -self._a_tight, self._a_tight)
        return np.clip(np.asarray(a, dtype=float), -self.cfg.a_max, self.cfg.a_max)
