"""Horizontal model-predictive controller for platform tracking.

Plans the drone's horizontal acceleration over a short horizon to drive the platform-relative
position and velocity to zero, assuming the platform continues at constant velocity. Because it
optimizes over the *future* trajectory it intercepts where the platform is going — eliminating the
steady-state lag a PD controller has against a moving/accelerating target. The first acceleration of
the optimal sequence is applied, then the problem is re-solved next step (receding horizon).

The output is a horizontal acceleration command that feeds the same geometric attitude inner loop +
control allocation as the geometric controller, so MPC and geometric are drop-in comparable.

Relative dynamics (platform - drone), platform assumed constant-velocity over the horizon:
    r_{k+1}  = r_k + v_k dt - 0.5 a_k dt^2
    v_{k+1}  = v_k - a_k dt
where r = platform_xy - drone_xy, v = d/dt r, and a = drone horizontal acceleration (the control).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover
    ca = None


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.05        # s   horizon step (coarser than the control loop)
    horizon: int = 20       # steps -> 1.0 s lookahead
    a_max: float = 3.0      # m/s^2  horizontal acceleration limit (tilt limit)
    w_pos: float = 12.0     # stage position-error weight
    w_vel: float = 1.5      # stage velocity-error weight
    w_ctrl: float = 0.08    # control-effort weight
    w_terminal: float = 60.0  # terminal position+velocity weight


class HorizontalMPC:
    """Receding-horizon horizontal tracker. Build once, call :meth:`compute` each control step."""

    def __init__(self, config: MPCConfig | None = None):
        if ca is None:
            raise ImportError("HorizontalMPC requires casadi. Install with: pip install -e .[mpc]")
        self.cfg = config or MPCConfig()
        self._build()
        self._warm = None

    def _build(self) -> None:
        c = self.cfg
        N, dt = c.horizon, c.dt
        opti = ca.Opti("conic")  # quadratic objective + linear constraints -> QP
        X = opti.variable(4, N + 1)   # [rx, ry, vrx, vry]
        U = opti.variable(2, N)       # [ax, ay]
        x0 = opti.parameter(4)
        opti.subject_to(X[:, 0] == x0)

        cost = 0
        for k in range(N):
            r, v, a = X[0:2, k], X[2:4, k], U[:, k]
            cost += c.w_pos * ca.sumsqr(r) + c.w_vel * ca.sumsqr(v) + c.w_ctrl * ca.sumsqr(a)
            r_next = r + v * dt - 0.5 * a * dt**2
            v_next = v - a * dt
            opti.subject_to(X[0:2, k + 1] == r_next)
            opti.subject_to(X[2:4, k + 1] == v_next)
            opti.subject_to(opti.bounded(-c.a_max, a, c.a_max))
        cost += c.w_terminal * (ca.sumsqr(X[0:2, N]) + ca.sumsqr(X[2:4, N]))
        opti.minimize(cost)
        opti.solver("qrqp", {"print_iter": False, "print_header": False,
                             "print_info": False, "error_on_fail": False})

        self._opti, self._X, self._U, self._x0 = opti, X, U, x0

    def reset(self) -> None:
        self._warm = None

    def compute(self, r_xy: np.ndarray, vr_xy: np.ndarray) -> np.ndarray:
        """Return the horizontal acceleration command [ax, ay] for the current relative state."""
        self._opti.set_value(self._x0, np.array([r_xy[0], r_xy[1], vr_xy[0], vr_xy[1]]))
        if self._warm is not None:
            self._opti.set_initial(self._X, self._warm[0])
            self._opti.set_initial(self._U, self._warm[1])
        try:
            sol = self._opti.solve()
            self._warm = (sol.value(self._X), sol.value(self._U))
            a = sol.value(self._U)[:, 0]
        except Exception:
            # fall back to a proportional command if the solver fails
            a = np.clip(self.cfg.w_pos / 12.0 * np.asarray(r_xy), -self.cfg.a_max, self.cfg.a_max)
        return np.clip(np.asarray(a, dtype=float), -self.cfg.a_max, self.cfg.a_max)
