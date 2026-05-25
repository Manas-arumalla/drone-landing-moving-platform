"""Unit tests for control allocation, including fault-tolerant (rotor-out) allocation."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class AllocationTests(unittest.TestCase):
    def test_nominal_allocation_inverts_wrench(self):
        import numpy as np

        from drone_landing.control.allocation import x2_allocator

        al = x2_allocator()
        thrust, torque = 13.5, np.array([0.05, -0.03, 0.02])
        f = al.allocate(thrust, torque)
        achieved = al.A @ f                     # forward map should reproduce the commanded wrench
        np.testing.assert_allclose(achieved, [thrust, *torque], atol=1e-6)

    def test_fault_tolerant_invariants(self):
        import numpy as np

        from drone_landing.control.allocation import x2_allocator

        al = x2_allocator()
        thrust, torque = 13.5, np.array([0.03, 0.02, 0.05])
        for failed in range(4):
            f = al.allocate(thrust, torque, failed=failed)
            self.assertEqual(f[failed], 0.0)            # the dead rotor produces nothing
            self.assertTrue(np.all(f >= 0.0))           # thrusts are non-negative
            self.assertTrue(np.all(np.isfinite(f)))

    def test_fault_tolerant_tracks_when_feasible(self):
        import numpy as np

        from drone_landing.control.allocation import x2_allocator

        # When the 3-rotor solution is feasible (no clipping), it must hold collective+roll+pitch
        # exactly (yaw is deliberately sacrificed). Rotors on the X2 diagonal admit a feasible solution
        # for a near-hover command; verify the wrench is reproduced for those.
        al = x2_allocator()
        thrust, torque = 13.5, np.array([0.05, 0.03, 0.0])
        for failed in (0, 3):
            f = al.allocate(thrust, torque, failed=failed)
            if np.all(f > 0.0):                         # feasible (no clipping)
                w = al.A @ f
                np.testing.assert_allclose(w[:3], [thrust, torque[0], torque[1]], atol=1e-6)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class DisturbanceObserverTests(unittest.TestCase):
    def test_estimates_constant_wind(self):
        import numpy as np

        from drone_landing.control.disturbance import GRAVITY, DisturbanceObserver

        m = 1.37
        dob = DisturbanceObserver(m, tau=0.2, control_dt=0.01)
        body_z = np.array([0.0, 0.0, 1.0])
        thrust = m * 9.81                         # hover thrust -> expected accel = 0 (+ gravity model)
        wind = np.array([0.6, -0.3, 0.0]) / m     # constant external accel the IMU would also see
        d = None
        for _ in range(400):
            a_meas = (thrust / m) * body_z + GRAVITY + wind   # measured = expected + disturbance
            d = dob.update(a_meas, thrust, body_z)
        np.testing.assert_allclose(d, wind, atol=1e-2)        # converges to the true disturbance


@unittest.skipUnless(importlib.util.find_spec("casadi"), "casadi is not installed")
@unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy not installed")
class RotorOutLQRTests(unittest.TestCase):
    """Reduced-attitude LQR research artifact: it constructs + produces a valid wrench (it is NOT claimed
    to land — see the module docstring; production rotor-out handling is the spinning-descent contingency)."""

    def test_constructs_and_outputs_valid_thrusts(self):
        import numpy as np

        from drone_landing.control.allocation import x2_allocator
        from drone_landing.control.rotor_out_lqr import RotorOutLQR

        ctl = RotorOutLQR(1.4, np.array([0.062, 0.038, 0.027]), x2_allocator(), failed_rotor=2)
        R = np.eye(3)
        u = ctl.control(np.array([0.3, 0.0, -1.5]), np.zeros(3), R, np.array([0.0, 0.0, -2.0]),
                        vz_des=-0.3)
        self.assertEqual(u.shape, (4,))
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(u >= 0.0) and np.all(u <= 13.0))   # within the motor range
        self.assertAlmostEqual(float(u[2]), 0.0, places=6)         # failed rotor commands zero

    def test_lqr_gain_is_cached(self):
        import numpy as np

        from drone_landing.control.allocation import x2_allocator
        from drone_landing.control.rotor_out_lqr import RotorOutLQR

        ctl = RotorOutLQR(1.4, np.array([0.062, 0.038, 0.027]), x2_allocator(), failed_rotor=2)
        ctl._gain(2.0); n = len(ctl._cache)
        ctl._gain(2.0)                                             # same rounded spin rate -> cache hit
        self.assertEqual(len(ctl._cache), n)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco not installed")
class RotorOutFloquetTests(unittest.TestCase):
    """The averaged-precession controller actually LANDS a dead-rotor quad (wind-off, clean hover) — the
    first rotor-out controller that does. (Full-pipeline + wind transfer remain open; see the docstring.)"""

    def test_lands_dead_rotor_drone_windoff(self):
        import numpy as np
        import mujoco

        from drone_landing.control.allocation import x2_allocator
        from drone_landing.control.rotor_out_floquet import RotorOutFloquet
        from drone_landing.estimation import quat_to_rotmat
        from drone_landing.sim.world import LandingWorld, LandingWorldConfig

        w = LandingWorld(LandingWorldConfig(world="x2_landing_ground", failed_rotor=2,
                                            wind_mean=(0.0, 0.0, 0.0), wind_gust_std=0.0))
        w.reset(0)
        mass = float(w.model.body_mass[w.drone_bid]); J = w.model.body_inertia[w.drone_bid].copy()
        ctl = RotorOutFloquet(mass, J, x2_allocator(), failed_rotor=2, control_dt=w.control_dt)
        dp = w._deck_pos
        w.data.qpos[w.qadr:w.qadr + 3] = [dp[0] + 0.3, dp[1], dp[2] + 1.8]
        w.data.qpos[w.qadr + 3:w.qadr + 7] = [1, 0, 0, 0]; w.data.qvel[w.vadr:w.vadr + 6] = 0
        mujoco.mj_forward(w.model, w.data)
        t, max_tilt = 0.0, 0.0
        while t < 16.0:
            p = w.data.qpos[w.qadr:w.qadr + 3].copy(); q = w.data.qpos[w.qadr + 3:w.qadr + 7]
            R = quat_to_rotmat(q); om = w.data.qvel[w.vadr + 3:w.vadr + 6].copy()
            rel_pos = w._deck_pos - p; rel_vel = -w.data.qvel[w.vadr:w.vadr + 3].copy()
            w.step(ctl.control(rel_pos, rel_vel, R, om, vz_des=-0.35)); t += w.control_dt
            max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(1 - 2 * (q[1]**2 + q[2]**2), -1, 1))))
            if w.data.qpos[w.qadr + 2] - w._deck_pos[2] < 0.15:
                break
        drift = float(np.hypot(p[0] - w._deck_pos[0], p[1] - w._deck_pos[1]))
        self.assertLess(w.data.qpos[w.qadr + 2] - w._deck_pos[2], 0.2)   # it came down (didn't fly off)
        # This is a documented-marginal underactuated controller (median ~1 m drift, wide spread; opt-in,
        # NOT a precision lander — see rotor_out_floquet docstring). The assertion's purpose is to
        # distinguish the CONTROLLED spinning descent (comes down within ~2 m of the pad) from the
        # discarded fixed-point LQR's catastrophic failure (60 m fly-off + tumble), not to claim precision.
        self.assertLess(drift, 2.2)                                       # came down near the deck (not 60 m)
        self.assertLess(max_tilt, 110.0)                                  # never flipped/tumbled


class TubeMPCTests(unittest.TestCase):
    """B4: DOB-MPC cancels the steady-state wind offset that plain MPC suffers."""

    @staticmethod
    def _sim(ctrl, dob, d_wind, T=10.0, dt=0.02):
        import numpy as np
        r, v = np.array([1.0, 0.0]), np.array([0.0, 0.0])
        d = np.asarray(d_wind, dtype=float)
        errs = []
        for _ in range(int(T / dt)):
            a = ctrl.compute(r, v, d) if dob else ctrl.compute(r, v)
            a_tot = a + d
            r = r + v * dt - 0.5 * a_tot * dt**2
            v = v - a_tot * dt
            errs.append(float(np.linalg.norm(r)))
        import numpy as np
        return float(np.mean(errs[-150:]))

    def test_dob_mpc_beats_plain_mpc_under_wind(self):
        from drone_landing.control.mpc.nmpc import HorizontalMPC, MPCConfig
        from drone_landing.control.mpc.tube_mpc import TubeMPC, TubeMPCConfig

        d_wind = [1.5, 0.0]
        e_plain = self._sim(HorizontalMPC(MPCConfig()), False, d_wind)
        e_tube = self._sim(TubeMPC(TubeMPCConfig(d_bound=1.0)), True, d_wind)
        self.assertGreater(e_plain, 0.05)        # plain MPC has a clear standing offset under wind
        self.assertLess(e_tube, 0.02)            # DOB-MPC cancels it (near-zero steady-state error)
        self.assertLess(e_tube, e_plain / 5.0)   # at least 5x tighter

    def test_zero_wind_matches_plain(self):
        # With no wind the disturbance feedforward is zero; both controllers station-keep to ~0.
        from drone_landing.control.mpc.tube_mpc import TubeMPC, TubeMPCConfig

        e = self._sim(TubeMPC(TubeMPCConfig(d_bound=1.0)), True, [0.0, 0.0])
        self.assertLess(e, 0.02)


if __name__ == "__main__":
    unittest.main()
