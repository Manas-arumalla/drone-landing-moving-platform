"""Minimum-snap / differential-flatness planner (planning/minsnap.py).

Pure-math properties (boundary conditions, genuine snap-optimality, feasibility time-stretch, the
flatness map) plus a closed-loop double-integrator convergence check and the autopilot wiring smoke.
"""

import importlib.util
import unittest

import numpy as np

from drone_landing.planning.minsnap import (
    GRAVITY,
    MinSnapConfig,
    MinSnapPlan,
    MinSnapTracker,
    flatness_feedforward,
)


class MinSnapPlanTests(unittest.TestCase):
    def test_boundary_conditions_met(self):
        rng = np.random.default_rng(0)
        b0 = rng.normal(0.0, 1.0, size=(4, 2))
        bT = rng.normal(0.0, 1.0, size=(4, 2))
        T = 2.3
        plan = MinSnapPlan(T, b0, bT)
        for k in range(4):
            np.testing.assert_allclose(plan.eval(0.0, k), b0[k], atol=1e-6)
            np.testing.assert_allclose(plan.eval(T, k), bT[k], atol=1e-6)

    def test_snap_optimality_beats_hermite(self):
        """The order-9 min-snap solution must have <= snap cost than the unique order-7 Hermite
        interpolant of the same boundary conditions (any other constraint-satisfying polynomial)."""
        rng = np.random.default_rng(1)
        b0 = rng.normal(0.0, 1.0, size=(4, 1))
        bT = rng.normal(0.0, 1.0, size=(4, 1))
        T = 2.0
        plan = MinSnapPlan(T, b0, bT)
        # order-7 Hermite: 8 coefficients fully determined by the 8 boundary conditions,
        # in normalized time: q^(k)(0) = b0[k]*T^k, q^(k)(1) = bT[k]*T^k
        from math import factorial
        A = np.zeros((8, 8))
        for k in range(4):
            A[k, k] = factorial(k)
            for j in range(k, 8):
                A[4 + k, j] = factorial(j) / factorial(j - k)
        rhs = np.concatenate([b0[:, 0] * T ** np.arange(4), bT[:, 0] * T ** np.arange(4)])
        c7 = np.linalg.solve(A, rhs)
        # numerical snap cost of the Hermite polynomial (4th derivative, normalized time)
        taus = np.linspace(0.0, 1.0, 4001)
        snap7 = np.zeros_like(taus)
        for j in range(4, 8):
            snap7 += factorial(j) / factorial(j - 4) * taus ** (j - 4) * c7[j]
        cost7 = float(np.trapezoid(snap7**2, taus)) / T**7
        self.assertLessEqual(plan.snap_cost(), cost7 + 1e-9)

    def test_rest_to_rest_hits_target(self):
        plan = MinSnapPlan(3.0, np.zeros((4, 2)), np.array([[1.0, -0.5]] + [[0.0, 0.0]] * 3))
        np.testing.assert_allclose(plan.eval(3.0, 0), [1.0, -0.5], atol=1e-8)
        np.testing.assert_allclose(plan.eval(3.0, 1), [0.0, 0.0], atol=1e-8)
        mid = plan.eval(1.5, 0)
        self.assertTrue(0.0 < mid[0] < 1.0)          # passes between the endpoints

    def test_longer_time_lowers_peak_accel(self):
        b0 = np.array([[2.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        bT = np.zeros((4, 2))
        self.assertGreater(MinSnapPlan(1.5, b0, bT).peak_accel(),
                           MinSnapPlan(3.0, b0, bT).peak_accel())


class FlatnessMapTests(unittest.TestCase):
    def test_hover(self):
        thrust, z_b, omega = flatness_feedforward(1.4, np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(thrust, 1.4 * GRAVITY, places=9)
        np.testing.assert_allclose(z_b, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(omega, [0.0, 0.0], atol=1e-12)

    def test_accel_tilts_thrust_vector(self):
        _, z_b, _ = flatness_feedforward(1.4, np.array([2.0, 0.0, 0.0]))
        self.assertGreater(z_b[0], 0.1)              # thrust vector leans into the acceleration
        self.assertGreater(z_b[2], 0.9)


class MinSnapTrackerTests(unittest.TestCase):
    def test_double_integrator_rendezvous(self):
        """Closed loop on the true relative dynamics r'' = -a: the tracker must drive r, v -> 0
        within the acceleration budget."""
        dt = 0.01
        trk = MinSnapTracker(control_dt=dt)
        r = np.array([2.0, -1.2])
        v = np.array([0.3, 0.4])
        for _ in range(int(8.0 / dt)):
            a = trk.compute(r, v)
            self.assertLessEqual(float(np.max(np.abs(a))), trk.cfg.a_max + 1e-9)
            v = v - a * dt
            r = r + v * dt
        self.assertLess(float(np.linalg.norm(r)), 0.05)
        self.assertLess(float(np.linalg.norm(v)), 0.05)

    def test_time_stretch_respects_accel_budget(self):
        cfg = MinSnapConfig(t_min=0.5)               # force an aggressive initial segment time
        trk = MinSnapTracker(control_dt=0.01, config=cfg)
        trk.compute(np.array([3.0, 0.0]), np.zeros(2))
        self.assertLessEqual(trk.plan.peak_accel(), 0.85 * cfg.a_max + 1e-9)

    def test_reset_clears_plan(self):
        trk = MinSnapTracker()
        trk.compute(np.array([1.0, 0.0]), np.zeros(2))
        self.assertIsNotNone(trk.plan)
        trk.reset()
        self.assertIsNone(trk.plan)

    def test_target_offset_rendezvous(self):
        """With target_xy set (up-slope lead) the closed loop settles at the offset, not the centre."""
        dt = 0.01
        trk = MinSnapTracker(control_dt=dt)
        target = np.array([0.15, 0.0])
        r = np.array([1.5, -0.8])
        v = np.zeros(2)
        for _ in range(int(8.0 / dt)):
            a = trk.compute(r, v, target_xy=target)
            v = v - a * dt
            r = r + v * dt
        self.assertLess(float(np.linalg.norm(r - target)), 0.05)   # settled at the lead point
        self.assertGreater(float(np.linalg.norm(r)), 0.09)         # NOT at the centre


@unittest.skipUnless(importlib.util.find_spec("cv2"), "opencv not installed")
class DeckNormalTests(unittest.TestCase):
    """Deck-surface normal from the ArUco PnP rotation (attitude-matched touchdown, Phase 2)."""

    def _camera(self):
        from drone_landing.perception import CameraModel
        return CameraModel(480, 480, 90.0)

    def test_flat_deck_gives_vertical_normal(self):
        from drone_landing.perception.aruco_detector import board_normal_world
        n = board_normal_world(np.zeros(3), self._camera())
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-9)

    def test_tilted_board_recovers_tilt_angle(self):
        from drone_landing.perception.aruco_detector import board_normal_world
        tilt = np.deg2rad(12.0)
        n = board_normal_world(np.array([tilt, 0.0, 0.0]), self._camera())
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=9)
        self.assertGreater(n[2], 0.0)                                  # upward by construction
        self.assertAlmostEqual(float(np.degrees(np.arccos(n[2]))), 12.0, places=6)


class PressNormalTests(unittest.TestCase):
    """Attitude-matched press: press_normal=None must be byte-identical to the legacy level press."""

    def _controller(self):
        from drone_landing.control.geometric import GeometricController
        return GeometricController(1.4, np.array([0.062, 0.038, 0.027]), control_dt=0.01)

    def test_none_press_unchanged(self):
        args = (np.array([0.05, -0.02, -0.02]), np.zeros(3), np.eye(3), np.zeros(3))
        legacy = self._controller().compute(*args, press=True)
        explicit = self._controller().compute(*args, press=True, press_normal=None)
        np.testing.assert_array_equal(legacy, explicit)

    def test_tilted_press_produces_valid_distinct_command(self):
        args = (np.array([0.05, -0.02, -0.02]), np.zeros(3), np.eye(3), np.zeros(3))
        tilt = np.deg2rad(12.0)
        n = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
        level = self._controller().compute(*args, press=True)
        tilted = self._controller().compute(*args, press=True, press_normal=n)
        self.assertTrue(np.all(np.isfinite(tilted)) and np.all(tilted >= 0.0))
        self.assertFalse(np.allclose(level, tilted))                   # it actually acts on the normal

    def test_velocity_hold_pretilt_only_in_terminal_window(self):
        # the pre-tilt is a last-instant maneuver: active inside the final 15 cm, inert above it
        # (holding g*tan(tilt) through the whole commit descent would carry the drone off the deck)
        tilt = np.deg2rad(12.0)
        n = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
        high = (np.array([0.0, 0.0, -0.3]), np.zeros(3), np.eye(3), np.zeros(3))   # 0.3 m up
        low = (np.array([0.0, 0.0, -0.1]), np.zeros(3), np.eye(3), np.zeros(3))    # 0.1 m up
        base_high = self._controller().compute(*high, velocity_hold=True)
        shaped_high = self._controller().compute(*high, velocity_hold=True, press_normal=n)
        np.testing.assert_array_equal(base_high, shaped_high)          # inert above the window
        base_low = self._controller().compute(*low, velocity_hold=True)
        shaped_low = self._controller().compute(*low, velocity_hold=True, press_normal=n)
        self.assertFalse(np.allclose(base_low, shaped_low))            # active inside it


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco not installed")
class MinSnapWiringTests(unittest.TestCase):
    def test_autopilot_builds_and_steps(self):
        """--controller minsnap constructs through the real build path and produces finite thrusts."""
        from drone_landing.cli import SimSpec, build

        world, ap = build(SimSpec("ground", "minsnap"))
        self.assertIsNotNone(ap.minsnap)
        sensors = world.reset(0)
        ap.reset()
        for _ in range(25):                           # SEARCH phase: no image needed
            ctrl = ap.step(None, sensors, world.observe_truth()["support_feet"])
            self.assertTrue(np.all(np.isfinite(ctrl)))
            sensors = world.step(ctrl).sensors
