"""Tests for B2: Hamilton-Jacobi reachability safe-landing set + runtime-assurance shield."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class ReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from drone_landing.control.reachability import LandingReachability, ReachabilityConfig
        cls.R = LandingReachability(ReachabilityConfig())

    def test_basic_membership(self):
        R = self.R
        self.assertTrue(R.is_safe(2.0, 0.0))        # hovering high -> safe
        self.assertTrue(R.is_safe(3.5, -1.0))       # high + gentle descent -> safe
        self.assertFalse(R.is_safe(0.2, -3.5))      # fast descent right at the deck -> can't brake -> unsafe

    def test_respects_analytic_braking_boundary(self):
        # The grid safe-set's lower boundary must sit above the analytic worst-case braking curve
        # (within one discrete step |w|*dt + one cell).
        import numpy as np

        R = self.R
        dh = R.h_grid[1] - R.h_grid[0]
        for w in (-1.5, -2.0, -2.5, -3.0, -3.5):
            iw = int(round((w + R.cfg.w_max) / (R.w_grid[1] - R.w_grid[0])))
            col = R.safe[:, iw]
            self.assertTrue(col.any())
            min_h = R.h_grid[int(np.argmax(col))]
            hb = float(R.braking_boundary(np.array([w]))[0])
            self.assertGreaterEqual(min_h, hb - (abs(w) * R.cfg.dt + dh))

    def test_shield_overrides_unsafe_command(self):
        R = self.R
        a, intervened = R.safe_action(0.4, -3.0, a_nominal=R.cfg.a_min)   # reckless dive at the deck
        self.assertTrue(intervened)
        self.assertGreater(a, 0.0)                                        # forced to brake upward
        a2, intervened2 = R.safe_action(2.5, -0.3, a_nominal=0.0)        # gentle, clearly safe
        self.assertFalse(intervened2)

    def test_safe_descent_speed_grows_with_altitude(self):
        # The velocity-level shield used by the autopilot: more altitude -> more braking room -> a faster
        # descent is still safe; near the deck only a slow descent is safe.
        R = self.R
        low = R.safe_descent_speed(0.3)
        high = R.safe_descent_speed(3.0)
        self.assertGreater(high, low)                  # more braking room up high -> faster safe descent
        self.assertGreater(low, 0.0)
        self.assertLessEqual(high, R.cfg.w_max + 1e-6)   # bounded by the modelled speed range

    def test_shield_guarantees_soft_touchdown_under_worst_case(self):
        # Closed-loop reduced-model sim: a reckless nominal controller (always max descent) + adversarial
        # disturbance. The shield must still produce a soft touchdown (|w| <= w_land at h<=0).
        import numpy as np

        R = self.R
        c = R.cfg
        h, w = 3.0, 0.0
        touchdown_w = None
        for _ in range(2000):
            a_nom = c.a_min                                  # reckless: command max downward accel
            a, _ = R.safe_action(h, w, a_nom)
            d = -c.d_max if w < 0 else c.d_max               # adversarial disturbance (opposes braking)
            w = w + (a + d) * c.dt
            h = h + w * c.dt
            if h <= 0:
                touchdown_w = w
                break
        self.assertIsNotNone(touchdown_w)
        self.assertLessEqual(abs(touchdown_w), c.w_land + 0.15)    # soft landing despite worst-case wind


if __name__ == "__main__":
    unittest.main()
