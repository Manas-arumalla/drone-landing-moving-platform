"""Unit tests for the green-deck motion predictor (planning/deck_predictor.py)."""

import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class DeckMotionPredictorTests(unittest.TestCase):
    def _feed(self, pred, signal_fn, duration=12.0, dt=0.02, noise=0.0, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        n = int(duration / dt)
        for k in range(n):
            t = k * dt
            pred.update(t, signal_fn(t) + (rng.normal(0, noise) if noise else 0.0))

    def test_locks_and_recovers_period(self):
        from drone_landing.planning import DeckMotionPredictor

        period = 5.0
        pred = DeckMotionPredictor()
        self._feed(pred, lambda t: 0.15 * math.sin(2 * math.pi * t / period), noise=0.01)
        self.assertTrue(pred.locked)
        est_period = 2 * math.pi / pred.omega
        self.assertAlmostEqual(est_period, period, delta=0.5)
        self.assertAlmostEqual(pred.amplitude, 0.15, delta=0.04)

    def test_heave_rate_tracks_truth(self):
        import numpy as np

        from drone_landing.planning import DeckMotionPredictor

        # The meaningful property is that the nowcast *tracks* the true deck vertical velocity over
        # time (sign + phase), not the exact value at one instant (a pure tone over a short window has
        # a frequency-resolution-limited phase error; real multi-component seas track better).
        period, amp, dt = 5.0, 0.15, 0.02
        w = 2 * math.pi / period
        pred = DeckMotionPredictor()
        true, est = [], []
        for k in range(int(16.0 / dt)):
            t = k * dt
            pred.update(t, amp * math.sin(w * t))
            if pred.locked and t > 9.0:
                true.append(amp * w * math.cos(w * t))
                est.append(pred.heave_rate(0.0))
        true, est = np.asarray(true), np.asarray(est)
        self.assertTrue(pred.locked)
        self.assertGreater(float(np.corrcoef(est, true)[0, 1]), 0.7)
        self.assertLess(float(np.sqrt(np.mean((est - true) ** 2))), 0.6 * true.std() + 0.05)

    def test_calm_water_is_always_green(self):
        from drone_landing.planning import DeckMotionPredictor

        pred = DeckMotionPredictor()
        self._feed(pred, lambda t: 0.003 * math.sin(2 * math.pi * t / 5.0), noise=0.005)
        # amplitude below min_amplitude -> calm -> every instant green
        self.assertTrue(pred.is_calm())
        self.assertTrue(pred.in_green_window(descent_time=0.6))
        self.assertEqual(pred.time_to_green(descent_time=0.6), 0.0)

    def test_green_window_detected_in_seaway(self):
        from drone_landing.planning import DeckMotionPredictor

        # a clear single-tone heave has obvious quiescent windows near its peaks/troughs
        pred = DeckMotionPredictor()
        self._feed(pred, lambda t: 0.15 * math.sin(2 * math.pi * t / 5.0), noise=0.005)
        self.assertTrue(pred.locked)
        self.assertFalse(pred.is_calm())
        # within a 5 s period there must be a green window reachable inside the horizon
        ttg = pred.time_to_green(descent_time=0.5)
        self.assertIsNotNone(ttg)
        self.assertLessEqual(ttg, pred.cfg.horizon)

    def test_not_locked_before_enough_samples(self):
        from drone_landing.planning import DeckMotionPredictor

        pred = DeckMotionPredictor()
        self._feed(pred, lambda t: 0.15 * math.sin(2 * math.pi * t / 5.0), duration=0.2, dt=0.02)
        self.assertFalse(pred.locked)
        # graceful no-ops while unlocked
        self.assertEqual(pred.heave_rate(0.0), 0.0)
        self.assertTrue(pred.in_green_window(0.6))  # unlocked -> treated as calm/green


if __name__ == "__main__":
    unittest.main()
