"""Tests for B1 maritime fidelity: wave spectra (JONSWAP/PM) + RAOs + ship air-wake."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class WaveSpectrumTests(unittest.TestCase):
    def test_jonswap_recovers_significant_height(self):
        import numpy as np

        from drone_landing.sim.platforms.wave_spectrum import jonswap, significant_height

        w = np.linspace(0.05, 4.0, 3000)
        for hs, tp in [(0.6, 5.5), (1.88, 8.8), (5.0, 12.4)]:
            s = jonswap(w, hs, tp, gamma=3.3)
            self.assertAlmostEqual(significant_height(w, s), hs, delta=0.05 * hs)

    def test_pm_is_jonswap_gamma_one(self):
        import numpy as np

        from drone_landing.sim.platforms.wave_spectrum import jonswap, pierson_moskowitz

        w = np.linspace(0.1, 3.0, 500)
        np.testing.assert_allclose(jonswap(w, 2.0, 9.0, gamma=1.0),
                                   pierson_moskowitz(w, 2.0, 9.0), rtol=1e-6, atol=1e-9)

    def test_peak_near_target_period(self):
        import numpy as np

        from drone_landing.sim.platforms.wave_spectrum import jonswap

        w = np.linspace(0.1, 3.0, 2000)
        tp = 8.8
        s = jonswap(w, 1.88, tp, 3.3)
        w_peak = w[int(np.argmax(s))]
        self.assertAlmostEqual(2 * np.pi / w_peak, tp, delta=0.6)   # spectral peak ~ Tp

    def test_spectral_matches_sinusoid_rms(self):
        # The calibrated spectral model reproduces the validated sum-of-sinusoids RMS motion (so the
        # ship-landing result is preserved) while using physically-correct spectral content.
        import numpy as np

        from drone_landing.sim.platforms import sea_state
        from drone_landing.sim.platforms.ship import ShipDeckMotion

        def heave_rms(cfg, seed=0):
            m = ShipDeckMotion(cfg); m.reset(np.random.default_rng(seed))
            zs = [m.step(0.05).pos[2] for _ in range(3000)]
            return float(np.std(zs))

        for nm in ["calm", "moderate", "rough"]:
            a = heave_rms(sea_state(nm))
            b = heave_rms(sea_state(nm, spectral=True))
            self.assertLess(abs(a - b), 0.35 * a + 0.01)   # within ~35% (random phases / finite sample)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class DataDrivenMotionTests(unittest.TestCase):
    def test_replays_recorded_trajectory(self):
        # Build a known sinusoidal 6-DOF trace; the model should replay it (interpolated pose + velocity).
        import numpy as np

        from drone_landing.sim.platforms.data_driven import DataDrivenDeckMotion

        t = np.linspace(0, 10, 1001)
        z = 0.30 + 0.1 * np.sin(2 * np.pi * t / 5.0)            # 0.1 m heave, 5 s period
        motion = np.column_stack([np.zeros_like(t), np.zeros_like(t), z,
                                  np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)])
        m = DataDrivenDeckMotion(t, motion, random_start=False)
        m.reset(np.random.default_rng(0))
        zs = [m.step(0.05).pos[2] for _ in range(200)]          # 10 s
        self.assertAlmostEqual(float(np.mean(zs)), 0.30, delta=0.02)
        self.assertAlmostEqual(float(np.std(zs)), 0.1 / np.sqrt(2), delta=0.02)   # RMS of a sine

    def test_relative_xy_starts_at_origin(self):
        # A trajectory with large surge offset should start at the world-origin XY (relative_xy).
        import numpy as np

        from drone_landing.sim.platforms.data_driven import DataDrivenDeckMotion

        t = np.linspace(0, 10, 101)
        motion = np.column_stack([100.0 + 0.4 * t, np.full_like(t, 50.0), np.full_like(t, 0.3),
                                  np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)])
        m = DataDrivenDeckMotion(t, motion, random_start=False, relative_xy=True)
        s0 = m.reset(np.random.default_rng(0))
        np.testing.assert_allclose(s0.pos[:2], [0.0, 0.0], atol=1e-6)   # surge offset stripped at reset


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class AirwakeTests(unittest.TestCase):
    def test_envelope_decays_with_height_and_radius(self):
        import numpy as np

        from drone_landing.sim.airwake import AirwakeConfig, ShipAirwake

        aw = ShipAirwake(AirwakeConfig())
        deck = np.array([0.0, 0.0, 0.3])
        at_deck = aw.envelope(np.array([0.0, 0.0, 0.4]), deck)
        high = aw.envelope(np.array([0.0, 0.0, 3.0]), deck)
        offset = aw.envelope(np.array([3.0, 0.0, 0.4]), deck)
        self.assertGreater(at_deck, high)        # decays with height
        self.assertGreater(at_deck, offset)      # decays with horizontal distance
        self.assertGreaterEqual(at_deck, 0.0)
        self.assertLessEqual(at_deck, 1.0)

    def test_force_stronger_near_deck(self):
        import numpy as np

        from drone_landing.sim.airwake import AirwakeConfig, ShipAirwake

        aw = ShipAirwake(AirwakeConfig()); rng = np.random.default_rng(0); aw.reset(rng)
        deck = np.array([0.0, 0.0, 0.3])
        near = np.mean([np.linalg.norm(aw.force(np.array([0, 0, 0.5]), deck, 0.0, rng, 0.01))
                        for _ in range(200)])
        far = np.mean([np.linalg.norm(aw.force(np.array([0, 0, 4.0]), deck, 0.0, rng, 0.01))
                       for _ in range(200)])
        self.assertGreater(near, 3 * far)        # the burble is local to the deck


if __name__ == "__main__":
    unittest.main()
