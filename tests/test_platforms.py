"""P4 platform motion-model tests: inclined deck, USV, truck (pure kinematics, no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from drone_landing.sim.platforms import (
    InclinedDeckMotion,
    PlatformState,
    TruckMotion,
    USVMotion,
    incline_preset,
)
from drone_landing.sim.platforms.inclined import InclinedDeckConfig
from drone_landing.sim.platforms.truck import TruckMotionConfig
from drone_landing.sim.platforms.usv import USVMotionConfig


def _rpy(quat):
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class InclinedDeckTests(unittest.TestCase):
    def test_persistent_tilt_present(self):
        # moderate = 12 deg tilt along +x -> mean pitch ~ 12 deg, roll ~ 0
        m = InclinedDeckMotion(incline_preset("moderate"))
        m.reset(np.random.default_rng(0))
        pitches = []
        for _ in range(400):
            s = m.step(0.02)
            _, pitch, _ = _rpy(s.quat)
            pitches.append(np.rad2deg(pitch))
        self.assertAlmostEqual(float(np.mean(pitches)), 12.0, delta=2.0)   # mean tilt held

    def test_tilt_scales_with_preset(self):
        def mean_tilt(name):
            m = InclinedDeckMotion(incline_preset(name)); m.reset(np.random.default_rng(1))
            ts = [abs(np.rad2deg(_rpy(m.step(0.02).quat)[1])) for _ in range(300)]
            return float(np.mean(ts))
        self.assertLess(mean_tilt("gentle"), mean_tilt("moderate"))
        self.assertLess(mean_tilt("moderate"), mean_tilt("steep"))

    def test_tilt_heading_rotates_into_roll(self):
        # sloping downhill toward +y should put the tilt into roll, not pitch
        m = InclinedDeckMotion(InclinedDeckConfig(incline_deg=12.0, tilt_heading=np.pi / 2))
        m.reset(np.random.default_rng(0))
        rolls, pitches = [], []
        for _ in range(300):
            r, p, _ = _rpy(m.step(0.02).quat)
            rolls.append(abs(np.rad2deg(r))); pitches.append(abs(np.rad2deg(p)))
        self.assertGreater(np.mean(rolls), np.mean(pitches))

    def test_returns_platform_state(self):
        m = InclinedDeckMotion(); s = m.reset(np.random.default_rng(0))
        self.assertIsInstance(s, PlatformState)
        self.assertEqual(s.pos.shape, (3,))
        self.assertEqual(s.quat.shape, (4,))


class USVTests(unittest.TestCase):
    def test_starts_at_origin_and_stays_bounded(self):
        cfg = USVMotionConfig(loop_radius=2.0)
        m = USVMotion(cfg); s0 = m.reset(np.random.default_rng(3))
        self.assertLess(float(np.linalg.norm(s0.pos[:2])), 1e-6)        # starts under the drone
        for _ in range(3000):
            s = m.step(0.02)                                            # loop shifted to start at origin
            self.assertLessEqual(abs(s.pos[0]), 2 * cfg.loop_radius + 0.2)
            self.assertLessEqual(abs(s.pos[1]), 2 * cfg.loop_radius + 0.2)

    def test_has_wave_roll(self):
        m = USVMotion(); m.reset(np.random.default_rng(0))
        rolls = [np.rad2deg(_rpy(m.step(0.02).quat)[0]) for _ in range(1500)]
        self.assertGreater(np.std(rolls), 1.0)            # lively roll, not flat

    def test_velocity_matches_finite_difference(self):
        m = USVMotion(); m.reset(np.random.default_rng(2))
        s0 = m.step(0.0)
        dt = 1e-3
        s1 = m.step(dt)
        fd = (s1.pos - s0.pos) / dt
        np.testing.assert_allclose(fd[:2], s1.lin_vel[:2], atol=0.2)


class TruckTests(unittest.TestCase):
    def test_constant_height_and_level(self):
        m = TruckMotion(); m.reset(np.random.default_rng(0))
        for _ in range(1000):
            s = m.step(0.02)
            self.assertAlmostEqual(s.pos[2], m.config.deck_z, places=6)   # flat bed
            r, p, _ = _rpy(s.quat)
            self.assertAlmostEqual(r, 0.0, places=6)
            self.assertAlmostEqual(p, 0.0, places=6)

    def test_starts_at_origin_and_stays_bounded(self):
        cfg = TruckMotionConfig(loop_radius=2.3)
        m = TruckMotion(cfg); s0 = m.reset(np.random.default_rng(5))
        self.assertLess(float(np.linalg.norm(s0.pos[:2])), 1e-6)        # starts under the drone
        for _ in range(3000):
            s = m.step(0.02)
            self.assertLessEqual(abs(s.pos[0]), 2 * cfg.loop_radius + 0.2)
            self.assertLessEqual(abs(s.pos[1]), 2 * cfg.loop_radius + 0.2)

    def test_moves_forward_along_heading(self):
        m = TruckMotion(); m.reset(np.random.default_rng(0))
        s = m.step(0.02)
        speed = float(np.linalg.norm(s.lin_vel[:2]))
        self.assertGreater(speed, 0.3)                    # cruising, not parked


if __name__ == "__main__":
    unittest.main()
