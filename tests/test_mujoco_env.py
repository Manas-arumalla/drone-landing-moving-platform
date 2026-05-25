import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is not installed")
class MuJoCoLandingEnvTests(unittest.TestCase):
    def test_mujoco_env_reset_and_step(self):
        from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv

        env = MuJoCoLandingEnv(MuJoCoLandingConfig(seed=1, platform_mode="static", max_steps=5))
        obs = env.reset(seed=1)
        self.assertEqual(len(obs), 17)
        result = env.step([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(result.observation), 17)
        self.assertIn("leg_contacts", result.info)
        self.assertIn("motor_rpm_fl", result.info)
        self.assertIn("wind_x", result.info)

    def test_strict_physics_landing_is_default(self):
        from drone_landing.envs.mujoco_env import MuJoCoLandingConfig

        self.assertFalse(MuJoCoLandingConfig().lock_after_success)
        self.assertLessEqual(MuJoCoLandingConfig().success_radius, 0.35)

    def test_landing_lock_tracks_platform_after_success(self):
        from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv

        env = MuJoCoLandingEnv(
            MuJoCoLandingConfig(
                seed=2,
                platform_mode="sinusoidal",
                platform_speed=0.2,
                platform_amplitude=0.5,
                lock_after_success=True,
            )
        )
        env.reset(seed=2)
        env._engage_landing_lock()
        before = env.observation()
        result = env.step([0.0, 0.0, 0.0, 0.0])
        after = result.observation

        self.assertTrue(result.info["landed"])
        self.assertAlmostEqual(before[0], after[0], places=6)
        self.assertAlmostEqual(before[1], after[1], places=6)
        self.assertAlmostEqual(after[7], 0.0, places=6)
        self.assertAlmostEqual(after[8], 0.0, places=6)

    def test_wind_config_is_reported(self):
        from drone_landing.envs.mujoco_env import MuJoCoLandingConfig, MuJoCoLandingEnv

        env = MuJoCoLandingEnv(
            MuJoCoLandingConfig(seed=3, platform_mode="static", wind_x=0.25, wind_gust_std=0.01)
        )
        env.reset(seed=3)
        result = env.step([0.0, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(result.info["wind_x"], 0.25)


if __name__ == "__main__":
    unittest.main()
