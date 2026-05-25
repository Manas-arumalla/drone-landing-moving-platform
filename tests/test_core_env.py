import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from drone_landing.control import CascadedPIDController
from drone_landing.envs import LandingConfig, LandingEnv


class LandingEnvTests(unittest.TestCase):
    def test_reset_observation_shape(self):
        env = LandingEnv(LandingConfig(seed=1))
        obs = env.reset()
        self.assertEqual(len(obs), 8)

    def test_success_when_centered_and_slow(self):
        env = LandingEnv(LandingConfig(platform_mode="static", seed=2))
        env.reset()
        env.drone.x = 0.0
        env.drone.y = 0.0
        env.drone.z = -0.01
        env.drone.vx = 0.0
        env.drone.vy = 0.0
        env.drone.vz = -0.1
        result = env.step([0.0, 0.0, 0.2])
        self.assertTrue(result.terminated)
        self.assertTrue(result.info["success"])
        self.assertTrue(result.info["turn_off"])

    def test_crash_when_landing_far_from_platform(self):
        env = LandingEnv(LandingConfig(platform_mode="static", seed=3))
        env.reset()
        env.drone.x = 3.0
        env.drone.y = 0.0
        env.drone.z = -0.01
        env.drone.vz = -0.1
        result = env.step([0.0, 0.0, 0.2])
        self.assertTrue(result.terminated)
        self.assertEqual(result.info["termination"], "crash")

    def test_pid_runs_without_invalid_actions(self):
        env = LandingEnv(LandingConfig(platform_mode="sinusoidal", seed=4, max_steps=80))
        policy = CascadedPIDController()
        obs = env.reset()
        for _ in range(10):
            action = policy.act(obs)
            self.assertEqual(len(action), 3)
            self.assertTrue(all(-1.0 <= value <= 1.0 for value in action))
            obs = env.step(action).observation

    def test_mujoco_visual_quadrotor_uses_x_frame_arms(self):
        xml_path = Path("assets/mujoco/landing_scene.xml")
        root = ET.parse(xml_path).getroot()
        geoms = {geom.attrib["name"]: geom.attrib for geom in root.iter("geom") if "name" in geom.attrib}

        for name in ("arm_fl", "arm_fr", "arm_rl", "arm_rr"):
            self.assertIn(name, geoms)

        self.assertEqual(geoms["arm_fl"]["fromto"].split()[-3:], geoms["rotor_fl"]["pos"].split())
        self.assertEqual(geoms["arm_fr"]["fromto"].split()[-3:], geoms["rotor_fr"]["pos"].split())
        self.assertEqual(geoms["arm_rl"]["fromto"].split()[-3:], geoms["rotor_rl"]["pos"].split())
        self.assertEqual(geoms["arm_rr"]["fromto"].split()[-3:], geoms["rotor_rr"]["pos"].split())


if __name__ == "__main__":
    unittest.main()
