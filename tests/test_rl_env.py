"""Tests for the RL landing environment (drone_landing.rl.landing_env).

Guards the two bugs we fixed: (1) potential-based reward must not incentivize early bail-out, and the
sign/convention must be consistent (platform-drone); (2) the residual baseline (action=0) must be the
competent supervisor+geometric controller, i.e. it should land on the easy distribution.
"""

import importlib.util
import unittest

_HAS = all(importlib.util.find_spec(m) for m in ("numpy", "gymnasium", "mujoco"))


@unittest.skipUnless(_HAS, "numpy/gymnasium/mujoco not installed")
class LandingEnvTests(unittest.TestCase):
    def test_env_checker_passes(self):
        from gymnasium.utils.env_checker import check_env

        from drone_landing.rl import LandingEnv, LandingEnvConfig

        check_env(LandingEnv(LandingEnvConfig(scenario="ground", control_mode="residual")),
                  skip_render_check=True)

    def test_residual_baseline_lands_on_easy(self):
        import numpy as np

        from drone_landing.rl import LandingEnv, LandingEnvConfig

        # action=0 in residual mode == the proven supervisor+geometric controller -> should land on the
        # easy/default (no domain randomization) distribution.
        env = LandingEnv(LandingEnvConfig(scenario="ground", domain_rand=False, control_mode="residual"))
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        succ = 0
        for ep in range(8):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(zero)
                done = term or trunc
            succ += info["termination"] == "success"
        self.assertGreaterEqual(succ, 6)  # competent baseline lands the large majority of easy episodes

    def test_reward_does_not_reward_flying_out(self):
        import numpy as np

        from drone_landing.rl import LandingEnv, LandingEnvConfig

        # potential-based shaping must NOT make bailing out attractive: a policy that flies straight out
        # should earn a clearly negative return (no accumulation-suicide incentive).
        env = LandingEnv(LandingEnvConfig(scenario="ground", domain_rand=False, control_mode="direct"))
        obs, _ = env.reset(seed=0)
        R, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))  # flee
            R += r
            done = term or trunc
        self.assertEqual(info["termination"], "out_of_bounds")
        self.assertLess(R, 0.0)


if __name__ == "__main__":
    unittest.main()
