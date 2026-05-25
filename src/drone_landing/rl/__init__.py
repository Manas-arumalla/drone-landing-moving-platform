"""Reinforcement-learning layer: a Gymnasium env for the landing guidance policy.

The policy learns the outer guidance law (horizontal accel + descent rate) executed by the validated
geometric attitude inner loop on true MuJoCo dynamics. Trained on a calibrated estimator-noise
surrogate for speed, then evaluated on the full vision pipeline. See ``landing_env.py``.
"""

from drone_landing.rl.landing_env import LandingEnv, LandingEnvConfig

__all__ = ["LandingEnv", "LandingEnvConfig"]
