from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from drone_landing.envs.core import LandingEnv


class Policy(Protocol):
    def act(self, observation: list[float]) -> list[float]:
        ...


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    success_rate: float
    mean_return: float
    mean_steps: float
    mean_touchdown_error: float


def evaluate_policy(env: LandingEnv, policy: Policy, episodes: int = 20) -> EvaluationSummary:
    returns: list[float] = []
    steps: list[int] = []
    errors: list[float] = []
    successes = 0

    for episode in range(episodes):
        obs = env.reset(seed=(env.config.seed or 0) + episode)
        done = False
        total_reward = 0.0
        step_count = 0
        last_error = env.horizontal_error()

        while not done:
            result = env.step(policy.act(obs))
            obs = result.observation
            total_reward += result.reward
            step_count += 1
            last_error = float(result.info["horizontal_error"])
            done = result.terminated or result.truncated

        successes += int(bool(result.info["success"]))
        returns.append(total_reward)
        steps.append(step_count)
        errors.append(last_error)

    return EvaluationSummary(
        episodes=episodes,
        success_rate=successes / episodes,
        mean_return=mean(returns),
        mean_steps=mean(steps),
        mean_touchdown_error=mean(errors),
    )

