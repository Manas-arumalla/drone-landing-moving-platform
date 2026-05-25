from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable, Protocol

from drone_landing.envs.mujoco_env import MuJoCoLandingEnv


class MuJoCoPolicy(Protocol):
    def act(self, observation: list[float]) -> list[float]:
        ...


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    termination: str
    steps: int
    episode_return: float
    touchdown_error: float
    relative_speed: float
    leg_contacts: int
    stable_support: bool
    post_landing_steps: int


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    success_rate: float
    mean_return: float
    mean_steps: float
    mean_touchdown_error: float
    mean_relative_speed: float
    results: list[EpisodeResult]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["results"] = [asdict(result) for result in self.results]
        return data


def evaluate_mujoco_policy(
    env_factory: Callable[[int], MuJoCoLandingEnv],
    policy_factory: Callable[[], MuJoCoPolicy],
    seeds: list[int],
    post_landing_steps: int = 0,
) -> EvaluationSummary:
    results: list[EpisodeResult] = []

    for seed in seeds:
        env = env_factory(seed)
        policy = policy_factory()
        obs = env.reset(seed=seed)
        episode_return = 0.0
        result = None

        while True:
            result = env.step(policy.act(obs))
            obs = result.observation
            episode_return += result.reward
            if result.terminated or result.truncated:
                break

        for _ in range(post_landing_steps):
            result = env.step(policy.act(obs))
            obs = result.observation

        assert result is not None
        results.append(
            EpisodeResult(
                seed=seed,
                success=bool(result.info["success"]),
                termination=str(result.info["termination"]),
                steps=env.steps,
                episode_return=episode_return,
                touchdown_error=float(result.info["horizontal_error"]),
                relative_speed=float(result.info["relative_horizontal_speed"]),
                leg_contacts=int(result.info["leg_contacts"]),
                stable_support=bool(result.info["stable_support"]),
                post_landing_steps=post_landing_steps,
            )
        )

    return EvaluationSummary(
        episodes=len(results),
        success_rate=sum(result.success for result in results) / max(1, len(results)),
        mean_return=mean(result.episode_return for result in results),
        mean_steps=mean(result.steps for result in results),
        mean_touchdown_error=mean(result.touchdown_error for result in results),
        mean_relative_speed=mean(result.relative_speed for result in results),
        results=results,
    )

