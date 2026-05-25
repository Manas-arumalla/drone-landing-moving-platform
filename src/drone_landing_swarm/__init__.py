"""Swarm flight-deck recovery — a SEPARATE application that coordinates N drones landing on one moving
deck (scheduling + CBF collision avoidance + holding stack), reusing the single-drone autopilot as a
black-box inner loop. It never modifies the single-drone simulation. See docs/SWARM.md.
"""

from drone_landing_swarm.avoidance import cbf_safe_velocity, min_pairwise_distance
from drone_landing_swarm.consensus import ConsensusConfig, ConsensusDeckEstimator
from drone_landing_swarm.coordinator import SwarmConfig, SwarmCoordinator
from drone_landing_swarm.holding import HoldingConfig, HoldingStack
from drone_landing_swarm.multi_deck import MultiDeckConfig, MultiDeckCoordinator
from drone_landing_swarm.safety import SafetyFilter, SafetySpec, verify_separation
from drone_landing_swarm.scheduler import LandingScheduler, SchedulerConfig, optimal_assignment
from drone_landing_swarm.sensing import SensingConfig, SwarmSensing

__all__ = [
    "cbf_safe_velocity",
    "min_pairwise_distance",
    "LandingScheduler",
    "SchedulerConfig",
    "optimal_assignment",
    "HoldingStack",
    "HoldingConfig",
    "SwarmCoordinator",
    "SwarmConfig",
    "SafetyFilter",
    "SafetySpec",
    "verify_separation",
    "SensingConfig",
    "SwarmSensing",
    "ConsensusConfig",
    "ConsensusDeckEstimator",
    "MultiDeckConfig",
    "MultiDeckCoordinator",
]
