"""Sense-and-avoid + contingency safety layer (Phase 3).

Static-obstacle sense-and-avoid (an onboard range sensor + a higher-order CBF) and a failsafe state
machine — the real-robotics safety pieces a setpoint tracker omits. All decision-only and estimate-driven
(no ground truth). See ``docs/SAFETY.md``."""

from drone_landing.safety.avoid import AvoidConfig, AvoidReport, HOCBFAvoider, cluster_returns
from drone_landing.safety.contingency import (
    Contingency,
    ContingencyConfig,
    ContingencySupervisor,
    GeofenceSpec,
    HealthStatus,
)
from drone_landing.safety.obstacles import (
    Obstacle,
    ObstacleField,
    RangeSensor,
    RangeSensorConfig,
)

__all__ = [
    "Obstacle",
    "ObstacleField",
    "RangeSensor",
    "RangeSensorConfig",
    "AvoidConfig",
    "AvoidReport",
    "HOCBFAvoider",
    "cluster_returns",
    "ContingencySupervisor",
    "ContingencyConfig",
    "Contingency",
    "GeofenceSpec",
    "HealthStatus",
]
