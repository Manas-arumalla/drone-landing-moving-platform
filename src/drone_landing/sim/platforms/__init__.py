"""Platform motion models. The platform's own motion is prescribed by a validated model
(honest per the Realism Charter, since the platform is far heavier than the drone); the
drone-deck interaction is always true contact physics."""

from drone_landing.sim.platforms.base import PlatformMotion, PlatformState, yaw_to_quat
from drone_landing.sim.platforms.ground import GroundMotionConfig, RandomGroundMotion
from drone_landing.sim.platforms.data_driven import DataDrivenDeckMotion
from drone_landing.sim.platforms.inclined import (
    InclinedDeckConfig,
    InclinedDeckMotion,
    incline_preset,
)
from drone_landing.sim.platforms.ship import (
    ShipDeckMotion,
    ShipMotionConfig,
    WaveComponent,
    sea_state,
)
from drone_landing.sim.platforms.truck import TruckMotion, TruckMotionConfig
from drone_landing.sim.platforms.usv import USVMotion, USVMotionConfig
from drone_landing.sim.platforms.wave_spectrum import (
    SeaSpectrum,
    jonswap,
    pierson_moskowitz,
    significant_height,
    spectral_sea_state,
)

__all__ = [
    "PlatformMotion",
    "PlatformState",
    "yaw_to_quat",
    "GroundMotionConfig",
    "RandomGroundMotion",
    "ShipDeckMotion",
    "ShipMotionConfig",
    "WaveComponent",
    "sea_state",
    "DataDrivenDeckMotion",
    "InclinedDeckMotion",
    "InclinedDeckConfig",
    "incline_preset",
    "USVMotion",
    "USVMotionConfig",
    "TruckMotion",
    "TruckMotionConfig",
    "SeaSpectrum",
    "spectral_sea_state",
    "jonswap",
    "pierson_moskowitz",
    "significant_height",
]
