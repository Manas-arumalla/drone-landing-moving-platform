"""Model Predictive Control: a CasADi optimal-control outer loop that plans to intercept the
platform's predicted future trajectory (predictive, no PD lag), feeding the geometric attitude
inner loop."""

from drone_landing.control.mpc.nmpc import HorizontalMPC, MPCConfig
from drone_landing.control.mpc.tube_mpc import TubeMPC, TubeMPCConfig

__all__ = ["HorizontalMPC", "MPCConfig", "TubeMPC", "TubeMPCConfig"]
