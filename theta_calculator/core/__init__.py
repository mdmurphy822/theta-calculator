"""Core module: ThetaState, interpolation, and regime detection."""

from .theta_state import ThetaState, PhysicalSystem, Regime, ThetaTrajectory
from .interpolation import ThetaCalculator, theta_interpolation_function

__all__ = [
    "ThetaState",
    "PhysicalSystem",
    "Regime",
    "ThetaTrajectory",
    "ThetaCalculator",
    "theta_interpolation_function",
]
