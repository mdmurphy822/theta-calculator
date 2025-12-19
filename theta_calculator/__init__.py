"""
Theta Calculator: Proving Theta as the Quantum-Classical Gradient

This library computes and proves theta (θ) exists as the interpolation
parameter between quantum (θ=1) and classical (θ=0) states.

Example:
    from theta_calculator import UnifiedThetaProof, PhysicalSystem

    electron = PhysicalSystem(
        name="electron",
        mass=9.109e-31,
        length_scale=2.818e-15,
        energy=8.187e-14,
        temperature=300.0
    )

    proof = UnifiedThetaProof()
    result = proof.prove_theta_exists(electron)
    print(f"Theta: {result.theta:.4f}")
"""

from .core.theta_state import ThetaState, PhysicalSystem, Regime, ThetaTrajectory
from .core.interpolation import ThetaCalculator
from .constants.values import FundamentalConstants
from .constants.planck_units import PlanckUnits
from .proofs.unified import UnifiedThetaProof, UnifiedProofResult

__version__ = "0.1.0"
__author__ = "Theta Proof Project"

__all__ = [
    "ThetaState",
    "PhysicalSystem",
    "Regime",
    "ThetaTrajectory",
    "ThetaCalculator",
    "FundamentalConstants",
    "PlanckUnits",
    "UnifiedThetaProof",
    "UnifiedProofResult",
]
