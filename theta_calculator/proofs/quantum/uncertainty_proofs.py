"""
Uncertainty Principle Proofs: Heisenberg, Energy-Time, and Entropic

This module implements theta derivations from quantum uncertainty relations.

Key Insight: Uncertainty relations set fundamental limits on measurement precision.
The degree to which a system saturates these bounds indicates quantum behavior.

Theta Mapping:
- theta ~ 0: Classical regime (uncertainties >> ℏ/2)
- theta ~ 1: Quantum regime (uncertainties ~ ℏ/2, saturating the bound)

Forms of Uncertainty:
1. Position-Momentum (Heisenberg): Δx·Δp ≥ ℏ/2
2. Energy-Time: ΔE·Δt ≥ ℏ/2
3. Entropic: H(X) + H(P) ≥ log(πeℏ)

References (see BIBLIOGRAPHY.bib):
    \cite{Heisenberg1927} - Über den anschaulichen Inhalt
    \cite{Robertson1929} - Generalized uncertainty principle
    \cite{Ozawa2003} - Universally valid reformulation
    \cite{Deutsch1983} - Entropic uncertainty relations
    \cite{MaassenUffink1988} - Generalized entropic uncertainty
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

# Physical constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]
PLANCK_CONSTANT = 6.62607015e-34  # Planck constant [J·s]


class UncertaintyType(Enum):
    """Types of uncertainty relations."""
    HEISENBERG = "heisenberg"        # Position-momentum
    ENERGY_TIME = "energy_time"       # Energy-time
    ENTROPIC = "entropic"             # Entropy-based
    NUMBER_PHASE = "number_phase"     # Photon number-phase
    ANGULAR = "angular"               # Angular momentum


@dataclass
class HeisenbergResult:
    """
    Result of Heisenberg uncertainty analysis.

    The position-momentum uncertainty relation:
    Δx · Δp ≥ ℏ/2

    Attributes:
        delta_x: Position uncertainty [m]
        delta_p: Momentum uncertainty [kg·m/s]
        uncertainty_product: Δx · Δp [J·s]
        minimum_uncertainty: ℏ/2 [J·s]
        saturation: How close to the bound (1 = saturated)
        theta: Quantum-classical interpolation parameter
        is_minimum_uncertainty: True if state saturates bound

    Reference: \cite{Heisenberg1927}
    """
    delta_x: float
    delta_p: float
    uncertainty_product: float
    minimum_uncertainty: float
    saturation: float
    theta: float
    is_minimum_uncertainty: bool

    def __post_init__(self):
        """Validate physical constraints."""
        if self.delta_x <= 0:
            raise ValueError("Position uncertainty must be positive")
        if self.delta_p <= 0:
            raise ValueError("Momentum uncertainty must be positive")
        if self.uncertainty_product < self.minimum_uncertainty * 0.999:
            raise ValueError("Uncertainty product violates Heisenberg bound")


@dataclass
class EnergyTimeResult:
    """
    Result of energy-time uncertainty analysis.

    The energy-time uncertainty relation:
    ΔE · Δt ≥ ℏ/2

    Note: Time is not an observable in QM. Δt represents:
    - Lifetime of excited state
    - Duration of measurement
    - Time for expectation value to change by ΔE

    Attributes:
        delta_E: Energy uncertainty [J]
        delta_t: Time uncertainty [s]
        uncertainty_product: ΔE · Δt [J·s]
        minimum_uncertainty: ℏ/2 [J·s]
        saturation: How close to the bound
        theta: Quantum-classical interpolation parameter

    Reference: \cite{MandelstamTamm1945}
    """
    delta_E: float
    delta_t: float
    uncertainty_product: float
    minimum_uncertainty: float
    saturation: float
    theta: float


@dataclass
class EntropicResult:
    """
    Result of entropic uncertainty analysis.

    The entropic uncertainty relation (Deutsch-Maassen-Uffink):
    H(X) + H(P) ≥ log(πeℏ)

    Where H(X) and H(P) are Shannon entropies of position and
    momentum probability distributions.

    Advantage: Works for arbitrary distributions, not just Gaussian.

    Attributes:
        entropy_x: Position entropy [bits]
        entropy_p: Momentum entropy [bits]
        entropy_sum: H(X) + H(P) [bits]
        entropy_bound: log(πeℏ) [bits]
        saturation: How close to the bound
        theta: Quantum-classical interpolation parameter

    Reference: \cite{MaassenUffink1988}
    """
    entropy_x: float
    entropy_p: float
    entropy_sum: float
    entropy_bound: float
    saturation: float
    theta: float


def compute_heisenberg_theta(
    delta_x: float,
    delta_p: float,
    normalize: bool = True
) -> HeisenbergResult:
    """
    Compute theta from Heisenberg position-momentum uncertainty.

    Theta = (ℏ/2) / (Δx · Δp)

    Physical interpretation:
    - theta = 1: Minimum uncertainty state (coherent state)
    - theta → 0: Classical regime (large uncertainties)

    Args:
        delta_x: Position uncertainty in meters
        delta_p: Momentum uncertainty in kg·m/s
        normalize: If True, ensure theta ∈ [0, 1]

    Returns:
        HeisenbergResult with theta and analysis

    Example:
        >>> # Electron in 1nm trap
        >>> result = compute_heisenberg_theta(1e-9, 1e-24)
        >>> print(f"theta = {result.theta:.3f}")

    Reference: \cite{Heisenberg1927}
    """
    if delta_x <= 0 or delta_p <= 0:
        raise ValueError("Uncertainties must be positive")

    uncertainty_product = delta_x * delta_p
    minimum_uncertainty = HBAR / 2

    # Check Heisenberg bound
    if uncertainty_product < minimum_uncertainty * 0.999:
        raise ValueError(
            f"Uncertainty product {uncertainty_product:.2e} J·s violates "
            f"Heisenberg bound {minimum_uncertainty:.2e} J·s"
        )

    # Theta: ratio of minimum to actual uncertainty
    # theta = 1 means saturating the bound (maximum quantum)
    # theta → 0 means far from bound (classical)
    theta = minimum_uncertainty / uncertainty_product

    if normalize:
        theta = np.clip(theta, 0.0, 1.0)

    saturation = theta  # Same as theta for this relation
    is_minimum = np.isclose(theta, 1.0, rtol=0.01)

    return HeisenbergResult(
        delta_x=delta_x,
        delta_p=delta_p,
        uncertainty_product=uncertainty_product,
        minimum_uncertainty=minimum_uncertainty,
        saturation=saturation,
        theta=theta,
        is_minimum_uncertainty=is_minimum
    )


def compute_energy_time_theta(
    delta_E: float,
    delta_t: float,
    normalize: bool = True
) -> EnergyTimeResult:
    """
    Compute theta from energy-time uncertainty.

    Theta = (ℏ/2) / (ΔE · Δt)

    Applications:
    - Excited state lifetime: ΔE = Γ (decay width), Δt = τ (lifetime)
    - Virtual particles: ΔE = mc², Δt = ℏ/(mc²)
    - Quantum tunneling: Time-energy tradeoff

    Args:
        delta_E: Energy uncertainty in Joules
        delta_t: Time uncertainty in seconds
        normalize: If True, ensure theta ∈ [0, 1]

    Returns:
        EnergyTimeResult with theta and analysis

    Example:
        >>> # Excited atomic state with 1ns lifetime
        >>> delta_E = HBAR / (2 * 1e-9)  # Natural linewidth
        >>> result = compute_energy_time_theta(delta_E, 1e-9)
        >>> print(f"theta = {result.theta:.3f}")  # Should be ~1

    Reference: \cite{MandelstamTamm1945}
    """
    if delta_E <= 0 or delta_t <= 0:
        raise ValueError("Uncertainties must be positive")

    uncertainty_product = delta_E * delta_t
    minimum_uncertainty = HBAR / 2

    theta = minimum_uncertainty / uncertainty_product

    if normalize:
        theta = np.clip(theta, 0.0, 1.0)

    saturation = theta

    return EnergyTimeResult(
        delta_E=delta_E,
        delta_t=delta_t,
        uncertainty_product=uncertainty_product,
        minimum_uncertainty=minimum_uncertainty,
        saturation=saturation,
        theta=theta
    )


def compute_entropic_theta(
    entropy_x: float,
    entropy_p: float,
    hbar_units: float = 1.0
) -> EntropicResult:
    """
    Compute theta from entropic uncertainty relation.

    H(X) + H(P) ≥ log(πeℏ)

    This is stronger than Heisenberg for non-Gaussian states.

    Args:
        entropy_x: Shannon entropy of position distribution [bits]
        entropy_p: Shannon entropy of momentum distribution [bits]
        hbar_units: Value of ℏ in chosen units (default: natural units)

    Returns:
        EntropicResult with theta and analysis

    Note:
        The bound log(πeℏ) depends on units. In natural units (ℏ=1):
        log(πe) ≈ 2.14 bits

    Reference: \cite{MaassenUffink1988}
    """
    entropy_sum = entropy_x + entropy_p

    # Bound in natural units
    entropy_bound = np.log2(np.pi * np.e * hbar_units)

    # Theta: how close to saturating the bound
    # High entropy_sum = classical (theta → 0)
    # entropy_sum = bound = maximally quantum (theta → 1)
    if entropy_sum <= entropy_bound:
        theta = 1.0  # At or violating bound (numerical issues)
    else:
        theta = entropy_bound / entropy_sum

    theta = np.clip(theta, 0.0, 1.0)
    saturation = theta

    return EntropicResult(
        entropy_x=entropy_x,
        entropy_p=entropy_p,
        entropy_sum=entropy_sum,
        entropy_bound=entropy_bound,
        saturation=saturation,
        theta=theta
    )


class UncertaintyProofs:
    """
    Unified interface for uncertainty-based theta calculations.

    Provides methods for computing theta from various uncertainty
    relations, with consistent output format.

    Reference: \cite{Ozawa2003}
    """

    @staticmethod
    def heisenberg(delta_x: float, delta_p: float) -> Dict[str, Any]:
        """
        Compute theta from Heisenberg uncertainty.

        Reference: \cite{Heisenberg1927}
        """
        result = compute_heisenberg_theta(delta_x, delta_p)
        return {
            "theta": result.theta,
            "proof_type": "heisenberg_uncertainty",
            "uncertainty_product": result.uncertainty_product,
            "bound": result.minimum_uncertainty,
            "saturation": result.saturation,
            "is_minimum_uncertainty": result.is_minimum_uncertainty,
            "confidence": min(0.99, result.saturation + 0.5),
            "citation": "\\cite{Heisenberg1927}"
        }

    @staticmethod
    def energy_time(delta_E: float, delta_t: float) -> Dict[str, Any]:
        """
        Compute theta from energy-time uncertainty.

        Reference: \cite{MandelstamTamm1945}
        """
        result = compute_energy_time_theta(delta_E, delta_t)
        return {
            "theta": result.theta,
            "proof_type": "energy_time_uncertainty",
            "uncertainty_product": result.uncertainty_product,
            "bound": result.minimum_uncertainty,
            "saturation": result.saturation,
            "confidence": min(0.99, result.saturation + 0.5),
            "citation": "\\cite{MandelstamTamm1945}"
        }

    @staticmethod
    def entropic(entropy_x: float, entropy_p: float) -> Dict[str, Any]:
        """
        Compute theta from entropic uncertainty.

        Reference: \cite{MaassenUffink1988}
        """
        result = compute_entropic_theta(entropy_x, entropy_p)
        return {
            "theta": result.theta,
            "proof_type": "entropic_uncertainty",
            "entropy_sum": result.entropy_sum,
            "bound": result.entropy_bound,
            "saturation": result.saturation,
            "confidence": min(0.99, result.saturation + 0.5),
            "citation": "\\cite{MaassenUffink1988}"
        }

    @staticmethod
    def from_wavefunction(
        psi_x: np.ndarray,
        x_grid: np.ndarray,
        mass: float
    ) -> Dict[str, Any]:
        """
        Compute theta directly from a wavefunction.

        Calculates position and momentum uncertainties from |ψ(x)|²
        and its Fourier transform.

        Args:
            psi_x: Wavefunction values ψ(x)
            x_grid: Position grid points [m]
            mass: Particle mass [kg]

        Returns:
            Dictionary with theta and uncertainty analysis
        """
        # Normalize
        dx = x_grid[1] - x_grid[0]
        norm = np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
        psi_x = psi_x / norm

        # Position probability
        prob_x = np.abs(psi_x)**2

        # Position moments
        x_mean = np.sum(x_grid * prob_x) * dx
        x2_mean = np.sum(x_grid**2 * prob_x) * dx
        delta_x = np.sqrt(x2_mean - x_mean**2)

        # Momentum space via FFT
        psi_p = np.fft.fft(psi_x) * dx
        p_grid = np.fft.fftfreq(len(x_grid), dx) * 2 * np.pi * HBAR

        # Momentum probability
        prob_p = np.abs(psi_p)**2
        prob_p = prob_p / (np.sum(prob_p) * (p_grid[1] - p_grid[0]))

        # Momentum moments
        dp = p_grid[1] - p_grid[0] if len(p_grid) > 1 else 1.0
        p_mean = np.sum(p_grid * prob_p) * dp
        p2_mean = np.sum(p_grid**2 * prob_p) * dp
        delta_p = np.sqrt(max(0, p2_mean - p_mean**2))

        if delta_p == 0:
            delta_p = HBAR / (2 * delta_x)  # Minimum uncertainty

        return UncertaintyProofs.heisenberg(delta_x, delta_p)


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

UNCERTAINTY_EXAMPLES = {
    "coherent_state": {
        "description": "Coherent state (minimum uncertainty)",
        "delta_x": 1e-11,  # 10 pm
        "delta_p": HBAR / (2 * 1e-11),  # Saturates bound
        "expected_theta": 1.0,
    },
    "electron_atom": {
        "description": "Electron in hydrogen atom",
        "delta_x": 5.29e-11,  # Bohr radius
        "delta_p": 1.99e-24,  # ~ℏ/a_0
        "expected_theta": 0.5,  # Approximate
    },
    "thermal_particle": {
        "description": "Thermal particle at 300K",
        "delta_x": 1e-9,  # 1 nm (de Broglie wavelength scale)
        "delta_p": 1e-23,  # ~sqrt(2mkT)
        "expected_theta": 0.005,  # Very classical
    },
    "quantum_dot": {
        "description": "Electron in 10nm quantum dot",
        "delta_x": 1e-8,  # 10 nm
        "delta_p": 6.6e-26,  # Confinement momentum
        "expected_theta": 0.08,
    },
}


def uncertainty_theta_summary():
    """Print theta analysis for example uncertainty systems."""
    print("=" * 70)
    print("UNCERTAINTY PRINCIPLE THETA ANALYSIS")
    print("=" * 70)
    print()
    print(f"{'System':<30} {'Δx':>12} {'Δp':>12} {'θ':>8}")
    print("-" * 70)

    for name, system in UNCERTAINTY_EXAMPLES.items():
        result = compute_heisenberg_theta(system["delta_x"], system["delta_p"])
        print(f"{system['description']:<30} "
              f"{system['delta_x']:.2e} "
              f"{system['delta_p']:.2e} "
              f"{result.theta:>8.4f}")

    print()
    print("Key: θ = 1 means minimum uncertainty (maximum quantum)")
    print("     θ → 0 means large uncertainties (classical)")


if __name__ == "__main__":
    uncertainty_theta_summary()
