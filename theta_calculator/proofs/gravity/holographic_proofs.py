"""
Holographic Entropy Proofs: Ryu-Takayanagi and Entanglement Wedge

This module implements theta derivations from the holographic principle
and AdS/CFT correspondence.

Key Insight: In holographic theories, entanglement entropy of a boundary
region equals the area of a minimal surface in the bulk:

S_A = Area(γ_A) / (4G_N)

This is the Ryu-Takayanagi formula, a quantum gravity result.

Theta Mapping:
- theta ~ 0: Classical gravity (large areas, classical geometry)
- theta ~ 1: Quantum gravity (Planck-scale areas, quantum geometry)

References (see BIBLIOGRAPHY.bib):
    \cite{RyuTakayanagi2006} - Holographic derivation of entanglement entropy
    \cite{Hubeny2007} - Covariant holographic entanglement entropy
    \cite{Maldacena1999} - AdS/CFT correspondence
    \cite{VanRaamsdonk2010} - Building up spacetime from entanglement
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

# Physical constants
HBAR = 1.054571817e-34  # [J·s]
C = 2.99792458e8  # [m/s]
G = 6.67430e-11  # [m³/kg/s²]
K_B = 1.380649e-23  # [J/K]

# Planck units
L_PLANCK = np.sqrt(HBAR * G / C**3)  # ~1.6e-35 m
A_PLANCK = L_PLANCK**2  # Planck area


class HolographicRegime(Enum):
    """Regimes of holographic systems."""
    CLASSICAL = "classical"  # Large N, weak coupling
    SEMICLASSICAL = "semiclassical"  # 1/N corrections
    QUANTUM = "quantum"  # Strong quantum effects


@dataclass
class RyuTakayanagiResult:
    """
    Result of Ryu-Takayanagi entropy analysis.

    The RT formula:
    S_A = Area(γ_A) / (4G_N)

    Where γ_A is the minimal surface homologous to boundary region A.

    Attributes:
        boundary_area: Area of boundary region [m²]
        minimal_surface_area: Area of RT surface [m²]
        entanglement_entropy: S_A [dimensionless, in Planck units]
        classical_entropy: Expected classical entropy
        theta: Quantum-classical interpolation

    Reference: \cite{RyuTakayanagi2006}
    """
    boundary_area: float
    minimal_surface_area: float
    entanglement_entropy: float
    classical_entropy: float
    theta: float
    ads_radius: Optional[float] = None


@dataclass
class EntanglementWedgeResult:
    """
    Result of entanglement wedge analysis.

    The entanglement wedge is the bulk region reconstructable
    from boundary region A. Its size indicates how much of
    the bulk is encoded in the boundary.

    Attributes:
        boundary_fraction: Fraction of boundary in region A
        wedge_volume: Volume of entanglement wedge [m³]
        bulk_volume: Total bulk volume [m³]
        wedge_fraction: V_wedge / V_bulk
        theta: Quantum information content

    Reference: \cite{Hubeny2007}
    """
    boundary_fraction: float
    wedge_volume: float
    bulk_volume: float
    wedge_fraction: float
    theta: float


def rt_entropy(area: float, G_N: float = G) -> float:
    """
    Compute Ryu-Takayanagi entanglement entropy.

    S = Area / (4G_N)

    In natural units with G_N = l_P²:
    S = Area / (4 l_P²)

    Reference: \cite{RyuTakayanagi2006}
    """
    return area / (4 * G_N)


def compute_rt_theta(
    minimal_surface_area: float,
    classical_entropy: Optional[float] = None,
    G_N: float = G
) -> RyuTakayanagiResult:
    """
    Compute theta from Ryu-Takayanagi formula.

    Theta measures quantum corrections to holographic entropy.

    Method 1: Area ratio
    theta = A_Planck / A_minimal (quantum if area is small)

    Method 2: Entropy ratio (if classical_entropy provided)
    theta = S_RT / S_classical (quantum if RT dominates)

    Args:
        minimal_surface_area: Area of RT surface [m²]
        classical_entropy: Classical thermodynamic entropy [bits]
        G_N: Newton's constant (for unit conversion)

    Returns:
        RyuTakayanagiResult with theta analysis

    Reference: \cite{RyuTakayanagi2006}
    """
    if minimal_surface_area <= 0:
        raise ValueError("Area must be positive")

    # RT entropy
    S_RT = rt_entropy(minimal_surface_area, G_N)

    # Theta method 1: Planck area ratio
    # Near Planck scale = quantum regime
    theta_area = A_PLANCK / minimal_surface_area

    # Theta method 2: Entropy ratio
    if classical_entropy is not None and classical_entropy > 0:
        # If RT entropy dominates, system is quantum
        theta_entropy = min(S_RT / classical_entropy, 1.0)
        theta = max(theta_area, theta_entropy)
    else:
        theta = theta_area

    theta = np.clip(theta, 0.0, 1.0)

    return RyuTakayanagiResult(
        boundary_area=0.0,  # Not specified
        minimal_surface_area=minimal_surface_area,
        entanglement_entropy=S_RT,
        classical_entropy=classical_entropy if classical_entropy else 0.0,
        theta=theta
    )


def compute_wedge_theta(
    boundary_fraction: float,
    ads_radius: float = 1.0,
    cutoff: float = 0.01
) -> EntanglementWedgeResult:
    """
    Compute theta from entanglement wedge geometry.

    In AdS/CFT, the entanglement wedge grows with boundary region size.
    The relationship encodes quantum information structure.

    For a boundary region of size l:
    - Small l: Wedge is small (theta ~ 0, local)
    - Large l (half space): Wedge is half bulk (theta ~ 1)

    Args:
        boundary_fraction: Fraction of boundary in region (0, 1)
        ads_radius: AdS radius in Planck units
        cutoff: UV cutoff (regularization)

    Returns:
        EntanglementWedgeResult with geometry analysis

    Reference: \cite{Hubeny2007}
    """
    if not 0 < boundary_fraction < 1:
        raise ValueError("Boundary fraction must be in (0, 1)")

    # Simplified model: wedge fraction scales with boundary fraction
    # In 2D CFT with bulk AdS3:
    # For interval of length l, RT surface depth ~ l/2

    # Wedge volume grows faster than linearly for small regions
    # Approaches bulk volume for half-space (f = 0.5)
    # Use continuous formula: sin²(π*f/2) which gives:
    #   f=0 → 0, f=0.5 → 0.5, f=1 → 1
    # This matches AdS3/CFT2 behavior and is continuous everywhere
    wedge_fraction = np.sin(np.pi * boundary_fraction / 2)**2

    wedge_fraction = np.clip(wedge_fraction, 0.0, 1.0)

    # Theta: how much of bulk is reconstructable
    # theta ~ 1 when wedge covers significant bulk
    theta = wedge_fraction

    # Volumes (relative units)
    bulk_volume = 4 * np.pi * ads_radius**3 / 3
    wedge_volume = wedge_fraction * bulk_volume

    return EntanglementWedgeResult(
        boundary_fraction=boundary_fraction,
        wedge_volume=wedge_volume,
        bulk_volume=bulk_volume,
        wedge_fraction=wedge_fraction,
        theta=theta
    )


def subregion_complexity(
    minimal_surface_area: float,
    boundary_volume: float,
    ads_radius: float = 1.0
) -> Dict[str, float]:
    """
    Compute holographic subregion complexity.

    Complexity = Volume(Wedge) / (G_N * l_AdS)

    Or action formulation:
    Complexity = Action(WdW patch) / (π * ℏ)

    Reference: \cite{Susskind2016}
    """
    # Volume-complexity duality
    # Simplified model
    wedge_volume = minimal_surface_area * ads_radius / 2

    complexity = wedge_volume / (G * ads_radius)

    return {
        "complexity": complexity,
        "volume": wedge_volume,
        "entropy_complexity_ratio": rt_entropy(minimal_surface_area) / max(complexity, 1e-100)
    }


def mutual_information_holographic(
    area_A: float,
    area_B: float,
    area_AB: float,
    G_N: float = G
) -> Dict[str, float]:
    """
    Compute holographic mutual information.

    I(A:B) = S_A + S_B - S_AB

    In holographic theories, mutual information is related to
    the correlation structure of spacetime.

    Reference: \cite{Headrick2010}
    """
    S_A = rt_entropy(area_A, G_N)
    S_B = rt_entropy(area_B, G_N)
    S_AB = rt_entropy(area_AB, G_N)

    I_AB = S_A + S_B - S_AB

    # Theta: quantum correlations
    # High mutual information = quantum correlated
    max_entropy = min(S_A, S_B)
    theta = I_AB / (2 * max_entropy) if max_entropy > 0 else 0.0
    theta = np.clip(theta, 0.0, 1.0)

    return {
        "entropy_A": S_A,
        "entropy_B": S_B,
        "entropy_AB": S_AB,
        "mutual_information": I_AB,
        "theta": theta
    }


class HolographicProofs:
    """
    Unified interface for holographic theta calculations.

    Reference: \cite{RyuTakayanagi2006}
    """

    @staticmethod
    def ryu_takayanagi(
        minimal_area: float,
        classical_entropy: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compute theta from RT formula."""
        result = compute_rt_theta(minimal_area, classical_entropy)
        return {
            "theta": result.theta,
            "proof_type": "ryu_takayanagi",
            "entanglement_entropy": result.entanglement_entropy,
            "minimal_area": result.minimal_surface_area,
            "confidence": 0.90,  # Well-established in AdS/CFT
            "citation": "\\cite{RyuTakayanagi2006}"
        }

    @staticmethod
    def entanglement_wedge(boundary_fraction: float) -> Dict[str, Any]:
        """Compute theta from entanglement wedge."""
        result = compute_wedge_theta(boundary_fraction)
        return {
            "theta": result.theta,
            "proof_type": "entanglement_wedge",
            "wedge_fraction": result.wedge_fraction,
            "boundary_fraction": result.boundary_fraction,
            "confidence": 0.85,
            "citation": "\\cite{Hubeny2007}"
        }

    @staticmethod
    def mutual_info(area_A: float, area_B: float, area_AB: float) -> Dict[str, Any]:
        """Compute theta from holographic mutual information."""
        result = mutual_information_holographic(area_A, area_B, area_AB)
        return {
            "theta": result["theta"],
            "proof_type": "holographic_mutual_information",
            "mutual_information": result["mutual_information"],
            "confidence": 0.85,
            "citation": "\\cite{Headrick2010}"
        }


# =============================================================================
# EXAMPLE HOLOGRAPHIC SYSTEMS
# =============================================================================

HOLOGRAPHIC_EXAMPLES = {
    "planck_surface": {
        "description": "Planck-area RT surface",
        "area": A_PLANCK,
        "expected_theta": 1.0,
    },
    "small_interval": {
        "description": "Small boundary interval (10%)",
        "boundary_fraction": 0.1,
        "expected_theta": 0.024,  # sin²(π*0.1/2) ≈ 0.024
    },
    "half_space": {
        "description": "Half-space (50%)",
        "boundary_fraction": 0.5,
        "expected_theta": 0.5,  # sin²(π/4) = 0.5
    },
    "large_interval": {
        "description": "Large boundary interval (90%)",
        "boundary_fraction": 0.9,
        "expected_theta": 0.976,  # sin²(π*0.9/2) ≈ 0.976
    },
    "black_hole_horizon": {
        "description": "Black hole horizon (10 M☉)",
        "area": 4 * np.pi * (2 * G * 10 * 1.989e30 / C**2)**2,
        "expected_theta": 0.0,  # Very classical
    },
}


def holographic_theta_summary():
    """Print theta analysis for example holographic systems."""
    print("=" * 70)
    print("HOLOGRAPHIC THETA ANALYSIS")
    print("=" * 70)
    print()
    print("Ryu-Takayanagi Examples:")
    print(f"{'System':<35} {'Area [m²]':>15} {'S_RT':>12} {'θ':>8}")
    print("-" * 70)

    for name, sys in HOLOGRAPHIC_EXAMPLES.items():
        if "area" in sys:
            result = compute_rt_theta(sys["area"])
            print(f"{sys['description']:<35} "
                  f"{sys['area']:.2e} "
                  f"{result.entanglement_entropy:.2e} "
                  f"{result.theta:>8.4f}")

    print()
    print("Entanglement Wedge Examples:")
    print(f"{'System':<35} {'Boundary %':>12} {'Wedge %':>12} {'θ':>8}")
    print("-" * 70)

    for name, sys in HOLOGRAPHIC_EXAMPLES.items():
        if "boundary_fraction" in sys:
            result = compute_wedge_theta(sys["boundary_fraction"])
            print(f"{sys['description']:<35} "
                  f"{sys['boundary_fraction']*100:>11.1f}% "
                  f"{result.wedge_fraction*100:>11.1f}% "
                  f"{result.theta:>8.4f}")


if __name__ == "__main__":
    holographic_theta_summary()
