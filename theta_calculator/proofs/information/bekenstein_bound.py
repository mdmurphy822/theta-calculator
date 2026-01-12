"""
Bekenstein Bound: Information-theoretic proof of theta.

The Bekenstein Bound sets the maximum entropy (information) that can be
contained in a spherical region of space with given radius and energy:

    S ≤ 2πkRE / (ℏc)

This is a profound connection between information theory, thermodynamics,
quantum mechanics, and gravity - all the constants that define theta.

Key insight: The Bekenstein bound defines the theta=1 (quantum) limit.
- A system at the Bekenstein bound is maximally quantum (like a black hole)
- A system far from the bound is classical

This module computes theta from how close a system is to saturating
the Bekenstein bound.

References (see BIBLIOGRAPHY.bib):
    \\cite{Bekenstein1981} - Universal upper bound on entropy-to-energy ratio
    \\cite{Bekenstein1973} - Black holes and entropy
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...constants.values import FundamentalConstants as FC
from ...core.theta_state import ThetaState, PhysicalSystem


@dataclass
class BekensteinResult:
    """
    Result of Bekenstein bound calculation.

    Attributes:
        entropy_bound: Maximum entropy in bits (Bekenstein bound)
        actual_entropy: Actual system entropy if known (bits)
        bound_saturation: How close to the bound (0 to 1)
        is_holographic: True if near saturation (black hole limit)
        theta_from_saturation: Theta derived from bound saturation
        radius: System radius used
        energy: System energy used
        explanation: Human-readable interpretation
    """
    entropy_bound: float
    actual_entropy: Optional[float]
    bound_saturation: float
    is_holographic: bool
    theta_from_saturation: float
    radius: float
    energy: float
    explanation: str


class BekensteinBound:
    """
    Calculates the Bekenstein entropy bound and derives theta from it.

    The Bekenstein Bound: S_max = 2πkRE / (ℏc)

    This is the maximum entropy that can be contained in a spherical
    region of radius R with total energy E. It emerges from:
    - Quantum mechanics (ℏ)
    - Relativity (c)
    - Thermodynamics (k)
    - Gravity (implicit in black hole saturation)

    Physical interpretation:
    - Information has a physical limit based on space and energy
    - Black holes saturate the bound (holographic principle)
    - Classical systems are far from saturation

    Theta interpretation:
    - θ → 1 when S_actual → S_bekenstein (quantum/holographic limit)
    - θ → 0 when S_actual << S_bekenstein (classical regime)
    """

    def __init__(self):
        self.c = FC.c.value
        self.h_bar = FC.h_bar.value
        self.k = FC.k.value
        self.G = FC.G.value

    def compute_bound_nats(self, radius: float, energy: float) -> float:
        """
        Compute the Bekenstein entropy bound in nats (dimensionless natural units).

        S_max = 2πRE / (ℏc)  [in nats, i.e., natural logarithm units]

        Note: The Bekenstein bound in SI units includes k_B: S = 2πkRE/(ℏc) [J/K].
        For dimensionless entropy in nats, we use S/k_B = 2πRE/(ℏc).
        To convert to bits, divide by ln(2): S_bits = S_nats / ln(2).

        Args:
            radius: System radius in meters
            energy: Total energy in Joules

        Returns:
            Maximum entropy in nats (dimensionless)
        """
        # Correct formula: S_nats = 2πRE/(ℏc) without k (dimensionless)
        return 2 * np.pi * radius * energy / (self.h_bar * self.c)

    def compute_bound_bits(self, radius: float, energy: float) -> float:
        """
        Compute the Bekenstein bound in bits.

        S_bits = S_nats / ln(2)

        This gives the maximum number of bits of information that
        can be stored in the region.

        Args:
            radius: System radius in meters
            energy: Total energy in Joules

        Returns:
            Maximum entropy in bits
        """
        S_nats = self.compute_bound_nats(radius, energy)
        # S_nats is now dimensionless, so just divide by ln(2) to get bits
        return S_nats / np.log(2)

    def schwarzschild_radius(self, mass: float) -> float:
        """
        Compute the Schwarzschild radius for a given mass.

        r_s = 2GM / c²

        This is the radius at which escape velocity equals c.
        A sphere of this radius with the given mass is a black hole.

        Args:
            mass: Mass in kg

        Returns:
            Schwarzschild radius in meters
        """
        return 2 * self.G * mass / self.c**2

    def black_hole_entropy_bits(self, mass: float) -> float:
        """
        Compute black hole entropy using Bekenstein-Hawking formula.

        S_BH = kc³A / (4Gℏ) = πkc³r_s² / (Gℏ)

        where A = 4πr_s² is the horizon area.

        This is the maximum entropy for a given mass - black holes
        saturate the Bekenstein bound.

        Args:
            mass: Black hole mass in kg

        Returns:
            Black hole entropy in bits
        """
        r_s = self.schwarzschild_radius(mass)
        A = 4 * np.pi * r_s**2  # Horizon area
        S_joules_per_kelvin = self.k * self.c**3 * A / (4 * self.G * self.h_bar)
        return S_joules_per_kelvin / (self.k * np.log(2))

    def estimate_thermal_entropy_nats(self, system: PhysicalSystem) -> float:
        """
        Estimate actual entropy of a thermal system in nats (dimensionless).

        For an ideal gas: S/k ≈ N(ln(V/N) + 3/2 ln(T) + const)

        We use a simpler estimate: S/k ≈ N for N particles,
        or S/k ≈ E/(kT) for thermal systems.

        Args:
            system: Physical system

        Returns:
            Estimated entropy in nats (dimensionless)
        """
        if system.entropy is not None:
            # Convert from J/K to nats by dividing by k
            return system.entropy / self.k

        if system.number_of_particles is not None:
            # Rough estimate: ~1 nat per particle
            return float(system.number_of_particles)

        if system.temperature > 0:
            # Thermal estimate: S/k ~ E/(kT)
            return system.energy / (self.k * system.temperature)

        return 0.0

    def compute_saturation(
        self,
        system: PhysicalSystem,
        actual_entropy: Optional[float] = None
    ) -> BekensteinResult:
        """
        Compute how close a system is to the Bekenstein bound.

        saturation = S_actual / S_bekenstein

        This saturation directly gives theta:
        - saturation = 1 → black hole (maximally quantum/holographic)
        - saturation → 0 → classical matter (far from bound)

        Args:
            system: Physical system to analyze
            actual_entropy: Actual entropy in J/K. If None, estimated.

        Returns:
            BekensteinResult with saturation and theta
        """
        radius = system.length_scale / 2  # Assume spherical, use radius

        # Use rest mass energy if energy seems like kinetic energy
        energy = max(system.energy, system.rest_energy)

        # Compute the Bekenstein bound (now returns dimensionless nats)
        S_max_nats = self.compute_bound_nats(radius, energy)
        S_max_bits = S_max_nats / np.log(2)  # Convert nats to bits

        # Get or estimate actual entropy (in nats)
        if actual_entropy is None:
            S_actual_nats = self.estimate_thermal_entropy_nats(system)
        else:
            # Convert from J/K to nats
            S_actual_nats = actual_entropy / self.k

        S_actual_bits = S_actual_nats / np.log(2)  # Convert nats to bits

        # Compute saturation (both in nats, so dimensionally consistent)
        if S_max_nats > 0:
            saturation = S_actual_nats / S_max_nats
            saturation = min(1.0, saturation)  # Cap at 1
        else:
            saturation = 0.0

        # Check if near black hole limit
        r_s = self.schwarzschild_radius(system.mass)
        is_holographic = radius <= 2 * r_s

        # Theta from saturation
        # Use smooth function: high saturation → high theta
        # θ = tanh(saturation × scale) for smooth transition
        theta = np.tanh(saturation * 5)  # Scale factor for sensitivity

        # If actually a black hole, theta = 1
        if is_holographic:
            theta = 1.0

        # Generate explanation
        if theta > 0.9:
            explanation = (
                f"System is near the Bekenstein bound (saturation={saturation:.2e}). "
                f"This indicates a maximally quantum/holographic state. "
                f"Information density approaches the black hole limit."
            )
        elif theta > 0.1:
            explanation = (
                f"System is in transition regime (saturation={saturation:.2e}). "
                f"Both quantum and classical descriptions contribute. "
                f"Information density is intermediate."
            )
        else:
            explanation = (
                f"System is far from Bekenstein bound (saturation={saturation:.2e}). "
                f"Classical description is adequate. "
                f"Information is spread diffusely, not concentrated."
            )

        return BekensteinResult(
            entropy_bound=S_max_bits,
            actual_entropy=S_actual_bits if actual_entropy is not None else None,
            bound_saturation=saturation,
            is_holographic=is_holographic,
            theta_from_saturation=theta,
            radius=radius,
            energy=energy,
            explanation=explanation
        )

    def theta_from_bekenstein(self, system: PhysicalSystem) -> ThetaState:
        """
        Create a ThetaState from Bekenstein bound analysis.

        This is the primary interface for using Bekenstein bound
        in the theta proof.

        Args:
            system: Physical system to analyze

        Returns:
            ThetaState with Bekenstein-derived theta
        """
        result = self.compute_saturation(system)

        return ThetaState(
            theta=result.theta_from_saturation,
            system=system,
            proof_method="bekenstein_bound",
            information_bits=result.entropy_bound,
            components={
                "bekenstein_saturation": result.bound_saturation,
                "is_holographic": float(result.is_holographic)
            },
            confidence=0.9 if result.bound_saturation > 0.1 else 0.7,
            validation_notes=result.explanation
        )

    def holographic_bits_per_area(self) -> float:
        """
        Compute the holographic information density.

        The holographic principle states that maximum information
        scales with area, not volume:

            I_max = A / (4 × l_P²) bits

        This gives about 10^66 bits per square meter.

        Returns:
            Bits per square meter at the holographic limit
        """
        l_P = np.sqrt(self.h_bar * self.G / self.c**3)
        return 1 / (4 * l_P**2 * np.log(2))

    def information_to_mass(self, bits: float, temperature: float) -> float:
        """
        Compute the minimum mass required to store given information.

        From Bekenstein bound and Landauer principle combined:
            M ≥ bits × k × T × ln(2) / c²

        Args:
            bits: Number of bits to store
            temperature: Temperature of the storage medium (K)

        Returns:
            Minimum mass in kg
        """
        return bits * self.k * temperature * np.log(2) / self.c**2


def bekenstein_bound_formula() -> str:
    """Return the Bekenstein bound formula as a string."""
    return "S ≤ 2πkRE/(ℏc)"


def bekenstein_hawking_formula() -> str:
    """Return the Bekenstein-Hawking black hole entropy formula."""
    return "S_BH = kc³A/(4Gℏ) = (A/4)×(c³/Gℏ)×k"
