"""
Black Hole Thermodynamics: Where all fundamental constants meet.

Black holes are unique objects that connect quantum mechanics (ℏ),
general relativity (G), thermodynamics (k), and special relativity (c)
in a single system. The Hawking temperature and Bekenstein-Hawking entropy
formulas contain ALL fundamental constants.

This makes black holes the ultimate test case for theta:
- Black holes saturate the Bekenstein bound → θ = 1 (maximally quantum)
- Black hole entropy is holographic (scales with area, not volume)
- Hawking radiation is a quantum effect at horizon scale

Black holes prove that theta = 1 at the information-theoretic limit.

References (see BIBLIOGRAPHY.bib):
    \\cite{Hawking1974} - Black hole explosions? (Hawking radiation prediction)
    \\cite{Hawking1975} - Particle creation by black holes (full derivation)
    \\cite{Bekenstein1973} - Black holes and entropy
    \\cite{Schwarzschild1916} - Schwarzschild metric and event horizon
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..constants.values import FundamentalConstants as FC
from ..constants.planck_units import PlanckUnits
from ..core.theta_state import ThetaState, PhysicalSystem


@dataclass
class BlackHoleProperties:
    """
    Thermodynamic properties of a black hole.

    Attributes:
        mass: Black hole mass (kg)
        schwarzschild_radius: Event horizon radius (m)
        hawking_temperature: Temperature from Hawking radiation (K)
        bekenstein_hawking_entropy: Entropy in J/K
        entropy_bits: Entropy in bits
        evaporation_time: Time to fully evaporate (s)
        luminosity: Hawking radiation power (W)
        area: Horizon area (m²)
        theta: Theta value (always 1 for black holes)
    """
    mass: float
    schwarzschild_radius: float
    hawking_temperature: float
    bekenstein_hawking_entropy: float
    entropy_bits: float
    evaporation_time: float
    luminosity: float
    area: float
    theta: float


class BlackHoleThermodynamics:
    """
    Calculate black hole thermodynamic properties.

    Black holes are the extreme case where theta = 1:
    - They saturate the Bekenstein entropy bound
    - Information is maximally compressed
    - Quantum effects (Hawking radiation) are essential

    Key formulas:
    - Schwarzschild radius: r_s = 2GM/c²
    - Hawking temperature: T_H = ℏc³/(8πGMk)
    - Bekenstein-Hawking entropy: S = kc³A/(4Gℏ)
    - Evaporation time: t_evap ∝ M³

    These formulas unify all fundamental constants, showing
    theta emerges at the deepest level of physics.
    """

    def __init__(self):
        self.c = FC.c.value
        self.G = FC.G.value
        self.h_bar = FC.h_bar.value
        self.k = FC.k.value

        # Stefan-Boltzmann constant for luminosity
        self.sigma = 5.67e-8  # W/(m²·K⁴)

    def schwarzschild_radius(self, mass: float) -> float:
        """
        Compute Schwarzschild radius. \\cite{Schwarzschild1916}

        r_s = 2GM/c²

        This is the event horizon radius for a non-rotating black hole.

        Args:
            mass: Black hole mass in kg

        Returns:
            Schwarzschild radius in meters
        """
        return 2 * self.G * mass / self.c**2

    def hawking_temperature(self, mass: float) -> float:
        """
        Compute Hawking temperature. \\cite{Hawking1974} \\cite{Hawking1975}

        T_H = ℏc³/(8πGMk)

        This remarkable formula contains ALL fundamental constants:
        - ℏ: quantum mechanics
        - c: special relativity
        - G: general relativity
        - k: thermodynamics

        The temperature is inversely proportional to mass:
        - Stellar black holes: ~10⁻⁸ K (colder than CMB)
        - Primordial black holes: can be hot

        Args:
            mass: Black hole mass in kg

        Returns:
            Hawking temperature in Kelvin
        """
        if mass <= 0:
            return float('inf')
        return self.h_bar * self.c**3 / (8 * np.pi * self.G * mass * self.k)

    def bekenstein_hawking_entropy(self, mass: float) -> float:
        """
        Compute Bekenstein-Hawking entropy. \\cite{Bekenstein1973}

        S = kc³A/(4Gℏ) = (A/4) × (1/l_P²) × k

        where A = 4πr_s² is the horizon area.

        This is the maximum entropy for a given mass.
        Entropy scales with AREA, not volume (holographic principle).

        Args:
            mass: Black hole mass in kg

        Returns:
            Entropy in J/K
        """
        r_s = self.schwarzschild_radius(mass)
        A = 4 * np.pi * r_s**2
        return self.k * self.c**3 * A / (4 * self.G * self.h_bar)

    def entropy_in_bits(self, mass: float) -> float:
        """
        Compute black hole entropy in bits.

        S_bits = S / (k ln 2) = A / (4 l_P² ln 2)

        One bit per 4 Planck areas.

        Args:
            mass: Black hole mass in kg

        Returns:
            Entropy in bits
        """
        S_joules_per_kelvin = self.bekenstein_hawking_entropy(mass)
        return S_joules_per_kelvin / (self.k * np.log(2))

    def evaporation_time(self, mass: float) -> float:
        """
        Compute black hole evaporation time.

        t_evap = 5120 π G² M³ / (ℏ c⁴)

        A solar-mass black hole would take ~10^67 years to evaporate.
        A 10^12 kg black hole (asteroid mass) evaporates in ~10^10 years.

        Args:
            mass: Black hole mass in kg

        Returns:
            Evaporation time in seconds
        """
        return 5120 * np.pi * self.G**2 * mass**3 / (self.h_bar * self.c**4)

    def hawking_luminosity(self, mass: float) -> float:
        """
        Compute Hawking radiation luminosity.

        L = ℏc⁶ / (15360 π G² M²)

        Power output from Hawking radiation.

        Args:
            mass: Black hole mass in kg

        Returns:
            Luminosity in Watts
        """
        if mass <= 0:
            return float('inf')
        return self.h_bar * self.c**6 / (15360 * np.pi * self.G**2 * mass**2)

    def compute_all_properties(self, mass: float) -> BlackHoleProperties:
        """
        Compute all thermodynamic properties for a black hole.

        Args:
            mass: Black hole mass in kg

        Returns:
            BlackHoleProperties with all computed values
        """
        r_s = self.schwarzschild_radius(mass)
        T_H = self.hawking_temperature(mass)
        S = self.bekenstein_hawking_entropy(mass)
        S_bits = self.entropy_in_bits(mass)
        t_evap = self.evaporation_time(mass)
        L = self.hawking_luminosity(mass)
        A = 4 * np.pi * r_s**2

        return BlackHoleProperties(
            mass=mass,
            schwarzschild_radius=r_s,
            hawking_temperature=T_H,
            bekenstein_hawking_entropy=S,
            entropy_bits=S_bits,
            evaporation_time=t_evap,
            luminosity=L,
            area=A,
            theta=1.0  # Black holes are maximally quantum
        )

    def mass_from_temperature(self, temperature: float) -> float:
        """
        Compute black hole mass from Hawking temperature.

        M = ℏc³/(8πGkT)

        Inverse of Hawking temperature formula.

        Args:
            temperature: Temperature in Kelvin

        Returns:
            Black hole mass in kg
        """
        if temperature <= 0:
            return float('inf')
        return self.h_bar * self.c**3 / (8 * np.pi * self.G * self.k * temperature)

    def minimum_black_hole_mass(self) -> float:
        """
        Compute minimum black hole mass (Planck mass).

        Below this mass, black hole evaporates instantly.
        This is the quantum gravity scale.

        Returns:
            Planck mass in kg
        """
        return PlanckUnits.planck_mass()

    def theta_for_black_hole(self, mass: float) -> ThetaState:
        """
        Create ThetaState for a black hole.

        Black holes always have θ = 1 because:
        1. They saturate the Bekenstein bound
        2. Hawking radiation is essentially quantum
        3. Information is maximally compressed

        Args:
            mass: Black hole mass in kg

        Returns:
            ThetaState with θ = 1
        """
        props = self.compute_all_properties(mass)

        system = PhysicalSystem(
            name=f"black_hole_{mass:.2e}kg",
            mass=mass,
            length_scale=props.schwarzschild_radius,
            energy=mass * self.c**2,
            temperature=props.hawking_temperature,
            entropy=props.bekenstein_hawking_entropy,
        )

        return ThetaState(
            theta=1.0,
            theta_uncertainty=0.0,
            system=system,
            proof_method="black_hole_thermodynamics",
            information_bits=props.entropy_bits,
            components={
                "bekenstein_saturation": 1.0,
                "hawking_temperature": props.hawking_temperature,
                "entropy_bits": props.entropy_bits,
            },
            confidence=1.0,
            validation_notes=(
                f"Black hole saturates Bekenstein bound. "
                f"T_H = {props.hawking_temperature:.2e} K, "
                f"S = {props.entropy_bits:.2e} bits, "
                f"θ = 1.0 (maximally quantum)"
            )
        )

    def information_paradox_summary(self) -> str:
        """
        Return summary of black hole information paradox.

        The paradox: What happens to information that falls into a black hole?
        This is directly related to theta - does theta remain 1 or go to 0?
        """
        return """
BLACK HOLE INFORMATION PARADOX AND THETA

The Problem:
- When matter falls into a black hole, information seems lost
- Hawking radiation appears thermal (no information content)
- This violates quantum mechanics (information must be conserved)

Theta Perspective:
- Black holes have θ = 1 (maximally quantum)
- Information must be preserved (θ cannot suddenly jump)
- Possible resolutions all involve theta:

1. Information in correlations: θ remains 1, info in subtle correlations
2. Remnants: θ = 1 remnant stores info
3. Firewall: θ changes discontinuously (problematic)
4. ER=EPR: Entanglement (θ = 1) creates wormholes

The resolution likely involves understanding how θ transitions
at the event horizon - the ultimate quantum-classical boundary.
"""


# Convenience functions
def stellar_black_hole(solar_masses: float = 10.0) -> BlackHoleProperties:
    """Create properties for a stellar black hole."""
    M_sun = 1.989e30  # kg
    bh = BlackHoleThermodynamics()
    return bh.compute_all_properties(solar_masses * M_sun)


def supermassive_black_hole(solar_masses: float = 1e9) -> BlackHoleProperties:
    """Create properties for a supermassive black hole (like Sagittarius A*)."""
    M_sun = 1.989e30  # kg
    bh = BlackHoleThermodynamics()
    return bh.compute_all_properties(solar_masses * M_sun)


def primordial_black_hole(mass_kg: float = 1e12) -> BlackHoleProperties:
    """Create properties for a primordial black hole."""
    bh = BlackHoleThermodynamics()
    return bh.compute_all_properties(mass_kg)
