"""
Planck Units: Natural unit system derived from fundamental constants.

Planck units define the boundaries where quantum gravity effects dominate.
They represent the natural scales of the universe - where theta = 1 (fully quantum).

Key insight: Planck units are constructed from c, ℏ, G to eliminate arbitrary
human-scale units. At these scales, quantum and gravitational effects are
equally important.

References (see BIBLIOGRAPHY.bib):
    \\cite{CODATA2022} - NIST CODATA 2022 for base constants
    \\cite{Planck1901} - Original Planck constant derivation
    \\cite{Rovelli2004} - Planck scale in quantum gravity
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from .values import FundamentalConstants as FC


@dataclass
class PlanckUnit:
    """Represents a Planck unit with its value and interpretation."""
    name: str
    symbol: str
    value: float
    si_unit: str
    formula: str
    interpretation: str


class PlanckUnits:
    """
    Natural unit system derived from c, ℏ, and G.

    These define the absolute boundaries of physical description:
    - l_P: Smallest meaningful length (spacetime foam scale)
    - t_P: Smallest meaningful time (quantum gravity timescale)
    - m_P: Mass where quantum = gravitational effects
    - E_P: Maximum localized energy at Planck scale
    - T_P: Temperature where quantum gravity dominates thermodynamics

    At Planck scales, theta = 1 by definition (fully quantum).
    """

    # Cache computed values
    _cache: Dict[str, float] = {}

    @classmethod
    def _get_constants(cls):
        """Get constant values for calculations."""
        return FC.h_bar.value, FC.G.value, FC.c.value, FC.k.value

    @classmethod
    def planck_length(cls) -> float:
        """
        Planck length: l_P = √(ℏG/c³)

        The smallest meaningful length scale. Below this, spacetime itself
        exhibits quantum fluctuations. Classical geometry breaks down.

        l_P ≈ 1.616 × 10⁻³⁵ m

        Returns:
            Planck length in meters
        """
        if 'l_P' not in cls._cache:
            h_bar, G, c, _ = cls._get_constants()
            cls._cache['l_P'] = np.sqrt(h_bar * G / c**3)
        return cls._cache['l_P']

    @classmethod
    def planck_time(cls) -> float:
        """
        Planck time: t_P = √(ℏG/c⁵)

        The smallest meaningful time interval. The time for light to
        travel one Planck length. Below this, causality is undefined.

        t_P ≈ 5.391 × 10⁻⁴⁴ s

        Returns:
            Planck time in seconds
        """
        if 't_P' not in cls._cache:
            h_bar, G, c, _ = cls._get_constants()
            cls._cache['t_P'] = np.sqrt(h_bar * G / c**5)
        return cls._cache['t_P']

    @classmethod
    def planck_mass(cls) -> float:
        """
        Planck mass: m_P = √(ℏc/G)

        The mass scale where quantum wavelength equals Schwarzschild radius.
        A particle of this mass would be a quantum black hole.

        m_P ≈ 2.176 × 10⁻⁸ kg ≈ 21.76 micrograms

        This is surprisingly macroscopic compared to particle masses.

        Returns:
            Planck mass in kg
        """
        if 'm_P' not in cls._cache:
            h_bar, G, c, _ = cls._get_constants()
            cls._cache['m_P'] = np.sqrt(h_bar * c / G)
        return cls._cache['m_P']

    @classmethod
    def planck_energy(cls) -> float:
        """
        Planck energy: E_P = √(ℏc⁵/G) = m_P c²

        Maximum energy that can be localized at the Planck scale.
        Beyond this, a black hole forms.

        E_P ≈ 1.956 × 10⁹ J ≈ 1.22 × 10¹⁹ GeV

        This is enormous - about the energy in a lightning bolt
        concentrated to a point.

        Returns:
            Planck energy in Joules
        """
        if 'E_P' not in cls._cache:
            h_bar, G, c, _ = cls._get_constants()
            cls._cache['E_P'] = np.sqrt(h_bar * c**5 / G)
        return cls._cache['E_P']

    @classmethod
    def planck_temperature(cls) -> float:
        """
        Planck temperature: T_P = √(ℏc⁵/(Gk²)) = E_P / k

        The temperature where quantum gravity effects dominate
        thermodynamics. The "hottest meaningful temperature."

        T_P ≈ 1.417 × 10³² K

        Nothing in the universe today is this hot.
        The Big Bang may have approached this briefly.

        Returns:
            Planck temperature in Kelvin
        """
        if 'T_P' not in cls._cache:
            h_bar, G, c, k = cls._get_constants()
            cls._cache['T_P'] = np.sqrt(h_bar * c**5 / (G * k**2))
        return cls._cache['T_P']

    @classmethod
    def planck_density(cls) -> float:
        """
        Planck density: ρ_P = c⁵/(ℏG²) = m_P / l_P³

        Maximum density before spacetime curvature becomes singular.
        The density of a Planck-mass black hole.

        ρ_P ≈ 5.155 × 10⁹⁶ kg/m³

        For comparison, nuclear density is only ~10¹⁷ kg/m³.

        Returns:
            Planck density in kg/m³
        """
        if 'rho_P' not in cls._cache:
            h_bar, G, c, _ = cls._get_constants()
            cls._cache['rho_P'] = c**5 / (h_bar * G**2)
        return cls._cache['rho_P']

    @classmethod
    def planck_charge(cls) -> float:
        """
        Planck charge: q_P = √(4πε₀ℏc) = e/√α

        The natural unit of charge in Planck units.
        Related to elementary charge by the fine-structure constant.

        q_P ≈ 1.876 × 10⁻¹⁸ C

        Returns:
            Planck charge in Coulombs
        """
        if 'q_P' not in cls._cache:
            h_bar = FC.h_bar.value
            c = FC.c.value
            eps_0 = FC.epsilon_0.value
            cls._cache['q_P'] = np.sqrt(4 * np.pi * eps_0 * h_bar * c)
        return cls._cache['q_P']

    @classmethod
    def planck_momentum(cls) -> float:
        """
        Planck momentum: p_P = √(ℏc³/G) = m_P c

        Returns:
            Planck momentum in kg·m/s
        """
        if 'p_P' not in cls._cache:
            cls._cache['p_P'] = cls.planck_mass() * FC.c.value
        return cls._cache['p_P']

    @classmethod
    def planck_force(cls) -> float:
        """
        Planck force: F_P = c⁴/G

        The maximum force in nature. Interestingly, this is
        independent of ℏ - it's purely classical.

        F_P ≈ 1.21 × 10⁴⁴ N

        Returns:
            Planck force in Newtons
        """
        if 'F_P' not in cls._cache:
            c = FC.c.value
            G = FC.G.value
            cls._cache['F_P'] = c**4 / G
        return cls._cache['F_P']

    @classmethod
    def get_all(cls) -> Dict[str, float]:
        """Return all Planck units as a dictionary."""
        return {
            'l_P': cls.planck_length(),
            't_P': cls.planck_time(),
            'm_P': cls.planck_mass(),
            'E_P': cls.planck_energy(),
            'T_P': cls.planck_temperature(),
            'rho_P': cls.planck_density(),
            'q_P': cls.planck_charge(),
            'p_P': cls.planck_momentum(),
            'F_P': cls.planck_force(),
        }

    @classmethod
    def get_detailed(cls) -> Dict[str, PlanckUnit]:
        """Return all Planck units with full metadata."""
        return {
            'l_P': PlanckUnit(
                name="Planck length",
                symbol="l_P",
                value=cls.planck_length(),
                si_unit="m",
                formula="√(ℏG/c³)",
                interpretation="Minimum meaningful length; spacetime foam scale"
            ),
            't_P': PlanckUnit(
                name="Planck time",
                symbol="t_P",
                value=cls.planck_time(),
                si_unit="s",
                formula="√(ℏG/c⁵)",
                interpretation="Minimum meaningful time interval"
            ),
            'm_P': PlanckUnit(
                name="Planck mass",
                symbol="m_P",
                value=cls.planck_mass(),
                si_unit="kg",
                formula="√(ℏc/G)",
                interpretation="Mass where quantum wavelength = Schwarzschild radius"
            ),
            'E_P': PlanckUnit(
                name="Planck energy",
                symbol="E_P",
                value=cls.planck_energy(),
                si_unit="J",
                formula="√(ℏc⁵/G)",
                interpretation="Maximum localized energy at Planck scale"
            ),
            'T_P': PlanckUnit(
                name="Planck temperature",
                symbol="T_P",
                value=cls.planck_temperature(),
                si_unit="K",
                formula="√(ℏc⁵/(Gk²))",
                interpretation="Temperature where quantum gravity dominates"
            ),
        }

    @classmethod
    def in_planck_units(cls, value: float, unit_type: str) -> float:
        """
        Convert an SI value to Planck units.

        Args:
            value: The value in SI units
            unit_type: One of 'length', 'time', 'mass', 'energy', 'temperature'

        Returns:
            The value in Planck units (dimensionless ratio to Planck scale)

        Example:
            >>> PlanckUnits.in_planck_units(1.0, 'length')  # 1 meter in Planck lengths
            6.19e+34  # meters / l_P
        """
        converters = {
            'length': cls.planck_length(),
            'time': cls.planck_time(),
            'mass': cls.planck_mass(),
            'energy': cls.planck_energy(),
            'temperature': cls.planck_temperature(),
            'density': cls.planck_density(),
            'charge': cls.planck_charge(),
            'momentum': cls.planck_momentum(),
            'force': cls.planck_force(),
        }

        if unit_type not in converters:
            raise ValueError(
                f"Unknown unit type: {unit_type}. "
                f"Choose from: {list(converters.keys())}"
            )

        return value / converters[unit_type]

    @classmethod
    def from_planck_units(cls, value: float, unit_type: str) -> float:
        """
        Convert a Planck unit value to SI units.

        Args:
            value: The value in Planck units (dimensionless)
            unit_type: One of 'length', 'time', 'mass', 'energy', 'temperature'

        Returns:
            The value in SI units
        """
        converters = {
            'length': cls.planck_length(),
            'time': cls.planck_time(),
            'mass': cls.planck_mass(),
            'energy': cls.planck_energy(),
            'temperature': cls.planck_temperature(),
        }

        if unit_type not in converters:
            raise ValueError(f"Unknown unit type: {unit_type}")

        return value * converters[unit_type]

    @classmethod
    def verify_relationships(cls) -> Dict[str, bool]:
        """
        Verify internal consistency of Planck unit relationships.

        Tests:
        1. l_P / t_P = c
        2. E_P = m_P * c²
        3. T_P = E_P / k

        Returns:
            Dict mapping relationship name to whether it holds.
        """
        c = FC.c.value
        k = FC.k.value

        results = {}

        # l_P / t_P should equal c
        results['l_P/t_P = c'] = np.isclose(
            cls.planck_length() / cls.planck_time(),
            c,
            rtol=1e-10
        )

        # E_P should equal m_P * c^2
        results['E_P = m_P*c²'] = np.isclose(
            cls.planck_energy(),
            cls.planck_mass() * c**2,
            rtol=1e-10
        )

        # T_P should equal E_P / k
        results['T_P = E_P/k'] = np.isclose(
            cls.planck_temperature(),
            cls.planck_energy() / k,
            rtol=1e-10
        )

        return results

    @classmethod
    def print_all(cls) -> None:
        """Print all Planck units with descriptions."""
        print("=" * 70)
        print("PLANCK UNITS (Natural Scales of the Universe)")
        print("=" * 70)

        for name, unit in cls.get_detailed().items():
            print(f"\n{unit.name} ({unit.symbol})")
            print(f"  Value: {unit.value:.4e} {unit.si_unit}")
            print(f"  Formula: {unit.formula}")
            print(f"  Meaning: {unit.interpretation}")
