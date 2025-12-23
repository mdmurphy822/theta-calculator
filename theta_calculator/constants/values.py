"""
Fundamental Physical Constants (CODATA 2022)

These constants define the boundary states between classical and quantum regimes.
They are not arbitrary but form a self-consistent, recursively interconnected system.

Key insight: Constants are operational thresholds that determine when different
physical descriptions become applicable.

References (see BIBLIOGRAPHY.bib):
    \\cite{CODATA2022} - NIST CODATA 2022 Recommended Values
    \\cite{Planck1901} - Planck constant original derivation
    \\cite{Einstein1905SR} - Special relativity (speed of light)
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class PhysicalConstant:
    """
    Represents a physical constant with uncertainty and metadata.

    Attributes:
        name: Full name of the constant
        symbol: Mathematical symbol
        value: Numerical value in SI units
        unit: SI unit string
        uncertainty: Measurement uncertainty (0 for exact values)
        description: Physical interpretation as a boundary state
    """
    name: str
    symbol: str
    value: float
    unit: str
    uncertainty: float
    description: str

    def __repr__(self) -> str:
        if self.uncertainty == 0:
            return f"{self.symbol} = {self.value:.10e} {self.unit} (exact)"
        return f"{self.symbol} = {self.value:.10e} +/- {self.uncertainty:.1e} {self.unit}"

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as a fraction."""
        if self.value == 0:
            return float('inf')
        return self.uncertainty / abs(self.value)


class FundamentalConstants:
    """
    CODATA 2022 fundamental physical constants.

    These constants define the boundaries between physical regimes:
    - c: Spacetime causality boundary (maximum information speed)
    - h_bar: Quantum action boundary (quantum/classical transition)
    - G: Curvature/mass boundary (gravity scale)
    - k: Thermal/statistical boundary (micro/macro bridge)
    - alpha: Electromagnetic coupling boundary (atomic structure)

    The 2019 SI redefinition made c, h, e, and k exact by definition.
    """

    # Speed of light in vacuum (exact by definition since 1983)
    c = PhysicalConstant(
        name="Speed of light in vacuum",
        symbol="c",
        value=299_792_458.0,  # m/s
        unit="m/s",
        uncertainty=0.0,
        description="Maximum speed of information transfer; spacetime causality boundary. "
                   "Connects space and time through c = dx/dt at the causal limit."
    )

    # Planck constant (exact by 2019 SI redefinition)
    h = PhysicalConstant(
        name="Planck constant",
        symbol="h",
        value=6.626_070_15e-34,  # J*s
        unit="J·s",
        uncertainty=0.0,
        description="Quantum of action. Defines the scale at which quantum effects "
                   "become significant. E = hf relates energy to frequency."
    )

    # Reduced Planck constant (derived from h)
    h_bar = PhysicalConstant(
        name="Reduced Planck constant",
        symbol="ℏ",
        value=1.054_571_817e-34,  # J*s (h / 2*pi)
        unit="J·s",
        uncertainty=0.0,
        description="Quantum action boundary. When action S ~ ℏ, quantum mechanics "
                   "dominates. When S >> ℏ, classical physics applies. This defines theta."
    )

    # Gravitational constant (least precisely known fundamental constant)
    G = PhysicalConstant(
        name="Newtonian constant of gravitation",
        symbol="G",
        value=6.674_30e-11,  # m^3/(kg*s^2)
        unit="m³/(kg·s²)",
        uncertainty=1.5e-15,
        description="Couples mass-energy to spacetime curvature. Least precisely known "
                   "constant due to gravity's weakness. Defines the Planck scale with c and ℏ."
    )

    # Boltzmann constant (exact by 2019 SI redefinition)
    k = PhysicalConstant(
        name="Boltzmann constant",
        symbol="k",
        value=1.380_649e-23,  # J/K
        unit="J/K",
        uncertainty=0.0,
        description="Bridges microscopic and macroscopic thermodynamics. Connects "
                   "temperature to energy: E_thermal = kT. Defines entropy scale S = k ln W."
    )

    # Fine-structure constant (dimensionless)
    alpha = PhysicalConstant(
        name="Fine-structure constant",
        symbol="α",
        value=7.297_352_5693e-3,  # dimensionless, approximately 1/137.036
        unit="dimensionless",
        uncertainty=1.1e-12,
        description="Electromagnetic coupling strength. Governs atomic structure and "
                   "all electromagnetic phenomena. α = e²/(4πε₀ℏc) ≈ 1/137. "
                   "Origin of this specific value is unknown - a key mystery."
    )

    # Elementary charge (exact by 2019 SI redefinition)
    e = PhysicalConstant(
        name="Elementary charge",
        symbol="e",
        value=1.602_176_634e-19,  # C
        unit="C",
        uncertainty=0.0,
        description="Fundamental unit of electric charge. All observed charges are "
                   "integer multiples of e (quarks have fractional charges e/3, 2e/3)."
    )

    # Vacuum permittivity (derived from c, mu_0)
    epsilon_0 = PhysicalConstant(
        name="Vacuum electric permittivity",
        symbol="ε₀",
        value=8.854_187_8128e-12,  # F/m
        unit="F/m",
        uncertainty=1.3e-21,
        description="Electric field permittivity in vacuum. Related to c by "
                   "c = 1/√(ε₀μ₀). Determines electric field propagation."
    )

    # Vacuum permeability (derived from c, epsilon_0)
    mu_0 = PhysicalConstant(
        name="Vacuum magnetic permeability",
        symbol="μ₀",
        value=1.256_637_062_12e-6,  # H/m (N/A^2)
        unit="H/m",
        uncertainty=1.9e-16,
        description="Magnetic field permeability in vacuum. μ₀ = 4π × 10⁻⁷ H/m "
                   "exactly in old SI. Now derived from c and ε₀."
    )

    # Electron mass
    m_e = PhysicalConstant(
        name="Electron mass",
        symbol="m_e",
        value=9.109_383_7015e-31,  # kg
        unit="kg",
        uncertainty=2.8e-40,
        description="Mass of the electron, the lightest charged lepton. "
                   "Defines the Compton wavelength λ_C = h/(m_e c)."
    )

    # Proton mass
    m_p = PhysicalConstant(
        name="Proton mass",
        symbol="m_p",
        value=1.672_621_923_69e-27,  # kg
        unit="kg",
        uncertainty=5.1e-37,
        description="Mass of the proton. m_p/m_e ≈ 1836.15, another unexplained ratio."
    )

    # Avogadro constant (exact by 2019 SI redefinition)
    N_A = PhysicalConstant(
        name="Avogadro constant",
        symbol="N_A",
        value=6.022_140_76e23,  # mol^-1
        unit="mol⁻¹",
        uncertainty=0.0,
        description="Number of entities in one mole. Bridges atomic and macroscopic scales."
    )

    @classmethod
    def get_all(cls) -> Dict[str, PhysicalConstant]:
        """Return all fundamental constants as a dictionary."""
        return {
            'c': cls.c,
            'h': cls.h,
            'h_bar': cls.h_bar,
            'G': cls.G,
            'k': cls.k,
            'alpha': cls.alpha,
            'e': cls.e,
            'epsilon_0': cls.epsilon_0,
            'mu_0': cls.mu_0,
            'm_e': cls.m_e,
            'm_p': cls.m_p,
            'N_A': cls.N_A,
        }

    @classmethod
    def get_exact(cls) -> Dict[str, PhysicalConstant]:
        """Return only constants that are exact by definition."""
        return {k: v for k, v in cls.get_all().items() if v.uncertainty == 0}

    @classmethod
    def print_all(cls) -> None:
        """Print all constants with their values and descriptions."""
        print("=" * 70)
        print("FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2022)")
        print("=" * 70)
        for name, const in cls.get_all().items():
            print(f"\n{const.name}")
            print(f"  {const}")
            print(f"  {const.description[:70]}...")


# Convenience accessors for direct value access
c = FundamentalConstants.c.value
h = FundamentalConstants.h.value
h_bar = FundamentalConstants.h_bar.value
G = FundamentalConstants.G.value
k_B = FundamentalConstants.k.value
alpha = FundamentalConstants.alpha.value
e = FundamentalConstants.e.value
epsilon_0 = FundamentalConstants.epsilon_0.value
mu_0 = FundamentalConstants.mu_0.value
m_e = FundamentalConstants.m_e.value
m_p = FundamentalConstants.m_p.value


def verify_relationships() -> Dict[str, bool]:
    """
    Verify internal consistency of constant relationships.

    Tests:
    1. c = 1/√(ε₀μ₀)
    2. α = e²/(4πε₀ℏc)
    3. ℏ = h/(2π)

    Returns:
        Dict mapping relationship name to whether it holds.
    """
    results = {}

    # Test c = 1/sqrt(epsilon_0 * mu_0)
    c_from_em = 1.0 / np.sqrt(epsilon_0 * mu_0)
    results['c_from_em'] = np.isclose(c, c_from_em, rtol=1e-9)

    # Test alpha = e^2 / (4*pi*epsilon_0*h_bar*c)
    alpha_computed = e**2 / (4 * np.pi * epsilon_0 * h_bar * c)
    results['alpha_consistency'] = np.isclose(alpha, alpha_computed, rtol=1e-9)

    # Test h_bar = h / (2*pi)
    h_bar_computed = h / (2 * np.pi)
    results['h_bar_from_h'] = np.isclose(h_bar, h_bar_computed, rtol=1e-10)

    return results
