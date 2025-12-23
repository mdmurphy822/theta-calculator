r"""
Hawking Radiation Proofs: Power, Page Time, and Area Quantization

This module implements theta derivations from black hole thermodynamics
and Hawking radiation.

Key Insight: Black holes are quantum objects with temperature and entropy.
The degree of quantum behavior depends on:
- Size: Small black holes are more quantum (higher temperature)
- Age: Near Page time, information recovery becomes important
- Area: Quantized in units of Planck area

Theta Mapping:
- theta ~ 0: Large/classical black holes (T_H << T_CMB)
- theta ~ 1: Planck-scale black holes (T_H ~ T_Planck)

References (see BIBLIOGRAPHY.bib):
    \cite{Einstein1916GR} - General Relativity foundation
    \cite{MisnerThorneWheeler1973} - Gravitation textbook
    \cite{Hawking1975} - Particle creation by black holes
    \cite{Page1993} - Information in black hole radiation
    \cite{Bekenstein1998} - Quantum black holes as atoms
    \cite{Ashtekar2004} - Loop quantum gravity black holes
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

# Physical constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]
C = 2.99792458e8  # Speed of light [m/s]
G = 6.67430e-11  # Gravitational constant [m³/kg/s²]
K_B = 1.380649e-23  # Boltzmann constant [J/K]

# Derived Planck units
L_PLANCK = np.sqrt(HBAR * G / C**3)  # Planck length [m]
M_PLANCK = np.sqrt(HBAR * C / G)  # Planck mass [kg]
T_PLANCK = np.sqrt(HBAR * C**5 / (G * K_B**2))  # Planck temperature [K]
T_P = M_PLANCK * C**2 / K_B  # Planck temperature [K]

# Immirzi parameter (from Loop Quantum Gravity)
IMMIRZI_GAMMA = 0.2375  # Ashtekar-Barbero-Immirzi parameter


class BlackHoleType(Enum):
    """Types of black holes for analysis."""
    SCHWARZSCHILD = "schwarzschild"  # Non-rotating, uncharged
    KERR = "kerr"  # Rotating
    REISSNER_NORDSTROM = "reissner_nordstrom"  # Charged
    KERR_NEWMAN = "kerr_newman"  # Rotating and charged
    PRIMORDIAL = "primordial"  # Early universe
    MICROSCOPIC = "microscopic"  # Quantum scale


@dataclass
class HawkingRadiationResult:
    r"""
    Result of Hawking radiation analysis.

    Black holes emit thermal radiation with temperature:
    T_H = ℏc³ / (8πGMk_B)

    And power:
    P = ℏc⁶ / (15360πG²M²)  [Stefan-Boltzmann for black body]

    Attributes:
        mass: Black hole mass [kg]
        radius: Schwarzschild radius [m]
        temperature: Hawking temperature [K]
        power: Radiation power [W]
        evaporation_time: Time to evaporate [s]
        theta: Quantum-classical interpolation parameter

    Reference: \cite{Hawking1975}
    """
    mass: float
    radius: float
    temperature: float
    power: float
    evaporation_time: float
    theta: float

    @property
    def is_quantum(self) -> bool:
        """True if T_H >> T_CMB (quantum dominates)."""
        T_CMB = 2.725  # CMB temperature [K]
        return self.temperature > 10 * T_CMB


@dataclass
class PageTimeResult:
    r"""
    Result of Page time analysis.

    The Page time is when half the black hole's entropy has been radiated.
    At this point, information starts to be released from the black hole.

    t_Page ≈ 5120π G²M³ / (ℏc⁴)

    After Page time: theta increases as information is recovered.

    Attributes:
        mass: Initial black hole mass [kg]
        page_time: Time to Page point [s]
        current_time: Current evolution time [s]
        scrambling_time: Time to scramble information [s]
        theta: Based on t/t_Page ratio

    Reference: \cite{Page1993}
    """
    mass: float
    page_time: float
    current_time: float
    scrambling_time: float
    information_fraction: float
    theta: float


@dataclass
class AreaQuantizationResult:
    r"""
    Result of area quantization analysis.

    In Loop Quantum Gravity, black hole area is quantized:
    A = 8πγ l_P² Σ √(j(j+1))

    Where j are spin quantum numbers and γ is the Immirzi parameter.

    The minimum area eigenvalue is:
    A_min = 4√3 πγ l_P² ≈ 5.17 l_P²

    Attributes:
        actual_area: Black hole horizon area [m²]
        planck_area: l_P² [m²]
        min_quantum_area: Minimum area eigenvalue [m²]
        area_quanta: Number of area quanta
        theta: A_min / A_actual

    Reference: \cite{Ashtekar2004}
    """
    actual_area: float
    planck_area: float
    min_quantum_area: float
    area_quanta: float
    theta: float


def schwarzschild_radius(mass: float) -> float:
    """Compute Schwarzschild radius r_s = 2GM/c²."""
    return 2 * G * mass / C**2


def hawking_temperature(mass: float) -> float:
    r"""
    Compute Hawking temperature.

    T_H = ℏc³ / (8πGMk_B)

    Reference: \cite{Hawking1975}
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")
    return HBAR * C**3 / (8 * np.pi * G * mass * K_B)


def hawking_power(mass: float) -> float:
    r"""
    Compute Hawking radiation power.

    P = ℏc⁶ / (15360πG²M²)

    This is the Stefan-Boltzmann law for a black body
    with area 4πr_s² and temperature T_H.

    Reference: \cite{Hawking1975}
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")
    return HBAR * C**6 / (15360 * np.pi * G**2 * mass**2)


def evaporation_time(mass: float) -> float:
    r"""
    Compute black hole evaporation time.

    t_evap = 5120π G²M³ / (ℏc⁴)

    Reference: \cite{Hawking1975}
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")
    return 5120 * np.pi * G**2 * mass**3 / (HBAR * C**4)


def compute_hawking_theta(mass: float) -> HawkingRadiationResult:
    r"""
    Compute theta from Hawking radiation properties.

    Theta = T_H / T_Planck = M_Planck / (8πM)

    Physical interpretation:
    - theta = 1: Planck-mass black hole (T_H = T_Planck)
    - theta → 0: Large black hole (T_H << T_Planck)

    Args:
        mass: Black hole mass in kg

    Returns:
        HawkingRadiationResult with theta and thermodynamic properties

    Reference: \cite{Hawking1975}
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")

    r_s = schwarzschild_radius(mass)
    T_H = hawking_temperature(mass)
    P = hawking_power(mass)
    t_evap = evaporation_time(mass)

    # Theta: ratio of Hawking temp to Planck temp
    # Equivalently: Planck mass to black hole mass
    theta = M_PLANCK / (8 * np.pi * mass)
    theta = np.clip(theta, 0.0, 1.0)

    return HawkingRadiationResult(
        mass=mass,
        radius=r_s,
        temperature=T_H,
        power=P,
        evaporation_time=t_evap,
        theta=theta
    )


def page_time(mass: float) -> float:
    r"""
    Compute Page time.

    t_Page = t_evap / 2 (approximately)

    More precisely, it's when half the entropy has been radiated.

    Reference: \cite{Page1993}
    """
    return evaporation_time(mass) / 2


def scrambling_time(mass: float) -> float:
    r"""
    Compute scrambling time.

    t_scramble = (r_s/c) * ln(S_BH) = (r_s/c) * ln(A/(4l_P²))

    This is the time for information to be scrambled across the horizon.

    Reference: \cite{HaydenPreskill2007}
    """
    r_s = schwarzschild_radius(mass)
    A = 4 * np.pi * r_s**2
    S_BH = A / (4 * L_PLANCK**2)
    return (r_s / C) * np.log(max(S_BH, 1))


def compute_page_time_theta(
    mass: float,
    current_time: float = 0.0
) -> PageTimeResult:
    r"""
    Compute theta based on Page time.

    Before Page time: Information is hidden (theta ~ 0)
    At Page time: Information starts emerging (theta ~ 0.5)
    After Page time: Information is recovered (theta → 1)

    Args:
        mass: Initial black hole mass [kg]
        current_time: Evolution time since formation [s]

    Returns:
        PageTimeResult with information analysis

    Reference: \cite{Page1993}
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")

    t_page = page_time(mass)
    t_scramble = scrambling_time(mass)
    t_evap = evaporation_time(mass)

    # Information fraction based on Page curve
    if current_time <= 0:
        info_fraction = 0.0
    elif current_time >= t_evap:
        info_fraction = 1.0
    elif current_time < t_page:
        # Before Page time: information is hidden
        info_fraction = 0.0
    else:
        # After Page time: information is released
        # Simplified linear model
        info_fraction = (current_time - t_page) / (t_evap - t_page)

    # Theta represents quantum information recovery
    theta = info_fraction

    return PageTimeResult(
        mass=mass,
        page_time=t_page,
        current_time=current_time,
        scrambling_time=t_scramble,
        information_fraction=info_fraction,
        theta=theta
    )


def minimum_area_eigenvalue() -> float:
    r"""
    Compute minimum area eigenvalue in LQG.

    A_min = 4√3 πγ l_P²

    Where γ ≈ 0.2375 (Immirzi parameter).

    Reference: \cite{Ashtekar2004}
    """
    return 4 * np.sqrt(3) * np.pi * IMMIRZI_GAMMA * L_PLANCK**2


def compute_area_quantization_theta(area: float) -> AreaQuantizationResult:
    r"""
    Compute theta from area quantization.

    Theta = A_min / A_actual

    Physical interpretation:
    - theta = 1: Minimum quantum black hole
    - theta → 0: Large classical black hole

    Args:
        area: Horizon area [m²]

    Returns:
        AreaQuantizationResult with quantization analysis

    Reference: \cite{Ashtekar2004}
    """
    if area <= 0:
        raise ValueError("Area must be positive")

    A_min = minimum_area_eigenvalue()
    A_planck = L_PLANCK**2

    # Number of area quanta (approximately)
    n_quanta = area / A_min

    # Theta: ratio of minimum to actual area
    theta = A_min / area
    theta = np.clip(theta, 0.0, 1.0)

    return AreaQuantizationResult(
        actual_area=area,
        planck_area=A_planck,
        min_quantum_area=A_min,
        area_quanta=n_quanta,
        theta=theta
    )


class HawkingProofs:
    r"""
    Unified interface for Hawking radiation theta calculations.

    Reference: \cite{Hawking1975}
    """

    @staticmethod
    def from_temperature(mass: float) -> Dict[str, Any]:
        """Compute theta from Hawking temperature."""
        result = compute_hawking_theta(mass)
        return {
            "theta": result.theta,
            "proof_type": "hawking_temperature",
            "temperature": result.temperature,
            "power": result.power,
            "evaporation_time": result.evaporation_time,
            "confidence": 0.95,
            "citation": "\\cite{Hawking1975}"
        }

    @staticmethod
    def from_page_time(mass: float, current_time: float) -> Dict[str, Any]:
        """Compute theta from Page time."""
        result = compute_page_time_theta(mass, current_time)
        return {
            "theta": result.theta,
            "proof_type": "page_time",
            "page_time": result.page_time,
            "information_fraction": result.information_fraction,
            "confidence": 0.85,  # Page curve is theoretical
            "citation": "\\cite{Page1993}"
        }

    @staticmethod
    def from_area_quantization(area: float) -> Dict[str, Any]:
        """Compute theta from LQG area quantization."""
        result = compute_area_quantization_theta(area)
        return {
            "theta": result.theta,
            "proof_type": "area_quantization",
            "area_quanta": result.area_quanta,
            "min_quantum_area": result.min_quantum_area,
            "confidence": 0.80,  # LQG is theoretical
            "citation": "\\cite{Ashtekar2004}"
        }


# =============================================================================
# EXAMPLE BLACK HOLES
# =============================================================================

# Solar masses to kg
M_SUN = 1.989e30  # kg

BLACKHOLE_EXAMPLES = {
    "planck_mass": {
        "description": "Planck-mass black hole",
        "mass": M_PLANCK,
        "expected_theta": 1.0 / (8 * np.pi),  # ~0.04
    },
    "primordial_small": {
        "description": "Small primordial BH (10¹¹ kg)",
        "mass": 1e11,
        "expected_theta": 0.0,  # Very small
    },
    "stellar": {
        "description": "Stellar black hole (10 M☉)",
        "mass": 10 * M_SUN,
        "expected_theta": 0.0,  # Essentially classical
    },
    "supermassive": {
        "description": "Supermassive BH (10⁶ M☉)",
        "mass": 1e6 * M_SUN,
        "expected_theta": 0.0,  # Very classical
    },
    "sagittarius_a": {
        "description": "Sagittarius A* (4×10⁶ M☉)",
        "mass": 4e6 * M_SUN,
        "expected_theta": 0.0,
    },
}


def hawking_theta_summary():
    """Print theta analysis for example black holes."""
    print("=" * 70)
    print("HAWKING RADIATION THETA ANALYSIS")
    print("=" * 70)
    print()
    print(f"{'Black Hole':<30} {'Mass [kg]':>12} {'T_H [K]':>12} {'θ':>8}")
    print("-" * 70)

    for name, bh in BLACKHOLE_EXAMPLES.items():
        result = compute_hawking_theta(bh["mass"])
        print(f"{bh['description']:<30} "
              f"{bh['mass']:.2e} "
              f"{result.temperature:.2e} "
              f"{result.theta:>8.2e}")

    print()
    print("Key: θ = M_Planck / (8πM)")
    print("     Only Planck-scale BHs have significant θ")


if __name__ == "__main__":
    hawking_theta_summary()
