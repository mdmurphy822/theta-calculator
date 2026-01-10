r"""
Chemistry Domain: Superconductivity, BEC, and Quantum Materials

This module implements theta as the quantum coherence parameter
for chemical and material systems.

Key Insight: Materials exhibit quantum-classical transitions:
- theta ~ 0: Classical/thermal behavior (T >> T_c)
- theta ~ 1: Quantum coherent state (T << T_c)

Theta Maps To:
1. Superconductivity: T_c / T (BCS gap parameter)
2. Bose-Einstein Condensate: N₀ / N (condensate fraction)
3. Quantum dots: E_confinement / E_thermal
4. Superfluidity: λ-transition behavior

References (see BIBLIOGRAPHY.bib):
    \cite{BCS1957} - Theory of superconductivity
    \cite{EinsteinBose1924} - Bose-Einstein condensation
    \cite{Alivisatos1996} - Semiconductor quantum dots
    \cite{Leggett2006} - Quantum Liquids
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


# Physical constants
K_B = 1.380649e-23  # Boltzmann constant [J/K]
HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]
M_ELECTRON = 9.10938e-31  # Electron mass [kg]
E_CHARGE = 1.602176634e-19  # Electron charge [C]


class MaterialPhase(Enum):
    """Material phase based on theta."""
    NORMAL = "normal"              # Classical behavior
    NEAR_TRANSITION = "near_transition"  # Close to T_c
    QUANTUM = "quantum"            # Quantum coherent


class SuperconductorType(Enum):
    """Types of superconductors."""
    TYPE_I = "type_I"              # Complete Meissner effect
    TYPE_II = "type_II"            # Vortex lattice
    HIGH_TC = "high_Tc"            # Cuprate, etc.
    CONVENTIONAL = "conventional"   # BCS type


@dataclass
class QuantumMaterial:
    """
    A quantum material for theta analysis.

    Attributes:
        name: Material identifier
        critical_temperature: Phase transition temperature [K]
        current_temperature: Operating temperature [K]
        gap_energy: Energy gap [eV]
        coherence_length: Coherence length [m]
        penetration_depth: London penetration depth [m]
    """
    name: str
    critical_temperature: float
    current_temperature: float
    gap_energy: Optional[float] = None
    coherence_length: Optional[float] = None
    penetration_depth: Optional[float] = None


# =============================================================================
# SUPERCONDUCTIVITY (BCS Theory)
# =============================================================================

def bcs_gap_temperature(T: float, T_c: float, Delta_0: float) -> float:
    r"""
    Compute BCS gap as function of temperature.

    Δ(T) ≈ Δ₀ × tanh(1.74√(T_c/T - 1)) for T < T_c

    BCS relation: Δ₀ = 1.76 k_B T_c

    Args:
        T: Temperature [K]
        T_c: Critical temperature [K]
        Delta_0: Zero-temperature gap [J]

    Returns:
        Energy gap Δ(T) [J]

    Reference: \cite{BCS1957}
    """
    if T >= T_c:
        return 0.0
    if T <= 0:
        return Delta_0

    ratio = T_c / T - 1
    if ratio <= 0:
        return 0.0

    return Delta_0 * np.tanh(1.74 * np.sqrt(ratio))


def bcs_gap_zero_temp(T_c: float) -> float:
    r"""
    Compute zero-temperature BCS gap.

    Δ₀ = 1.76 k_B T_c

    Args:
        T_c: Critical temperature [K]

    Returns:
        Zero-temperature gap [J]

    Reference: \cite{BCS1957}
    """
    return 1.76 * K_B * T_c


def compute_superconductor_theta(
    T: float,
    T_c: float
) -> float:
    r"""
    Compute theta for superconductor.

    Theta = T_c / T for T > T_c (normal state)
    Theta = 1 for T < T_c (superconducting)

    More precisely: theta = Δ(T) / Δ₀

    Args:
        T: Current temperature [K]
        T_c: Critical temperature [K]

    Returns:
        theta in [0, 1]

    Reference: \cite{BCS1957}
    """
    if T <= 0:
        return 1.0

    if T >= T_c:
        # Normal state: theta decreases with T/T_c
        theta = T_c / T
    else:
        # Superconducting: gap ratio
        Delta_0 = bcs_gap_zero_temp(T_c)
        Delta_T = bcs_gap_temperature(T, T_c, Delta_0)
        theta = Delta_T / Delta_0

    return np.clip(theta, 0.0, 1.0)


def ginzburg_landau_kappa(
    coherence_length: float,
    penetration_depth: float
) -> float:
    """
    Compute Ginzburg-Landau parameter κ.

    κ = λ / ξ

    κ < 1/√2: Type I superconductor
    κ > 1/√2: Type II superconductor

    Args:
        coherence_length: ξ [m]
        penetration_depth: λ [m]

    Returns:
        GL parameter κ
    """
    if coherence_length <= 0:
        return float('inf')
    return penetration_depth / coherence_length


def classify_superconductor(kappa: float) -> SuperconductorType:
    """Classify superconductor type from κ."""
    threshold = 1 / np.sqrt(2)
    if kappa < threshold:
        return SuperconductorType.TYPE_I
    else:
        return SuperconductorType.TYPE_II


# =============================================================================
# BOSE-EINSTEIN CONDENSATION
# =============================================================================

def bec_critical_temperature(
    n: float,
    mass: float
) -> float:
    r"""
    Compute BEC critical temperature.

    T_c = (2πℏ²/mk_B) × (n/ζ(3/2))^(2/3)

    Where ζ(3/2) ≈ 2.612

    Args:
        n: Particle density [m⁻³]
        mass: Particle mass [kg]

    Returns:
        Critical temperature [K]

    Reference: \cite{EinsteinBose1924}
    """
    zeta_3_2 = 2.612
    prefactor = (2 * np.pi * HBAR**2) / (mass * K_B)
    return prefactor * (n / zeta_3_2) ** (2/3)


def bec_condensate_fraction(
    T: float,
    T_c: float
) -> float:
    r"""
    Compute BEC condensate fraction.

    N₀/N = 1 - (T/T_c)^(3/2) for T < T_c
    N₀/N = 0 for T ≥ T_c

    Args:
        T: Temperature [K]
        T_c: Critical temperature [K]

    Returns:
        Condensate fraction [0, 1]

    Reference: \cite{EinsteinBose1924}
    """
    if T >= T_c:
        return 0.0
    if T <= 0:
        return 1.0

    return 1 - (T / T_c) ** 1.5


def compute_bec_theta(T: float, T_c: float) -> float:
    r"""
    Compute theta for BEC.

    Theta = condensate fraction = N₀/N

    Args:
        T: Temperature [K]
        T_c: Critical temperature [K]

    Returns:
        theta in [0, 1]

    Reference: \cite{EinsteinBose1924}
    """
    return bec_condensate_fraction(T, T_c)


# =============================================================================
# QUANTUM DOTS
# =============================================================================

def quantum_dot_confinement_energy(
    size: float,
    mass: float = M_ELECTRON
) -> float:
    r"""
    Compute quantum confinement energy in quantum dot.

    E_conf = ℏ²π² / (2mL²)

    For electrons in InAs dot, m* ≈ 0.023 m_e

    Args:
        size: Dot size [m]
        mass: Effective mass [kg]

    Returns:
        Confinement energy [J]

    Reference: \cite{Alivisatos1996}
    """
    return (HBAR * np.pi) ** 2 / (2 * mass * size ** 2)


def compute_quantum_dot_theta(
    size: float,
    temperature: float,
    effective_mass: float = 0.023 * M_ELECTRON
) -> float:
    r"""
    Compute theta for quantum dot.

    Theta = E_confinement / E_thermal
          = E_conf / (k_B T)

    High theta: Quantum confinement dominates (discrete levels)
    Low theta: Thermal excitation dominates (classical)

    Args:
        size: Quantum dot size [m]
        temperature: Temperature [K]
        effective_mass: Effective mass [kg]

    Returns:
        theta (can be > 1 for strong confinement)

    Reference: \cite{Alivisatos1996}
    """
    if temperature <= 0:
        return 1.0

    E_conf = quantum_dot_confinement_energy(size, effective_mass)
    E_thermal = K_B * temperature

    theta = E_conf / E_thermal

    # Normalize to [0, 1] using typical values
    # Strong confinement: E_conf >> k_B T, theta ~ 1
    # Weak confinement: E_conf << k_B T, theta ~ 0
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# SUPERFLUIDITY
# =============================================================================

def lambda_transition_exponent(T: float, T_lambda: float) -> float:
    r"""
    Compute superfluid fraction near lambda transition.

    Helium-4 superfluid transition at T_λ = 2.17 K.

    ρ_s/ρ ~ |T - T_λ|^ν with ν ≈ 0.67 (3D XY universality)

    Args:
        T: Temperature [K]
        T_lambda: Lambda temperature [K]

    Returns:
        Superfluid fraction

    Reference: \cite{Leggett2006}
    """
    if T >= T_lambda:
        return 0.0
    if T <= 0:
        return 1.0

    nu = 0.67  # 3D XY critical exponent
    t = (T_lambda - T) / T_lambda
    return t ** nu


def compute_superfluid_theta(T: float, T_lambda: float = 2.17) -> float:
    r"""
    Compute theta for superfluid helium.

    Theta = superfluid fraction ρ_s/ρ

    Args:
        T: Temperature [K]
        T_lambda: Lambda temperature [K]

    Returns:
        theta in [0, 1]

    Reference: \cite{Leggett2006}
    """
    return lambda_transition_exponent(T, T_lambda)


# =============================================================================
# UNIFIED CALCULATIONS
# =============================================================================

def compute_chemistry_theta(material: QuantumMaterial) -> float:
    """
    Compute unified theta for quantum material.

    Args:
        material: QuantumMaterial to analyze

    Returns:
        theta in [0, 1]
    """
    return compute_superconductor_theta(
        material.current_temperature,
        material.critical_temperature
    )


def classify_material_phase(theta: float) -> MaterialPhase:
    """Classify material phase from theta."""
    if theta < 0.3:
        return MaterialPhase.NORMAL
    elif theta < 0.8:
        return MaterialPhase.NEAR_TRANSITION
    else:
        return MaterialPhase.QUANTUM


# =============================================================================
# EXAMPLE MATERIALS
# =============================================================================

SUPERCONDUCTORS: Dict[str, QuantumMaterial] = {
    "aluminum": QuantumMaterial(
        name="Aluminum",
        critical_temperature=1.2,  # K
        current_temperature=0.5,
        gap_energy=0.34e-3,  # eV
        coherence_length=1600e-9,  # m
        penetration_depth=16e-9,
    ),
    "niobium": QuantumMaterial(
        name="Niobium",
        critical_temperature=9.3,
        current_temperature=4.2,
        gap_energy=1.5e-3,
        coherence_length=38e-9,
        penetration_depth=39e-9,
    ),
    "YBCO": QuantumMaterial(
        name="YBCO (High-Tc)",
        critical_temperature=92.0,
        current_temperature=77.0,  # Liquid nitrogen
        gap_energy=20e-3,
        coherence_length=1.5e-9,
        penetration_depth=150e-9,
    ),
    "MgB2": QuantumMaterial(
        name="Magnesium Diboride",
        critical_temperature=39.0,
        current_temperature=20.0,
        gap_energy=2.3e-3,
        coherence_length=5e-9,
        penetration_depth=85e-9,
    ),
    "room_temp_superconductor": QuantumMaterial(
        name="Hypothetical Room-Temp SC",
        critical_temperature=300.0,
        current_temperature=293.0,
        gap_energy=45e-3,
    ),
    # NEW: Additional quantum materials
    "lead": QuantumMaterial(
        name="Lead (Pb)",
        critical_temperature=7.2,
        current_temperature=4.2,
        gap_energy=1.35e-3,
        coherence_length=83e-9,
        penetration_depth=37e-9,
    ),
    "mercury": QuantumMaterial(
        name="Mercury (Hg) - First SC",
        critical_temperature=4.15,
        current_temperature=2.0,
        gap_energy=0.82e-3,
        coherence_length=200e-9,
        penetration_depth=45e-9,
    ),
    "nb3sn": QuantumMaterial(
        name="Nb3Sn (A15)",
        critical_temperature=18.3,
        current_temperature=10.0,
        gap_energy=3.4e-3,
        coherence_length=3e-9,
        penetration_depth=80e-9,
    ),
}

BEC_EXAMPLES = {
    "Rb87_cold": {"T": 100e-9, "T_c": 170e-9, "name": "Rb-87 BEC (100 nK)"},
    "Rb87_critical": {"T": 170e-9, "T_c": 170e-9, "name": "Rb-87 at T_c"},
    "Rb87_warm": {"T": 500e-9, "T_c": 170e-9, "name": "Rb-87 (500 nK)"},
    "Na23": {"T": 50e-9, "T_c": 2e-6, "name": "Na-23 BEC"},
}

QUANTUM_DOT_EXAMPLES = {
    "small_cold": {"size": 5e-9, "T": 4.2, "name": "5nm QD at 4K"},
    "small_room": {"size": 5e-9, "T": 300, "name": "5nm QD at 300K"},
    "large_cold": {"size": 20e-9, "T": 4.2, "name": "20nm QD at 4K"},
    "large_room": {"size": 20e-9, "T": 300, "name": "20nm QD at 300K"},
}


def chemistry_theta_summary():
    """Print theta analysis for example quantum materials."""
    print("=" * 70)
    print("CHEMISTRY / QUANTUM MATERIALS THETA ANALYSIS")
    print("=" * 70)
    print()

    # Superconductors
    print("SUPERCONDUCTORS:")
    print(f"{'Material':<25} {'T_c [K]':>10} {'T [K]':>10} {'θ':>8} {'Phase':<15}")
    print("-" * 70)

    for name, mat in SUPERCONDUCTORS.items():
        theta = compute_superconductor_theta(
            mat.current_temperature,
            mat.critical_temperature
        )
        phase = classify_material_phase(theta)
        print(f"{mat.name:<25} "
              f"{mat.critical_temperature:>10.1f} "
              f"{mat.current_temperature:>10.1f} "
              f"{theta:>8.3f} "
              f"{phase.value:<15}")

    print()

    # BEC
    print("BOSE-EINSTEIN CONDENSATES:")
    print(f"{'System':<25} {'T [nK]':>10} {'T_c [nK]':>10} {'N₀/N':>8} {'θ':>8}")
    print("-" * 60)

    for name, bec in BEC_EXAMPLES.items():
        theta = compute_bec_theta(bec["T"], bec["T_c"])
        fraction = bec_condensate_fraction(bec["T"], bec["T_c"])
        print(f"{bec['name']:<25} "
              f"{bec['T']*1e9:>10.0f} "
              f"{bec['T_c']*1e9:>10.0f} "
              f"{fraction:>8.3f} "
              f"{theta:>8.3f}")

    print()

    # Quantum dots
    print("QUANTUM DOTS:")
    print(f"{'System':<25} {'Size [nm]':>10} {'T [K]':>8} {'E_c/E_th':>10} {'θ':>8}")
    print("-" * 60)

    for name, qd in QUANTUM_DOT_EXAMPLES.items():
        E_conf = quantum_dot_confinement_energy(qd["size"])
        E_thermal = K_B * qd["T"]
        ratio = E_conf / E_thermal
        theta = compute_quantum_dot_theta(qd["size"], qd["T"])
        print(f"{qd['name']:<25} "
              f"{qd['size']*1e9:>10.0f} "
              f"{qd['T']:>8.1f} "
              f"{ratio:>10.1f} "
              f"{theta:>8.3f}")


if __name__ == "__main__":
    chemistry_theta_summary()
