"""
Condensed Matter Physics Domain Module

This module maps theta to condensed matter systems including phase transitions,
electronic transport, Anderson localization, and topological phases.

Theta Mapping:
    theta -> 0: Disordered/classical/insulating state
    theta -> 1: Ordered/quantum/conducting/topological state
    theta = (T_c - T)/T_c: Phase transition proximity
    theta = W_c/W: Localization transition
    theta = nu (quantized): Hall conductance

Key Features:
    - Phase transition analysis (Ising, XY, Heisenberg)
    - Electronic transport regimes (diffusive, ballistic, localized)
    - Anderson localization and metal-insulator transitions
    - Quantum Hall effects (integer and fractional)
    - Topological insulators and superconductors
    - Critical phenomena and scaling

References:
    @article{Ising1925,
      author = {Ising, Ernst},
      title = {Contribution to the theory of ferromagnetism},
      journal = {Z. Phys.},
      year = {1925}
    }
    @article{Onsager1944,
      author = {Onsager, Lars},
      title = {Crystal statistics. I. A two-dimensional model with an order-disorder transition},
      journal = {Phys. Rev.},
      year = {1944}
    }
    @article{Anderson1958,
      author = {Anderson, P. W.},
      title = {Absence of diffusion in certain random lattices},
      journal = {Phys. Rev.},
      year = {1958}
    }
    @article{Klitzing1980,
      author = {von Klitzing, K. and Dorda, G. and Pepper, M.},
      title = {New method for high-accuracy determination of the fine-structure constant},
      journal = {Phys. Rev. Lett.},
      year = {1980}
    }
    @article{Laughlin1983,
      author = {Laughlin, R. B.},
      title = {Anomalous quantum Hall effect},
      journal = {Phys. Rev. Lett.},
      year = {1983}
    }
    @article{KaneMele2005,
      author = {Kane, C. L. and Mele, E. J.},
      title = {Quantum spin Hall effect in graphene},
      journal = {Phys. Rev. Lett.},
      year = {2005}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


# =============================================================================
# Physical Constants
# =============================================================================

K_B = 1.380649e-23          # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
H_PLANCK = 6.62607015e-34   # Planck constant (J*s)
HBAR = H_PLANCK / (2 * np.pi)  # Reduced Planck constant
G_0 = E_CHARGE**2 / H_PLANCK   # Conductance quantum (S) = e^2/h ~ 3.87e-5 S
PHI_0 = H_PLANCK / (2 * E_CHARGE)  # Magnetic flux quantum (Wb) ~ 2.07e-15 Wb

# 2D Ising critical temperature (in units of J/k_B)
ISING_2D_TC = 2.0 / np.log(1 + np.sqrt(2))  # ~2.269

# Anderson localization critical disorder (3D, W_c/t ~ 16.5)
ANDERSON_WC_3D = 16.5


# =============================================================================
# Enums for Regime Classification
# =============================================================================

class PhaseRegime(Enum):
    """Classification of phase regimes based on theta."""
    DISORDERED = "disordered"           # theta < 0.2: High T, paramagnetic
    FLUCTUATING = "fluctuating"         # 0.2 <= theta < 0.4: Near critical
    CRITICAL = "critical"               # 0.4 <= theta < 0.6: At phase transition
    ORDERED = "ordered"                 # 0.6 <= theta < 0.8: Broken symmetry
    FULLY_ORDERED = "fully_ordered"     # theta >= 0.8: Deep in ordered phase


class TransportType(Enum):
    """Classification of electronic transport regimes."""
    INSULATING = "insulating"           # No conduction, theta ~ 0
    HOPPING = "hopping"                 # Variable range hopping
    DIFFUSIVE = "diffusive"             # Ohmic, mean free path << L
    BALLISTIC = "ballistic"             # Mean free path >> L
    QUANTUM_HALL = "quantum_hall"       # Quantized Hall conductance


class DisorderLevel(Enum):
    """Classification of disorder strength for Anderson localization."""
    CLEAN = "clean"                     # W/t < 1, extended states
    WEAKLY_DISORDERED = "weakly_disordered"  # 1 < W/t < 5
    CRITICAL_DISORDER = "critical_disorder"   # W/t ~ W_c
    ANDERSON_LOCALIZED = "anderson_localized" # W/t > W_c, localized states


class TopologicalPhase(Enum):
    """Classification of topological phases."""
    TRIVIAL = "trivial"                 # No topological protection
    WEAK_TI = "weak_topological"        # Weak topological insulator
    STRONG_TI = "strong_topological"    # Strong topological insulator
    TOPOLOGICAL_SC = "topological_superconductor"  # Majorana modes
    QUANTUM_SPIN_HALL = "quantum_spin_hall"  # 2D TI


class CriticalExponentClass(Enum):
    """Universality classes for critical phenomena."""
    ISING_2D = "ising_2d"               # beta=1/8, nu=1
    ISING_3D = "ising_3d"               # beta~0.326, nu~0.63
    XY_3D = "xy_3d"                     # beta~0.35, nu~0.67
    HEISENBERG_3D = "heisenberg_3d"     # beta~0.37, nu~0.71
    MEAN_FIELD = "mean_field"           # beta=1/2, nu=1/2


# =============================================================================
# Dataclass for Condensed Matter Systems
# =============================================================================

@dataclass
class CondensedMatterSystem:
    """
    A condensed matter system for theta analysis.

    Attributes:
        name: System identifier
        temperature: Operating temperature (K)
        critical_temperature: Phase transition temperature (K)
        coupling_constant: Dimensionless interaction strength J/k_B*T
        disorder_strength: Disorder parameter W/t
        system_size: Characteristic length (m)
        dimension: Spatial dimension (1, 2, or 3)
        carrier_density: Electron density (m^-3)
        magnetic_field: Applied magnetic field (T)
        hall_conductance: Hall conductance in units of e^2/h
        correlation_length: Correlation length (m)
        order_parameter: Order parameter magnitude [0, 1]
        mean_free_path: Mean free path (m)
        universality_class: Critical exponent universality class
    """
    name: str
    temperature: float
    critical_temperature: float
    coupling_constant: float = 1.0
    disorder_strength: float = 0.0
    system_size: float = 1e-6
    dimension: int = 3
    carrier_density: Optional[float] = None
    magnetic_field: float = 0.0
    hall_conductance: Optional[float] = None
    correlation_length: Optional[float] = None
    order_parameter: Optional[float] = None
    mean_free_path: Optional[float] = None
    universality_class: Optional[CriticalExponentClass] = None

    @property
    def reduced_temperature(self) -> float:
        """Reduced temperature t = (T - T_c) / T_c."""
        if self.critical_temperature == 0:
            return float('inf')
        return (self.temperature - self.critical_temperature) / self.critical_temperature

    @property
    def is_ordered(self) -> bool:
        """Whether system is below critical temperature."""
        return self.temperature < self.critical_temperature

    @property
    def is_localized(self) -> bool:
        """Whether system is Anderson localized (3D)."""
        if self.dimension == 3:
            return self.disorder_strength > ANDERSON_WC_3D
        # In 1D and 2D, any disorder localizes
        return self.disorder_strength > 0

    @property
    def transport_regime(self) -> TransportType:
        """Determine transport regime from mean free path and system size."""
        if self.mean_free_path is None:
            return TransportType.DIFFUSIVE
        if self.hall_conductance is not None and abs(self.hall_conductance) > 0.5:
            return TransportType.QUANTUM_HALL
        if self.mean_free_path < 1e-9:
            return TransportType.INSULATING
        if self.mean_free_path < self.system_size * 0.01:
            return TransportType.HOPPING
        if self.mean_free_path > self.system_size:
            return TransportType.BALLISTIC
        return TransportType.DIFFUSIVE


# =============================================================================
# Theta Calculation Functions
# =============================================================================

def compute_phase_theta(
    temperature: float,
    critical_temperature: float,
    exponent: float = 0.5
) -> float:
    """
    Compute theta for phase transition based on temperature.

    theta = max(0, (T_c - T) / T_c)^exponent for T < T_c
    theta = 0 for T >= T_c

    Args:
        temperature: Current temperature (K)
        critical_temperature: Phase transition temperature (K)
        exponent: Critical exponent modifier (default 0.5 for order parameter)

    Returns:
        Theta in [0, 1] where 1 = deep in ordered phase, 0 = disordered
    """
    if critical_temperature <= 0:
        return 0.0
    if temperature >= critical_temperature:
        return 0.0

    reduced = (critical_temperature - temperature) / critical_temperature
    theta = np.clip(reduced ** exponent, 0.0, 1.0)
    return float(theta)


def compute_order_theta(order_parameter: float) -> float:
    """
    Compute theta directly from order parameter.

    Args:
        order_parameter: Magnitude of order parameter [0, 1]

    Returns:
        Theta in [0, 1], directly equal to order parameter
    """
    return float(np.clip(order_parameter, 0.0, 1.0))


def compute_localization_theta(
    disorder_strength: float,
    critical_disorder: float = ANDERSON_WC_3D,
    dimension: int = 3
) -> float:
    """
    Compute theta for Anderson localization transition.

    For 3D: theta = W_c / W for extended states (W < W_c)
            theta = 0 for localized states (W > W_c)
    For 1D/2D: All states localized, theta = exp(-L/xi)

    Args:
        disorder_strength: Disorder parameter W/t
        critical_disorder: Critical disorder W_c/t (3D only)
        dimension: Spatial dimension

    Returns:
        Theta in [0, 1] where 1 = extended states, 0 = localized
    """
    if disorder_strength <= 0:
        return 1.0  # Clean limit

    if dimension == 3:
        if disorder_strength >= critical_disorder:
            return 0.0
        return float(np.clip(1.0 - disorder_strength / critical_disorder, 0.0, 1.0))
    else:
        # 1D and 2D: exponential localization
        # Localization length xi ~ 1/W^2 (1D), 1/W (2D weak disorder)
        if dimension == 1:
            return float(np.clip(np.exp(-disorder_strength**2 / 10), 0.0, 1.0))
        else:  # 2D
            return float(np.clip(np.exp(-disorder_strength / 5), 0.0, 1.0))


def compute_transport_theta(
    conductivity: float,
    reference_conductivity: Optional[float] = None
) -> float:
    """
    Compute theta from electrical conductivity.

    theta = sigma / sigma_0, capped at 1

    Args:
        conductivity: Electrical conductivity (S/m)
        reference_conductivity: Reference conductivity (default: quantum of conductance)

    Returns:
        Theta in [0, 1] where 1 = good conductor
    """
    if reference_conductivity is None:
        reference_conductivity = G_0 / 1e-9  # G_0 per nm

    if reference_conductivity <= 0:
        return 0.0
    if conductivity <= 0:
        return 0.0

    theta = conductivity / reference_conductivity
    return float(np.clip(theta, 0.0, 1.0))


def compute_hall_theta(filling_factor: float) -> float:
    """
    Compute theta from quantum Hall filling factor.

    For integer QHE: theta = |nu| if nu is close to integer
    For fractional QHE: theta = |nu| * (fractional enhancement)

    Args:
        filling_factor: Hall filling factor nu = n*h/(e*B)

    Returns:
        Theta in [0, 1] indicating degree of quantization
    """
    if filling_factor == 0:
        return 0.0

    nu = abs(filling_factor)

    # Check for integer quantization
    nearest_int = round(nu)
    if nearest_int > 0 and abs(nu - nearest_int) < 0.1:
        # Close to integer: high theta
        deviation = abs(nu - nearest_int) / nearest_int
        return float(np.clip(1.0 - deviation * 10, 0.5, 1.0))

    # Check for common fractions (Laughlin states)
    laughlin_fractions = [1/3, 2/5, 3/7, 2/3, 3/5, 4/7]
    for frac in laughlin_fractions:
        if abs(nu - frac) < 0.05:
            return float(np.clip(0.8 + 0.2 * (1 - abs(nu - frac) / 0.05), 0.5, 1.0))

    # General case: proportional to filling
    return float(np.clip(nu / 3, 0.0, 0.5))


def compute_correlation_theta(
    correlation_length: float,
    system_size: float
) -> float:
    """
    Compute theta from correlation length relative to system size.

    theta = xi / L, indicating how "coherent" the system is

    Args:
        correlation_length: Correlation length (m)
        system_size: System size (m)

    Returns:
        Theta in [0, 1] where 1 = correlation spans system
    """
    if system_size <= 0:
        return 0.0
    if correlation_length <= 0:
        return 0.0

    theta = correlation_length / system_size
    return float(np.clip(theta, 0.0, 1.0))


def compute_topological_theta(
    gap: float,
    bandwidth: float,
    z2_invariant: int = 0
) -> float:
    """
    Compute theta for topological phase.

    theta considers both band topology (Z2) and gap robustness.

    Args:
        gap: Band gap (eV)
        bandwidth: Total bandwidth (eV)
        z2_invariant: Z2 topological invariant (0 = trivial, 1 = non-trivial)

    Returns:
        Theta in [0, 1] where 1 = robust topological phase
    """
    if bandwidth <= 0:
        return 0.0

    gap_ratio = gap / bandwidth
    gap_theta = float(np.clip(gap_ratio, 0.0, 0.5))

    if z2_invariant == 1:
        # Topological: add 0.5 for non-trivial topology
        return float(np.clip(0.5 + gap_theta, 0.0, 1.0))
    else:
        return gap_theta


def compute_condensed_matter_theta(system: CondensedMatterSystem) -> float:
    """
    Compute unified theta for a condensed matter system.

    Combines phase transition, order parameter, localization, and transport.

    Args:
        system: CondensedMatterSystem instance

    Returns:
        Theta in [0, 1]
    """
    thetas = []
    weights = []

    # Phase transition contribution
    phase_theta = compute_phase_theta(
        system.temperature,
        system.critical_temperature
    )
    thetas.append(phase_theta)
    weights.append(0.4)

    # Order parameter contribution (if available)
    if system.order_parameter is not None:
        order_theta = compute_order_theta(system.order_parameter)
        thetas.append(order_theta)
        weights.append(0.3)
    else:
        weights[0] += 0.3  # Give weight to phase theta

    # Localization contribution
    if system.disorder_strength > 0:
        loc_theta = compute_localization_theta(
            system.disorder_strength,
            dimension=system.dimension
        )
        thetas.append(loc_theta)
        weights.append(0.2)
    else:
        thetas.append(1.0)  # Clean system
        weights.append(0.2)

    # Hall contribution (if in magnetic field)
    if system.hall_conductance is not None:
        hall_theta = compute_hall_theta(system.hall_conductance)
        thetas.append(hall_theta)
        weights.append(0.1)
    else:
        weights[0] += 0.1

    # Weighted average
    total_weight = sum(weights)
    theta = sum(t * w for t, w in zip(thetas, weights)) / total_weight

    return float(np.clip(theta, 0.0, 1.0))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_phase_regime(theta: float) -> PhaseRegime:
    """Classify phase regime based on theta value."""
    if theta < 0.2:
        return PhaseRegime.DISORDERED
    elif theta < 0.4:
        return PhaseRegime.FLUCTUATING
    elif theta < 0.6:
        return PhaseRegime.CRITICAL
    elif theta < 0.8:
        return PhaseRegime.ORDERED
    else:
        return PhaseRegime.FULLY_ORDERED


def classify_transport(theta: float) -> TransportType:
    """Classify transport regime based on theta value."""
    if theta < 0.1:
        return TransportType.INSULATING
    elif theta < 0.3:
        return TransportType.HOPPING
    elif theta < 0.7:
        return TransportType.DIFFUSIVE
    else:
        return TransportType.BALLISTIC


def classify_disorder(
    disorder_strength: float,
    dimension: int = 3
) -> DisorderLevel:
    """Classify disorder level."""
    if disorder_strength < 1:
        return DisorderLevel.CLEAN
    elif disorder_strength < 5:
        return DisorderLevel.WEAKLY_DISORDERED
    elif dimension == 3 and abs(disorder_strength - ANDERSON_WC_3D) < 2:
        return DisorderLevel.CRITICAL_DISORDER
    else:
        return DisorderLevel.ANDERSON_LOCALIZED


def classify_topological(
    z2_invariant: int,
    dimension: int = 3,
    has_time_reversal: bool = True
) -> TopologicalPhase:
    """Classify topological phase."""
    if z2_invariant == 0:
        return TopologicalPhase.TRIVIAL
    if dimension == 2 and has_time_reversal:
        return TopologicalPhase.QUANTUM_SPIN_HALL
    if dimension == 3:
        # Could distinguish weak vs strong based on indices
        return TopologicalPhase.STRONG_TI
    return TopologicalPhase.WEAK_TI


# =============================================================================
# Example Systems
# =============================================================================

CONDENSED_MATTER_SYSTEMS: Dict[str, CondensedMatterSystem] = {
    "ising_2d_below_tc": CondensedMatterSystem(
        name="2D Ising below T_c",
        temperature=2.0,  # In units of J/k_B
        critical_temperature=ISING_2D_TC,
        coupling_constant=1.0,
        dimension=2,
        order_parameter=0.8,
        universality_class=CriticalExponentClass.ISING_2D,
    ),
    "ising_2d_critical": CondensedMatterSystem(
        name="2D Ising at T_c",
        temperature=ISING_2D_TC,
        critical_temperature=ISING_2D_TC,
        coupling_constant=1.0,
        dimension=2,
        order_parameter=0.0,
        correlation_length=1e-3,  # Diverging
        universality_class=CriticalExponentClass.ISING_2D,
    ),
    "ising_2d_above_tc": CondensedMatterSystem(
        name="2D Ising above T_c",
        temperature=3.0,
        critical_temperature=ISING_2D_TC,
        coupling_constant=1.0,
        dimension=2,
        order_parameter=0.0,
        universality_class=CriticalExponentClass.ISING_2D,
    ),
    "quantum_hall_nu1": CondensedMatterSystem(
        name="Integer QHE nu=1",
        temperature=0.1,
        critical_temperature=10.0,  # Cyclotron gap
        magnetic_field=10.0,
        carrier_density=2.4e15,  # m^-2
        hall_conductance=1.0,  # e^2/h
        dimension=2,
        system_size=1e-4,
    ),
    "fqhe_laughlin_1_3": CondensedMatterSystem(
        name="FQHE Laughlin 1/3",
        temperature=0.05,
        critical_temperature=5.0,
        magnetic_field=15.0,
        carrier_density=8e14,
        hall_conductance=1/3,
        dimension=2,
        system_size=1e-4,
    ),
    "anderson_metal": CondensedMatterSystem(
        name="Anderson metal (3D)",
        temperature=10.0,
        critical_temperature=0.0,  # No phase transition
        disorder_strength=5.0,  # W/t < W_c
        dimension=3,
        system_size=1e-6,
        mean_free_path=1e-7,
    ),
    "anderson_insulator": CondensedMatterSystem(
        name="Anderson insulator (3D)",
        temperature=10.0,
        critical_temperature=0.0,
        disorder_strength=20.0,  # W/t > W_c
        dimension=3,
        system_size=1e-6,
        mean_free_path=1e-9,
    ),
    "topological_insulator_bi2se3": CondensedMatterSystem(
        name="Bi2Se3 TI",
        temperature=4.0,
        critical_temperature=300.0,  # Gap temperature scale
        dimension=3,
        disorder_strength=0.5,
        system_size=1e-7,
        order_parameter=0.9,  # Gap/bandwidth ratio as proxy
    ),
    "graphene_qsh": CondensedMatterSystem(
        name="Graphene QSH",
        temperature=0.01,
        critical_temperature=0.001,  # Very small SOC gap
        dimension=2,
        disorder_strength=0.1,
        carrier_density=1e16,
        system_size=1e-6,
    ),
    "cuprate_ybco": CondensedMatterSystem(
        name="YBCO superconductor",
        temperature=77.0,  # Liquid nitrogen
        critical_temperature=93.0,  # T_c of YBCO
        dimension=3,
        order_parameter=0.7,  # Below T_c
        coupling_constant=2.0,  # Strong coupling
        correlation_length=1e-9,  # Coherence length
    ),
    "heavy_fermion_cecoini5": CondensedMatterSystem(
        name="CeCoIn5 heavy fermion",
        temperature=1.0,
        critical_temperature=2.3,  # Superconducting T_c
        dimension=3,
        coupling_constant=5.0,  # Very strong correlations
        carrier_density=1e28,
        order_parameter=0.5,
    ),
    "weyl_semimetal": CondensedMatterSystem(
        name="TaAs Weyl semimetal",
        temperature=4.0,
        critical_temperature=1000.0,  # Structural stability
        dimension=3,
        disorder_strength=0.2,
        carrier_density=1e24,
        system_size=1e-6,
    ),
}


# =============================================================================
# Demonstration Function
# =============================================================================

def demonstrate_condensed_matter() -> Dict[str, Dict]:
    """
    Demonstrate theta calculations for example systems.

    Returns:
        Dictionary mapping system names to their analysis results.
    """
    results = {}

    for name, system in CONDENSED_MATTER_SYSTEMS.items():
        theta = compute_condensed_matter_theta(system)
        phase = classify_phase_regime(theta)

        results[name] = {
            "system": system.name,
            "theta": round(theta, 4),
            "phase_regime": phase.value,
            "temperature": system.temperature,
            "T_c": system.critical_temperature,
            "is_ordered": system.is_ordered,
            "is_localized": system.is_localized,
        }

    return results


if __name__ == "__main__":
    results = demonstrate_condensed_matter()
    print("\nCondensed Matter Systems Theta Analysis")
    print("=" * 60)
    for name, data in results.items():
        print(f"\n{data['system']}:")
        print(f"  theta = {data['theta']}")
        print(f"  Phase: {data['phase_regime']}")
        print(f"  T = {data['temperature']}, T_c = {data['T_c']}")
        print(f"  Ordered: {data['is_ordered']}, Localized: {data['is_localized']}")
