r"""
Quantum Foundations Domain: Decoherence, Measurement, and Quantum-Classical Transitions

This module implements theta as the quantum-classical interpolation parameter
for foundational quantum mechanics phenomena including decoherence, the
measurement problem, and the emergence of classical behavior.

## Mapping Definition

This domain maps quantum systems to theta via coherence and classicality metrics:

**Inputs (Physical Analogs):**
- decoherence_time (tau_D) -> Characteristic decoherence timescale
- observation_time (t) -> Time of observation/measurement
- environment_coupling -> Strength of system-environment interaction
- hilbert_dim -> Dimension of system Hilbert space
- superposition_size -> Spatial extent of quantum superposition

**Theta Mapping:**
theta = exp(-t / tau_D) for simple decoherence
theta = |<psi|pointer>|^2 for measurement (overlap with pointer states)
theta = hbar_eff / hbar for effective classicality

**Interpretation:**
- theta -> 0: Classical regime (decoherence complete, pointer states selected)
- theta -> 1: Quantum regime (coherent superpositions preserved)
- 0.1 < theta < 0.9: Transition regime (partial decoherence)

**Key Feature:** The quantum-classical boundary is not sharp but represents
a continuous transition governed by environmental monitoring.

**Important:** This is a PHYSICAL SCORE based on coherence survival.

References (see BIBLIOGRAPHY.bib):
    \cite{Zurek2003} - Quantum Darwinism and decoherence
    \cite{JoosZeh1985} - Environmental decoherence mechanisms
    \cite{Penrose1996} - Gravitational state reduction
    \cite{Chisholm2025} - Pointer states for non-commuting observables
    \cite{Zhang2024Decoherence} - Decoherence without einselection
    \cite{Doucet2024} - Measurement compatibility and classical emergence
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class DecoherenceRegime(Enum):
    """Classification of decoherence regimes based on theta."""
    COHERENT = "coherent"                  # theta > 0.8: Quantum superposition intact
    PARTIAL = "partial_decoherence"        # 0.4 < theta < 0.8: Mixed state
    DECOHERED = "decohered"                # 0.1 < theta < 0.4: Nearly classical
    CLASSICAL = "classical"                # theta < 0.1: Pointer basis reached


class MeasurementType(Enum):
    """Types of quantum measurements."""
    PROJECTIVE = "projective"              # von Neumann measurement
    WEAK = "weak"                          # Minimal disturbance measurement
    CONTINUOUS = "continuous"              # Quantum trajectory measurement
    QND = "quantum_non_demolition"         # Preserves measured observable


class QuantumClassicalMechanism(Enum):
    """Mechanisms for quantum-to-classical transition."""
    ENVIRONMENTAL = "environmental_decoherence"      # Standard Zurek mechanism
    GRAVITATIONAL = "gravitational_decoherence"      # Penrose-Diosi mechanism
    SPONTANEOUS = "spontaneous_localization"         # GRW/CSL models
    EINSELECTION = "einselection"                    # Environment-induced selection


class PointerBasis(Enum):
    """Common pointer bases selected by einselection."""
    POSITION = "position"                  # Spatial localization
    ENERGY = "energy"                      # Energy eigenstates
    COHERENT = "coherent_states"           # Phase space localization
    NUMBER = "number"                      # Fock states


@dataclass
class QuantumFoundationSystem:
    """
    A quantum foundations system for theta analysis.

    Represents a quantum system undergoing decoherence or measurement,
    characterized by its coherence properties and environment coupling.

    Attributes:
        name: System identifier
        hilbert_dim: Dimension of the Hilbert space (2 for qubit, etc.)
        decoherence_time: Characteristic decoherence timescale (seconds)
        observation_time: Time of measurement/observation (seconds)
        environment_coupling: System-environment coupling strength (Hz)
        temperature: Environment temperature (Kelvin)
        mechanism: Type of decoherence mechanism
        pointer_basis: Selected pointer basis type
        mass: System mass for gravitational effects (kg, optional)
        superposition_size: Spatial superposition scale (m, optional)
    """
    name: str
    hilbert_dim: int
    decoherence_time: float
    observation_time: float
    environment_coupling: float
    temperature: float
    mechanism: QuantumClassicalMechanism = QuantumClassicalMechanism.ENVIRONMENTAL
    pointer_basis: PointerBasis = PointerBasis.POSITION
    mass: Optional[float] = None
    superposition_size: Optional[float] = None

    @property
    def decoherence_rate(self) -> float:
        """Decoherence rate gamma = 1/tau_D (Hz)."""
        if self.decoherence_time > 0:
            return 1.0 / self.decoherence_time
        return float('inf')

    @property
    def quantum_lifetime_ratio(self) -> float:
        """Ratio of observation time to decoherence time."""
        if self.decoherence_time > 0:
            return self.observation_time / self.decoherence_time
        return float('inf')


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

HBAR = 1.054571817e-34    # J*s (reduced Planck constant)
K_B = 1.380649e-23        # J/K (Boltzmann constant)
G = 6.67430e-11           # m^3/(kg*s^2) (gravitational constant)
C = 299792458             # m/s (speed of light)
M_PLANCK = 2.176434e-8    # kg (Planck mass)
L_PLANCK = 1.616255e-35   # m (Planck length)
T_PLANCK = 5.391247e-44   # s (Planck time)


def thermal_time(temperature: float) -> float:
    """
    Compute thermal fluctuation timescale.

    tau_th = hbar / (k_B * T)

    This is the characteristic time for thermal fluctuations
    to randomize quantum phases.
    """
    if temperature <= 0:
        return float('inf')
    return HBAR / (K_B * temperature)


def thermal_wavelength(temperature: float, mass: float) -> float:
    """
    Compute thermal de Broglie wavelength.

    lambda_th = h / sqrt(2 * pi * m * k_B * T)

    Sets the scale at which quantum effects become relevant.
    """
    if temperature <= 0 or mass <= 0:
        return float('inf')
    return HBAR * np.sqrt(2 * np.pi / (mass * K_B * temperature))


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_decoherence_theta(
    observation_time: float,
    decoherence_time: float
) -> float:
    r"""
    Compute theta from exponential decoherence dynamics.

    theta = exp(-t / tau_D)

    This represents the survival probability of quantum coherence
    after time t, given characteristic decoherence time tau_D.

    Args:
        observation_time: Time of observation (seconds)
        decoherence_time: Characteristic decoherence time (seconds)

    Returns:
        theta in [0, 1]

    Reference: \cite{Zurek2003}, \cite{JoosZeh1985}
    """
    if decoherence_time <= 0:
        return 0.0
    if observation_time <= 0:
        return 1.0

    theta = np.exp(-observation_time / decoherence_time)
    return np.clip(theta, 0.0, 1.0)


def compute_measurement_theta(
    overlap: float,
    measurement_strength: float = 1.0
) -> float:
    r"""
    Compute theta for a measurement process.

    theta = 1 - |<psi|pointer>|^2 * strength

    High overlap with pointer basis means classical (low theta).
    Low overlap means quantum superposition preserved (high theta).

    Args:
        overlap: |<psi|pointer>|^2, overlap with pointer state [0, 1]
        measurement_strength: Measurement strength parameter [0, 1]

    Returns:
        theta in [0, 1]

    Reference: \cite{Doucet2024}
    """
    theta = 1.0 - overlap * measurement_strength
    return np.clip(theta, 0.0, 1.0)


def compute_zurek_decoherence_time(
    temperature: float,
    mass: float,
    superposition_size: float,
    scattering_rate: Optional[float] = None
) -> float:
    r"""
    Compute Zurek's decoherence time for spatial superpositions.

    For photon/air scattering:
    tau_D ~ (lambda_th / Delta_x)^2 * tau_scattering

    For thermal photons at temperature T with superposition size Delta_x:
    tau_D ~ (hbar / k_B T) * (lambda_th / Delta_x)^2

    Args:
        temperature: Environment temperature (K)
        mass: System mass (kg)
        superposition_size: Spatial separation of superposition (m)
        scattering_rate: Optional scattering rate (Hz)

    Returns:
        Decoherence time (seconds)

    Reference: \cite{Zurek2003}
    """
    if superposition_size <= 0:
        return float('inf')

    lambda_th = thermal_wavelength(temperature, mass)

    if lambda_th == float('inf'):
        return float('inf')

    ratio_squared = (lambda_th / superposition_size) ** 2

    if scattering_rate is not None and scattering_rate > 0:
        tau_scatter = 1.0 / scattering_rate
        return ratio_squared * tau_scatter
    else:
        tau_thermal = thermal_time(temperature)
        return ratio_squared * tau_thermal


def compute_penrose_time(
    mass: float,
    superposition_size: float
) -> float:
    r"""
    Compute Penrose gravitational decoherence time.

    tau_G ~ hbar / E_G

    where E_G ~ G * m^2 / r is the gravitational self-energy
    of the superposition.

    This represents gravitational instability of quantum superpositions
    of different mass distributions.

    Args:
        mass: System mass (kg)
        superposition_size: Spatial separation (m)

    Returns:
        Gravitational decoherence time (seconds)

    Reference: \cite{Penrose1996}
    """
    if superposition_size <= 0 or mass <= 0:
        return float('inf')

    E_gravity = G * mass**2 / superposition_size
    if E_gravity <= 0:
        return float('inf')

    return HBAR / E_gravity


def compute_diosi_decoherence_rate(
    mass_density: float,
    volume: float
) -> float:
    r"""
    Compute Diosi gravitational decoherence rate.

    gamma_D ~ G * rho^2 * V / hbar

    where rho is mass density and V is volume.

    Reference: \cite{Penrose1996} (Penrose-Diosi model)
    """
    rate = G * mass_density**2 * volume / HBAR
    return rate


def compute_pointer_overlap_theta(
    coherence_matrix: np.ndarray,
    pointer_projector: np.ndarray
) -> float:
    r"""
    Compute theta from density matrix overlap with pointer states.

    theta = 1 - Tr(rho * P_pointer)

    where P_pointer projects onto the pointer basis.

    Args:
        coherence_matrix: Density matrix rho
        pointer_projector: Projector onto pointer subspace

    Returns:
        theta in [0, 1]
    """
    overlap = np.real(np.trace(coherence_matrix @ pointer_projector))
    theta = 1.0 - np.clip(overlap, 0.0, 1.0)
    return theta


def compute_quantum_foundations_theta(system: QuantumFoundationSystem) -> float:
    """
    Compute unified theta for a quantum foundations system.

    Combines environmental decoherence with gravitational effects
    if mass and superposition size are specified.

    Args:
        system: QuantumFoundationSystem instance

    Returns:
        theta in [0, 1]
    """
    # Environmental decoherence contribution
    theta_env = compute_decoherence_theta(
        system.observation_time,
        system.decoherence_time
    )

    # Gravitational decoherence (if applicable)
    theta_grav = 1.0
    if system.mass is not None and system.superposition_size is not None:
        tau_penrose = compute_penrose_time(system.mass, system.superposition_size)
        if tau_penrose < float('inf'):
            theta_grav = compute_decoherence_theta(
                system.observation_time,
                tau_penrose
            )

    # Both mechanisms must preserve coherence
    # Use minimum (most restrictive) theta
    return min(theta_env, theta_grav)


def classify_decoherence_regime(theta: float) -> DecoherenceRegime:
    """Classify decoherence regime from theta value."""
    if theta > 0.8:
        return DecoherenceRegime.COHERENT
    elif theta > 0.4:
        return DecoherenceRegime.PARTIAL
    elif theta > 0.1:
        return DecoherenceRegime.DECOHERED
    else:
        return DecoherenceRegime.CLASSICAL


# =============================================================================
# SPECIFIC DECOHERENCE SCENARIOS
# =============================================================================

def analyze_schrodinger_cat(
    mass_kg: float = 4.0,
    superposition_m: float = 0.1,
    temperature_k: float = 300.0,
    observation_s: float = 1.0
) -> Dict[str, float]:
    """
    Analyze decoherence for a Schrodinger cat-like macroscopic superposition.

    Returns theta values and timescales for different mechanisms.
    """
    tau_env = compute_zurek_decoherence_time(
        temperature_k, mass_kg, superposition_m
    )
    tau_grav = compute_penrose_time(mass_kg, superposition_m)

    return {
        "tau_environmental": tau_env,
        "tau_gravitational": tau_grav,
        "tau_dominant": min(tau_env, tau_grav),
        "theta_environmental": compute_decoherence_theta(observation_s, tau_env),
        "theta_gravitational": compute_decoherence_theta(observation_s, tau_grav),
        "regime": "instantly_classical",  # For macroscopic objects
    }


def analyze_matter_wave_interference(
    mass_kg: float,
    slit_separation_m: float,
    temperature_k: float = 300.0,
    flight_time_s: float = 1e-3
) -> Dict[str, float]:
    """
    Analyze coherence in matter-wave interferometry.

    Relevant for molecular interferometry experiments.
    """
    lambda_dB = HBAR / np.sqrt(2 * mass_kg * K_B * temperature_k)
    tau_deco = compute_zurek_decoherence_time(
        temperature_k, mass_kg, slit_separation_m
    )
    theta = compute_decoherence_theta(flight_time_s, tau_deco)

    return {
        "de_broglie_wavelength": lambda_dB,
        "decoherence_time": tau_deco,
        "theta": theta,
        "regime": classify_decoherence_regime(theta).value,
        "coherence_preserved": theta > 0.5,
    }


# =============================================================================
# EXAMPLE QUANTUM SYSTEMS
# =============================================================================

QUANTUM_FOUNDATIONS_SYSTEMS: Dict[str, QuantumFoundationSystem] = {
    # Microscopic quantum systems
    "single_photon": QuantumFoundationSystem(
        name="Single Photon (vacuum)",
        hilbert_dim=2,
        decoherence_time=1e6,           # Extremely long in vacuum
        observation_time=1e-9,          # Nanosecond observation
        environment_coupling=1e3,       # Minimal coupling
        temperature=3.0,                # CMB temperature
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.NUMBER,
    ),
    "superconducting_qubit": QuantumFoundationSystem(
        name="Superconducting Qubit",
        hilbert_dim=2,
        decoherence_time=100e-6,        # ~100 microseconds T2
        observation_time=1e-6,          # Microsecond gate
        environment_coupling=1e6,       # MHz coupling
        temperature=0.020,              # 20 mK dilution fridge
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.ENERGY,
    ),
    "trapped_ion": QuantumFoundationSystem(
        name="Trapped Ion Qubit",
        hilbert_dim=2,
        decoherence_time=1.0,           # ~1 second coherence
        observation_time=1e-3,          # Millisecond operations
        environment_coupling=1e3,
        temperature=1e-6,               # Doppler cooled ~microK
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.ENERGY,
    ),
    "nv_center": QuantumFoundationSystem(
        name="NV Center in Diamond",
        hilbert_dim=3,
        decoherence_time=1e-3,          # ~1 ms at room temp
        observation_time=1e-6,
        environment_coupling=1e9,       # GHz hyperfine
        temperature=300,                # Room temperature
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.ENERGY,
    ),
    # Mesoscopic systems
    "c60_fullerene": QuantumFoundationSystem(
        name="C60 Fullerene (interferometry)",
        hilbert_dim=1000,               # Many internal modes
        decoherence_time=1e-6,          # Microsecond coherence
        observation_time=1e-3,          # Flight time
        environment_coupling=1e6,
        temperature=900,                # Oven temperature
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.POSITION,
        mass=60 * 12 * 1.66e-27,        # 60 carbon atoms
        superposition_size=1e-7,        # 100 nm grating
    ),
    "squid_ring": QuantumFoundationSystem(
        name="SQUID Supercurrent",
        hilbert_dim=2,
        decoherence_time=1e-6,          # Microsecond
        observation_time=1e-9,          # Fast readout
        environment_coupling=1e9,
        temperature=0.050,              # 50 mK
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.COHERENT,
        mass=1e-15,                     # Effective mass
        superposition_size=1e-6,        # Micron-scale ring
    ),
    # Optomechanical systems
    "optomechanical_mirror": QuantumFoundationSystem(
        name="Optomechanical Mirror",
        hilbert_dim=100,                # Many phonon modes
        decoherence_time=1e-3,          # Millisecond
        observation_time=1e-6,
        environment_coupling=1e6,
        temperature=0.010,              # 10 mK
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.POSITION,
        mass=1e-12,                     # Picogram mirror
        superposition_size=1e-15,       # Femtometer motion
    ),
    # Gedanken experiments
    "schrodinger_cat": QuantumFoundationSystem(
        name="Schrodinger Cat (Gedanken)",
        hilbert_dim=int(1e23),          # Avogadro-scale
        decoherence_time=1e-40,         # Instantly decohered
        observation_time=1.0,           # 1 second
        environment_coupling=1e20,      # Maximal coupling
        temperature=300,
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.POSITION,
        mass=4.0,                       # 4 kg cat
        superposition_size=0.1,         # 10 cm alive/dead separation
    ),
    "wigner_friend": QuantumFoundationSystem(
        name="Wigner's Friend",
        hilbert_dim=int(1e26),          # Human-scale
        decoherence_time=1e-35,
        observation_time=0.1,           # Reaction time
        environment_coupling=1e22,
        temperature=310,                # Body temperature
        mechanism=QuantumClassicalMechanism.ENVIRONMENTAL,
        pointer_basis=PointerBasis.POSITION,
        mass=70.0,                      # 70 kg person
        superposition_size=0.01,        # 1 cm
    ),
    # Gravitational decoherence tests
    "gravity_test_mass": QuantumFoundationSystem(
        name="Penrose Gravity Test Mass",
        hilbert_dim=2,
        decoherence_time=1.0,           # Set by gravity
        observation_time=0.1,
        environment_coupling=1e3,
        temperature=1e-3,               # milliKelvin
        mechanism=QuantumClassicalMechanism.GRAVITATIONAL,
        pointer_basis=PointerBasis.POSITION,
        mass=1e-14,                     # 10 fg mass
        superposition_size=1e-6,        # Micron separation
    ),
}


# =============================================================================
# QUANTUM DARWINISM ANALYSIS
# =============================================================================

def compute_redundancy_theta(
    n_copies: int,
    information_per_copy: float
) -> float:
    """
    Compute theta from quantum Darwinism redundancy.

    In quantum Darwinism, classical objectivity emerges when
    information about pointer states is redundantly recorded
    in many environmental fragments.

    theta = 1 / (1 + log(n_copies))

    More copies -> lower theta (more classical)

    Reference: \cite{Zurek2003}
    """
    if n_copies <= 1:
        return 1.0

    # Logarithmic scaling of redundancy
    redundancy = np.log(n_copies) * information_per_copy
    theta = 1.0 / (1.0 + redundancy)

    return np.clip(theta, 0.0, 1.0)


def compute_mutual_information_theta(
    system_entropy: float,
    environment_entropy: float,
    total_entropy: float
) -> float:
    """
    Compute theta from quantum mutual information.

    I(S:E) = S(rho_S) + S(rho_E) - S(rho_SE)

    High mutual information indicates strong system-environment
    correlation and thus classicalization.

    theta = 1 - I(S:E) / (2 * min(S_S, S_E))
    """
    mutual_info = system_entropy + environment_entropy - total_entropy
    max_info = 2 * min(system_entropy, environment_entropy)

    if max_info <= 0:
        return 1.0

    theta = 1.0 - mutual_info / max_info
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def quantum_foundations_summary():
    """Print theta analysis for quantum foundations systems."""
    print("=" * 90)
    print("QUANTUM FOUNDATIONS THETA ANALYSIS")
    print("=" * 90)
    print()
    print(f"{'System':<30} {'theta':>8} {'tau_D':>12} {'t_obs':>10} "
          f"{'Mechanism':<15} {'Regime':<12}")
    print("-" * 90)

    for name, system in QUANTUM_FOUNDATIONS_SYSTEMS.items():
        theta = compute_quantum_foundations_theta(system)
        regime = classify_decoherence_regime(theta)

        # Format decoherence time
        tau = system.decoherence_time
        if tau >= 1.0:
            tau_str = f"{tau:.1f} s"
        elif tau >= 1e-3:
            tau_str = f"{tau*1e3:.1f} ms"
        elif tau >= 1e-6:
            tau_str = f"{tau*1e6:.1f} us"
        elif tau >= 1e-9:
            tau_str = f"{tau*1e9:.1f} ns"
        elif tau >= 1e-12:
            tau_str = f"{tau*1e12:.1f} ps"
        else:
            tau_str = f"{tau:.1e} s"

        # Format observation time
        t = system.observation_time
        if t >= 1.0:
            t_str = f"{t:.1f} s"
        elif t >= 1e-3:
            t_str = f"{t*1e3:.1f} ms"
        elif t >= 1e-6:
            t_str = f"{t*1e6:.1f} us"
        else:
            t_str = f"{t*1e9:.1f} ns"

        mech = system.mechanism.value.split('_')[0][:15]

        print(f"{system.name:<30} {theta:>8.4f} {tau_str:>12} {t_str:>10} "
              f"{mech:<15} {regime.value:<12}")

    print()
    print("Key: Microscopic systems maintain coherence (high theta)")
    print("     Macroscopic superpositions instantly decohere (theta ~ 0)")


if __name__ == "__main__":
    quantum_foundations_summary()
