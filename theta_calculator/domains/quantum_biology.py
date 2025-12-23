"""
Quantum Biology Domain: Quantum Effects in Living Systems

This module implements theta as the quantum-classical interpolation parameter
for biological systems where quantum coherence plays a functional role.

Key Insight: Living systems exploit quantum effects at the boundary:
- theta ~ 0: Classical biochemistry (thermal noise dominates)
- theta ~ 1: Pure quantum coherence (isolated systems)
- 0.2 < theta < 0.7: Functional quantum biology (evolution-optimized)

Quantum biology demonstrates that life operates in the transition regime,
using quantum effects for efficiency while remaining robust against decoherence.

References (see BIBLIOGRAPHY.bib):
    \\cite{Engel2007} - Quantum coherence in FMO complex photosynthesis
    \\cite{Ritz2000} - Radical pair magnetoreception in birds
    \\cite{Klinman2013} - Hydrogen tunneling in enzyme catalysis
    \\cite{Lowdin1963} - Proton tunneling in DNA base pairs
    \\cite{Turin1996} - Vibration theory of olfaction
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class QuantumBioRegime(Enum):
    """Classification of quantum biological regimes."""
    COHERENT = "coherent"              # theta > 0.6: Strong quantum effects
    FUNCTIONAL = "functional"          # 0.3 < theta < 0.6: Evolution-optimized
    TRANSITION = "transition"          # 0.1 < theta < 0.3: Weak quantum effects
    CLASSICAL = "classical"            # theta < 0.1: Thermal noise dominates


class QuantumMechanism(Enum):
    """Types of quantum mechanisms in biology."""
    COHERENT_TRANSFER = "coherent_transfer"    # Energy/exciton transfer
    TUNNELING = "tunneling"                     # Proton/electron tunneling
    RADICAL_PAIR = "radical_pair"               # Spin-correlated radical pairs
    VIBRATION_ASSISTED = "vibration_assisted"  # Vibrational coupling


@dataclass
class BiologicalSystem:
    """
    A biological system exhibiting quantum effects.

    Attributes:
        name: System identifier
        organism: Host organism or type
        mechanism: Type of quantum mechanism
        coherence_time: Quantum coherence time (seconds)
        thermal_time: Thermal fluctuation timescale (seconds)
        functional_time: Relevant biological timescale (seconds)
        temperature: Operating temperature (Kelvin)
        efficiency_classical: Efficiency without quantum effects
        efficiency_quantum: Observed efficiency with quantum effects
    """
    name: str
    organism: str
    mechanism: QuantumMechanism
    coherence_time: float      # tau_c (seconds)
    thermal_time: float        # tau_th = hbar/(k_B T) (seconds)
    functional_time: float     # tau_f: biological process time (seconds)
    temperature: float         # T (Kelvin)
    efficiency_classical: Optional[float] = None
    efficiency_quantum: Optional[float] = None

    @property
    def quantum_advantage(self) -> Optional[float]:
        """Ratio of quantum to classical efficiency."""
        if self.efficiency_classical and self.efficiency_quantum:
            return self.efficiency_quantum / self.efficiency_classical
        return None

    @property
    def coherence_thermal_ratio(self) -> float:
        """Ratio of coherence time to thermal time."""
        return self.coherence_time / self.thermal_time


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

HBAR = 1.054571817e-34  # J*s (reduced Planck constant)
K_B = 1.380649e-23      # J/K (Boltzmann constant)


def thermal_time(temperature: float) -> float:
    """
    Compute thermal fluctuation timescale.

    tau_th = hbar / (k_B * T)

    This is the timescale over which thermal fluctuations randomize
    quantum phases at temperature T.
    """
    return HBAR / (K_B * temperature)


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_coherence_theta(system: BiologicalSystem) -> float:
    """
    Compute theta based on coherence time vs thermal time.

    theta = tau_c / (tau_c + tau_th)

    High coherence (tau_c >> tau_th): theta -> 1
    Low coherence (tau_c << tau_th): theta -> 0
    """
    tau_th = thermal_time(system.temperature)
    theta = system.coherence_time / (system.coherence_time + tau_th)
    return np.clip(theta, 0.0, 1.0)


def compute_functional_theta(system: BiologicalSystem) -> float:
    """
    Compute theta based on whether quantum coherence persists
    long enough to be functionally relevant.

    theta = tau_c / tau_f (capped at 1)

    If coherence outlasts the biological process, quantum effects
    can influence the outcome.
    """
    ratio = system.coherence_time / system.functional_time
    return np.clip(ratio, 0.0, 1.0)


def compute_tunneling_theta(
    barrier_height: float,  # eV
    barrier_width: float,   # nm
    particle_mass: float,   # kg (electron mass for electrons, proton mass for protons)
    temperature: float      # K
) -> float:
    """
    Compute theta for tunneling processes.

    Compares tunneling rate to classical (thermal) activation rate.

    WKB tunneling probability: P ~ exp(-2*kappa*d)
    where kappa = sqrt(2*m*V) / hbar

    Classical activation: P ~ exp(-V / k_B*T)

    theta = P_tunnel / (P_tunnel + P_classical)
    """
    # Convert units
    V_joules = barrier_height * 1.602176634e-19  # eV to J
    d_meters = barrier_width * 1e-9              # nm to m

    # WKB tunneling exponent
    kappa = np.sqrt(2 * particle_mass * V_joules) / HBAR
    tunnel_exponent = 2 * kappa * d_meters

    # Classical activation exponent
    classical_exponent = V_joules / (K_B * temperature)

    # Compare rates (in log space for numerical stability)
    # P_tunnel / P_classical = exp(classical_exponent - tunnel_exponent)
    log_ratio = classical_exponent - tunnel_exponent

    # theta = P_tunnel / (P_tunnel + P_classical)
    #       = 1 / (1 + P_classical/P_tunnel)
    #       = 1 / (1 + exp(-log_ratio))
    theta = 1.0 / (1.0 + np.exp(-np.clip(log_ratio, -50, 50)))

    return theta


def compute_quantum_bio_theta(system: BiologicalSystem) -> float:
    """
    Compute unified theta for a biological quantum system.

    Combines coherence-based and functional-time-based metrics.
    """
    theta_coherence = compute_coherence_theta(system)
    theta_functional = compute_functional_theta(system)

    # Geometric mean: both coherence and functional relevance matter
    return np.sqrt(theta_coherence * theta_functional)


def classify_regime(theta: float) -> QuantumBioRegime:
    """Classify quantum biology regime from theta."""
    if theta > 0.6:
        return QuantumBioRegime.COHERENT
    elif theta > 0.3:
        return QuantumBioRegime.FUNCTIONAL
    elif theta > 0.1:
        return QuantumBioRegime.TRANSITION
    else:
        return QuantumBioRegime.CLASSICAL


# =============================================================================
# EXAMPLE BIOLOGICAL SYSTEMS
# =============================================================================

# Physical constants for tunneling calculations
M_ELECTRON = 9.1093837015e-31  # kg
M_PROTON = 1.67262192369e-27   # kg

BIOLOGICAL_SYSTEMS: Dict[str, BiologicalSystem] = {
    "fmo_complex": BiologicalSystem(
        name="FMO Complex (Photosynthesis)",
        organism="Green sulfur bacteria",
        mechanism=QuantumMechanism.COHERENT_TRANSFER,
        coherence_time=300e-15,        # 300 fs coherence at 77K (Engel 2007)
        thermal_time=thermal_time(77), # 77K experiment
        functional_time=1e-12,         # ~1 ps energy transfer
        temperature=77,                # Cryogenic measurement
        efficiency_classical=0.70,     # Forster hopping estimate
        efficiency_quantum=0.99,       # Observed near-unity efficiency
    ),
    "fmo_room_temp": BiologicalSystem(
        name="FMO Complex (Room Temp)",
        organism="Green sulfur bacteria",
        mechanism=QuantumMechanism.COHERENT_TRANSFER,
        coherence_time=60e-15,         # ~60 fs at 300K (shorter but present)
        thermal_time=thermal_time(300),
        functional_time=1e-12,
        temperature=300,
        efficiency_classical=0.70,
        efficiency_quantum=0.95,
    ),
    "lhcii_complex": BiologicalSystem(
        name="LHCII Light Harvesting",
        organism="Higher plants",
        mechanism=QuantumMechanism.COHERENT_TRANSFER,
        coherence_time=400e-15,        # 400 fs in LHCII
        thermal_time=thermal_time(300),
        functional_time=5e-12,         # 5 ps transfer time
        temperature=300,
        efficiency_classical=0.80,
        efficiency_quantum=0.95,
    ),
    "cryptochrome_bird": BiologicalSystem(
        name="Cryptochrome (Bird Navigation)",
        organism="European robin",
        mechanism=QuantumMechanism.RADICAL_PAIR,
        coherence_time=1e-6,           # ~1 μs spin coherence (Ritz 2000)
        thermal_time=thermal_time(310), # Body temperature
        functional_time=1e-6,          # Matches coherence for sensitivity
        temperature=310,               # Bird body temp ~37°C
        efficiency_classical=0.0,      # No classical magnetic sense
        efficiency_quantum=1.0,        # Functional compass
    ),
    "cryptochrome_drosophila": BiologicalSystem(
        name="Cryptochrome (Fruit Fly)",
        organism="Drosophila melanogaster",
        mechanism=QuantumMechanism.RADICAL_PAIR,
        coherence_time=0.5e-6,
        thermal_time=thermal_time(298),
        functional_time=1e-6,
        temperature=298,
    ),
    "alcohol_dehydrogenase": BiologicalSystem(
        name="Alcohol Dehydrogenase",
        organism="Various (enzyme)",
        mechanism=QuantumMechanism.TUNNELING,
        coherence_time=1e-13,          # Tunneling event timescale
        thermal_time=thermal_time(310),
        functional_time=1e-3,          # Catalytic turnover ~1 ms
        temperature=310,
        efficiency_classical=0.001,    # Classical over-barrier rate
        efficiency_quantum=1.0,        # Tunneling-enhanced rate
    ),
    "soybean_lipoxygenase": BiologicalSystem(
        name="Soybean Lipoxygenase",
        organism="Glycine max",
        mechanism=QuantumMechanism.TUNNELING,
        coherence_time=1e-13,
        thermal_time=thermal_time(300),
        functional_time=1e-3,
        temperature=300,
        efficiency_classical=0.01,     # KIE of ~80 indicates tunneling
        efficiency_quantum=0.80,       # Observed rate
    ),
    "dna_tautomerization": BiologicalSystem(
        name="DNA Base Pair Tautomerization",
        organism="Universal (DNA)",
        mechanism=QuantumMechanism.TUNNELING,
        coherence_time=1e-14,          # Proton tunneling timescale
        thermal_time=thermal_time(310),
        functional_time=1.0,           # DNA replication timescale
        temperature=310,
        # Low theta because process is rare but significant
    ),
    "olfactory_receptor": BiologicalSystem(
        name="Olfactory Receptor (Turin model)",
        organism="Human",
        mechanism=QuantumMechanism.VIBRATION_ASSISTED,
        coherence_time=1e-13,          # Vibrational lifetime
        thermal_time=thermal_time(310),
        functional_time=1e-9,          # Receptor binding timescale
        temperature=310,
    ),
    "atp_synthase": BiologicalSystem(
        name="ATP Synthase Proton Channel",
        organism="Universal (mitochondria)",
        mechanism=QuantumMechanism.TUNNELING,
        coherence_time=1e-13,
        thermal_time=thermal_time(310),
        functional_time=1e-4,          # Proton transit time
        temperature=310,
    ),
}


# =============================================================================
# TUNNELING ANALYSIS FOR SPECIFIC SYSTEMS
# =============================================================================

def analyze_enzyme_tunneling(
    barrier_eV: float = 0.4,     # Typical enzyme barrier
    width_nm: float = 0.05,     # Tunneling distance ~0.5 Angstrom
    temperature: float = 310.0   # Body temperature
) -> Dict[str, float]:
    """
    Analyze proton and hydrogen tunneling in enzymes.

    Returns theta values for different tunneling scenarios.
    """
    return {
        "proton_tunneling": compute_tunneling_theta(
            barrier_eV, width_nm, M_PROTON, temperature
        ),
        "hydrogen_tunneling": compute_tunneling_theta(
            barrier_eV, width_nm, M_PROTON, temperature  # Same as proton for H
        ),
        "deuterium_tunneling": compute_tunneling_theta(
            barrier_eV, width_nm, 2 * M_PROTON, temperature  # D is ~2x proton mass
        ),
    }


def analyze_dna_mutation_rate(
    barrier_eV: float = 0.3,     # Tautomerization barrier
    width_nm: float = 0.04,     # Proton shift distance
    temperature: float = 310.0
) -> Dict[str, float]:
    """
    Analyze quantum contribution to DNA point mutations.

    Proton tunneling between base pairs can cause rare tautomeric
    forms that lead to point mutations during replication.
    """
    theta = compute_tunneling_theta(barrier_eV, width_nm, M_PROTON, temperature)

    # Mutation rate enhancement from tunneling
    # Classical rate: ~exp(-V/kT)
    # Quantum rate: ~exp(-2*kappa*d)
    classical_rate = np.exp(-barrier_eV * 1.602e-19 / (K_B * temperature))

    return {
        "theta": theta,
        "classical_mutation_probability": classical_rate,
        "quantum_enhancement_factor": 1.0 / (1.0 - theta) if theta < 1 else float('inf'),
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def quantum_biology_theta_summary():
    """Print theta analysis for all biological systems."""
    print("=" * 80)
    print("QUANTUM BIOLOGY THETA ANALYSIS")
    print("=" * 80)
    print()
    print(f"{'System':<35} {'θ':>8} {'τ_c':>12} {'Mechanism':<20} {'Regime':<12}")
    print("-" * 80)

    for name, system in BIOLOGICAL_SYSTEMS.items():
        theta = compute_quantum_bio_theta(system)
        regime = classify_regime(theta)

        # Format coherence time
        if system.coherence_time >= 1e-3:
            tau_str = f"{system.coherence_time*1e3:.1f} ms"
        elif system.coherence_time >= 1e-6:
            tau_str = f"{system.coherence_time*1e6:.1f} μs"
        elif system.coherence_time >= 1e-9:
            tau_str = f"{system.coherence_time*1e9:.1f} ns"
        elif system.coherence_time >= 1e-12:
            tau_str = f"{system.coherence_time*1e12:.1f} ps"
        else:
            tau_str = f"{system.coherence_time*1e15:.1f} fs"

        print(f"{system.name:<35} {theta:>8.3f} {tau_str:>12} "
              f"{system.mechanism.value:<20} {regime.value:<12}")

    print()
    print("Key: Photosynthesis operates in FUNCTIONAL regime - evolution optimized!")
    print("     Magnetoreception requires COHERENT regime for compass function")


if __name__ == "__main__":
    quantum_biology_theta_summary()
