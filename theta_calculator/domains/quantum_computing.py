"""
Quantum Computing Domain: Error Thresholds and Decoherence

This module implements theta as the quantum-classical interpolation parameter
for quantum computing systems using coherence and error rates.

Key Insight: Quantum computers transition between:
- theta ~ 1: Coherent quantum state (useful quantum computation)
- theta ~ 0: Decoherent classical state (noise-dominated)

The critical parameter is the error threshold:
- p < p_threshold: Quantum error correction works (exponential suppression)
- p > p_threshold: Errors accumulate faster than correction

Recent milestone (2024): Google Willow achieved below-threshold operation!

References:
- Shor (1995): Fault-Tolerant Quantum Computation
- Kitaev (2003): Fault-tolerant quantum computation by anyons
- Google (2024): Quantum error correction below threshold (Nature)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class QubitType(Enum):
    """Types of physical qubit implementations."""
    TRANSMON = "transmon"              # Superconducting (Google, IBM)
    TRAPPED_ION = "trapped_ion"        # Ionic (IonQ, Quantinuum)
    NV_CENTER = "nv_center"            # Diamond defect
    PHOTONIC = "photonic"              # Optical qubits
    TOPOLOGICAL = "topological"        # Majorana (Microsoft)
    NEUTRAL_ATOM = "neutral_atom"      # Cold atoms (QuEra)
    CAT_QUBIT = "cat_qubit"            # Bosonic (Amazon)


class CoherenceRegime(Enum):
    """Coherence regime classification."""
    HIGHLY_COHERENT = "highly_coherent"   # theta > 0.9
    COHERENT = "coherent"                  # 0.7 < theta < 0.9
    PARTIALLY_COHERENT = "partial"         # 0.3 < theta < 0.7
    MOSTLY_DECOHERENT = "mostly_decoherent"  # 0.1 < theta < 0.3
    CLASSICAL = "classical"                # theta < 0.1


@dataclass
class QubitSystem:
    """
    A quantum computing system for theta analysis.

    Attributes:
        name: System identifier
        qubit_type: Physical implementation
        n_qubits: Number of qubits
        T1: Relaxation time (energy decay) in seconds
        T2: Dephasing time (phase coherence) in seconds
        gate_time: Single-qubit gate duration in seconds
        error_rate: Error per gate operation
        error_threshold: Threshold for error correction
        code_distance: QEC code distance (if applicable)
    """
    name: str
    qubit_type: QubitType
    n_qubits: int
    T1: float              # Relaxation time (seconds)
    T2: float              # Dephasing time (seconds)
    gate_time: float       # Gate duration (seconds)
    error_rate: float      # Error per gate
    error_threshold: float = 0.01  # ~1% for surface codes
    code_distance: Optional[int] = None

    @property
    def coherence_ratio(self) -> float:
        """Ratio of coherence time to gate time."""
        T_eff = min(self.T1, self.T2)
        return T_eff / self.gate_time

    @property
    def operations_per_coherence(self) -> float:
        """Number of gates possible within coherence time."""
        return self.coherence_ratio

    @property
    def is_below_threshold(self) -> bool:
        """Check if operating below error threshold."""
        return self.error_rate < self.error_threshold


# =============================================================================
# DECOHERENCE MODELS
# =============================================================================

def amplitude_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply amplitude damping channel (T1 decay).

    Models energy relaxation: |1⟩ → |0⟩ with probability gamma.

    Kraus operators:
    K0 = [[1, 0], [0, sqrt(1-gamma)]]
    K1 = [[0, sqrt(gamma)], [0, 0]]
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])

    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def phase_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply phase damping channel (T2 dephasing).

    Models loss of phase coherence without energy loss.

    For qubit: off-diagonal elements decay by sqrt(1-gamma).
    """
    # Decay off-diagonal elements
    result = rho.copy()
    result[0, 1] *= np.sqrt(1 - gamma)
    result[1, 0] *= np.sqrt(1 - gamma)
    return result


def depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Apply depolarizing channel.

    With probability p, replace state with maximally mixed state.
    ρ → (1-p)ρ + (p/d)I

    This is the standard error model for quantum error correction.
    """
    d = rho.shape[0]
    I = np.eye(d) / d
    return (1 - p) * rho + p * I


def compute_T2_star(T1: float, T_phi: float) -> float:
    """
    Compute effective T2* from T1 and pure dephasing time.

    1/T2* = 1/(2*T1) + 1/T_phi

    T2* ≤ 2*T1 always (fundamental limit).
    """
    return 1.0 / (1.0 / (2 * T1) + 1.0 / T_phi)


# =============================================================================
# ERROR CORRECTION
# =============================================================================

def logical_error_rate(
    physical_error: float,
    threshold: float,
    code_distance: int,
    suppression_factor: float = 2.14  # From Google Willow
) -> float:
    """
    Compute logical error rate for surface code.

    Below threshold (p < p_th):
        ε_L ∝ (p / p_th)^((d+1)/2)

    The suppression factor Λ determines how much error decreases
    when code distance increases by 2.

    Google Willow achieved Λ = 2.14 ± 0.02 (2024).
    """
    if physical_error >= threshold:
        # Above threshold: errors accumulate
        return min(physical_error * code_distance, 1.0)

    # Below threshold: exponential suppression
    ratio = physical_error / threshold
    exponent = (code_distance + 1) / 2

    return ratio ** exponent


def required_code_distance(
    target_logical_error: float,
    physical_error: float,
    threshold: float
) -> int:
    """
    Compute code distance needed for target logical error rate.

    Solves: (p/p_th)^((d+1)/2) = ε_target

    Returns:
        Required code distance (odd integer)
    """
    if physical_error >= threshold:
        return float('inf')  # Impossible

    ratio = physical_error / threshold
    log_ratio = np.log(ratio)
    log_target = np.log(target_logical_error)

    # (d+1)/2 = log(ε_target) / log(ratio)
    d_plus_1_over_2 = log_target / log_ratio
    d = 2 * d_plus_1_over_2 - 1

    # Round up to nearest odd integer
    d_int = int(np.ceil(d))
    if d_int % 2 == 0:
        d_int += 1

    return d_int


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_coherence_theta(system: QubitSystem) -> float:
    """
    Compute theta based on coherence time vs gate time.

    theta = 1 - exp(-T_gate / T_coherence)

    High coherence (T >> T_gate): theta → 1 (quantum)
    Low coherence (T << T_gate): theta → 0 (classical)
    """
    T_eff = min(system.T1, system.T2)

    if T_eff <= 0:
        return 0.0

    # Ratio of gate time to coherence time
    ratio = system.gate_time / T_eff

    # theta = 1 when ratio → 0 (fast gates, long coherence)
    # theta = 0 when ratio → ∞ (slow gates, short coherence)
    theta = np.exp(-ratio)

    return np.clip(theta, 0.0, 1.0)


def compute_error_threshold_theta(system: QubitSystem) -> float:
    """
    Compute theta based on error rate vs threshold.

    theta = 1 - p / p_threshold (for p < p_threshold)
    theta = 0 (for p >= p_threshold)

    Below threshold: quantum error correction works → high theta
    Above threshold: noise wins → theta = 0
    """
    if system.error_rate >= system.error_threshold:
        return 0.0

    theta = 1.0 - system.error_rate / system.error_threshold
    return np.clip(theta, 0.0, 1.0)


def compute_quantum_computing_theta(system: QubitSystem) -> float:
    """
    Compute unified theta for quantum computing system.

    Combines:
    1. Coherence-based theta
    2. Error threshold theta

    A system needs BOTH long coherence AND low errors to be useful.
    """
    theta_coherence = compute_coherence_theta(system)
    theta_error = compute_error_threshold_theta(system)

    # Geometric mean: both must be high for overall theta to be high
    return np.sqrt(theta_coherence * theta_error)


def classify_coherence_regime(theta: float) -> CoherenceRegime:
    """Classify coherence regime from theta."""
    if theta > 0.9:
        return CoherenceRegime.HIGHLY_COHERENT
    elif theta > 0.7:
        return CoherenceRegime.COHERENT
    elif theta > 0.3:
        return CoherenceRegime.PARTIALLY_COHERENT
    elif theta > 0.1:
        return CoherenceRegime.MOSTLY_DECOHERENT
    else:
        return CoherenceRegime.CLASSICAL


# =============================================================================
# CURRENT QUANTUM HARDWARE (2024-2025)
# =============================================================================

QUANTUM_HARDWARE: Dict[str, QubitSystem] = {
    "google_sycamore": QubitSystem(
        name="Google Sycamore (2019)",
        qubit_type=QubitType.TRANSMON,
        n_qubits=53,
        T1=20e-6,      # 20 μs
        T2=10e-6,      # 10 μs
        gate_time=20e-9,  # 20 ns
        error_rate=0.006,  # 0.6% two-qubit
        error_threshold=0.01,
    ),
    "google_willow": QubitSystem(
        name="Google Willow (2024)",
        qubit_type=QubitType.TRANSMON,
        n_qubits=105,
        T1=68e-6,      # 68 μs (improved)
        T2=30e-6,      # ~30 μs
        gate_time=25e-9,
        error_rate=0.00143,  # 0.143% logical error per cycle!
        error_threshold=0.01,
        code_distance=7,  # Surface code d=7
    ),
    "ibm_heron": QubitSystem(
        name="IBM Heron (2024)",
        qubit_type=QubitType.TRANSMON,
        n_qubits=133,
        T1=200e-6,     # 200 μs
        T2=100e-6,     # 100 μs
        gate_time=70e-9,
        error_rate=0.003,  # 0.3%
        error_threshold=0.01,
    ),
    "ionq_forte": QubitSystem(
        name="IonQ Forte (2024)",
        qubit_type=QubitType.TRAPPED_ION,
        n_qubits=36,
        T1=10.0,       # 10 seconds!
        T2=1.0,        # 1 second
        gate_time=100e-6,  # 100 μs (slow but accurate)
        error_rate=0.003,
        error_threshold=0.01,
    ),
    "quantinuum_h2": QubitSystem(
        name="Quantinuum H2 (2024)",
        qubit_type=QubitType.TRAPPED_ION,
        n_qubits=56,
        T1=30.0,       # 30 seconds
        T2=3.0,        # 3 seconds
        gate_time=200e-6,
        error_rate=0.001,  # 0.1% (best 2-qubit fidelity)
        error_threshold=0.01,
    ),
    "nv_center_lab": QubitSystem(
        name="NV Center (Lab, 2024)",
        qubit_type=QubitType.NV_CENTER,
        n_qubits=10,
        T1=60.0,       # 1 minute at room temp!
        T2=2e-3,       # 2 ms (limited by nuclear bath)
        gate_time=1e-6,
        error_rate=0.01,
        error_threshold=0.01,
    ),
    "neutral_atom_quera": QubitSystem(
        name="QuEra Aquila (2024)",
        qubit_type=QubitType.NEUTRAL_ATOM,
        n_qubits=256,
        T1=1.0,        # ~1 second
        T2=0.5,        # ~0.5 seconds
        gate_time=1e-6,
        error_rate=0.005,
        error_threshold=0.01,
    ),
    "noisy_classical": QubitSystem(
        name="Highly Noisy System",
        qubit_type=QubitType.TRANSMON,
        n_qubits=10,
        T1=1e-6,       # 1 μs
        T2=0.5e-6,     # 0.5 μs
        gate_time=100e-9,
        error_rate=0.1,  # 10% error!
        error_threshold=0.01,
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def quantum_computing_theta_summary():
    """Print theta analysis for quantum hardware."""
    print("=" * 80)
    print("QUANTUM COMPUTING THETA ANALYSIS (Error Threshold Framework)")
    print("=" * 80)
    print()
    print(f"{'System':<25} {'θ':>8} {'θ_coh':>8} {'θ_err':>8} "
          f"{'T₁':>10} {'p':>10} {'Regime':<15}")
    print("-" * 80)

    for name, system in QUANTUM_HARDWARE.items():
        theta = compute_quantum_computing_theta(system)
        theta_coh = compute_coherence_theta(system)
        theta_err = compute_error_threshold_theta(system)
        regime = classify_coherence_regime(theta)

        # Format T1
        if system.T1 >= 1:
            T1_str = f"{system.T1:.1f}s"
        elif system.T1 >= 1e-3:
            T1_str = f"{system.T1*1e3:.1f}ms"
        else:
            T1_str = f"{system.T1*1e6:.1f}μs"

        # Format error rate
        p_str = f"{system.error_rate*100:.3f}%"

        below_threshold = "✓" if system.is_below_threshold else "✗"

        print(f"{system.name:<25} {theta:>8.3f} {theta_coh:>8.3f} {theta_err:>8.3f} "
              f"{T1_str:>10} {p_str:>10} {regime.value:<15}")

    print()
    print("Key: Google Willow (2024) first to achieve below-threshold operation!")


if __name__ == "__main__":
    quantum_computing_theta_summary()
