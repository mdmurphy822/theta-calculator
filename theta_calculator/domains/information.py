"""
Information Theory Domain: Shannon vs Von Neumann Entropy

This module implements theta as the quantum-classical interpolation parameter
for information systems using entropy measures.

## Mapping Definition

This domain maps information systems to theta via entropy and purity:

**Inputs (Physical Analogs):**
- dimension (d) → Hilbert space size / number of possible states
- purity (P) → Tr(ρ²), ranges from 1/d (maximally mixed) to 1 (pure)
- entropy_shannon → Classical Shannon entropy H(X)
- entropy_von_neumann → Quantum von Neumann entropy S(ρ)

**Theta Mapping:**
θ = S(ρ) / S_max = S(ρ) / log(d)

Or equivalently: θ = 1 - P (for normalized purity)

**Interpretation:**
- θ → 0: Pure/deterministic state (low entropy, highly ordered information)
- θ → 1: Maximally mixed state (maximum entropy, maximum uncertainty)

**Key Quantum Feature:** Quantum entropy is NON-ADDITIVE due to entanglement.
S(A∪B) ≤ S(A) + S(B) for classical, but S(A∪B) can be < S(A) for quantum.

**Important:** This is an ANALOGY SCORE mapping entropy/purity to the [0,1] scale.

References (see BIBLIOGRAPHY.bib):
    \\cite{Shannon1948} - A Mathematical Theory of Communication
    \\cite{VonNeumann1932} - Mathematical Foundations of Quantum Mechanics
    \\cite{Hastings2009} - Non-additivity of quantum channel capacity
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class InformationRegime(Enum):
    """Information regime classification based on theta."""
    DETERMINISTIC = "deterministic"  # theta < 0.1: Pure state
    LOW_ENTROPY = "low_entropy"      # 0.1 <= theta < 0.3
    MODERATE = "moderate"            # 0.3 <= theta < 0.7
    HIGH_ENTROPY = "high_entropy"    # 0.7 <= theta < 0.9
    MAXIMALLY_MIXED = "max_mixed"    # theta >= 0.9


@dataclass
class InformationSystem:
    """
    An information system for theta analysis.

    Attributes:
        name: System identifier
        dimension: Hilbert space dimension (number of states)
        probabilities: Classical probability distribution
        density_matrix: Quantum density matrix (if applicable)
        entropy_shannon: Classical Shannon entropy
        entropy_von_neumann: Quantum von Neumann entropy
        purity: Tr(ρ²), quantum purity measure
    """
    name: str
    dimension: int
    probabilities: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    entropy_shannon: Optional[float] = None
    entropy_von_neumann: Optional[float] = None
    purity: Optional[float] = None

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy for this dimension."""
        return np.log2(self.dimension)


# =============================================================================
# ENTROPY CALCULATIONS
# =============================================================================

def compute_shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Compute classical Shannon entropy.

    H(X) = -∑_i p_i log₂(p_i)

    Properties:
    - H ≥ 0
    - H = 0 iff one p_i = 1 (deterministic)
    - H_max = log₂(n) for uniform distribution

    Args:
        probabilities: Probability distribution (must sum to 1)

    Returns:
        Shannon entropy in bits
    """
    # Filter out zeros to avoid log(0)
    p = probabilities[probabilities > 0]
    return -np.sum(p * np.log2(p))


def compute_von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """
    Compute quantum von Neumann entropy.

    S(ρ) = -Tr(ρ log₂ ρ) = -∑_i λ_i log₂(λ_i)

    Where λ_i are eigenvalues of ρ.

    Properties:
    - S ≥ 0
    - S = 0 iff ρ is a pure state (ρ = |ψ⟩⟨ψ|)
    - S_max = log₂(d) for maximally mixed state (ρ = I/d)
    - S(ρ_AB) can be < S(ρ_A) for entangled states!

    Args:
        density_matrix: Quantum density matrix (Hermitian, Tr(ρ) = 1)

    Returns:
        Von Neumann entropy in bits
    """
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(density_matrix)

    # Filter out zeros and negative values (numerical errors)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    return -np.sum(eigenvalues * np.log2(eigenvalues))


def compute_purity(density_matrix: np.ndarray) -> float:
    """
    Compute quantum purity.

    P = Tr(ρ²)

    Properties:
    - 1/d ≤ P ≤ 1
    - P = 1 iff pure state
    - P = 1/d iff maximally mixed

    The purity directly relates to theta:
    - High purity (P → 1): Classical-like (definite state)
    - Low purity (P → 1/d): Quantum-like (uncertain state)
    """
    return np.real(np.trace(density_matrix @ density_matrix))


def compute_linear_entropy(density_matrix: np.ndarray) -> float:
    """
    Compute linear entropy (simpler alternative to von Neumann).

    S_L = (d/(d-1)) * (1 - Tr(ρ²)) = (d/(d-1)) * (1 - P)

    Ranges from 0 (pure) to 1 (maximally mixed).
    """
    d = density_matrix.shape[0]
    P = compute_purity(density_matrix)
    if d == 1:
        return 0.0
    return (d / (d - 1)) * (1 - P)


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_information_theta(system: InformationSystem) -> float:
    """
    Compute theta for an information system.

    Theta measures deviation from pure/deterministic state.

    Methods:
    1. Purity-based: theta = 1 - P (for quantum systems)
    2. Entropy-based: theta = S / S_max
    3. Linear entropy: theta = S_L

    Returns:
        theta in [0, 1] where:
        - 0 = pure/deterministic state
        - 1 = maximally mixed state
    """
    if system.density_matrix is not None:
        # Quantum system: use purity
        P = system.purity if system.purity else compute_purity(system.density_matrix)
        d = system.dimension

        # Normalize purity to [0, 1]
        # P ranges from 1/d to 1
        # theta = (1 - P) / (1 - 1/d)
        P_min = 1.0 / d
        if P >= 1.0:
            return 0.0
        theta = (1 - P) / (1 - P_min)
        return np.clip(theta, 0.0, 1.0)

    elif system.probabilities is not None:
        # Classical system: use entropy ratio
        S = compute_shannon_entropy(system.probabilities)
        S_max = system.max_entropy
        if S_max == 0:
            return 0.0
        return np.clip(S / S_max, 0.0, 1.0)

    elif system.entropy_shannon is not None:
        # Use provided entropy
        S_max = system.max_entropy
        return np.clip(system.entropy_shannon / S_max, 0.0, 1.0)

    return 0.5  # Default: unknown


def classify_information_regime(theta: float) -> InformationRegime:
    """Classify information regime from theta."""
    if theta < 0.1:
        return InformationRegime.DETERMINISTIC
    elif theta < 0.3:
        return InformationRegime.LOW_ENTROPY
    elif theta < 0.7:
        return InformationRegime.MODERATE
    elif theta < 0.9:
        return InformationRegime.HIGH_ENTROPY
    else:
        return InformationRegime.MAXIMALLY_MIXED


# =============================================================================
# SPECIAL QUANTUM STATES
# =============================================================================

def pure_state_density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Create density matrix for pure state: ρ = |ψ⟩⟨ψ|"""
    psi = state_vector.reshape(-1, 1)
    return psi @ psi.conj().T


def maximally_mixed_state(dimension: int) -> np.ndarray:
    """Create maximally mixed state: ρ = I/d"""
    return np.eye(dimension) / dimension


def thermal_state(hamiltonian: np.ndarray, temperature: float) -> np.ndarray:
    """
    Create thermal (Gibbs) state: ρ = exp(-H/kT) / Z

    In natural units where k = 1.

    Uses eigendecomposition: exp(-βH) = V exp(-βD) V†
    where H = V D V†
    """
    if temperature <= 0:
        # Ground state (T = 0)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        return pure_state_density_matrix(eigenvectors[:, 0])

    beta = 1.0 / temperature

    # Eigendecomposition for matrix exponential
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    exp_eigenvalues = np.exp(-beta * eigenvalues)

    # Reconstruct exp(-βH) = V diag(exp(-βλ)) V†
    exp_H = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.conj().T

    Z = np.trace(exp_H)
    return exp_H / Z


def bell_state(which: str = "phi_plus") -> np.ndarray:
    """
    Create Bell state density matrix.

    Bell states are maximally entangled 2-qubit states.
    For the reduced state of one qubit, they give maximally mixed!

    States:
    - phi_plus: (|00⟩ + |11⟩) / √2
    - phi_minus: (|00⟩ - |11⟩) / √2
    - psi_plus: (|01⟩ + |10⟩) / √2
    - psi_minus: (|01⟩ - |10⟩) / √2
    """
    sqrt2 = np.sqrt(2)
    if which == "phi_plus":
        psi = np.array([1, 0, 0, 1]) / sqrt2
    elif which == "phi_minus":
        psi = np.array([1, 0, 0, -1]) / sqrt2
    elif which == "psi_plus":
        psi = np.array([0, 1, 1, 0]) / sqrt2
    elif which == "psi_minus":
        psi = np.array([0, 1, -1, 0]) / sqrt2
    else:
        raise ValueError(f"Unknown Bell state: {which}")

    return pure_state_density_matrix(psi)


def partial_trace(rho: np.ndarray, dims: Tuple[int, int], trace_out: int) -> np.ndarray:
    """
    Compute partial trace of bipartite density matrix.

    For Bell states, tracing out one qubit gives maximally mixed state
    even though the full state is pure! This is the quantum-classical
    boundary: local measurements appear random (theta = 1) but the
    global state is pure (theta = 0).
    """
    d1, d2 = dims
    if trace_out == 0:
        # Trace out first system
        rho_reshaped = rho.reshape(d1, d2, d1, d2)
        return np.trace(rho_reshaped, axis1=0, axis2=2)
    else:
        # Trace out second system
        rho_reshaped = rho.reshape(d1, d2, d1, d2)
        return np.trace(rho_reshaped, axis1=1, axis2=3)


# =============================================================================
# EXAMPLE INFORMATION SYSTEMS
# =============================================================================

# Pure state (|0⟩)
_pure_state = pure_state_density_matrix(np.array([1, 0]))

# Maximally mixed qubit (I/2)
_mixed_qubit = maximally_mixed_state(2)

# Thermal qubit at various temperatures
_thermal_hot = thermal_state(np.array([[0, 0], [0, 1]]), temperature=10.0)
_thermal_cold = thermal_state(np.array([[0, 0], [0, 1]]), temperature=0.1)

# Bell state reduced density matrix (trace out one qubit)
_bell_full = bell_state("phi_plus")
_bell_reduced = partial_trace(_bell_full, (2, 2), trace_out=1)

INFORMATION_SYSTEMS: Dict[str, InformationSystem] = {
    "pure_qubit": InformationSystem(
        name="Pure Qubit |0⟩",
        dimension=2,
        density_matrix=_pure_state,
        purity=1.0,
        entropy_von_neumann=0.0,
    ),
    "mixed_qubit": InformationSystem(
        name="Maximally Mixed Qubit",
        dimension=2,
        density_matrix=_mixed_qubit,
        purity=0.5,
        entropy_von_neumann=1.0,
    ),
    "thermal_hot": InformationSystem(
        name="Hot Thermal Qubit (T=10)",
        dimension=2,
        density_matrix=_thermal_hot,
        purity=compute_purity(_thermal_hot),
    ),
    "thermal_cold": InformationSystem(
        name="Cold Thermal Qubit (T=0.1)",
        dimension=2,
        density_matrix=_thermal_cold,
        purity=compute_purity(_thermal_cold),
    ),
    "bell_reduced": InformationSystem(
        name="Entangled Qubit (Bell state reduced)",
        dimension=2,
        density_matrix=_bell_reduced,
        purity=compute_purity(_bell_reduced),
    ),
    "fair_coin": InformationSystem(
        name="Fair Coin (classical)",
        dimension=2,
        probabilities=np.array([0.5, 0.5]),
        entropy_shannon=1.0,
    ),
    "biased_coin": InformationSystem(
        name="Biased Coin (90/10)",
        dimension=2,
        probabilities=np.array([0.9, 0.1]),
        entropy_shannon=compute_shannon_entropy(np.array([0.9, 0.1])),
    ),
    "deterministic": InformationSystem(
        name="Deterministic (100/0)",
        dimension=2,
        probabilities=np.array([1.0, 0.0]),
        entropy_shannon=0.0,
    ),
    "uniform_die": InformationSystem(
        name="Fair 6-sided Die",
        dimension=6,
        probabilities=np.ones(6) / 6,
        entropy_shannon=np.log2(6),
    ),
    "8_level_mixed": InformationSystem(
        name="8-level Maximally Mixed",
        dimension=8,
        density_matrix=maximally_mixed_state(8),
        purity=1/8,
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def information_theta_summary():
    """Print theta analysis for all example information systems."""
    print("=" * 70)
    print("INFORMATION THETA ANALYSIS (Entropy Framework)")
    print("=" * 70)
    print()
    print(f"{'System':<30} {'θ':>8} {'S':>8} {'P':>8} {'Regime':<15}")
    print("-" * 70)

    for name, system in INFORMATION_SYSTEMS.items():
        theta = compute_information_theta(system)
        regime = classify_information_regime(theta)

        # Get entropy
        if system.entropy_von_neumann is not None:
            S = system.entropy_von_neumann
        elif system.entropy_shannon is not None:
            S = system.entropy_shannon
        elif system.density_matrix is not None:
            S = compute_von_neumann_entropy(system.density_matrix)
        elif system.probabilities is not None:
            S = compute_shannon_entropy(system.probabilities)
        else:
            S = None

        # Get purity
        P = system.purity
        if P is None and system.density_matrix is not None:
            P = compute_purity(system.density_matrix)

        S_str = f"{S:.3f}" if S is not None else "N/A"
        P_str = f"{P:.3f}" if P is not None else "N/A"

        print(f"{system.name:<30} {theta:>8.3f} {S_str:>8} {P_str:>8} {regime.value:<15}")


if __name__ == "__main__":
    information_theta_summary()
