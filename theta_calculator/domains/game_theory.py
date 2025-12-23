"""
Game Theory Domain: Quantum Games and Entanglement Parameter

This module implements theta as the quantum-classical interpolation parameter
for strategic games using entanglement as the key measure.

## Mapping Definition

This domain maps game-theoretic systems to theta via entanglement:

**Inputs (Physical Analogs):**
- gamma (γ) → Entanglement parameter [0, π/2]
- game_type → Type of strategic game (Prisoner's Dilemma, Chicken, etc.)
- payoff_matrix → Reward structure for player strategies

**Theta Mapping:**
θ = sin²(γ)

Or equivalently: θ = 2γ/π (linear approximation)

**Interpretation:**
- θ → 0 (γ = 0): Classical Nash equilibria, no entanglement, classical strategies
- θ → 1 (γ = π/2): Quantum strategies with maximal entanglement

**Key Feature:** Quantum games enable new equilibria that don't exist classically.
Example: In quantum Prisoner's Dilemma (θ ≈ 1), cooperation can emerge.

**Important:** This is an ANALOGY SCORE based on quantum game theory.

References (see BIBLIOGRAPHY.bib):
    \\cite{Eisert1999} - Quantum Games and Quantum Strategies
    \\cite{Meyer1999} - Quantum Strategies
    \\cite{Du2002} - Experimental Realization of Quantum Games
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class GameType(Enum):
    """Types of strategic games."""
    PRISONERS_DILEMMA = "prisoners_dilemma"
    CHICKEN = "chicken"
    BATTLE_OF_SEXES = "battle_of_sexes"
    COORDINATION = "coordination"
    ZERO_SUM = "zero_sum"


class StrategyType(Enum):
    """Types of strategies available."""
    CLASSICAL = "classical"      # Standard mixed strategies
    QUANTUM = "quantum"          # Unitary operations on qubits
    HYBRID = "hybrid"            # Mix of classical and quantum


@dataclass
class QuantumGame:
    """
    A quantum game for theta analysis.

    In Eisert-Wilkens-Lewenstein (EWL) protocol:
    1. Initial state: |00⟩ (both players start with |0⟩)
    2. Entangling gate J(γ): Creates shared entanglement
    3. Player strategies: Unitary operations U_A, U_B
    4. Disentangling gate J†(γ)
    5. Measurement and payoff

    The entanglement parameter γ ∈ [0, π/2] controls quantum advantage.

    Attributes:
        name: Game identifier
        game_type: Type of strategic game
        payoff_matrix: Classical payoff matrix
        gamma: Entanglement parameter
        n_players: Number of players
        n_strategies: Number of strategies per player
    """
    name: str
    game_type: GameType
    payoff_matrix: np.ndarray  # Shape: (n_strategies, n_strategies, n_players)
    gamma: float  # Entanglement parameter [0, π/2]
    n_players: int = 2
    n_strategies: int = 2

    @property
    def is_fully_quantum(self) -> bool:
        """Check if game is at maximum entanglement."""
        return np.isclose(self.gamma, np.pi / 2)

    @property
    def is_classical(self) -> bool:
        """Check if game is classical (no entanglement)."""
        return np.isclose(self.gamma, 0.0)


# =============================================================================
# QUANTUM GATES AND OPERATIONS
# =============================================================================

def entangling_gate(gamma: float) -> np.ndarray:
    """
    EWL entangling gate J(γ).

    J(γ) = exp(i*γ*σ_x ⊗ σ_x / 2)

    For 2-qubit system:
    J(0) = I (identity, classical)
    J(π/2) = (I + i*σ_x⊗σ_x) / √2 (maximum entanglement)
    """
    cos_g = np.cos(gamma / 2)
    sin_g = np.sin(gamma / 2) * 1j

    # In computational basis |00⟩, |01⟩, |10⟩, |11⟩
    J = np.array([
        [cos_g, 0, 0, sin_g],
        [0, cos_g, sin_g, 0],
        [0, sin_g, cos_g, 0],
        [sin_g, 0, 0, cos_g]
    ])
    return J


def strategy_operator(theta: float, phi: float) -> np.ndarray:
    """
    General single-qubit strategy operator.

    U(θ, φ) = [[exp(iφ)cos(θ/2), sin(θ/2)],
               [-sin(θ/2), exp(-iφ)cos(θ/2)]]

    Special cases:
    - U(0, 0) = I: Cooperate
    - U(π, 0) = σ_x: Defect
    - U(0, π/2) = Phase gate
    """
    cos_t = np.cos(theta / 2)
    sin_t = np.sin(theta / 2)
    exp_p = np.exp(1j * phi)
    exp_m = np.exp(-1j * phi)

    return np.array([
        [exp_p * cos_t, sin_t],
        [-sin_t, exp_m * cos_t]
    ])


# Classical strategies
COOPERATE = strategy_operator(0, 0)  # Identity
DEFECT = strategy_operator(np.pi, 0)  # σ_x (bit flip)

# Quantum strategies (only available when γ > 0)
QUANTUM_STRATEGY = strategy_operator(0, np.pi / 2)  # "Magic" strategy


# =============================================================================
# PAYOFF CALCULATIONS
# =============================================================================

def classical_payoff(
    payoff_matrix: np.ndarray,
    strategy_a: int,
    strategy_b: int
) -> Tuple[float, float]:
    """Get classical payoff for pure strategies."""
    return payoff_matrix[strategy_a, strategy_b, 0], payoff_matrix[strategy_a, strategy_b, 1]


def quantum_payoff(
    payoff_matrix: np.ndarray,
    U_A: np.ndarray,
    U_B: np.ndarray,
    gamma: float
) -> Tuple[float, float]:
    """
    Compute payoff for quantum strategies.

    Protocol:
    1. Start with |00⟩
    2. Apply J(γ)
    3. Apply U_A ⊗ U_B
    4. Apply J†(γ)
    5. Measure in computational basis
    6. Compute expected payoff
    """
    # Initial state
    psi_0 = np.array([1, 0, 0, 0])  # |00⟩

    # Entangling gate
    J = entangling_gate(gamma)

    # Player strategies (tensor product)
    U_AB = np.kron(U_A, U_B)

    # Final state
    psi_f = J.conj().T @ U_AB @ J @ psi_0

    # Probabilities
    probs = np.abs(psi_f) ** 2  # P(00), P(01), P(10), P(11)

    # Expected payoffs
    payoff_A = sum(probs[i*2 + j] * payoff_matrix[i, j, 0]
                   for i in range(2) for j in range(2))
    payoff_B = sum(probs[i*2 + j] * payoff_matrix[i, j, 1]
                   for i in range(2) for j in range(2))

    return float(np.real(payoff_A)), float(np.real(payoff_B))


def prisoners_dilemma_payoff(cooperate_a: bool, cooperate_b: bool) -> Tuple[int, int]:
    """
    Classic Prisoner's Dilemma payoffs.

    Payoff matrix (row player A, column player B):
             B: Coop   B: Defect
    A: Coop    (3,3)      (0,5)
    A: Defect  (5,0)      (1,1)

    Classical Nash: Both defect (1,1)
    Quantum Nash: Both cooperate (3,3) possible!
    """
    if cooperate_a and cooperate_b:
        return (3, 3)
    elif cooperate_a and not cooperate_b:
        return (0, 5)
    elif not cooperate_a and cooperate_b:
        return (5, 0)
    else:
        return (1, 1)


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_entanglement_theta(game: QuantumGame) -> float:
    """
    Compute theta for a quantum game.

    theta = γ / (π/2) = 2γ/π

    At γ = 0: Classical game (theta = 0)
    At γ = π/2: Maximum entanglement (theta = 1)

    Returns:
        theta in [0, 1]
    """
    return 2 * game.gamma / np.pi


def compute_quantum_advantage(game: QuantumGame) -> float:
    """
    Compute the quantum advantage in payoff.

    For Prisoner's Dilemma:
    - Classical optimal: (1, 1) from mutual defection
    - Quantum optimal: (3, 3) from quantum cooperation

    Returns:
        Quantum payoff / Classical payoff ratio
    """
    # Classical Nash equilibrium payoff
    classical_nash_payoff = 1.0  # Both defect in PD

    # Quantum optimal (both use quantum strategy)
    payoff_A, payoff_B = quantum_payoff(
        game.payoff_matrix,
        QUANTUM_STRATEGY,
        QUANTUM_STRATEGY,
        game.gamma
    )

    if classical_nash_payoff == 0:
        return float('inf') if payoff_A > 0 else 1.0

    return (payoff_A + payoff_B) / (2 * classical_nash_payoff)


# =============================================================================
# EXAMPLE GAME SYSTEMS
# =============================================================================

# Prisoner's Dilemma payoff matrix
PD_PAYOFF = np.array([
    [[3, 3], [0, 5]],   # A cooperates: (C,C)=(3,3), (C,D)=(0,5)
    [[5, 0], [1, 1]]    # A defects:    (D,C)=(5,0), (D,D)=(1,1)
])

# Chicken (Hawk-Dove) payoff matrix
CHICKEN_PAYOFF = np.array([
    [[0, 0], [-1, 1]],   # Swerve vs Swerve, Swerve vs Straight
    [[1, -1], [-10, -10]]  # Straight vs Swerve, Straight vs Straight
])

# Battle of Sexes payoff matrix
BOS_PAYOFF = np.array([
    [[3, 2], [0, 0]],   # Opera vs Opera, Opera vs Football
    [[0, 0], [2, 3]]    # Football vs Opera, Football vs Football
])

GAME_SYSTEMS: Dict[str, QuantumGame] = {
    "classical_pd": QuantumGame(
        name="Classical Prisoner's Dilemma",
        game_type=GameType.PRISONERS_DILEMMA,
        payoff_matrix=PD_PAYOFF,
        gamma=0.0,
    ),
    "partial_quantum_pd": QuantumGame(
        name="Partial Quantum PD (γ=π/4)",
        game_type=GameType.PRISONERS_DILEMMA,
        payoff_matrix=PD_PAYOFF,
        gamma=np.pi / 4,
    ),
    "quantum_pd": QuantumGame(
        name="Quantum Prisoner's Dilemma",
        game_type=GameType.PRISONERS_DILEMMA,
        payoff_matrix=PD_PAYOFF,
        gamma=np.pi / 2,
    ),
    "classical_chicken": QuantumGame(
        name="Classical Chicken",
        game_type=GameType.CHICKEN,
        payoff_matrix=CHICKEN_PAYOFF,
        gamma=0.0,
    ),
    "quantum_chicken": QuantumGame(
        name="Quantum Chicken",
        game_type=GameType.CHICKEN,
        payoff_matrix=CHICKEN_PAYOFF,
        gamma=np.pi / 2,
    ),
    "classical_bos": QuantumGame(
        name="Classical Battle of Sexes",
        game_type=GameType.BATTLE_OF_SEXES,
        payoff_matrix=BOS_PAYOFF,
        gamma=0.0,
    ),
    "quantum_bos": QuantumGame(
        name="Quantum Battle of Sexes",
        game_type=GameType.BATTLE_OF_SEXES,
        payoff_matrix=BOS_PAYOFF,
        gamma=np.pi / 2,
    ),
}


# =============================================================================
# NASH EQUILIBRIUM ANALYSIS
# =============================================================================

def find_classical_nash(payoff_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find Nash equilibria for 2x2 game (pure strategies).

    A Nash equilibrium is where neither player can improve by
    unilaterally changing strategy.
    """
    nash = []
    n = payoff_matrix.shape[0]

    for i in range(n):
        for j in range(n):
            # Check if (i,j) is Nash
            a_payoff = payoff_matrix[i, j, 0]
            b_payoff = payoff_matrix[i, j, 1]

            # Can A improve by switching?
            a_can_improve = any(
                payoff_matrix[k, j, 0] > a_payoff for k in range(n) if k != i
            )

            # Can B improve by switching?
            b_can_improve = any(
                payoff_matrix[i, k, 1] > b_payoff for k in range(n) if k != j
            )

            if not a_can_improve and not b_can_improve:
                nash.append((i, j))

    return nash


# =============================================================================
# DEMONSTRATION
# =============================================================================

def game_theory_theta_summary():
    """Print theta analysis for all example games."""
    print("=" * 70)
    print("GAME THEORY THETA ANALYSIS (Entanglement Framework)")
    print("=" * 70)
    print()
    print(f"{'Game':<35} {'θ':>8} {'γ':>10} {'Advantage':>12}")
    print("-" * 70)

    for name, game in GAME_SYSTEMS.items():
        theta = compute_entanglement_theta(game)

        # Only compute advantage for PD games
        if game.game_type == GameType.PRISONERS_DILEMMA:
            advantage = compute_quantum_advantage(game)
            adv_str = f"{advantage:.2f}x"
        else:
            adv_str = "N/A"

        gamma_str = f"{game.gamma:.4f}" if game.gamma < 1 else "π/2"
        if np.isclose(game.gamma, np.pi/4):
            gamma_str = "π/4"
        elif np.isclose(game.gamma, np.pi/2):
            gamma_str = "π/2"

        print(f"{game.name:<35} {theta:>8.3f} {gamma_str:>10} {adv_str:>12}")

    print()
    print("Key Insight: At γ = π/2, quantum strategies enable cooperation")
    print("in Prisoner's Dilemma where classical Nash is mutual defection!")


if __name__ == "__main__":
    game_theory_theta_summary()
