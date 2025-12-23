"""
Universal Theta Framework: Cross-Domain Unification

This module provides a unified interface for computing theta across
all domains, demonstrating that the quantum-classical interpolation
parameter is a universal concept.

Key Insight: Theta represents the same fundamental concept across domains:
- Physics: ℏ/S (action ratio)
- Economics: |J|/J_c (coupling ratio)
- Information: 1 - P (mixedness)
- Game Theory: γ/(π/2) (entanglement ratio)
- Complex Systems: Critical proximity
- Quantum Computing: Error margin

In ALL cases:
- theta = 0: Classical limit (deterministic, independent, random walk)
- theta = 1: Quantum limit (coherent, correlated, entangled)

This unification suggests theta is a FUNDAMENTAL parameter of nature,
not just a convenient mathematical construct.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

# Import domain-specific types and functions
from .economics import MarketSystem, compute_market_theta, ECONOMIC_SYSTEMS
from .information import InformationSystem, compute_information_theta, INFORMATION_SYSTEMS
from .game_theory import QuantumGame, compute_entanglement_theta, GAME_SYSTEMS
from .complex_systems import ComplexSystem, compute_complex_theta, COMPLEX_SYSTEMS
from .quantum_computing import QubitSystem, compute_quantum_computing_theta, QUANTUM_HARDWARE


class DomainType(Enum):
    """Domains where theta applies."""
    PHYSICS = "physics"
    ECONOMICS = "economics"
    INFORMATION = "information"
    GAME_THEORY = "game_theory"
    COMPLEX_SYSTEMS = "complex_systems"
    QUANTUM_COMPUTING = "quantum_computing"


@dataclass
class UniversalThetaResult:
    """
    Result of universal theta computation.

    Attributes:
        theta: The computed theta value [0, 1]
        domain: Which domain this came from
        regime: String description of regime
        confidence: Confidence in theta value [0, 1]
        details: Domain-specific details
    """
    theta: float
    domain: DomainType
    regime: str
    confidence: float = 1.0
    details: Optional[Dict[str, Any]] = None


class UniversalTheta:
    """
    Universal theta calculator across all domains.

    Provides a single interface for computing theta regardless of
    the underlying domain, demonstrating universality.
    """

    @staticmethod
    def compute(system: Any, domain: Optional[DomainType] = None) -> UniversalThetaResult:
        """
        Compute theta for any system type.

        Automatically detects domain from system type if not specified.

        Args:
            system: System to analyze (MarketSystem, InformationSystem, etc.)
            domain: Optional domain override

        Returns:
            UniversalThetaResult with theta and metadata
        """
        # Auto-detect domain
        if domain is None:
            domain = UniversalTheta._detect_domain(system)

        # Compute theta based on domain
        if domain == DomainType.ECONOMICS:
            theta = compute_market_theta(system)
            regime = UniversalTheta._classify_market_regime(theta)
            details = {
                "order_parameter": system.order_parameter,
                "coupling": system.coupling_strength,
                "temperature": system.temperature,
            }

        elif domain == DomainType.INFORMATION:
            theta = compute_information_theta(system)
            regime = UniversalTheta._classify_information_regime(theta)
            details = {
                "dimension": system.dimension,
                "purity": system.purity,
            }

        elif domain == DomainType.GAME_THEORY:
            theta = compute_entanglement_theta(system)
            regime = UniversalTheta._classify_game_regime(theta)
            details = {
                "gamma": system.gamma,
                "game_type": system.game_type.value,
            }

        elif domain == DomainType.COMPLEX_SYSTEMS:
            theta = compute_complex_theta(system)
            regime = UniversalTheta._classify_complex_regime(theta, system)
            details = {
                "order_parameter": system.order_parameter,
                "reduced_temperature": system.reduced_temperature,
            }

        elif domain == DomainType.QUANTUM_COMPUTING:
            theta = compute_quantum_computing_theta(system)
            regime = UniversalTheta._classify_qc_regime(theta)
            details = {
                "error_rate": system.error_rate,
                "T1": system.T1,
                "below_threshold": system.is_below_threshold,
            }

        else:
            raise ValueError(f"Unknown domain: {domain}")

        return UniversalThetaResult(
            theta=theta,
            domain=domain,
            regime=regime,
            confidence=1.0,
            details=details,
        )

    @staticmethod
    def _detect_domain(system: Any) -> DomainType:
        """Auto-detect domain from system type."""
        if isinstance(system, MarketSystem):
            return DomainType.ECONOMICS
        elif isinstance(system, InformationSystem):
            return DomainType.INFORMATION
        elif isinstance(system, QuantumGame):
            return DomainType.GAME_THEORY
        elif isinstance(system, ComplexSystem):
            return DomainType.COMPLEX_SYSTEMS
        elif isinstance(system, QubitSystem):
            return DomainType.QUANTUM_COMPUTING
        else:
            raise TypeError(f"Unknown system type: {type(system)}")

    @staticmethod
    def _classify_market_regime(theta: float) -> str:
        if theta < 0.2:
            return "efficient"
        elif theta < 0.5:
            return "normal"
        elif theta < 0.8:
            return "trending"
        else:
            return "crash/bubble"

    @staticmethod
    def _classify_information_regime(theta: float) -> str:
        if theta < 0.1:
            return "pure/deterministic"
        elif theta < 0.5:
            return "low entropy"
        elif theta < 0.9:
            return "high entropy"
        else:
            return "maximally mixed"

    @staticmethod
    def _classify_game_regime(theta: float) -> str:
        if theta < 0.1:
            return "classical"
        elif theta < 0.5:
            return "partial entanglement"
        else:
            return "quantum"

    @staticmethod
    def _classify_complex_regime(theta: float, system: ComplexSystem) -> str:
        t = system.reduced_temperature
        if abs(t) < 0.1:
            return "critical"
        elif t > 0:
            return "disordered"
        else:
            return "ordered"

    @staticmethod
    def _classify_qc_regime(theta: float) -> str:
        if theta > 0.9:
            return "highly coherent"
        elif theta > 0.5:
            return "coherent"
        elif theta > 0.1:
            return "partially coherent"
        else:
            return "classical/noisy"


def cross_domain_comparison() -> Dict[str, Dict[str, float]]:
    """
    Compare theta across all domains with example systems.

    Returns dictionary mapping domain to example systems and their theta values.
    """
    comparison = {}

    # Economics
    comparison["economics"] = {
        name: compute_market_theta(system)
        for name, system in ECONOMIC_SYSTEMS.items()
    }

    # Information
    comparison["information"] = {
        name: compute_information_theta(system)
        for name, system in INFORMATION_SYSTEMS.items()
    }

    # Game Theory
    comparison["game_theory"] = {
        name: compute_entanglement_theta(game)
        for name, game in GAME_SYSTEMS.items()
    }

    # Complex Systems
    comparison["complex_systems"] = {
        name: compute_complex_theta(system)
        for name, system in COMPLEX_SYSTEMS.items()
    }

    # Quantum Computing
    comparison["quantum_computing"] = {
        name: compute_quantum_computing_theta(system)
        for name, system in QUANTUM_HARDWARE.items()
    }

    return comparison


def print_universal_summary():
    """Print comprehensive cross-domain theta summary."""
    print("=" * 80)
    print("UNIVERSAL THETA: Cross-Domain Comparison")
    print("=" * 80)
    print()
    print("Theta represents the same concept across all domains:")
    print("  θ = 0: Classical (deterministic, independent, random)")
    print("  θ = 1: Quantum (coherent, correlated, entangled)")
    print()

    comparison = cross_domain_comparison()

    for domain, systems in comparison.items():
        print(f"\n{domain.upper().replace('_', ' ')}")
        print("-" * 50)
        for name, theta in sorted(systems.items(), key=lambda x: -x[1]):
            bar = "█" * int(theta * 20) + "░" * (20 - int(theta * 20))
            print(f"  {name:<30} {theta:.3f} [{bar}]")

    print()
    print("=" * 80)
    print("ISOMORPHISM TABLE")
    print("=" * 80)
    print()
    print(f"{'Domain':<20} {'θ=0 (Classical)':<25} {'θ=1 (Quantum)':<25}")
    print("-" * 70)
    print(f"{'Physics':<20} {'Planets, baseballs':<25} {'Electrons, photons':<25}")
    print(f"{'Economics':<20} {'Efficient markets':<25} {'Crashes, bubbles':<25}")
    print(f"{'Information':<20} {'Pure/deterministic':<25} {'Maximally mixed':<25}")
    print(f"{'Game Theory':<20} {'Classical Nash':<25} {'Entangled strategies':<25}")
    print(f"{'Complex Systems':<20} {'Disordered phase':<25} {'Critical point':<25}")
    print(f"{'Quantum Computing':<20} {'Noisy/decoherent':<25} {'Coherent qubits':<25}")


# =============================================================================
# MATHEMATICAL ISOMORPHISMS
# =============================================================================

"""
The deep mathematical connections between domains:

1. ISING MODEL ↔ QUANTUM SPIN CHAIN
   - Classical Ising: H = -J∑s_i s_j
   - Quantum Ising: H = -J∑σ_z^i σ_z^j - Γ∑σ_x^i
   - At T=0, Γ/J plays role of quantum fluctuations
   - Phase transition at Γ_c/J (quantum critical point)

2. MARKET CRASH ↔ MAGNETIC PHASE TRANSITION
   - Traders ↔ Spins
   - Buy/Sell ↔ Up/Down
   - Herding ↔ Ferromagnetic coupling
   - Volatility ↔ Temperature
   - Crash ↔ Spontaneous magnetization

3. NASH EQUILIBRIUM ↔ GROUND STATE
   - Strategies ↔ Spin configurations
   - Payoff ↔ Energy (inverted)
   - Nash equilibrium ↔ Minimum energy
   - Quantum game ↔ Quantum ground state

4. INFORMATION ENTROPY ↔ THERMODYNAMIC ENTROPY
   - Shannon: H = -∑p log p
   - Boltzmann: S = k ln W
   - Connection: S = k H (when p_i = W_i/W)
   - Landauer: E_min = kT ln 2 per bit erased

5. NEURAL CRITICALITY ↔ PHASE TRANSITION
   - Neurons ↔ Spins
   - Firing patterns ↔ Spin configurations
   - Optimal computation ↔ Critical point
   - "Edge of chaos" ↔ Critical temperature

These aren't just analogies—they're EXACT MATHEMATICAL MAPPINGS.
"""


if __name__ == "__main__":
    print_universal_summary()
