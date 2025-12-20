"""
Multi-Domain Theta Framework

This module extends theta beyond physics to economics, game theory,
information theory, complex systems, quantum computing, quantum biology,
and cosmology.

Key Insight: Theta represents a universal interpolation parameter
between discrete/quantum-like and continuous/classical behavior
across ALL domains exhibiting phase transitions.

Domains:
- economics: Ising model of markets, phase transitions
- information: Shannon vs von Neumann entropy
- game_theory: Entanglement parameter in quantum games
- complex_systems: Critical exponents, social phase transitions
- quantum_computing: Error thresholds, decoherence
- quantum_biology: Coherence in photosynthesis, tunneling in enzymes
- cosmology: Theta evolution from Big Bang to heat death

Each domain maps to the universal theta framework:
    theta = 0: Classical limit (random, independent, deterministic)
    theta = 1: Quantum limit (correlated, entangled, coherent)
"""

from .economics import (
    MarketSystem,
    IsingMarket,
    compute_market_theta,
    ECONOMIC_SYSTEMS,
    compute_coupling_from_correlation,
    detect_phase_transition,
)

from .information import (
    InformationSystem,
    compute_shannon_entropy,
    compute_von_neumann_entropy,
    compute_purity,
    compute_information_theta,
    INFORMATION_SYSTEMS,
)

from .game_theory import (
    QuantumGame,
    compute_entanglement_theta,
    GAME_SYSTEMS,
    prisoners_dilemma_payoff,
)

from .complex_systems import (
    ComplexSystem,
    CriticalExponents,
    compute_order_parameter,
    compute_susceptibility,
    compute_correlation_length,
    detect_critical_point,
    compute_complex_theta,
    COMPLEX_SYSTEMS,
)

from .quantum_computing import (
    QubitSystem,
    compute_coherence_theta,
    compute_error_threshold_theta,
    compute_quantum_computing_theta,
    QUANTUM_HARDWARE,
)

from .quantum_biology import (
    BiologicalSystem,
    compute_quantum_bio_theta,
    compute_tunneling_theta,
    BIOLOGICAL_SYSTEMS,
)

from .cosmology import (
    CosmicEpoch,
    compute_cosmic_theta,
    compute_thermal_theta,
    COSMIC_TIMELINE,
)

from .control_theory import (
    ControlSystem,
    ControllerType,
    StabilityRegime,
    compute_control_theta,
    compute_gain_margin_theta,
    compute_phase_margin_theta,
    classify_stability,
    CONTROL_SYSTEMS,
)

from .nonlinear_dynamics import (
    DynamicalSystem,
    DynamicalRegime,
    AttractorType,
    compute_dynamics_theta,
    compute_lyapunov_theta,
    classify_regime as classify_dynamics_regime,
    logistic_map,
    logistic_lyapunov,
    DYNAMICAL_SYSTEMS,
    FEIGENBAUM_DELTA,
    FEIGENBAUM_ALPHA,
)

from .quantum_gravity import (
    QuantumGravitySystem,
    SpacetimeRegime,
    QuantumGravityTheory,
    compute_quantum_gravity_theta,
    compute_length_theta,
    compute_energy_theta,
    classify_regime as classify_spacetime_regime,
    black_hole_theta,
    hawking_temperature_k,
    bekenstein_entropy,
    QUANTUM_GRAVITY_SYSTEMS,
    L_PLANCK,
    E_PLANCK_EV,
    M_PLANCK,
)

from .universal import (
    UniversalTheta,
    DomainType,
    cross_domain_comparison,
)

__all__ = [
    # Economics
    "MarketSystem",
    "IsingMarket",
    "compute_market_theta",
    "ECONOMIC_SYSTEMS",
    "compute_coupling_from_correlation",
    "detect_phase_transition",
    # Information
    "InformationSystem",
    "compute_shannon_entropy",
    "compute_von_neumann_entropy",
    "compute_purity",
    "compute_information_theta",
    "INFORMATION_SYSTEMS",
    # Game Theory
    "QuantumGame",
    "compute_entanglement_theta",
    "GAME_SYSTEMS",
    "prisoners_dilemma_payoff",
    # Complex Systems
    "ComplexSystem",
    "CriticalExponents",
    "compute_order_parameter",
    "compute_susceptibility",
    "compute_correlation_length",
    "detect_critical_point",
    "COMPLEX_SYSTEMS",
    # Quantum Computing
    "QubitSystem",
    "compute_coherence_theta",
    "compute_error_threshold_theta",
    "QUANTUM_HARDWARE",
    # Quantum Biology
    "BiologicalSystem",
    "compute_quantum_bio_theta",
    "compute_tunneling_theta",
    "BIOLOGICAL_SYSTEMS",
    # Cosmology
    "CosmicEpoch",
    "compute_cosmic_theta",
    "compute_thermal_theta",
    "COSMIC_TIMELINE",
    # Control Theory
    "ControlSystem",
    "ControllerType",
    "StabilityRegime",
    "compute_control_theta",
    "compute_gain_margin_theta",
    "compute_phase_margin_theta",
    "classify_stability",
    "CONTROL_SYSTEMS",
    # Nonlinear Dynamics
    "DynamicalSystem",
    "DynamicalRegime",
    "AttractorType",
    "compute_dynamics_theta",
    "compute_lyapunov_theta",
    "classify_dynamics_regime",
    "logistic_map",
    "logistic_lyapunov",
    "DYNAMICAL_SYSTEMS",
    "FEIGENBAUM_DELTA",
    "FEIGENBAUM_ALPHA",
    # Quantum Gravity
    "QuantumGravitySystem",
    "SpacetimeRegime",
    "QuantumGravityTheory",
    "compute_quantum_gravity_theta",
    "compute_length_theta",
    "compute_energy_theta",
    "classify_spacetime_regime",
    "black_hole_theta",
    "hawking_temperature_k",
    "bekenstein_entropy",
    "QUANTUM_GRAVITY_SYSTEMS",
    "L_PLANCK",
    "E_PLANCK_EV",
    "M_PLANCK",
    # Universal
    "UniversalTheta",
    "DomainType",
    "cross_domain_comparison",
]
