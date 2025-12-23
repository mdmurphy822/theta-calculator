"""
Multi-Domain Theta Framework

This module extends theta beyond physics to economics, game theory,
information theory, complex systems, quantum computing, quantum biology,
cosmology, education, mechanical systems, networks, cognition, social
dynamics, and chemistry.

Key Insight: Theta represents a universal interpolation parameter
between discrete/quantum-like and continuous/classical behavior
across ALL domains exhibiting phase transitions.

Original Domains:
- economics: Ising model of markets, phase transitions
- information: Shannon vs von Neumann entropy
- game_theory: Entanglement parameter in quantum games
- complex_systems: Critical exponents, social phase transitions
- quantum_computing: Error thresholds, decoherence
- quantum_biology: Coherence in photosynthesis, tunneling in enzymes
- cosmology: Theta evolution from Big Bang to heat death

New Domains (2024):
- education: Learning curves, memory retention, knowledge integration
- mechanical_systems: Engine efficiency, motors, batteries, damping
- networks: Shannon capacity, percolation, QKD
- cognition: Integrated information, neural criticality, working memory
- social_systems: Opinion dynamics, epidemics, urban scaling, traffic
- chemistry: Superconductivity, BEC, quantum dots, superfluidity
- work_life_balance: Burnout, effort-reward, work-family conflict

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

# New domains (2024)
from .education import (
    LearningSystem,
    LearningPhase,
    MemoryRetention,
    LearningCurve,
    compute_education_theta,
    compute_retention_theta,
    compute_learning_theta,
    classify_learning_phase,
    ebbinghaus_retention,
    EDUCATION_SYSTEMS,
)

from .mechanical_systems import (
    MechanicalSystem,
    SystemType,
    EfficiencyRegime,
    compute_mechanical_theta,
    compute_engine_theta,
    compute_motor_theta,
    compute_battery_theta,
    compute_damping_theta,
    carnot_efficiency,
    MECHANICAL_SYSTEMS,
)

from .networks import (
    NetworkSystem,
    NetworkType,
    ConnectivityRegime,
    compute_network_theta,
    compute_shannon_theta,
    compute_percolation_theta,
    compute_qkd_theta,
    shannon_capacity,
    percolation_threshold,
    NETWORK_SYSTEMS,
)

from .cognition import (
    CognitiveSystem,
    ConsciousnessState,
    BrainState,
    compute_cognition_theta,
    compute_phi_ratio,
    compute_criticality_theta,
    compute_working_memory_theta,
    classify_brain_state,
    COGNITIVE_SYSTEMS,
)

from .social_systems import (
    SocialSystem,
    SocialPhase,
    EpidemicPhase,
    TrafficPhase,
    compute_social_theta,
    compute_opinion_theta,
    compute_epidemic_theta,
    compute_urban_theta,
    compute_traffic_theta,
    classify_social_phase,
    herd_immunity_threshold,
    SOCIAL_SYSTEMS,
)

from .chemistry import (
    QuantumMaterial,
    MaterialPhase,
    SuperconductorType,
    compute_chemistry_theta,
    compute_superconductor_theta,
    compute_bec_theta,
    compute_quantum_dot_theta,
    compute_superfluid_theta,
    bcs_gap_zero_temp,
    bec_condensate_fraction,
    SUPERCONDUCTORS,
)

from .work_life_balance import (
    WorkLifeSystem,
    WellbeingPhase,
    BurnoutDimension,
    ConflictDirection,
    compute_burnout_theta,
    compute_effort_reward_theta,
    compute_work_family_conflict_theta,
    compute_cognitive_load_theta,
    compute_jdr_theta,
    compute_recovery_theta,
    compute_job_strain_theta,
    compute_work_life_theta,
    classify_burnout,
    classify_wellbeing_phase,
    classify_job_strain,
    WORK_LIFE_SYSTEMS,
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
    # Education
    "LearningSystem",
    "LearningPhase",
    "MemoryRetention",
    "LearningCurve",
    "compute_education_theta",
    "compute_retention_theta",
    "compute_learning_theta",
    "classify_learning_phase",
    "ebbinghaus_retention",
    "EDUCATION_SYSTEMS",
    # Mechanical Systems
    "MechanicalSystem",
    "SystemType",
    "EfficiencyRegime",
    "compute_mechanical_theta",
    "compute_engine_theta",
    "compute_motor_theta",
    "compute_battery_theta",
    "compute_damping_theta",
    "carnot_efficiency",
    "MECHANICAL_SYSTEMS",
    # Networks
    "NetworkSystem",
    "NetworkType",
    "ConnectivityRegime",
    "compute_network_theta",
    "compute_shannon_theta",
    "compute_percolation_theta",
    "compute_qkd_theta",
    "shannon_capacity",
    "percolation_threshold",
    "NETWORK_SYSTEMS",
    # Cognition
    "CognitiveSystem",
    "ConsciousnessState",
    "BrainState",
    "compute_cognition_theta",
    "compute_phi_ratio",
    "compute_criticality_theta",
    "compute_working_memory_theta",
    "classify_brain_state",
    "COGNITIVE_SYSTEMS",
    # Social Systems
    "SocialSystem",
    "SocialPhase",
    "EpidemicPhase",
    "TrafficPhase",
    "compute_social_theta",
    "compute_opinion_theta",
    "compute_epidemic_theta",
    "compute_urban_theta",
    "compute_traffic_theta",
    "classify_social_phase",
    "herd_immunity_threshold",
    "SOCIAL_SYSTEMS",
    # Chemistry
    "QuantumMaterial",
    "MaterialPhase",
    "SuperconductorType",
    "compute_chemistry_theta",
    "compute_superconductor_theta",
    "compute_bec_theta",
    "compute_quantum_dot_theta",
    "compute_superfluid_theta",
    "bcs_gap_zero_temp",
    "bec_condensate_fraction",
    "SUPERCONDUCTORS",
    # Work-Life Balance
    "WorkLifeSystem",
    "WellbeingPhase",
    "BurnoutDimension",
    "ConflictDirection",
    "compute_burnout_theta",
    "compute_effort_reward_theta",
    "compute_work_family_conflict_theta",
    "compute_cognitive_load_theta",
    "compute_jdr_theta",
    "compute_recovery_theta",
    "compute_job_strain_theta",
    "compute_work_life_theta",
    "classify_burnout",
    "classify_wellbeing_phase",
    "classify_job_strain",
    "WORK_LIFE_SYSTEMS",
]
