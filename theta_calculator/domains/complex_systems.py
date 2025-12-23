"""
Complex Systems Domain: Phase Transitions and Critical Exponents

This module implements theta as the quantum-classical interpolation parameter
for complex systems using critical phenomena and universality.

## Mapping Definition

This domain maps complex systems to theta via proximity to critical point:

**Inputs (Physical Analogs):**
- order_parameter (m) → Magnetization, opinion consensus, etc. [0, 1]
- reduced_temperature (t) → (T - T_c) / T_c, distance from critical point
- correlation_length (ξ) → Spatial extent of correlations
- susceptibility (χ) → Response to external perturbation

**Theta Mapping:**
θ = exp(-|t|) × f(ξ/L) × g(m)

Near critical point (t → 0): θ → 1
Far from critical point (|t| >> 1): θ → 0

**Interpretation:**
- θ → 0: Disordered phase (random, independent behavior, high entropy)
- θ → 1: At critical point (diverging correlations, power-law scaling)

**Key Feature:** At θ ≈ 1, systems exhibit universal behavior independent
of microscopic details (critical exponents, scaling laws).

**Important:** This is an ANALOGY SCORE based on statistical mechanics.

References (see BIBLIOGRAPHY.bib):
    \\cite{WilsonKogut1974} - Renormalization Group theory
    \\cite{Stanley1971} - Phase Transitions and Critical Phenomena
    \\cite{BakTangWiesenfeld1987} - Self-Organized Criticality
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PhaseType(Enum):
    """Phase classification for complex systems."""
    DISORDERED = "disordered"    # High temperature, random
    CRITICAL = "critical"         # At phase transition
    ORDERED = "ordered"           # Low temperature, aligned
    SUBCRITICAL = "subcritical"   # Just below T_c
    SUPERCRITICAL = "supercritical"  # Just above T_c


@dataclass
class CriticalExponents:
    """
    Universal critical exponents characterizing phase transitions.

    Near T_c, physical quantities scale as power laws:
    - Order parameter: m ~ |T - T_c|^β (for T < T_c)
    - Susceptibility: χ ~ |T - T_c|^(-γ)
    - Correlation length: ξ ~ |T - T_c|^(-ν)
    - Specific heat: C ~ |T - T_c|^(-α)
    - Critical isotherm: m ~ h^(1/δ) (at T = T_c)

    These exponents are UNIVERSAL - they depend only on:
    - Dimensionality (d)
    - Symmetry of order parameter
    - Range of interactions

    NOT on microscopic details! This is the power of universality.
    """
    alpha: float  # Specific heat
    beta: float   # Order parameter
    gamma: float  # Susceptibility
    delta: float  # Critical isotherm
    nu: float     # Correlation length
    eta: float    # Correlation function decay

    def verify_scaling_relations(self) -> Dict[str, Tuple[float, float]]:
        """
        Verify hyperscaling relations between exponents.

        Rushbrooke: α + 2β + γ = 2
        Widom: γ = β(δ - 1)
        Fisher: γ = (2 - η)ν
        Josephson: νd = 2 - α (d = dimension)
        """
        relations = {}

        # Rushbrooke identity
        rushbrooke = self.alpha + 2 * self.beta + self.gamma
        relations["rushbrooke"] = (rushbrooke, 2.0)

        # Widom identity
        widom = self.gamma
        widom_rhs = self.beta * (self.delta - 1)
        relations["widom"] = (widom, widom_rhs)

        # Fisher identity
        fisher = self.gamma
        fisher_rhs = (2 - self.eta) * self.nu
        relations["fisher"] = (fisher, fisher_rhs)

        return relations


# Standard universality classes
MEAN_FIELD = CriticalExponents(
    alpha=0.0, beta=0.5, gamma=1.0, delta=3.0, nu=0.5, eta=0.0
)

ISING_2D = CriticalExponents(
    alpha=0.0,  # Logarithmic divergence
    beta=0.125,  # 1/8
    gamma=1.75,  # 7/4
    delta=15.0,
    nu=1.0,
    eta=0.25   # 1/4
)

ISING_3D = CriticalExponents(
    alpha=0.110, beta=0.326, gamma=1.237, delta=4.789, nu=0.630, eta=0.036
)

XY_3D = CriticalExponents(
    alpha=-0.015, beta=0.348, gamma=1.316, delta=4.780, nu=0.671, eta=0.038
)

HEISENBERG_3D = CriticalExponents(
    alpha=-0.12, beta=0.365, gamma=1.39, delta=4.80, nu=0.705, eta=0.035
)


@dataclass
class ComplexSystem:
    """
    A complex system for theta analysis.

    Attributes:
        name: System identifier
        dimension: Spatial dimension
        n_agents: Number of agents/nodes
        temperature: Control parameter (not necessarily thermal)
        critical_temperature: Phase transition point
        order_parameter: Collective alignment measure
        correlation_length: Spatial extent of correlations
        susceptibility: Response to perturbation
        exponents: Universality class
    """
    name: str
    dimension: int
    n_agents: int
    temperature: float
    critical_temperature: float
    order_parameter: float = 0.0
    correlation_length: float = 1.0
    susceptibility: float = 1.0
    exponents: CriticalExponents = field(default_factory=lambda: MEAN_FIELD)

    @property
    def reduced_temperature(self) -> float:
        """t = (T - T_c) / T_c: Distance from critical point."""
        if self.critical_temperature == 0:
            return float('inf')
        return (self.temperature - self.critical_temperature) / self.critical_temperature


# =============================================================================
# SCALING FUNCTIONS
# =============================================================================

def compute_order_parameter(
    temperature: float,
    T_c: float,
    exponents: CriticalExponents,
    amplitude: float = 1.0
) -> float:
    """
    Compute order parameter from temperature.

    For T < T_c: m = B * |t|^β
    For T > T_c: m = 0

    Args:
        temperature: Current temperature
        T_c: Critical temperature
        exponents: Universality class
        amplitude: Critical amplitude B

    Returns:
        Order parameter m in [0, 1]
    """
    if T_c == 0:
        return 0.0

    t = (temperature - T_c) / T_c

    if t >= 0:  # Above T_c
        return 0.0
    else:  # Below T_c
        return amplitude * abs(t) ** exponents.beta


def compute_susceptibility(
    temperature: float,
    T_c: float,
    exponents: CriticalExponents,
    amplitude: float = 1.0
) -> float:
    """
    Compute susceptibility from temperature.

    χ = Γ * |t|^(-γ)

    Diverges at T_c! This indicates instability.

    Args:
        temperature: Current temperature
        T_c: Critical temperature
        exponents: Universality class
        amplitude: Critical amplitude Γ

    Returns:
        Susceptibility χ
    """
    if T_c == 0:
        return 1.0

    t = (temperature - T_c) / T_c

    if abs(t) < 1e-6:  # At critical point
        return float('inf')

    return amplitude * abs(t) ** (-exponents.gamma)


def compute_correlation_length(
    temperature: float,
    T_c: float,
    exponents: CriticalExponents,
    amplitude: float = 1.0
) -> float:
    """
    Compute correlation length from temperature.

    ξ = ξ_0 * |t|^(-ν)

    Diverges at T_c! Correlations become long-range.

    Returns:
        Correlation length ξ
    """
    if T_c == 0:
        return 1.0

    t = (temperature - T_c) / T_c

    if abs(t) < 1e-6:
        return float('inf')

    return amplitude * abs(t) ** (-exponents.nu)


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_complex_theta(system: ComplexSystem) -> float:
    """
    Compute theta for a complex system.

    Theta measures proximity to criticality and degree of order.

    Methods:
    1. Order parameter: |m| near 1 means ordered (moderate theta)
    2. Critical proximity: |t| near 0 means critical (theta → 1)
    3. Correlation divergence: ξ/L large means correlated

    Returns:
        theta in [0, 1] where:
        - 0 = disordered (high T, no correlations)
        - 1 = critical point (maximum correlations)
    """
    t = system.reduced_temperature

    # Near critical point (|t| < 0.1): theta is high
    if abs(t) < 0.1:
        # At criticality, correlations are maximal
        theta_critical = 1.0 - abs(t) * 10  # Linear in t near T_c

    elif t > 0:  # Above T_c (disordered)
        # theta decreases with distance from T_c
        theta_critical = np.exp(-t)

    else:  # Below T_c (ordered)
        # Ordered phase has intermediate theta
        # Order develops, but correlations not maximal
        theta_critical = 0.5 + 0.5 * np.exp(t)

    # Contribution from order parameter
    theta_order = abs(system.order_parameter)

    # Combined theta
    # At critical point: theta = 1 (maximum fluctuations)
    # Deep in ordered phase: theta ~ 0.5 (partial order)
    # Deep in disordered phase: theta ~ 0 (random)
    theta = max(theta_critical, theta_order * 0.5)

    return np.clip(theta, 0.0, 1.0)


def detect_critical_point(
    order_parameters: List[float],
    temperatures: List[float]
) -> Tuple[bool, Optional[float], Optional[CriticalExponents]]:
    """
    Detect phase transition and estimate critical exponents.

    Method: Look for where susceptibility (dm/dT) is maximized.

    Returns:
        (detected, T_c, estimated_exponents)
    """
    if len(order_parameters) < 5:
        return False, None, None

    # Compute susceptibility (negative derivative of m)
    dm_dT = -np.gradient(order_parameters, temperatures)

    # Find peak (divergence)
    max_idx = np.argmax(dm_dT)
    T_c = temperatures[max_idx]

    # Check if this is a real transition (susceptibility spike)
    if dm_dT[max_idx] < 0.5:
        return False, None, None

    # Estimate beta from power law fit
    # For T < T_c: m ~ |t|^beta
    below_Tc = [(T, m) for T, m in zip(temperatures, order_parameters) if T < T_c]
    if len(below_Tc) < 3:
        return True, T_c, MEAN_FIELD

    T_below, m_below = zip(*below_Tc)
    t_below = [(T - T_c) / T_c for T in T_below]

    # Log-log fit for power law
    try:
        log_t = np.log(np.abs(t_below))
        log_m = np.log(np.abs(m_below))
        beta_est = np.polyfit(log_t, log_m, 1)[0]

        # Create estimated exponents (assume mean-field relations)
        estimated = CriticalExponents(
            alpha=0.0,
            beta=beta_est,
            gamma=1.0,  # Would need susceptibility data
            delta=3.0,
            nu=0.5,
            eta=0.0
        )
        return True, T_c, estimated

    except Exception:
        return True, T_c, MEAN_FIELD


def classify_phase(system: ComplexSystem) -> PhaseType:
    """Classify the phase of a complex system."""
    t = system.reduced_temperature

    if abs(t) < 0.05:
        return PhaseType.CRITICAL
    elif t > 0:
        if t < 0.2:
            return PhaseType.SUPERCRITICAL
        else:
            return PhaseType.DISORDERED
    else:
        if abs(t) < 0.2:
            return PhaseType.SUBCRITICAL
        else:
            return PhaseType.ORDERED


# =============================================================================
# EXAMPLE COMPLEX SYSTEMS
# =============================================================================

COMPLEX_SYSTEMS: Dict[str, ComplexSystem] = {
    "ferromagnet_hot": ComplexSystem(
        name="Ferromagnet (T >> T_c)",
        dimension=3,
        n_agents=10**6,
        temperature=1000.0,
        critical_temperature=1044.0,  # Iron Curie temperature in K
        order_parameter=0.0,
        exponents=ISING_3D,
    ),
    "ferromagnet_critical": ComplexSystem(
        name="Ferromagnet (T = T_c)",
        dimension=3,
        n_agents=10**6,
        temperature=1044.0,
        critical_temperature=1044.0,
        order_parameter=0.0,  # Fluctuating
        exponents=ISING_3D,
    ),
    "ferromagnet_cold": ComplexSystem(
        name="Ferromagnet (T << T_c)",
        dimension=3,
        n_agents=10**6,
        temperature=300.0,
        critical_temperature=1044.0,
        order_parameter=0.9,  # Strong magnetization
        exponents=ISING_3D,
    ),
    "opinion_polarized": ComplexSystem(
        name="Polarized Society",
        dimension=2,  # Social network effectively 2D
        n_agents=10**8,
        temperature=0.5,
        critical_temperature=1.0,
        order_parameter=0.8,  # Strong polarization
        exponents=MEAN_FIELD,  # Mean-field due to long-range connections
    ),
    "opinion_diverse": ComplexSystem(
        name="Diverse Society",
        dimension=2,
        n_agents=10**8,
        temperature=2.0,
        critical_temperature=1.0,
        order_parameter=0.1,  # Low polarization
        exponents=MEAN_FIELD,
    ),
    "epidemic_spreading": ComplexSystem(
        name="Epidemic at Critical R0",
        dimension=3,
        n_agents=10**7,
        temperature=1.0,  # R0 = 1
        critical_temperature=1.0,  # R0_c = 1
        order_parameter=0.3,
        exponents=MEAN_FIELD,  # Percolation universality
    ),
    "neural_criticality": ComplexSystem(
        name="Brain at Criticality",
        dimension=3,
        n_agents=10**11,  # Neurons
        temperature=1.0,
        critical_temperature=1.0,
        order_parameter=0.5,
        exponents=MEAN_FIELD,
    ),
    "civil_unrest": ComplexSystem(
        name="Pre-Revolution Society",
        dimension=2,
        n_agents=10**7,
        temperature=0.95,  # Just below critical
        critical_temperature=1.0,
        order_parameter=0.6,
        exponents=ISING_2D,
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def complex_systems_theta_summary():
    """Print theta analysis for all example complex systems."""
    print("=" * 70)
    print("COMPLEX SYSTEMS THETA ANALYSIS (Critical Phenomena)")
    print("=" * 70)
    print()
    print(f"{'System':<30} {'θ':>8} {'|m|':>8} {'t':>10} {'Phase':<15}")
    print("-" * 70)

    for name, system in COMPLEX_SYSTEMS.items():
        theta = compute_complex_theta(system)
        phase = classify_phase(system)
        t = system.reduced_temperature

        t_str = f"{t:.4f}" if abs(t) < 10 else f"{t:.1e}"

        print(f"{system.name:<30} {theta:>8.3f} "
              f"{abs(system.order_parameter):>8.2f} {t_str:>10} {phase.value:<15}")

    print()
    print("Key: At critical point (t=0), theta=1: maximum fluctuations!")


if __name__ == "__main__":
    complex_systems_theta_summary()
