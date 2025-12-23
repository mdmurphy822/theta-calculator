"""
Social Systems Domain: Opinion Dynamics, Epidemics, Urban Scaling, and Traffic

This module implements theta as the collective behavior parameter
for social and complex adaptive systems.

Key Insight: Social systems exhibit phase transitions between:
- theta ~ 0: Individual behavior (random, uncorrelated)
- theta ~ 1: Collective behavior (herding, consensus, cascades)

Theta Maps To:
1. Opinion dynamics: Polarization level (voter model)
2. Epidemic spreading: R₀ / R₀_critical (SIR threshold)
3. Urban scaling: (β - 1) / 0.15 (superlinear scaling)
4. Traffic flow: density / ρ_critical (jamming transition)

References (see BIBLIOGRAPHY.bib):
    \cite{Castellano2009} - Statistical physics of social dynamics
    \cite{Kermack1927} - SIR epidemic model
    \cite{Bettencourt2007} - Urban scaling laws
    \cite{NagelSchreckenberg1992} - Cellular automaton for traffic
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SocialPhase(Enum):
    """Social system phases based on theta."""
    INDIVIDUAL = "individual"      # theta < 0.3
    CLUSTERING = "clustering"      # 0.3 <= theta < 0.6
    COLLECTIVE = "collective"      # 0.6 <= theta < 0.9
    CONSENSUS = "consensus"        # theta >= 0.9


class EpidemicPhase(Enum):
    """Epidemic phases."""
    DYING_OUT = "dying_out"        # R₀ < 1
    CRITICAL = "critical"          # R₀ ≈ 1
    SPREADING = "spreading"        # 1 < R₀ < 3
    EXPLOSIVE = "explosive"        # R₀ > 3


class TrafficPhase(Enum):
    """Traffic flow phases."""
    FREE_FLOW = "free_flow"        # Below critical density
    SYNCHRONIZED = "synchronized"   # Near critical
    JAMMED = "jammed"              # Above critical


@dataclass
class SocialSystem:
    """
    A social system for theta analysis.

    Attributes:
        name: System identifier
        population: Number of agents
        connectivity: Average connections per agent
        polarization: Opinion polarization [0, 1]
        clustering: Social clustering coefficient
        cascade_probability: Probability of information cascade
    """
    name: str
    population: int
    connectivity: float
    polarization: float
    clustering: float
    cascade_probability: float


# =============================================================================
# OPINION DYNAMICS
# =============================================================================

def compute_polarization(opinions: List[float]) -> float:
    """
    Compute opinion polarization.

    Polarization is high when opinions cluster at extremes.
    Uses bimodality coefficient.

    Args:
        opinions: List of opinion values in [-1, 1]

    Returns:
        Polarization in [0, 1]
    """
    if len(opinions) < 2:
        return 0.0

    opinions = np.array(opinions)

    # Bimodality: high variance + low mean suggests polarization
    variance = np.var(opinions)
    mean_abs = np.abs(np.mean(opinions))

    # Polarization high when variance high and |mean| low
    # (opinions at both extremes)
    polarization = variance * (1 - mean_abs)

    return np.clip(polarization, 0.0, 1.0)


def voter_model_magnetization(
    opinions: List[int],
    time: float,
    n: int
) -> float:
    """
    Compute magnetization in voter model.

    In voter model, agents copy random neighbor's opinion.
    System reaches consensus in time O(N) for complete graph.

    Args:
        opinions: List of +1 or -1 opinions
        time: Time steps elapsed
        n: Population size

    Returns:
        Magnetization m = |Σs_i| / N

    Reference: \cite{Castellano2009}
    """
    return abs(np.mean(opinions))


def compute_opinion_theta(
    polarization: float,
    consensus_threshold: float = 0.9
) -> float:
    """
    Compute theta for opinion dynamics.

    High polarization = high theta (collective separation)
    High consensus = high theta (collective agreement)

    Args:
        polarization: Current polarization level [0, 1]
        consensus_threshold: Threshold for consensus

    Returns:
        theta in [0, 1]

    Reference: \cite{Castellano2009}
    """
    # Both high polarization and high consensus indicate
    # collective behavior (high theta)
    return polarization


# =============================================================================
# EPIDEMIC SPREADING (SIR Model)
# =============================================================================

@dataclass
class SIRState:
    """
    State of SIR epidemic model.

    dS/dt = -βSI
    dI/dt = βSI - γI
    dR/dt = γI

    Attributes:
        susceptible: Fraction susceptible
        infected: Fraction infected
        recovered: Fraction recovered
        R0: Basic reproduction number
        herd_immunity_threshold: 1 - 1/R₀

    Reference: \cite{Kermack1927}
    """
    susceptible: float
    infected: float
    recovered: float
    R0: float
    herd_immunity_threshold: float


def compute_R0(
    transmission_rate: float,
    recovery_rate: float,
    contacts_per_day: float
) -> float:
    """
    Compute basic reproduction number R₀.

    R₀ = β/γ = (transmission × contacts) / recovery

    R₀ < 1: Epidemic dies out
    R₀ > 1: Epidemic spreads
    R₀ = 1: Critical threshold

    Args:
        transmission_rate: Probability of transmission per contact
        recovery_rate: Recovery rate (1/infectious_period)
        contacts_per_day: Average daily contacts

    Returns:
        R₀

    Reference: \cite{Kermack1927}
    """
    if recovery_rate <= 0:
        return float('inf')
    return (transmission_rate * contacts_per_day) / recovery_rate


def compute_epidemic_theta(
    R0: float,
    R0_critical: float = 1.0
) -> float:
    """
    Compute theta for epidemic spreading.

    Theta = R₀ / R₀_max (normalized by typical pandemic R₀)

    R₀ typical values:
    - Measles: 12-18
    - COVID-19: 2-3
    - Flu: 1.3
    - Common cold: 2-3

    Args:
        R0: Basic reproduction number
        R0_critical: Critical threshold (1.0)

    Returns:
        theta in [0, 1]

    Reference: \cite{Kermack1927}
    """
    # Normalize by maximum concerning R₀ (measles ~ 18)
    R0_max = 18.0

    if R0 < R0_critical:
        # Below threshold: theta < 0.5
        theta = 0.5 * R0 / R0_critical
    else:
        # Above threshold: theta > 0.5
        theta = 0.5 + 0.5 * (R0 - R0_critical) / (R0_max - R0_critical)

    return np.clip(theta, 0.0, 1.0)


def herd_immunity_threshold(R0: float) -> float:
    """
    Compute herd immunity threshold.

    HIT = 1 - 1/R₀

    When this fraction is immune, epidemic cannot spread.

    Reference: \cite{Kermack1927}
    """
    if R0 <= 1:
        return 0.0
    return 1 - 1/R0


def classify_epidemic(R0: float) -> EpidemicPhase:
    """Classify epidemic phase from R₀."""
    if R0 < 0.9:
        return EpidemicPhase.DYING_OUT
    elif R0 < 1.1:
        return EpidemicPhase.CRITICAL
    elif R0 < 3.0:
        return EpidemicPhase.SPREADING
    else:
        return EpidemicPhase.EXPLOSIVE


# =============================================================================
# URBAN SCALING (Bettencourt's Law)
# =============================================================================

def urban_scaling_exponent(
    city_sizes: List[float],
    metric_values: List[float]
) -> float:
    """
    Compute urban scaling exponent β.

    Y = Y₀ × N^β

    β > 1: Superlinear (innovation, crime, disease)
    β = 1: Linear (infrastructure per capita)
    β < 1: Sublinear (infrastructure total)

    Args:
        city_sizes: Population of cities
        metric_values: Metric value for each city

    Returns:
        Scaling exponent β

    Reference: \cite{Bettencourt2007}
    """
    if len(city_sizes) < 3:
        return 1.0

    # Log-log regression
    log_N = np.log(city_sizes)
    log_Y = np.log(metric_values)

    beta, _ = np.polyfit(log_N, log_Y, 1)
    return beta


def compute_urban_theta(
    beta: float,
    beta_typical: float = 1.15
) -> float:
    """
    Compute theta for urban scaling.

    Superlinear scaling (β > 1) indicates emergent collective effects.
    β ≈ 1.15 for socioeconomic metrics (GDP, patents, crime)
    β ≈ 0.85 for infrastructure (roads, pipes)

    Theta = (β - 1) / 0.15 for superlinear metrics

    Args:
        beta: Scaling exponent
        beta_typical: Typical superlinear exponent (1.15)

    Returns:
        theta in [0, 1]

    Reference: \cite{Bettencourt2007}
    """
    # For superlinear: beta = 1.15 → theta = 1
    # For linear: beta = 1.0 → theta = 0
    max_deviation = beta_typical - 1.0  # 0.15

    if beta >= 1.0:
        theta = (beta - 1.0) / max_deviation
    else:
        # Sublinear: still map to positive theta
        theta = (1.0 - beta) / max_deviation * 0.5

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# TRAFFIC FLOW (Nagel-Schreckenberg Model)
# =============================================================================

def fundamental_diagram(
    density: float,
    v_max: float = 1.0,
    critical_density: float = 0.3
) -> float:
    """
    Compute traffic flow from fundamental diagram.

    Flow = density × velocity
    At low density: velocity = v_max, flow ~ density
    At high density: velocity decreases, flow peaks then decreases

    Args:
        density: Vehicle density [0, 1]
        v_max: Maximum velocity
        critical_density: Density at maximum flow

    Returns:
        Traffic flow

    Reference: \cite{NagelSchreckenberg1992}
    """
    if density <= 0:
        return 0.0
    if density >= 1:
        return 0.0

    if density < critical_density:
        # Free flow regime
        return density * v_max
    else:
        # Congested regime
        return (1 - density) * v_max


def compute_traffic_theta(
    density: float,
    critical_density: float = 0.3
) -> float:
    """
    Compute theta for traffic flow.

    Theta = density / critical_density (before jamming)
    Theta represents how close to congestion transition.

    Args:
        density: Current vehicle density [0, 1]
        critical_density: Jamming transition density

    Returns:
        theta in [0, 1]

    Reference: \cite{NagelSchreckenberg1992}
    """
    if critical_density <= 0:
        return 0.0

    # Below critical: theta < 1
    # Above critical: theta approaches 1 (jammed)
    if density <= critical_density:
        theta = density / critical_density
    else:
        # In congestion: high theta
        excess = (density - critical_density) / (1 - critical_density)
        theta = 1.0 - 0.1 * (1 - excess)  # Stays near 1

    return np.clip(theta, 0.0, 1.0)


def classify_traffic(density: float, rho_c: float = 0.3) -> TrafficPhase:
    """Classify traffic phase from density."""
    if density < 0.8 * rho_c:
        return TrafficPhase.FREE_FLOW
    elif density < 1.2 * rho_c:
        return TrafficPhase.SYNCHRONIZED
    else:
        return TrafficPhase.JAMMED


# =============================================================================
# UNIFIED THETA CALCULATION
# =============================================================================

def compute_social_theta(system: SocialSystem) -> float:
    """
    Compute unified theta for social system.

    Combines:
    - Polarization
    - Clustering
    - Cascade probability

    Args:
        system: SocialSystem to analyze

    Returns:
        theta in [0, 1]
    """
    theta = (
        0.4 * system.polarization +
        0.3 * system.clustering +
        0.3 * system.cascade_probability
    )
    return np.clip(theta, 0.0, 1.0)


def classify_social_phase(theta: float) -> SocialPhase:
    """Classify social phase from theta."""
    if theta < 0.3:
        return SocialPhase.INDIVIDUAL
    elif theta < 0.6:
        return SocialPhase.CLUSTERING
    elif theta < 0.9:
        return SocialPhase.COLLECTIVE
    else:
        return SocialPhase.CONSENSUS


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

SOCIAL_SYSTEMS: Dict[str, SocialSystem] = {
    "diverse_democracy": SocialSystem(
        name="Diverse Democracy",
        population=1000000,
        connectivity=150,
        polarization=0.2,
        clustering=0.3,
        cascade_probability=0.1,
    ),
    "polarized_society": SocialSystem(
        name="Polarized Society",
        population=1000000,
        connectivity=100,
        polarization=0.85,
        clustering=0.7,
        cascade_probability=0.5,
    ),
    "echo_chamber": SocialSystem(
        name="Social Media Echo Chamber",
        population=10000,
        connectivity=500,
        polarization=0.95,
        clustering=0.9,
        cascade_probability=0.8,
    ),
    "small_community": SocialSystem(
        name="Small Rural Community",
        population=1000,
        connectivity=50,
        polarization=0.1,
        clustering=0.8,
        cascade_probability=0.3,
    ),
}

EPIDEMIC_EXAMPLES = {
    "common_cold": {"R0": 2.0, "name": "Common Cold"},
    "seasonal_flu": {"R0": 1.3, "name": "Seasonal Flu"},
    "covid_original": {"R0": 2.5, "name": "COVID-19 (original)"},
    "covid_delta": {"R0": 5.0, "name": "COVID-19 (Delta)"},
    "measles": {"R0": 15.0, "name": "Measles"},
    "ebola": {"R0": 1.8, "name": "Ebola"},
}

TRAFFIC_EXAMPLES = {
    "empty_highway": {"density": 0.05, "name": "Empty Highway"},
    "light_traffic": {"density": 0.15, "name": "Light Traffic"},
    "rush_hour_approaching": {"density": 0.25, "name": "Rush Hour Approaching"},
    "critical_density": {"density": 0.30, "name": "Critical Density"},
    "stop_and_go": {"density": 0.50, "name": "Stop-and-Go"},
    "gridlock": {"density": 0.80, "name": "Gridlock"},
}


def social_theta_summary():
    """Print theta analysis for example social systems."""
    print("=" * 70)
    print("SOCIAL SYSTEMS THETA ANALYSIS")
    print("=" * 70)
    print()

    # Opinion dynamics
    print("OPINION DYNAMICS:")
    print(f"{'System':<30} {'Polarization':>12} {'Clustering':>10} {'θ':>8} {'Phase':<15}")
    print("-" * 70)

    for name, system in SOCIAL_SYSTEMS.items():
        theta = compute_social_theta(system)
        phase = classify_social_phase(theta)
        print(f"{system.name:<30} "
              f"{system.polarization:>12.2f} "
              f"{system.clustering:>10.2f} "
              f"{theta:>8.3f} "
              f"{phase.value:<15}")

    print()

    # Epidemics
    print("EPIDEMIC SPREADING:")
    print(f"{'Disease':<25} {'R₀':>8} {'HIT':>8} {'θ':>8} {'Phase':<15}")
    print("-" * 60)

    for name, epi in EPIDEMIC_EXAMPLES.items():
        R0 = epi["R0"]
        theta = compute_epidemic_theta(R0)
        HIT = herd_immunity_threshold(R0)
        phase = classify_epidemic(R0)
        print(f"{epi['name']:<25} {R0:>8.1f} {HIT:>8.1%} {theta:>8.3f} {phase.value:<15}")

    print()

    # Traffic
    print("TRAFFIC FLOW:")
    print(f"{'Condition':<25} {'Density':>10} {'θ':>8} {'Phase':<15}")
    print("-" * 60)

    for name, traffic in TRAFFIC_EXAMPLES.items():
        theta = compute_traffic_theta(traffic["density"])
        phase = classify_traffic(traffic["density"])
        print(f"{traffic['name']:<25} {traffic['density']:>10.2f} {theta:>8.3f} {phase.value:<15}")


if __name__ == "__main__":
    social_theta_summary()
