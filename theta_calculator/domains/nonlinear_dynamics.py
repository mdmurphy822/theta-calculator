"""
Nonlinear Dynamics Domain: Chaos and Bifurcations as Theta

This module implements theta as the order-chaos interpolation parameter
for nonlinear dynamical systems.

Key Insight: Dynamical systems exhibit a spectrum between:
- theta ~ 0: Ordered/periodic regime (predictable, low-dimensional)
- theta ~ 0.5: Edge of chaos (maximum computational capacity)
- theta ~ 1: Chaotic regime (sensitive dependence, high-dimensional)

The dynamical theta measures where a system sits on the order-chaos continuum,
with special significance at theta ~ 0.5 (edge of chaos).

Key Mappings:
- Lyapunov exponent: λ > 0 → chaotic → high theta
- Bifurcation parameter: Distance to bifurcation → theta
- Attractor dimension: Higher dimension → higher theta
- Feigenbaum constant: Universal at period-doubling transitions

References:
- Strogatz (2015): Nonlinear Dynamics and Chaos
- Feigenbaum (1978): Quantitative universality for nonlinear transformations
- Lorenz (1963): Deterministic nonperiodic flow
- Ott (2002): Chaos in Dynamical Systems
- Langton (1990): Computation at the edge of chaos
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from enum import Enum


class DynamicalRegime(Enum):
    """Classification of dynamical regimes."""
    FIXED_POINT = "fixed_point"        # theta < 0.1: Stable equilibrium
    PERIODIC = "periodic"               # 0.1 <= theta < 0.3: Limit cycles
    QUASIPERIODIC = "quasiperiodic"    # 0.3 <= theta < 0.5: Torus
    EDGE_OF_CHAOS = "edge_of_chaos"    # 0.4 <= theta < 0.6: Critical
    WEAKLY_CHAOTIC = "weakly_chaotic"  # 0.6 <= theta < 0.8: Intermittent
    CHAOTIC = "chaotic"                 # theta >= 0.8: Strange attractor


class AttractorType(Enum):
    """Types of attractors in phase space."""
    POINT = "point"                     # Fixed point
    LIMIT_CYCLE = "limit_cycle"         # Periodic orbit
    TORUS = "torus"                     # Quasiperiodic
    STRANGE = "strange"                 # Chaotic attractor


@dataclass
class DynamicalSystem:
    """
    A dynamical system for theta analysis.

    Attributes:
        name: System identifier
        dimension: Phase space dimension
        lyapunov_exponents: List of Lyapunov exponents (largest first)
        attractor_type: Type of attractor
        attractor_dimension: Fractal dimension of attractor
        bifurcation_parameter: Current value of control parameter
        bifurcation_critical: Critical value where bifurcation occurs
        period: Period for periodic orbits (None if chaotic)
        description: Physical description
    """
    name: str
    dimension: int
    lyapunov_exponents: List[float]
    attractor_type: AttractorType
    attractor_dimension: Optional[float] = None
    bifurcation_parameter: Optional[float] = None
    bifurcation_critical: Optional[float] = None
    period: Optional[float] = None
    description: Optional[str] = None

    @property
    def max_lyapunov(self) -> float:
        """Largest Lyapunov exponent."""
        return max(self.lyapunov_exponents)

    @property
    def is_chaotic(self) -> bool:
        """Check if system is chaotic (positive max Lyapunov)."""
        return self.max_lyapunov > 0

    @property
    def lyapunov_sum(self) -> float:
        """Sum of all Lyapunov exponents (related to phase space contraction)."""
        return sum(self.lyapunov_exponents)


# =============================================================================
# UNIVERSAL CONSTANTS
# =============================================================================

# Feigenbaum constants (universal for period-doubling bifurcations)
FEIGENBAUM_DELTA = 4.669201609  # Ratio of successive bifurcation intervals
FEIGENBAUM_ALPHA = 2.502907875  # Scaling of function values


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_lyapunov_theta(system: DynamicalSystem) -> float:
    """
    Compute theta from Lyapunov exponent.

    Reference: Ott (2002), Ch. 4

    λ < 0: Stable, theta -> 0
    λ = 0: Edge of chaos, theta = 0.5
    λ > 0: Chaotic, theta -> 1

    theta = 1 / (1 + exp(-λ * scale))
    """
    lambda_max = system.max_lyapunov

    # Scale factor to map typical Lyapunov exponents to [0, 1]
    scale = 2.0  # Adjust based on typical exponent magnitudes

    theta = 1.0 / (1.0 + np.exp(-lambda_max * scale))
    return np.clip(theta, 0.0, 1.0)


def compute_dimension_theta(system: DynamicalSystem) -> float:
    """
    Compute theta from attractor dimension.

    Integer dimension = periodic = low theta
    Fractional dimension = strange attractor = high theta

    theta = (D_attractor - floor(D_attractor))
    """
    if system.attractor_dimension is None:
        return compute_lyapunov_theta(system)

    D = system.attractor_dimension

    # Fractional part indicates "strangeness"
    fractional_part = D - np.floor(D)

    # Also consider how close to phase space dimension
    relative_dimension = D / system.dimension

    theta = 0.5 * fractional_part + 0.5 * relative_dimension
    return np.clip(theta, 0.0, 1.0)


def compute_bifurcation_theta(system: DynamicalSystem) -> float:
    """
    Compute theta from distance to bifurcation point.

    Near bifurcation (critical slowing down) = high sensitivity = high theta
    Far from bifurcation = stable regime = low theta

    theta = exp(-|r - r_c| / scale)
    """
    if system.bifurcation_parameter is None or system.bifurcation_critical is None:
        return compute_lyapunov_theta(system)

    distance = abs(system.bifurcation_parameter - system.bifurcation_critical)

    # Scale by typical parameter range
    scale = abs(system.bifurcation_critical) * 0.1 if system.bifurcation_critical != 0 else 0.1

    theta = np.exp(-distance / scale)
    return np.clip(theta, 0.0, 1.0)


def compute_dynamics_theta(system: DynamicalSystem) -> float:
    """
    Compute unified theta for a dynamical system.

    Combines Lyapunov exponent, attractor dimension, and bifurcation proximity.
    """
    theta_lyap = compute_lyapunov_theta(system)
    theta_dim = compute_dimension_theta(system)

    # Weighted combination
    theta = 0.6 * theta_lyap + 0.4 * theta_dim

    return np.clip(theta, 0.0, 1.0)


def classify_regime(theta: float) -> DynamicalRegime:
    """Classify dynamical regime from theta."""
    if theta < 0.1:
        return DynamicalRegime.FIXED_POINT
    elif theta < 0.3:
        return DynamicalRegime.PERIODIC
    elif theta < 0.45:
        return DynamicalRegime.QUASIPERIODIC
    elif theta < 0.6:
        return DynamicalRegime.EDGE_OF_CHAOS
    elif theta < 0.8:
        return DynamicalRegime.WEAKLY_CHAOTIC
    else:
        return DynamicalRegime.CHAOTIC


# =============================================================================
# LOGISTIC MAP ANALYSIS
# =============================================================================

def logistic_map(x: float, r: float) -> float:
    """
    The logistic map: x_{n+1} = r * x_n * (1 - x_n)

    Reference: Feigenbaum (1978)

    Bifurcation points:
    - r = 1: Stable fixed point at 0
    - r = 3: Period-2 bifurcation
    - r = 3.449: Period-4
    - r = 3.544: Period-8
    - r = 3.5699...: Onset of chaos (accumulation point)
    - r = 4: Fully developed chaos
    """
    return r * x * (1 - x)


def logistic_lyapunov(r: float, n_iter: int = 1000) -> float:
    """
    Compute Lyapunov exponent for logistic map.

    λ = lim (1/n) Σ ln|f'(x_i)| = lim (1/n) Σ ln|r(1 - 2x_i)|
    """
    x = 0.5  # Initial condition
    lyap_sum = 0.0

    # Transient
    for _ in range(100):
        x = logistic_map(x, r)

    # Compute Lyapunov exponent
    for _ in range(n_iter):
        x = logistic_map(x, r)
        derivative = abs(r * (1 - 2 * x))
        if derivative > 0:
            lyap_sum += np.log(derivative)

    return lyap_sum / n_iter


def logistic_theta_sweep(r_min: float = 2.5, r_max: float = 4.0, n_points: int = 100) -> List[Tuple[float, float]]:
    """
    Compute theta across logistic map parameter range.

    Returns list of (r, theta) pairs.
    """
    results = []
    for r in np.linspace(r_min, r_max, n_points):
        lyap = logistic_lyapunov(r)
        theta = 1.0 / (1.0 + np.exp(-lyap * 2))
        results.append((r, theta))
    return results


# =============================================================================
# LORENZ SYSTEM
# =============================================================================

def lorenz_derivatives(state: np.ndarray, sigma: float = 10, rho: float = 28, beta: float = 8/3) -> np.ndarray:
    """
    Lorenz system derivatives.

    Reference: Lorenz (1963)

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Standard chaotic parameters: sigma=10, rho=28, beta=8/3
    """
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


# Lorenz system Lyapunov exponents (standard parameters)
LORENZ_LYAPUNOV_EXPONENTS = [0.906, 0.0, -14.57]  # Approximate values


# =============================================================================
# EXAMPLE DYNAMICAL SYSTEMS
# =============================================================================

DYNAMICAL_SYSTEMS: Dict[str, DynamicalSystem] = {
    "logistic_stable": DynamicalSystem(
        name="Logistic Map (r=2.5, stable)",
        dimension=1,
        lyapunov_exponents=[-0.69],  # ln(0.5)
        attractor_type=AttractorType.POINT,
        attractor_dimension=0,
        bifurcation_parameter=2.5,
        bifurcation_critical=3.0,
        description="Stable fixed point regime",
    ),
    "logistic_period2": DynamicalSystem(
        name="Logistic Map (r=3.2, period-2)",
        dimension=1,
        lyapunov_exponents=[-0.2],
        attractor_type=AttractorType.LIMIT_CYCLE,
        attractor_dimension=0,
        bifurcation_parameter=3.2,
        bifurcation_critical=3.449,
        period=2,
        description="Period-2 oscillation",
    ),
    "logistic_edge_of_chaos": DynamicalSystem(
        name="Logistic Map (r=3.57, edge)",
        dimension=1,
        lyapunov_exponents=[0.0],  # At accumulation point
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=0.538,  # Feigenbaum attractor
        bifurcation_parameter=3.57,
        bifurcation_critical=3.5699,
        description="Edge of chaos (onset)",
    ),
    "logistic_chaotic": DynamicalSystem(
        name="Logistic Map (r=4, chaotic)",
        dimension=1,
        lyapunov_exponents=[0.693],  # ln(2)
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=1.0,
        bifurcation_parameter=4.0,
        bifurcation_critical=3.5699,
        description="Fully developed chaos",
    ),
    "lorenz_attractor": DynamicalSystem(
        name="Lorenz Attractor (standard)",
        dimension=3,
        lyapunov_exponents=LORENZ_LYAPUNOV_EXPONENTS,
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=2.06,
        bifurcation_parameter=28,  # rho
        bifurcation_critical=24.74,  # Onset of chaos
        description="Classic strange attractor",
    ),
    "lorenz_stable": DynamicalSystem(
        name="Lorenz System (rho=10, stable)",
        dimension=3,
        lyapunov_exponents=[-1.0, -1.5, -2.0],
        attractor_type=AttractorType.POINT,
        attractor_dimension=0,
        bifurcation_parameter=10,
        bifurcation_critical=24.74,
        description="Stable fixed point regime",
    ),
    "double_pendulum": DynamicalSystem(
        name="Double Pendulum (high energy)",
        dimension=4,
        lyapunov_exponents=[0.5, 0.0, 0.0, -0.5],
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=2.5,
        description="Chaotic mechanical system",
    ),
    "henon_map": DynamicalSystem(
        name="Henon Map (a=1.4, b=0.3)",
        dimension=2,
        lyapunov_exponents=[0.42, -1.62],
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=1.26,
        description="Classic 2D chaotic map",
    ),
    "cardiac_normal": DynamicalSystem(
        name="Heart Rhythm (healthy sinus)",
        dimension=3,
        lyapunov_exponents=[-0.1, -0.2, -0.3],
        attractor_type=AttractorType.LIMIT_CYCLE,
        attractor_dimension=1,
        period=1.0,  # ~60 bpm
        description="Normal periodic heartbeat",
    ),
    "cardiac_fibrillation": DynamicalSystem(
        name="Heart Rhythm (fibrillation)",
        dimension=3,
        lyapunov_exponents=[0.3, 0.0, -0.5],
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=2.2,
        description="Chaotic cardiac dynamics",
    ),
    "brain_criticality": DynamicalSystem(
        name="Neural Activity (critical)",
        dimension=100,  # Many coupled neurons
        lyapunov_exponents=[0.01, 0.0, -0.01],  # Near zero
        attractor_type=AttractorType.STRANGE,
        attractor_dimension=50,  # Half of phase space
        description="Brain at edge of chaos (optimal computation)",
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def nonlinear_dynamics_theta_summary():
    """Print theta analysis for dynamical systems."""
    print("=" * 85)
    print("NONLINEAR DYNAMICS THETA ANALYSIS")
    print("=" * 85)
    print()
    print(f"{'System':<35} {'θ':>8} {'λ_max':>10} {'D':>8} {'Regime':<18}")
    print("-" * 85)

    for name, system in DYNAMICAL_SYSTEMS.items():
        theta = compute_dynamics_theta(system)
        regime = classify_regime(theta)

        lyap_str = f"{system.max_lyapunov:.3f}"
        dim_str = f"{system.attractor_dimension:.2f}" if system.attractor_dimension else "N/A"

        print(f"{system.name:<35} {theta:>8.3f} {lyap_str:>10} {dim_str:>8} {regime.value:<18}")

    print()
    print("Key: θ ≈ 0.5 is 'edge of chaos' (optimal for computation)")
    print("     Feigenbaum delta = 4.669... (universal period-doubling ratio)")


if __name__ == "__main__":
    nonlinear_dynamics_theta_summary()
