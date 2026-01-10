"""
Applied Mathematics Domain Module

This module maps theta to applied mathematical systems including
PDEs, optimization, numerical analysis, and probability.

Theta Mapping:
    theta -> 0: Ill-posed/unstable/slow convergence
    theta -> 1: Well-posed/stable/fast convergence
    theta = order / order_max: Convergence order
    theta = 1 / kappa: Conditioning measure
    theta = gap / gap_max: Duality gap closure

Key Features:
    - PDE regularity and stability
    - Optimization convergence
    - Numerical conditioning
    - Stochastic process analysis

References:
    @book{Evans2010,
      author = {Evans, Lawrence C.},
      title = {Partial Differential Equations},
      publisher = {American Mathematical Society},
      year = {2010}
    }
    @book{BoydVandenberghe2004,
      author = {Boyd, Stephen and Vandenberghe, Lieven},
      title = {Convex Optimization},
      publisher = {Cambridge University Press},
      year = {2004}
    }
    @book{TrefethenBau1997,
      author = {Trefethen, Lloyd N. and Bau, David},
      title = {Numerical Linear Algebra},
      publisher = {SIAM},
      year = {1997}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

# Target condition number for "well-conditioned"
KAPPA_GOOD = 100.0

# Target convergence order
ORDER_SPECTRAL = 10.0


# =============================================================================
# Enums for Classification
# =============================================================================

class PDEType(Enum):
    """Classification of PDE type."""
    ELLIPTIC = "elliptic"            # Laplace, Poisson
    PARABOLIC = "parabolic"          # Heat equation
    HYPERBOLIC = "hyperbolic"        # Wave equation
    MIXED = "mixed"                  # Mixed type


class OptimizationClass(Enum):
    """Classification of optimization problem."""
    CONVEX = "convex"                # Convex optimization
    NONCONVEX = "nonconvex"          # Non-convex
    COMBINATORIAL = "combinatorial"  # Discrete optimization
    STOCHASTIC = "stochastic"        # Stochastic optimization


class ConvergenceType(Enum):
    """Classification of convergence type."""
    SUBLINEAR = "sublinear"          # O(1/k)
    LINEAR = "linear"                # O(r^k), r < 1
    SUPERLINEAR = "superlinear"      # Faster than linear
    QUADRATIC = "quadratic"          # O(r^{2^k})


class StabilityClass(Enum):
    """Classification of numerical stability."""
    UNSTABLE = "unstable"            # Blow-up possible
    CONDITIONALLY = "conditionally"  # CFL condition
    UNCONDITIONALLY = "unconditionally"  # Always stable
    ASYMPTOTIC = "asymptotic"        # Stable at large t


# =============================================================================
# Dataclass for Applied Math Systems
# =============================================================================

@dataclass
class AppliedMathSystem:
    """
    An applied mathematics system.

    Attributes:
        name: Descriptive name
        dimension: Problem dimension
        condition_number: Condition number
        convergence_order: Order of accuracy
        convergence_rate: Linear convergence rate
        regularity: Solution regularity (Sobolev index)
        cfl_number: CFL number for stability
        pde_type: Type of PDE (if applicable)
    """
    name: str
    dimension: int = 1
    condition_number: float = 1.0
    convergence_order: float = 2.0
    convergence_rate: Optional[float] = None
    regularity: float = 2.0
    cfl_number: Optional[float] = None
    pde_type: Optional[PDEType] = None


# =============================================================================
# PDE Analysis
# =============================================================================

def compute_regularity_theta(
    regularity: float,
    regularity_target: float = 2.0
) -> float:
    r"""
    Compute theta from solution regularity.

    Higher regularity = smoother solution = better behaved.

    Args:
        regularity: Sobolev index of solution
        regularity_target: Target regularity (e.g., H^2)

    Returns:
        theta in [0, 1]

    Reference: \cite{Evans2010}
    """
    if regularity <= 0:
        return 0.0
    if regularity_target <= 0:
        return 0.0

    theta = regularity / regularity_target
    return np.clip(theta, 0.0, 1.0)


def compute_stability_theta(
    cfl: float,
    cfl_max: float = 1.0
) -> float:
    r"""
    Compute theta from CFL condition.

    CFL <= CFL_max for stability.

    Args:
        cfl: Actual CFL number
        cfl_max: Maximum stable CFL

    Returns:
        theta in [0, 1]: 1 = well within stability

    Reference: \cite{TrefethenBau1997}
    """
    if cfl <= 0:
        return 1.0  # No constraint
    if cfl_max <= 0:
        return 0.0

    if cfl >= cfl_max:
        return 0.0  # Unstable

    theta = 1 - (cfl / cfl_max)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Optimization
# =============================================================================

def compute_convergence_theta(
    order: float,
    order_max: float = ORDER_SPECTRAL
) -> float:
    r"""
    Compute theta from convergence order.

    Higher order = faster convergence.

    Args:
        order: Convergence order (e.g., 2 for O(h^2))
        order_max: Maximum/target order

    Returns:
        theta in [0, 1]

    Reference: \cite{BoydVandenberghe2004}
    """
    if order <= 0:
        return 0.0
    if order_max <= 0:
        return 0.0

    theta = order / order_max
    return np.clip(theta, 0.0, 1.0)


def compute_rate_theta(
    rate: float,
    rate_good: float = 0.5
) -> float:
    r"""
    Compute theta from linear convergence rate.

    Lower rate = faster convergence (r < 1).

    Args:
        rate: Linear convergence rate (0 < r < 1)
        rate_good: Target "fast" rate

    Returns:
        theta in [0, 1]: 1 = fast convergence

    Reference: \cite{BoydVandenberghe2004}
    """
    if rate <= 0:
        return 1.0  # Instantaneous
    if rate >= 1:
        return 0.0  # Not converging

    theta = rate_good / rate
    return np.clip(theta, 0.0, 1.0)


def compute_duality_gap_theta(
    primal: float,
    dual: float
) -> float:
    r"""
    Compute theta from duality gap.

    Zero gap = optimal.

    Args:
        primal: Primal objective value
        dual: Dual objective value

    Returns:
        theta in [0, 1]: 1 = no gap

    Reference: \cite{BoydVandenberghe2004}
    """
    if primal == 0:
        return 1.0 if dual == 0 else 0.0

    gap = abs(primal - dual) / abs(primal)
    theta = 1 - gap
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Numerical Analysis
# =============================================================================

def compute_condition_theta(
    kappa: float,
    kappa_good: float = KAPPA_GOOD
) -> float:
    r"""
    Compute theta from condition number.

    Lower kappa = better conditioned.

    Args:
        kappa: Condition number
        kappa_good: Target "good" conditioning

    Returns:
        theta in [0, 1]

    Reference: \cite{TrefethenBau1997}
    """
    if kappa <= 1:
        return 1.0
    if kappa == float('inf'):
        return 0.0

    theta = kappa_good / kappa
    return np.clip(theta, 0.0, 1.0)


def compute_accuracy_theta(
    error: float,
    tolerance: float = 1e-6
) -> float:
    r"""
    Compute theta from numerical error.

    Args:
        error: Numerical error
        tolerance: Target tolerance

    Returns:
        theta in [0, 1]: 1 = within tolerance
    """
    if error <= 0:
        return 1.0
    if tolerance <= 0:
        return 0.0

    if error <= tolerance:
        return 1.0

    theta = tolerance / error
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Stochastic Analysis
# =============================================================================

def compute_mixing_theta(
    mixing_time: float,
    target_time: float = 100.0
) -> float:
    r"""
    Compute theta from mixing time.

    Faster mixing = higher theta.

    Args:
        mixing_time: Time to mix
        target_time: Target mixing time

    Returns:
        theta in [0, 1]
    """
    if mixing_time <= 0:
        return 1.0
    if target_time <= 0:
        return 0.0

    theta = target_time / mixing_time
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Unified Applied Math Theta
# =============================================================================

def compute_applied_math_theta(system: AppliedMathSystem) -> float:
    r"""
    Compute unified theta for applied math system.

    Args:
        system: AppliedMathSystem dataclass

    Returns:
        theta in [0, 1]
    """
    thetas = []

    # Condition number contribution
    if system.condition_number > 0:
        theta_cond = compute_condition_theta(system.condition_number)
        thetas.append(theta_cond)

    # Convergence order contribution
    if system.convergence_order > 0:
        theta_order = compute_convergence_theta(system.convergence_order)
        thetas.append(theta_order)

    # Convergence rate contribution
    if system.convergence_rate is not None:
        theta_rate = compute_rate_theta(system.convergence_rate)
        thetas.append(theta_rate)

    # Regularity contribution
    if system.regularity > 0:
        theta_reg = compute_regularity_theta(system.regularity)
        thetas.append(theta_reg)

    # CFL stability contribution
    if system.cfl_number is not None:
        theta_cfl = compute_stability_theta(system.cfl_number)
        thetas.append(theta_cfl)

    if not thetas:
        return 0.5

    return np.prod(thetas)**(1/len(thetas))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_pde_type(
    a: float,
    b: float,
    c: float
) -> PDEType:
    """
    Classify PDE from coefficients a*u_xx + b*u_xy + c*u_yy.

    Args:
        a, b, c: Second-order coefficients

    Returns:
        PDEType enum
    """
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return PDEType.ELLIPTIC
    elif discriminant == 0:
        return PDEType.PARABOLIC
    else:
        return PDEType.HYPERBOLIC


def classify_optimization(
    is_convex: bool,
    is_discrete: bool,
    is_stochastic: bool
) -> OptimizationClass:
    """
    Classify optimization problem.

    Args:
        is_convex: Whether objective is convex
        is_discrete: Whether variables are discrete
        is_stochastic: Whether problem is stochastic

    Returns:
        OptimizationClass enum
    """
    if is_stochastic:
        return OptimizationClass.STOCHASTIC
    elif is_discrete:
        return OptimizationClass.COMBINATORIAL
    elif is_convex:
        return OptimizationClass.CONVEX
    else:
        return OptimizationClass.NONCONVEX


def classify_convergence(
    order: float,
    rate: Optional[float] = None
) -> ConvergenceType:
    """
    Classify convergence type.

    Args:
        order: Convergence order
        rate: Linear rate (if applicable)

    Returns:
        ConvergenceType enum
    """
    if order >= 2:
        return ConvergenceType.QUADRATIC
    elif order > 1:
        return ConvergenceType.SUPERLINEAR
    elif order == 1:
        return ConvergenceType.LINEAR
    else:
        return ConvergenceType.SUBLINEAR


def classify_stability(theta: float) -> StabilityClass:
    """
    Classify stability from theta.

    Args:
        theta: Stability theta [0, 1]

    Returns:
        StabilityClass enum
    """
    if theta < 0.25:
        return StabilityClass.UNSTABLE
    elif theta < 0.5:
        return StabilityClass.CONDITIONALLY
    elif theta < 0.75:
        return StabilityClass.ASYMPTOTIC
    else:
        return StabilityClass.UNCONDITIONALLY


# =============================================================================
# Example Systems Dictionary
# =============================================================================

APPLIED_MATH_SYSTEMS: Dict[str, AppliedMathSystem] = {
    "laplace_fd": AppliedMathSystem(
        name="Laplace Equation (Finite Difference)",
        dimension=2,
        condition_number=100.0,
        convergence_order=2.0,
        regularity=2.0,
        pde_type=PDEType.ELLIPTIC
    ),
    "heat_explicit": AppliedMathSystem(
        name="Heat Equation (Explicit)",
        dimension=2,
        condition_number=10.0,
        convergence_order=2.0,
        cfl_number=0.4,
        pde_type=PDEType.PARABOLIC
    ),
    "wave_leapfrog": AppliedMathSystem(
        name="Wave Equation (Leapfrog)",
        dimension=2,
        condition_number=1.0,
        convergence_order=2.0,
        cfl_number=0.8,
        pde_type=PDEType.HYPERBOLIC
    ),
    "navier_stokes": AppliedMathSystem(
        name="Navier-Stokes (DNS)",
        dimension=3,
        condition_number=1e6,
        convergence_order=2.0,
        regularity=1.5,
        pde_type=PDEType.MIXED
    ),
    "newton_optimization": AppliedMathSystem(
        name="Newton's Method",
        dimension=100,
        condition_number=10.0,
        convergence_order=2.0,
        convergence_rate=0.1
    ),
    "gradient_descent": AppliedMathSystem(
        name="Gradient Descent",
        dimension=1000,
        condition_number=100.0,
        convergence_order=1.0,
        convergence_rate=0.9
    ),
    "interior_point": AppliedMathSystem(
        name="Interior Point Method",
        dimension=500,
        condition_number=50.0,
        convergence_order=2.0,
        convergence_rate=0.3
    ),
    "spectral_method": AppliedMathSystem(
        name="Spectral Method",
        dimension=2,
        condition_number=1000.0,
        convergence_order=10.0,  # Exponential for smooth
        regularity=10.0
    ),
    "multigrid": AppliedMathSystem(
        name="Multigrid Solver",
        dimension=3,
        condition_number=10.0,
        convergence_order=2.0,
        convergence_rate=0.1
    ),
    "monte_carlo": AppliedMathSystem(
        name="Monte Carlo Integration",
        dimension=100,
        condition_number=1.0,
        convergence_order=0.5,  # 1/sqrt(n)
        regularity=0.0
    ),
}


# Precomputed theta values
APPLIED_MATH_THETA_VALUES: Dict[str, float] = {
    name: compute_applied_math_theta(system)
    for name, system in APPLIED_MATH_SYSTEMS.items()
}
