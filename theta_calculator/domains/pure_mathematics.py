"""
Pure Mathematics Domain Module

This module maps theta to pure mathematical structures including
algebraic geometry, representation theory, functional analysis, and combinatorics.

Theta Mapping:
    theta -> 0: Simple/trivial structure
    theta -> 1: Complex/rich structure
    theta = g / g_max: Genus complexity
    theta = dim(V) / dim(V_max): Representation dimension
    theta = 1 - 1/kappa: Condition number measure
    theta = R(n) / R_max(n): Ramsey density

Key Features:
    - Algebraic variety complexity (genus, singularities)
    - Representation theory (dimension, unitarity)
    - Functional analysis (spectral gaps, operator norms)
    - Combinatorics (Ramsey bounds, chromatic numbers)
    - Topological invariants

References:
    @book{Hartshorne1977,
      author = {Hartshorne, Robin},
      title = {Algebraic Geometry},
      publisher = {Springer},
      year = {1977}
    }
    @book{Fulton1991,
      author = {Fulton, William and Harris, Joe},
      title = {Representation Theory: A First Course},
      publisher = {Springer},
      year = {1991}
    }
    @book{Conway1990,
      author = {Conway, John B.},
      title = {A Course in Functional Analysis},
      publisher = {Springer},
      year = {1990}
    }
    @book{GrahamRothschildSpencer1990,
      author = {Graham, Ronald L. and Rothschild, Bruce L. and Spencer, Joel H.},
      title = {Ramsey Theory},
      publisher = {Wiley},
      year = {1990}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


# =============================================================================
# Mathematical Constants
# =============================================================================

# Euler-Mascheroni constant
EULER_GAMMA = 0.5772156649

# Ramsey number R(3,3) = 6
R_3_3 = 6

# Maximum known genus for "interesting" curves
GENUS_MAX_TYPICAL = 50


# =============================================================================
# Enums for Classification
# =============================================================================

class AlgebraicComplexity(Enum):
    """Classification of algebraic variety complexity."""
    TRIVIAL = "trivial"              # Point, affine/projective space
    SMOOTH = "smooth"                # Smooth variety, genus 0-1
    SINGULAR = "singular"            # Has singularities
    MODULI = "moduli"                # Moduli space structure
    DERIVED = "derived"              # Derived category level


class RepresentationType(Enum):
    """Classification of representation type."""
    FINITE_DIM = "finite_dim"        # Finite dimensional
    INFINITE_DIM = "infinite_dim"    # Infinite dimensional
    UNITARY = "unitary"              # Unitary representation
    IRREDUCIBLE = "irreducible"      # Irreducible representation
    MODULAR = "modular"              # Modular representation


class FunctionalClass(Enum):
    """Classification of functional analysis space."""
    BANACH = "banach"                # Banach space
    HILBERT = "hilbert"              # Hilbert space
    SOBOLEV = "sobolev"              # Sobolev space
    OPERATOR_ALGEBRA = "operator_algebra"  # C*-algebra, von Neumann
    DISTRIBUTION = "distribution"    # Distribution space


class CombinatorialPhase(Enum):
    """Classification of combinatorial complexity."""
    POLYNOMIAL = "polynomial"        # Polynomial bounds
    QUASI_POLYNOMIAL = "quasi_polynomial"  # n^(log n) type
    EXPONENTIAL = "exponential"      # Exponential growth
    TOWER = "tower"                  # Tower function growth


# =============================================================================
# Dataclass for Mathematical Systems
# =============================================================================

@dataclass
class PureMathSystem:
    """
    A pure mathematical system.

    Attributes:
        name: Descriptive name
        dimension: Dimension of the object
        genus: Genus (for curves/surfaces)
        rank: Rank (for groups, bundles)
        singularity_count: Number of singular points
        spectral_gap: Spectral gap (if applicable)
        condition_number: Condition number (if applicable)
        is_compact: Whether space is compact
        is_simply_connected: Whether simply connected
    """
    name: str
    dimension: int = 1
    genus: int = 0
    rank: int = 1
    singularity_count: int = 0
    spectral_gap: Optional[float] = None
    condition_number: Optional[float] = None
    is_compact: bool = True
    is_simply_connected: bool = True


# =============================================================================
# Algebraic Geometry
# =============================================================================

def genus_from_degree(degree: int, dimension: int = 1) -> int:
    r"""
    Compute genus for smooth plane curve of given degree.

    For plane curve: g = (d-1)(d-2)/2

    Args:
        degree: Degree of the curve
        dimension: Ambient dimension (default 1 for curves)

    Returns:
        Genus

    Reference: \cite{Hartshorne1977}
    """
    if degree <= 1:
        return 0
    return (degree - 1) * (degree - 2) // 2


def euler_characteristic(genus: int) -> int:
    r"""
    Euler characteristic for surface of given genus.

    chi = 2 - 2g

    Args:
        genus: Genus of the surface

    Returns:
        Euler characteristic
    """
    return 2 - 2 * genus


def compute_genus_theta(
    genus: int,
    genus_max: int = GENUS_MAX_TYPICAL
) -> float:
    r"""
    Compute theta from genus.

    Higher genus = more topological complexity.

    Args:
        genus: Genus of the variety
        genus_max: Maximum reference genus

    Returns:
        theta in [0, 1]

    Reference: \cite{Hartshorne1977}
    """
    if genus <= 0:
        return 0.0
    if genus_max <= 0:
        return 0.0

    theta = genus / genus_max
    return np.clip(theta, 0.0, 1.0)


def compute_singularity_theta(
    n_singularities: int,
    degree: int
) -> float:
    r"""
    Compute theta from singularity count.

    More singularities = higher complexity requiring resolution.

    Args:
        n_singularities: Number of singular points
        degree: Degree of variety

    Returns:
        theta in [0, 1]
    """
    if n_singularities <= 0:
        return 0.0
    if degree <= 0:
        return 0.0

    # Maximum singularities scales with degree^dim
    max_singularities = degree**2
    theta = n_singularities / max_singularities
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Representation Theory
# =============================================================================

def dimension_formula_sl2(highest_weight: int) -> int:
    r"""
    Dimension of SL(2) irreducible representation.

    dim(V_n) = n + 1 for highest weight n

    Args:
        highest_weight: Highest weight (non-negative integer)

    Returns:
        Dimension of representation

    Reference: \cite{Fulton1991}
    """
    return highest_weight + 1


def dimension_formula_sln(highest_weight: List[int], n: int) -> int:
    r"""
    Dimension of SL(n) irreducible representation using Weyl formula.

    Simplified for dominant weights.

    Args:
        highest_weight: Highest weight as partition
        n: Rank of SL(n)

    Returns:
        Dimension of representation

    Reference: \cite{Fulton1991}
    """
    if not highest_weight:
        return 1

    # Simplified: product formula for hooks
    dim = 1
    for i, w in enumerate(highest_weight):
        for j in range(w):
            dim *= (n + j - i) / (j + 1)
    return max(1, int(dim))


def compute_rank_theta(
    rank: int,
    rank_max: int = 100
) -> float:
    r"""
    Compute theta from representation rank.

    Higher rank = more complex representation structure.

    Args:
        rank: Rank of the representation/group
        rank_max: Maximum reference rank

    Returns:
        theta in [0, 1]

    Reference: \cite{Fulton1991}
    """
    if rank <= 0:
        return 0.0
    if rank_max <= 0:
        return 0.0

    theta = rank / rank_max
    return np.clip(theta, 0.0, 1.0)


def compute_dimension_theta(
    dim: int,
    dim_max: int = 1000
) -> float:
    r"""
    Compute theta from representation dimension.

    Args:
        dim: Dimension of representation
        dim_max: Maximum reference dimension

    Returns:
        theta in [0, 1]
    """
    if dim <= 1:
        return 0.0
    if dim_max <= 1:
        return 0.0

    theta = np.log(dim) / np.log(dim_max)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Functional Analysis
# =============================================================================

def spectral_gap(eigenvalues: List[float]) -> float:
    r"""
    Compute spectral gap from eigenvalue list.

    gap = lambda_1 - lambda_0 (for sorted eigenvalues)

    Args:
        eigenvalues: List of eigenvalues (sorted)

    Returns:
        Spectral gap
    """
    if len(eigenvalues) < 2:
        return 0.0

    sorted_eigs = sorted(eigenvalues)
    return sorted_eigs[1] - sorted_eigs[0]


def condition_number(eigenvalues: List[float]) -> float:
    r"""
    Compute condition number from eigenvalues.

    kappa = |lambda_max| / |lambda_min|

    Args:
        eigenvalues: List of eigenvalues

    Returns:
        Condition number
    """
    if not eigenvalues:
        return float('inf')

    abs_eigs = [abs(e) for e in eigenvalues]
    min_eig = min(abs_eigs)
    max_eig = max(abs_eigs)

    if min_eig == 0:
        return float('inf')

    return max_eig / min_eig


def compute_spectral_theta(
    gap: float,
    gap_target: float = 1.0
) -> float:
    r"""
    Compute theta from spectral gap.

    Larger gap = better separation = higher theta.

    Args:
        gap: Spectral gap
        gap_target: Target gap value

    Returns:
        theta in [0, 1]

    Reference: \cite{Conway1990}
    """
    if gap <= 0:
        return 0.0
    if gap_target <= 0:
        return 0.0

    theta = gap / gap_target
    return np.clip(theta, 0.0, 1.0)


def compute_condition_theta(
    kappa: float,
    kappa_good: float = 10.0
) -> float:
    r"""
    Compute theta from condition number.

    Lower condition number = better conditioned = higher theta.

    Args:
        kappa: Condition number
        kappa_good: Target "good" condition number

    Returns:
        theta in [0, 1]
    """
    if kappa <= 1:
        return 1.0
    if kappa == float('inf'):
        return 0.0

    theta = kappa_good / kappa
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Combinatorics
# =============================================================================

def ramsey_lower_bound(r: int, s: int) -> int:
    r"""
    Lower bound for Ramsey number R(r, s).

    R(r, s) >= 2^((r+s-2)/2) / sqrt(2)

    Args:
        r: First color class
        s: Second color class

    Returns:
        Lower bound for R(r, s)

    Reference: \cite{GrahamRothschildSpencer1990}
    """
    if r <= 1 or s <= 1:
        return 1

    return int(2**((r + s - 2) / 2) / np.sqrt(2))


def ramsey_upper_bound(r: int, s: int) -> int:
    r"""
    Upper bound for Ramsey number R(r, s).

    R(r, s) <= C(r+s-2, r-1)

    Args:
        r: First color class
        s: Second color class

    Returns:
        Upper bound for R(r, s)

    Reference: \cite{GrahamRothschildSpencer1990}
    """
    if r <= 1 or s <= 1:
        return 1

    from math import comb
    return comb(r + s - 2, r - 1)


def chromatic_number_estimate(n_vertices: int, n_edges: int) -> int:
    r"""
    Estimate chromatic number from graph statistics.

    chi >= n / (n - 2m/n) for m edges, n vertices

    Args:
        n_vertices: Number of vertices
        n_edges: Number of edges

    Returns:
        Lower bound on chromatic number
    """
    if n_vertices <= 0:
        return 0
    if n_edges <= 0:
        return 1

    avg_degree = 2 * n_edges / n_vertices
    if avg_degree >= n_vertices - 1:
        return n_vertices

    return max(1, int(n_vertices / (n_vertices - avg_degree)))


def compute_ramsey_theta(
    known_value: int,
    lower_bound: int,
    upper_bound: int
) -> float:
    r"""
    Compute theta for Ramsey number knowledge.

    Theta measures how tight our bounds are.

    Args:
        known_value: Best known value (or estimate)
        lower_bound: Lower bound
        upper_bound: Upper bound

    Returns:
        theta in [0, 1]: 1 = exact value known

    Reference: \cite{GrahamRothschildSpencer1990}
    """
    if upper_bound <= lower_bound:
        return 1.0  # Exact value known

    if known_value < lower_bound or known_value > upper_bound:
        return 0.0

    gap = upper_bound - lower_bound
    if gap == 0:
        return 1.0

    # How close are bounds to each other?
    theta = 1.0 - (gap / upper_bound)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Unified Pure Math Theta
# =============================================================================

def compute_pure_math_theta(system: PureMathSystem) -> float:
    r"""
    Compute unified theta for pure math system.

    Combines multiple structural aspects.

    Args:
        system: PureMathSystem dataclass

    Returns:
        theta in [0, 1]
    """
    thetas = []

    # Genus contribution
    if system.genus > 0:
        theta_genus = compute_genus_theta(system.genus)
        thetas.append(theta_genus)

    # Rank contribution
    if system.rank > 1:
        theta_rank = compute_rank_theta(system.rank)
        thetas.append(theta_rank)

    # Singularity contribution
    if system.singularity_count > 0:
        theta_sing = compute_singularity_theta(
            system.singularity_count,
            system.dimension + 2
        )
        thetas.append(theta_sing)

    # Spectral gap contribution
    if system.spectral_gap is not None:
        theta_spec = compute_spectral_theta(system.spectral_gap)
        thetas.append(theta_spec)

    # Condition number contribution
    if system.condition_number is not None:
        theta_cond = compute_condition_theta(system.condition_number)
        thetas.append(theta_cond)

    if not thetas:
        # Default based on dimension
        return np.clip(system.dimension / 10, 0.0, 1.0)

    return np.prod(thetas)**(1/len(thetas))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_algebraic_complexity(
    genus: int,
    n_singularities: int
) -> AlgebraicComplexity:
    """
    Classify algebraic variety complexity.

    Args:
        genus: Genus of the variety
        n_singularities: Number of singular points

    Returns:
        AlgebraicComplexity enum
    """
    if n_singularities > 0:
        return AlgebraicComplexity.SINGULAR
    elif genus == 0:
        return AlgebraicComplexity.TRIVIAL
    elif genus <= 1:
        return AlgebraicComplexity.SMOOTH
    elif genus <= 10:
        return AlgebraicComplexity.MODULI
    else:
        return AlgebraicComplexity.DERIVED


def classify_representation(
    dimension: int,
    is_unitary: bool = False
) -> RepresentationType:
    """
    Classify representation type.

    Args:
        dimension: Dimension of representation
        is_unitary: Whether representation is unitary

    Returns:
        RepresentationType enum
    """
    if is_unitary:
        return RepresentationType.UNITARY
    elif dimension == 1:
        return RepresentationType.IRREDUCIBLE
    elif dimension < float('inf'):
        return RepresentationType.FINITE_DIM
    else:
        return RepresentationType.INFINITE_DIM


def classify_functional_space(
    has_inner_product: bool,
    is_complete: bool
) -> FunctionalClass:
    """
    Classify functional analysis space.

    Args:
        has_inner_product: Whether space has inner product
        is_complete: Whether space is complete

    Returns:
        FunctionalClass enum
    """
    if has_inner_product and is_complete:
        return FunctionalClass.HILBERT
    elif is_complete:
        return FunctionalClass.BANACH
    else:
        return FunctionalClass.SOBOLEV


def classify_combinatorial_phase(theta: float) -> CombinatorialPhase:
    """
    Classify combinatorial complexity from theta.

    Args:
        theta: Theta value [0, 1]

    Returns:
        CombinatorialPhase enum
    """
    if theta < 0.25:
        return CombinatorialPhase.POLYNOMIAL
    elif theta < 0.5:
        return CombinatorialPhase.QUASI_POLYNOMIAL
    elif theta < 0.75:
        return CombinatorialPhase.EXPONENTIAL
    else:
        return CombinatorialPhase.TOWER


# =============================================================================
# Example Systems Dictionary
# =============================================================================

PURE_MATH_SYSTEMS: Dict[str, PureMathSystem] = {
    "elliptic_curve": PureMathSystem(
        name="Elliptic Curve",
        dimension=1,
        genus=1,
        rank=1
    ),
    "k3_surface": PureMathSystem(
        name="K3 Surface",
        dimension=2,
        genus=0,  # Trivial canonical
        rank=22,  # Picard rank up to 22
        is_simply_connected=True
    ),
    "moduli_curves": PureMathSystem(
        name="Moduli of Genus g Curves",
        dimension=10,  # M_g for g=4: dim = 3g-3 = 9
        genus=4,
        rank=4
    ),
    "sl2r_reps": PureMathSystem(
        name="SL(2,R) Representations",
        dimension=1000,  # Discrete series
        rank=1,
        spectral_gap=0.25
    ),
    "laplacian_sphere": PureMathSystem(
        name="Laplacian on Sphere",
        dimension=2,
        spectral_gap=2.0,  # First nonzero eigenvalue
        condition_number=1.0,
        is_compact=True
    ),
    "ramsey_r55": PureMathSystem(
        name="Ramsey R(5,5)",
        dimension=0,
        # R(5,5) in [43, 48]
        rank=5
    ),
    "hilbert_l2": PureMathSystem(
        name="L^2 Hilbert Space",
        dimension=1000000,  # Infinite, but represented finitely
        spectral_gap=0.0,
        is_compact=False
    ),
    "calabi_yau": PureMathSystem(
        name="Calabi-Yau Threefold",
        dimension=3,
        genus=0,  # Trivial canonical
        rank=100,  # Hodge numbers
        is_simply_connected=True
    ),
    "graph_chromatic": PureMathSystem(
        name="Graph Coloring Problem",
        dimension=100,  # vertices
        rank=4,  # chromatic number
        singularity_count=0
    ),
    "abelian_variety": PureMathSystem(
        name="Abelian Variety",
        dimension=4,
        genus=2,
        rank=4,
        is_compact=True
    ),
}


# Precomputed theta values
PURE_MATH_THETA_VALUES: Dict[str, float] = {
    name: compute_pure_math_theta(system)
    for name, system in PURE_MATH_SYSTEMS.items()
}
