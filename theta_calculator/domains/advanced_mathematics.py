"""
Advanced Mathematics Domain Module

This module maps theta to mathematical structures including topology,
differential geometry, and mathematical physics.

Theta Mapping:
    theta -> 0: Trivial/simple/integrable structure
    theta -> 1: Complex/non-trivial/chaotic structure
    theta = sum(betti)/dim: Topological complexity
    theta = |K|/K_max: Curvature measure
    theta = 1 - (n_conserved/n_dof): Non-integrability

Key Features:
    - Topological invariants (Betti numbers, Euler characteristic)
    - Differential geometric properties (curvature, dimension)
    - Dynamical systems (integrability, chaos)
    - Symmetry analysis (Lie groups, gauge theory)

References:
    @book{Milnor1963,
      author = {Milnor, John},
      title = {Morse Theory},
      publisher = {Princeton University Press},
      year = {1963}
    }
    @article{Thurston1982,
      author = {Thurston, William},
      title = {Three-dimensional manifolds, Kleinian groups and hyperbolic geometry},
      journal = {Bull. AMS},
      year = {1982}
    }
    @article{Perelman2003,
      author = {Perelman, Grigori},
      title = {Ricci flow with surgery on three-manifolds},
      journal = {arXiv},
      year = {2003}
    }
    @book{Arnold1989,
      author = {Arnold, Vladimir I.},
      title = {Mathematical Methods of Classical Mechanics},
      publisher = {Springer},
      year = {1989}
    }
    @article{Witten1982,
      author = {Witten, Edward},
      title = {Supersymmetry and Morse theory},
      journal = {J. Diff. Geom.},
      year = {1982}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


# =============================================================================
# Mathematical Constants
# =============================================================================

EULER_MASCHERONI = 0.5772156649015329  # gamma
FEIGENBAUM_DELTA = 4.669201609102990   # Period-doubling constant
FEIGENBAUM_ALPHA = 2.502907875095892   # Scaling constant
APERY_CONSTANT = 1.2020569031595942    # zeta(3)
PI = np.pi
E = np.e

# Topological constants
EULER_CHAR_S2 = 2      # Euler characteristic of 2-sphere
EULER_CHAR_T2 = 0      # Euler characteristic of 2-torus
EULER_CHAR_RP2 = 1     # Euler characteristic of real projective plane


# =============================================================================
# Enums for Classification
# =============================================================================

class TopologicalPhase(Enum):
    """Classification of topological complexity."""
    TRIVIAL = "trivial"                 # Contractible, no topology
    SIMPLY_CONNECTED = "simply_connected"  # pi_1 = 0 but may have higher
    NON_TRIVIAL_PI1 = "non_trivial_pi1"   # Non-trivial fundamental group
    HIGHER_HOMOTOPY = "higher_homotopy"   # Non-trivial higher homotopy


class GeometricRegime(Enum):
    """Classification of geometric curvature."""
    FLAT = "flat"                       # K = 0 (Euclidean)
    POSITIVE_CURVED = "positive"        # K > 0 (Spherical)
    NEGATIVE_CURVED = "negative"        # K < 0 (Hyperbolic)
    MIXED = "mixed"                     # Variable curvature


class IntegrabilityLevel(Enum):
    """Classification of dynamical system integrability."""
    COMPLETELY_INTEGRABLE = "complete"  # Liouville integrable
    PARTIALLY_INTEGRABLE = "partial"    # Some conserved quantities
    CHAOTIC = "chaotic"                 # Non-integrable, sensitive to IC


class SymmetryType(Enum):
    """Classification of symmetry."""
    DISCRETE = "discrete"               # Finite group
    CONTINUOUS = "continuous"           # Lie group
    GAUGE = "gauge"                     # Local symmetry
    SUPERSYMMETRIC = "supersymmetric"   # SUSY


class ManifoldType(Enum):
    """Classification of manifold types."""
    EUCLIDEAN = "euclidean"             # R^n
    SPHERICAL = "spherical"             # S^n
    HYPERBOLIC = "hyperbolic"           # H^n
    TORUS = "torus"                     # T^n
    PROJECTIVE = "projective"           # RP^n, CP^n
    CALABI_YAU = "calabi_yau"           # CY manifolds


# =============================================================================
# Dataclass for Mathematical Systems
# =============================================================================

@dataclass
class MathematicalSystem:
    """
    A mathematical system for theta analysis.

    Attributes:
        name: System identifier
        domain: Mathematical domain (topology, geometry, physics)
        dimension: Topological/geometric dimension
        curvature: Scalar curvature (or representative value)
        euler_characteristic: Topological invariant chi
        betti_numbers: List of Betti numbers [b0, b1, b2, ...]
        fundamental_group_rank: Rank of pi_1
        symmetry_dimension: Dimension of symmetry group
        n_conserved: Number of conserved quantities (for dynamics)
        n_degrees_of_freedom: Number of degrees of freedom
        genus: Surface genus (for 2D surfaces)
        is_orientable: Whether manifold is orientable
        is_compact: Whether manifold is compact
    """
    name: str
    domain: str  # "topology", "geometry", "dynamics"
    dimension: int
    curvature: float = 0.0
    euler_characteristic: Optional[int] = None
    betti_numbers: Optional[List[int]] = None
    fundamental_group_rank: int = 0
    symmetry_dimension: int = 0
    n_conserved: int = 0
    n_degrees_of_freedom: int = 1
    genus: int = 0
    is_orientable: bool = True
    is_compact: bool = True

    @property
    def total_betti(self) -> int:
        """Sum of all Betti numbers."""
        if self.betti_numbers is None:
            return 0
        return sum(self.betti_numbers)

    @property
    def is_simply_connected(self) -> bool:
        """Check if pi_1 = 0."""
        return self.fundamental_group_rank == 0

    @property
    def integrability_ratio(self) -> float:
        """Ratio of conserved quantities to degrees of freedom."""
        if self.n_degrees_of_freedom == 0:
            return 0.0
        return self.n_conserved / self.n_degrees_of_freedom

    @property
    def is_integrable(self) -> bool:
        """Check if completely integrable (Liouville)."""
        return self.n_conserved >= self.n_degrees_of_freedom


# =============================================================================
# Theta Calculation Functions
# =============================================================================

def compute_topological_theta(
    betti_numbers: List[int],
    dimension: int
) -> float:
    """
    Compute theta from topological complexity.

    theta = sum(betti_numbers) / (dimension + 1), capped at 1

    Higher Betti numbers indicate more "holes" and topological complexity.

    Args:
        betti_numbers: List of Betti numbers [b0, b1, b2, ...]
        dimension: Manifold dimension

    Returns:
        Theta in [0, 1] where 1 = topologically complex
    """
    if not betti_numbers or dimension < 0:
        return 0.0

    total_betti = sum(betti_numbers)

    # Normalize by dimension + 1 (expected number of Betti numbers)
    # For n-sphere: only b0 = bn = 1, so total = 2
    # For n-torus: all betti = C(n,k), total = 2^n
    max_expected = 2 ** dimension  # Upper bound like n-torus

    theta = total_betti / max_expected

    return float(np.clip(theta, 0.0, 1.0))


def compute_euler_theta(
    euler_characteristic: int,
    dimension: int
) -> float:
    """
    Compute theta from Euler characteristic.

    Deviations from expected values indicate topological complexity.

    Args:
        euler_characteristic: Euler characteristic chi
        dimension: Manifold dimension

    Returns:
        Theta in [0, 1]
    """
    # Expected for sphere: chi = 1 + (-1)^n
    if dimension % 2 == 0:
        expected = 2  # Even-dimensional spheres
    else:
        expected = 0  # Odd-dimensional

    deviation = abs(euler_characteristic - expected)
    # Larger deviation = more interesting topology
    theta = 1.0 - 1.0 / (1.0 + deviation)

    return float(np.clip(theta, 0.0, 1.0))


def compute_curvature_theta(
    curvature: float,
    curvature_scale: float = 1.0
) -> float:
    """
    Compute theta from curvature.

    theta = |K| / (|K| + scale), sigmoid-like mapping

    Args:
        curvature: Scalar curvature
        curvature_scale: Scale parameter for normalization

    Returns:
        Theta in [0, 1] where 1 = highly curved
    """
    if curvature_scale <= 0:
        return 0.5

    abs_k = abs(curvature)
    theta = abs_k / (abs_k + curvature_scale)

    return float(np.clip(theta, 0.0, 1.0))


def compute_integrability_theta(
    n_conserved: int,
    n_dof: int
) -> float:
    """
    Compute theta for dynamical system integrability.

    theta = 1 - (n_conserved / n_dof)

    Higher theta = more chaotic/non-integrable

    Args:
        n_conserved: Number of independent conserved quantities
        n_dof: Number of degrees of freedom

    Returns:
        Theta in [0, 1] where 1 = fully chaotic
    """
    if n_dof <= 0:
        return 0.5

    ratio = min(n_conserved / n_dof, 1.0)
    theta = 1.0 - ratio

    return float(np.clip(theta, 0.0, 1.0))


def compute_symmetry_theta(
    symmetry_dim: int,
    space_dim: int
) -> float:
    """
    Compute theta from symmetry richness.

    theta = symmetry_dim / max_possible

    Args:
        symmetry_dim: Dimension of symmetry group
        space_dim: Dimension of space

    Returns:
        Theta in [0, 1] where 1 = maximally symmetric
    """
    if space_dim <= 0:
        return 0.0

    # Maximum symmetry: isometry group of sphere
    # dim(SO(n+1)) = n(n+1)/2
    max_sym = space_dim * (space_dim + 1) // 2

    if max_sym <= 0:
        return 0.0

    theta = symmetry_dim / max_sym

    return float(np.clip(theta, 0.0, 1.0))


def compute_homotopy_theta(
    fundamental_group_rank: int,
    higher_homotopy_complexity: int = 0
) -> float:
    """
    Compute theta from homotopy groups.

    Args:
        fundamental_group_rank: Rank of pi_1
        higher_homotopy_complexity: Measure of higher pi_n complexity

    Returns:
        Theta in [0, 1] where 1 = homotopically complex
    """
    # Fundamental group contribution
    pi1_theta = 1.0 - 1.0 / (1.0 + fundamental_group_rank)

    # Higher homotopy contribution
    higher_theta = 1.0 - 1.0 / (1.0 + higher_homotopy_complexity)

    # Combine
    theta = 0.6 * pi1_theta + 0.4 * higher_theta

    return float(np.clip(theta, 0.0, 1.0))


def compute_math_theta(system: MathematicalSystem) -> float:
    """
    Compute unified theta for a mathematical system.

    Routes to domain-specific calculation.

    Args:
        system: MathematicalSystem instance

    Returns:
        Theta in [0, 1]
    """
    domain = system.domain.lower()
    thetas = []
    weights = []

    if domain == "topology":
        # Topological complexity
        if system.betti_numbers:
            thetas.append(compute_topological_theta(
                system.betti_numbers, system.dimension
            ))
            weights.append(0.4)

        if system.euler_characteristic is not None:
            thetas.append(compute_euler_theta(
                system.euler_characteristic, system.dimension
            ))
            weights.append(0.3)

        thetas.append(compute_homotopy_theta(system.fundamental_group_rank))
        weights.append(0.3)

    elif domain == "geometry":
        # Geometric properties
        thetas.append(compute_curvature_theta(system.curvature))
        weights.append(0.5)

        thetas.append(compute_symmetry_theta(
            system.symmetry_dimension, system.dimension
        ))
        weights.append(0.3)

        if system.euler_characteristic is not None:
            thetas.append(compute_euler_theta(
                system.euler_characteristic, system.dimension
            ))
            weights.append(0.2)

    elif domain == "dynamics":
        # Dynamical systems
        thetas.append(compute_integrability_theta(
            system.n_conserved, system.n_degrees_of_freedom
        ))
        weights.append(0.6)

        thetas.append(compute_symmetry_theta(
            system.symmetry_dimension, system.n_degrees_of_freedom
        ))
        weights.append(0.4)

    else:
        # Default: combine available metrics
        if system.betti_numbers:
            thetas.append(compute_topological_theta(
                system.betti_numbers, system.dimension
            ))
            weights.append(1.0)
        else:
            thetas.append(compute_curvature_theta(system.curvature))
            weights.append(1.0)

    if not thetas:
        return 0.5

    total_weight = sum(weights)
    theta = sum(t * w for t, w in zip(thetas, weights)) / total_weight

    return float(np.clip(theta, 0.0, 1.0))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_topological_phase(
    fundamental_group_rank: int,
    has_higher_homotopy: bool = False
) -> TopologicalPhase:
    """Classify topological phase from homotopy data."""
    if fundamental_group_rank == 0:
        if has_higher_homotopy:
            return TopologicalPhase.SIMPLY_CONNECTED
        return TopologicalPhase.TRIVIAL
    elif fundamental_group_rank > 0:
        if has_higher_homotopy:
            return TopologicalPhase.HIGHER_HOMOTOPY
        return TopologicalPhase.NON_TRIVIAL_PI1
    return TopologicalPhase.TRIVIAL


def classify_geometric_regime(curvature: float, tolerance: float = 1e-6) -> GeometricRegime:
    """Classify geometric regime from curvature."""
    if abs(curvature) < tolerance:
        return GeometricRegime.FLAT
    elif curvature > 0:
        return GeometricRegime.POSITIVE_CURVED
    else:
        return GeometricRegime.NEGATIVE_CURVED


def classify_integrability(
    n_conserved: int,
    n_dof: int
) -> IntegrabilityLevel:
    """Classify integrability level."""
    if n_dof <= 0:
        return IntegrabilityLevel.CHAOTIC

    ratio = n_conserved / n_dof

    if ratio >= 1.0:
        return IntegrabilityLevel.COMPLETELY_INTEGRABLE
    elif ratio >= 0.5:
        return IntegrabilityLevel.PARTIALLY_INTEGRABLE
    else:
        return IntegrabilityLevel.CHAOTIC


def classify_symmetry(
    symmetry_dim: int,
    is_local: bool = False,
    has_susy: bool = False
) -> SymmetryType:
    """Classify symmetry type."""
    if has_susy:
        return SymmetryType.SUPERSYMMETRIC
    if is_local:
        return SymmetryType.GAUGE
    if symmetry_dim > 0:
        return SymmetryType.CONTINUOUS
    return SymmetryType.DISCRETE


# =============================================================================
# Example Systems
# =============================================================================

MATHEMATICAL_SYSTEMS: Dict[str, MathematicalSystem] = {
    # Topology examples
    "euclidean_r3": MathematicalSystem(
        name="Euclidean R^3",
        domain="topology",
        dimension=3,
        curvature=0.0,
        euler_characteristic=1,
        betti_numbers=[1, 0, 0, 0],
        fundamental_group_rank=0,
        is_compact=False,
    ),
    "sphere_s2": MathematicalSystem(
        name="2-Sphere S^2",
        domain="geometry",
        dimension=2,
        curvature=1.0,  # Positive constant
        euler_characteristic=2,
        betti_numbers=[1, 0, 1],
        fundamental_group_rank=0,
        symmetry_dimension=3,  # SO(3)
    ),
    "sphere_s3": MathematicalSystem(
        name="3-Sphere S^3",
        domain="topology",
        dimension=3,
        curvature=1.0,
        euler_characteristic=0,
        betti_numbers=[1, 0, 0, 1],
        fundamental_group_rank=0,
    ),
    "hyperbolic_h2": MathematicalSystem(
        name="Hyperbolic Plane H^2",
        domain="geometry",
        dimension=2,
        curvature=-1.0,  # Negative constant
        euler_characteristic=0,  # Non-compact
        betti_numbers=[1, 0, 0],
        fundamental_group_rank=0,
        is_compact=False,
    ),
    "torus_t2": MathematicalSystem(
        name="2-Torus T^2",
        domain="topology",
        dimension=2,
        curvature=0.0,  # Flat
        euler_characteristic=0,
        betti_numbers=[1, 2, 1],
        fundamental_group_rank=2,  # Z x Z
        genus=1,
    ),
    "torus_t3": MathematicalSystem(
        name="3-Torus T^3",
        domain="topology",
        dimension=3,
        curvature=0.0,
        euler_characteristic=0,
        betti_numbers=[1, 3, 3, 1],
        fundamental_group_rank=3,
    ),
    "klein_bottle": MathematicalSystem(
        name="Klein Bottle",
        domain="topology",
        dimension=2,
        curvature=0.0,
        euler_characteristic=0,
        betti_numbers=[1, 1, 0],  # b1=1 for Z coefficient
        fundamental_group_rank=2,
        is_orientable=False,
    ),
    "projective_plane_rp2": MathematicalSystem(
        name="Real Projective Plane RP^2",
        domain="topology",
        dimension=2,
        curvature=1.0,
        euler_characteristic=1,
        betti_numbers=[1, 0, 0],  # Over R
        fundamental_group_rank=1,  # Z/2Z
        is_orientable=False,
    ),
    "calabi_yau_3fold": MathematicalSystem(
        name="Calabi-Yau 3-fold",
        domain="geometry",
        dimension=6,  # Complex dim 3
        curvature=0.0,  # Ricci-flat
        euler_characteristic=-200,  # Typical quintic
        betti_numbers=[1, 0, 1, 204, 1, 0, 1],  # Quintic
        fundamental_group_rank=0,
    ),

    # Dynamical systems
    "harmonic_oscillator": MathematicalSystem(
        name="Harmonic Oscillator",
        domain="dynamics",
        dimension=2,
        n_conserved=1,  # Energy
        n_degrees_of_freedom=1,
        symmetry_dimension=1,  # U(1) phase
    ),
    "kepler_problem": MathematicalSystem(
        name="Kepler Problem",
        domain="dynamics",
        dimension=6,
        n_conserved=5,  # E, L (3), LRL hidden
        n_degrees_of_freedom=3,
        symmetry_dimension=6,  # SO(4) hidden
    ),
    "henon_heiles": MathematicalSystem(
        name="Henon-Heiles System",
        domain="dynamics",
        dimension=4,
        n_conserved=1,  # Only energy
        n_degrees_of_freedom=2,
        symmetry_dimension=0,
    ),
    "three_body_problem": MathematicalSystem(
        name="Three-Body Problem",
        domain="dynamics",
        dimension=18,  # 3 bodies x 6 coords
        n_conserved=10,  # E, P, L, center of mass
        n_degrees_of_freedom=9,
        symmetry_dimension=0,  # No continuous symmetry
    ),
    "lorenz_system": MathematicalSystem(
        name="Lorenz Attractor",
        domain="dynamics",
        dimension=3,
        n_conserved=0,  # Dissipative
        n_degrees_of_freedom=3,
        symmetry_dimension=0,
    ),

    # Geometric structures
    "schwarzschild_manifold": MathematicalSystem(
        name="Schwarzschild Geometry",
        domain="geometry",
        dimension=4,
        curvature=0.0,  # Ricci-flat vacuum
        euler_characteristic=2,
        symmetry_dimension=4,  # Static spherical
    ),
    "ads5_x_s5": MathematicalSystem(
        name="AdS5 x S5",
        domain="geometry",
        dimension=10,
        curvature=-1.0,  # AdS part
        symmetry_dimension=45,  # SO(4,2) x SO(6)
    ),
}


# =============================================================================
# Demonstration Function
# =============================================================================

def demonstrate_mathematics() -> Dict[str, Dict]:
    """
    Demonstrate theta calculations for example systems.

    Returns:
        Dictionary mapping system names to their analysis results.
    """
    results = {}

    for name, system in MATHEMATICAL_SYSTEMS.items():
        theta = compute_math_theta(system)

        # Get classifications
        if system.domain == "dynamics":
            classification = classify_integrability(
                system.n_conserved, system.n_degrees_of_freedom
            ).value
        elif system.domain == "geometry":
            classification = classify_geometric_regime(system.curvature).value
        else:
            classification = classify_topological_phase(
                system.fundamental_group_rank
            ).value

        results[name] = {
            "system": system.name,
            "domain": system.domain,
            "dimension": system.dimension,
            "theta": round(theta, 4),
            "classification": classification,
        }

    return results


if __name__ == "__main__":
    results = demonstrate_mathematics()
    print("\nAdvanced Mathematics Systems Theta Analysis")
    print("=" * 60)
    for name, data in results.items():
        print(f"\n{data['system']} ({data['domain']}, dim={data['dimension']}):")
        print(f"  theta = {data['theta']}")
        print(f"  Classification: {data['classification']}")
