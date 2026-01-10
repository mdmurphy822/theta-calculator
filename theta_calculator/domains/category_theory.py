r"""
Category Theory Domain: Abstract Structure as Theta

This module implements theta as the abstraction level parameter
using frameworks from category theory and mathematical logic.

Key Insight: Mathematical structures exhibit transitions between:
- theta ~ 0: Concrete/set-theoretic (elements, functions)
- theta ~ 1: Abstract/categorical (universal properties, functors)

Theta Maps To:
1. Abstraction Level: Set -> Cat -> 2-Cat -> ∞-Cat
2. Functoriality: Degree of structure preservation
3. Naturality: Coherence of transformations
4. Universality: Distance from universal constructions
5. Coherence: Satisfaction of coherence conditions

Abstraction Regimes:
- SET_THEORETIC (theta < 0.2): Concrete sets and functions
- ALGEBRAIC (0.2 <= theta < 0.4): Groups, rings, algebraic structures
- CATEGORICAL (0.4 <= theta < 0.6): Categories, functors, natural transformations
- HIGHER (0.6 <= theta < 0.8): 2-categories, bicategories, n-categories
- HOMOTOPICAL (theta >= 0.8): ∞-categories, derived categories, motives

Physical Analogy:
Category theory abstracts structure in a way analogous to the
renormalization group in physics. Higher categories "integrate out"
lower-level details, keeping only the essential structural relationships.
This mirrors how effective field theories abstract away high-energy
degrees of freedom.

The Yoneda lemma states that objects are determined by their
relationships (morphisms), analogous to how quantum particles are
defined by their interactions rather than intrinsic properties.

References (see BIBLIOGRAPHY.bib):
    \cite{MacLane1998} - Categories for the Working Mathematician
    \cite{Awodey2010} - Category Theory
    \cite{Baez2010} - Physics, Topology, Logic and Computation: A Rosetta Stone
    \cite{Riehl2017} - Category Theory in Context
    \cite{Lurie2009} - Higher Topos Theory
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class AbstractionLevel(Enum):
    """Levels of categorical abstraction."""
    SET = "set"                      # theta ~ 0.1
    MONOID = "monoid"                # theta ~ 0.2
    GROUP = "group"                  # theta ~ 0.25
    CATEGORY = "category"            # theta ~ 0.4
    FUNCTOR = "functor"              # theta ~ 0.5
    NATURAL = "natural"              # theta ~ 0.6
    TWO_CATEGORY = "2-category"      # theta ~ 0.7
    BICATEGORY = "bicategory"        # theta ~ 0.75
    INFINITY_CATEGORY = "infinity"   # theta ~ 0.9
    TOPOS = "topos"                  # theta ~ 0.95


class MorphismType(Enum):
    """Types of morphisms/arrows."""
    FUNCTION = "function"
    HOMOMORPHISM = "homomorphism"
    FUNCTOR = "functor"
    NATURAL_TRANSFORMATION = "natural_transformation"
    ADJUNCTION = "adjunction"
    EQUIVALENCE = "equivalence"


class StructureType(Enum):
    """Types of categorical structures."""
    DISCRETE = "discrete"            # No non-identity morphisms
    PREORDER = "preorder"            # At most one morphism between objects
    GROUPOID = "groupoid"            # All morphisms invertible
    MONOIDAL = "monoidal"            # Tensor product structure
    ENRICHED = "enriched"            # Hom-sets are objects in V
    FIBERED = "fibered"              # Fibration structure


@dataclass
class CategoricalSystem:
    """
    A categorical system for theta analysis.

    Attributes:
        name: System identifier
        abstraction_level: Level of categorical abstraction
        n_objects: Number of objects (0-cells)
        n_morphisms: Number of morphisms (1-cells)
        n_two_morphisms: Number of 2-morphisms (for 2-categories)
        functoriality: Degree of structure preservation [0, 1]
        naturality_satisfaction: Fraction of natural conditions satisfied
        universal_distance: Distance from universal construction [0, ∞)
        coherence_satisfied: Number of coherence conditions satisfied
        coherence_total: Total coherence conditions
        has_limits: Whether category has (co)limits
        has_adjunctions: Whether key adjunctions exist
    """
    name: str
    abstraction_level: AbstractionLevel
    n_objects: int
    n_morphisms: int
    n_two_morphisms: int = 0
    functoriality: float = 0.0
    naturality_satisfaction: float = 0.0
    universal_distance: float = float('inf')
    coherence_satisfied: int = 0
    coherence_total: int = 0
    has_limits: bool = False
    has_adjunctions: bool = False

    @property
    def morphism_density(self) -> float:
        """Average morphisms per object pair."""
        if self.n_objects <= 1:
            return 0.0
        max_morphisms = self.n_objects * (self.n_objects - 1)
        return self.n_morphisms / max_morphisms if max_morphisms > 0 else 0.0


# =============================================================================
# ABSTRACTION LEVEL THETA
# =============================================================================

def compute_abstraction_theta(level: AbstractionLevel) -> float:
    r"""
    Compute theta from abstraction level.

    Higher abstraction = higher theta.

    Args:
        level: Categorical abstraction level

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Categorical hierarchy
    """
    level_thetas = {
        AbstractionLevel.SET: 0.1,
        AbstractionLevel.MONOID: 0.2,
        AbstractionLevel.GROUP: 0.25,
        AbstractionLevel.CATEGORY: 0.4,
        AbstractionLevel.FUNCTOR: 0.5,
        AbstractionLevel.NATURAL: 0.6,
        AbstractionLevel.TWO_CATEGORY: 0.7,
        AbstractionLevel.BICATEGORY: 0.75,
        AbstractionLevel.INFINITY_CATEGORY: 0.9,
        AbstractionLevel.TOPOS: 0.95,
    }
    return level_thetas.get(level, 0.5)


# =============================================================================
# FUNCTORIALITY THETA
# =============================================================================

def compute_functoriality_theta(
    composition_preserved: int,
    composition_total: int,
    identity_preserved: bool = True
) -> float:
    r"""
    Compute theta from functoriality (structure preservation).

    A functor F preserves:
    - Composition: F(g ∘ f) = F(g) ∘ F(f)
    - Identity: F(id_A) = id_{F(A)}

    Args:
        composition_preserved: Number of compositions preserved
        composition_total: Total composable pairs
        identity_preserved: Whether identities are preserved

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Functors
    """
    if composition_total == 0:
        comp_theta = 1.0 if identity_preserved else 0.5
    else:
        comp_theta = composition_preserved / composition_total

    # Identity is worth 20% of the score
    id_theta = 1.0 if identity_preserved else 0.0

    theta = 0.8 * comp_theta + 0.2 * id_theta
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# NATURALITY THETA
# =============================================================================

def compute_naturality_theta(
    commuting_squares: int,
    total_squares: int
) -> float:
    r"""
    Compute theta from naturality satisfaction.

    A natural transformation η: F → G satisfies:
    For all f: A → B, the square commutes:
    G(f) ∘ η_A = η_B ∘ F(f)

    Args:
        commuting_squares: Number of commuting naturality squares
        total_squares: Total squares that should commute

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Natural Transformations
    """
    if total_squares == 0:
        return 1.0  # Vacuously natural

    theta = commuting_squares / total_squares
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIVERSALITY THETA
# =============================================================================

def compute_universality_theta(
    distance_from_universal: float,
    characteristic_scale: float = 1.0
) -> float:
    r"""
    Compute theta from distance to universal construction.

    Universal constructions (limits, colimits, adjunctions) represent
    "optimal" solutions. Distance measures deviation from optimality.

    Distance = 0: Universal property satisfied exactly
    Distance = ∞: No universal property

    Args:
        distance_from_universal: Deviation from universal construction
        characteristic_scale: Scale for normalization

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Universal Properties
    """
    if distance_from_universal == float('inf'):
        return 0.0
    if distance_from_universal <= 0:
        return 1.0

    # Exponential decay
    theta = np.exp(-distance_from_universal / characteristic_scale)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# COHERENCE THETA
# =============================================================================

def compute_coherence_theta(
    satisfied: int,
    total: int
) -> float:
    r"""
    Compute theta from coherence condition satisfaction.

    Higher categories require coherence conditions:
    - Pentagon identity (monoidal)
    - Triangle identity (monoidal)
    - Coherence for bicategories, tricategories, etc.

    Mac Lane's coherence theorem: All diagrams commute if
    basic ones do.

    Args:
        satisfied: Coherence conditions satisfied
        total: Total coherence conditions required

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Coherence Theorems
    """
    if total == 0:
        return 1.0  # No coherence conditions (discrete category)

    # Coherence is often all-or-nothing, but partial counts
    theta = satisfied / total
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# YONEDA THETA
# =============================================================================

def compute_yoneda_theta(
    representable_fraction: float,
    embedding_faithful: bool = True
) -> float:
    r"""
    Compute theta from Yoneda lemma applicability.

    The Yoneda lemma: Nat(Hom(A, -), F) ≅ F(A)
    The Yoneda embedding: y: C → Set^{C^op} is fully faithful

    Higher theta when:
    - More objects are representable
    - Yoneda embedding is well-behaved

    Args:
        representable_fraction: Fraction of functors that are representable
        embedding_faithful: Whether Yoneda embedding is faithful

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Yoneda Lemma
    """
    base_theta = representable_fraction

    # Bonus for faithful embedding
    if embedding_faithful:
        base_theta = 0.3 + 0.7 * base_theta
    else:
        base_theta = 0.7 * base_theta

    return np.clip(base_theta, 0.0, 1.0)


# =============================================================================
# ADJUNCTION THETA
# =============================================================================

def compute_adjunction_theta(
    unit_natural: bool,
    counit_natural: bool,
    triangle_left: bool,
    triangle_right: bool
) -> float:
    r"""
    Compute theta from adjunction quality.

    An adjunction F ⊣ G consists of:
    - Unit η: 1 → GF (natural transformation)
    - Counit ε: FG → 1 (natural transformation)
    - Triangle identities:
      (εF) ∘ (Fη) = 1_F
      (Gε) ∘ (ηG) = 1_G

    Args:
        unit_natural: Whether unit is natural
        counit_natural: Whether counit is natural
        triangle_left: Left triangle identity holds
        triangle_right: Right triangle identity holds

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Adjunctions
    """
    conditions = [unit_natural, counit_natural, triangle_left, triangle_right]
    theta = sum(1 for c in conditions if c) / len(conditions)
    return theta


# =============================================================================
# ENRICHED CATEGORY THETA
# =============================================================================

def compute_enriched_theta(
    base_category_level: AbstractionLevel,
    hom_objects_defined: bool = True,
    composition_enriched: bool = True
) -> float:
    r"""
    Compute theta from enriched category structure.

    V-enriched categories have Hom-objects in V rather than sets.
    Examples:
    - Ab-enriched = preadditive categories
    - Cat-enriched = 2-categories
    - sSet-enriched = ∞-categories

    Args:
        base_category_level: Abstraction level of base V
        hom_objects_defined: Whether hom-objects are well-defined
        composition_enriched: Whether composition respects V-structure

    Returns:
        theta in [0, 1]

    Reference: \cite{Riehl2017} - Enriched Categories
    """
    base_theta = compute_abstraction_theta(base_category_level)

    # Boost for proper enrichment
    if hom_objects_defined and composition_enriched:
        theta = 0.2 + 0.8 * base_theta
    elif hom_objects_defined or composition_enriched:
        theta = 0.1 + 0.7 * base_theta
    else:
        theta = 0.5 * base_theta

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# TOPOS THETA
# =============================================================================

def compute_topos_theta(
    has_finite_limits: bool,
    has_exponentials: bool,
    has_subobject_classifier: bool,
    is_grothendieck: bool = False
) -> float:
    r"""
    Compute theta from topos-theoretic properties.

    An elementary topos has:
    - Finite limits
    - Exponentials (internal hom)
    - Subobject classifier Ω

    Grothendieck topoi are sheaf categories.

    Args:
        has_finite_limits: Has all finite limits
        has_exponentials: Has exponential objects
        has_subobject_classifier: Has Ω with characteristic morphisms
        is_grothendieck: Is a Grothendieck topos (sheaves)

    Returns:
        theta in [0, 1]

    Reference: \cite{MacLane1998} - Topoi
    """
    score = 0.0

    if has_finite_limits:
        score += 0.25
    if has_exponentials:
        score += 0.25
    if has_subobject_classifier:
        score += 0.30

    # Grothendieck is the "gold standard"
    if is_grothendieck:
        score += 0.20

    return np.clip(score, 0.0, 1.0)


# =============================================================================
# UNIFIED CATEGORY THEORY THETA
# =============================================================================

def compute_category_theta(system: CategoricalSystem) -> float:
    """
    Compute unified theta for categorical system.

    Combines:
    - Abstraction level (25%)
    - Functoriality (20%)
    - Naturality (20%)
    - Universality (15%)
    - Coherence (10%)
    - Structure bonuses (10%)

    Args:
        system: CategoricalSystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Abstraction level
    theta_level = compute_abstraction_theta(system.abstraction_level)

    # Functoriality
    theta_func = system.functoriality

    # Naturality
    theta_nat = system.naturality_satisfaction

    # Universality
    theta_univ = compute_universality_theta(system.universal_distance)

    # Coherence
    if system.coherence_total > 0:
        theta_coh = compute_coherence_theta(
            system.coherence_satisfied,
            system.coherence_total
        )
    else:
        theta_coh = 1.0

    # Structure bonuses
    structure_bonus = 0.0
    if system.has_limits:
        structure_bonus += 0.05
    if system.has_adjunctions:
        structure_bonus += 0.05

    # Weighted combination
    theta = (
        0.25 * theta_level +
        0.20 * theta_func +
        0.20 * theta_nat +
        0.15 * theta_univ +
        0.10 * theta_coh +
        structure_bonus
    )

    return np.clip(theta, 0.0, 1.0)


def classify_abstraction_regime(theta: float) -> str:
    """Classify abstraction regime from theta."""
    if theta < 0.2:
        return "set_theoretic"
    elif theta < 0.4:
        return "algebraic"
    elif theta < 0.6:
        return "categorical"
    elif theta < 0.8:
        return "higher"
    else:
        return "homotopical"


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

CATEGORY_SYSTEMS: Dict[str, CategoricalSystem] = {
    "set_function": CategoricalSystem(
        name="Function between Sets",
        abstraction_level=AbstractionLevel.SET,
        n_objects=2,
        n_morphisms=1,
        functoriality=0.1,
        naturality_satisfaction=0.0,
        universal_distance=float('inf'),
        coherence_satisfied=0,
        coherence_total=0,
        has_limits=False,
        has_adjunctions=False,
    ),
    "group_homomorphism": CategoricalSystem(
        name="Group Homomorphism",
        abstraction_level=AbstractionLevel.GROUP,
        n_objects=2,
        n_morphisms=1,
        functoriality=0.8,
        naturality_satisfaction=0.0,
        universal_distance=1.0,
        coherence_satisfied=0,
        coherence_total=0,
        has_limits=True,
        has_adjunctions=False,
    ),
    "small_category": CategoricalSystem(
        name="Small Category",
        abstraction_level=AbstractionLevel.CATEGORY,
        n_objects=10,
        n_morphisms=50,
        functoriality=1.0,
        naturality_satisfaction=0.5,
        universal_distance=0.5,
        coherence_satisfied=0,
        coherence_total=0,
        has_limits=True,
        has_adjunctions=True,
    ),
    "functor_category": CategoricalSystem(
        name="Functor between Categories",
        abstraction_level=AbstractionLevel.FUNCTOR,
        n_objects=2,
        n_morphisms=1,
        functoriality=1.0,
        naturality_satisfaction=0.8,
        universal_distance=0.2,
        coherence_satisfied=0,
        coherence_total=0,
        has_limits=True,
        has_adjunctions=True,
    ),
    "natural_transformation": CategoricalSystem(
        name="Natural Transformation",
        abstraction_level=AbstractionLevel.NATURAL,
        n_objects=2,
        n_morphisms=1,
        n_two_morphisms=1,
        functoriality=1.0,
        naturality_satisfaction=1.0,
        universal_distance=0.1,
        coherence_satisfied=0,
        coherence_total=0,
        has_limits=True,
        has_adjunctions=True,
    ),
    "adjoint_pair": CategoricalSystem(
        name="Adjoint Functor Pair",
        abstraction_level=AbstractionLevel.FUNCTOR,
        n_objects=2,
        n_morphisms=2,
        n_two_morphisms=2,
        functoriality=1.0,
        naturality_satisfaction=1.0,
        universal_distance=0.0,  # Universal property satisfied
        coherence_satisfied=2,
        coherence_total=2,  # Triangle identities
        has_limits=True,
        has_adjunctions=True,
    ),
    "monad": CategoricalSystem(
        name="Monad (T, η, μ)",
        abstraction_level=AbstractionLevel.NATURAL,
        n_objects=1,
        n_morphisms=1,  # T: C → C
        n_two_morphisms=2,  # η, μ
        functoriality=1.0,
        naturality_satisfaction=1.0,
        universal_distance=0.0,
        coherence_satisfied=2,
        coherence_total=2,  # Associativity, unit laws
        has_limits=True,
        has_adjunctions=True,
    ),
    "bicategory": CategoricalSystem(
        name="Bicategory",
        abstraction_level=AbstractionLevel.BICATEGORY,
        n_objects=5,
        n_morphisms=20,
        n_two_morphisms=50,
        functoriality=1.0,
        naturality_satisfaction=0.95,
        universal_distance=0.1,
        coherence_satisfied=4,
        coherence_total=5,  # Pentagon, triangle, etc.
        has_limits=True,
        has_adjunctions=True,
    ),
    "elementary_topos": CategoricalSystem(
        name="Elementary Topos",
        abstraction_level=AbstractionLevel.TOPOS,
        n_objects=100,
        n_morphisms=1000,
        n_two_morphisms=0,
        functoriality=1.0,
        naturality_satisfaction=1.0,
        universal_distance=0.0,
        coherence_satisfied=3,
        coherence_total=3,
        has_limits=True,
        has_adjunctions=True,
    ),
    "infinity_category": CategoricalSystem(
        name="∞-Category (Quasi-category)",
        abstraction_level=AbstractionLevel.INFINITY_CATEGORY,
        n_objects=10,
        n_morphisms=100,
        n_two_morphisms=1000,
        functoriality=1.0,
        naturality_satisfaction=1.0,
        universal_distance=0.0,
        coherence_satisfied=10,
        coherence_total=10,  # All higher coherences
        has_limits=True,
        has_adjunctions=True,
    ),
}


def category_theta_summary():
    """Print theta analysis for example categorical systems."""
    print("=" * 85)
    print("CATEGORY THEORY THETA ANALYSIS (Abstract Structure)")
    print("=" * 85)
    print()
    print(f"{'System':<30} {'Level':<12} {'Func':>6} {'Nat':>6} "
          f"{'Univ':>6} {'Coh':>6} {'θ':>8} {'Regime':<12}")
    print("-" * 85)

    for name, system in CATEGORY_SYSTEMS.items():
        theta = compute_category_theta(system)
        regime = classify_abstraction_regime(theta)
        level = system.abstraction_level.value[:10]

        univ_theta = compute_universality_theta(system.universal_distance)
        coh_theta = (system.coherence_satisfied / system.coherence_total
                     if system.coherence_total > 0 else 1.0)

        print(f"{system.name:<30} "
              f"{level:<12} "
              f"{system.functoriality:>6.2f} "
              f"{system.naturality_satisfaction:>6.2f} "
              f"{univ_theta:>6.2f} "
              f"{coh_theta:>6.2f} "
              f"{theta:>8.3f} "
              f"{regime:<12}")

    print()
    print("Key: θ measures abstraction level & structural coherence")
    print("     Topos/∞-category represents highest abstraction (θ ~ 0.9+)")


if __name__ == "__main__":
    category_theta_summary()
