r"""
Recursive Learning Domain: Meta-Cognition as Theta

This module implements theta as the meta-learning parameter
using frameworks from self-improvement and metacognition.

Key Insight: Learning systems exhibit phase transitions between:
- theta ~ 0: Static/object-level (fixed algorithm, no adaptation)
- theta ~ 1: Recursive/meta-level (self-improving, self-aware)

Theta Maps To:
1. Meta-Awareness: Levels of self-modeling
2. Self-Improvement: Performance gain per iteration
3. Recursion Depth: Effective depth of meta-reasoning
4. Feedback Loops: Fraction of closed feedback loops
5. Abstraction Climbing: Rate of moving up abstraction hierarchy

Recursive Regimes:
- OBJECT_LEVEL (theta < 0.25): Fixed algorithm, no meta-learning
- ADAPTIVE (0.25 <= theta < 0.5): Online learning, adapts to distribution
- META_LEVEL (0.5 <= theta < 0.75): Learns to learn, meta-gradients
- RECURSIVE (0.75 <= theta < 0.9): Self-modifying, recursive improvement
- OMEGA_LEVEL (theta >= 0.9): Full self-model, reflective architecture

Physical Analogy:
Recursive learning exhibits renormalization group flow. Each level of
meta-learning "integrates out" details from the level below, keeping
only the essential structure. Fixed points of the RG flow correspond
to stable learning strategies.

The Gödel-like self-reference in recursive systems parallels quantum
measurement: observing the learning process changes the learner,
creating an inherent complementarity between learning and observing.

References (see BIBLIOGRAPHY.bib):
    \cite{Schmidhuber1987} - Evolutionary Principles in Self-Referential Learning
    \cite{Finn2017} - Model-Agnostic Meta-Learning (MAML)
    \cite{Bengio2009} - Curriculum Learning
    \cite{Lake2017} - Building Machines That Learn and Think Like People
    \cite{Hofstadter1979} - Gödel, Escher, Bach: An Eternal Golden Braid
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class RecursionLevel(Enum):
    """Levels of recursive learning."""
    OBJECT_LEVEL = "object_level"      # theta < 0.25
    ADAPTIVE = "adaptive"              # 0.25 <= theta < 0.5
    META_LEVEL = "meta_level"          # 0.5 <= theta < 0.75
    RECURSIVE = "recursive"            # 0.75 <= theta < 0.9
    OMEGA_LEVEL = "omega_level"        # theta >= 0.9


class ImprovementType(Enum):
    """Types of self-improvement."""
    NONE = "none"                      # No improvement
    LINEAR = "linear"                  # Constant improvement
    LOGARITHMIC = "logarithmic"        # Diminishing returns
    EXPONENTIAL = "exponential"        # Accelerating improvement
    SUPEREXPONENTIAL = "superexponential"  # Singularity-like


class FeedbackType(Enum):
    """Types of feedback loops."""
    OPEN = "open"                      # No feedback
    DELAYED = "delayed"                # Feedback with latency
    IMMEDIATE = "immediate"            # Real-time feedback
    RECURSIVE = "recursive"            # Feedback feeds back


@dataclass
class RecursiveSystem:
    """
    A recursive learning system for theta analysis.

    Attributes:
        name: System identifier
        meta_levels: Number of meta-learning levels
        max_meta_levels: Maximum possible meta-levels
        improvement_rate: Performance improvement per iteration
        recursion_depth: Effective recursion depth
        max_safe_depth: Maximum safe recursion before divergence
        closed_loops: Number of closed feedback loops
        total_pathways: Total information pathways
        abstraction_rate: Rate of abstraction climbing
        has_self_model: Whether system has explicit self-model
        convergence_distance: Distance from fixed point
    """
    name: str
    meta_levels: int
    max_meta_levels: int = 5
    improvement_rate: float = 0.0
    recursion_depth: int = 0
    max_safe_depth: int = 10
    closed_loops: int = 0
    total_pathways: int = 1
    abstraction_rate: float = 0.0
    has_self_model: bool = False
    convergence_distance: float = float('inf')

    @property
    def meta_ratio(self) -> float:
        """Fraction of meta-levels utilized."""
        if self.max_meta_levels == 0:
            return 0.0
        return self.meta_levels / self.max_meta_levels

    @property
    def loop_closure(self) -> float:
        """Fraction of closed feedback loops."""
        if self.total_pathways == 0:
            return 0.0
        return self.closed_loops / self.total_pathways


# =============================================================================
# META-AWARENESS THETA
# =============================================================================

def compute_meta_awareness_theta(
    meta_levels: int,
    max_levels: int = 5
) -> float:
    r"""
    Compute theta from meta-awareness depth.

    Higher meta-levels = higher theta.
    Typical levels:
    - 0: Object-level (direct task)
    - 1: Meta-level (learning to learn)
    - 2: Meta-meta (learning how to learn to learn)
    - ω: Omega level (full self-reference)

    Args:
        meta_levels: Current meta-level count
        max_levels: Maximum meta-levels (typically 5)

    Returns:
        theta in [0, 1]

    Reference: \cite{Schmidhuber1987} - Self-referential learning
    """
    if max_levels <= 0:
        return 0.0

    # Logarithmic scaling (harder to add more meta-levels)
    normalized = np.log1p(meta_levels) / np.log1p(max_levels)
    return np.clip(normalized, 0.0, 1.0)


# =============================================================================
# SELF-IMPROVEMENT THETA
# =============================================================================

def compute_improvement_theta(
    improvement_rate: float,
    optimal_rate: float = 0.1
) -> float:
    r"""
    Compute theta from self-improvement rate.

    Improvement rate = performance gain per iteration.
    Positive rate = improving, negative = degrading.

    Args:
        improvement_rate: Fractional improvement per iteration
        optimal_rate: Target improvement rate (10% default)

    Returns:
        theta in [0, 1]

    Reference: \cite{Finn2017} - MAML improvement rates
    """
    if improvement_rate <= 0:
        return 0.0

    # Compare to optimal rate
    if improvement_rate >= optimal_rate:
        theta = 1.0
    else:
        theta = improvement_rate / optimal_rate

    return np.clip(theta, 0.0, 1.0)


def classify_improvement_type(rate: float) -> ImprovementType:
    """Classify improvement type from rate."""
    if rate <= 0:
        return ImprovementType.NONE
    elif rate < 0.01:
        return ImprovementType.LOGARITHMIC
    elif rate < 0.1:
        return ImprovementType.LINEAR
    elif rate < 1.0:
        return ImprovementType.EXPONENTIAL
    else:
        return ImprovementType.SUPEREXPONENTIAL


# =============================================================================
# RECURSION DEPTH THETA
# =============================================================================

def compute_recursion_theta(
    depth: int,
    max_safe_depth: int = 10
) -> float:
    r"""
    Compute theta from recursion depth.

    Deeper recursion = higher theta, but bounded by safety.
    Too deep can lead to divergence/infinite loops.

    Args:
        depth: Current effective recursion depth
        max_safe_depth: Maximum stable depth

    Returns:
        theta in [0, 1]
    """
    if max_safe_depth <= 0:
        return 0.0

    # Optimal is around 70% of max safe depth
    optimal_depth = 0.7 * max_safe_depth

    if depth <= optimal_depth:
        theta = depth / optimal_depth
    else:
        # Penalty for approaching unsafe depth
        excess = (depth - optimal_depth) / (max_safe_depth - optimal_depth)
        theta = 1 - 0.3 * excess

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# FEEDBACK LOOP THETA
# =============================================================================

def compute_feedback_theta(
    closed_loops: int,
    total_pathways: int
) -> float:
    r"""
    Compute theta from feedback loop closure.

    Closed loops enable recursive information flow.
    More closure = higher self-referential capacity.

    Args:
        closed_loops: Number of closed feedback loops
        total_pathways: Total information pathways

    Returns:
        theta in [0, 1]
    """
    if total_pathways == 0:
        return 0.0

    closure_ratio = closed_loops / total_pathways
    return np.clip(closure_ratio, 0.0, 1.0)


# =============================================================================
# ABSTRACTION CLIMBING THETA
# =============================================================================

def compute_abstraction_theta(
    abstraction_rate: float,
    max_rate: float = 0.5
) -> float:
    r"""
    Compute theta from abstraction climbing rate.

    Higher abstraction = more general learning.
    Rate measures how quickly system moves up abstraction hierarchy.

    Args:
        abstraction_rate: Rate of abstraction increase per iteration
        max_rate: Maximum expected rate (50% per iteration)

    Returns:
        theta in [0, 1]

    Reference: \cite{Bengio2009} - Curriculum learning abstraction
    """
    if abstraction_rate <= 0:
        return 0.0

    theta = min(1.0, abstraction_rate / max_rate)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# SELF-MODEL THETA
# =============================================================================

def compute_self_model_theta(
    has_explicit_model: bool,
    model_accuracy: float = 0.5,
    model_completeness: float = 0.5
) -> float:
    r"""
    Compute theta from self-model quality.

    A self-model represents the system's own behavior.
    Better self-models enable better meta-learning.

    Args:
        has_explicit_model: Whether explicit self-model exists
        model_accuracy: Accuracy of self-predictions [0, 1]
        model_completeness: Coverage of self-model [0, 1]

    Returns:
        theta in [0, 1]

    Reference: \cite{Hofstadter1979} - Self-reference and strange loops
    """
    if not has_explicit_model:
        return 0.1  # Implicit self-modeling exists

    # Combine accuracy and completeness
    theta = 0.1 + 0.45 * model_accuracy + 0.45 * model_completeness
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# CONVERGENCE THETA
# =============================================================================

def compute_convergence_theta(
    distance: float,
    characteristic_scale: float = 1.0
) -> float:
    r"""
    Compute theta from fixed-point convergence.

    Recursive processes converge to fixed points.
    Closer to fixed point = more stable/mature learning.

    Args:
        distance: Distance from fixed point
        characteristic_scale: Normalization scale

    Returns:
        theta in [0, 1]
    """
    if distance == float('inf'):
        return 0.0
    if distance <= 0:
        return 1.0

    # Exponential approach to fixed point
    theta = np.exp(-distance / characteristic_scale)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# GÖDEL SELF-REFERENCE THETA
# =============================================================================

def compute_godel_theta(
    can_represent_self: bool,
    can_reason_about_self: bool,
    has_fixed_point: bool
) -> float:
    r"""
    Compute theta from Gödelian self-reference capacity.

    Gödel's incompleteness shows limits of self-reference.
    Systems that can represent themselves face fundamental limits.

    Args:
        can_represent_self: System can encode itself
        can_reason_about_self: System can prove things about itself
        has_fixed_point: System has reached self-referential fixed point

    Returns:
        theta in [0, 1]

    Reference: \cite{Hofstadter1979} - Gödel, Escher, Bach
    """
    score = 0.0

    if can_represent_self:
        score += 0.3
    if can_reason_about_self:
        score += 0.4
    if has_fixed_point:
        score += 0.3

    return score


# =============================================================================
# CURRICULUM LEARNING THETA
# =============================================================================

def compute_curriculum_theta(
    task_ordering_quality: float,
    difficulty_progression: float,
    transfer_success: float
) -> float:
    r"""
    Compute theta from curriculum learning quality.

    Curriculum learning orders tasks from easy to hard.
    Good curricula enable faster learning.

    Args:
        task_ordering_quality: Quality of task ordering [0, 1]
        difficulty_progression: Smoothness of difficulty increase [0, 1]
        transfer_success: Success of knowledge transfer [0, 1]

    Returns:
        theta in [0, 1]

    Reference: \cite{Bengio2009} - Curriculum Learning
    """
    theta = (
        0.3 * task_ordering_quality +
        0.3 * difficulty_progression +
        0.4 * transfer_success
    )
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED RECURSIVE LEARNING THETA
# =============================================================================

def compute_recursive_theta(system: RecursiveSystem) -> float:
    """
    Compute unified theta for recursive learning system.

    Combines:
    - Meta-awareness (25%)
    - Self-improvement (25%)
    - Recursion depth (15%)
    - Feedback loops (15%)
    - Abstraction rate (10%)
    - Self-model (10%)

    Args:
        system: RecursiveSystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Meta-awareness
    theta_meta = compute_meta_awareness_theta(
        system.meta_levels,
        system.max_meta_levels
    )

    # Self-improvement
    theta_improve = compute_improvement_theta(system.improvement_rate)

    # Recursion depth
    theta_recurse = compute_recursion_theta(
        system.recursion_depth,
        system.max_safe_depth
    )

    # Feedback loops
    theta_feedback = compute_feedback_theta(
        system.closed_loops,
        system.total_pathways
    )

    # Abstraction climbing
    theta_abstract = compute_abstraction_theta(system.abstraction_rate)

    # Self-model bonus
    theta_self = 0.0
    if system.has_self_model:
        theta_self = 0.8

    # Convergence bonus (if close to fixed point)
    theta_conv = compute_convergence_theta(system.convergence_distance)

    # Weighted combination
    theta = (
        0.25 * theta_meta +
        0.25 * theta_improve +
        0.15 * theta_recurse +
        0.15 * theta_feedback +
        0.10 * theta_abstract +
        0.05 * theta_self +
        0.05 * theta_conv
    )

    return np.clip(theta, 0.0, 1.0)


def classify_recursion_level(theta: float) -> RecursionLevel:
    """Classify recursion level from theta."""
    if theta < 0.25:
        return RecursionLevel.OBJECT_LEVEL
    elif theta < 0.5:
        return RecursionLevel.ADAPTIVE
    elif theta < 0.75:
        return RecursionLevel.META_LEVEL
    elif theta < 0.9:
        return RecursionLevel.RECURSIVE
    else:
        return RecursionLevel.OMEGA_LEVEL


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

RECURSIVE_SYSTEMS: Dict[str, RecursiveSystem] = {
    "static_algorithm": RecursiveSystem(
        name="Static Algorithm",
        meta_levels=0,
        max_meta_levels=5,
        improvement_rate=0.0,
        recursion_depth=0,
        max_safe_depth=10,
        closed_loops=0,
        total_pathways=10,
        abstraction_rate=0.0,
        has_self_model=False,
        convergence_distance=float('inf'),
    ),
    "online_learner": RecursiveSystem(
        name="Online Learning (SGD)",
        meta_levels=0,
        max_meta_levels=5,
        improvement_rate=0.02,
        recursion_depth=1,
        max_safe_depth=10,
        closed_loops=1,
        total_pathways=5,
        abstraction_rate=0.01,
        has_self_model=False,
        convergence_distance=5.0,
    ),
    "curriculum_learner": RecursiveSystem(
        name="Curriculum Learner",
        meta_levels=1,
        max_meta_levels=5,
        improvement_rate=0.05,
        recursion_depth=2,
        max_safe_depth=10,
        closed_loops=2,
        total_pathways=5,
        abstraction_rate=0.1,
        has_self_model=False,
        convergence_distance=2.0,
    ),
    "maml_meta_learner": RecursiveSystem(
        name="MAML Meta-Learner",
        meta_levels=2,
        max_meta_levels=5,
        improvement_rate=0.1,
        recursion_depth=3,
        max_safe_depth=10,
        closed_loops=3,
        total_pathways=5,
        abstraction_rate=0.15,
        has_self_model=False,
        convergence_distance=1.0,
    ),
    "neural_architecture_search": RecursiveSystem(
        name="Neural Architecture Search",
        meta_levels=2,
        max_meta_levels=5,
        improvement_rate=0.08,
        recursion_depth=4,
        max_safe_depth=10,
        closed_loops=3,
        total_pathways=5,
        abstraction_rate=0.2,
        has_self_model=True,
        convergence_distance=0.8,
    ),
    "self_play_agent": RecursiveSystem(
        name="Self-Play Agent (AlphaZero)",
        meta_levels=3,
        max_meta_levels=5,
        improvement_rate=0.15,
        recursion_depth=5,
        max_safe_depth=10,
        closed_loops=4,
        total_pathways=5,
        abstraction_rate=0.25,
        has_self_model=True,
        convergence_distance=0.5,
    ),
    "recursive_self_improver": RecursiveSystem(
        name="Recursive Self-Improver",
        meta_levels=4,
        max_meta_levels=5,
        improvement_rate=0.2,
        recursion_depth=7,
        max_safe_depth=10,
        closed_loops=5,
        total_pathways=5,
        abstraction_rate=0.35,
        has_self_model=True,
        convergence_distance=0.2,
    ),
    "reflective_architecture": RecursiveSystem(
        name="Full Reflective Architecture",
        meta_levels=5,
        max_meta_levels=5,
        improvement_rate=0.25,
        recursion_depth=7,
        max_safe_depth=10,
        closed_loops=5,
        total_pathways=5,
        abstraction_rate=0.4,
        has_self_model=True,
        convergence_distance=0.05,
    ),
}


def recursive_theta_summary():
    """Print theta analysis for example recursive systems."""
    print("=" * 90)
    print("RECURSIVE LEARNING THETA ANALYSIS (Meta-Cognition)")
    print("=" * 90)
    print()
    print(f"{'System':<35} {'Meta':>5} {'Impr':>6} {'Depth':>6} "
          f"{'Loops':>6} {'Self':>5} {'θ':>8} {'Level':<12}")
    print("-" * 90)

    for name, system in RECURSIVE_SYSTEMS.items():
        theta = compute_recursive_theta(system)
        level = classify_recursion_level(theta)

        loop_frac = f"{system.closed_loops}/{system.total_pathways}"
        self_model = "Yes" if system.has_self_model else "No"

        print(f"{system.name:<35} "
              f"{system.meta_levels:>5} "
              f"{system.improvement_rate:>6.2f} "
              f"{system.recursion_depth:>6} "
              f"{loop_frac:>6} "
              f"{self_model:>5} "
              f"{theta:>8.3f} "
              f"{level.value:<12}")

    print()
    print("Key: θ combines meta-awareness, improvement rate, recursion, feedback loops")
    print("     Reflective architecture approaches θ ~ 0.9 (omega level)")


if __name__ == "__main__":
    recursive_theta_summary()
