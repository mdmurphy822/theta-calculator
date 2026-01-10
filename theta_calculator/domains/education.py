r"""
Education Domain: Learning, Memory, and Knowledge Integration

This module implements theta as the learning effectiveness parameter
for educational systems using cognitive science principles.

Key Insight: Learning systems exhibit phase transitions between:
- theta ~ 0: Rote memorization (isolated facts, fast decay)
- theta ~ 1: Deep understanding (integrated knowledge, slow decay)

Theta Maps To:
1. Power law of learning: Performance = A * t^(-β), theta ~ 1/β
2. Spaced repetition: Retention = exp(-t/τ), theta ~ τ/τ_max
3. Knowledge integration: Like integrated information (IIT)
4. Adaptive feedback: Stability margin in control theory

References (see BIBLIOGRAPHY.bib):
    \cite{Anderson1982} - Acquisition of Cognitive Skill
    \cite{Ebbinghaus1885} - Memory: A Contribution to Experimental Psychology
    \cite{Pimsleur1967} - Memory schedule for vocabulary learning
    \cite{Dunlosky2013} - Improving Students' Learning
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class LearningPhase(Enum):
    """Phases of learning based on theta."""
    ACQUISITION = "acquisition"      # theta < 0.3: Initial learning
    CONSOLIDATION = "consolidation"  # 0.3 <= theta < 0.6: Practice
    MASTERY = "mastery"              # 0.6 <= theta < 0.9: Fluency
    EXPERTISE = "expertise"          # theta >= 0.9: Automaticity


class KnowledgeType(Enum):
    """Types of knowledge for theta analysis."""
    DECLARATIVE = "declarative"  # Facts, concepts
    PROCEDURAL = "procedural"    # Skills, procedures
    EPISODIC = "episodic"        # Personal experiences
    SEMANTIC = "semantic"        # Abstract meaning
    METACOGNITIVE = "metacognitive"  # Learning about learning


@dataclass
class LearningSystem:
    """
    A learning system for theta analysis.

    Attributes:
        name: System identifier
        learner_count: Number of learners
        practice_time: Total practice time [hours]
        retention_rate: Fraction retained after delay
        transfer_rate: Fraction transferable to new contexts
        integration_level: How well knowledge is connected
        feedback_delay: Time between action and feedback [s]
    """
    name: str
    learner_count: int
    practice_time: float  # hours
    retention_rate: float  # [0, 1]
    transfer_rate: float  # [0, 1]
    integration_level: float  # [0, 1]
    feedback_delay: float  # seconds


@dataclass
class MemoryRetention:
    r"""
    Memory retention analysis result.

    Based on Ebbinghaus forgetting curve:
    R(t) = exp(-t/S) or R(t) = (1 + t/S)^(-1)

    Where S is the memory strength.

    Attributes:
        initial_strength: Memory strength at encoding
        current_retention: Current retention level
        decay_constant: Time constant τ [hours]
        optimal_review_time: Best time for review
        theta: Retention quality measure

    Reference: \cite{Ebbinghaus1885}
    """
    initial_strength: float
    current_retention: float
    decay_constant: float
    optimal_review_time: float
    theta: float


@dataclass
class LearningCurve:
    r"""
    Power law of learning analysis.

    Performance improves with practice:
    T = A * N^(-β)

    Where:
    - T = time to complete task
    - N = number of practice trials
    - A = initial performance
    - β = learning rate (typically 0.2-0.5)

    Attributes:
        initial_time: Time for first trial
        current_time: Current task time
        trials: Number of practice trials
        learning_rate: Power law exponent β
        asymptote: Performance floor
        theta: Normalized learning rate

    Reference: \cite{Anderson1982}
    """
    initial_time: float
    current_time: float
    trials: int
    learning_rate: float
    asymptote: float
    theta: float


# =============================================================================
# FORGETTING CURVE FUNCTIONS
# =============================================================================

def ebbinghaus_retention(
    time: float,
    strength: float = 1.0,
    decay_model: str = "exponential"
) -> float:
    r"""
    Compute memory retention using Ebbinghaus forgetting curve.

    Models:
    - exponential: R = exp(-t/S)
    - power: R = (1 + t/S)^(-1)
    - wickelgren: R = (1 + βt)^(-ψ)

    Args:
        time: Time since learning [hours]
        strength: Memory strength S
        decay_model: Type of decay function

    Returns:
        Retention level in [0, 1]

    Reference: \cite{Ebbinghaus1885}
    """
    if time < 0:
        raise ValueError("Time must be non-negative")
    if strength <= 0:
        raise ValueError("Strength must be positive")

    if decay_model == "exponential":
        return np.exp(-time / strength)
    elif decay_model == "power":
        return 1 / (1 + time / strength)
    elif decay_model == "wickelgren":
        beta = 0.1  # Decay rate
        psi = 0.5   # Power
        return (1 + beta * time) ** (-psi)
    else:
        raise ValueError(f"Unknown decay model: {decay_model}")


def optimal_review_time(
    current_strength: float,
    target_retention: float = 0.9
) -> float:
    r"""
    Compute optimal time for spaced repetition review.

    Review should happen when retention drops to target level.
    After review, strength increases.

    Args:
        current_strength: Current memory strength
        target_retention: Target retention before review (default 90%)

    Returns:
        Optimal review interval [hours]

    Reference: \cite{Pimsleur1967}
    """
    # From R = exp(-t/S), solve for t when R = target
    return -current_strength * np.log(target_retention)


def compute_retention_theta(
    time_since_learning: float,
    strength: float,
    max_strength: float = 168.0  # 1 week in hours
) -> MemoryRetention:
    r"""
    Compute theta for memory retention.

    Theta = S / S_max = τ / τ_max

    Long-term memory (high strength) = high theta
    Short-term memory (low strength) = low theta

    Args:
        time_since_learning: Hours since initial learning
        strength: Memory strength (decay constant) [hours]
        max_strength: Maximum expected strength [hours]

    Returns:
        MemoryRetention with theta analysis

    Reference: \cite{Ebbinghaus1885}
    """
    retention = ebbinghaus_retention(time_since_learning, strength)
    review_time = optimal_review_time(strength)

    # Theta: normalized memory strength
    theta = strength / max_strength
    theta = np.clip(theta, 0.0, 1.0)

    return MemoryRetention(
        initial_strength=strength,
        current_retention=retention,
        decay_constant=strength,
        optimal_review_time=review_time,
        theta=theta
    )


# =============================================================================
# POWER LAW OF LEARNING
# =============================================================================

def power_law_performance(
    trials: int,
    initial_time: float,
    learning_rate: float = 0.3,
    asymptote: float = 0.0
) -> float:
    r"""
    Compute performance using power law of learning.

    T = A * N^(-β) + asymptote

    Args:
        trials: Number of practice trials
        initial_time: Time for first trial
        learning_rate: Power law exponent β
        asymptote: Performance floor

    Returns:
        Task completion time

    Reference: \cite{Anderson1982}
    """
    if trials < 1:
        raise ValueError("Trials must be at least 1")
    return initial_time * trials ** (-learning_rate) + asymptote


def estimate_learning_rate(
    times: List[float],
    trials: Optional[List[int]] = None
) -> float:
    r"""
    Estimate learning rate from performance data.

    Fits T = A * N^(-β) using log-linear regression.

    Args:
        times: Task completion times
        trials: Trial numbers (default: 1, 2, 3, ...)

    Returns:
        Estimated learning rate β

    Reference: \cite{Anderson1982}
    """
    if trials is None:
        trials = list(range(1, len(times) + 1))

    if len(times) < 3:
        return 0.3  # Default

    # Log-log regression: log(T) = log(A) - β*log(N)
    log_N = np.log(trials)
    log_T = np.log(times)

    # Linear regression slope = -β
    slope, _ = np.polyfit(log_N, log_T, 1)
    return -slope


def compute_learning_theta(
    trials: int,
    initial_time: float,
    current_time: float,
    max_learning_rate: float = 0.5
) -> LearningCurve:
    r"""
    Compute theta from power law of learning.

    Theta = β / β_max

    Fast learning (high β) = high theta
    Slow learning (low β) = low theta

    Args:
        trials: Number of practice trials
        initial_time: First trial time
        current_time: Current trial time
        max_learning_rate: Maximum expected learning rate

    Returns:
        LearningCurve with theta analysis

    Reference: \cite{Anderson1982}
    """
    if trials < 2:
        return LearningCurve(
            initial_time=initial_time,
            current_time=current_time,
            trials=trials,
            learning_rate=0.0,
            asymptote=0.0,
            theta=0.0
        )

    # Estimate β from T1 and T_n
    # T_n = T_1 * n^(-β) => β = log(T_1/T_n) / log(n)
    ratio = initial_time / current_time
    if ratio <= 1:
        beta = 0.0
    else:
        beta = np.log(ratio) / np.log(trials)

    beta = max(0.0, min(beta, 1.0))

    # Theta: normalized learning rate
    theta = beta / max_learning_rate
    theta = np.clip(theta, 0.0, 1.0)

    return LearningCurve(
        initial_time=initial_time,
        current_time=current_time,
        trials=trials,
        learning_rate=beta,
        asymptote=0.0,
        theta=theta
    )


# =============================================================================
# KNOWLEDGE INTEGRATION
# =============================================================================

def compute_integration_theta(
    connections: int,
    max_connections: int,
    transfer_rate: float = 0.0
) -> float:
    r"""
    Compute theta for knowledge integration.

    Like integrated information (Φ) in consciousness theory,
    knowledge integration measures how well concepts connect.

    Theta = connections / max_connections + transfer_bonus

    Args:
        connections: Number of concept connections
        max_connections: Maximum possible connections
        transfer_rate: Ability to apply knowledge to new contexts

    Returns:
        Integration theta in [0, 1]

    Reference: \cite{Tononi2016}
    """
    if max_connections <= 0:
        return 0.0

    connection_ratio = connections / max_connections
    transfer_bonus = 0.2 * transfer_rate  # Transfer adds to integration

    theta = connection_ratio + transfer_bonus
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ADAPTIVE FEEDBACK / CONTROL THEORY
# =============================================================================

def compute_feedback_theta(
    feedback_delay: float,
    max_acceptable_delay: float = 10.0,
    feedback_specificity: float = 1.0
) -> float:
    r"""
    Compute theta for feedback quality.

    Based on control theory: faster, more specific feedback
    leads to better learning (higher theta).

    Args:
        feedback_delay: Time between action and feedback [seconds]
        max_acceptable_delay: Maximum useful delay [seconds]
        feedback_specificity: How targeted the feedback is [0, 1]

    Returns:
        Feedback theta in [0, 1]

    Reference: \cite{Hattie2007}
    """
    # Delay penalty
    delay_factor = np.exp(-feedback_delay / max_acceptable_delay)

    # Combined theta
    theta = delay_factor * feedback_specificity
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED THETA CALCULATION
# =============================================================================

def compute_education_theta(system: LearningSystem) -> float:
    """
    Compute unified theta for an educational system.

    Combines:
    - Retention rate (memory strength)
    - Transfer rate (generalization)
    - Integration level (connectedness)
    - Feedback quality (delay and specificity)

    Args:
        system: LearningSystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Component thetas
    theta_retention = system.retention_rate
    theta_transfer = system.transfer_rate
    theta_integration = system.integration_level
    theta_feedback = compute_feedback_theta(system.feedback_delay)

    # Weighted combination
    weights = {
        "retention": 0.3,
        "transfer": 0.3,
        "integration": 0.25,
        "feedback": 0.15
    }

    theta = (
        weights["retention"] * theta_retention +
        weights["transfer"] * theta_transfer +
        weights["integration"] * theta_integration +
        weights["feedback"] * theta_feedback
    )

    return np.clip(theta, 0.0, 1.0)


def classify_learning_phase(theta: float) -> LearningPhase:
    """Classify learning phase from theta."""
    if theta < 0.3:
        return LearningPhase.ACQUISITION
    elif theta < 0.6:
        return LearningPhase.CONSOLIDATION
    elif theta < 0.9:
        return LearningPhase.MASTERY
    else:
        return LearningPhase.EXPERTISE


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

EDUCATION_SYSTEMS: Dict[str, LearningSystem] = {
    "cramming": LearningSystem(
        name="Cramming (night before exam)",
        learner_count=1,
        practice_time=8.0,
        retention_rate=0.3,  # High initial, fast decay
        transfer_rate=0.1,   # Poor transfer
        integration_level=0.1,
        feedback_delay=0.0,  # Immediate during practice
    ),
    "spaced_repetition": LearningSystem(
        name="Spaced Repetition (Anki)",
        learner_count=1,
        practice_time=20.0,  # Distributed over weeks
        retention_rate=0.9,  # Excellent long-term
        transfer_rate=0.4,
        integration_level=0.5,
        feedback_delay=0.5,
    ),
    "lecture_passive": LearningSystem(
        name="Passive Lecture",
        learner_count=100,
        practice_time=40.0,  # Semester
        retention_rate=0.2,
        transfer_rate=0.15,
        integration_level=0.2,
        feedback_delay=604800.0,  # 1 week for exam feedback
    ),
    "project_based": LearningSystem(
        name="Project-Based Learning",
        learner_count=20,
        practice_time=60.0,
        retention_rate=0.75,
        transfer_rate=0.8,  # Excellent transfer
        integration_level=0.85,
        feedback_delay=60.0,  # Minutes
    ),
    "expert_tutoring": LearningSystem(
        name="Expert 1-on-1 Tutoring",
        learner_count=1,
        practice_time=30.0,
        retention_rate=0.85,
        transfer_rate=0.7,
        integration_level=0.9,
        feedback_delay=2.0,  # Immediate
    ),
    "language_immersion": LearningSystem(
        name="Language Immersion",
        learner_count=1,
        practice_time=500.0,  # Intensive months
        retention_rate=0.95,
        transfer_rate=0.9,
        integration_level=0.95,
        feedback_delay=1.0,  # Real-time conversation
    ),
    # Mastery-based learning (competency-based)
    "mastery_learning": LearningSystem(
        name="Mastery Learning (Bloom)",
        learner_count=25,
        practice_time=45.0,
        retention_rate=0.88,
        transfer_rate=0.65,
        integration_level=0.75,
        feedback_delay=5.0,  # Quick formative assessment
    ),
    # Peer instruction (active learning)
    "peer_instruction": LearningSystem(
        name="Peer Instruction (Mazur)",
        learner_count=150,
        practice_time=40.0,
        retention_rate=0.70,
        transfer_rate=0.55,
        integration_level=0.60,
        feedback_delay=30.0,  # ConcepTest feedback cycle
    ),
    # Retrieval practice (testing effect)
    "retrieval_practice": LearningSystem(
        name="Retrieval Practice",
        learner_count=1,
        practice_time=15.0,
        retention_rate=0.82,
        transfer_rate=0.50,
        integration_level=0.45,
        feedback_delay=1.0,  # Immediate flashcard feedback
    ),
}


def education_theta_summary():
    """Print theta analysis for example education systems."""
    print("=" * 70)
    print("EDUCATION THETA ANALYSIS (Learning Effectiveness)")
    print("=" * 70)
    print()
    print(f"{'System':<30} {'θ':>8} {'Ret':>6} {'Trf':>6} {'Int':>6} {'Phase':<15}")
    print("-" * 70)

    for name, system in EDUCATION_SYSTEMS.items():
        theta = compute_education_theta(system)
        phase = classify_learning_phase(theta)
        print(f"{system.name:<30} {theta:>8.3f} "
              f"{system.retention_rate:>6.2f} "
              f"{system.transfer_rate:>6.2f} "
              f"{system.integration_level:>6.2f} "
              f"{phase.value:<15}")

    print()
    print("Key: High θ = deep learning, Low θ = surface learning")


if __name__ == "__main__":
    education_theta_summary()
