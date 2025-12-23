r"""
Cognition Domain: Integrated Information, Neural Criticality, and Working Memory

This module implements theta as the consciousness/cognition parameter
using frameworks from neuroscience and information theory.

Key Insight: Cognitive systems exhibit phase transitions between:
- theta ~ 0: Unconscious processing (feedforward, modular)
- theta ~ 1: Conscious awareness (integrated, global)

Theta Maps To:
1. Integrated Information (IIT): Φ / Φ_max
2. Neural criticality: Power-law exponent τ / 1.5
3. Working memory: Items / capacity (Miller's 7±2)
4. Attention: Focused vs diffuse states

References (see BIBLIOGRAPHY.bib):
    \cite{Tononi2016} - Integrated information theory
    \cite{Beggs2003} - Neuronal avalanches
    \cite{Miller1956} - The magical number seven
    \cite{Dehaene2006} - Global neuronal workspace
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class ConsciousnessState(Enum):
    """States of consciousness based on theta."""
    UNCONSCIOUS = "unconscious"    # theta < 0.2
    SUBLIMINAL = "subliminal"      # 0.2 <= theta < 0.4
    PRECONSCIOUS = "preconscious"  # 0.4 <= theta < 0.6
    CONSCIOUS = "conscious"        # 0.6 <= theta < 0.8
    HIGHLY_AWARE = "highly_aware"  # theta >= 0.8


class BrainState(Enum):
    """Neural activity states."""
    SLEEP_DEEP = "deep_sleep"       # Slow-wave, high Φ
    SLEEP_REM = "rem_sleep"         # Dreaming
    DROWSY = "drowsy"               # Transition
    RELAXED = "relaxed"             # Alpha rhythm
    ALERT = "alert"                 # Beta rhythm
    FOCUSED = "focused"             # Gamma rhythm
    FLOW = "flow"                   # Optimal performance


@dataclass
class CognitiveSystem:
    """
    A cognitive system for theta analysis.

    Attributes:
        name: System identifier
        n_modules: Number of brain modules/regions
        integration: Information integration level Φ
        differentiation: Information differentiation level
        criticality_exponent: Power-law avalanche exponent
        working_memory_load: Current WM items
        attention_focus: Attention concentration [0, 1]
    """
    name: str
    n_modules: int
    integration: float  # Φ in IIT
    differentiation: float
    criticality_exponent: float
    working_memory_load: int
    attention_focus: float


# =============================================================================
# INTEGRATED INFORMATION THEORY (IIT)
# =============================================================================

def compute_phi_simple(
    connectivity_matrix: np.ndarray,
    states: np.ndarray
) -> float:
    r"""
    Compute simplified integrated information Φ.

    Φ measures how much a system is "more than the sum of its parts."
    It quantifies irreducible information integration.

    Full IIT computation is NP-hard, so this is simplified.

    Args:
        connectivity_matrix: N x N connection weights
        states: Current state of each element

    Returns:
        Integrated information Φ

    Reference: \cite{Tononi2016}
    """
    n = len(states)
    if n == 0:
        return 0.0

    # Simplified: use synergy measure
    # Φ ≈ I(system) - Σ I(parts)

    # Total mutual information (simplified as connectivity sum)
    I_whole = np.sum(np.abs(connectivity_matrix)) / (n * n)

    # Information in parts (diagonal + local)
    # This is a rough approximation
    I_parts = np.sum(np.diag(np.abs(connectivity_matrix))) / n

    # Integrated information
    phi = max(0, I_whole - I_parts)

    return phi


def compute_phi_ratio(phi: float, phi_max: float = 1.0) -> float:
    r"""
    Compute theta from integrated information.

    Theta = Φ / Φ_max

    Args:
        phi: Computed Φ value
        phi_max: Maximum possible Φ for this architecture

    Returns:
        theta in [0, 1]

    Reference: \cite{Tononi2016}
    """
    if phi_max <= 0:
        return 0.0
    return np.clip(phi / phi_max, 0.0, 1.0)


def consciousness_from_phi(phi: float) -> ConsciousnessState:
    r"""
    Classify consciousness level from Φ.

    Reference: \cite{Tononi2016}
    """
    if phi < 0.2:
        return ConsciousnessState.UNCONSCIOUS
    elif phi < 0.4:
        return ConsciousnessState.SUBLIMINAL
    elif phi < 0.6:
        return ConsciousnessState.PRECONSCIOUS
    elif phi < 0.8:
        return ConsciousnessState.CONSCIOUS
    else:
        return ConsciousnessState.HIGHLY_AWARE


# =============================================================================
# NEURAL CRITICALITY
# =============================================================================

def criticality_exponent(avalanche_sizes: List[int]) -> float:
    r"""
    Compute power-law exponent from neural avalanche sizes.

    At criticality: P(s) ~ s^(-τ) with τ ≈ 1.5

    This is the "edge of chaos" where information processing
    is optimal.

    Args:
        avalanche_sizes: List of avalanche sizes

    Returns:
        Power-law exponent τ

    Reference: \cite{Beggs2003}
    """
    if len(avalanche_sizes) < 10:
        return 0.0

    # Log-log regression for power law
    sizes = np.array([s for s in avalanche_sizes if s > 0])
    unique, counts = np.unique(sizes, return_counts=True)

    if len(unique) < 3:
        return 0.0

    # P(s) = N(s) / total
    probs = counts / len(sizes)

    # Fit log(P) = -τ * log(s) + const
    log_s = np.log(unique)
    log_p = np.log(probs)

    slope, _ = np.polyfit(log_s, log_p, 1)
    return -slope


def compute_criticality_theta(
    tau: float,
    tau_critical: float = 1.5,
    tau_range: float = 0.5
) -> float:
    r"""
    Compute theta from criticality exponent.

    At criticality (τ = 1.5), information processing is optimal.
    Deviations indicate subcritical or supercritical regimes.

    Args:
        tau: Measured power-law exponent
        tau_critical: Critical exponent (1.5 for mean-field)
        tau_range: Acceptable deviation range

    Returns:
        theta in [0, 1], with 1.0 at criticality

    Reference: \cite{Beggs2003}
    """
    # Theta is highest at tau = tau_critical
    deviation = abs(tau - tau_critical)

    if deviation >= tau_range:
        return 0.0

    theta = 1 - deviation / tau_range
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# WORKING MEMORY (Miller's Law)
# =============================================================================

def miller_capacity() -> Tuple[int, int, int]:
    r"""
    Return Miller's magical number: 7 ± 2.

    Working memory can hold about 7 (±2) chunks of information.

    Returns:
        (min, typical, max) capacity

    Reference: \cite{Miller1956}
    """
    return 5, 7, 9


def compute_working_memory_theta(
    items: int,
    capacity: int = 7
) -> float:
    r"""
    Compute theta from working memory load.

    Theta = items / capacity

    At capacity, cognitive load is maximum.
    Beyond capacity, performance degrades.

    Args:
        items: Current number of items in WM
        capacity: Individual's WM capacity (typically 7)

    Returns:
        theta in [0, 1]

    Reference: \cite{Miller1956}
    """
    if capacity <= 0:
        return 0.0

    theta = items / capacity
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ATTENTION AND FOCUS
# =============================================================================

def compute_attention_theta(
    focus_intensity: float,
    distractors: int = 0,
    max_distractors: int = 10
) -> float:
    """
    Compute theta from attention state.

    High focus = high theta (concentrated processing)
    Many distractors = low theta (diffuse processing)

    Args:
        focus_intensity: Self-reported focus [0, 1]
        distractors: Number of competing stimuli
        max_distractors: Maximum expected distractors

    Returns:
        theta in [0, 1]
    """
    # Distractor penalty
    distractor_factor = 1 - distractors / max_distractors
    distractor_factor = max(0, distractor_factor)

    theta = focus_intensity * distractor_factor
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# GLOBAL WORKSPACE THEORY
# =============================================================================

def compute_global_broadcast_theta(
    activated_modules: int,
    total_modules: int,
    broadcast_strength: float = 1.0
) -> float:
    r"""
    Compute theta from global workspace theory.

    Conscious processing involves "global broadcast" where
    information becomes available to multiple brain modules.

    Theta = (activated / total) * broadcast_strength

    Args:
        activated_modules: Modules receiving broadcast
        total_modules: Total number of modules
        broadcast_strength: Strength of global signal

    Returns:
        theta in [0, 1]

    Reference: \cite{Dehaene2006}
    """
    if total_modules <= 0:
        return 0.0

    coverage = activated_modules / total_modules
    theta = coverage * broadcast_strength
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED THETA CALCULATION
# =============================================================================

def compute_cognition_theta(system: CognitiveSystem) -> float:
    """
    Compute unified theta for cognitive system.

    Combines:
    - Integrated information (Φ)
    - Neural criticality
    - Working memory load
    - Attention focus

    Args:
        system: CognitiveSystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Component thetas
    theta_phi = compute_phi_ratio(system.integration)
    theta_criticality = compute_criticality_theta(system.criticality_exponent)
    theta_wm = compute_working_memory_theta(system.working_memory_load)
    theta_attention = system.attention_focus

    # Weighted combination
    # IIT and criticality are most fundamental
    theta = (
        0.35 * theta_phi +
        0.30 * theta_criticality +
        0.15 * theta_wm +
        0.20 * theta_attention
    )

    return np.clip(theta, 0.0, 1.0)


def classify_brain_state(theta: float) -> BrainState:
    """Classify brain state from theta."""
    if theta < 0.15:
        return BrainState.SLEEP_DEEP
    elif theta < 0.30:
        return BrainState.SLEEP_REM
    elif theta < 0.45:
        return BrainState.DROWSY
    elif theta < 0.60:
        return BrainState.RELAXED
    elif theta < 0.75:
        return BrainState.ALERT
    elif theta < 0.90:
        return BrainState.FOCUSED
    else:
        return BrainState.FLOW


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

COGNITIVE_SYSTEMS: Dict[str, CognitiveSystem] = {
    "deep_sleep": CognitiveSystem(
        name="Deep Sleep (N3)",
        n_modules=6,
        integration=0.8,  # High Φ but synchronized
        differentiation=0.1,
        criticality_exponent=2.0,  # Supercritical
        working_memory_load=0,
        attention_focus=0.0,
    ),
    "rem_sleep": CognitiveSystem(
        name="REM Sleep (Dreaming)",
        n_modules=6,
        integration=0.6,
        differentiation=0.7,  # Rich imagery
        criticality_exponent=1.6,
        working_memory_load=2,
        attention_focus=0.3,
    ),
    "drowsy": CognitiveSystem(
        name="Drowsy/Hypnagogia",
        n_modules=6,
        integration=0.4,
        differentiation=0.4,
        criticality_exponent=1.7,
        working_memory_load=3,
        attention_focus=0.3,
    ),
    "relaxed_awake": CognitiveSystem(
        name="Relaxed Wakefulness",
        n_modules=6,
        integration=0.5,
        differentiation=0.6,
        criticality_exponent=1.5,  # At criticality
        working_memory_load=4,
        attention_focus=0.5,
    ),
    "focused_work": CognitiveSystem(
        name="Focused Work",
        n_modules=6,
        integration=0.7,
        differentiation=0.7,
        criticality_exponent=1.4,
        working_memory_load=6,
        attention_focus=0.85,
    ),
    "flow_state": CognitiveSystem(
        name="Flow State",
        n_modules=6,
        integration=0.9,
        differentiation=0.8,
        criticality_exponent=1.5,  # At criticality
        working_memory_load=5,
        attention_focus=0.95,
    ),
    "meditation": CognitiveSystem(
        name="Deep Meditation",
        n_modules=6,
        integration=0.95,
        differentiation=0.3,  # Unified experience
        criticality_exponent=1.5,
        working_memory_load=1,
        attention_focus=0.98,
    ),
    "anesthesia": CognitiveSystem(
        name="General Anesthesia",
        n_modules=6,
        integration=0.1,  # Very low Φ
        differentiation=0.1,
        criticality_exponent=2.5,  # Supercritical
        working_memory_load=0,
        attention_focus=0.0,
    ),
}


def cognition_theta_summary():
    """Print theta analysis for example cognitive systems."""
    print("=" * 70)
    print("COGNITION THETA ANALYSIS (Consciousness & Criticality)")
    print("=" * 70)
    print()
    print(f"{'State':<25} {'Φ':>6} {'τ':>6} {'WM':>4} {'Attn':>6} {'θ':>8} {'Level':<15}")
    print("-" * 70)

    for name, system in COGNITIVE_SYSTEMS.items():
        theta = compute_cognition_theta(system)
        state = classify_brain_state(theta)
        print(f"{system.name:<25} "
              f"{system.integration:>6.2f} "
              f"{system.criticality_exponent:>6.2f} "
              f"{system.working_memory_load:>4} "
              f"{system.attention_focus:>6.2f} "
              f"{theta:>8.3f} "
              f"{state.value:<15}")

    print()
    print("Key: θ combines integrated information (Φ), criticality, WM, attention")
    print("     Flow state has highest θ (optimal consciousness)")


if __name__ == "__main__":
    cognition_theta_summary()
