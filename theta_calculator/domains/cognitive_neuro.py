r"""
Cognitive Neuroscience Domain: Attention, Metacognition, and Working Memory

This module implements theta as the cognitive control interpolation parameter
for neural systems involved in attention, metacognition, and executive function.

## Mapping Definition

This domain maps neurocognitive systems to theta via resource and efficiency metrics:

**Inputs (Physical Analogs):**
- attention_resources -> Available cognitive resources [0, 1]
- attention_demand -> Task demand on attention [0, 1]
- d_prime -> Signal detection sensitivity
- meta_d_prime -> Metacognitive sensitivity
- wm_items -> Working memory load (items)
- wm_capacity -> Working memory capacity (items)
- prefrontal_activation -> PFC activity level [0, 1]
- recurrent_connectivity -> Feedback connection strength [0, 1]

**Theta Mapping:**
theta_attention = resources / demand
theta_metacog = meta-d' / d'
theta_wm = 1 - |items - optimal| / capacity
theta_exec = sqrt(PFC * recurrent)

**Interpretation:**
- theta -> 0: Automatic/feedforward processing (inattentional)
- theta -> 1: Controlled/recurrent processing (metacognitive access)
- 0.3 < theta < 0.7: Adaptive processing (flexible control)

**Key Feature:** Cognitive systems operate near criticality, balancing
efficiency with flexibility through theta-modulated control.

**Important:** This is a COGNITIVE SCORE based on neural mechanisms.

References (see BIBLIOGRAPHY.bib):
    \cite{Dehaene2006} - Global Neuronal Workspace theory
    \cite{Tononi2016} - Integrated Information Theory
    \cite{Miller1956} - Working memory capacity 7+/-2
    \cite{Sejnowski2025} - Dynamical mechanisms for working memory
    \cite{Staub2025} - Temporal coherence in neural systems
    \cite{Kalburge2025} - Functional information bottleneck
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class AttentionState(Enum):
    """Classification of attention states based on theta."""
    INATTENTIONAL = "inattentional"        # theta < 0.2: Stimuli not processed
    DIVIDED = "divided"                     # 0.2 <= theta < 0.5: Split attention
    SELECTIVE = "selective"                 # 0.5 <= theta < 0.8: Focused but flexible
    FOCUSED = "focused"                     # theta >= 0.8: Maximal concentration


class MetacognitiveLevel(Enum):
    """Levels of metacognitive awareness."""
    ABSENT = "no_metacognition"             # meta-d'/d' ~ 0
    IMPLICIT = "implicit_monitoring"        # 0 < meta-d'/d' < 0.5
    EXPLICIT = "explicit_access"            # 0.5 <= meta-d'/d' < 1.0
    REFLECTIVE = "reflective_control"       # meta-d'/d' >= 1.0 (optimal)


class WorkingMemoryPhase(Enum):
    """Working memory load phases."""
    UNDERLOAD = "underload"                 # Few items, spare capacity
    OPTIMAL = "optimal"                     # Near Cowan's 4, efficient
    NEAR_CAPACITY = "near_capacity"         # Approaching limit
    OVERLOAD = "overload"                   # Exceeds capacity, errors increase


class ExecutiveMode(Enum):
    """Executive function operating modes."""
    AUTOMATIC = "automatic"                 # Fast, stimulus-driven
    ROUTINE = "routine"                     # Practiced, efficient
    ADAPTIVE = "adaptive"                   # Flexible, learning
    DELIBERATE = "deliberate"               # Slow, effortful control


class ConsciousnessLevel(Enum):
    """Levels of conscious access (Global Workspace)."""
    SUBLIMINAL = "subliminal"               # Below threshold
    PRECONSCIOUS = "preconscious"           # Available but not accessed
    CONSCIOUS = "conscious"                 # Global broadcast
    INTROSPECTIVE = "introspective"         # Metacognitive reflection


@dataclass
class NeuroCognitiveSystem:
    """
    A neurocognitive system for theta analysis.

    Represents a cognitive state characterized by attention, metacognition,
    working memory, and executive control parameters.

    Attributes:
        name: System/state identifier
        attention_resources: Available attention resources [0, 1]
        attention_demand: Task demand on attention [0, 1]
        d_prime: Signal detection sensitivity (d')
        meta_d_prime: Metacognitive sensitivity (meta-d')
        wm_items: Number of items in working memory
        wm_capacity: Working memory capacity (typically 4-7)
        prefrontal_activation: Prefrontal cortex activity [0, 1]
        recurrent_connectivity: Feedback loop strength [0, 1]
        arousal: Physiological arousal level [0, 1]
        cognitive_load: Intrinsic + extraneous load [0, 1]
    """
    name: str
    attention_resources: float
    attention_demand: float
    d_prime: float
    meta_d_prime: float
    wm_items: int
    wm_capacity: int
    prefrontal_activation: float
    recurrent_connectivity: float
    arousal: float = 0.5
    cognitive_load: float = 0.5

    @property
    def metacognitive_efficiency(self) -> float:
        """Ratio of meta-d' to d' (metacognitive efficiency)."""
        if self.d_prime <= 0:
            return 0.0
        return self.meta_d_prime / self.d_prime

    @property
    def attention_ratio(self) -> float:
        """Ratio of resources to demand."""
        if self.attention_demand <= 0:
            return 1.0
        return min(self.attention_resources / self.attention_demand, 1.0)

    @property
    def wm_load_ratio(self) -> float:
        """Working memory utilization ratio."""
        if self.wm_capacity <= 0:
            return 1.0
        return self.wm_items / self.wm_capacity


# =============================================================================
# NEUROPHYSIOLOGICAL CONSTANTS
# =============================================================================

# Cowan's limit: working memory capacity
COWAN_LIMIT = 4

# Miller's number: 7 +/- 2
MILLER_LIMIT = 7

# Yerkes-Dodson optimal arousal
OPTIMAL_AROUSAL = 0.6

# Global Workspace ignition threshold
GW_THRESHOLD = 0.3


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_attention_theta(
    resources: float,
    demand: float
) -> float:
    r"""
    Compute theta from attention resource allocation.

    theta = resources / demand (capped at 1)

    High resources relative to demand: theta -> 1 (focused)
    Low resources relative to demand: theta -> 0 (inattentional)

    Args:
        resources: Available attention resources [0, 1]
        demand: Task attention demand [0, 1]

    Returns:
        theta in [0, 1]
    """
    if demand <= 0:
        return 1.0
    theta = resources / demand
    return np.clip(theta, 0.0, 1.0)


def compute_metacognition_theta(
    meta_d: float,
    d_prime: float
) -> float:
    r"""
    Compute theta from metacognitive efficiency.

    theta = meta-d' / d'

    Perfect metacognition: meta-d' = d' -> theta = 1
    No metacognition: meta-d' = 0 -> theta = 0
    Super-metacognition: meta-d' > d' -> theta > 1 (capped)

    Args:
        meta_d: Metacognitive sensitivity (meta-d')
        d_prime: Signal detection sensitivity (d')

    Returns:
        theta in [0, 1]

    Reference: \cite{Dehaene2006}
    """
    if d_prime <= 0:
        return 0.0
    theta = meta_d / d_prime
    return np.clip(theta, 0.0, 1.0)


def compute_wm_theta(items: int, capacity: int) -> float:
    r"""
    Compute theta from working memory load.

    theta = 1 - |items - optimal| / capacity

    Optimal is around Cowan's 4 items.
    Too few items: underutilized (theta < 1)
    Too many items: overloaded (theta < 1)

    Args:
        items: Number of items in working memory
        capacity: Working memory capacity

    Returns:
        theta in [0, 1]

    Reference: \cite{Miller1956}
    """
    if capacity <= 0:
        return 0.0

    optimal = min(COWAN_LIMIT, capacity)
    deviation = abs(items - optimal)
    theta = 1.0 - deviation / capacity

    return np.clip(theta, 0.0, 1.0)


def compute_executive_theta(
    pfc_activation: float,
    recurrent: float
) -> float:
    r"""
    Compute theta from executive control indicators.

    theta = sqrt(PFC_activation * recurrent_connectivity)

    Geometric mean captures need for both:
    - Prefrontal engagement (top-down control)
    - Recurrent processing (maintaining representations)

    Args:
        pfc_activation: Prefrontal cortex activity [0, 1]
        recurrent: Feedback connection strength [0, 1]

    Returns:
        theta in [0, 1]
    """
    theta = np.sqrt(pfc_activation * recurrent)
    return np.clip(theta, 0.0, 1.0)


def compute_arousal_theta(arousal: float) -> float:
    r"""
    Compute theta from arousal level (Yerkes-Dodson).

    theta = 1 - |arousal - optimal|^2 / 0.25

    Inverted-U relationship: optimal arousal around 0.5-0.7

    Args:
        arousal: Physiological arousal level [0, 1]

    Returns:
        theta in [0, 1]
    """
    deviation = abs(arousal - OPTIMAL_AROUSAL)
    theta = 1.0 - (deviation ** 2) / 0.25  # Parabolic penalty
    return np.clip(theta, 0.0, 1.0)


def compute_global_workspace_theta(
    activation: float,
    threshold: float = GW_THRESHOLD
) -> float:
    r"""
    Compute theta for Global Neuronal Workspace ignition.

    theta = 0 if activation < threshold (subliminal)
    theta = (activation - threshold) / (1 - threshold) otherwise

    Models the nonlinear ignition to conscious access.

    Args:
        activation: Neural activation level [0, 1]
        threshold: Ignition threshold (default 0.3)

    Returns:
        theta in [0, 1]

    Reference: \cite{Dehaene2006}
    """
    if activation < threshold:
        return 0.0
    theta = (activation - threshold) / (1.0 - threshold)
    return np.clip(theta, 0.0, 1.0)


def compute_cognitive_neuro_theta(system: NeuroCognitiveSystem) -> float:
    """
    Compute unified theta for a neurocognitive system.

    Weighted combination of component thetas:
    - Attention (30%): Resource allocation
    - Metacognition (25%): Self-monitoring
    - Working Memory (25%): Information maintenance
    - Executive (20%): Top-down control

    Args:
        system: NeuroCognitiveSystem instance

    Returns:
        theta in [0, 1]
    """
    theta_attention = compute_attention_theta(
        system.attention_resources,
        system.attention_demand
    )
    theta_metacog = compute_metacognition_theta(
        system.meta_d_prime,
        system.d_prime
    )
    theta_wm = compute_wm_theta(system.wm_items, system.wm_capacity)
    theta_exec = compute_executive_theta(
        system.prefrontal_activation,
        system.recurrent_connectivity
    )

    # Weighted combination
    theta = (0.30 * theta_attention +
             0.25 * theta_metacog +
             0.25 * theta_wm +
             0.20 * theta_exec)

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_attention_state(theta: float) -> AttentionState:
    """Classify attention state from theta value."""
    if theta < 0.2:
        return AttentionState.INATTENTIONAL
    elif theta < 0.5:
        return AttentionState.DIVIDED
    elif theta < 0.8:
        return AttentionState.SELECTIVE
    else:
        return AttentionState.FOCUSED


def classify_metacognition_level(theta: float) -> MetacognitiveLevel:
    """Classify metacognitive level from theta."""
    if theta < 0.1:
        return MetacognitiveLevel.ABSENT
    elif theta < 0.5:
        return MetacognitiveLevel.IMPLICIT
    elif theta < 0.9:
        return MetacognitiveLevel.EXPLICIT
    else:
        return MetacognitiveLevel.REFLECTIVE


def classify_wm_phase(items: int, capacity: int) -> WorkingMemoryPhase:
    """Classify working memory load phase."""
    if capacity <= 0:
        return WorkingMemoryPhase.OVERLOAD

    ratio = items / capacity
    if ratio < 0.3:
        return WorkingMemoryPhase.UNDERLOAD
    elif ratio < 0.7:
        return WorkingMemoryPhase.OPTIMAL
    elif ratio <= 1.0:
        return WorkingMemoryPhase.NEAR_CAPACITY
    else:
        return WorkingMemoryPhase.OVERLOAD


def classify_executive_mode(theta: float) -> ExecutiveMode:
    """Classify executive function mode from theta."""
    if theta < 0.25:
        return ExecutiveMode.AUTOMATIC
    elif theta < 0.5:
        return ExecutiveMode.ROUTINE
    elif theta < 0.75:
        return ExecutiveMode.ADAPTIVE
    else:
        return ExecutiveMode.DELIBERATE


def classify_consciousness_level(
    activation: float,
    threshold: float = GW_THRESHOLD
) -> ConsciousnessLevel:
    """Classify consciousness level (Global Workspace)."""
    if activation < threshold * 0.5:
        return ConsciousnessLevel.SUBLIMINAL
    elif activation < threshold:
        return ConsciousnessLevel.PRECONSCIOUS
    elif activation < 0.8:
        return ConsciousnessLevel.CONSCIOUS
    else:
        return ConsciousnessLevel.INTROSPECTIVE


# =============================================================================
# INTEGRATED INFORMATION ANALYSIS
# =============================================================================

def compute_phi_approximation(
    n_modules: int,
    connectivity: float,
    integration: float
) -> float:
    r"""
    Approximate Integrated Information (Phi) for neural systems.

    Phi ~ n_modules * connectivity * integration

    Full IIT computation is exponentially complex; this provides
    a tractable approximation for large systems.

    Args:
        n_modules: Number of functionally distinct modules
        connectivity: Inter-module connectivity [0, 1]
        integration: Information integration level [0, 1]

    Returns:
        Approximate Phi value (unbounded)

    Reference: \cite{Tononi2016}
    """
    return n_modules * connectivity * integration


def compute_phi_theta(phi: float, phi_max: float = 10.0) -> float:
    """
    Convert Phi to theta scale.

    theta = phi / (phi + phi_max)

    Saturating function mapping Phi to [0, 1].
    """
    if phi < 0:
        return 0.0
    theta = phi / (phi + phi_max)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# EXAMPLE COGNITIVE SYSTEMS
# =============================================================================

COGNITIVE_NEURO_SYSTEMS: Dict[str, NeuroCognitiveSystem] = {
    # Optimal states
    "vigilant": NeuroCognitiveSystem(
        name="Vigilant State",
        attention_resources=0.9,
        attention_demand=0.3,
        d_prime=2.5,
        meta_d_prime=2.3,
        wm_items=4,
        wm_capacity=7,
        prefrontal_activation=0.8,
        recurrent_connectivity=0.85,
        arousal=0.6,
        cognitive_load=0.3,
    ),
    "flow_state": NeuroCognitiveSystem(
        name="Flow State",
        attention_resources=0.95,
        attention_demand=0.9,
        d_prime=3.0,
        meta_d_prime=2.8,
        wm_items=4,
        wm_capacity=6,
        prefrontal_activation=0.7,
        recurrent_connectivity=0.9,
        arousal=0.65,
        cognitive_load=0.85,
    ),
    "mindful_attention": NeuroCognitiveSystem(
        name="Mindful Attention",
        attention_resources=0.85,
        attention_demand=0.4,
        d_prime=2.8,
        meta_d_prime=2.9,  # Enhanced metacognition
        wm_items=3,
        wm_capacity=7,
        prefrontal_activation=0.75,
        recurrent_connectivity=0.8,
        arousal=0.5,
        cognitive_load=0.35,
    ),
    # Impaired states
    "cognitive_overload": NeuroCognitiveSystem(
        name="Cognitive Overload",
        attention_resources=0.3,
        attention_demand=0.95,
        d_prime=1.5,
        meta_d_prime=0.5,
        wm_items=9,
        wm_capacity=6,
        prefrontal_activation=0.4,
        recurrent_connectivity=0.3,
        arousal=0.85,
        cognitive_load=0.95,
    ),
    "fatigue": NeuroCognitiveSystem(
        name="Fatigue State",
        attention_resources=0.4,
        attention_demand=0.6,
        d_prime=1.8,
        meta_d_prime=1.2,
        wm_items=3,
        wm_capacity=5,  # Reduced capacity
        prefrontal_activation=0.35,
        recurrent_connectivity=0.45,
        arousal=0.25,
        cognitive_load=0.6,
    ),
    "mind_wandering": NeuroCognitiveSystem(
        name="Mind Wandering",
        attention_resources=0.3,
        attention_demand=0.2,
        d_prime=1.2,
        meta_d_prime=0.3,  # Low meta-awareness
        wm_items=2,
        wm_capacity=7,
        prefrontal_activation=0.4,
        recurrent_connectivity=0.6,  # Default mode network
        arousal=0.4,
        cognitive_load=0.15,
    ),
    # Task-specific states
    "problem_solving": NeuroCognitiveSystem(
        name="Active Problem Solving",
        attention_resources=0.85,
        attention_demand=0.8,
        d_prime=2.2,
        meta_d_prime=2.0,
        wm_items=5,
        wm_capacity=6,
        prefrontal_activation=0.85,
        recurrent_connectivity=0.75,
        arousal=0.7,
        cognitive_load=0.8,
    ),
    "memory_retrieval": NeuroCognitiveSystem(
        name="Memory Retrieval",
        attention_resources=0.7,
        attention_demand=0.5,
        d_prime=2.0,
        meta_d_prime=1.8,
        wm_items=4,
        wm_capacity=7,
        prefrontal_activation=0.6,
        recurrent_connectivity=0.85,  # Hippocampal-cortical loop
        arousal=0.5,
        cognitive_load=0.5,
    ),
    "dual_task": NeuroCognitiveSystem(
        name="Dual-Task Performance",
        attention_resources=0.5,  # Split resources
        attention_demand=0.9,     # High total demand
        d_prime=1.6,
        meta_d_prime=1.0,
        wm_items=6,
        wm_capacity=6,
        prefrontal_activation=0.7,
        recurrent_connectivity=0.5,
        arousal=0.7,
        cognitive_load=0.85,
    ),
    # Clinical approximations
    "adhd_inattentive": NeuroCognitiveSystem(
        name="ADHD Inattentive Profile",
        attention_resources=0.4,
        attention_demand=0.6,
        d_prime=1.5,
        meta_d_prime=0.8,
        wm_items=3,
        wm_capacity=5,
        prefrontal_activation=0.4,
        recurrent_connectivity=0.5,
        arousal=0.4,
        cognitive_load=0.5,
    ),
    "meditation_expert": NeuroCognitiveSystem(
        name="Meditation Expert (Focused)",
        attention_resources=0.95,
        attention_demand=0.3,
        d_prime=3.2,
        meta_d_prime=3.5,  # Superior metacognition
        wm_items=2,
        wm_capacity=7,
        prefrontal_activation=0.6,
        recurrent_connectivity=0.9,
        arousal=0.45,
        cognitive_load=0.2,
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def cognitive_neuro_summary():
    """Print theta analysis for cognitive neuroscience systems."""
    print("=" * 95)
    print("COGNITIVE NEUROSCIENCE THETA ANALYSIS")
    print("=" * 95)
    print()
    print(f"{'System':<25} {'theta':>7} {'Attn':>7} {'Meta':>7} "
          f"{'WM':>7} {'Exec':>7} {'State':<15}")
    print("-" * 95)

    for name, system in COGNITIVE_NEURO_SYSTEMS.items():
        theta = compute_cognitive_neuro_theta(system)
        theta_attn = compute_attention_theta(
            system.attention_resources, system.attention_demand
        )
        theta_meta = compute_metacognition_theta(
            system.meta_d_prime, system.d_prime
        )
        theta_wm = compute_wm_theta(system.wm_items, system.wm_capacity)
        theta_exec = compute_executive_theta(
            system.prefrontal_activation, system.recurrent_connectivity
        )
        state = classify_attention_state(theta)

        print(f"{system.name:<25} {theta:>7.3f} {theta_attn:>7.3f} {theta_meta:>7.3f} "
              f"{theta_wm:>7.3f} {theta_exec:>7.3f} {state.value:<15}")

    print()
    print("Key: Flow state achieves high theta through matched demand/resources")
    print("     Cognitive overload shows low theta from resource exhaustion")


if __name__ == "__main__":
    cognitive_neuro_summary()
