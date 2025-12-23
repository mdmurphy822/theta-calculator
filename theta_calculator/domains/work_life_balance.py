"""
Work-Life Balance Domain: Burnout, Stress, and Recovery as Theta

This module implements theta as the strain/imbalance parameter for
work-life systems, measuring deviation from optimal functioning.

Key Insight: Work-life systems exhibit a spectrum between:
- theta ~ 0: Balanced, sustainable, thriving
- theta ~ 1: Burnout, conflict, unsustainable strain

Theta Maps To:
1. Burnout Index: (Exhaustion + Cynicism) / (2 × MaxScore)
2. Effort-Reward Imbalance: E/R ratio (Siegrist model)
3. Work-Family Conflict: Bidirectional conflict / MaxConflict
4. Cognitive Load: (Intrinsic + Extraneous) / Capacity
5. Recovery Deficit: 1 - Recovery/Demands

References (see BIBLIOGRAPHY.bib):
    \\cite{Maslach1981} - Maslach Burnout Inventory
    \\cite{Siegrist1996} - Effort-Reward Imbalance model
    \\cite{Greenhaus1985} - Work-family conflict theory
    \\cite{Sweller1988} - Cognitive Load Theory
    \\cite{Demerouti2001} - Job Demands-Resources model
    \\cite{Bakker2007} - JD-R model extension
    \\cite{Karasek1979} - Job strain model
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class WellbeingPhase(Enum):
    """Work-life wellbeing phases based on theta."""
    THRIVING = "thriving"          # theta < 0.2
    BALANCED = "balanced"          # 0.2 <= theta < 0.4
    STRAINED = "strained"          # 0.4 <= theta < 0.6
    AT_RISK = "at_risk"            # 0.6 <= theta < 0.8
    BURNOUT = "burnout"            # theta >= 0.8


class BurnoutDimension(Enum):
    """Maslach Burnout Inventory dimensions."""
    EXHAUSTION = "exhaustion"           # Emotional exhaustion
    CYNICISM = "cynicism"               # Depersonalization
    INEFFICACY = "inefficacy"           # Reduced accomplishment


class ConflictDirection(Enum):
    """Work-family conflict direction."""
    WORK_TO_FAMILY = "work_to_family"   # Work interferes with family
    FAMILY_TO_WORK = "family_to_work"   # Family interferes with work


@dataclass
class WorkLifeSystem:
    """
    A work-life system for theta analysis.

    Attributes:
        name: System identifier
        exhaustion: Emotional exhaustion score (0-6 MBI scale)
        cynicism: Cynicism/depersonalization score (0-6)
        efficacy: Professional efficacy score (0-6, inverted)
        effort: Work effort level (0-100)
        reward: Perceived reward (0-100)
        work_to_family_conflict: WIF conflict (0-5)
        family_to_work_conflict: FIW conflict (0-5)
        cognitive_load: Mental workload (0-100)
        recovery_time: Hours of recovery per day
        work_demands: Hours of work demands per day

    Reference: \\cite{Maslach1981}
    """
    name: str
    exhaustion: float = 0.0
    cynicism: float = 0.0
    efficacy: float = 6.0  # Higher is better, inverted for theta
    effort: float = 50.0
    reward: float = 50.0
    work_to_family_conflict: float = 0.0
    family_to_work_conflict: float = 0.0
    cognitive_load: float = 50.0
    recovery_time: float = 8.0
    work_demands: float = 8.0


# =============================================================================
# BURNOUT (MASLACH BURNOUT INVENTORY)
# =============================================================================

def compute_burnout_theta(
    exhaustion: float,
    cynicism: float,
    efficacy: float = 6.0,
    max_score: float = 6.0
) -> float:
    """
    Compute theta from Maslach Burnout Inventory scores.

    θ = (Exhaustion + Cynicism + (MaxScore - Efficacy)) / (3 × MaxScore)

    Args:
        exhaustion: Emotional exhaustion (0-6)
        cynicism: Cynicism/depersonalization (0-6)
        efficacy: Professional efficacy (0-6, higher = better)
        max_score: Maximum score on scale

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Maslach1981}
    """
    # Invert efficacy (low efficacy = high burnout)
    inefficacy = max_score - efficacy

    # Average all three dimensions
    theta = (exhaustion + cynicism + inefficacy) / (3 * max_score)

    return np.clip(theta, 0.0, 1.0)


def classify_burnout(theta: float) -> str:
    """
    Classify burnout level based on theta.

    Reference: \\cite{Maslach1981}
    """
    if theta < 0.3:
        return "low_burnout"
    elif theta < 0.5:
        return "moderate_burnout"
    elif theta < 0.7:
        return "high_burnout"
    else:
        return "severe_burnout"


# =============================================================================
# EFFORT-REWARD IMBALANCE (SIEGRIST MODEL)
# =============================================================================

def compute_effort_reward_theta(
    effort: float,
    reward: float,
    correction_factor: float = 0.5
) -> float:
    """
    Compute theta from Effort-Reward Imbalance model.

    θ = E / (R × c)

    Where c is a correction factor. E/R > 1 indicates imbalance.

    Args:
        effort: Work effort (0-100)
        reward: Perceived reward (0-100)
        correction_factor: Adjustment factor (default 0.5)

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Siegrist1996}
    """
    if reward <= 0:
        return 1.0

    ratio = effort / (reward * (1 / correction_factor))

    # Normalize: ratio of 1 = balanced, >1 = imbalanced
    # Map to theta where 0.5 = balanced
    theta = ratio / (1 + ratio)

    return np.clip(theta, 0.0, 1.0)


def compute_overcommitment_theta(overcommitment: float, max_score: float = 24.0) -> float:
    """
    Compute theta from overcommitment scale.

    Overcommitment is the intrinsic component of ERI model.

    Reference: \\cite{Siegrist1996}
    """
    return np.clip(overcommitment / max_score, 0.0, 1.0)


# =============================================================================
# WORK-FAMILY CONFLICT
# =============================================================================

def compute_work_family_conflict_theta(
    work_to_family: float,
    family_to_work: float,
    max_conflict: float = 5.0
) -> float:
    """
    Compute theta from bidirectional work-family conflict.

    θ = (WIF + FIW) / (2 × MaxConflict)

    Args:
        work_to_family: Work interfering with family (0-5)
        family_to_work: Family interfering with work (0-5)
        max_conflict: Maximum conflict score

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Greenhaus1985}
    """
    total_conflict = work_to_family + family_to_work
    theta = total_conflict / (2 * max_conflict)

    return np.clip(theta, 0.0, 1.0)


def compute_work_family_enrichment_theta(
    work_to_family_enrichment: float,
    family_to_work_enrichment: float,
    max_enrichment: float = 5.0
) -> float:
    """
    Compute theta from work-family enrichment (inverse of conflict).

    θ = 1 - (WFE + FWE) / (2 × MaxEnrichment)

    Higher enrichment = lower theta (more balanced).

    Reference: \\cite{Greenhaus1985}
    """
    total_enrichment = work_to_family_enrichment + family_to_work_enrichment
    enrichment_ratio = total_enrichment / (2 * max_enrichment)

    # Invert: high enrichment = low theta
    theta = 1 - enrichment_ratio

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# COGNITIVE LOAD (SWELLER)
# =============================================================================

def compute_cognitive_load_theta(
    intrinsic_load: float,
    extraneous_load: float,
    germane_load: float = 0.0,
    capacity: float = 100.0
) -> float:
    """
    Compute theta from cognitive load theory.

    θ = (Intrinsic + Extraneous) / Capacity

    Germane load is productive and doesn't contribute to overload.

    Args:
        intrinsic_load: Task complexity (0-100)
        extraneous_load: Inefficient instruction/environment (0-100)
        germane_load: Productive schema construction (0-100)
        capacity: Working memory capacity

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Sweller1988}
    """
    total_load = intrinsic_load + extraneous_load
    theta = total_load / capacity

    return np.clip(theta, 0.0, 1.0)


def classify_cognitive_load(theta: float) -> str:
    """
    Classify cognitive load level.

    Reference: \\cite{Sweller1988}
    """
    if theta < 0.4:
        return "low_load"
    elif theta < 0.7:
        return "optimal_load"
    elif theta < 0.9:
        return "high_load"
    else:
        return "overload"


# =============================================================================
# JOB DEMANDS-RESOURCES (DEMEROUTI)
# =============================================================================

def compute_jdr_theta(
    demands: float,
    resources: float,
    max_scale: float = 100.0
) -> float:
    """
    Compute theta from Job Demands-Resources model.

    θ = Demands / (Demands + Resources)

    High demands + low resources = high strain (theta → 1)
    Low demands + high resources = motivation (theta → 0)

    Args:
        demands: Job demands (0-100)
        resources: Job resources (0-100)
        max_scale: Maximum scale value

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Demerouti2001}
    """
    total = demands + resources
    if total <= 0:
        return 0.5

    theta = demands / total

    return np.clip(theta, 0.0, 1.0)


def compute_recovery_theta(
    recovery_time: float,
    work_demands: float,
    min_recovery_ratio: float = 0.5
) -> float:
    """
    Compute theta from recovery-demand ratio.

    θ = 1 - min(Recovery/Demands, 1)

    Args:
        recovery_time: Hours of recovery (sleep, leisure)
        work_demands: Hours of work demands
        min_recovery_ratio: Minimum healthy ratio

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Demerouti2001}
    """
    if work_demands <= 0:
        return 0.0

    ratio = recovery_time / work_demands
    # If ratio >= 1, full recovery → theta = 0
    # If ratio = 0, no recovery → theta = 1
    theta = 1 - min(ratio, 1.0)

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# KARASEK JOB STRAIN MODEL
# =============================================================================

def compute_job_strain_theta(
    demands: float,
    control: float,
    support: float = 50.0,
    max_scale: float = 100.0
) -> float:
    """
    Compute theta from Karasek's Job Strain model.

    θ = Demands / (Control + Support/2)

    High demands + low control = high strain

    Args:
        demands: Psychological job demands (0-100)
        control: Job decision latitude (0-100)
        support: Social support (0-100)
        max_scale: Maximum scale value

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Karasek1979}
    """
    buffered_control = control + support / 2

    if buffered_control <= 0:
        return 1.0

    ratio = demands / buffered_control
    # Normalize so ratio=1 gives theta=0.5
    theta = ratio / (1 + ratio)

    return np.clip(theta, 0.0, 1.0)


def classify_job_strain(theta: float) -> str:
    """
    Classify job strain quadrant based on theta.

    Reference: \\cite{Karasek1979}
    """
    if theta < 0.3:
        return "low_strain"      # Low demands, high control
    elif theta < 0.5:
        return "active"          # High demands, high control
    elif theta < 0.7:
        return "passive"         # Low demands, low control
    else:
        return "high_strain"     # High demands, low control


# =============================================================================
# COMPOSITE WORK-LIFE THETA
# =============================================================================

def compute_work_life_theta(system: WorkLifeSystem) -> Dict[str, float]:
    """
    Compute comprehensive work-life theta from all dimensions.

    Returns individual and composite theta values.

    Args:
        system: WorkLifeSystem with all measurements

    Returns:
        Dictionary with theta values for each dimension

    Reference: \\cite{Demerouti2001}
    """
    # Individual dimensions
    burnout = compute_burnout_theta(
        system.exhaustion, system.cynicism, system.efficacy
    )

    effort_reward = compute_effort_reward_theta(
        system.effort, system.reward
    )

    work_family = compute_work_family_conflict_theta(
        system.work_to_family_conflict, system.family_to_work_conflict
    )

    cognitive = compute_cognitive_load_theta(
        system.cognitive_load * 0.6,  # Intrinsic
        system.cognitive_load * 0.4   # Extraneous estimate
    )

    recovery = compute_recovery_theta(
        system.recovery_time, system.work_demands
    )

    # Composite: weighted average
    # Burnout is strongest predictor
    composite = (
        0.30 * burnout +
        0.25 * effort_reward +
        0.20 * work_family +
        0.15 * cognitive +
        0.10 * recovery
    )

    return {
        "burnout": burnout,
        "effort_reward": effort_reward,
        "work_family_conflict": work_family,
        "cognitive_load": cognitive,
        "recovery_deficit": recovery,
        "composite": np.clip(composite, 0.0, 1.0),
        "phase": classify_wellbeing_phase(composite)
    }


def classify_wellbeing_phase(theta: float) -> WellbeingPhase:
    """Classify wellbeing phase from theta."""
    if theta < 0.2:
        return WellbeingPhase.THRIVING
    elif theta < 0.4:
        return WellbeingPhase.BALANCED
    elif theta < 0.6:
        return WellbeingPhase.STRAINED
    elif theta < 0.8:
        return WellbeingPhase.AT_RISK
    else:
        return WellbeingPhase.BURNOUT


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

WORK_LIFE_SYSTEMS: Dict[str, WorkLifeSystem] = {
    "balanced_professional": WorkLifeSystem(
        name="Balanced Professional",
        exhaustion=1.5,
        cynicism=1.0,
        efficacy=5.0,
        effort=60,
        reward=70,
        work_to_family_conflict=1.0,
        family_to_work_conflict=0.5,
        cognitive_load=40,
        recovery_time=10,
        work_demands=8
    ),
    "high_performer": WorkLifeSystem(
        name="High Performer",
        exhaustion=2.5,
        cynicism=1.5,
        efficacy=5.5,
        effort=80,
        reward=75,
        work_to_family_conflict=2.0,
        family_to_work_conflict=1.0,
        cognitive_load=60,
        recovery_time=7,
        work_demands=10
    ),
    "overworked_parent": WorkLifeSystem(
        name="Overworked Parent",
        exhaustion=4.0,
        cynicism=2.5,
        efficacy=4.0,
        effort=85,
        reward=50,
        work_to_family_conflict=4.0,
        family_to_work_conflict=3.5,
        cognitive_load=75,
        recovery_time=5,
        work_demands=12
    ),
    "burnout_case": WorkLifeSystem(
        name="Burnout Case",
        exhaustion=5.5,
        cynicism=5.0,
        efficacy=2.0,
        effort=90,
        reward=30,
        work_to_family_conflict=4.5,
        family_to_work_conflict=4.0,
        cognitive_load=90,
        recovery_time=4,
        work_demands=14
    ),
    "remote_worker": WorkLifeSystem(
        name="Remote Worker",
        exhaustion=2.0,
        cynicism=2.0,
        efficacy=4.5,
        effort=65,
        reward=60,
        work_to_family_conflict=2.5,
        family_to_work_conflict=2.5,
        cognitive_load=55,
        recovery_time=8,
        work_demands=9
    ),
    "healthcare_worker": WorkLifeSystem(
        name="Healthcare Worker",
        exhaustion=4.5,
        cynicism=3.0,
        efficacy=4.5,
        effort=85,
        reward=55,
        work_to_family_conflict=3.5,
        family_to_work_conflict=2.0,
        cognitive_load=70,
        recovery_time=6,
        work_demands=12
    ),
    "engaged_employee": WorkLifeSystem(
        name="Engaged Employee",
        exhaustion=1.0,
        cynicism=0.5,
        efficacy=5.5,
        effort=70,
        reward=80,
        work_to_family_conflict=1.0,
        family_to_work_conflict=1.0,
        cognitive_load=50,
        recovery_time=9,
        work_demands=8
    ),
    "new_graduate": WorkLifeSystem(
        name="New Graduate",
        exhaustion=2.0,
        cynicism=1.0,
        efficacy=4.0,
        effort=75,
        reward=45,
        work_to_family_conflict=1.5,
        family_to_work_conflict=0.5,
        cognitive_load=70,
        recovery_time=7,
        work_demands=9
    ),
}


def work_life_theta_summary():
    """Print theta analysis for example work-life systems."""
    print("=" * 75)
    print("WORK-LIFE BALANCE THETA ANALYSIS")
    print("=" * 75)
    print()
    print(f"{'System':<25} {'Burnout':>8} {'E-R':>8} {'W-F':>8} {'Cog':>8} {'θ':>8} {'Phase':<12}")
    print("-" * 75)

    for name, system in WORK_LIFE_SYSTEMS.items():
        result = compute_work_life_theta(system)
        print(f"{system.name:<25} "
              f"{result['burnout']:>8.3f} "
              f"{result['effort_reward']:>8.3f} "
              f"{result['work_family_conflict']:>8.3f} "
              f"{result['cognitive_load']:>8.3f} "
              f"{result['composite']:>8.3f} "
              f"{result['phase'].value:<12}")


if __name__ == "__main__":
    work_life_theta_summary()
