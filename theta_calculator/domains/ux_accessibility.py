"""
UX/Accessibility Domain Module

This module maps theta to user experience and accessibility metrics including
WCAG compliance, cognitive load, usability, and inclusive design.

Theta Mapping:
    theta -> 0: Poor accessibility/usability, high cognitive load
    theta -> 1: Excellent accessibility/usability, optimal cognitive load
    theta = wcag_score: Accessibility compliance
    theta = sus_score/100: Usability rating
    theta = 1 - cognitive_load: Mental capacity available

Key Features:
    - WCAG accessibility compliance levels (A, AA, AAA)
    - Cognitive load theory (intrinsic, extraneous, germane)
    - System Usability Scale (SUS) scoring
    - Nielsen heuristic evaluation
    - Flesch-Kincaid readability
    - Task success and efficiency metrics

References:
    @article{Nielsen1990,
      author = {Nielsen, Jakob and Molich, Rolf},
      title = {Heuristic evaluation of user interfaces},
      journal = {CHI},
      year = {1990}
    }
    @article{Brooke1996,
      author = {Brooke, John},
      title = {SUS: A 'quick and dirty' usability scale},
      journal = {Usability Evaluation in Industry},
      year = {1996}
    }
    @standard{WCAG21,
      author = {W3C},
      title = {Web Content Accessibility Guidelines 2.1},
      year = {2018}
    }
    @article{Sweller1988,
      author = {Sweller, John},
      title = {Cognitive load during problem solving},
      journal = {Cognitive Science},
      year = {1988}
    }
    @article{FleschKincaid1975,
      author = {Kincaid, J. Peter and Fishburne, Robert P.},
      title = {Derivation of new readability formulas},
      journal = {Navy Research},
      year = {1975}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

SUS_AVERAGE = 68.0              # Average System Usability Scale score
SUS_GOOD = 80.3                 # 'Good' threshold (Grade B)
SUS_EXCELLENT = 90.0            # 'Excellent' threshold (Grade A)

WCAG_AA_CONTRAST = 4.5          # Minimum AA contrast ratio (normal text)
WCAG_AA_CONTRAST_LARGE = 3.0    # Minimum AA contrast ratio (large text)
WCAG_AAA_CONTRAST = 7.0         # Minimum AAA contrast ratio (normal text)
WCAG_AAA_CONTRAST_LARGE = 4.5   # Minimum AAA contrast ratio (large text)

OPTIMAL_READING_LEVEL = 8.0     # 8th grade (target for general audience)
MAX_READING_LEVEL = 12.0        # 12th grade (advanced)

NIELSEN_THRESHOLD = 7.0         # Good heuristics score (out of 10)
NIELSEN_MAX = 10.0              # Perfect heuristics score

COWAN_LIMIT = 4                 # Working memory chunk limit
MILLER_LIMIT = 7                # Miller's 7 +/- 2


# =============================================================================
# Enums for Classification
# =============================================================================

class AccessibilityLevel(Enum):
    """WCAG accessibility compliance levels."""
    NONE = "non_compliant"              # No accessibility consideration
    A = "wcag_a"                        # Level A (minimum)
    AA = "wcag_aa"                      # Level AA (recommended)
    AAA = "wcag_aaa"                    # Level AAA (enhanced)


class CognitiveLoadState(Enum):
    """Cognitive load states based on available mental capacity."""
    MINIMAL = "minimal"                 # theta > 0.8: Low load, easy task
    MANAGEABLE = "manageable"           # 0.5 <= theta < 0.8: Moderate load
    CHALLENGING = "challenging"         # 0.3 <= theta < 0.5: High load
    OVERWHELMING = "overwhelming"       # theta < 0.3: Overloaded


class UsabilityLevel(Enum):
    """Usability levels based on SUS and heuristics."""
    POOR = "poor"                       # SUS < 51 (F)
    FAIR = "fair"                       # 51 <= SUS < 68 (D)
    GOOD = "good"                       # 68 <= SUS < 80.3 (C)
    EXCELLENT = "excellent"             # SUS >= 80.3 (B+)


class InteractionMode(Enum):
    """User interaction modality support."""
    VISUAL_ONLY = "visual"              # Mouse/touch only
    KEYBOARD_ACCESSIBLE = "keyboard"    # Keyboard navigation supported
    SCREEN_READER = "screen_reader"     # Full screen reader support
    MULTIMODAL = "multimodal"           # All modalities supported


class ReadabilityLevel(Enum):
    """Content readability levels."""
    SIMPLE = "simple"                   # Grade 1-5
    STANDARD = "standard"               # Grade 6-8
    ADVANCED = "advanced"               # Grade 9-12
    EXPERT = "expert"                   # Grade 12+


# =============================================================================
# Dataclass for UX/Accessibility Systems
# =============================================================================

@dataclass
class UXAccessibilitySystem:
    """
    A UX/Accessibility system state for theta analysis.

    Attributes:
        name: System/interface identifier
        wcag_level: WCAG compliance level (0=None, 1=A, 2=AA, 3=AAA)
        color_contrast_ratio: Actual contrast ratio
        cognitive_load: Combined cognitive load [0, 1]
        task_complexity: Task complexity score [0, 1]
        time_on_task: Average time to complete task (seconds)
        error_rate: User error rate [0, 1]
        task_success_rate: Task completion rate [0, 1]
        sus_score: System Usability Scale score [0, 100]
        nielsen_heuristics_score: Nielsen heuristics compliance [0, 10]
        aria_coverage: ARIA attribute coverage [0, 1]
        keyboard_navigable: Fraction keyboard accessible [0, 1]
        reading_level: Flesch-Kincaid grade level
        animation_respects_preferences: Respects prefers-reduced-motion
        focus_visible: Focus indicators visible
        alt_text_coverage: Images with alt text [0, 1]
    """
    name: str
    wcag_level: int = 0  # 0=None, 1=A, 2=AA, 3=AAA
    color_contrast_ratio: float = 4.5
    cognitive_load: float = 0.5
    task_complexity: float = 0.5
    time_on_task: float = 30.0
    error_rate: float = 0.1
    task_success_rate: float = 0.9
    sus_score: float = SUS_AVERAGE
    nielsen_heuristics_score: float = 7.0
    aria_coverage: float = 0.5
    keyboard_navigable: float = 0.8
    reading_level: float = 8.0
    animation_respects_preferences: bool = True
    focus_visible: bool = True
    alt_text_coverage: float = 0.8

    @property
    def is_wcag_aa(self) -> bool:
        """Check if system meets WCAG AA."""
        return self.wcag_level >= 2

    @property
    def is_wcag_aaa(self) -> bool:
        """Check if system meets WCAG AAA."""
        return self.wcag_level >= 3

    @property
    def sus_grade(self) -> str:
        """Get letter grade from SUS score."""
        if self.sus_score >= 90:
            return "A"
        elif self.sus_score >= 80.3:
            return "B"
        elif self.sus_score >= 68:
            return "C"
        elif self.sus_score >= 51:
            return "D"
        else:
            return "F"

    @property
    def effective_complexity(self) -> float:
        """Combined task and cognitive complexity."""
        return min(1.0, self.cognitive_load * self.task_complexity)


# =============================================================================
# Theta Calculation Functions
# =============================================================================

def compute_accessibility_theta(
    wcag_level: int,
    contrast_ratio: float,
    aria_coverage: float = 0.5,
    keyboard_navigable: float = 0.5
) -> float:
    """
    Compute theta for accessibility compliance.

    Combines WCAG level, contrast, ARIA, and keyboard accessibility.

    Args:
        wcag_level: WCAG level (0=None, 1=A, 2=AA, 3=AAA)
        contrast_ratio: Color contrast ratio
        aria_coverage: ARIA attribute coverage [0, 1]
        keyboard_navigable: Keyboard accessibility [0, 1]

    Returns:
        Theta in [0, 1] where 1 = fully accessible
    """
    # WCAG level contribution (0.4 weight)
    wcag_theta = wcag_level / 3.0

    # Contrast contribution (0.2 weight)
    if contrast_ratio >= WCAG_AAA_CONTRAST:
        contrast_theta = 1.0
    elif contrast_ratio >= WCAG_AA_CONTRAST:
        contrast_theta = 0.7
    elif contrast_ratio >= WCAG_AA_CONTRAST_LARGE:
        contrast_theta = 0.4
    else:
        contrast_theta = contrast_ratio / WCAG_AA_CONTRAST

    # ARIA and keyboard (0.2 each)
    aria_theta = np.clip(aria_coverage, 0.0, 1.0)
    keyboard_theta = np.clip(keyboard_navigable, 0.0, 1.0)

    theta = (0.4 * wcag_theta +
             0.2 * contrast_theta +
             0.2 * aria_theta +
             0.2 * keyboard_theta)

    return float(np.clip(theta, 0.0, 1.0))


def compute_usability_theta(
    sus_score: float,
    nielsen_score: float = 7.0
) -> float:
    """
    Compute theta from usability metrics.

    Combines SUS score and Nielsen heuristics.

    Args:
        sus_score: System Usability Scale score [0, 100]
        nielsen_score: Nielsen heuristics compliance [0, 10]

    Returns:
        Theta in [0, 1] where 1 = excellent usability
    """
    # Normalize SUS (0-100 -> 0-1)
    sus_theta = np.clip(sus_score / 100.0, 0.0, 1.0)

    # Normalize Nielsen (0-10 -> 0-1)
    nielsen_theta = np.clip(nielsen_score / NIELSEN_MAX, 0.0, 1.0)

    # Weighted combination
    theta = 0.6 * sus_theta + 0.4 * nielsen_theta

    return float(np.clip(theta, 0.0, 1.0))


def compute_cognitive_theta(
    cognitive_load: float,
    task_complexity: float = 0.5
) -> float:
    """
    Compute theta from cognitive load.

    theta = 1 - (load * complexity), representing available mental capacity.

    Args:
        cognitive_load: Current cognitive load [0, 1]
        task_complexity: Task complexity [0, 1]

    Returns:
        Theta in [0, 1] where 1 = minimal load, high capacity
    """
    load = np.clip(cognitive_load, 0.0, 1.0)
    complexity = np.clip(task_complexity, 0.0, 1.0)

    # Higher load + complexity = lower theta
    effective_load = load * (0.5 + 0.5 * complexity)
    theta = 1.0 - effective_load

    return float(np.clip(theta, 0.0, 1.0))


def compute_comprehension_theta(
    reading_level: float,
    target_level: float = OPTIMAL_READING_LEVEL
) -> float:
    """
    Compute theta from reading level appropriateness.

    Optimal is when reading_level matches target (e.g., 8th grade).
    Deviations in either direction reduce theta.

    Args:
        reading_level: Flesch-Kincaid grade level
        target_level: Target reading level (default 8th grade)

    Returns:
        Theta in [0, 1] where 1 = optimal readability
    """
    if target_level <= 0:
        return 0.5

    # Deviation from target
    deviation = abs(reading_level - target_level)

    # Asymmetric penalty: too complex is worse than too simple
    if reading_level > target_level:
        penalty = deviation / (MAX_READING_LEVEL - target_level + 1)
    else:
        penalty = deviation / (target_level + 1) * 0.5  # Half penalty for simpler

    theta = 1.0 - np.clip(penalty, 0.0, 1.0)

    return float(np.clip(theta, 0.0, 1.0))


def compute_task_theta(
    success_rate: float,
    error_rate: float,
    time_ratio: float = 1.0
) -> float:
    """
    Compute theta from task performance metrics.

    Args:
        success_rate: Task completion rate [0, 1]
        error_rate: Error rate [0, 1]
        time_ratio: Actual/expected time ratio (1.0 = on time)

    Returns:
        Theta in [0, 1] where 1 = efficient, error-free completion
    """
    success = np.clip(success_rate, 0.0, 1.0)
    errors = np.clip(error_rate, 0.0, 1.0)

    # Time factor: penalize if over expected time
    if time_ratio <= 1.0:
        time_factor = 1.0
    else:
        time_factor = 1.0 / time_ratio

    # Combine: success weighted highest
    theta = 0.5 * success + 0.3 * (1 - errors) + 0.2 * time_factor

    return float(np.clip(theta, 0.0, 1.0))


def compute_inclusive_theta(
    visual_support: float,
    keyboard_support: float,
    screen_reader_support: float,
    motor_support: float = 0.5
) -> float:
    """
    Compute theta for inclusive/multimodal design.

    Args:
        visual_support: Visual interface quality [0, 1]
        keyboard_support: Keyboard accessibility [0, 1]
        screen_reader_support: Screen reader compatibility [0, 1]
        motor_support: Motor accessibility (large targets, etc.) [0, 1]

    Returns:
        Theta in [0, 1] where 1 = fully inclusive
    """
    # All modalities matter; use minimum as baseline
    minimum = min(visual_support, keyboard_support,
                  screen_reader_support, motor_support)

    # Average for overall quality
    average = (visual_support + keyboard_support +
               screen_reader_support + motor_support) / 4

    # Theta favors balance (minimum) but rewards overall quality
    theta = 0.6 * minimum + 0.4 * average

    return float(np.clip(theta, 0.0, 1.0))


def compute_ux_accessibility_theta(system: UXAccessibilitySystem) -> float:
    """
    Compute unified theta for a UX/Accessibility system.

    Combines accessibility, usability, cognitive load, and task metrics.

    Args:
        system: UXAccessibilitySystem instance

    Returns:
        Theta in [0, 1]
    """
    # Accessibility contribution
    a11y_theta = compute_accessibility_theta(
        system.wcag_level,
        system.color_contrast_ratio,
        system.aria_coverage,
        system.keyboard_navigable
    )

    # Usability contribution
    usability_theta = compute_usability_theta(
        system.sus_score,
        system.nielsen_heuristics_score
    )

    # Cognitive load contribution
    cognitive_theta = compute_cognitive_theta(
        system.cognitive_load,
        system.task_complexity
    )

    # Task performance contribution
    task_theta = compute_task_theta(
        system.task_success_rate,
        system.error_rate
    )

    # Weighted combination
    theta = (0.3 * a11y_theta +
             0.3 * usability_theta +
             0.2 * cognitive_theta +
             0.2 * task_theta)

    return float(np.clip(theta, 0.0, 1.0))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_accessibility_level(wcag_level: int) -> AccessibilityLevel:
    """Classify accessibility level from WCAG compliance."""
    if wcag_level >= 3:
        return AccessibilityLevel.AAA
    elif wcag_level >= 2:
        return AccessibilityLevel.AA
    elif wcag_level >= 1:
        return AccessibilityLevel.A
    else:
        return AccessibilityLevel.NONE


def classify_cognitive_load(theta: float) -> CognitiveLoadState:
    """Classify cognitive load state based on theta value."""
    if theta >= 0.8:
        return CognitiveLoadState.MINIMAL
    elif theta >= 0.5:
        return CognitiveLoadState.MANAGEABLE
    elif theta >= 0.3:
        return CognitiveLoadState.CHALLENGING
    else:
        return CognitiveLoadState.OVERWHELMING


def classify_usability(theta: float) -> UsabilityLevel:
    """Classify usability level based on theta value."""
    if theta >= 0.803:  # SUS 80.3+
        return UsabilityLevel.EXCELLENT
    elif theta >= 0.68:  # SUS 68+
        return UsabilityLevel.GOOD
    elif theta >= 0.51:  # SUS 51+
        return UsabilityLevel.FAIR
    else:
        return UsabilityLevel.POOR


def classify_readability(reading_level: float) -> ReadabilityLevel:
    """Classify readability level from grade level."""
    if reading_level <= 5:
        return ReadabilityLevel.SIMPLE
    elif reading_level <= 8:
        return ReadabilityLevel.STANDARD
    elif reading_level <= 12:
        return ReadabilityLevel.ADVANCED
    else:
        return ReadabilityLevel.EXPERT


def classify_interaction_mode(
    keyboard: float,
    screen_reader: float,
    visual: float = 1.0
) -> InteractionMode:
    """Classify supported interaction modality."""
    if keyboard >= 0.9 and screen_reader >= 0.9:
        return InteractionMode.MULTIMODAL
    elif screen_reader >= 0.8:
        return InteractionMode.SCREEN_READER
    elif keyboard >= 0.8:
        return InteractionMode.KEYBOARD_ACCESSIBLE
    else:
        return InteractionMode.VISUAL_ONLY


# =============================================================================
# Example Systems
# =============================================================================

UX_ACCESSIBILITY_SYSTEMS: Dict[str, UXAccessibilitySystem] = {
    "gov_accessible_site": UXAccessibilitySystem(
        name="Government Accessible Website",
        wcag_level=3,  # AAA
        color_contrast_ratio=8.0,
        cognitive_load=0.3,
        task_complexity=0.4,
        sus_score=75.0,
        nielsen_heuristics_score=8.5,
        aria_coverage=0.95,
        keyboard_navigable=0.98,
        reading_level=7.0,
        alt_text_coverage=0.99,
    ),
    "startup_mvp_ui": UXAccessibilitySystem(
        name="Startup MVP Interface",
        wcag_level=0,  # Non-compliant
        color_contrast_ratio=3.5,
        cognitive_load=0.6,
        task_complexity=0.5,
        sus_score=55.0,
        nielsen_heuristics_score=5.0,
        aria_coverage=0.1,
        keyboard_navigable=0.3,
        reading_level=10.0,
        alt_text_coverage=0.2,
    ),
    "mobile_banking_app": UXAccessibilitySystem(
        name="Mobile Banking Application",
        wcag_level=2,  # AA
        color_contrast_ratio=5.5,
        cognitive_load=0.4,
        task_complexity=0.6,
        sus_score=82.0,
        nielsen_heuristics_score=8.0,
        aria_coverage=0.85,
        keyboard_navigable=0.90,
        reading_level=8.0,
        alt_text_coverage=0.95,
    ),
    "complex_dashboard": UXAccessibilitySystem(
        name="Enterprise Analytics Dashboard",
        wcag_level=1,  # A
        color_contrast_ratio=4.5,
        cognitive_load=0.8,
        task_complexity=0.9,
        sus_score=58.0,
        nielsen_heuristics_score=6.0,
        aria_coverage=0.5,
        keyboard_navigable=0.6,
        reading_level=12.0,
    ),
    "simple_landing_page": UXAccessibilitySystem(
        name="Simple Marketing Landing Page",
        wcag_level=2,  # AA
        color_contrast_ratio=6.0,
        cognitive_load=0.2,
        task_complexity=0.2,
        sus_score=85.0,
        nielsen_heuristics_score=9.0,
        aria_coverage=0.7,
        keyboard_navigable=0.95,
        reading_level=6.0,
    ),
    "gaming_interface": UXAccessibilitySystem(
        name="Gaming User Interface",
        wcag_level=0,
        color_contrast_ratio=3.0,
        cognitive_load=0.7,
        task_complexity=0.8,
        sus_score=70.0,
        nielsen_heuristics_score=6.5,
        aria_coverage=0.1,
        keyboard_navigable=0.4,
        reading_level=9.0,
        alt_text_coverage=0.3,
    ),
    "medical_device_ui": UXAccessibilitySystem(
        name="Medical Device Interface",
        wcag_level=2,
        color_contrast_ratio=7.5,
        cognitive_load=0.5,
        task_complexity=0.7,
        sus_score=78.0,
        nielsen_heuristics_score=9.0,
        aria_coverage=0.9,
        keyboard_navigable=0.95,
        reading_level=10.0,
        error_rate=0.02,
        task_success_rate=0.99,
    ),
    "elderly_focused_app": UXAccessibilitySystem(
        name="Senior-Friendly Application",
        wcag_level=3,  # AAA
        color_contrast_ratio=10.0,
        cognitive_load=0.2,
        task_complexity=0.3,
        sus_score=88.0,
        nielsen_heuristics_score=9.5,
        aria_coverage=0.95,
        keyboard_navigable=0.99,
        reading_level=5.0,  # Simple
        alt_text_coverage=1.0,
    ),
    "developer_ide": UXAccessibilitySystem(
        name="Developer IDE",
        wcag_level=1,
        color_contrast_ratio=5.0,
        cognitive_load=0.7,
        task_complexity=0.9,
        sus_score=72.0,
        nielsen_heuristics_score=7.5,
        aria_coverage=0.6,
        keyboard_navigable=0.85,
        reading_level=14.0,  # Expert level
    ),
    "e_commerce_checkout": UXAccessibilitySystem(
        name="E-Commerce Checkout Flow",
        wcag_level=2,
        color_contrast_ratio=5.0,
        cognitive_load=0.4,
        task_complexity=0.5,
        sus_score=80.0,
        nielsen_heuristics_score=8.5,
        aria_coverage=0.8,
        keyboard_navigable=0.9,
        reading_level=7.0,
        error_rate=0.05,
        task_success_rate=0.95,
    ),
}


# =============================================================================
# Demonstration Function
# =============================================================================

def demonstrate_ux_accessibility() -> Dict[str, Dict]:
    """
    Demonstrate theta calculations for example systems.

    Returns:
        Dictionary mapping system names to their analysis results.
    """
    results = {}

    for name, system in UX_ACCESSIBILITY_SYSTEMS.items():
        theta = compute_ux_accessibility_theta(system)
        a11y_level = classify_accessibility_level(system.wcag_level)
        usability = classify_usability(theta)
        cognitive = classify_cognitive_load(
            compute_cognitive_theta(system.cognitive_load, system.task_complexity)
        )

        results[name] = {
            "system": system.name,
            "theta": round(theta, 4),
            "accessibility": a11y_level.value,
            "usability": usability.value,
            "cognitive_state": cognitive.value,
            "sus_grade": system.sus_grade,
        }

    return results


if __name__ == "__main__":
    results = demonstrate_ux_accessibility()
    print("\nUX/Accessibility Systems Theta Analysis")
    print("=" * 60)
    for name, data in results.items():
        print(f"\n{data['system']}:")
        print(f"  theta = {data['theta']}")
        print(f"  Accessibility: {data['accessibility']}")
        print(f"  Usability: {data['usability']} (SUS Grade: {data['sus_grade']})")
        print(f"  Cognitive: {data['cognitive_state']}")
