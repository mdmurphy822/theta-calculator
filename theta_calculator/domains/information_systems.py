"""
Information Systems Domain Module

This module maps theta to information systems including information retrieval,
computer graphics, software engineering, and robotics/autonomy.

Theta Mapping:
    theta -> 0: Poor performance/quality/autonomy
    theta -> 1: Optimal performance/quality/full autonomy
    theta = F1 score: Retrieval effectiveness
    theta = 1 - intervention_rate: Autonomy level
    theta = quality * (fps/target): Graphics fidelity

Key Features:
    - Information retrieval metrics (precision, recall, NDCG)
    - Graphics rendering quality and performance
    - Software quality metrics (coverage, complexity)
    - Robotic autonomy levels (SAE-style classification)
    - Latency and throughput optimization

References:
    @book{SaltonMcGill1983,
      author = {Salton, Gerard and McGill, Michael J.},
      title = {Introduction to Modern Information Retrieval},
      publisher = {McGraw-Hill},
      year = {1983}
    }
    @article{Robertson1994,
      author = {Robertson, Stephen E. and Walker, Steve},
      title = {Some simple effective approximations to the 2-Poisson model},
      journal = {SIGIR},
      year = {1994}
    }
    @article{Kajiya1986,
      author = {Kajiya, James T.},
      title = {The rendering equation},
      journal = {ACM SIGGRAPH},
      year = {1986}
    }
    @standard{SAE_J3016,
      author = {SAE International},
      title = {Taxonomy and Definitions for Terms Related to Driving Automation Systems},
      number = {J3016},
      year = {2021}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

IDEAL_LATENCY_MS = 100      # Google's target response time
HUMAN_REACTION_MS = 200     # Minimum human response time
TARGET_FPS = 60             # Standard gaming target
CINEMATIC_FPS = 24          # Film standard
MIN_COVERAGE = 0.80         # Enterprise minimum test coverage
OPTIMAL_COMPLEXITY = 10     # Ideal cyclomatic complexity threshold


# =============================================================================
# Enums for Regime Classification
# =============================================================================

class RetrievalQuality(Enum):
    """Classification of information retrieval quality."""
    RANDOM = "random"                   # theta < 0.2: No better than chance
    LOW = "low"                         # 0.2 <= theta < 0.4
    MODERATE = "moderate"               # 0.4 <= theta < 0.6
    HIGH = "high"                       # 0.6 <= theta < 0.8
    OPTIMAL = "optimal"                 # theta >= 0.8


class AutonomyLevel(Enum):
    """Classification of robotic/vehicle autonomy (SAE J3016-style)."""
    TELEOPERATED = "teleoperated"       # Level 0: Full human control
    ASSISTED = "assisted"               # Level 1-2: Driver assistance
    CONDITIONAL = "conditional"         # Level 3: Conditional automation
    HIGH = "high"                       # Level 4: High automation
    FULL = "full"                       # Level 5: Full autonomy


class RenderingFidelity(Enum):
    """Classification of graphics rendering quality."""
    WIREFRAME = "wireframe"             # Basic geometry only
    FLAT_SHADED = "flat_shaded"         # Simple flat shading
    GOURAUD = "gouraud"                 # Per-vertex shading
    PHONG = "phong"                     # Per-pixel shading
    PHOTOREALISTIC = "photorealistic"   # Ray tracing / path tracing


class CodeQuality(Enum):
    """Classification of software code quality."""
    PROTOTYPE = "prototype"             # Quick implementation
    FUNCTIONAL = "functional"           # Works but needs polish
    PRODUCTION = "production"           # Ready for deployment
    ENTERPRISE = "enterprise"           # Fully tested, documented


class SystemDomain(Enum):
    """Sub-domains within information systems."""
    RETRIEVAL = "retrieval"             # Information retrieval (cs.IR)
    GRAPHICS = "graphics"               # Computer graphics (cs.GR)
    SOFTWARE = "software"               # Software engineering (cs.SE)
    ROBOTICS = "robotics"               # Robotics (cs.RO)


# =============================================================================
# Dataclass for Information Systems
# =============================================================================

@dataclass
class InformationSystemState:
    """
    An information system state for theta analysis.

    Attributes:
        name: System identifier
        domain: Sub-domain (retrieval, graphics, software, robotics)
        precision: Retrieval precision [0, 1]
        recall: Retrieval recall [0, 1]
        f1_score: Harmonic mean of precision/recall (computed if None)
        ndcg: Normalized Discounted Cumulative Gain [0, 1]
        query_complexity: Query complexity measure [0, inf)
        latency_ms: Response latency (milliseconds)
        throughput: Operations per second
        autonomy_score: Autonomy level [0, 1] for robotics
        intervention_rate: Human intervention rate [0, 1]
        rendering_quality: Visual fidelity score [0, 1]
        frame_rate: Frames per second
        samples_per_pixel: Ray tracing samples
        code_coverage: Test coverage [0, 1]
        cyclomatic_complexity: Code complexity metric
        defect_rate: Defects per KLOC
    """
    name: str
    domain: str  # "retrieval", "graphics", "software", "robotics"
    precision: float = 0.5
    recall: float = 0.5
    f1_score: Optional[float] = None
    ndcg: Optional[float] = None
    query_complexity: float = 1.0
    latency_ms: float = 100.0
    throughput: float = 100.0
    autonomy_score: float = 0.0
    intervention_rate: float = 1.0
    rendering_quality: float = 0.5
    frame_rate: float = 30.0
    samples_per_pixel: int = 1
    code_coverage: float = 0.0
    cyclomatic_complexity: float = 10.0
    defect_rate: float = 10.0

    def __post_init__(self):
        """Compute F1 score if not provided."""
        if self.f1_score is None and (self.precision + self.recall) > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def computed_f1(self) -> float:
        """Compute F1 score from precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def latency_ratio(self) -> float:
        """Ratio of actual to ideal latency (lower is better)."""
        return self.latency_ms / IDEAL_LATENCY_MS

    @property
    def fps_ratio(self) -> float:
        """Ratio of actual to target FPS."""
        return min(self.frame_rate / TARGET_FPS, 1.0)


# =============================================================================
# Theta Calculation Functions
# =============================================================================

def compute_retrieval_theta(
    precision: float,
    recall: float,
    complexity_penalty: float = 0.0
) -> float:
    """
    Compute theta for information retrieval based on F1 score.

    theta = F1 * (1 - complexity_penalty)

    Args:
        precision: Retrieval precision [0, 1]
        recall: Retrieval recall [0, 1]
        complexity_penalty: Penalty for query complexity [0, 1]

    Returns:
        Theta in [0, 1] where 1 = perfect retrieval
    """
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    theta = f1 * (1 - np.clip(complexity_penalty, 0, 0.5))

    return float(np.clip(theta, 0.0, 1.0))


def compute_ndcg_theta(
    relevance_scores: List[float],
    ideal_scores: Optional[List[float]] = None,
    k: Optional[int] = None
) -> float:
    """
    Compute theta from Normalized Discounted Cumulative Gain.

    NDCG = DCG / IDCG where DCG = sum(rel_i / log2(i+1))

    Args:
        relevance_scores: List of relevance scores in ranking order
        ideal_scores: Ideal ranking (sorted desc if None)
        k: Cutoff position (all if None)

    Returns:
        Theta in [0, 1] where 1 = perfect ranking
    """
    if not relevance_scores:
        return 0.0

    if k is not None:
        relevance_scores = relevance_scores[:k]

    if ideal_scores is None:
        ideal_scores = sorted(relevance_scores, reverse=True)
    elif k is not None:
        ideal_scores = ideal_scores[:k]

    def dcg(scores):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))

    dcg_actual = dcg(relevance_scores)
    dcg_ideal = dcg(ideal_scores)

    if dcg_ideal == 0:
        return 0.0

    return float(np.clip(dcg_actual / dcg_ideal, 0.0, 1.0))


def compute_autonomy_theta(
    intervention_rate: float,
    task_complexity: float = 1.0
) -> float:
    """
    Compute theta for robotic/vehicle autonomy.

    theta = (1 - intervention_rate) * task_complexity_factor

    Args:
        intervention_rate: Rate of human intervention [0, 1]
        task_complexity: Complexity of autonomous task (1.0 = standard)

    Returns:
        Theta in [0, 1] where 1 = full autonomy
    """
    base_theta = 1.0 - np.clip(intervention_rate, 0.0, 1.0)

    # Complex tasks get bonus for same intervention rate
    complexity_factor = min(1.0 + 0.1 * (task_complexity - 1.0), 1.2)

    theta = base_theta * complexity_factor

    return float(np.clip(theta, 0.0, 1.0))


def compute_graphics_theta(
    quality: float,
    frame_rate: float,
    target_fps: float = TARGET_FPS
) -> float:
    """
    Compute theta for graphics rendering.

    theta = quality * min(fps/target, 1.0)

    Args:
        quality: Visual quality score [0, 1]
        frame_rate: Actual frames per second
        target_fps: Target frame rate (default 60)

    Returns:
        Theta in [0, 1] where 1 = high quality at target fps
    """
    fps_factor = min(frame_rate / target_fps, 1.0) if target_fps > 0 else 0.0
    theta = quality * fps_factor

    return float(np.clip(theta, 0.0, 1.0))


def compute_rendering_theta(
    samples: int,
    noise_level: float = 0.5,
    target_samples: int = 1024
) -> float:
    """
    Compute theta for ray tracing convergence.

    theta = (samples/target) * (1 - noise_level)

    Args:
        samples: Samples per pixel
        noise_level: Remaining noise [0, 1]
        target_samples: Target samples for convergence

    Returns:
        Theta in [0, 1] where 1 = converged render
    """
    sample_factor = min(samples / target_samples, 1.0) if target_samples > 0 else 0.0
    noise_factor = 1.0 - np.clip(noise_level, 0.0, 1.0)

    theta = sample_factor * noise_factor

    return float(np.clip(theta, 0.0, 1.0))


def compute_code_theta(
    coverage: float,
    complexity: float,
    defect_rate: float = 0.0
) -> float:
    """
    Compute theta for software code quality.

    theta = coverage * complexity_factor * defect_factor

    Args:
        coverage: Test coverage [0, 1]
        complexity: Cyclomatic complexity (lower is better)
        defect_rate: Defects per KLOC (lower is better)

    Returns:
        Theta in [0, 1] where 1 = high quality code
    """
    # Complexity factor: optimal around 10, penalize > 20
    if complexity <= OPTIMAL_COMPLEXITY:
        complexity_factor = 1.0
    else:
        complexity_factor = max(0.5, 1.0 - (complexity - OPTIMAL_COMPLEXITY) / 40)

    # Defect factor: exponential decay
    defect_factor = np.exp(-defect_rate / 20)

    theta = coverage * complexity_factor * defect_factor

    return float(np.clip(theta, 0.0, 1.0))


def compute_latency_theta(
    latency_ms: float,
    ideal_latency_ms: float = IDEAL_LATENCY_MS
) -> float:
    """
    Compute theta from response latency.

    theta = ideal / actual for actual >= ideal
    theta = 1.0 for actual < ideal

    Args:
        latency_ms: Actual latency in milliseconds
        ideal_latency_ms: Target latency (default 100ms)

    Returns:
        Theta in [0, 1] where 1 = at or below ideal latency
    """
    if latency_ms <= 0:
        return 1.0
    if latency_ms <= ideal_latency_ms:
        return 1.0

    theta = ideal_latency_ms / latency_ms

    return float(np.clip(theta, 0.0, 1.0))


def compute_information_system_theta(system: InformationSystemState) -> float:
    """
    Compute unified theta for an information system.

    Routes to domain-specific calculation based on system.domain.

    Args:
        system: InformationSystemState instance

    Returns:
        Theta in [0, 1]
    """
    domain = system.domain.lower()

    if domain == "retrieval":
        # IR: F1-based with latency consideration
        retrieval_theta = compute_retrieval_theta(
            system.precision,
            system.recall
        )
        latency_theta = compute_latency_theta(system.latency_ms)
        return float(np.clip(0.8 * retrieval_theta + 0.2 * latency_theta, 0.0, 1.0))

    elif domain == "graphics":
        # Graphics: quality and frame rate
        return compute_graphics_theta(
            system.rendering_quality,
            system.frame_rate
        )

    elif domain == "software":
        # Software: code quality metrics
        return compute_code_theta(
            system.code_coverage,
            system.cyclomatic_complexity,
            system.defect_rate
        )

    elif domain == "robotics":
        # Robotics: autonomy level
        return compute_autonomy_theta(
            system.intervention_rate
        )

    else:
        # Default: average of available metrics
        thetas = []
        if system.precision > 0 or system.recall > 0:
            thetas.append(compute_retrieval_theta(system.precision, system.recall))
        if system.rendering_quality > 0:
            thetas.append(system.rendering_quality)
        if system.code_coverage > 0:
            thetas.append(system.code_coverage)

        return float(np.mean(thetas)) if thetas else 0.5


# =============================================================================
# Classification Functions
# =============================================================================

def classify_retrieval_quality(theta: float) -> RetrievalQuality:
    """Classify retrieval quality based on theta value."""
    if theta < 0.2:
        return RetrievalQuality.RANDOM
    elif theta < 0.4:
        return RetrievalQuality.LOW
    elif theta < 0.6:
        return RetrievalQuality.MODERATE
    elif theta < 0.8:
        return RetrievalQuality.HIGH
    else:
        return RetrievalQuality.OPTIMAL


def classify_autonomy_level(theta: float) -> AutonomyLevel:
    """Classify autonomy level based on theta value."""
    if theta < 0.2:
        return AutonomyLevel.TELEOPERATED
    elif theta < 0.4:
        return AutonomyLevel.ASSISTED
    elif theta < 0.6:
        return AutonomyLevel.CONDITIONAL
    elif theta < 0.8:
        return AutonomyLevel.HIGH
    else:
        return AutonomyLevel.FULL


def classify_rendering_fidelity(theta: float) -> RenderingFidelity:
    """Classify rendering fidelity based on theta value."""
    if theta < 0.2:
        return RenderingFidelity.WIREFRAME
    elif theta < 0.4:
        return RenderingFidelity.FLAT_SHADED
    elif theta < 0.6:
        return RenderingFidelity.GOURAUD
    elif theta < 0.8:
        return RenderingFidelity.PHONG
    else:
        return RenderingFidelity.PHOTOREALISTIC


def classify_code_quality(theta: float) -> CodeQuality:
    """Classify code quality based on theta value."""
    if theta < 0.3:
        return CodeQuality.PROTOTYPE
    elif theta < 0.5:
        return CodeQuality.FUNCTIONAL
    elif theta < 0.8:
        return CodeQuality.PRODUCTION
    else:
        return CodeQuality.ENTERPRISE


# =============================================================================
# Example Systems
# =============================================================================

INFORMATION_SYSTEMS: Dict[str, InformationSystemState] = {
    # Information Retrieval Systems
    "google_search": InformationSystemState(
        name="Google Search",
        domain="retrieval",
        precision=0.85,
        recall=0.90,
        latency_ms=50,
        throughput=10000,
    ),
    "simple_keyword_search": InformationSystemState(
        name="Simple Keyword Search",
        domain="retrieval",
        precision=0.40,
        recall=0.60,
        latency_ms=200,
        throughput=1000,
    ),
    "semantic_search_bert": InformationSystemState(
        name="BERT Semantic Search",
        domain="retrieval",
        precision=0.80,
        recall=0.85,
        latency_ms=150,
        throughput=500,
    ),
    "elasticsearch_basic": InformationSystemState(
        name="Elasticsearch Basic",
        domain="retrieval",
        precision=0.65,
        recall=0.75,
        latency_ms=80,
        throughput=5000,
    ),

    # Graphics Systems
    "real_time_raytracing": InformationSystemState(
        name="RTX Real-time Ray Tracing",
        domain="graphics",
        rendering_quality=0.90,
        frame_rate=60,
        samples_per_pixel=4,
    ),
    "mobile_game_renderer": InformationSystemState(
        name="Mobile Game Renderer",
        domain="graphics",
        rendering_quality=0.50,
        frame_rate=30,
        samples_per_pixel=1,
    ),
    "path_tracer_offline": InformationSystemState(
        name="Offline Path Tracer",
        domain="graphics",
        rendering_quality=0.99,
        frame_rate=0.01,  # Offline rendering
        samples_per_pixel=4096,
    ),
    "webgl_visualization": InformationSystemState(
        name="WebGL Visualization",
        domain="graphics",
        rendering_quality=0.60,
        frame_rate=45,
        samples_per_pixel=1,
    ),

    # Software Engineering Systems
    "enterprise_java": InformationSystemState(
        name="Enterprise Java Application",
        domain="software",
        code_coverage=0.90,
        cyclomatic_complexity=8,
        defect_rate=2.0,
    ),
    "startup_mvp": InformationSystemState(
        name="Startup MVP",
        domain="software",
        code_coverage=0.30,
        cyclomatic_complexity=15,
        defect_rate=25.0,
    ),
    "legacy_codebase": InformationSystemState(
        name="Legacy COBOL System",
        domain="software",
        code_coverage=0.10,
        cyclomatic_complexity=40,
        defect_rate=50.0,
    ),
    "modern_typescript": InformationSystemState(
        name="Modern TypeScript App",
        domain="software",
        code_coverage=0.85,
        cyclomatic_complexity=6,
        defect_rate=5.0,
    ),

    # Robotics/Autonomy Systems
    "waymo_autonomy": InformationSystemState(
        name="Waymo Level 4",
        domain="robotics",
        autonomy_score=0.90,
        intervention_rate=0.05,
    ),
    "tesla_autopilot": InformationSystemState(
        name="Tesla Autopilot",
        domain="robotics",
        autonomy_score=0.60,
        intervention_rate=0.30,
    ),
    "industrial_robot_arm": InformationSystemState(
        name="Industrial Robot Arm",
        domain="robotics",
        autonomy_score=0.95,
        intervention_rate=0.02,
    ),
    "drone_assisted": InformationSystemState(
        name="Assisted Drone",
        domain="robotics",
        autonomy_score=0.40,
        intervention_rate=0.50,
    ),
    "telepresence_robot": InformationSystemState(
        name="Telepresence Robot",
        domain="robotics",
        autonomy_score=0.10,
        intervention_rate=0.95,
    ),
}


# =============================================================================
# Demonstration Function
# =============================================================================

def demonstrate_information_systems() -> Dict[str, Dict]:
    """
    Demonstrate theta calculations for example systems.

    Returns:
        Dictionary mapping system names to their analysis results.
    """
    results = {}

    for name, system in INFORMATION_SYSTEMS.items():
        theta = compute_information_system_theta(system)

        # Get appropriate classification
        if system.domain == "retrieval":
            classification = classify_retrieval_quality(theta).value
        elif system.domain == "graphics":
            classification = classify_rendering_fidelity(theta).value
        elif system.domain == "software":
            classification = classify_code_quality(theta).value
        elif system.domain == "robotics":
            classification = classify_autonomy_level(theta).value
        else:
            classification = "unknown"

        results[name] = {
            "system": system.name,
            "domain": system.domain,
            "theta": round(theta, 4),
            "classification": classification,
        }

    return results


if __name__ == "__main__":
    results = demonstrate_information_systems()
    print("\nInformation Systems Theta Analysis")
    print("=" * 60)
    for name, data in results.items():
        print(f"\n{data['system']} ({data['domain']}):")
        print(f"  theta = {data['theta']}")
        print(f"  Classification: {data['classification']}")
