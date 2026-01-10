"""
Tests for Information Systems Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Retrieval quality metrics
- Graphics and autonomy calculations
- Domain classification
"""

import pytest

from theta_calculator.domains.information_systems import (
    InformationSystemState,
    RetrievalQuality,
    AutonomyLevel,
    RenderingFidelity,
    CodeQuality,
    SystemDomain,
    compute_retrieval_theta,
    compute_ndcg_theta,
    compute_autonomy_theta,
    compute_graphics_theta,
    compute_rendering_theta,
    compute_code_theta,
    compute_latency_theta,
    compute_information_system_theta,
    classify_retrieval_quality,
    classify_autonomy_level,
    classify_rendering_fidelity,
    classify_code_quality,
    INFORMATION_SYSTEMS,
    IDEAL_LATENCY_MS,
    TARGET_FPS,
    MIN_COVERAGE,
    OPTIMAL_COMPLEXITY,
)


class TestInformationSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """INFORMATION_SYSTEMS dict should exist."""
        assert INFORMATION_SYSTEMS is not None
        assert isinstance(INFORMATION_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(INFORMATION_SYSTEMS) >= 5

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = [
            "google_search",
            "real_time_raytracing",
            "enterprise_java",
            "waymo_autonomy",
        ]
        for name in expected:
            assert name in INFORMATION_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in INFORMATION_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "domain")
            assert hasattr(system, "precision")
            assert hasattr(system, "recall")


class TestRetrievalTheta:
    """Test information retrieval theta calculation."""

    def test_perfect_retrieval(self):
        """P=1, R=1 -> theta = 1."""
        theta = compute_retrieval_theta(1.0, 1.0)
        assert theta == 1.0

    def test_zero_retrieval(self):
        """P=0, R=0 -> theta = 0."""
        theta = compute_retrieval_theta(0.0, 0.0)
        assert theta == 0.0

    def test_balanced_pr(self):
        """P=R -> theta = P = R."""
        theta = compute_retrieval_theta(0.5, 0.5)
        assert theta == pytest.approx(0.5)

    def test_high_precision_low_recall(self):
        """High P, low R -> moderate theta."""
        theta = compute_retrieval_theta(0.9, 0.1)
        # F1 = 2*0.9*0.1 / (0.9+0.1) = 0.18
        assert theta == pytest.approx(0.18, rel=0.01)

    def test_complexity_penalty(self):
        """Complexity penalty reduces theta."""
        theta_no_penalty = compute_retrieval_theta(0.8, 0.8, complexity_penalty=0.0)
        theta_with_penalty = compute_retrieval_theta(0.8, 0.8, complexity_penalty=0.3)
        assert theta_with_penalty < theta_no_penalty

    def test_theta_always_in_range(self):
        """Theta should always be in [0, 1]."""
        test_cases = [
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.9, 0.1),
        ]
        for p, r in test_cases:
            theta = compute_retrieval_theta(p, r)
            assert 0.0 <= theta <= 1.0


class TestNDCGTheta:
    """Test NDCG theta calculation."""

    def test_perfect_ranking(self):
        """Perfect ranking -> theta = 1."""
        scores = [3, 2, 1, 0]
        theta = compute_ndcg_theta(scores)
        assert theta == 1.0

    def test_reversed_ranking(self):
        """Reversed ranking -> low theta."""
        scores = [0, 1, 2, 3]
        theta = compute_ndcg_theta(scores)
        assert theta < 0.7

    def test_empty_list(self):
        """Empty list -> theta = 0."""
        theta = compute_ndcg_theta([])
        assert theta == 0.0

    def test_single_element(self):
        """Single element -> theta = 1."""
        theta = compute_ndcg_theta([5])
        assert theta == 1.0

    def test_cutoff_k(self):
        """Cutoff should limit consideration."""
        scores = [3, 2, 0, 0, 0]
        theta_full = compute_ndcg_theta(scores)
        theta_k2 = compute_ndcg_theta(scores, k=2)
        # Both should be valid
        assert 0.0 <= theta_full <= 1.0
        assert 0.0 <= theta_k2 <= 1.0


class TestAutonomyTheta:
    """Test autonomy theta calculation."""

    def test_no_intervention(self):
        """Zero intervention -> theta = 1."""
        theta = compute_autonomy_theta(0.0)
        assert theta == 1.0

    def test_full_intervention(self):
        """Full intervention -> theta = 0."""
        theta = compute_autonomy_theta(1.0)
        assert theta == 0.0

    def test_partial_intervention(self):
        """50% intervention -> theta ~ 0.5."""
        theta = compute_autonomy_theta(0.5)
        assert theta == pytest.approx(0.5)

    def test_complexity_bonus(self):
        """Complex tasks get bonus."""
        theta_simple = compute_autonomy_theta(0.3, task_complexity=1.0)
        theta_complex = compute_autonomy_theta(0.3, task_complexity=2.0)
        assert theta_complex >= theta_simple


class TestGraphicsTheta:
    """Test graphics theta calculation."""

    def test_high_quality_high_fps(self):
        """High quality at target FPS -> theta ~ 1."""
        theta = compute_graphics_theta(1.0, 60)
        assert theta == 1.0

    def test_high_quality_low_fps(self):
        """High quality at low FPS -> reduced theta."""
        theta = compute_graphics_theta(1.0, 30)
        assert theta == pytest.approx(0.5)

    def test_low_quality_high_fps(self):
        """Low quality at high FPS -> reduced theta."""
        theta = compute_graphics_theta(0.5, 60)
        assert theta == pytest.approx(0.5)

    def test_zero_quality(self):
        """Zero quality -> theta = 0."""
        theta = compute_graphics_theta(0.0, 60)
        assert theta == 0.0

    def test_fps_capped_at_target(self):
        """FPS above target doesn't increase theta."""
        theta_60 = compute_graphics_theta(0.8, 60)
        theta_120 = compute_graphics_theta(0.8, 120)
        assert theta_60 == theta_120


class TestRenderingTheta:
    """Test ray tracing rendering theta."""

    def test_converged_render(self):
        """High samples, low noise -> high theta."""
        theta = compute_rendering_theta(1024, noise_level=0.1)
        assert theta > 0.8

    def test_noisy_render(self):
        """Low samples, high noise -> low theta."""
        theta = compute_rendering_theta(10, noise_level=0.9)
        assert theta < 0.1

    def test_zero_samples(self):
        """Zero samples -> low theta."""
        theta = compute_rendering_theta(0, noise_level=0.5)
        assert theta == 0.0


class TestCodeTheta:
    """Test code quality theta calculation."""

    def test_high_coverage_low_complexity(self):
        """High coverage, low complexity -> high theta."""
        theta = compute_code_theta(0.95, 5, defect_rate=1.0)
        assert theta > 0.8

    def test_low_coverage(self):
        """Low coverage -> low theta."""
        theta = compute_code_theta(0.20, 10)
        assert theta < 0.3

    def test_high_complexity_penalty(self):
        """High complexity reduces theta."""
        theta_low = compute_code_theta(0.8, 5)
        theta_high = compute_code_theta(0.8, 50)
        assert theta_high < theta_low

    def test_defect_penalty(self):
        """High defect rate reduces theta."""
        theta_few = compute_code_theta(0.8, 10, defect_rate=1.0)
        theta_many = compute_code_theta(0.8, 10, defect_rate=50.0)
        assert theta_many < theta_few


class TestLatencyTheta:
    """Test latency theta calculation."""

    def test_ideal_latency(self):
        """At ideal latency -> theta = 1."""
        theta = compute_latency_theta(100)
        assert theta == 1.0

    def test_below_ideal(self):
        """Below ideal latency -> theta = 1."""
        theta = compute_latency_theta(50)
        assert theta == 1.0

    def test_double_ideal(self):
        """Double ideal latency -> theta = 0.5."""
        theta = compute_latency_theta(200)
        assert theta == pytest.approx(0.5)

    def test_zero_latency(self):
        """Zero latency -> theta = 1."""
        theta = compute_latency_theta(0)
        assert theta == 1.0


class TestUnifiedTheta:
    """Test unified theta calculation for systems."""

    def test_all_systems_have_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in INFORMATION_SYSTEMS.items():
            theta = compute_information_system_theta(system)
            assert 0.0 <= theta <= 1.0, f"{name}: theta={theta}"

    def test_google_high_theta(self):
        """Google Search should have high theta."""
        google = INFORMATION_SYSTEMS["google_search"]
        theta = compute_information_system_theta(google)
        assert theta > 0.7

    def test_waymo_high_autonomy(self):
        """Waymo should have high autonomy theta."""
        waymo = INFORMATION_SYSTEMS["waymo_autonomy"]
        theta = compute_information_system_theta(waymo)
        assert theta > 0.9

    def test_legacy_low_quality(self):
        """Legacy codebase should have low theta."""
        legacy = INFORMATION_SYSTEMS["legacy_codebase"]
        theta = compute_information_system_theta(legacy)
        assert theta < 0.2


class TestRetrievalQualityClassification:
    """Test retrieval quality classification."""

    def test_random(self):
        """theta < 0.2 -> RANDOM."""
        assert classify_retrieval_quality(0.1) == RetrievalQuality.RANDOM

    def test_low(self):
        """0.2 <= theta < 0.4 -> LOW."""
        assert classify_retrieval_quality(0.3) == RetrievalQuality.LOW

    def test_moderate(self):
        """0.4 <= theta < 0.6 -> MODERATE."""
        assert classify_retrieval_quality(0.5) == RetrievalQuality.MODERATE

    def test_high(self):
        """0.6 <= theta < 0.8 -> HIGH."""
        assert classify_retrieval_quality(0.7) == RetrievalQuality.HIGH

    def test_optimal(self):
        """theta >= 0.8 -> OPTIMAL."""
        assert classify_retrieval_quality(0.9) == RetrievalQuality.OPTIMAL


class TestAutonomyLevelClassification:
    """Test autonomy level classification."""

    def test_teleoperated(self):
        """theta < 0.2 -> TELEOPERATED."""
        assert classify_autonomy_level(0.1) == AutonomyLevel.TELEOPERATED

    def test_assisted(self):
        """0.2 <= theta < 0.4 -> ASSISTED."""
        assert classify_autonomy_level(0.3) == AutonomyLevel.ASSISTED

    def test_conditional(self):
        """0.4 <= theta < 0.6 -> CONDITIONAL."""
        assert classify_autonomy_level(0.5) == AutonomyLevel.CONDITIONAL

    def test_high(self):
        """0.6 <= theta < 0.8 -> HIGH."""
        assert classify_autonomy_level(0.7) == AutonomyLevel.HIGH

    def test_full(self):
        """theta >= 0.8 -> FULL."""
        assert classify_autonomy_level(0.9) == AutonomyLevel.FULL


class TestRenderingFidelityClassification:
    """Test rendering fidelity classification."""

    def test_wireframe(self):
        """theta < 0.2 -> WIREFRAME."""
        assert classify_rendering_fidelity(0.1) == RenderingFidelity.WIREFRAME

    def test_photorealistic(self):
        """theta >= 0.8 -> PHOTOREALISTIC."""
        assert classify_rendering_fidelity(0.9) == RenderingFidelity.PHOTOREALISTIC


class TestCodeQualityClassification:
    """Test code quality classification."""

    def test_prototype(self):
        """theta < 0.3 -> PROTOTYPE."""
        assert classify_code_quality(0.2) == CodeQuality.PROTOTYPE

    def test_enterprise(self):
        """theta >= 0.8 -> ENTERPRISE."""
        assert classify_code_quality(0.9) == CodeQuality.ENTERPRISE


class TestConstants:
    """Test domain constants."""

    def test_ideal_latency(self):
        """IDEAL_LATENCY_MS should be 100."""
        assert IDEAL_LATENCY_MS == 100

    def test_target_fps(self):
        """TARGET_FPS should be 60."""
        assert TARGET_FPS == 60

    def test_min_coverage(self):
        """MIN_COVERAGE should be 0.80."""
        assert MIN_COVERAGE == 0.80

    def test_optimal_complexity(self):
        """OPTIMAL_COMPLEXITY should be 10."""
        assert OPTIMAL_COMPLEXITY == 10


class TestSystemDataclass:
    """Test InformationSystemState dataclass."""

    def test_create_minimal_system(self):
        """Should create system with required parameters."""
        system = InformationSystemState(
            name="Test",
            domain="retrieval",
        )
        assert system.name == "Test"
        assert system.domain == "retrieval"

    def test_f1_auto_computed(self):
        """F1 should be auto-computed from P and R."""
        system = InformationSystemState(
            name="Test",
            domain="retrieval",
            precision=0.8,
            recall=0.6,
        )
        expected_f1 = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert system.f1_score == pytest.approx(expected_f1)

    def test_computed_f1_property(self):
        """computed_f1 should return F1 from P and R."""
        system = InformationSystemState(
            name="Test",
            domain="retrieval",
            precision=0.5,
            recall=0.5,
        )
        assert system.computed_f1 == pytest.approx(0.5)

    def test_latency_ratio(self):
        """latency_ratio should be actual/ideal."""
        system = InformationSystemState(
            name="Test",
            domain="retrieval",
            latency_ms=200,
        )
        assert system.latency_ratio == pytest.approx(2.0)

    def test_fps_ratio(self):
        """fps_ratio should be fps/target, capped at 1."""
        system = InformationSystemState(
            name="Test",
            domain="graphics",
            frame_rate=30,
        )
        assert system.fps_ratio == pytest.approx(0.5)

        system_fast = InformationSystemState(
            name="Test Fast",
            domain="graphics",
            frame_rate=120,
        )
        assert system_fast.fps_ratio == 1.0


class TestEnums:
    """Test enum definitions."""

    def test_retrieval_quality(self):
        """All retrieval quality levels should be defined."""
        assert RetrievalQuality.RANDOM.value == "random"
        assert RetrievalQuality.OPTIMAL.value == "optimal"

    def test_autonomy_levels(self):
        """All autonomy levels should be defined."""
        assert AutonomyLevel.TELEOPERATED.value == "teleoperated"
        assert AutonomyLevel.FULL.value == "full"

    def test_rendering_fidelity(self):
        """All rendering fidelity levels should be defined."""
        assert RenderingFidelity.WIREFRAME.value == "wireframe"
        assert RenderingFidelity.PHOTOREALISTIC.value == "photorealistic"

    def test_code_quality(self):
        """All code quality levels should be defined."""
        assert CodeQuality.PROTOTYPE.value == "prototype"
        assert CodeQuality.ENTERPRISE.value == "enterprise"

    def test_system_domain(self):
        """All system domains should be defined."""
        assert SystemDomain.RETRIEVAL.value == "retrieval"
        assert SystemDomain.ROBOTICS.value == "robotics"
