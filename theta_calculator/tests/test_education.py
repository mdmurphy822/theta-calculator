"""
Tests for Education Domain Module

Tests cover:
- Ebbinghaus forgetting curve
- Power law of learning
- Spaced repetition
- Knowledge integration
- Feedback effectiveness
- Theta range validation [0, 1]
"""

import pytest
import numpy as np

from theta_calculator.domains.education import (
    LearningSystem,
    LearningPhase,
    KnowledgeType,
    MemoryRetention,
    LearningCurve,
    compute_education_theta,
    compute_retention_theta,
    compute_learning_theta,
    compute_integration_theta,
    compute_feedback_theta,
    ebbinghaus_retention,
    optimal_review_time,
    power_law_performance,
    estimate_learning_rate,
    classify_learning_phase,
    EDUCATION_SYSTEMS,
)


class TestEducationSystemsExist:
    """Test that example education systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """EDUCATION_SYSTEMS dict should exist."""
        assert EDUCATION_SYSTEMS is not None
        assert isinstance(EDUCATION_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 6 systems."""
        assert len(EDUCATION_SYSTEMS) >= 6

    def test_key_systems_defined(self):
        """Key learning systems should be defined."""
        expected = [
            "cramming",
            "spaced_repetition",
            "project_based",
            "expert_tutoring"
        ]
        for name in expected:
            assert name in EDUCATION_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in EDUCATION_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "retention_rate")
            assert hasattr(system, "transfer_rate")
            assert hasattr(system, "integration_level")
            assert hasattr(system, "feedback_delay")


class TestEbbinghausRetention:
    """Test Ebbinghaus forgetting curve."""

    def test_zero_time(self):
        """At t=0, retention = 1."""
        retention = ebbinghaus_retention(0, strength=10.0)
        assert retention == pytest.approx(1.0)

    def test_decay_over_time(self):
        """Retention decreases over time."""
        r_early = ebbinghaus_retention(1.0, strength=10.0)
        r_late = ebbinghaus_retention(10.0, strength=10.0)
        assert r_late < r_early

    def test_stronger_memory_slower_decay(self):
        """Stronger memories decay slower."""
        r_weak = ebbinghaus_retention(5.0, strength=1.0)
        r_strong = ebbinghaus_retention(5.0, strength=10.0)
        assert r_strong > r_weak

    def test_exponential_model(self):
        """Exponential model: R = exp(-t/S)."""
        retention = ebbinghaus_retention(10.0, strength=10.0, decay_model="exponential")
        expected = np.exp(-1.0)  # e^(-10/10)
        assert retention == pytest.approx(expected)

    def test_power_model(self):
        """Power model: R = 1/(1 + t/S)."""
        retention = ebbinghaus_retention(10.0, strength=10.0, decay_model="power")
        expected = 0.5  # 1/(1+1)
        assert retention == pytest.approx(expected)

    def test_wickelgren_model(self):
        """Wickelgren model should work."""
        retention = ebbinghaus_retention(10.0, strength=10.0, decay_model="wickelgren")
        assert 0 < retention < 1

    def test_negative_time_raises(self):
        """Negative time should raise ValueError."""
        with pytest.raises(ValueError):
            ebbinghaus_retention(-5.0, strength=10.0)

    def test_zero_strength_raises(self):
        """Zero strength should raise ValueError."""
        with pytest.raises(ValueError):
            ebbinghaus_retention(5.0, strength=0.0)

    def test_negative_strength_raises(self):
        """Negative strength should raise ValueError."""
        with pytest.raises(ValueError):
            ebbinghaus_retention(5.0, strength=-1.0)

    def test_unknown_model_raises(self):
        """Unknown decay model should raise ValueError."""
        with pytest.raises(ValueError):
            ebbinghaus_retention(5.0, strength=10.0, decay_model="unknown")


class TestOptimalReviewTime:
    """Test optimal review time calculation."""

    def test_stronger_memory_longer_interval(self):
        """Stronger memories can wait longer for review."""
        t_weak = optimal_review_time(1.0)
        t_strong = optimal_review_time(10.0)
        assert t_strong > t_weak

    def test_higher_target_shorter_interval(self):
        """Higher target retention means shorter interval."""
        t_high = optimal_review_time(10.0, target_retention=0.95)
        t_low = optimal_review_time(10.0, target_retention=0.80)
        assert t_high < t_low

    def test_formula(self):
        """Verify formula: t = -S * ln(R)."""
        S = 10.0
        R = 0.9
        t = optimal_review_time(S, target_retention=R)
        expected = -S * np.log(R)
        assert t == pytest.approx(expected)


class TestRetentionTheta:
    """Test retention theta calculation."""

    def test_returns_memory_retention(self):
        """Should return MemoryRetention dataclass."""
        result = compute_retention_theta(5.0, strength=10.0)
        assert isinstance(result, MemoryRetention)

    def test_theta_normalized(self):
        """Theta should be normalized to max_strength."""
        result = compute_retention_theta(5.0, strength=84.0, max_strength=168.0)
        assert result.theta == pytest.approx(0.5)

    def test_theta_clipped(self):
        """Theta should be clipped to [0, 1]."""
        result = compute_retention_theta(5.0, strength=500.0, max_strength=100.0)
        assert result.theta == 1.0

    def test_high_strength_high_theta(self):
        """High strength = high theta (good retention)."""
        result = compute_retention_theta(5.0, strength=150.0, max_strength=168.0)
        assert result.theta > 0.8

    def test_low_strength_low_theta(self):
        """Low strength = low theta (poor retention)."""
        result = compute_retention_theta(5.0, strength=10.0, max_strength=168.0)
        assert result.theta < 0.1


class TestPowerLawPerformance:
    """Test power law of learning."""

    def test_first_trial(self):
        """First trial gives initial_time."""
        time = power_law_performance(1, initial_time=100.0, learning_rate=0.3)
        assert time == pytest.approx(100.0)

    def test_improvement_with_practice(self):
        """Performance improves with practice."""
        t1 = power_law_performance(1, initial_time=100.0)
        t10 = power_law_performance(10, initial_time=100.0)
        assert t10 < t1

    def test_higher_learning_rate_faster(self):
        """Higher learning rate = faster improvement."""
        t_slow = power_law_performance(10, initial_time=100.0, learning_rate=0.2)
        t_fast = power_law_performance(10, initial_time=100.0, learning_rate=0.5)
        assert t_fast < t_slow

    def test_asymptote(self):
        """Performance floor (asymptote)."""
        t = power_law_performance(1000, initial_time=100.0, asymptote=10.0)
        assert t >= 10.0  # Never below asymptote

    def test_zero_trials_raises(self):
        """Zero trials should raise ValueError."""
        with pytest.raises(ValueError):
            power_law_performance(0, initial_time=100.0)

    def test_formula(self):
        """Verify formula: T = A * N^(-beta) + asymptote."""
        N = 10
        A = 100.0
        beta = 0.3
        asymptote = 5.0
        t = power_law_performance(N, A, beta, asymptote)
        expected = A * (N ** (-beta)) + asymptote
        assert t == pytest.approx(expected)


class TestEstimateLearningRate:
    """Test learning rate estimation."""

    def test_insufficient_data(self):
        """Too few data points returns default."""
        beta = estimate_learning_rate([100, 50])
        assert beta == 0.3  # Default

    def test_power_law_data(self):
        """Power-law data gives reasonable estimate."""
        # Generate data with beta = 0.3
        times = [100 * (n ** -0.3) for n in range(1, 20)]
        beta = estimate_learning_rate(times)
        assert 0.2 < beta < 0.4

    def test_no_improvement(self):
        """Flat performance gives low learning rate."""
        times = [100, 100, 100, 100, 100]
        beta = estimate_learning_rate(times)
        assert beta < 0.1


class TestLearningTheta:
    """Test learning theta calculation."""

    def test_returns_learning_curve(self):
        """Should return LearningCurve dataclass."""
        result = compute_learning_theta(10, initial_time=100.0, current_time=50.0)
        assert isinstance(result, LearningCurve)

    def test_single_trial_zero_theta(self):
        """Single trial gives zero theta (no learning yet)."""
        result = compute_learning_theta(1, initial_time=100.0, current_time=100.0)
        assert result.theta == 0.0

    def test_fast_learning_high_theta(self):
        """Fast improvement = high theta."""
        result = compute_learning_theta(
            10, initial_time=100.0, current_time=20.0, max_learning_rate=0.5
        )
        assert result.theta > 0.5

    def test_no_improvement_low_theta(self):
        """No improvement = low theta."""
        result = compute_learning_theta(10, initial_time=100.0, current_time=100.0)
        assert result.theta == 0.0

    def test_theta_normalized(self):
        """Theta should be normalized by max_learning_rate."""
        result = compute_learning_theta(
            10, initial_time=100.0, current_time=50.0, max_learning_rate=1.0
        )
        assert 0 <= result.theta <= 1


class TestIntegrationTheta:
    """Test knowledge integration theta."""

    def test_no_connections(self):
        """No connections = zero theta."""
        theta = compute_integration_theta(0, max_connections=100)
        assert theta == 0.0

    def test_all_connected(self):
        """Full connectivity = high theta."""
        theta = compute_integration_theta(100, max_connections=100)
        assert theta >= 1.0  # May exceed due to transfer bonus

    def test_half_connected(self):
        """Half connectivity = 0.5 theta (without transfer)."""
        theta = compute_integration_theta(50, max_connections=100, transfer_rate=0.0)
        assert theta == pytest.approx(0.5)

    def test_transfer_bonus(self):
        """Transfer adds bonus."""
        theta_no_transfer = compute_integration_theta(50, 100, transfer_rate=0.0)
        theta_with_transfer = compute_integration_theta(50, 100, transfer_rate=1.0)
        assert theta_with_transfer > theta_no_transfer

    def test_zero_max_connections(self):
        """Zero max connections = zero theta."""
        theta = compute_integration_theta(10, max_connections=0)
        assert theta == 0.0


class TestFeedbackTheta:
    """Test feedback quality theta."""

    def test_instant_feedback(self):
        """Zero delay = high theta."""
        theta = compute_feedback_theta(0.0, feedback_specificity=1.0)
        assert theta == pytest.approx(1.0)

    def test_delayed_feedback(self):
        """Long delay = low theta."""
        theta = compute_feedback_theta(100.0, max_acceptable_delay=10.0)
        assert theta < 0.01

    def test_specificity_matters(self):
        """Low specificity reduces theta."""
        theta_specific = compute_feedback_theta(1.0, feedback_specificity=1.0)
        theta_vague = compute_feedback_theta(1.0, feedback_specificity=0.5)
        assert theta_vague < theta_specific

    def test_exponential_decay(self):
        """Delay follows exponential decay."""
        theta = compute_feedback_theta(10.0, max_acceptable_delay=10.0)
        expected = np.exp(-1.0)
        assert theta == pytest.approx(expected)


class TestUnifiedEducationTheta:
    """Test unified education theta calculation."""

    def test_all_systems_valid_theta(self):
        """All education systems should have theta in [0, 1]."""
        for name, system in EDUCATION_SYSTEMS.items():
            theta = compute_education_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_cramming_low_theta(self):
        """Cramming has low theta (poor learning)."""
        cramming = EDUCATION_SYSTEMS["cramming"]
        theta = compute_education_theta(cramming)
        assert theta < 0.4

    def test_immersion_high_theta(self):
        """Language immersion has high theta."""
        immersion = EDUCATION_SYSTEMS["language_immersion"]
        theta = compute_education_theta(immersion)
        assert theta > 0.8

    def test_spaced_repetition_good_theta(self):
        """Spaced repetition has good theta."""
        sr = EDUCATION_SYSTEMS["spaced_repetition"]
        theta = compute_education_theta(sr)
        assert theta > 0.5


class TestClassifyLearningPhase:
    """Test learning phase classification."""

    def test_acquisition(self):
        """Low theta -> ACQUISITION."""
        assert classify_learning_phase(0.1) == LearningPhase.ACQUISITION
        assert classify_learning_phase(0.25) == LearningPhase.ACQUISITION

    def test_consolidation(self):
        """Medium theta -> CONSOLIDATION."""
        assert classify_learning_phase(0.4) == LearningPhase.CONSOLIDATION
        assert classify_learning_phase(0.55) == LearningPhase.CONSOLIDATION

    def test_mastery(self):
        """High theta -> MASTERY."""
        assert classify_learning_phase(0.7) == LearningPhase.MASTERY
        assert classify_learning_phase(0.85) == LearningPhase.MASTERY

    def test_expertise(self):
        """Very high theta -> EXPERTISE."""
        assert classify_learning_phase(0.92) == LearningPhase.EXPERTISE
        assert classify_learning_phase(0.99) == LearningPhase.EXPERTISE

    def test_boundaries(self):
        """Test boundary values."""
        assert classify_learning_phase(0.3) == LearningPhase.CONSOLIDATION
        assert classify_learning_phase(0.6) == LearningPhase.MASTERY
        assert classify_learning_phase(0.9) == LearningPhase.EXPERTISE


class TestDataclasses:
    """Test dataclass definitions."""

    def test_learning_system_creation(self):
        """Should create LearningSystem with all parameters."""
        system = LearningSystem(
            name="Test",
            learner_count=10,
            practice_time=20.0,
            retention_rate=0.8,
            transfer_rate=0.6,
            integration_level=0.7,
            feedback_delay=5.0
        )
        assert system.name == "Test"
        assert system.retention_rate == 0.8

    def test_memory_retention_creation(self):
        """Should create MemoryRetention."""
        result = MemoryRetention(
            initial_strength=10.0,
            current_retention=0.8,
            decay_constant=10.0,
            optimal_review_time=2.0,
            theta=0.5
        )
        assert result.theta == 0.5

    def test_learning_curve_creation(self):
        """Should create LearningCurve."""
        result = LearningCurve(
            initial_time=100.0,
            current_time=50.0,
            trials=10,
            learning_rate=0.3,
            asymptote=10.0,
            theta=0.6
        )
        assert result.learning_rate == 0.3


class TestEnums:
    """Test enum definitions."""

    def test_learning_phases(self):
        """All learning phases should be defined."""
        assert LearningPhase.ACQUISITION.value == "acquisition"
        assert LearningPhase.CONSOLIDATION.value == "consolidation"
        assert LearningPhase.MASTERY.value == "mastery"
        assert LearningPhase.EXPERTISE.value == "expertise"

    def test_knowledge_types(self):
        """All knowledge types should be defined."""
        assert KnowledgeType.DECLARATIVE.value == "declarative"
        assert KnowledgeType.PROCEDURAL.value == "procedural"
        assert KnowledgeType.EPISODIC.value == "episodic"
        assert KnowledgeType.SEMANTIC.value == "semantic"
        assert KnowledgeType.METACOGNITIVE.value == "metacognitive"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_delay(self):
        """Very long feedback delay gives theta ~ 0."""
        theta = compute_feedback_theta(1e6, max_acceptable_delay=10.0)
        assert theta < 0.001

    def test_very_high_retention(self):
        """Retention rate = 1 is valid."""
        system = LearningSystem(
            name="Perfect",
            learner_count=1,
            practice_time=100.0,
            retention_rate=1.0,
            transfer_rate=1.0,
            integration_level=1.0,
            feedback_delay=0.0
        )
        theta = compute_education_theta(system)
        assert 0.95 <= theta <= 1.0

    def test_zero_everything(self):
        """System with all zeros gives low theta."""
        system = LearningSystem(
            name="Nothing",
            learner_count=1,
            practice_time=0.0,
            retention_rate=0.0,
            transfer_rate=0.0,
            integration_level=0.0,
            feedback_delay=1e6  # Very delayed feedback
        )
        theta = compute_education_theta(system)
        assert theta < 0.1
