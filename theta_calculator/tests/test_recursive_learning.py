"""
Tests for Recursive Learning Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Recursion level classification
- Component theta calculations
"""

import pytest

from theta_calculator.domains.recursive_learning import (
    RecursionLevel,
    ImprovementType,
    compute_meta_awareness_theta,
    compute_improvement_theta,
    compute_recursion_theta,
    compute_feedback_theta,
    compute_abstraction_theta,
    compute_self_model_theta,
    compute_convergence_theta,
    compute_godel_theta,
    compute_curriculum_theta,
    compute_recursive_theta,
    classify_recursion_level,
    classify_improvement_type,
    RECURSIVE_SYSTEMS,
)


class TestRecursiveSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """RECURSIVE_SYSTEMS dict should exist."""
        assert RECURSIVE_SYSTEMS is not None
        assert isinstance(RECURSIVE_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(RECURSIVE_SYSTEMS) >= 5

    def test_system_names(self):
        """Key systems should be defined."""
        expected = [
            "static_algorithm",
            "maml_meta_learner",
            "self_play_agent",
            "reflective_architecture",
        ]
        for name in expected:
            assert name in RECURSIVE_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in RECURSIVE_SYSTEMS.items():
            assert hasattr(system, "name")
            assert hasattr(system, "meta_levels")
            assert hasattr(system, "improvement_rate")
            assert hasattr(system, "recursion_depth")
            assert hasattr(system, "closed_loops")


class TestMetaAwarenessTheta:
    """Test meta-awareness theta calculation."""

    def test_no_meta(self):
        """Zero meta-levels -> theta ~ 0."""
        theta = compute_meta_awareness_theta(0)
        assert theta == pytest.approx(0.0)

    def test_full_meta(self):
        """Max meta-levels -> theta = 1."""
        theta = compute_meta_awareness_theta(5, max_levels=5)
        assert theta == pytest.approx(1.0)

    def test_partial_meta(self):
        """Partial meta-levels -> medium theta."""
        theta = compute_meta_awareness_theta(2, max_levels=5)
        assert 0.3 < theta < 0.7

    def test_logarithmic_scaling(self):
        """Adding meta-levels has diminishing returns."""
        theta1 = compute_meta_awareness_theta(1, max_levels=10)
        theta2 = compute_meta_awareness_theta(2, max_levels=10)
        theta5 = compute_meta_awareness_theta(5, max_levels=10)

        # Differences should decrease
        diff1 = theta2 - theta1
        diff2 = theta5 - theta2
        assert diff1 > diff2 / 3  # Logarithmic scaling


class TestImprovementTheta:
    """Test self-improvement theta calculation."""

    def test_no_improvement(self):
        """Zero improvement -> theta = 0."""
        theta = compute_improvement_theta(0.0)
        assert theta == 0.0

    def test_negative_improvement(self):
        """Negative improvement -> theta = 0."""
        theta = compute_improvement_theta(-0.1)
        assert theta == 0.0

    def test_optimal_improvement(self):
        """Optimal improvement -> theta = 1."""
        theta = compute_improvement_theta(0.1, optimal_rate=0.1)
        assert theta == 1.0

    def test_above_optimal(self):
        """Above optimal -> theta = 1 (capped)."""
        theta = compute_improvement_theta(0.5, optimal_rate=0.1)
        assert theta == 1.0


class TestImprovementTypeClassification:
    """Test improvement type classification."""

    def test_none(self):
        assert classify_improvement_type(0.0) == ImprovementType.NONE

    def test_logarithmic(self):
        assert classify_improvement_type(0.005) == ImprovementType.LOGARITHMIC

    def test_linear(self):
        assert classify_improvement_type(0.05) == ImprovementType.LINEAR

    def test_exponential(self):
        assert classify_improvement_type(0.5) == ImprovementType.EXPONENTIAL

    def test_superexponential(self):
        assert classify_improvement_type(2.0) == ImprovementType.SUPEREXPONENTIAL


class TestRecursionTheta:
    """Test recursion depth theta calculation."""

    def test_no_recursion(self):
        """Zero depth -> theta = 0."""
        theta = compute_recursion_theta(0)
        assert theta == pytest.approx(0.0)

    def test_optimal_recursion(self):
        """Optimal depth (~70% of max) -> high theta."""
        theta = compute_recursion_theta(7, max_safe_depth=10)
        assert theta > 0.9

    def test_unsafe_recursion(self):
        """At max safe depth -> slightly reduced theta."""
        theta_optimal = compute_recursion_theta(7, max_safe_depth=10)
        theta_max = compute_recursion_theta(10, max_safe_depth=10)
        assert theta_max < theta_optimal


class TestFeedbackTheta:
    """Test feedback loop theta calculation."""

    def test_no_feedback(self):
        """No closed loops -> theta = 0."""
        theta = compute_feedback_theta(0, 10)
        assert theta == 0.0

    def test_full_feedback(self):
        """All loops closed -> theta = 1."""
        theta = compute_feedback_theta(10, 10)
        assert theta == 1.0

    def test_partial_feedback(self):
        """Some loops closed -> proportional theta."""
        theta = compute_feedback_theta(5, 10)
        assert theta == pytest.approx(0.5)


class TestAbstractionTheta:
    """Test abstraction climbing theta calculation."""

    def test_no_abstraction(self):
        """Zero abstraction rate -> theta = 0."""
        theta = compute_abstraction_theta(0.0)
        assert theta == 0.0

    def test_high_abstraction(self):
        """High abstraction rate -> high theta."""
        theta = compute_abstraction_theta(0.4, max_rate=0.5)
        assert theta > 0.7


class TestSelfModelTheta:
    """Test self-model theta calculation."""

    def test_no_self_model(self):
        """No explicit self-model -> low theta."""
        theta = compute_self_model_theta(False)
        assert theta == pytest.approx(0.1)

    def test_perfect_self_model(self):
        """Perfect self-model -> theta = 1."""
        theta = compute_self_model_theta(True, 1.0, 1.0)
        assert theta == 1.0

    def test_partial_self_model(self):
        """Partial self-model -> medium theta."""
        theta = compute_self_model_theta(True, 0.5, 0.5)
        assert 0.4 < theta < 0.7


class TestConvergenceTheta:
    """Test fixed-point convergence theta calculation."""

    def test_at_fixed_point(self):
        """Distance = 0 -> theta = 1."""
        theta = compute_convergence_theta(0.0)
        assert theta == 1.0

    def test_infinite_distance(self):
        """Distance = inf -> theta = 0."""
        theta = compute_convergence_theta(float('inf'))
        assert theta == 0.0

    def test_exponential_decay(self):
        """Exponential approach to fixed point."""
        theta1 = compute_convergence_theta(1.0)
        theta2 = compute_convergence_theta(2.0)
        assert theta1 > theta2


class TestGodelTheta:
    """Test Gödelian self-reference theta calculation."""

    def test_full_self_reference(self):
        """All Gödel properties -> theta = 1."""
        theta = compute_godel_theta(
            can_represent_self=True,
            can_reason_about_self=True,
            has_fixed_point=True
        )
        assert theta == 1.0

    def test_no_self_reference(self):
        """No Gödel properties -> theta = 0."""
        theta = compute_godel_theta(
            can_represent_self=False,
            can_reason_about_self=False,
            has_fixed_point=False
        )
        assert theta == 0.0


class TestCurriculumTheta:
    """Test curriculum learning theta calculation."""

    def test_perfect_curriculum(self):
        """Perfect curriculum -> theta = 1."""
        theta = compute_curriculum_theta(1.0, 1.0, 1.0)
        assert theta == 1.0

    def test_no_curriculum(self):
        """No curriculum structure -> theta = 0."""
        theta = compute_curriculum_theta(0.0, 0.0, 0.0)
        assert theta == 0.0


class TestUnifiedRecursiveTheta:
    """Test unified recursive learning theta calculation."""

    def test_all_systems_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in RECURSIVE_SYSTEMS.items():
            theta = compute_recursive_theta(system)
            assert 0 <= theta <= 1, f"{name} has invalid theta: {theta}"

    def test_static_algorithm_low_theta(self):
        """Static algorithm should have low theta."""
        system = RECURSIVE_SYSTEMS["static_algorithm"]
        theta = compute_recursive_theta(system)
        assert theta < 0.15

    def test_reflective_high_theta(self):
        """Reflective architecture should have high theta."""
        system = RECURSIVE_SYSTEMS["reflective_architecture"]
        theta = compute_recursive_theta(system)
        assert theta > 0.75

    def test_ordering_preserved(self):
        """More meta-learning -> higher theta."""
        theta_static = compute_recursive_theta(RECURSIVE_SYSTEMS["static_algorithm"])
        theta_maml = compute_recursive_theta(RECURSIVE_SYSTEMS["maml_meta_learner"])
        theta_reflect = compute_recursive_theta(RECURSIVE_SYSTEMS["reflective_architecture"])

        assert theta_static < theta_maml < theta_reflect


class TestRecursionLevelClassification:
    """Test recursion level classification."""

    def test_object_level(self):
        assert classify_recursion_level(0.1) == RecursionLevel.OBJECT_LEVEL

    def test_adaptive(self):
        assert classify_recursion_level(0.35) == RecursionLevel.ADAPTIVE

    def test_meta_level(self):
        assert classify_recursion_level(0.6) == RecursionLevel.META_LEVEL

    def test_recursive(self):
        assert classify_recursion_level(0.8) == RecursionLevel.RECURSIVE

    def test_omega_level(self):
        assert classify_recursion_level(0.95) == RecursionLevel.OMEGA_LEVEL


class TestDocstrings:
    """Test that functions have proper documentation."""

    def test_module_docstring(self):
        """Module should have docstring with citations."""
        import theta_calculator.domains.recursive_learning as module
        assert module.__doc__ is not None
        assert "\\cite{" in module.__doc__

    def test_function_docstrings(self):
        """Key functions should have docstrings."""
        functions = [
            compute_meta_awareness_theta,
            compute_improvement_theta,
            compute_recursive_theta,
            classify_recursion_level,
        ]
        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
