"""
Tests for Applied Mathematics Domain Module

Tests cover:
- PDE analysis and stability
- Optimization convergence
- Numerical conditioning
- Theta range validation [0, 1]
"""


from theta_calculator.domains.applied_mathematics import (
    AppliedMathSystem,
    PDEType,
    OptimizationClass,
    ConvergenceType,
    StabilityClass,
    compute_applied_math_theta,
    compute_regularity_theta,
    compute_stability_theta,
    compute_convergence_theta,
    compute_rate_theta,
    compute_duality_gap_theta,
    compute_condition_theta,
    compute_accuracy_theta,
    compute_mixing_theta,
    classify_pde_type,
    classify_optimization,
    classify_convergence,
    classify_stability,
    APPLIED_MATH_SYSTEMS,
)


class TestAppliedMathSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """APPLIED_MATH_SYSTEMS dict should exist."""
        assert APPLIED_MATH_SYSTEMS is not None
        assert isinstance(APPLIED_MATH_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(APPLIED_MATH_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = ["laplace_fd", "newton_optimization", "spectral_method"]
        for name in expected:
            assert name in APPLIED_MATH_SYSTEMS, f"Missing system: {name}"


class TestRegularityTheta:
    """Test regularity theta calculation."""

    def test_zero_regularity(self):
        """Zero regularity gives theta = 0."""
        theta = compute_regularity_theta(0)
        assert theta == 0.0

    def test_target_regularity(self):
        """Target regularity gives theta = 1."""
        theta = compute_regularity_theta(2.0, 2.0)
        assert theta == 1.0

    def test_half_regularity(self):
        """Half target gives theta = 0.5."""
        theta = compute_regularity_theta(1.0, 2.0)
        assert theta == 0.5


class TestStabilityTheta:
    """Test CFL stability theta calculation."""

    def test_zero_cfl(self):
        """Zero CFL gives theta = 1 (no constraint)."""
        theta = compute_stability_theta(0)
        assert theta == 1.0

    def test_max_cfl(self):
        """At max CFL gives theta = 0 (unstable)."""
        theta = compute_stability_theta(1.0, 1.0)
        assert theta == 0.0

    def test_half_cfl(self):
        """Half max CFL gives theta = 0.5."""
        theta = compute_stability_theta(0.5, 1.0)
        assert theta == 0.5


class TestConvergenceTheta:
    """Test convergence order theta calculation."""

    def test_zero_order(self):
        """Zero order gives theta = 0."""
        theta = compute_convergence_theta(0)
        assert theta == 0.0

    def test_max_order(self):
        """Max order gives theta = 1."""
        theta = compute_convergence_theta(10, 10)
        assert theta == 1.0


class TestRateTheta:
    """Test convergence rate theta calculation."""

    def test_instant(self):
        """Zero rate (instant) gives theta = 1."""
        theta = compute_rate_theta(0)
        assert theta == 1.0

    def test_slow(self):
        """Rate >= 1 gives theta = 0."""
        theta = compute_rate_theta(1.0)
        assert theta == 0.0

    def test_typical(self):
        """Typical rate gives intermediate theta."""
        theta = compute_rate_theta(0.5, 0.5)
        assert theta == 1.0


class TestDualityGapTheta:
    """Test duality gap theta calculation."""

    def test_no_gap(self):
        """Equal primal/dual gives theta = 1."""
        theta = compute_duality_gap_theta(100, 100)
        assert theta == 1.0

    def test_large_gap(self):
        """Large gap gives low theta."""
        theta = compute_duality_gap_theta(100, 50)
        assert theta < 1.0


class TestConditionTheta:
    """Test condition number theta calculation."""

    def test_perfect(self):
        """kappa = 1 gives theta = 1."""
        theta = compute_condition_theta(1.0)
        assert theta == 1.0

    def test_singular(self):
        """kappa = inf gives theta = 0."""
        theta = compute_condition_theta(float('inf'))
        assert theta == 0.0


class TestAccuracyTheta:
    """Test accuracy theta calculation."""

    def test_within_tolerance(self):
        """Error <= tolerance gives theta = 1."""
        theta = compute_accuracy_theta(1e-8, 1e-6)
        assert theta == 1.0

    def test_large_error(self):
        """Large error gives low theta."""
        theta = compute_accuracy_theta(0.1, 1e-6)
        assert theta < 0.01


class TestMixingTheta:
    """Test mixing time theta calculation."""

    def test_fast_mixing(self):
        """Fast mixing gives high theta."""
        theta = compute_mixing_theta(10, 100)
        assert theta == 1.0

    def test_slow_mixing(self):
        """Slow mixing gives low theta."""
        theta = compute_mixing_theta(1000, 100)
        assert theta < 0.2


class TestUnifiedAppliedMathTheta:
    """Test unified applied math theta calculation."""

    def test_all_systems_valid_theta(self):
        """All systems should have theta in [0, 1]."""
        for name, system in APPLIED_MATH_SYSTEMS.items():
            theta = compute_applied_math_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_spectral_high_order(self):
        """Spectral method has high convergence order."""
        spectral = APPLIED_MATH_SYSTEMS["spectral_method"]
        theta = compute_applied_math_theta(spectral)
        assert theta > 0.3


class TestClassifyPDEType:
    """Test PDE type classification."""

    def test_elliptic(self):
        """Negative discriminant -> ELLIPTIC."""
        result = classify_pde_type(1, 0, 1)  # Laplace
        assert result == PDEType.ELLIPTIC

    def test_parabolic(self):
        """Zero discriminant -> PARABOLIC."""
        result = classify_pde_type(1, 2, 1)  # b^2 - 4ac = 0
        assert result == PDEType.PARABOLIC

    def test_hyperbolic(self):
        """Positive discriminant -> HYPERBOLIC."""
        result = classify_pde_type(1, 0, -1)  # Wave equation
        assert result == PDEType.HYPERBOLIC


class TestClassifyOptimization:
    """Test optimization classification."""

    def test_convex(self):
        """Convex continuous -> CONVEX."""
        result = classify_optimization(True, False, False)
        assert result == OptimizationClass.CONVEX

    def test_stochastic(self):
        """Stochastic flag -> STOCHASTIC."""
        result = classify_optimization(True, False, True)
        assert result == OptimizationClass.STOCHASTIC

    def test_combinatorial(self):
        """Discrete -> COMBINATORIAL."""
        result = classify_optimization(False, True, False)
        assert result == OptimizationClass.COMBINATORIAL


class TestClassifyConvergence:
    """Test convergence type classification."""

    def test_quadratic(self):
        """Order >= 2 -> QUADRATIC."""
        result = classify_convergence(2.0)
        assert result == ConvergenceType.QUADRATIC

    def test_linear(self):
        """Order = 1 -> LINEAR."""
        result = classify_convergence(1.0)
        assert result == ConvergenceType.LINEAR

    def test_sublinear(self):
        """Order < 1 -> SUBLINEAR."""
        result = classify_convergence(0.5)
        assert result == ConvergenceType.SUBLINEAR


class TestClassifyStability:
    """Test stability classification."""

    def test_unstable(self):
        """Low theta -> UNSTABLE."""
        result = classify_stability(0.1)
        assert result == StabilityClass.UNSTABLE

    def test_unconditional(self):
        """High theta -> UNCONDITIONALLY."""
        result = classify_stability(0.9)
        assert result == StabilityClass.UNCONDITIONALLY


class TestEnums:
    """Test enum definitions."""

    def test_pde_types(self):
        """All PDE types defined."""
        assert PDEType.ELLIPTIC.value == "elliptic"
        assert PDEType.HYPERBOLIC.value == "hyperbolic"

    def test_optimization_classes(self):
        """All optimization classes defined."""
        assert OptimizationClass.CONVEX.value == "convex"
        assert OptimizationClass.STOCHASTIC.value == "stochastic"


class TestAppliedMathSystemDataclass:
    """Test AppliedMathSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with name only."""
        system = AppliedMathSystem(name="Test")
        assert system.name == "Test"
        assert system.dimension == 1
        assert system.condition_number == 1.0

    def test_custom_values(self):
        """Can set custom values."""
        system = AppliedMathSystem(
            name="Custom",
            dimension=3,
            condition_number=50.0,
            pde_type=PDEType.PARABOLIC
        )
        assert system.condition_number == 50.0
        assert system.pde_type == PDEType.PARABOLIC


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_target(self):
        """Zero target gives theta = 0."""
        theta = compute_regularity_theta(1.0, 0)
        assert theta == 0.0

    def test_negative_values(self):
        """Negative values handled gracefully."""
        theta = compute_convergence_theta(-1)
        assert theta == 0.0
