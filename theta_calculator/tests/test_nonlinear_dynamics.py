"""
Tests for nonlinear_dynamics module.

Tests chaos and bifurcation theta calculations for dynamical systems
including logistic map, Lorenz attractor, and biological systems.
"""

import pytest
import numpy as np

from theta_calculator.domains.nonlinear_dynamics import (
    DYNAMICAL_SYSTEMS,
    DynamicalSystem,
    DynamicalRegime,
    AttractorType,
    compute_dynamics_theta,
    compute_lyapunov_theta,
    compute_dimension_theta,
    compute_bifurcation_theta,
    classify_regime,
    logistic_map,
    logistic_lyapunov,
    logistic_theta_sweep,
    lorenz_derivatives,
    FEIGENBAUM_DELTA,
    FEIGENBAUM_ALPHA,
    LORENZ_LYAPUNOV_EXPONENTS,
)


class TestDynamicalSystems:
    """Test the predefined dynamical systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "logistic_stable", "logistic_period2", "logistic_edge_of_chaos",
            "logistic_chaotic", "lorenz_attractor", "lorenz_stable",
            "double_pendulum", "henon_map", "cardiac_normal",
            "cardiac_fibrillation", "brain_criticality"
        ]
        for name in expected:
            assert name in DYNAMICAL_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in DYNAMICAL_SYSTEMS.items():
            assert isinstance(system, DynamicalSystem)
            assert system.name
            assert system.dimension > 0
            assert len(system.lyapunov_exponents) > 0
            assert isinstance(system.attractor_type, AttractorType)


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in DYNAMICAL_SYSTEMS.items():
            theta = compute_dynamics_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_stable_logistic_low_theta(self):
        """Stable logistic map should have low theta."""
        stable = DYNAMICAL_SYSTEMS["logistic_stable"]
        theta = compute_dynamics_theta(stable)
        assert theta < 0.5, f"Stable should have low theta: {theta}"

    def test_chaotic_logistic_high_theta(self):
        """Chaotic logistic map should have high theta."""
        chaotic = DYNAMICAL_SYSTEMS["logistic_chaotic"]
        theta = compute_dynamics_theta(chaotic)
        assert theta > 0.5, f"Chaotic should have high theta: {theta}"

    def test_edge_of_chaos_middle_theta(self):
        """Edge of chaos should have theta around 0.5."""
        edge = DYNAMICAL_SYSTEMS["logistic_edge_of_chaos"]
        theta = compute_dynamics_theta(edge)
        # Edge of chaos has lambda ≈ 0
        assert 0.3 < theta < 0.7, f"Edge of chaos theta: {theta}"

    def test_lorenz_chaotic(self):
        """Lorenz attractor should have high theta (chaotic)."""
        lorenz = DYNAMICAL_SYSTEMS["lorenz_attractor"]
        theta = compute_dynamics_theta(lorenz)
        assert theta > 0.5, f"Lorenz should be chaotic: {theta}"
        assert lorenz.is_chaotic


class TestLyapunovTheta:
    """Test Lyapunov exponent theta calculation."""

    def test_negative_lyapunov_low_theta(self):
        """Negative Lyapunov exponent should give theta < 0.5."""
        system = DynamicalSystem(
            name="Stable",
            dimension=1,
            lyapunov_exponents=[-1.0],
            attractor_type=AttractorType.POINT
        )
        theta = compute_lyapunov_theta(system)
        assert theta < 0.5

    def test_positive_lyapunov_high_theta(self):
        """Positive Lyapunov exponent should give theta > 0.5."""
        system = DynamicalSystem(
            name="Chaotic",
            dimension=1,
            lyapunov_exponents=[1.0],
            attractor_type=AttractorType.STRANGE
        )
        theta = compute_lyapunov_theta(system)
        assert theta > 0.5

    def test_zero_lyapunov_half_theta(self):
        """Zero Lyapunov exponent should give theta ≈ 0.5."""
        system = DynamicalSystem(
            name="Edge",
            dimension=1,
            lyapunov_exponents=[0.0],
            attractor_type=AttractorType.STRANGE
        )
        theta = compute_lyapunov_theta(system)
        assert 0.45 < theta < 0.55


class TestDimensionTheta:
    """Test attractor dimension theta calculation."""

    def test_integer_dimension_lower_theta(self):
        """Integer dimension (periodic) should contribute lower theta."""
        system = DynamicalSystem(
            name="Periodic",
            dimension=3,
            lyapunov_exponents=[-0.1],
            attractor_type=AttractorType.LIMIT_CYCLE,
            attractor_dimension=1.0
        )
        theta = compute_dimension_theta(system)
        # D=1 in 3D space: fractional=0, relative=0.33, weighted ~0.17
        assert theta < 0.5

    def test_fractional_dimension_higher_theta(self):
        """Fractional dimension (strange) should contribute higher theta."""
        system = DynamicalSystem(
            name="Strange",
            dimension=3,
            lyapunov_exponents=[0.5],
            attractor_type=AttractorType.STRANGE,
            attractor_dimension=2.5
        )
        theta = compute_dimension_theta(system)
        # D=2.5 in 3D: fractional=0.5, relative=0.83, weighted ~0.67
        assert theta > 0.5


class TestRegimeClassification:
    """Test dynamical regime classification."""

    def test_fixed_point_regime(self):
        """Theta < 0.1 should be FIXED_POINT."""
        assert classify_regime(0.05) == DynamicalRegime.FIXED_POINT

    def test_periodic_regime(self):
        """0.1 <= theta < 0.3 should be PERIODIC."""
        assert classify_regime(0.2) == DynamicalRegime.PERIODIC

    def test_quasiperiodic_regime(self):
        """0.3 <= theta < 0.45 should be QUASIPERIODIC."""
        assert classify_regime(0.35) == DynamicalRegime.QUASIPERIODIC

    def test_edge_of_chaos_regime(self):
        """0.45 <= theta < 0.6 should be EDGE_OF_CHAOS."""
        assert classify_regime(0.5) == DynamicalRegime.EDGE_OF_CHAOS

    def test_weakly_chaotic_regime(self):
        """0.6 <= theta < 0.8 should be WEAKLY_CHAOTIC."""
        assert classify_regime(0.7) == DynamicalRegime.WEAKLY_CHAOTIC

    def test_chaotic_regime(self):
        """theta >= 0.8 should be CHAOTIC."""
        assert classify_regime(0.9) == DynamicalRegime.CHAOTIC


class TestLogisticMap:
    """Test logistic map functions."""

    def test_logistic_map_iteration(self):
        """Test logistic map iteration."""
        x = 0.5
        r = 3.5
        x_next = logistic_map(x, r)
        # x' = 3.5 * 0.5 * 0.5 = 0.875
        assert abs(x_next - 0.875) < 0.001

    def test_logistic_stable_lyapunov(self):
        """Logistic map at r=2.5 should have negative Lyapunov."""
        lyap = logistic_lyapunov(2.5)
        assert lyap < 0

    def test_logistic_chaotic_lyapunov(self):
        """Logistic map at r=4 should have positive Lyapunov."""
        lyap = logistic_lyapunov(4.0)
        # At r=4, fully developed chaos with positive exponent
        assert lyap > 0
        assert lyap > 0.5  # Should be strongly chaotic

    def test_theta_sweep_increases(self):
        """Theta should generally increase with r."""
        sweep = logistic_theta_sweep(r_min=2.5, r_max=4.0, n_points=20)
        thetas = [theta for r, theta in sweep]
        # Last theta should be higher than first
        assert thetas[-1] > thetas[0]


class TestLorenzSystem:
    """Test Lorenz system functions."""

    def test_lorenz_derivatives(self):
        """Test Lorenz system derivative calculation."""
        state = np.array([1.0, 1.0, 1.0])
        deriv = lorenz_derivatives(state)
        # dx/dt = 10*(1-1) = 0
        # dy/dt = 1*(28-1) - 1 = 26
        # dz/dt = 1*1 - 8/3*1 ≈ -1.67
        assert abs(deriv[0]) < 0.01
        assert abs(deriv[1] - 26) < 0.01
        assert abs(deriv[2] - (-8/3 + 1)) < 0.01

    def test_lorenz_lyapunov_positive(self):
        """Lorenz system should have positive max Lyapunov."""
        assert LORENZ_LYAPUNOV_EXPONENTS[0] > 0

    def test_lorenz_lyapunov_sum_negative(self):
        """Lorenz system should be dissipative (sum of exponents < 0)."""
        assert sum(LORENZ_LYAPUNOV_EXPONENTS) < 0


class TestFeigenbaumConstants:
    """Test Feigenbaum universal constants."""

    def test_feigenbaum_delta(self):
        """Feigenbaum delta should be approximately 4.669."""
        assert abs(FEIGENBAUM_DELTA - 4.669201609) < 1e-6

    def test_feigenbaum_alpha(self):
        """Feigenbaum alpha should be approximately 2.503."""
        assert abs(FEIGENBAUM_ALPHA - 2.502907875) < 1e-6


class TestBiologicalDynamics:
    """Test biological dynamical systems."""

    def test_cardiac_normal_periodic(self):
        """Normal cardiac rhythm should be periodic."""
        cardiac = DYNAMICAL_SYSTEMS["cardiac_normal"]
        assert cardiac.attractor_type == AttractorType.LIMIT_CYCLE
        assert not cardiac.is_chaotic

    def test_cardiac_fibrillation_chaotic(self):
        """Cardiac fibrillation should be chaotic."""
        fibrillation = DYNAMICAL_SYSTEMS["cardiac_fibrillation"]
        assert fibrillation.is_chaotic
        assert fibrillation.attractor_type == AttractorType.STRANGE

    def test_brain_at_edge_of_chaos(self):
        """Brain criticality should be near edge of chaos."""
        brain = DYNAMICAL_SYSTEMS["brain_criticality"]
        theta = compute_dynamics_theta(brain)
        # Brain operates at edge of chaos for optimal computation
        assert 0.3 < theta < 0.7


class TestSystemProperties:
    """Test system property calculations."""

    def test_max_lyapunov(self):
        """Test max Lyapunov property."""
        system = DynamicalSystem(
            name="Test",
            dimension=3,
            lyapunov_exponents=[0.5, 0.0, -1.5],
            attractor_type=AttractorType.STRANGE
        )
        assert system.max_lyapunov == 0.5

    def test_lyapunov_sum(self):
        """Test Lyapunov sum property."""
        system = DynamicalSystem(
            name="Test",
            dimension=3,
            lyapunov_exponents=[0.5, 0.0, -1.5],
            attractor_type=AttractorType.STRANGE
        )
        assert system.lyapunov_sum == -1.0

    def test_is_chaotic_property(self):
        """Test is_chaotic property."""
        chaotic = DynamicalSystem(
            name="Chaotic",
            dimension=1,
            lyapunov_exponents=[0.5],
            attractor_type=AttractorType.STRANGE
        )
        stable = DynamicalSystem(
            name="Stable",
            dimension=1,
            lyapunov_exponents=[-0.5],
            attractor_type=AttractorType.POINT
        )
        assert chaotic.is_chaotic
        assert not stable.is_chaotic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
