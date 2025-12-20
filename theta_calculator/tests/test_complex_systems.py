"""
Tests for complex_systems module.

Tests complex systems theta calculations using critical phenomena
and phase transition analysis.
"""

import pytest
import numpy as np

from theta_calculator.domains.complex_systems import (
    COMPLEX_SYSTEMS,
    ComplexSystem,
    CriticalExponents,
    PhaseType,
    compute_complex_theta,
    compute_order_parameter,
    compute_susceptibility,
    compute_correlation_length,
    detect_critical_point,
    classify_phase,
    MEAN_FIELD,
    ISING_2D,
    ISING_3D,
    XY_3D,
    HEISENBERG_3D,
)


class TestComplexSystems:
    """Test the predefined complex systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "ferromagnet_hot", "ferromagnet_critical", "ferromagnet_cold",
            "opinion_polarized", "opinion_diverse", "epidemic_spreading",
            "neural_criticality", "civil_unrest"
        ]
        for name in expected:
            assert name in COMPLEX_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in COMPLEX_SYSTEMS.items():
            assert isinstance(system, ComplexSystem)
            assert system.name
            assert system.dimension > 0
            assert system.n_agents > 0
            assert system.temperature > 0
            assert system.critical_temperature > 0


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in COMPLEX_SYSTEMS.items():
            theta = compute_complex_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_critical_system_high_theta(self):
        """System at critical point should have high theta."""
        critical = COMPLEX_SYSTEMS["ferromagnet_critical"]
        theta = compute_complex_theta(critical)
        assert theta > 0.8, f"Critical system should have high theta: {theta}"

    def test_hot_system_low_theta(self):
        """System well above T_c should have lower theta."""
        hot = COMPLEX_SYSTEMS["ferromagnet_hot"]
        theta = compute_complex_theta(hot)
        # Hot system is disordered, but still has some theta
        assert theta < 0.8, f"Hot system should have lower theta: {theta}"

    def test_cold_system_moderate_theta(self):
        """System well below T_c should have moderate theta."""
        cold = COMPLEX_SYSTEMS["ferromagnet_cold"]
        theta = compute_complex_theta(cold)
        # Ordered phase has intermediate theta
        assert 0.3 < theta < 0.9, f"Cold system theta: {theta}"


class TestCriticalExponents:
    """Test critical exponent definitions and relations."""

    def test_mean_field_exponents(self):
        """Mean field exponents should have correct values."""
        assert MEAN_FIELD.beta == 0.5
        assert MEAN_FIELD.gamma == 1.0
        assert MEAN_FIELD.nu == 0.5
        assert MEAN_FIELD.delta == 3.0

    def test_ising_2d_exact_exponents(self):
        """2D Ising exponents should match Onsager solution."""
        assert abs(ISING_2D.beta - 0.125) < 0.01  # 1/8
        assert abs(ISING_2D.gamma - 1.75) < 0.01  # 7/4
        assert abs(ISING_2D.nu - 1.0) < 0.01

    def test_ising_3d_exponents(self):
        """3D Ising exponents should be non-trivial."""
        # 3D exponents differ from mean-field
        assert ISING_3D.beta != MEAN_FIELD.beta
        assert 0.3 < ISING_3D.beta < 0.35

    def test_scaling_relations(self):
        """Critical exponents should satisfy scaling relations."""
        # Test Rushbrooke: alpha + 2*beta + gamma = 2
        for exponents in [MEAN_FIELD, ISING_2D, ISING_3D]:
            rushbrooke = exponents.alpha + 2 * exponents.beta + exponents.gamma
            assert abs(rushbrooke - 2.0) < 0.1, f"Rushbrooke violation: {rushbrooke}"

    def test_verify_scaling_relations_method(self):
        """verify_scaling_relations should return dict."""
        relations = MEAN_FIELD.verify_scaling_relations()
        assert "rushbrooke" in relations
        assert "widom" in relations
        assert "fisher" in relations


class TestOrderParameter:
    """Test order parameter calculation."""

    def test_above_tc_zero_order(self):
        """Above T_c, order parameter should be 0."""
        m = compute_order_parameter(1200, T_c=1000, exponents=MEAN_FIELD)
        assert m == 0.0

    def test_below_tc_nonzero_order(self):
        """Below T_c, order parameter should be positive."""
        m = compute_order_parameter(800, T_c=1000, exponents=MEAN_FIELD)
        assert m > 0

    def test_order_increases_as_temp_decreases(self):
        """Order parameter should increase as T decreases below T_c."""
        m1 = compute_order_parameter(900, T_c=1000, exponents=MEAN_FIELD)
        m2 = compute_order_parameter(800, T_c=1000, exponents=MEAN_FIELD)
        m3 = compute_order_parameter(700, T_c=1000, exponents=MEAN_FIELD)
        assert m1 < m2 < m3


class TestSusceptibility:
    """Test susceptibility calculation."""

    def test_at_tc_diverges(self):
        """At T_c, susceptibility should diverge."""
        chi = compute_susceptibility(1000, T_c=1000, exponents=MEAN_FIELD)
        assert chi == float('inf')

    def test_away_from_tc_finite(self):
        """Away from T_c, susceptibility should be finite."""
        chi = compute_susceptibility(1200, T_c=1000, exponents=MEAN_FIELD)
        assert np.isfinite(chi)

    def test_susceptibility_positive(self):
        """Susceptibility should be positive."""
        chi = compute_susceptibility(800, T_c=1000, exponents=MEAN_FIELD)
        assert chi > 0


class TestCorrelationLength:
    """Test correlation length calculation."""

    def test_at_tc_diverges(self):
        """At T_c, correlation length should diverge."""
        xi = compute_correlation_length(1000, T_c=1000, exponents=MEAN_FIELD)
        assert xi == float('inf')

    def test_away_from_tc_finite(self):
        """Away from T_c, correlation length should be finite."""
        xi = compute_correlation_length(1200, T_c=1000, exponents=MEAN_FIELD)
        assert np.isfinite(xi)

    def test_correlation_length_positive(self):
        """Correlation length should be positive."""
        xi = compute_correlation_length(800, T_c=1000, exponents=MEAN_FIELD)
        assert xi > 0


class TestPhaseClassification:
    """Test phase classification."""

    def test_critical_phase(self):
        """System at T_c should be classified as CRITICAL."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=1.0, critical_temperature=1.0,
            exponents=MEAN_FIELD
        )
        assert classify_phase(system) == PhaseType.CRITICAL

    def test_disordered_phase(self):
        """System well above T_c should be DISORDERED."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=2.0, critical_temperature=1.0,
            exponents=MEAN_FIELD
        )
        assert classify_phase(system) == PhaseType.DISORDERED

    def test_ordered_phase(self):
        """System well below T_c should be ORDERED."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=0.5, critical_temperature=1.0,
            exponents=MEAN_FIELD
        )
        assert classify_phase(system) == PhaseType.ORDERED

    def test_supercritical_phase(self):
        """System just above T_c should be SUPERCRITICAL."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=1.1, critical_temperature=1.0,
            exponents=MEAN_FIELD
        )
        assert classify_phase(system) == PhaseType.SUPERCRITICAL

    def test_subcritical_phase(self):
        """System just below T_c should be SUBCRITICAL."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=0.9, critical_temperature=1.0,
            exponents=MEAN_FIELD
        )
        assert classify_phase(system) == PhaseType.SUBCRITICAL


class TestDetectCriticalPoint:
    """Test critical point detection."""

    def test_detect_clear_transition(self):
        """Should detect a clear phase transition."""
        # Create data with clear transition at T_c = 1.0
        temps = np.linspace(0.5, 1.5, 50)
        # Order parameter drops at T_c
        orders = [0.8 * (1 - T)**0.5 if T < 1.0 else 0.0 for T in temps]

        detected, T_c, exponents = detect_critical_point(orders, temps.tolist())
        assert detected, "Should detect phase transition"
        assert 0.9 < T_c < 1.1, f"T_c should be near 1.0: {T_c}"

    def test_no_transition(self):
        """Should not detect transition for constant data."""
        temps = np.linspace(0.5, 1.5, 50)
        orders = [0.5] * 50  # Constant

        detected, T_c, exponents = detect_critical_point(orders, temps.tolist())
        assert not detected, "Should not detect transition for constant data"

    def test_too_few_points(self):
        """Should not detect with insufficient data."""
        detected, _, _ = detect_critical_point([0.5, 0.4], [1.0, 1.1])
        assert not detected, "Should not detect with too few points"


class TestReducedTemperature:
    """Test reduced temperature property."""

    def test_at_tc_zero(self):
        """At T_c, reduced temperature should be 0."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=1000, critical_temperature=1000,
            exponents=MEAN_FIELD
        )
        assert system.reduced_temperature == 0.0

    def test_above_tc_positive(self):
        """Above T_c, reduced temperature should be positive."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=1200, critical_temperature=1000,
            exponents=MEAN_FIELD
        )
        assert system.reduced_temperature > 0

    def test_below_tc_negative(self):
        """Below T_c, reduced temperature should be negative."""
        system = ComplexSystem(
            name="Test", dimension=3, n_agents=1000,
            temperature=800, critical_temperature=1000,
            exponents=MEAN_FIELD
        )
        assert system.reduced_temperature < 0


class TestUniversalityClasses:
    """Test different universality classes."""

    def test_all_exponents_positive(self):
        """Most critical exponents should be positive."""
        for exponents in [MEAN_FIELD, ISING_3D, XY_3D, HEISENBERG_3D]:
            assert exponents.beta > 0
            assert exponents.gamma > 0
            assert exponents.nu > 0
            assert exponents.delta > 0

    def test_heisenberg_vs_ising(self):
        """Heisenberg and Ising should have different exponents."""
        assert HEISENBERG_3D.beta != ISING_3D.beta
        assert HEISENBERG_3D.nu != ISING_3D.nu


class TestThetaOrdering:
    """Test theta ordering across systems."""

    def test_critical_higher_than_ordered(self):
        """Critical system should have higher theta than ordered."""
        critical = COMPLEX_SYSTEMS["ferromagnet_critical"]
        cold = COMPLEX_SYSTEMS["ferromagnet_cold"]
        theta_critical = compute_complex_theta(critical)
        theta_cold = compute_complex_theta(cold)
        assert theta_critical > theta_cold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
