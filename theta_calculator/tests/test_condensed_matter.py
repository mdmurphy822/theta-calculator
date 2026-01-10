"""
Tests for Condensed Matter Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Phase regime classification
- Transport and localization theta
- Physical constants
"""

import pytest
import numpy as np

from theta_calculator.domains.condensed_matter import (
    CondensedMatterSystem,
    PhaseRegime,
    TransportType,
    DisorderLevel,
    TopologicalPhase,
    CriticalExponentClass,
    compute_phase_theta,
    compute_order_theta,
    compute_localization_theta,
    compute_transport_theta,
    compute_hall_theta,
    compute_correlation_theta,
    compute_topological_theta,
    compute_condensed_matter_theta,
    classify_phase_regime,
    classify_transport,
    classify_disorder,
    classify_topological,
    CONDENSED_MATTER_SYSTEMS,
    K_B,
    E_CHARGE,
    H_PLANCK,
    G_0,
    PHI_0,
    ISING_2D_TC,
    ANDERSON_WC_3D,
)


class TestCondensedMatterSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """CONDENSED_MATTER_SYSTEMS dict should exist."""
        assert CONDENSED_MATTER_SYSTEMS is not None
        assert isinstance(CONDENSED_MATTER_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(CONDENSED_MATTER_SYSTEMS) >= 5

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = [
            "ising_2d_critical",
            "quantum_hall_nu1",
            "anderson_insulator",
            "topological_insulator_bi2se3",
        ]
        for name in expected:
            assert name in CONDENSED_MATTER_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in CONDENSED_MATTER_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "temperature")
            assert hasattr(system, "critical_temperature")
            assert hasattr(system, "dimension")


class TestPhaseTheta:
    """Test phase transition theta calculation."""

    def test_below_tc_positive_theta(self):
        """T < T_c -> positive theta."""
        theta = compute_phase_theta(1.0, 2.0)
        assert theta > 0

    def test_at_tc_zero_theta(self):
        """T = T_c -> theta = 0."""
        theta = compute_phase_theta(2.0, 2.0)
        assert theta == 0.0

    def test_above_tc_zero_theta(self):
        """T > T_c -> theta = 0."""
        theta = compute_phase_theta(3.0, 2.0)
        assert theta == 0.0

    def test_deep_below_tc(self):
        """T << T_c -> theta ~ 1."""
        theta = compute_phase_theta(0.1, 10.0)
        assert theta > 0.9

    def test_zero_tc_returns_zero(self):
        """T_c = 0 -> theta = 0."""
        theta = compute_phase_theta(1.0, 0.0)
        assert theta == 0.0

    def test_theta_always_in_range(self):
        """Theta should always be in [0, 1]."""
        test_cases = [
            (0.5, 1.0),
            (1.0, 0.5),  # Above T_c
            (0.0, 1.0),  # T = 0
            (100.0, 50.0),  # Above T_c
        ]
        for t, tc in test_cases:
            theta = compute_phase_theta(t, tc)
            assert 0.0 <= theta <= 1.0, f"theta={theta} for T={t}, T_c={tc}"


class TestOrderTheta:
    """Test order parameter theta calculation."""

    def test_full_order(self):
        """Order parameter = 1 -> theta = 1."""
        theta = compute_order_theta(1.0)
        assert theta == 1.0

    def test_no_order(self):
        """Order parameter = 0 -> theta = 0."""
        theta = compute_order_theta(0.0)
        assert theta == 0.0

    def test_partial_order(self):
        """Order parameter = 0.5 -> theta = 0.5."""
        theta = compute_order_theta(0.5)
        assert theta == pytest.approx(0.5)

    def test_clipping(self):
        """Values outside [0,1] should be clipped."""
        assert compute_order_theta(1.5) == 1.0
        assert compute_order_theta(-0.5) == 0.0


class TestLocalizationTheta:
    """Test Anderson localization theta calculation."""

    def test_clean_system(self):
        """W = 0 -> theta = 1 (extended states)."""
        theta = compute_localization_theta(0.0)
        assert theta == 1.0

    def test_critical_disorder_3d(self):
        """W = W_c -> theta ~ 0 in 3D."""
        theta = compute_localization_theta(ANDERSON_WC_3D, dimension=3)
        assert theta == pytest.approx(0.0, abs=0.01)

    def test_localized_3d(self):
        """W > W_c -> theta = 0 in 3D."""
        theta = compute_localization_theta(20.0, dimension=3)
        assert theta == 0.0

    def test_weak_disorder_3d(self):
        """Weak disorder -> high theta."""
        theta = compute_localization_theta(2.0, dimension=3)
        assert theta > 0.8

    def test_1d_localization(self):
        """Any disorder localizes in 1D."""
        theta_clean = compute_localization_theta(0.0, dimension=1)
        theta_disordered = compute_localization_theta(5.0, dimension=1)
        assert theta_clean == 1.0
        assert theta_disordered < theta_clean

    def test_2d_localization(self):
        """Any disorder localizes in 2D."""
        theta_disordered = compute_localization_theta(3.0, dimension=2)
        assert 0.0 < theta_disordered < 1.0


class TestTransportTheta:
    """Test transport theta calculation."""

    def test_zero_conductivity(self):
        """Zero conductivity -> theta = 0."""
        theta = compute_transport_theta(0.0)
        assert theta == 0.0

    def test_good_conductor(self):
        """High conductivity -> high theta."""
        theta = compute_transport_theta(1e6)  # Metallic
        assert theta == 1.0  # Capped at 1

    def test_poor_conductor(self):
        """Low conductivity -> low theta."""
        theta = compute_transport_theta(1e-10)
        assert theta < 0.1


class TestHallTheta:
    """Test quantum Hall theta calculation."""

    def test_integer_filling(self):
        """nu = 1 -> high theta."""
        theta = compute_hall_theta(1.0)
        assert theta > 0.9

    def test_near_integer(self):
        """nu ~ 1 -> high theta."""
        theta = compute_hall_theta(0.95)
        assert theta >= 0.5

    def test_laughlin_fraction(self):
        """nu = 1/3 -> high theta (FQHE)."""
        theta = compute_hall_theta(1/3)
        assert theta > 0.7

    def test_zero_filling(self):
        """nu = 0 -> theta = 0."""
        theta = compute_hall_theta(0.0)
        assert theta == 0.0

    def test_non_quantized(self):
        """Random nu -> low theta."""
        theta = compute_hall_theta(0.15)  # Not close to integer or common fraction
        assert 0.0 < theta < 0.2


class TestCorrelationTheta:
    """Test correlation length theta calculation."""

    def test_spanning_correlation(self):
        """xi >= L -> theta = 1."""
        theta = compute_correlation_theta(1e-6, 1e-6)
        assert theta == 1.0

    def test_short_correlation(self):
        """xi << L -> low theta."""
        theta = compute_correlation_theta(1e-9, 1e-6)
        assert theta < 0.01

    def test_zero_system_size(self):
        """L = 0 -> theta = 0."""
        theta = compute_correlation_theta(1e-6, 0.0)
        assert theta == 0.0


class TestTopologicalTheta:
    """Test topological phase theta calculation."""

    def test_trivial_phase(self):
        """Z2 = 0 -> lower theta."""
        theta = compute_topological_theta(0.1, 1.0, z2_invariant=0)
        assert theta < 0.5

    def test_topological_phase(self):
        """Z2 = 1 -> higher theta."""
        theta = compute_topological_theta(0.1, 1.0, z2_invariant=1)
        assert theta > 0.5

    def test_large_gap(self):
        """Large gap -> higher theta."""
        theta_small = compute_topological_theta(0.1, 1.0, z2_invariant=1)
        theta_large = compute_topological_theta(0.4, 1.0, z2_invariant=1)
        assert theta_large > theta_small


class TestUnifiedTheta:
    """Test unified theta calculation for systems."""

    def test_all_systems_have_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in CONDENSED_MATTER_SYSTEMS.items():
            theta = compute_condensed_matter_theta(system)
            assert 0.0 <= theta <= 1.0, f"{name}: theta={theta}"

    def test_ordered_higher_than_disordered(self):
        """Below T_c should have higher theta than above T_c."""
        below = CONDENSED_MATTER_SYSTEMS["ising_2d_below_tc"]
        above = CONDENSED_MATTER_SYSTEMS["ising_2d_above_tc"]

        theta_below = compute_condensed_matter_theta(below)
        theta_above = compute_condensed_matter_theta(above)

        assert theta_below > theta_above

    def test_qhe_high_theta(self):
        """Quantum Hall state should have high theta."""
        qhe = CONDENSED_MATTER_SYSTEMS["quantum_hall_nu1"]
        theta = compute_condensed_matter_theta(qhe)
        assert theta > 0.5


class TestPhaseRegimeClassification:
    """Test phase regime classification."""

    def test_disordered(self):
        """theta < 0.2 -> DISORDERED."""
        assert classify_phase_regime(0.1) == PhaseRegime.DISORDERED

    def test_fluctuating(self):
        """0.2 <= theta < 0.4 -> FLUCTUATING."""
        assert classify_phase_regime(0.3) == PhaseRegime.FLUCTUATING

    def test_critical(self):
        """0.4 <= theta < 0.6 -> CRITICAL."""
        assert classify_phase_regime(0.5) == PhaseRegime.CRITICAL

    def test_ordered(self):
        """0.6 <= theta < 0.8 -> ORDERED."""
        assert classify_phase_regime(0.7) == PhaseRegime.ORDERED

    def test_fully_ordered(self):
        """theta >= 0.8 -> FULLY_ORDERED."""
        assert classify_phase_regime(0.9) == PhaseRegime.FULLY_ORDERED


class TestTransportClassification:
    """Test transport regime classification."""

    def test_insulating(self):
        """theta < 0.1 -> INSULATING."""
        assert classify_transport(0.05) == TransportType.INSULATING

    def test_hopping(self):
        """0.1 <= theta < 0.3 -> HOPPING."""
        assert classify_transport(0.2) == TransportType.HOPPING

    def test_diffusive(self):
        """0.3 <= theta < 0.7 -> DIFFUSIVE."""
        assert classify_transport(0.5) == TransportType.DIFFUSIVE

    def test_ballistic(self):
        """theta >= 0.7 -> BALLISTIC."""
        assert classify_transport(0.8) == TransportType.BALLISTIC


class TestDisorderClassification:
    """Test disorder level classification."""

    def test_clean(self):
        """W < 1 -> CLEAN."""
        assert classify_disorder(0.5) == DisorderLevel.CLEAN

    def test_weakly_disordered(self):
        """1 < W < 5 -> WEAKLY_DISORDERED."""
        assert classify_disorder(3.0) == DisorderLevel.WEAKLY_DISORDERED

    def test_critical(self):
        """W ~ W_c -> CRITICAL_DISORDER."""
        assert classify_disorder(16.0, dimension=3) == DisorderLevel.CRITICAL_DISORDER

    def test_localized(self):
        """W >> W_c -> ANDERSON_LOCALIZED."""
        assert classify_disorder(25.0, dimension=3) == DisorderLevel.ANDERSON_LOCALIZED


class TestTopologicalClassification:
    """Test topological phase classification."""

    def test_trivial(self):
        """Z2 = 0 -> TRIVIAL."""
        assert classify_topological(0) == TopologicalPhase.TRIVIAL

    def test_qsh_2d(self):
        """Z2 = 1 in 2D with TRS -> QUANTUM_SPIN_HALL."""
        assert classify_topological(1, dimension=2, has_time_reversal=True) == \
               TopologicalPhase.QUANTUM_SPIN_HALL

    def test_strong_ti_3d(self):
        """Z2 = 1 in 3D -> STRONG_TI."""
        assert classify_topological(1, dimension=3) == TopologicalPhase.STRONG_TI


class TestPhysicalConstants:
    """Test that physical constants are correct."""

    def test_boltzmann(self):
        """K_B should be Boltzmann constant."""
        assert K_B == pytest.approx(1.380649e-23, rel=1e-6)

    def test_electron_charge(self):
        """E_CHARGE should be elementary charge."""
        assert E_CHARGE == pytest.approx(1.602176634e-19, rel=1e-6)

    def test_planck(self):
        """H_PLANCK should be Planck constant."""
        assert H_PLANCK == pytest.approx(6.62607015e-34, rel=1e-6)

    def test_conductance_quantum(self):
        """G_0 should be e^2/h."""
        expected = E_CHARGE**2 / H_PLANCK
        assert G_0 == pytest.approx(expected, rel=1e-6)

    def test_flux_quantum(self):
        """PHI_0 should be h/2e."""
        expected = H_PLANCK / (2 * E_CHARGE)
        assert PHI_0 == pytest.approx(expected, rel=1e-6)

    def test_ising_2d_tc(self):
        """Ising 2D T_c should be 2/ln(1+sqrt(2))."""
        expected = 2.0 / np.log(1 + np.sqrt(2))
        assert ISING_2D_TC == pytest.approx(expected, rel=1e-6)


class TestSystemDataclass:
    """Test CondensedMatterSystem dataclass."""

    def test_create_minimal_system(self):
        """Should create system with required parameters."""
        system = CondensedMatterSystem(
            name="Test",
            temperature=10.0,
            critical_temperature=5.0,
        )
        assert system.name == "Test"
        assert system.temperature == 10.0

    def test_reduced_temperature(self):
        """reduced_temperature should be (T - T_c) / T_c."""
        system = CondensedMatterSystem(
            name="Test",
            temperature=3.0,
            critical_temperature=2.0,
        )
        assert system.reduced_temperature == pytest.approx(0.5)

    def test_is_ordered_property(self):
        """is_ordered should check T < T_c."""
        below = CondensedMatterSystem(
            name="Below",
            temperature=1.0,
            critical_temperature=2.0,
        )
        above = CondensedMatterSystem(
            name="Above",
            temperature=3.0,
            critical_temperature=2.0,
        )
        assert below.is_ordered is True
        assert above.is_ordered is False

    def test_is_localized_property(self):
        """is_localized should check disorder vs W_c."""
        metal = CondensedMatterSystem(
            name="Metal",
            temperature=10.0,
            critical_temperature=0.0,
            disorder_strength=5.0,
            dimension=3,
        )
        insulator = CondensedMatterSystem(
            name="Insulator",
            temperature=10.0,
            critical_temperature=0.0,
            disorder_strength=20.0,
            dimension=3,
        )
        assert metal.is_localized is False
        assert insulator.is_localized is True


class TestEnums:
    """Test enum definitions."""

    def test_phase_regimes(self):
        """All phase regimes should be defined."""
        assert PhaseRegime.DISORDERED.value == "disordered"
        assert PhaseRegime.CRITICAL.value == "critical"
        assert PhaseRegime.ORDERED.value == "ordered"

    def test_transport_types(self):
        """All transport types should be defined."""
        assert TransportType.INSULATING.value == "insulating"
        assert TransportType.BALLISTIC.value == "ballistic"
        assert TransportType.QUANTUM_HALL.value == "quantum_hall"

    def test_disorder_levels(self):
        """All disorder levels should be defined."""
        assert DisorderLevel.CLEAN.value == "clean"
        assert DisorderLevel.ANDERSON_LOCALIZED.value == "anderson_localized"

    def test_topological_phases(self):
        """All topological phases should be defined."""
        assert TopologicalPhase.TRIVIAL.value == "trivial"
        assert TopologicalPhase.STRONG_TI.value == "strong_topological"

    def test_universality_classes(self):
        """All universality classes should be defined."""
        assert CriticalExponentClass.ISING_2D.value == "ising_2d"
        assert CriticalExponentClass.MEAN_FIELD.value == "mean_field"
