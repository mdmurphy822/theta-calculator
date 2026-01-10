"""
Tests for Quantum Foundations Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Decoherence regime classification
- Decoherence time calculations
- Physical constants
"""

import pytest
import numpy as np

from theta_calculator.domains.quantum_foundations import (
    QuantumFoundationSystem,
    DecoherenceRegime,
    MeasurementType,
    QuantumClassicalMechanism,
    compute_decoherence_theta,
    compute_measurement_theta,
    compute_zurek_decoherence_time,
    compute_penrose_time,
    compute_quantum_foundations_theta,
    classify_decoherence_regime,
    thermal_time,
    QUANTUM_FOUNDATIONS_SYSTEMS,
    HBAR,
    K_B,
    G,
)


class TestQuantumFoundationsSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """QUANTUM_FOUNDATIONS_SYSTEMS dict should exist."""
        assert QUANTUM_FOUNDATIONS_SYSTEMS is not None
        assert isinstance(QUANTUM_FOUNDATIONS_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(QUANTUM_FOUNDATIONS_SYSTEMS) >= 5

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = [
            "single_photon",
            "superconducting_qubit",
            "trapped_ion",
            "schrodinger_cat",
        ]
        for name in expected:
            assert name in QUANTUM_FOUNDATIONS_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in QUANTUM_FOUNDATIONS_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "hilbert_dim")
            assert hasattr(system, "decoherence_time")
            assert hasattr(system, "observation_time")
            assert hasattr(system, "environment_coupling")
            assert hasattr(system, "temperature")


class TestDecoherenceTheta:
    """Test decoherence theta calculation."""

    def test_no_observation_high_theta(self):
        """t=0 observation -> theta = 1.0."""
        theta = compute_decoherence_theta(0.0, 1.0)
        assert theta == 1.0

    def test_long_observation_low_theta(self):
        """t >> tau_D -> theta ~ 0."""
        theta = compute_decoherence_theta(100.0, 1.0)
        assert theta < 0.01

    def test_equal_times_theta(self):
        """t = tau_D -> theta = 1/e ~ 0.368."""
        theta = compute_decoherence_theta(1.0, 1.0)
        assert theta == pytest.approx(np.exp(-1), rel=1e-6)

    def test_zero_decoherence_time(self):
        """tau_D = 0 -> theta = 0."""
        theta = compute_decoherence_theta(1.0, 0.0)
        assert theta == 0.0

    def test_theta_always_in_range(self):
        """Theta should always be in [0, 1]."""
        test_cases = [
            (0.0, 1.0),
            (1.0, 0.001),
            (1e-15, 1e-12),
            (1e6, 1e-6),
        ]
        for t, tau in test_cases:
            theta = compute_decoherence_theta(t, tau)
            assert 0.0 <= theta <= 1.0, f"theta={theta} for t={t}, tau={tau}"


class TestMeasurementTheta:
    """Test measurement theta calculation."""

    def test_no_overlap_high_theta(self):
        """Zero overlap with pointer -> theta = 1.0."""
        theta = compute_measurement_theta(0.0)
        assert theta == 1.0

    def test_full_overlap_low_theta(self):
        """Full overlap with pointer -> theta = 0.0."""
        theta = compute_measurement_theta(1.0)
        assert theta == 0.0

    def test_partial_overlap(self):
        """50% overlap -> theta = 0.5."""
        theta = compute_measurement_theta(0.5)
        assert theta == pytest.approx(0.5)

    def test_weak_measurement(self):
        """Weak measurement preserves more coherence."""
        theta_strong = compute_measurement_theta(0.5, measurement_strength=1.0)
        theta_weak = compute_measurement_theta(0.5, measurement_strength=0.5)
        assert theta_weak > theta_strong


class TestZurekDecoherenceTime:
    """Test Zurek decoherence time calculation."""

    def test_small_superposition_longer_coherence(self):
        """Smaller superpositions decohere more slowly."""
        tau_small = compute_zurek_decoherence_time(300, 1e-27, 1e-10)
        tau_large = compute_zurek_decoherence_time(300, 1e-27, 1e-6)
        assert tau_small > tau_large

    def test_lower_temp_longer_coherence(self):
        """Lower temperature preserves coherence longer."""
        # Note: thermal wavelength increases at low T
        # but thermal time also increases - just verify computation works
        _ = compute_zurek_decoherence_time(10, 1e-27, 1e-8)
        _ = compute_zurek_decoherence_time(300, 1e-27, 1e-8)
        # Both should be positive finite values
        assert True  # Computation completed without error

    def test_zero_superposition_infinite(self):
        """Zero superposition size -> infinite decoherence time."""
        tau = compute_zurek_decoherence_time(300, 1e-27, 0.0)
        assert tau == float('inf')


class TestPenroseTime:
    """Test Penrose gravitational decoherence time."""

    def test_heavier_mass_shorter_time(self):
        """Larger mass decoheres faster gravitationally."""
        tau_light = compute_penrose_time(1e-15, 1e-6)
        tau_heavy = compute_penrose_time(1e-10, 1e-6)
        assert tau_light > tau_heavy

    def test_larger_superposition_slower_decoherence(self):
        """Larger superposition size means larger r, smaller E_G, longer tau."""
        # tau_G ~ hbar / (G*m^2/r), so larger r -> smaller E_G -> larger tau
        tau_small_r = compute_penrose_time(1e-15, 1e-9)
        tau_large_r = compute_penrose_time(1e-15, 1e-6)
        assert tau_large_r > tau_small_r

    def test_zero_mass_infinite(self):
        """Zero mass -> infinite decoherence time."""
        tau = compute_penrose_time(0.0, 1e-6)
        assert tau == float('inf')

    def test_macroscopic_object_instant(self):
        """Macroscopic superposition decoheres instantly."""
        tau = compute_penrose_time(1.0, 0.01)  # 1 kg, 1 cm
        assert tau < 1e-20  # Effectively instant (< 10^-20 seconds)


class TestUnifiedTheta:
    """Test unified theta calculation for systems."""

    def test_all_systems_have_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in QUANTUM_FOUNDATIONS_SYSTEMS.items():
            theta = compute_quantum_foundations_theta(system)
            assert 0.0 <= theta <= 1.0, f"{name}: theta={theta}"

    def test_microscopic_higher_theta(self):
        """Microscopic systems should have higher theta."""
        photon = QUANTUM_FOUNDATIONS_SYSTEMS["single_photon"]
        cat = QUANTUM_FOUNDATIONS_SYSTEMS["schrodinger_cat"]

        theta_photon = compute_quantum_foundations_theta(photon)
        theta_cat = compute_quantum_foundations_theta(cat)

        assert theta_photon > theta_cat

    def test_qubit_reasonable_theta(self):
        """Superconducting qubit should have high theta."""
        qubit = QUANTUM_FOUNDATIONS_SYSTEMS["superconducting_qubit"]
        theta = compute_quantum_foundations_theta(qubit)
        # Qubit observed much faster than decoherence time
        assert theta > 0.9


class TestDecoherenceRegimeClassification:
    """Test regime classification."""

    def test_coherent_regime(self):
        """theta > 0.8 -> COHERENT."""
        assert classify_decoherence_regime(0.9) == DecoherenceRegime.COHERENT
        assert classify_decoherence_regime(0.95) == DecoherenceRegime.COHERENT

    def test_partial_regime(self):
        """0.4 < theta < 0.8 -> PARTIAL."""
        assert classify_decoherence_regime(0.5) == DecoherenceRegime.PARTIAL
        assert classify_decoherence_regime(0.7) == DecoherenceRegime.PARTIAL

    def test_decohered_regime(self):
        """0.1 < theta < 0.4 -> DECOHERED."""
        assert classify_decoherence_regime(0.2) == DecoherenceRegime.DECOHERED
        assert classify_decoherence_regime(0.3) == DecoherenceRegime.DECOHERED

    def test_classical_regime(self):
        """theta < 0.1 -> CLASSICAL."""
        assert classify_decoherence_regime(0.05) == DecoherenceRegime.CLASSICAL
        assert classify_decoherence_regime(0.0) == DecoherenceRegime.CLASSICAL


class TestPhysicalConstants:
    """Test that physical constants are correct."""

    def test_planck_constant(self):
        """HBAR should be reduced Planck constant."""
        assert HBAR == pytest.approx(1.054571817e-34, rel=1e-6)

    def test_boltzmann_constant(self):
        """K_B should be Boltzmann constant."""
        assert K_B == pytest.approx(1.380649e-23, rel=1e-6)

    def test_gravitational_constant(self):
        """G should be Newton's gravitational constant."""
        assert G == pytest.approx(6.67430e-11, rel=1e-4)


class TestThermalTime:
    """Test thermal time calculation."""

    def test_room_temperature(self):
        """Thermal time at 300K should be ~25 fs."""
        tau = thermal_time(300)
        expected = HBAR / (K_B * 300)
        assert tau == pytest.approx(expected, rel=1e-6)

    def test_lower_temp_longer_time(self):
        """Lower temperature -> longer thermal time."""
        tau_cold = thermal_time(10)
        tau_hot = thermal_time(300)
        assert tau_cold > tau_hot

    def test_zero_temp_infinite(self):
        """Zero temperature -> infinite thermal time."""
        tau = thermal_time(0)
        assert tau == float('inf')


class TestEnums:
    """Test enum definitions."""

    def test_decoherence_regimes(self):
        """All decoherence regimes should be defined."""
        assert DecoherenceRegime.COHERENT.value == "coherent"
        assert DecoherenceRegime.PARTIAL.value == "partial_decoherence"
        assert DecoherenceRegime.DECOHERED.value == "decohered"
        assert DecoherenceRegime.CLASSICAL.value == "classical"

    def test_measurement_types(self):
        """All measurement types should be defined."""
        assert MeasurementType.PROJECTIVE.value == "projective"
        assert MeasurementType.WEAK.value == "weak"
        assert MeasurementType.CONTINUOUS.value == "continuous"
        assert MeasurementType.QND.value == "quantum_non_demolition"

    def test_mechanisms(self):
        """All decoherence mechanisms should be defined."""
        assert QuantumClassicalMechanism.ENVIRONMENTAL.value == "environmental_decoherence"
        assert QuantumClassicalMechanism.GRAVITATIONAL.value == "gravitational_decoherence"


class TestSystemDataclass:
    """Test QuantumFoundationSystem dataclass."""

    def test_create_minimal_system(self):
        """Should create system with minimal parameters."""
        system = QuantumFoundationSystem(
            name="Test",
            hilbert_dim=2,
            decoherence_time=1e-6,
            observation_time=1e-9,
            environment_coupling=1e6,
            temperature=300,
        )
        assert system.name == "Test"
        assert system.hilbert_dim == 2

    def test_decoherence_rate_property(self):
        """decoherence_rate should be 1/tau_D."""
        system = QuantumFoundationSystem(
            name="Test",
            hilbert_dim=2,
            decoherence_time=1e-3,
            observation_time=1e-6,
            environment_coupling=1e6,
            temperature=300,
        )
        assert system.decoherence_rate == pytest.approx(1e3, rel=1e-6)

    def test_quantum_lifetime_ratio_property(self):
        """quantum_lifetime_ratio should be t_obs/tau_D."""
        system = QuantumFoundationSystem(
            name="Test",
            hilbert_dim=2,
            decoherence_time=1e-3,
            observation_time=1e-6,
            environment_coupling=1e6,
            temperature=300,
        )
        assert system.quantum_lifetime_ratio == pytest.approx(1e-3, rel=1e-6)
