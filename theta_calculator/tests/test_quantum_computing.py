"""
Tests for quantum_computing module.

Tests quantum computing theta calculations using coherence times,
error rates, and error correction thresholds.
"""

import pytest
import numpy as np

from theta_calculator.domains.quantum_computing import (
    QUANTUM_HARDWARE,
    QubitSystem,
    QubitType,
    CoherenceRegime,
    compute_coherence_theta,
    compute_error_threshold_theta,
    compute_quantum_computing_theta,
    compute_logical_theta,
    classify_coherence_regime,
    amplitude_damping,
    phase_damping,
    depolarizing,
    compute_T2_star,
    logical_error_rate,
    required_code_distance,
    threshold_crossing_timeline,
)


class TestQuantumHardware:
    """Test the predefined quantum hardware systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "google_sycamore", "google_willow", "ibm_heron",
            "ionq_forte", "quantinuum_h2", "nv_center_lab",
            "neutral_atom_quera", "noisy_classical"
        ]
        for name in expected:
            assert name in QUANTUM_HARDWARE, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in QUANTUM_HARDWARE.items():
            assert isinstance(system, QubitSystem)
            assert system.name
            assert isinstance(system.qubit_type, QubitType)
            assert system.n_qubits > 0
            assert system.T1 > 0
            assert system.T2 > 0
            assert system.gate_time > 0
            assert 0 < system.error_rate <= 1


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in QUANTUM_HARDWARE.items():
            theta = compute_quantum_computing_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_noisy_system_low_theta(self):
        """Highly noisy system should have low theta."""
        noisy = QUANTUM_HARDWARE["noisy_classical"]
        theta = compute_quantum_computing_theta(noisy)
        assert theta < 0.5, f"Noisy system should have low theta: {theta}"

    def test_high_fidelity_system_high_theta(self):
        """High fidelity system should have high theta."""
        good = QUANTUM_HARDWARE["quantinuum_h2"]
        theta = compute_quantum_computing_theta(good)
        assert theta > 0.5, f"High fidelity system should have high theta: {theta}"


class TestCoherenceTheta:
    """Test coherence-based theta calculation."""

    def test_long_coherence_high_theta(self):
        """Long coherence time should give high theta."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRAPPED_ION,
            n_qubits=1,
            T1=10.0,       # 10 seconds
            T2=1.0,        # 1 second
            gate_time=1e-6,  # 1 microsecond
            error_rate=0.001,
        )
        theta = compute_coherence_theta(system)
        assert theta > 0.99, f"Long coherence should give theta ~1: {theta}"

    def test_short_coherence_low_theta(self):
        """Short coherence time should give low theta."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=1,
            T1=1e-9,       # 1 nanosecond
            T2=1e-9,
            gate_time=1e-6,  # Gate longer than coherence!
            error_rate=0.1,
        )
        theta = compute_coherence_theta(system)
        assert theta < 0.01, f"Short coherence should give low theta: {theta}"


class TestErrorThresholdTheta:
    """Test error threshold theta calculation."""

    def test_below_threshold_positive_theta(self):
        """Below threshold should give positive theta."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=1,
            T1=100e-6,
            T2=50e-6,
            gate_time=20e-9,
            error_rate=0.001,    # 0.1%
            error_threshold=0.01  # 1%
        )
        theta = compute_error_threshold_theta(system)
        assert theta > 0.5, f"Below threshold should give high theta: {theta}"

    def test_above_threshold_zero_theta(self):
        """Above threshold should give zero theta."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=1,
            T1=100e-6,
            T2=50e-6,
            gate_time=20e-9,
            error_rate=0.05,    # 5%
            error_threshold=0.01  # 1%
        )
        theta = compute_error_threshold_theta(system)
        assert theta == 0.0, f"Above threshold should give theta=0: {theta}"


class TestDecoherenceChannels:
    """Test decoherence channel operations."""

    def test_amplitude_damping_pure_state(self):
        """Amplitude damping should move |1> toward |0>."""
        rho = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
        rho_damped = amplitude_damping(rho, gamma=0.5)
        # Population should transfer to |0>
        assert rho_damped[0, 0] > 0

    def test_amplitude_damping_preserves_trace(self):
        """Amplitude damping should preserve trace."""
        rho = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=complex)
        rho_damped = amplitude_damping(rho, gamma=0.3)
        assert abs(np.trace(rho_damped) - 1.0) < 1e-10

    def test_phase_damping_preserves_diagonal(self):
        """Phase damping should preserve diagonal elements."""
        rho = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=complex)
        rho_damped = phase_damping(rho, gamma=0.5)
        assert abs(rho_damped[0, 0] - 0.3) < 1e-10
        assert abs(rho_damped[1, 1] - 0.7) < 1e-10

    def test_phase_damping_reduces_coherence(self):
        """Phase damping should reduce off-diagonal elements."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        rho_damped = phase_damping(rho, gamma=0.5)
        assert abs(rho_damped[0, 1]) < abs(rho[0, 1])

    def test_depolarizing_at_p_one(self):
        """Full depolarizing should give maximally mixed state."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
        rho_depol = depolarizing(rho, p=1.0)
        expected = np.eye(2) / 2
        assert np.allclose(rho_depol, expected)

    def test_depolarizing_at_p_zero(self):
        """No depolarizing should leave state unchanged."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_depol = depolarizing(rho, p=0.0)
        assert np.allclose(rho_depol, rho)


class TestT2Star:
    """Test T2* calculation."""

    def test_t2_star_limit(self):
        """T2* should be at most 2*T1."""
        T1 = 100e-6
        T_phi = 1000e-6  # Very long pure dephasing
        T2_star = compute_T2_star(T1, T_phi)
        assert T2_star <= 2 * T1 + 1e-15

    def test_t2_star_formula(self):
        """Test T2* formula: 1/T2* = 1/(2*T1) + 1/T_phi."""
        T1 = 100e-6
        T_phi = 50e-6
        T2_star = compute_T2_star(T1, T_phi)
        expected = 1.0 / (1.0 / (2 * T1) + 1.0 / T_phi)
        assert abs(T2_star - expected) < 1e-15


class TestLogicalErrorRate:
    """Test logical error rate calculation."""

    def test_below_threshold_suppression(self):
        """Below threshold, logical error should be suppressed."""
        logical = logical_error_rate(
            physical_error=0.001,
            threshold=0.01,
            code_distance=5
        )
        # Should be at most equal to physical error (suppressed)
        assert logical <= 0.001 + 1e-10

    def test_above_threshold_no_suppression(self):
        """Above threshold, errors should accumulate."""
        logical = logical_error_rate(
            physical_error=0.02,  # Above threshold
            threshold=0.01,
            code_distance=5
        )
        # Should be worse than physical error
        assert logical >= 0.02

    def test_higher_distance_better(self):
        """Higher code distance should give lower logical error."""
        logical_d3 = logical_error_rate(0.001, 0.01, code_distance=3)
        logical_d5 = logical_error_rate(0.001, 0.01, code_distance=5)
        logical_d7 = logical_error_rate(0.001, 0.01, code_distance=7)
        assert logical_d3 > logical_d5 > logical_d7


class TestRequiredCodeDistance:
    """Test required code distance calculation."""

    def test_lower_error_needs_less_distance(self):
        """Lower physical error should need smaller code distance."""
        d_high_error = required_code_distance(1e-10, 0.005, 0.01)
        d_low_error = required_code_distance(1e-10, 0.001, 0.01)
        assert d_low_error < d_high_error

    def test_above_threshold_impossible(self):
        """Above threshold, required distance should be infinite."""
        d = required_code_distance(1e-10, 0.02, 0.01)
        assert d == float('inf')

    def test_returns_odd_integer(self):
        """Required distance should be an odd integer."""
        d = required_code_distance(1e-10, 0.001, 0.01)
        assert isinstance(d, int)
        assert d % 2 == 1


class TestCoherenceRegimeClassification:
    """Test coherence regime classification."""

    def test_highly_coherent_regime(self):
        """Theta > 0.9 should be HIGHLY_COHERENT."""
        assert classify_coherence_regime(0.95) == CoherenceRegime.HIGHLY_COHERENT

    def test_coherent_regime(self):
        """0.7 < theta < 0.9 should be COHERENT."""
        assert classify_coherence_regime(0.8) == CoherenceRegime.COHERENT

    def test_partially_coherent_regime(self):
        """0.3 < theta < 0.7 should be PARTIALLY_COHERENT."""
        assert classify_coherence_regime(0.5) == CoherenceRegime.PARTIALLY_COHERENT

    def test_mostly_decoherent_regime(self):
        """0.1 < theta < 0.3 should be MOSTLY_DECOHERENT."""
        assert classify_coherence_regime(0.2) == CoherenceRegime.MOSTLY_DECOHERENT

    def test_classical_regime(self):
        """Theta < 0.1 should be CLASSICAL."""
        assert classify_coherence_regime(0.05) == CoherenceRegime.CLASSICAL


class TestQubitSystemProperties:
    """Test QubitSystem properties."""

    def test_coherence_ratio(self):
        """Test coherence ratio calculation."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=1,
            T1=100e-6,
            T2=50e-6,
            gate_time=20e-9,
            error_rate=0.01,
        )
        # T_eff = min(T1, T2) = 50e-6
        # ratio = T_eff / gate_time = 50e-6 / 20e-9 = 2500
        assert abs(system.coherence_ratio - 2500) < 1

    def test_is_below_threshold(self):
        """Test is_below_threshold property."""
        below = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=1,
            T1=100e-6, T2=50e-6, gate_time=20e-9,
            error_rate=0.001,
            error_threshold=0.01
        )
        above = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=1,
            T1=100e-6, T2=50e-6, gate_time=20e-9,
            error_rate=0.02,
            error_threshold=0.01
        )
        assert below.is_below_threshold is True
        assert above.is_below_threshold is False


class TestLogicalTheta:
    """Test logical qubit theta calculation."""

    def test_below_threshold_with_qec(self):
        """Below threshold with QEC should have positive theta."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=100,
            T1=100e-6, T2=50e-6, gate_time=20e-9,
            error_rate=0.001,
            error_threshold=0.01,
            code_distance=7
        )
        theta = compute_logical_theta(system)
        assert theta > 0

    def test_above_threshold_zero(self):
        """Above threshold should give theta = 0."""
        system = QubitSystem(
            name="Test",
            qubit_type=QubitType.TRANSMON,
            n_qubits=100,
            T1=100e-6, T2=50e-6, gate_time=20e-9,
            error_rate=0.02,
            error_threshold=0.01,
            code_distance=7
        )
        theta = compute_logical_theta(system)
        assert theta == 0.0


class TestThresholdCrossingTimeline:
    """Test historical timeline data."""

    def test_timeline_has_entries(self):
        """Timeline should have historical entries."""
        timeline = threshold_crossing_timeline()
        assert len(timeline) > 0

    def test_google_willow_below_threshold(self):
        """Google Willow (2024) should be below threshold."""
        timeline = threshold_crossing_timeline()
        willow = timeline["2024_google_willow"]
        assert willow["below_threshold"] is True
        assert willow["physical_error"] < willow["threshold"]


class TestThetaOrdering:
    """Test theta ordering across systems."""

    def test_trapped_ion_higher_than_transmon(self):
        """Trapped ion systems generally have higher theta."""
        transmon = QUANTUM_HARDWARE["google_sycamore"]
        ion = QUANTUM_HARDWARE["ionq_forte"]
        theta_transmon = compute_quantum_computing_theta(transmon)
        theta_ion = compute_quantum_computing_theta(ion)
        # Trapped ions have longer coherence
        assert theta_ion > theta_transmon


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
