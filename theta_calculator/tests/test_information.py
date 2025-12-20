"""
Tests for information module.

Tests information theory theta calculations using Shannon and
Von Neumann entropy measures.
"""

import pytest
import numpy as np

from theta_calculator.domains.information import (
    INFORMATION_SYSTEMS,
    InformationSystem,
    InformationRegime,
    compute_information_theta,
    compute_shannon_entropy,
    compute_von_neumann_entropy,
    compute_purity,
    compute_linear_entropy,
    classify_information_regime,
    pure_state_density_matrix,
    maximally_mixed_state,
    thermal_state,
    bell_state,
    partial_trace,
)


class TestInformationSystems:
    """Test the predefined information systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "pure_qubit", "mixed_qubit", "thermal_hot", "thermal_cold",
            "bell_reduced", "fair_coin", "biased_coin", "deterministic",
            "uniform_die", "8_level_mixed"
        ]
        for name in expected:
            assert name in INFORMATION_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in INFORMATION_SYSTEMS.items():
            assert isinstance(system, InformationSystem)
            assert system.name
            assert system.dimension > 0


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in INFORMATION_SYSTEMS.items():
            theta = compute_information_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_pure_qubit_zero_theta(self):
        """Pure qubit should have theta near 0."""
        pure = INFORMATION_SYSTEMS["pure_qubit"]
        theta = compute_information_theta(pure)
        assert theta < 0.1, f"Pure qubit should have low theta: {theta}"

    def test_mixed_qubit_high_theta(self):
        """Maximally mixed qubit should have theta near 1."""
        mixed = INFORMATION_SYSTEMS["mixed_qubit"]
        theta = compute_information_theta(mixed)
        assert theta > 0.9, f"Mixed qubit should have high theta: {theta}"

    def test_deterministic_zero_theta(self):
        """Deterministic system should have theta = 0."""
        det = INFORMATION_SYSTEMS["deterministic"]
        theta = compute_information_theta(det)
        assert theta == 0.0, f"Deterministic should have zero theta: {theta}"

    def test_fair_coin_high_theta(self):
        """Fair coin should have theta = 1."""
        fair = INFORMATION_SYSTEMS["fair_coin"]
        theta = compute_information_theta(fair)
        assert abs(theta - 1.0) < 0.01, f"Fair coin should have theta=1: {theta}"


class TestShannonEntropy:
    """Test Shannon entropy calculation."""

    def test_deterministic_zero_entropy(self):
        """Deterministic distribution has zero entropy."""
        p = np.array([1.0, 0.0])
        H = compute_shannon_entropy(p)
        assert H == 0.0

    def test_fair_coin_one_bit(self):
        """Fair coin flip has 1 bit of entropy."""
        p = np.array([0.5, 0.5])
        H = compute_shannon_entropy(p)
        assert abs(H - 1.0) < 1e-10

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution has maximum entropy."""
        n = 8
        p = np.ones(n) / n
        H = compute_shannon_entropy(p)
        assert abs(H - np.log2(n)) < 1e-10

    def test_biased_distribution(self):
        """Biased distribution has less entropy than uniform."""
        p_uniform = np.array([0.5, 0.5])
        p_biased = np.array([0.9, 0.1])
        H_uniform = compute_shannon_entropy(p_uniform)
        H_biased = compute_shannon_entropy(p_biased)
        assert H_biased < H_uniform

    def test_entropy_non_negative(self):
        """Entropy should always be non-negative."""
        for _ in range(10):
            n = np.random.randint(2, 10)
            p = np.random.random(n)
            p = p / p.sum()
            H = compute_shannon_entropy(p)
            assert H >= 0


class TestVonNeumannEntropy:
    """Test von Neumann entropy calculation."""

    def test_pure_state_zero_entropy(self):
        """Pure state has zero entropy."""
        rho = pure_state_density_matrix(np.array([1, 0]))
        S = compute_von_neumann_entropy(rho)
        assert S < 1e-10

    def test_maximally_mixed_max_entropy(self):
        """Maximally mixed state has log(d) entropy."""
        d = 4
        rho = maximally_mixed_state(d)
        S = compute_von_neumann_entropy(rho)
        assert abs(S - np.log2(d)) < 1e-10

    def test_entropy_non_negative(self):
        """Von Neumann entropy should always be non-negative."""
        for d in [2, 3, 4]:
            rho = maximally_mixed_state(d)
            S = compute_von_neumann_entropy(rho)
            assert S >= 0


class TestPurity:
    """Test purity calculation."""

    def test_pure_state_purity_one(self):
        """Pure state has purity = 1."""
        rho = pure_state_density_matrix(np.array([1, 0]))
        P = compute_purity(rho)
        assert abs(P - 1.0) < 1e-10

    def test_maximally_mixed_purity(self):
        """Maximally mixed state has purity = 1/d."""
        d = 4
        rho = maximally_mixed_state(d)
        P = compute_purity(rho)
        assert abs(P - 1.0/d) < 1e-10

    def test_purity_bounds(self):
        """Purity should be in [1/d, 1]."""
        d = 4
        rho = maximally_mixed_state(d)
        P = compute_purity(rho)
        assert 1.0/d - 1e-10 <= P <= 1.0 + 1e-10


class TestLinearEntropy:
    """Test linear entropy calculation."""

    def test_pure_state_zero_linear_entropy(self):
        """Pure state has zero linear entropy."""
        rho = pure_state_density_matrix(np.array([1, 0]))
        S_L = compute_linear_entropy(rho)
        assert S_L < 1e-10

    def test_maximally_mixed_max_linear_entropy(self):
        """Maximally mixed state has linear entropy = 1."""
        d = 4
        rho = maximally_mixed_state(d)
        S_L = compute_linear_entropy(rho)
        assert abs(S_L - 1.0) < 1e-10


class TestRegimeClassification:
    """Test information regime classification."""

    def test_deterministic_regime(self):
        """Theta < 0.1 should be DETERMINISTIC."""
        assert classify_information_regime(0.05) == InformationRegime.DETERMINISTIC

    def test_low_entropy_regime(self):
        """0.1 <= theta < 0.3 should be LOW_ENTROPY."""
        assert classify_information_regime(0.2) == InformationRegime.LOW_ENTROPY

    def test_moderate_regime(self):
        """0.3 <= theta < 0.7 should be MODERATE."""
        assert classify_information_regime(0.5) == InformationRegime.MODERATE

    def test_high_entropy_regime(self):
        """0.7 <= theta < 0.9 should be HIGH_ENTROPY."""
        assert classify_information_regime(0.8) == InformationRegime.HIGH_ENTROPY

    def test_maximally_mixed_regime(self):
        """theta >= 0.9 should be MAXIMALLY_MIXED."""
        assert classify_information_regime(0.95) == InformationRegime.MAXIMALLY_MIXED


class TestDensityMatrixCreation:
    """Test density matrix creation functions."""

    def test_pure_state_is_hermitian(self):
        """Pure state density matrix should be Hermitian."""
        psi = np.array([1, 1j]) / np.sqrt(2)
        rho = pure_state_density_matrix(psi)
        assert np.allclose(rho, rho.conj().T)

    def test_pure_state_trace_one(self):
        """Pure state density matrix should have trace 1."""
        psi = np.array([1, 0])
        rho = pure_state_density_matrix(psi)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_maximally_mixed_is_identity(self):
        """Maximally mixed state is I/d."""
        d = 4
        rho = maximally_mixed_state(d)
        expected = np.eye(d) / d
        assert np.allclose(rho, expected)


class TestThermalState:
    """Test thermal state creation."""

    def test_high_temp_approaches_mixed(self):
        """High temperature should approach maximally mixed."""
        H = np.diag([0, 1])  # Simple qubit Hamiltonian
        rho = thermal_state(H, temperature=1000)
        mixed = maximally_mixed_state(2)
        assert np.allclose(rho, mixed, atol=0.01)

    def test_low_temp_approaches_ground(self):
        """Low temperature should approach ground state."""
        H = np.diag([0, 1])
        rho = thermal_state(H, temperature=0.01)
        # Should be close to |0><0|
        ground = pure_state_density_matrix(np.array([1, 0]))
        assert np.allclose(rho, ground, atol=0.01)

    def test_thermal_state_is_valid(self):
        """Thermal state should be valid density matrix."""
        H = np.diag([0, 1, 2])
        rho = thermal_state(H, temperature=1.0)
        # Check trace = 1
        assert abs(np.trace(rho) - 1.0) < 1e-10
        # Check Hermitian
        assert np.allclose(rho, rho.conj().T)
        # Check positive semidefinite
        eigenvalues = np.linalg.eigvalsh(rho)
        assert all(ev >= -1e-10 for ev in eigenvalues)


class TestBellStates:
    """Test Bell state creation."""

    def test_bell_phi_plus(self):
        """Test phi_plus Bell state."""
        rho = bell_state("phi_plus")
        # Should be pure state
        P = compute_purity(rho)
        assert abs(P - 1.0) < 1e-10

    def test_all_bell_states_pure(self):
        """All Bell states should be pure."""
        for which in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
            rho = bell_state(which)
            P = compute_purity(rho)
            assert abs(P - 1.0) < 1e-10

    def test_invalid_bell_state_raises(self):
        """Invalid Bell state name should raise."""
        with pytest.raises(ValueError):
            bell_state("invalid")


class TestPartialTrace:
    """Test partial trace operation."""

    def test_bell_reduced_maximally_mixed(self):
        """Reduced Bell state should be maximally mixed."""
        rho_full = bell_state("phi_plus")
        rho_reduced = partial_trace(rho_full, (2, 2), trace_out=1)
        # Should be I/2
        expected = maximally_mixed_state(2)
        assert np.allclose(rho_reduced, expected, atol=1e-10)

    def test_partial_trace_preserves_trace(self):
        """Partial trace should preserve total trace."""
        rho_full = bell_state("phi_plus")
        rho_reduced = partial_trace(rho_full, (2, 2), trace_out=0)
        assert abs(np.trace(rho_reduced) - 1.0) < 1e-10


class TestInformationSystemProperties:
    """Test InformationSystem properties."""

    def test_max_entropy_qubit(self):
        """Qubit max entropy should be 1 bit."""
        system = INFORMATION_SYSTEMS["pure_qubit"]
        assert abs(system.max_entropy - 1.0) < 1e-10

    def test_max_entropy_die(self):
        """6-sided die max entropy should be log2(6)."""
        system = INFORMATION_SYSTEMS["uniform_die"]
        assert abs(system.max_entropy - np.log2(6)) < 1e-10


class TestThetaOrdering:
    """Test that theta increases with mixedness."""

    def test_cold_vs_hot_thermal(self):
        """Cold thermal should have lower theta than hot."""
        cold = INFORMATION_SYSTEMS["thermal_cold"]
        hot = INFORMATION_SYSTEMS["thermal_hot"]
        theta_cold = compute_information_theta(cold)
        theta_hot = compute_information_theta(hot)
        assert theta_cold < theta_hot

    def test_biased_vs_fair_coin(self):
        """Biased coin should have lower theta than fair coin."""
        biased = INFORMATION_SYSTEMS["biased_coin"]
        fair = INFORMATION_SYSTEMS["fair_coin"]
        theta_biased = compute_information_theta(biased)
        theta_fair = compute_information_theta(fair)
        assert theta_biased < theta_fair


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
