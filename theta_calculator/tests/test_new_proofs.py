"""
Tests for new proof modules: uncertainty, hawking, holographic, cosmology.

These tests verify the theta derivations from fundamental physics.
"""

import pytest
import numpy as np

from theta_calculator.proofs import (
    # Uncertainty proofs
    UncertaintyProofs,
    compute_heisenberg_theta,
    compute_energy_time_theta,
    compute_entropic_theta,
    HeisenbergResult,
    # Hawking proofs
    HawkingProofs,
    compute_hawking_theta,
    compute_page_time_theta,
    compute_area_quantization_theta,
    # Holographic proofs
    HolographicProofs,
    compute_rt_theta,
    compute_wedge_theta,
    # Cosmology proofs
    VacuumEnergyProofs,
    compute_vacuum_theta,
    compute_dark_energy_theta,
)


# Physical constants
HBAR = 1.054571817e-34
M_PLANCK = 2.176434e-8


class TestUncertaintyProofs:
    """Tests for uncertainty principle theta derivations."""

    def test_heisenberg_minimum_uncertainty(self):
        """Minimum uncertainty state should give theta = 1."""
        delta_x = 1e-10  # 0.1 nm
        delta_p = HBAR / (2 * delta_x)  # Saturates bound
        result = compute_heisenberg_theta(delta_x, delta_p)
        assert result.theta == pytest.approx(1.0, rel=0.01)

    def test_heisenberg_classical_regime(self):
        """Large uncertainties should give theta ~ 0."""
        delta_x = 1e-3  # 1 mm (macroscopic)
        delta_p = 1e-20  # Very uncertain momentum
        result = compute_heisenberg_theta(delta_x, delta_p)
        assert result.theta < 0.01

    def test_heisenberg_bound_violation_raises(self):
        """Violating Heisenberg bound should raise error."""
        delta_x = 1e-10
        delta_p = HBAR / (4 * delta_x)  # Below minimum
        with pytest.raises(ValueError):
            compute_heisenberg_theta(delta_x, delta_p)

    def test_energy_time_excited_state(self):
        """Excited state with natural linewidth should be quantum."""
        delta_t = 1e-9  # 1 ns lifetime
        delta_E = HBAR / (2 * delta_t)  # Natural linewidth
        result = compute_energy_time_theta(delta_E, delta_t)
        assert result.theta == pytest.approx(1.0, rel=0.01)

    def test_entropic_theta_range(self):
        """Entropic theta should be in [0, 1]."""
        # High entropy = classical
        result_classical = compute_entropic_theta(5.0, 5.0)
        assert 0 <= result_classical.theta <= 1

        # Low entropy = quantum
        result_quantum = compute_entropic_theta(1.5, 1.5)
        assert 0 <= result_quantum.theta <= 1

    def test_uncertainty_proofs_interface(self):
        """Test unified UncertaintyProofs interface."""
        result = UncertaintyProofs.heisenberg(1e-10, 1e-24)
        assert "theta" in result
        assert "proof_type" in result
        assert result["proof_type"] == "heisenberg_uncertainty"


class TestHawkingProofs:
    """Tests for Hawking radiation theta derivations."""

    def test_planck_mass_black_hole(self):
        """Planck-mass black hole should have high theta."""
        result = compute_hawking_theta(M_PLANCK)
        # theta = M_Planck / (8*pi*M) for M = M_Planck
        expected = 1 / (8 * np.pi)
        assert result.theta == pytest.approx(expected, rel=0.01)

    def test_stellar_black_hole_classical(self):
        """Stellar black hole should be essentially classical."""
        M_sun = 1.989e30  # kg
        result = compute_hawking_theta(10 * M_sun)
        assert result.theta < 1e-40  # Incredibly small

    def test_hawking_temperature_positive(self):
        """Hawking temperature should be positive."""
        result = compute_hawking_theta(1e15)  # 10^15 kg
        assert result.temperature > 0

    def test_page_time_information(self):
        """After Page time, information should be recoverable."""
        mass = 1e12  # kg
        t_page = compute_page_time_theta(mass, 0).page_time

        # Before Page time
        result_before = compute_page_time_theta(mass, 0.5 * t_page)
        assert result_before.theta == 0.0  # No info yet

        # After Page time
        evap_time = 2 * t_page  # Approximate
        result_after = compute_page_time_theta(mass, 1.5 * t_page)
        assert result_after.theta > 0  # Info emerging

    def test_area_quantization(self):
        """Small area should give high theta."""
        L_PLANCK = 1.616255e-35  # m
        A_PLANCK = L_PLANCK**2

        # Planck area
        result_planck = compute_area_quantization_theta(A_PLANCK)
        assert result_planck.theta > 0.5

        # Large area
        result_large = compute_area_quantization_theta(1.0)  # 1 m²
        assert result_large.theta < 1e-60

    def test_hawking_proofs_interface(self):
        """Test unified HawkingProofs interface."""
        result = HawkingProofs.from_temperature(1e15)
        assert "theta" in result
        assert "proof_type" in result


class TestHolographicProofs:
    """Tests for holographic entropy theta derivations."""

    def test_rt_planck_area(self):
        """Planck-area surface should be quantum."""
        L_PLANCK = 1.616255e-35
        A_PLANCK = L_PLANCK**2
        result = compute_rt_theta(A_PLANCK)
        assert result.theta == 1.0  # Maximum quantum

    def test_rt_large_area(self):
        """Large area should be classical."""
        result = compute_rt_theta(1.0)  # 1 m²
        assert result.theta < 1e-60

    def test_wedge_half_space(self):
        """Half-space should give theta = 0.5 (continuous formula)."""
        result = compute_wedge_theta(0.5)
        assert abs(result.theta - 0.5) < 1e-10  # sin²(π/4) = 0.5 exactly

    def test_wedge_small_region(self):
        """Small boundary region gives small wedge."""
        result = compute_wedge_theta(0.1)
        assert result.theta < 0.2

    def test_wedge_boundary_limits(self):
        """Wedge theta should increase with boundary fraction."""
        theta_small = compute_wedge_theta(0.1).theta
        theta_large = compute_wedge_theta(0.9).theta
        assert theta_large > theta_small

    def test_holographic_proofs_interface(self):
        """Test unified HolographicProofs interface."""
        result = HolographicProofs.ryu_takayanagi(1e-60)
        assert "theta" in result
        assert "proof_type" in result


class TestVacuumEnergyProofs:
    """Tests for cosmological vacuum energy theta derivations."""

    def test_cosmological_constant_problem(self):
        """Vacuum energy theta should be incredibly small."""
        result = compute_vacuum_theta()
        # The 10^-122 problem!
        assert result.theta < 1e-100
        assert result.discrepancy > 1e100

    def test_dark_energy_lambda(self):
        """Pure cosmological constant has w = -1."""
        result = compute_dark_energy_theta(-1.0)
        assert result.theta == 0.0  # Exactly Λ

    def test_dark_energy_quintessence(self):
        """Quintessence deviates from w = -1."""
        result = compute_dark_energy_theta(-0.9)
        assert result.theta > 0
        assert result.model.value == "quintessence"

    def test_dark_energy_phantom(self):
        """Phantom energy has w < -1."""
        result = compute_dark_energy_theta(-1.1)
        assert result.theta > 0
        assert result.model.value == "phantom"

    def test_vacuum_energy_proofs_interface(self):
        """Test unified VacuumEnergyProofs interface."""
        result = VacuumEnergyProofs.cosmological_constant()
        assert "theta" in result
        assert "proof_type" in result
        assert "discrepancy" in result


class TestProofConsistency:
    """Cross-validation of proof methods."""

    def test_all_thetas_in_range(self):
        """All theta values should be in [0, 1]."""
        thetas = [
            compute_heisenberg_theta(1e-10, 1e-24).theta,
            compute_energy_time_theta(1e-20, 1e-10).theta,
            compute_hawking_theta(1e15).theta,
            compute_rt_theta(1e-60).theta,
            compute_dark_energy_theta(-0.95).theta,
        ]
        for theta in thetas:
            assert 0 <= theta <= 1

    def test_quantum_limit_high_theta(self):
        """Quantum limit should give high theta."""
        # Minimum uncertainty
        result = compute_heisenberg_theta(1e-10, HBAR / (2 * 1e-10))
        assert result.theta > 0.9

    def test_classical_limit_low_theta(self):
        """Classical limit should give low theta."""
        # Macroscopic uncertainties
        result = compute_heisenberg_theta(1e-3, 1e-20)
        assert result.theta < 0.1
