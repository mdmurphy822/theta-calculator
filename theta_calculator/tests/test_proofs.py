"""Tests for theta proof components."""

import numpy as np
import pytest

from theta_calculator.core.theta_state import PhysicalSystem, Regime, EXAMPLE_SYSTEMS
from theta_calculator.core.interpolation import ThetaCalculator, compute_theta
from theta_calculator.proofs.unified import UnifiedThetaProof, prove_theta
from theta_calculator.proofs.information.bekenstein_bound import BekensteinBound
from theta_calculator.proofs.information.landauer_limit import LandauerLimit
from theta_calculator.proofs.mathematical.constant_bootstrap import ConstantBootstrap


class TestThetaCalculator:
    """Test the core theta calculation methods."""

    @pytest.fixture
    def calculator(self):
        return ThetaCalculator()

    @pytest.fixture
    def electron(self):
        return EXAMPLE_SYSTEMS["electron"]

    @pytest.fixture
    def baseball(self):
        return EXAMPLE_SYSTEMS["baseball"]

    def test_action_theta_electron(self, calculator, electron):
        """Electron should have high theta from action method."""
        state = calculator.compute_action_theta(electron)
        assert state.theta > 0.5  # Quantum
        assert state.proof_method == "action_ratio"

    def test_action_theta_baseball(self, calculator, baseball):
        """Baseball should have very low theta from action method."""
        state = calculator.compute_action_theta(baseball)
        assert state.theta < 0.01  # Classical
        assert state.regime == Regime.CLASSICAL

    def test_thermal_theta(self, calculator, electron):
        """Thermal theta should work."""
        state = calculator.compute_thermal_theta(electron)
        assert 0 <= state.theta <= 1
        assert state.proof_method == "thermal_ratio"

    def test_scale_theta(self, calculator, electron):
        """Scale theta should work."""
        state = calculator.compute_scale_theta(electron)
        assert 0 <= state.theta <= 1
        assert state.proof_method == "scale_ratio"

    def test_decoherence_theta(self, calculator, electron):
        """Decoherence theta should work."""
        state = calculator.compute_decoherence_theta(electron)
        assert 0 <= state.theta <= 1
        assert state.proof_method == "decoherence"

    def test_unified_theta(self, calculator, electron):
        """Unified theta should combine all methods."""
        state = calculator.compute_unified_theta(electron)
        assert 0 <= state.theta <= 1
        assert state.proof_method == "unified"
        assert len(state.components) == 4  # All four methods

    def test_compute_all_methods(self, calculator, electron):
        """compute_all_methods should return dict of all results."""
        results = calculator.compute_all_methods(electron)
        assert "action_ratio" in results
        assert "thermal_ratio" in results
        assert "scale_ratio" in results
        assert "decoherence" in results
        assert "unified" in results

    def test_convergence_analysis(self, calculator, electron):
        """Convergence analysis should return statistics."""
        stats = calculator.analyze_convergence(electron)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_theta_ordering(self, calculator):
        """Theta should decrease with increasing size/mass."""
        electron = EXAMPLE_SYSTEMS["electron"]
        atom = EXAMPLE_SYSTEMS["hydrogen_atom"]
        baseball = EXAMPLE_SYSTEMS["baseball"]

        theta_electron = calculator.compute_unified_theta(electron).theta
        theta_atom = calculator.compute_unified_theta(atom).theta
        theta_baseball = calculator.compute_unified_theta(baseball).theta

        # Electron is most quantum, baseball is most classical
        assert theta_electron > theta_atom
        assert theta_atom > theta_baseball


class TestBekensteinBound:
    """Test Bekenstein bound calculations."""

    @pytest.fixture
    def bekenstein(self):
        return BekensteinBound()

    def test_compute_bound_positive(self, bekenstein):
        """Bekenstein bound should be positive."""
        S = bekenstein.compute_bound_nats(radius=1.0, energy=1e9)
        assert S > 0

    def test_compute_bound_bits(self, bekenstein):
        """Bound in bits should be positive."""
        S_bits = bekenstein.compute_bound_bits(radius=1.0, energy=1e9)
        assert S_bits > 0

    def test_schwarzschild_radius(self, bekenstein):
        """Schwarzschild radius should scale with mass."""
        r1 = bekenstein.schwarzschild_radius(mass=1e30)
        r2 = bekenstein.schwarzschild_radius(mass=2e30)
        assert np.isclose(r2, 2 * r1, rtol=1e-10)

    def test_black_hole_entropy(self, bekenstein):
        """Black hole entropy should be positive."""
        S_bits = bekenstein.black_hole_entropy_bits(mass=1e30)
        assert S_bits > 0
        assert np.isfinite(S_bits)

    def test_theta_from_bekenstein(self, bekenstein):
        """Should produce valid ThetaState."""
        electron = EXAMPLE_SYSTEMS["electron"]
        state = bekenstein.theta_from_bekenstein(electron)
        assert 0 <= state.theta <= 1
        assert state.proof_method == "bekenstein_bound"


class TestLandauerLimit:
    """Test Landauer limit calculations."""

    @pytest.fixture
    def landauer(self):
        return LandauerLimit()

    def test_minimum_erasure_energy(self, landauer):
        """Minimum erasure energy should be positive."""
        E = landauer.minimum_erasure_energy(temperature=300.0)
        assert E > 0

    def test_erasure_energy_proportional_to_temp(self, landauer):
        """Erasure energy should scale with temperature."""
        E1 = landauer.minimum_erasure_energy(temperature=300.0)
        E2 = landauer.minimum_erasure_energy(temperature=600.0)
        assert np.isclose(E2, 2 * E1, rtol=1e-10)

    def test_max_operations_quantum(self, landauer):
        """Quantum limit should be positive."""
        N = landauer.max_operations_quantum(energy=1e-15, time=1e-9)
        assert N > 0

    def test_theta_from_landauer(self, landauer):
        """Should produce valid ThetaState."""
        electron = EXAMPLE_SYSTEMS["electron"]
        state = landauer.theta_from_landauer(electron)
        assert 0 <= state.theta <= 1
        assert state.proof_method == "landauer_limit"


class TestConstantBootstrap:
    """Test constant bootstrap derivations."""

    @pytest.fixture
    def bootstrap(self):
        return ConstantBootstrap()

    def test_derive_c_from_electromagnetic(self, bootstrap):
        """Speed of light derivation should be consistent."""
        result = bootstrap.derive_c_from_electromagnetic()
        assert result.is_consistent
        assert result.relative_error < 1e-6

    def test_derive_c_from_alpha(self, bootstrap):
        """Speed of light from alpha should be consistent."""
        result = bootstrap.derive_c_from_alpha()
        assert result.is_consistent
        assert result.relative_error < 1e-6

    def test_derive_alpha(self, bootstrap):
        """Alpha derivation should be consistent."""
        result = bootstrap.derive_alpha_from_definition()
        assert result.is_consistent
        assert result.relative_error < 1e-6

    def test_derive_G_from_planck(self, bootstrap):
        """G from Planck mass should be consistent."""
        result = bootstrap.derive_G_from_planck_mass()
        assert result.is_consistent
        assert result.relative_error < 1e-6

    def test_verify_all_bootstraps(self, bootstrap):
        """All bootstrap derivations should pass."""
        results = bootstrap.verify_all_bootstraps()
        for name, passed in results.items():
            assert passed, f"Bootstrap {name} failed"

    def test_dependency_graph(self, bootstrap):
        """Dependency graph should contain expected constants."""
        graph = bootstrap.get_dependency_graph()
        assert "c" in graph
        assert "α" in graph
        assert "G" in graph


class TestUnifiedProof:
    """Test unified theta proof."""

    @pytest.fixture
    def proof(self):
        return UnifiedThetaProof()

    def test_electron_is_quantum(self, proof):
        """Electron should have high theta (quantum)."""
        electron = EXAMPLE_SYSTEMS["electron"]
        result = proof.prove_theta_exists(electron)

        assert result.theta > 0.5
        assert result.regime in [Regime.QUANTUM, Regime.TRANSITION]
        assert result.is_valid

    def test_baseball_is_classical(self, proof):
        """Baseball should have low theta (near-classical)."""
        baseball = EXAMPLE_SYSTEMS["baseball"]
        result = proof.prove_theta_exists(baseball)

        # Baseball is macroscopic, theta depends on methodology
        assert result.theta < 0.5  # Should be classical-leaning
        # Note: is_valid may be False due to high method divergence

    def test_proof_has_all_components(self, proof):
        """Proof should include all three methodologies."""
        electron = EXAMPLE_SYSTEMS["electron"]
        result = proof.prove_theta_exists(electron)

        assert "mathematical" in result.theta_values
        assert "numerical_unified" in result.theta_values
        assert "bekenstein" in result.theta_values

    def test_method_agreement_calculated(self, proof):
        """Method agreement should be between 0 and 1."""
        electron = EXAMPLE_SYSTEMS["electron"]
        result = proof.prove_theta_exists(electron)

        assert 0 <= result.method_agreement <= 1
        assert 0 <= result.consistency_score <= 1

    def test_summary_generated(self, proof):
        """Proof should generate summary."""
        electron = EXAMPLE_SYSTEMS["electron"]
        result = proof.prove_theta_exists(electron)

        assert len(result.summary) > 0
        assert len(result.detailed_explanation) > 0

    def test_validation_notes(self, proof):
        """Validation notes should be populated."""
        electron = EXAMPLE_SYSTEMS["electron"]
        result = proof.prove_theta_exists(electron)

        assert isinstance(result.validation_notes, list)
        assert len(result.validation_notes) > 0

    def test_compare_systems(self, proof):
        """Should compare multiple systems."""
        systems = [EXAMPLE_SYSTEMS["electron"], EXAMPLE_SYSTEMS["baseball"]]
        results = proof.compare_systems(systems)

        assert "electron" in results
        assert "baseball" in results
        assert results["electron"].theta > results["baseball"].theta


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_theta(self):
        """compute_theta convenience function should work."""
        electron = EXAMPLE_SYSTEMS["electron"]
        state = compute_theta(electron)

        assert 0 <= state.theta <= 1
        assert state.proof_method == "unified"

    def test_prove_theta(self):
        """prove_theta convenience function should work."""
        electron = EXAMPLE_SYSTEMS["electron"]
        result = prove_theta(electron)

        assert isinstance(result.theta, float)
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
