"""
Snapshot tests against PRECALCULATED_RESULTS.md values.

These tests verify that theta calculations remain consistent with
documented, reviewed values. Any changes to the underlying calculations
should be intentional and reviewed.
"""

import pytest
from theta_calculator.core.theta_state import PhysicalSystem, EXAMPLE_SYSTEMS
from theta_calculator.proofs.unified import UnifiedThetaProof, score_theta

# Snapshot values from PRECALCULATED_RESULTS.md
# Format: system_name -> (expected_theta, tolerance)
PHYSICS_SNAPSHOTS = {
    "electron": (0.87, 0.05),
    "proton": (0.61, 0.05),
    "hydrogen_atom": (0.77, 0.05),
    "water_molecule": (0.77, 0.05),
    "virus": (0.42, 0.05),
    "human_cell": (0.23, 0.05),
    "baseball": (0.13, 0.05),
    "human": (0.13, 0.05),
    "earth": (0.13, 0.05),
    "stellar_black_hole": (0.51, 0.05),
}


class TestPhysicsSnapshots:
    """Test that theta values match documented snapshots."""

    @pytest.fixture
    def estimator(self):
        return UnifiedThetaProof()

    @pytest.mark.parametrize("system_name,expected", PHYSICS_SNAPSHOTS.items())
    def test_example_system_theta(self, estimator, system_name, expected):
        """Test that example systems match expected theta values."""
        expected_theta, tolerance = expected

        if system_name not in EXAMPLE_SYSTEMS:
            pytest.skip(f"System {system_name} not in EXAMPLE_SYSTEMS")

        system = EXAMPLE_SYSTEMS[system_name]
        result = estimator.compute_theta(system)

        assert abs(result.theta - expected_theta) < tolerance, (
            f"System {system_name}: expected θ ≈ {expected_theta}, got {result.theta:.4f}"
        )

    def test_all_theta_in_valid_range(self, estimator):
        """Test that all systems produce theta in [0, 1]."""
        for name, system in EXAMPLE_SYSTEMS.items():
            result = estimator.compute_theta(system)
            assert 0 <= result.theta <= 1, (
                f"System {name}: θ = {result.theta} is outside [0, 1]"
            )

    def test_quantum_systems_have_high_theta(self, estimator):
        """Test that quantum systems (electron, proton) have high theta."""
        quantum_systems = ["electron", "proton", "hydrogen_atom"]
        for name in quantum_systems:
            if name in EXAMPLE_SYSTEMS:
                result = estimator.compute_theta(EXAMPLE_SYSTEMS[name])
                assert result.theta > 0.5, (
                    f"Quantum system {name}: θ = {result.theta} should be > 0.5"
                )

    def test_classical_systems_have_low_theta(self, estimator):
        """Test that classical systems (baseball, human, earth) have low theta."""
        classical_systems = ["baseball", "human", "earth"]
        for name in classical_systems:
            if name in EXAMPLE_SYSTEMS:
                result = estimator.compute_theta(EXAMPLE_SYSTEMS[name])
                assert result.theta < 0.5, (
                    f"Classical system {name}: θ = {result.theta} should be < 0.5"
                )

    def test_theta_ordering_preserved(self, estimator):
        """Test that theta ordering matches physical intuition."""
        # Electron should have higher theta than baseball
        if "electron" in EXAMPLE_SYSTEMS and "baseball" in EXAMPLE_SYSTEMS:
            electron_theta = estimator.compute_theta(EXAMPLE_SYSTEMS["electron"]).theta
            baseball_theta = estimator.compute_theta(EXAMPLE_SYSTEMS["baseball"]).theta
            assert electron_theta > baseball_theta, (
                f"Electron (θ={electron_theta}) should have higher theta than baseball (θ={baseball_theta})"
            )


class TestDomainSnapshots:
    """Test domain-specific theta calculations."""

    def test_economics_theta_range(self):
        """Test economics domain produces valid theta values."""
        from theta_calculator.domains import ECONOMIC_SYSTEMS, compute_market_theta

        for name, system in ECONOMIC_SYSTEMS.items():
            theta = compute_market_theta(system)
            assert 0 <= theta <= 1, f"Economics {name}: θ = {theta} outside [0, 1]"

    def test_quantum_computing_theta_range(self):
        """Test quantum computing domain produces valid theta values."""
        from theta_calculator.domains import QUANTUM_HARDWARE, compute_quantum_computing_theta

        for name, system in QUANTUM_HARDWARE.items():
            theta = compute_quantum_computing_theta(system)
            assert 0 <= theta <= 1, f"QC {name}: θ = {theta} outside [0, 1]"

    def test_information_theta_range(self):
        """Test information domain produces valid theta values."""
        from theta_calculator.domains import INFORMATION_SYSTEMS, compute_information_theta

        for name, system in INFORMATION_SYSTEMS.items():
            theta = compute_information_theta(system)
            assert 0 <= theta <= 1, f"Info {name}: θ = {theta} outside [0, 1]"


class TestRegressionGuards:
    """Regression tests to catch accidental changes."""

    def test_score_theta_function(self):
        """Test score_theta convenience function."""
        electron = PhysicalSystem(
            name="electron",
            mass=9.109e-31,
            length_scale=2.818e-15,
            energy=8.187e-14,
            temperature=300.0
        )
        result = score_theta(electron)
        assert result.theta > 0.5, "Electron should be quantum-like"
        assert result.is_valid, "Result should be valid"

    def test_result_structure(self):
        """Test that result has all expected fields."""
        electron = EXAMPLE_SYSTEMS.get("electron")
        if electron is None:
            pytest.skip("electron not in EXAMPLE_SYSTEMS")

        result = score_theta(electron)

        # Check all expected fields exist
        assert hasattr(result, 'theta')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'method_agreement')
        assert hasattr(result, 'theta_values')
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'summary')

    def test_method_agreement_reasonable(self):
        """Test that method agreement is reasonable for well-defined systems."""
        electron = EXAMPLE_SYSTEMS.get("electron")
        if electron is None:
            pytest.skip("electron not in EXAMPLE_SYSTEMS")

        result = score_theta(electron)

        # Methods should generally agree for standard systems
        assert result.method_agreement > 0.3, (
            f"Method agreement {result.method_agreement} is suspiciously low"
        )
