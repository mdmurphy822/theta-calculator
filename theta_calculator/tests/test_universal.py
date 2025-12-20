"""
Tests for universal module.

Tests cross-domain theta unification and the universal theta interface.
"""

import pytest
import numpy as np

from theta_calculator.domains.universal import (
    UniversalTheta,
    UniversalThetaResult,
    DomainType,
    cross_domain_comparison,
)
from theta_calculator.domains.economics import ECONOMIC_SYSTEMS
from theta_calculator.domains.information import INFORMATION_SYSTEMS
from theta_calculator.domains.game_theory import GAME_SYSTEMS
from theta_calculator.domains.complex_systems import COMPLEX_SYSTEMS
from theta_calculator.domains.quantum_computing import QUANTUM_HARDWARE


class TestUniversalTheta:
    """Test UniversalTheta class."""

    def test_compute_economics(self):
        """Test computing theta for economics domain."""
        system = ECONOMIC_SYSTEMS["efficient_market"]
        result = UniversalTheta.compute(system)
        assert isinstance(result, UniversalThetaResult)
        assert result.domain == DomainType.ECONOMICS
        assert 0 <= result.theta <= 1

    def test_compute_information(self):
        """Test computing theta for information domain."""
        system = INFORMATION_SYSTEMS["pure_qubit"]
        result = UniversalTheta.compute(system)
        assert result.domain == DomainType.INFORMATION
        assert 0 <= result.theta <= 1

    def test_compute_game_theory(self):
        """Test computing theta for game theory domain."""
        system = GAME_SYSTEMS["quantum_pd"]
        result = UniversalTheta.compute(system)
        assert result.domain == DomainType.GAME_THEORY
        assert 0 <= result.theta <= 1

    def test_compute_complex_systems(self):
        """Test computing theta for complex systems domain."""
        system = COMPLEX_SYSTEMS["ferromagnet_critical"]
        result = UniversalTheta.compute(system)
        assert result.domain == DomainType.COMPLEX_SYSTEMS
        assert 0 <= result.theta <= 1

    def test_compute_quantum_computing(self):
        """Test computing theta for quantum computing domain."""
        system = QUANTUM_HARDWARE["google_willow"]
        result = UniversalTheta.compute(system)
        assert result.domain == DomainType.QUANTUM_COMPUTING
        assert 0 <= result.theta <= 1


class TestDomainDetection:
    """Test automatic domain detection."""

    def test_detect_economics(self):
        """Should detect economics from MarketSystem."""
        system = ECONOMIC_SYSTEMS["efficient_market"]
        domain = UniversalTheta._detect_domain(system)
        assert domain == DomainType.ECONOMICS

    def test_detect_information(self):
        """Should detect information from InformationSystem."""
        system = INFORMATION_SYSTEMS["pure_qubit"]
        domain = UniversalTheta._detect_domain(system)
        assert domain == DomainType.INFORMATION

    def test_detect_game_theory(self):
        """Should detect game theory from QuantumGame."""
        system = GAME_SYSTEMS["quantum_pd"]
        domain = UniversalTheta._detect_domain(system)
        assert domain == DomainType.GAME_THEORY

    def test_detect_complex_systems(self):
        """Should detect complex systems from ComplexSystem."""
        system = COMPLEX_SYSTEMS["ferromagnet_critical"]
        domain = UniversalTheta._detect_domain(system)
        assert domain == DomainType.COMPLEX_SYSTEMS

    def test_detect_quantum_computing(self):
        """Should detect quantum computing from QubitSystem."""
        system = QUANTUM_HARDWARE["google_willow"]
        domain = UniversalTheta._detect_domain(system)
        assert domain == DomainType.QUANTUM_COMPUTING

    def test_unknown_type_raises(self):
        """Should raise TypeError for unknown system type."""
        with pytest.raises(TypeError):
            UniversalTheta._detect_domain("not a system")


class TestUniversalThetaResult:
    """Test UniversalThetaResult dataclass."""

    def test_result_has_required_fields(self):
        """Result should have all required fields."""
        system = ECONOMIC_SYSTEMS["efficient_market"]
        result = UniversalTheta.compute(system)

        assert hasattr(result, 'theta')
        assert hasattr(result, 'domain')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'details')

    def test_result_details_populated(self):
        """Result details should be populated."""
        system = ECONOMIC_SYSTEMS["efficient_market"]
        result = UniversalTheta.compute(system)
        assert result.details is not None
        assert isinstance(result.details, dict)


class TestRegimeClassification:
    """Test regime classification methods."""

    def test_market_regime_efficient(self):
        """Low theta should be efficient market."""
        regime = UniversalTheta._classify_market_regime(0.1)
        assert regime == "efficient"

    def test_market_regime_crash(self):
        """High theta should be crash/bubble."""
        regime = UniversalTheta._classify_market_regime(0.9)
        assert regime == "crash/bubble"

    def test_information_regime_pure(self):
        """Very low theta should be pure/deterministic."""
        regime = UniversalTheta._classify_information_regime(0.05)
        assert regime == "pure/deterministic"

    def test_information_regime_mixed(self):
        """High theta should be maximally mixed."""
        regime = UniversalTheta._classify_information_regime(0.95)
        assert regime == "maximally mixed"

    def test_game_regime_classical(self):
        """Low theta should be classical."""
        regime = UniversalTheta._classify_game_regime(0.05)
        assert regime == "classical"

    def test_game_regime_quantum(self):
        """High theta should be quantum."""
        regime = UniversalTheta._classify_game_regime(0.9)
        assert regime == "quantum"

    def test_qc_regime_coherent(self):
        """High theta should be coherent."""
        regime = UniversalTheta._classify_qc_regime(0.95)
        assert regime == "highly coherent"

    def test_qc_regime_noisy(self):
        """Low theta should be noisy."""
        regime = UniversalTheta._classify_qc_regime(0.05)
        assert regime == "classical/noisy"


class TestCrossDomainComparison:
    """Test cross_domain_comparison function."""

    def test_returns_all_domains(self):
        """Should return results for all domains."""
        comparison = cross_domain_comparison()
        expected_domains = [
            "economics", "information", "game_theory",
            "complex_systems", "quantum_computing"
        ]
        for domain in expected_domains:
            assert domain in comparison

    def test_all_thetas_in_range(self):
        """All theta values should be in [0, 1]."""
        comparison = cross_domain_comparison()
        for domain, systems in comparison.items():
            for name, theta in systems.items():
                assert 0 <= theta <= 1, f"{domain}/{name} theta out of range: {theta}"

    def test_economics_has_systems(self):
        """Economics should have expected systems."""
        comparison = cross_domain_comparison()
        assert "efficient_market" in comparison["economics"]
        assert "market_crash" in comparison["economics"]

    def test_game_theory_has_systems(self):
        """Game theory should have expected systems."""
        comparison = cross_domain_comparison()
        assert "classical_pd" in comparison["game_theory"]
        assert "quantum_pd" in comparison["game_theory"]


class TestThetaConsistency:
    """Test that theta behaves consistently across domains."""

    def test_classical_systems_low_theta(self):
        """Classical-like systems should have low theta."""
        results = []

        # Classical game
        result = UniversalTheta.compute(GAME_SYSTEMS["classical_pd"])
        results.append(("classical game", result.theta))

        # Pure qubit
        result = UniversalTheta.compute(INFORMATION_SYSTEMS["deterministic"])
        results.append(("deterministic", result.theta))

        # Efficient market
        result = UniversalTheta.compute(ECONOMIC_SYSTEMS["efficient_market"])
        results.append(("efficient market", result.theta))

        for name, theta in results:
            assert theta < 0.5, f"{name} should have low theta: {theta}"

    def test_quantum_systems_high_theta(self):
        """Quantum-like systems should have high theta."""
        results = []

        # Quantum game
        result = UniversalTheta.compute(GAME_SYSTEMS["quantum_pd"])
        results.append(("quantum game", result.theta))

        # Maximally mixed
        result = UniversalTheta.compute(INFORMATION_SYSTEMS["mixed_qubit"])
        results.append(("mixed qubit", result.theta))

        # Critical system
        result = UniversalTheta.compute(COMPLEX_SYSTEMS["ferromagnet_critical"])
        results.append(("critical system", result.theta))

        for name, theta in results:
            assert theta > 0.5, f"{name} should have high theta: {theta}"


class TestDomainOverride:
    """Test explicit domain override."""

    def test_explicit_domain(self):
        """Should use explicit domain when provided."""
        system = ECONOMIC_SYSTEMS["efficient_market"]
        result = UniversalTheta.compute(system, domain=DomainType.ECONOMICS)
        assert result.domain == DomainType.ECONOMICS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
