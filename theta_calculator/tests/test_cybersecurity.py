"""
Tests for Cybersecurity Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Security posture classification
- Component theta calculations
- Cross-domain consistency
"""

import pytest

from theta_calculator.domains.cybersecurity import (
    SecuritySystem,
    SecurityPosture,
    CryptoStrength,
    compute_attack_surface_theta,
    compute_detection_theta,
    compute_response_theta,
    compute_defense_depth_theta,
    compute_crypto_theta,
    compute_patch_theta,
    compute_auth_theta,
    compute_zero_trust_theta,
    compute_security_theta,
    classify_security_posture,
    classify_crypto_strength,
    cvss_to_theta,
    compute_threat_coupling,
    critical_coupling,
    SECURITY_SYSTEMS,
)


class TestSecuritySystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """SECURITY_SYSTEMS dict should exist."""
        assert SECURITY_SYSTEMS is not None
        assert isinstance(SECURITY_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(SECURITY_SYSTEMS) >= 5

    def test_system_names(self):
        """Key systems should be defined."""
        expected = [
            "unpatched_legacy",
            "enterprise_standard",
            "zero_trust",
            "active_breach",
        ]
        for name in expected:
            assert name in SECURITY_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in SECURITY_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "total_services")
            assert hasattr(system, "exposed_services")
            assert hasattr(system, "mttd_hours")
            assert hasattr(system, "mttr_hours")
            assert hasattr(system, "defense_layers")
            assert hasattr(system, "crypto_bits")
            assert hasattr(system, "patch_coverage")
            assert hasattr(system, "mfa_coverage")


class TestAttackSurfaceTheta:
    """Test attack surface theta calculation."""

    def test_no_exposure_high_theta(self):
        """No exposed services -> theta = 1.0."""
        theta = compute_attack_surface_theta(0, 100)
        assert theta == 1.0

    def test_full_exposure_low_theta(self):
        """All services exposed -> theta = 0.0."""
        theta = compute_attack_surface_theta(100, 100)
        assert theta == 0.0

    def test_partial_exposure(self):
        """50% exposure -> theta = 0.5."""
        theta = compute_attack_surface_theta(50, 100)
        assert theta == pytest.approx(0.5)

    def test_zero_total_services(self):
        """Zero total services -> theta = 0.0."""
        theta = compute_attack_surface_theta(0, 0)
        assert theta == 0.0


class TestDetectionTheta:
    """Test detection speed (MTTD) theta calculation."""

    def test_instant_detection(self):
        """Zero MTTD -> theta = 1.0."""
        theta = compute_detection_theta(0)
        assert theta == 1.0

    def test_optimal_detection(self):
        """MTTD at optimal -> theta = 1.0."""
        theta = compute_detection_theta(1.0, optimal_hours=1.0)
        assert theta == 1.0

    def test_slow_detection(self):
        """MTTD 100x optimal -> theta ~ 0.01."""
        theta = compute_detection_theta(100.0, optimal_hours=1.0)
        assert theta == pytest.approx(0.01)

    def test_very_slow_detection(self):
        """MTTD 1000x optimal -> theta capped at 0."""
        theta = compute_detection_theta(1000.0, optimal_hours=1.0)
        assert theta == pytest.approx(0.001)


class TestResponseTheta:
    """Test response speed (MTTR) theta calculation."""

    def test_instant_response(self):
        """Zero MTTR -> theta = 1.0."""
        theta = compute_response_theta(0)
        assert theta == 1.0

    def test_optimal_response(self):
        """MTTR at optimal -> theta = 1.0."""
        theta = compute_response_theta(4.0, optimal_hours=4.0)
        assert theta == 1.0

    def test_slow_response(self):
        """MTTR 10x optimal -> theta = 0.1."""
        theta = compute_response_theta(40.0, optimal_hours=4.0)
        assert theta == pytest.approx(0.1)


class TestDefenseDepthTheta:
    """Test defense-in-depth theta calculation."""

    def test_no_layers(self):
        """Zero layers -> theta = 0.0."""
        theta = compute_defense_depth_theta(0)
        assert theta == 0.0

    def test_full_layers(self):
        """Max layers -> theta = 1.0."""
        theta = compute_defense_depth_theta(7, max_layers=7)
        assert theta == 1.0

    def test_partial_layers(self):
        """Half layers -> theta ~ 0.5."""
        theta = compute_defense_depth_theta(3, max_layers=6)
        assert theta == pytest.approx(0.5)

    def test_zero_max_layers(self):
        """Zero max layers -> theta = 0.0."""
        theta = compute_defense_depth_theta(5, max_layers=0)
        assert theta == 0.0


class TestCryptoTheta:
    """Test cryptographic strength theta calculation."""

    def test_broken_crypto(self):
        """DES (56-bit) -> low theta."""
        theta = compute_crypto_theta(56)
        assert theta < 0.3

    def test_aes128(self):
        """AES-128 -> theta = 0.5."""
        theta = compute_crypto_theta(128)
        assert theta == pytest.approx(0.5)

    def test_aes256(self):
        """AES-256 -> theta = 1.0."""
        theta = compute_crypto_theta(256)
        assert theta == 1.0

    def test_quantum_safe(self):
        """512-bit -> theta capped at 1.0."""
        theta = compute_crypto_theta(512)
        assert theta == 1.0


class TestCryptoStrengthClassification:
    """Test cryptographic strength classification."""

    def test_broken(self):
        assert classify_crypto_strength(56) == CryptoStrength.BROKEN

    def test_weak(self):
        assert classify_crypto_strength(112) == CryptoStrength.WEAK

    def test_standard(self):
        assert classify_crypto_strength(128) == CryptoStrength.STANDARD

    def test_strong(self):
        assert classify_crypto_strength(192) == CryptoStrength.STRONG

    def test_quantum_safe(self):
        assert classify_crypto_strength(256) == CryptoStrength.QUANTUM_SAFE


class TestPatchTheta:
    """Test patch coverage theta calculation."""

    def test_no_vulns(self):
        """No vulnerabilities -> theta = 1.0."""
        theta = compute_patch_theta(0, 0)
        assert theta == 1.0

    def test_fully_patched(self):
        """All patched -> theta = 1.0."""
        theta = compute_patch_theta(100, 100)
        assert theta == 1.0

    def test_unpatched(self):
        """None patched -> theta = 0.0."""
        theta = compute_patch_theta(0, 100)
        assert theta == 0.0

    def test_partial_patch(self):
        """50% patched -> theta = 0.5."""
        theta = compute_patch_theta(50, 100)
        assert theta == pytest.approx(0.5)


class TestAuthTheta:
    """Test authentication strength theta calculation."""

    def test_full_mfa(self):
        """Full MFA -> high theta."""
        theta = compute_auth_theta(1.0, password_strength=1.0)
        assert theta == 1.0

    def test_no_mfa(self):
        """No MFA, weak passwords -> low theta."""
        theta = compute_auth_theta(0.0, password_strength=0.0)
        assert theta == 0.0

    def test_mfa_weight(self):
        """MFA should be weighted 70%."""
        theta = compute_auth_theta(1.0, password_strength=0.0)
        assert theta == pytest.approx(0.7)


class TestZeroTrustTheta:
    """Test Zero Trust maturity theta calculation."""

    def test_full_maturity(self):
        """All pillars at 100% -> theta = 1.0."""
        theta = compute_zero_trust_theta(1.0, 1.0, 1.0, 1.0, 1.0)
        assert theta == 1.0

    def test_no_maturity(self):
        """All pillars at 0% -> theta = 0.0."""
        theta = compute_zero_trust_theta(0.0, 0.0, 0.0, 0.0, 0.0)
        assert theta == 0.0

    def test_partial_maturity(self):
        """Average maturity."""
        theta = compute_zero_trust_theta(0.5, 0.5, 0.5, 0.5, 0.5)
        assert theta == pytest.approx(0.5)


class TestCVSSToTheta:
    """Test CVSS score to theta conversion."""

    def test_cvss_zero(self):
        """CVSS 0 -> theta = 1.0."""
        theta = cvss_to_theta(0.0)
        assert theta == 1.0

    def test_cvss_critical(self):
        """CVSS 10 -> theta = 0.0."""
        theta = cvss_to_theta(10.0)
        assert theta == 0.0

    def test_cvss_medium(self):
        """CVSS 5 -> theta = 0.5."""
        theta = cvss_to_theta(5.0)
        assert theta == pytest.approx(0.5)


class TestUnifiedSecurityTheta:
    """Test unified security theta calculation."""

    def test_all_systems_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in SECURITY_SYSTEMS.items():
            theta = compute_security_theta(system)
            assert 0 <= theta <= 1, f"{name} has invalid theta: {theta}"

    def test_compromised_system_low_theta(self):
        """Active breach should have very low theta."""
        system = SECURITY_SYSTEMS["active_breach"]
        theta = compute_security_theta(system)
        assert theta < 0.15

    def test_zero_trust_high_theta(self):
        """Zero trust should have high theta."""
        system = SECURITY_SYSTEMS["zero_trust"]
        theta = compute_security_theta(system)
        assert theta > 0.8

    def test_ordering_preserved(self):
        """Better security should have higher theta."""
        theta_legacy = compute_security_theta(SECURITY_SYSTEMS["unpatched_legacy"])
        theta_enterprise = compute_security_theta(SECURITY_SYSTEMS["enterprise_standard"])
        theta_zero_trust = compute_security_theta(SECURITY_SYSTEMS["zero_trust"])

        assert theta_legacy < theta_enterprise < theta_zero_trust


class TestSecurityPostureClassification:
    """Test security posture classification."""

    def test_compromised(self):
        assert classify_security_posture(0.1) == SecurityPosture.COMPROMISED

    def test_vulnerable(self):
        assert classify_security_posture(0.3) == SecurityPosture.VULNERABLE

    def test_baseline(self):
        assert classify_security_posture(0.5) == SecurityPosture.BASELINE

    def test_hardened(self):
        assert classify_security_posture(0.7) == SecurityPosture.HARDENED

    def test_fortified(self):
        assert classify_security_posture(0.9) == SecurityPosture.FORTIFIED


class TestThreatCoupling:
    """Test threat coupling (Ising analog) calculations."""

    def test_no_correlation(self):
        """Zero correlation -> zero coupling."""
        J = compute_threat_coupling(100, 0.0)
        assert J == 0.0

    def test_full_correlation(self):
        """Full correlation -> J = sqrt(N)."""
        J = compute_threat_coupling(100, 1.0)
        assert J == pytest.approx(10.0)

    def test_critical_coupling(self):
        """Critical coupling should be 1/sqrt(N)."""
        J_c = critical_coupling(100)
        assert J_c == pytest.approx(0.1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_theta_bounds(self):
        """All theta functions should return values in [0, 1]."""
        # Attack surface
        assert 0 <= compute_attack_surface_theta(-1, 100) <= 1
        assert 0 <= compute_attack_surface_theta(200, 100) <= 1

        # Detection
        assert 0 <= compute_detection_theta(-1) <= 1
        assert 0 <= compute_detection_theta(1e10) <= 1

        # Response
        assert 0 <= compute_response_theta(-1) <= 1

        # Defense depth
        assert 0 <= compute_defense_depth_theta(-1) <= 1
        assert 0 <= compute_defense_depth_theta(100) <= 1

        # Crypto
        assert 0 <= compute_crypto_theta(-1) <= 1
        assert 0 <= compute_crypto_theta(1024) <= 1

    def test_system_with_extreme_values(self):
        """System with extreme values should still compute valid theta."""
        extreme = SecuritySystem(
            name="Extreme",
            total_services=1000000,
            exposed_services=0,
            mttd_hours=0.001,
            mttr_hours=0.001,
            defense_layers=100,
            crypto_bits=4096,
            patch_coverage=1.0,
            mfa_coverage=1.0,
        )
        theta = compute_security_theta(extreme)
        assert 0 <= theta <= 1


class TestDocstrings:
    """Test that functions have proper documentation."""

    def test_module_docstring(self):
        """Module should have docstring with citations."""
        import theta_calculator.domains.cybersecurity as module
        assert module.__doc__ is not None
        assert "\\cite{" in module.__doc__

    def test_function_docstrings(self):
        """Key functions should have docstrings."""
        functions = [
            compute_attack_surface_theta,
            compute_detection_theta,
            compute_security_theta,
            classify_security_posture,
        ]
        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
