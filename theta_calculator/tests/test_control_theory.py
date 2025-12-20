"""
Tests for control_theory module.

Tests control theory theta calculations for stability margins,
feedback systems, and control architectures.
"""

import pytest
import numpy as np

from theta_calculator.domains.control_theory import (
    CONTROL_SYSTEMS,
    ControlSystem,
    ControllerType,
    StabilityRegime,
    compute_control_theta,
    compute_gain_margin_theta,
    compute_phase_margin_theta,
    compute_pole_theta,
    classify_stability,
    bode_stability_margins,
    settling_time_theta,
    overshoot_theta,
    nyquist_encirclements,
)


class TestControlSystems:
    """Test the predefined control systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "open_loop", "thermostat_simple", "pid_tuned",
            "inverted_pendulum", "spacecraft_attitude", "lqr_optimal",
            "h_infinity", "quantum_error_correction", "marginally_stable",
            "neural_feedback"
        ]
        for name in expected:
            assert name in CONTROL_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in CONTROL_SYSTEMS.items():
            assert isinstance(system, ControlSystem)
            assert system.name
            assert isinstance(system.controller_type, ControllerType)
            # Gain margin can be negative (unstable) or infinite
            assert system.phase_margin_deg is not None


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in CONTROL_SYSTEMS.items():
            theta = compute_control_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_open_loop_unstable(self):
        """Open loop system should have low theta (unstable)."""
        open_loop = CONTROL_SYSTEMS["open_loop"]
        theta = compute_control_theta(open_loop)
        assert theta < 0.1, f"Open loop should be unstable: {theta}"

    def test_lqr_highly_stable(self):
        """LQR optimal should have high theta."""
        lqr = CONTROL_SYSTEMS["lqr_optimal"]
        theta = compute_control_theta(lqr)
        # LQR has infinite gain margin
        assert theta > 0.5, f"LQR should be stable: {theta}"

    def test_stability_ordering(self):
        """More stable systems should have higher theta."""
        marginal = CONTROL_SYSTEMS["marginally_stable"]
        tuned = CONTROL_SYSTEMS["pid_tuned"]
        robust = CONTROL_SYSTEMS["h_infinity"]

        theta_marginal = compute_control_theta(marginal)
        theta_tuned = compute_control_theta(tuned)
        theta_robust = compute_control_theta(robust)

        assert theta_marginal < theta_tuned, "PID should beat marginal"
        assert theta_tuned < theta_robust, "H-infinity should beat PID"


class TestGainMarginTheta:
    """Test gain margin theta calculation."""

    def test_negative_margin_gives_zero(self):
        """Negative gain margin should give theta = 0."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.PROPORTIONAL,
            gain_margin_db=-5,
            phase_margin_deg=30
        )
        theta = compute_gain_margin_theta(system)
        assert theta == 0.0

    def test_high_margin_gives_high_theta(self):
        """Large gain margin should give theta near 1."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.ROBUST,
            gain_margin_db=20,
            phase_margin_deg=60
        )
        theta = compute_gain_margin_theta(system)
        assert theta > 0.9, f"High margin should give high theta: {theta}"

    def test_typical_margin(self):
        """6 dB margin should give moderate theta."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.PID,
            gain_margin_db=6,
            phase_margin_deg=45
        )
        theta = compute_gain_margin_theta(system)
        # At 6 dB, theta = 1 - exp(-1) ≈ 0.632
        assert 0.5 < theta < 0.8


class TestPhaseMarginTheta:
    """Test phase margin theta calculation."""

    def test_zero_margin_gives_zero(self):
        """Zero phase margin should give theta = 0."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.PROPORTIONAL,
            gain_margin_db=10,
            phase_margin_deg=0
        )
        theta = compute_phase_margin_theta(system)
        assert theta == 0.0

    def test_90_deg_gives_one(self):
        """90 degree phase margin should give theta = 1."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.ROBUST,
            gain_margin_db=10,
            phase_margin_deg=90
        )
        theta = compute_phase_margin_theta(system)
        assert theta == 1.0

    def test_45_deg_gives_half(self):
        """45 degree phase margin should give theta = 0.5."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.PID,
            gain_margin_db=10,
            phase_margin_deg=45
        )
        theta = compute_phase_margin_theta(system)
        assert theta == 0.5


class TestPoleTheta:
    """Test pole-based theta calculation."""

    def test_positive_pole_gives_zero(self):
        """Positive real pole (unstable) should give theta = 0."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.OPEN_LOOP,
            gain_margin_db=-10,
            phase_margin_deg=-10,
            poles_real_parts=[0.5, -1]
        )
        theta = compute_pole_theta(system)
        assert theta == 0.0

    def test_negative_poles_give_positive_theta(self):
        """Negative poles should give positive theta."""
        system = ControlSystem(
            name="Test",
            controller_type=ControllerType.PID,
            gain_margin_db=10,
            phase_margin_deg=45,
            poles_real_parts=[-2, -1.5, -1]
        )
        theta = compute_pole_theta(system)
        assert theta > 0


class TestStabilityClassification:
    """Test stability regime classification."""

    def test_unstable_regime(self):
        """Theta < 0.1 should be UNSTABLE."""
        assert classify_stability(0.05) == StabilityRegime.UNSTABLE

    def test_marginal_regime(self):
        """0.1 <= theta < 0.3 should be MARGINAL."""
        assert classify_stability(0.2) == StabilityRegime.MARGINALLY_STABLE

    def test_stable_regime(self):
        """0.3 <= theta < 0.7 should be STABLE."""
        assert classify_stability(0.5) == StabilityRegime.STABLE

    def test_robust_regime(self):
        """0.7 <= theta < 0.9 should be ROBUST."""
        assert classify_stability(0.8) == StabilityRegime.ROBUST

    def test_optimal_regime(self):
        """theta >= 0.9 should be OPTIMAL."""
        assert classify_stability(0.95) == StabilityRegime.OPTIMAL


class TestBodeMargins:
    """Test Bode stability margin calculation."""

    def test_margin_calculation(self):
        """Test correct margin computation from Bode data."""
        result = bode_stability_margins(
            gain_crossover_hz=10,
            phase_crossover_hz=50,
            gain_at_phase_crossover_db=-6,  # -6 dB at phase crossover
            phase_at_gain_crossover_deg=-135  # -135 deg at gain crossover
        )
        assert result["gain_margin_db"] == 6
        assert result["phase_margin_deg"] == 45


class TestSettlingTimeTheta:
    """Test settling time theta calculation."""

    def test_fast_settling_high_theta(self):
        """Fast settling should give high theta."""
        theta = settling_time_theta(settling_time=0.5, time_constant=1.0)
        assert theta > 0.8

    def test_slow_settling_low_theta(self):
        """Slow settling should give low theta."""
        theta = settling_time_theta(settling_time=10, time_constant=1.0)
        assert theta < 0.5


class TestOvershootTheta:
    """Test overshoot theta calculation."""

    def test_no_overshoot_gives_one(self):
        """No overshoot should give theta = 1."""
        theta = overshoot_theta(0)
        assert theta == 1.0

    def test_large_overshoot_gives_low_theta(self):
        """Large overshoot should give low theta."""
        theta = overshoot_theta(50)
        assert theta < 0.5


class TestNyquist:
    """Test Nyquist stability criterion."""

    def test_stable_system(self):
        """Test stable system identification."""
        result = nyquist_encirclements(
            poles_rhp=0,
            zeros_rhp=0,
            encirclements_ccw=0
        )
        assert result["is_stable"] is True
        assert result["theta"] == 1.0

    def test_unstable_system(self):
        """Test unstable system identification."""
        result = nyquist_encirclements(
            poles_rhp=2,
            zeros_rhp=0,
            encirclements_ccw=0
        )
        assert result["is_stable"] is False
        assert result["theta"] == 0.0


class TestControllerTypes:
    """Test different controller type behaviors."""

    def test_quantum_error_correction_high_theta(self):
        """Quantum error correction should have very high theta."""
        qec = CONTROL_SYSTEMS["quantum_error_correction"]
        theta = compute_control_theta(qec)
        assert theta > 0.8, f"QEC should be highly robust: {theta}"

    def test_adaptive_controller(self):
        """Neural feedback (adaptive) should have positive theta."""
        neural = CONTROL_SYSTEMS["neural_feedback"]
        theta = compute_control_theta(neural)
        # Neural feedback has slow poles but good margins
        assert 0.1 < theta < 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
