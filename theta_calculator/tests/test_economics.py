"""
Tests for economics module.

Tests market theta calculations using Ising model framework
for financial market phase transitions.
"""

import pytest
import numpy as np

from theta_calculator.domains.economics import (
    ECONOMIC_SYSTEMS,
    MarketSystem,
    MarketRegime,
    IsingMarket,
    compute_market_theta,
    compute_coupling_from_correlation,
    detect_phase_transition,
    classify_regime,
    MEAN_FIELD_EXPONENTS,
    ISING_3D_EXPONENTS,
)


class TestMarketSystems:
    """Test the predefined market systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "efficient_market", "normal_trading", "trending_market",
            "bubble_forming", "market_crash", "flash_crash", "dotcom_bubble"
        ]
        for name in expected:
            assert name in ECONOMIC_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, market in ECONOMIC_SYSTEMS.items():
            assert isinstance(market, MarketSystem)
            assert market.name
            assert market.n_traders > 0
            assert market.coupling_strength >= 0
            assert market.temperature > 0
            assert -1 <= market.order_parameter <= 1


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, market in ECONOMIC_SYSTEMS.items():
            theta = compute_market_theta(market)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_efficient_market_low_theta(self):
        """Efficient market should have low theta."""
        efficient = ECONOMIC_SYSTEMS["efficient_market"]
        theta = compute_market_theta(efficient)
        assert theta < 0.5, f"Efficient market should have low theta: {theta}"

    def test_crash_high_theta(self):
        """Market crash should have high theta."""
        crash = ECONOMIC_SYSTEMS["market_crash"]
        theta = compute_market_theta(crash)
        assert theta > 0.5, f"Market crash should have high theta: {theta}"

    def test_flash_crash_high_theta(self):
        """Flash crash should have very high theta."""
        flash = ECONOMIC_SYSTEMS["flash_crash"]
        theta = compute_market_theta(flash)
        assert theta > 0.7, f"Flash crash should have very high theta: {theta}"

    def test_theta_ordering(self):
        """Crash should have higher theta than efficient market."""
        efficient = ECONOMIC_SYSTEMS["efficient_market"]
        crash = ECONOMIC_SYSTEMS["market_crash"]

        theta_efficient = compute_market_theta(efficient)
        theta_crash = compute_market_theta(crash)

        # Crash has higher correlation/coupling, so higher theta
        assert theta_crash > theta_efficient


class TestMarketProperties:
    """Test market system properties."""

    def test_critical_temperature(self):
        """Critical temperature should be positive."""
        for name, market in ECONOMIC_SYSTEMS.items():
            T_c = market.critical_temperature
            assert T_c > 0, f"{name} should have positive T_c"

    def test_reduced_temperature(self):
        """Reduced temperature should be finite for normal markets."""
        normal = ECONOMIC_SYSTEMS["normal_trading"]
        t = normal.reduced_temperature
        assert np.isfinite(t), "Reduced temperature should be finite"


class TestIsingMarket:
    """Test Ising market model."""

    def test_create_ising_market(self):
        """Test creating an Ising market."""
        n = 10
        J = np.random.randn(n, n)
        J = (J + J.T) / 2  # Symmetrize
        np.fill_diagonal(J, 0)

        market = IsingMarket(n_spins=n, coupling_matrix=J)
        assert market.n_spins == n
        assert market.external_field == 0.0

    def test_magnetization(self):
        """Test magnetization calculation."""
        n = 10
        J = np.zeros((n, n))
        market = IsingMarket(n_spins=n, coupling_matrix=J)

        # All up spins
        spins_up = np.ones(n)
        assert market.magnetization(spins_up) == 1.0

        # All down spins
        spins_down = -np.ones(n)
        assert market.magnetization(spins_down) == -1.0

        # Mixed spins
        spins_mixed = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        assert market.magnetization(spins_mixed) == 0.0

    def test_energy(self):
        """Test energy calculation."""
        n = 4
        J = np.ones((n, n)) * 0.5
        np.fill_diagonal(J, 0)
        market = IsingMarket(n_spins=n, coupling_matrix=J)

        # All aligned spins should have lower energy
        spins_aligned = np.ones(n)
        spins_random = np.array([1, -1, 1, -1])

        E_aligned = market.energy(spins_aligned)
        E_random = market.energy(spins_random)

        assert E_aligned < E_random, "Aligned spins should have lower energy"


class TestRegimeClassification:
    """Test regime classification function."""

    def test_efficient_regime(self):
        """Theta < 0.2 should be EFFICIENT."""
        regime = classify_regime(0.1, 0.0)
        assert regime == MarketRegime.EFFICIENT

    def test_normal_regime(self):
        """0.2 <= theta < 0.5 should be NORMAL."""
        regime = classify_regime(0.3, 0.1)
        assert regime == MarketRegime.NORMAL

    def test_trending_regime(self):
        """0.5 <= theta < 0.8 should be TRENDING."""
        regime = classify_regime(0.6, 0.5)
        assert regime == MarketRegime.TRENDING

    def test_bubble_regime(self):
        """theta >= 0.8 with positive sentiment should be BUBBLE."""
        regime = classify_regime(0.9, 0.8)
        assert regime == MarketRegime.BUBBLE

    def test_crash_regime(self):
        """theta >= 0.8 with negative sentiment should be CRASH."""
        regime = classify_regime(0.9, -0.9)
        assert regime == MarketRegime.CRASH


class TestCouplingFromCorrelation:
    """Test coupling inference from correlation matrix."""

    def test_zero_correlation_gives_low_coupling(self):
        """No correlation should give low coupling."""
        n = 5
        C = np.eye(n)  # No off-diagonal correlation
        J = compute_coupling_from_correlation(C)
        assert J < 0.5, "No correlation should give low coupling"

    def test_high_correlation_gives_high_coupling(self):
        """High correlation should give high coupling."""
        n = 5
        C = 0.9 * np.ones((n, n)) + 0.1 * np.eye(n)
        J = compute_coupling_from_correlation(C)
        # High correlation should give measurable coupling
        assert J > 0


class TestPhaseTransition:
    """Test phase transition detection."""

    def test_detect_transition(self):
        """Test detecting a phase transition."""
        # Create data with sudden change
        temperatures = np.linspace(0.5, 1.5, 20)
        # Order parameter drops at T=1.0
        order_params = [0.9 if T < 1.0 else 0.1 for T in temperatures]

        detected, T_c = detect_phase_transition(order_params, temperatures.tolist())
        assert detected, "Should detect the phase transition"
        assert 0.8 < T_c < 1.2, f"Critical temperature should be near 1.0: {T_c}"

    def test_no_transition(self):
        """Test when there's no phase transition."""
        temperatures = np.linspace(0.5, 1.5, 20)
        # Constant order parameter
        order_params = [0.5] * 20

        detected, _ = detect_phase_transition(order_params, temperatures.tolist())
        assert not detected, "Should not detect transition for constant data"

    def test_too_few_points(self):
        """Test with insufficient data."""
        detected, _ = detect_phase_transition([0.5, 0.4], [1.0, 1.1])
        assert not detected, "Should not detect with too few points"


class TestCriticalExponents:
    """Test critical exponents are defined correctly."""

    def test_mean_field_exponents(self):
        """Mean-field exponents should have expected values."""
        assert MEAN_FIELD_EXPONENTS["beta"] == 0.5
        assert MEAN_FIELD_EXPONENTS["gamma"] == 1.0
        assert MEAN_FIELD_EXPONENTS["nu"] == 0.5

    def test_ising_3d_exponents(self):
        """3D Ising exponents should be non-trivial."""
        # 3D exponents differ from mean-field
        assert ISING_3D_EXPONENTS["beta"] != MEAN_FIELD_EXPONENTS["beta"]
        assert 0.3 < ISING_3D_EXPONENTS["beta"] < 0.35


class TestMarketSystemProperties:
    """Test individual market system properties."""

    def test_efficient_market_properties(self):
        """Efficient market should have weak correlations."""
        efficient = ECONOMIC_SYSTEMS["efficient_market"]
        assert efficient.order_parameter == 0.0
        assert efficient.correlation == 0.0

    def test_crash_properties(self):
        """Market crash should have negative sentiment."""
        crash = ECONOMIC_SYSTEMS["market_crash"]
        assert crash.order_parameter < 0
        assert crash.correlation > 0.5

    def test_bubble_properties(self):
        """Bubble should have positive sentiment."""
        bubble = ECONOMIC_SYSTEMS["dotcom_bubble"]
        assert bubble.order_parameter > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
