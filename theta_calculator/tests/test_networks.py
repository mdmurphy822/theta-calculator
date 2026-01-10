"""
Tests for Networks Domain Module

Tests cover:
- Shannon channel capacity
- Network percolation
- Quantum key distribution
- Scale-free networks
- Theta range validation [0, 1]
"""

import pytest
import numpy as np

from theta_calculator.domains.networks import (
    NetworkSystem,
    NetworkType,
    ConnectivityRegime,
    compute_network_theta,
    compute_shannon_theta,
    compute_percolation_theta,
    compute_qkd_theta,
    compute_scalefree_theta,
    shannon_capacity,
    snr_from_db,
    percolation_threshold,
    giant_component_fraction,
    bb84_key_rate,
    barabasi_albert_degree_dist,
    classify_connectivity,
    NETWORK_SYSTEMS,
)


class TestNetworkSystemsExist:
    """Test that example network systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """NETWORK_SYSTEMS dict should exist."""
        assert NETWORK_SYSTEMS is not None
        assert isinstance(NETWORK_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 6 systems."""
        assert len(NETWORK_SYSTEMS) >= 6

    def test_key_systems_defined(self):
        """Key networks should be defined."""
        expected = ["wifi_home", "5g_cellular", "qkd_link", "social_sparse"]
        for name in expected:
            assert name in NETWORK_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in NETWORK_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "network_type")
            assert hasattr(system, "n_nodes")
            assert hasattr(system, "connection_probability")


class TestShannonCapacity:
    """Test Shannon channel capacity calculation."""

    def test_basic_formula(self):
        """Verify C = B * log2(1 + SNR)."""
        B = 1e6  # 1 MHz
        SNR = 1  # 0 dB
        C = shannon_capacity(B, SNR)
        expected = B * np.log2(2)
        assert C == pytest.approx(expected)

    def test_high_snr(self):
        """High SNR gives higher capacity."""
        B = 1e6
        C_low = shannon_capacity(B, 1)
        C_high = shannon_capacity(B, 100)
        assert C_high > C_low

    def test_wider_bandwidth(self):
        """Wider bandwidth gives higher capacity."""
        SNR = 10
        C_narrow = shannon_capacity(1e6, SNR)
        C_wide = shannon_capacity(10e6, SNR)
        assert C_wide == pytest.approx(10 * C_narrow)

    def test_zero_bandwidth(self):
        """Zero bandwidth gives zero capacity."""
        C = shannon_capacity(0, 10)
        assert C == 0.0

    def test_zero_snr(self):
        """Zero SNR gives zero capacity (log2(1) = 0)."""
        C = shannon_capacity(1e6, 0)
        assert C == 0.0

    def test_negative_snr(self):
        """Negative SNR gives zero capacity."""
        C = shannon_capacity(1e6, -1)
        assert C == 0.0


class TestSNRFromDB:
    """Test dB to linear conversion."""

    def test_zero_db(self):
        """0 dB = 1 linear."""
        assert snr_from_db(0) == pytest.approx(1.0)

    def test_ten_db(self):
        """10 dB = 10 linear."""
        assert snr_from_db(10) == pytest.approx(10.0)

    def test_twenty_db(self):
        """20 dB = 100 linear."""
        assert snr_from_db(20) == pytest.approx(100.0)

    def test_negative_db(self):
        """Negative dB gives < 1."""
        assert snr_from_db(-10) == pytest.approx(0.1)


class TestShannonTheta:
    """Test Shannon theta calculation."""

    def test_at_capacity(self):
        """Achieving capacity gives theta = 1."""
        B, SNR = 1e6, 10
        C = shannon_capacity(B, SNR)
        theta = compute_shannon_theta(C, B, SNR)
        assert theta == pytest.approx(1.0)

    def test_half_capacity(self):
        """Half capacity gives theta = 0.5."""
        B, SNR = 1e6, 10
        C = shannon_capacity(B, SNR)
        theta = compute_shannon_theta(C / 2, B, SNR)
        assert theta == pytest.approx(0.5)

    def test_zero_rate(self):
        """Zero rate gives theta = 0."""
        theta = compute_shannon_theta(0, 1e6, 10)
        assert theta == 0.0

    def test_exceeds_capacity(self):
        """Exceeding capacity clips to 1."""
        B, SNR = 1e6, 10
        C = shannon_capacity(B, SNR)
        theta = compute_shannon_theta(C * 2, B, SNR)
        assert theta == 1.0


class TestPercolationThreshold:
    """Test percolation threshold values."""

    def test_2d_square(self):
        """2D square lattice threshold."""
        p_c = percolation_threshold("2d_square")
        assert p_c == pytest.approx(0.5927, rel=0.01)

    def test_2d_triangular(self):
        """2D triangular lattice is exactly 0.5."""
        p_c = percolation_threshold("2d_triangular")
        assert p_c == 0.5

    def test_3d_cubic(self):
        """3D cubic lattice threshold."""
        p_c = percolation_threshold("3d_cubic")
        assert p_c == pytest.approx(0.3116, rel=0.01)

    def test_unknown_type(self):
        """Unknown type returns default 0.5."""
        p_c = percolation_threshold("unknown")
        assert p_c == 0.5


class TestGiantComponentFraction:
    """Test giant component fraction calculation."""

    def test_below_threshold(self):
        """Below threshold, fraction = 0."""
        fraction = giant_component_fraction(0.4, p_c=0.5)
        assert fraction == 0.0

    def test_at_threshold(self):
        """At threshold, fraction = 0."""
        fraction = giant_component_fraction(0.5, p_c=0.5)
        assert fraction == 0.0

    def test_above_threshold(self):
        """Above threshold, fraction > 0."""
        fraction = giant_component_fraction(0.7, p_c=0.5)
        assert fraction > 0

    def test_fully_connected(self):
        """p = 1 gives fraction = 1."""
        fraction = giant_component_fraction(1.0, p_c=0.5)
        assert fraction == pytest.approx(1.0)


class TestPercolationTheta:
    """Test percolation theta calculation."""

    def test_zero_connection(self):
        """Zero connection probability gives theta = 0."""
        theta = compute_percolation_theta(0, p_c=0.5)
        assert theta == 0.0

    def test_at_threshold(self):
        """At threshold, theta = 0.5 (threshold transition)."""
        theta = compute_percolation_theta(0.5, p_c=0.5)
        # At exactly p_c: theta = p_c + (1-p_c)*0 = 0.5
        assert theta == pytest.approx(0.5)

    def test_below_threshold(self):
        """Below threshold, theta scales with p/p_c."""
        theta = compute_percolation_theta(0.25, p_c=0.5)
        assert theta == pytest.approx(0.5)

    def test_above_threshold(self):
        """Above threshold, theta includes giant component."""
        theta = compute_percolation_theta(0.8, p_c=0.5)
        assert theta > 0.5

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for p in [0, 0.25, 0.5, 0.75, 1.0]:
            theta = compute_percolation_theta(p)
            assert 0 <= theta <= 1


class TestBB84KeyRate:
    """Test BB84 QKD key rate calculation."""

    def test_zero_qber(self):
        """Zero QBER gives maximum key rate."""
        raw, secure = bb84_key_rate(
            pulse_rate=1e9,
            detection_efficiency=0.1,
            channel_loss=0.1,
            dark_count_rate=1e-6,
            qber=0.0
        )
        assert secure == raw  # h(0) = 0

    def test_high_qber(self):
        """QBER > 11% gives zero secure rate."""
        raw, secure = bb84_key_rate(
            pulse_rate=1e9,
            detection_efficiency=0.1,
            channel_loss=0.1,
            dark_count_rate=1e-6,
            qber=0.15
        )
        assert secure == 0.0

    def test_threshold_qber(self):
        """QBER > 11% gives zero secure rate."""
        raw, secure = bb84_key_rate(
            pulse_rate=1e9,
            detection_efficiency=0.1,
            channel_loss=0.1,
            dark_count_rate=1e-6,
            qber=0.12
        )
        assert secure == 0.0


class TestQKDTheta:
    """Test QKD advantage theta."""

    def test_qkd_viable(self):
        """Viable QKD has positive theta."""
        theta = compute_qkd_theta(1000, 10000)
        assert theta > 0

    def test_no_secure_rate(self):
        """Zero secure rate gives theta = 0."""
        theta = compute_qkd_theta(0, 10000)
        assert theta == 0.0

    def test_zero_classical(self):
        """Zero classical rate with positive QKD gives theta = 1."""
        theta = compute_qkd_theta(1000, 0)
        assert theta == 1.0


class TestBarabasiAlbert:
    """Test Barabasi-Albert degree distribution."""

    def test_minimum_degree(self):
        """k < m gives P(k) = 0."""
        prob = barabasi_albert_degree_dist(1, m=2)
        assert prob == 0.0

    def test_power_law(self):
        """P(k) ~ k^(-3) for BA model."""
        p_4 = barabasi_albert_degree_dist(4, m=2)
        p_8 = barabasi_albert_degree_dist(8, m=2)
        # P(8) / P(4) should be (8/4)^(-3) = 1/8
        ratio = p_8 / p_4
        assert ratio == pytest.approx(1/8)


class TestScaleFreeTheta:
    """Test scale-free network theta."""

    def test_ba_model(self):
        """BA model gamma = 3 gives theta = 0.5."""
        theta = compute_scalefree_theta(3.0, gamma_min=2.0, gamma_max=4.0)
        assert theta == pytest.approx(0.5)

    def test_hub_dominated(self):
        """Low gamma (hub-dominated) gives high theta."""
        theta = compute_scalefree_theta(2.0, gamma_min=2.0, gamma_max=4.0)
        assert theta == 1.0

    def test_random_like(self):
        """High gamma (random-like) gives low theta."""
        theta = compute_scalefree_theta(4.0, gamma_min=2.0, gamma_max=4.0)
        assert theta == 0.0


class TestUnifiedNetworkTheta:
    """Test unified network theta calculation."""

    def test_all_systems_valid_theta(self):
        """All network systems should have theta in [0, 1]."""
        for name, system in NETWORK_SYSTEMS.items():
            theta = compute_network_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_quantum_network_high_theta(self):
        """QKD network should have high theta."""
        qkd = NETWORK_SYSTEMS["qkd_link"]
        theta = compute_network_theta(qkd)
        assert theta > 0.5

    def test_communication_network(self):
        """Communication network has typical theta."""
        wifi = NETWORK_SYSTEMS["wifi_home"]
        theta = compute_network_theta(wifi)
        assert 0.5 < theta < 1.0


class TestClassifyConnectivity:
    """Test connectivity regime classification."""

    def test_disconnected(self):
        """Low theta -> DISCONNECTED."""
        assert classify_connectivity(0.2, p_c=0.5) == ConnectivityRegime.DISCONNECTED

    def test_critical(self):
        """Near threshold -> CRITICAL."""
        assert classify_connectivity(0.45, p_c=0.5) == ConnectivityRegime.CRITICAL
        assert classify_connectivity(0.55, p_c=0.5) == ConnectivityRegime.CRITICAL

    def test_connected(self):
        """Above threshold -> CONNECTED."""
        assert classify_connectivity(0.7, p_c=0.5) == ConnectivityRegime.CONNECTED

    def test_fully_connected(self):
        """High theta -> FULLY_CONNECTED."""
        assert classify_connectivity(0.98, p_c=0.5) == ConnectivityRegime.FULLY_CONNECTED


class TestEnums:
    """Test enum definitions."""

    def test_network_types(self):
        """All network types should be defined."""
        assert NetworkType.COMMUNICATION.value == "communication"
        assert NetworkType.SOCIAL.value == "social"
        assert NetworkType.QUANTUM.value == "quantum"
        assert NetworkType.INFRASTRUCTURE.value == "infrastructure"

    def test_connectivity_regimes(self):
        """All connectivity regimes should be defined."""
        assert ConnectivityRegime.DISCONNECTED.value == "disconnected"
        assert ConnectivityRegime.CRITICAL.value == "critical"
        assert ConnectivityRegime.CONNECTED.value == "connected"
        assert ConnectivityRegime.FULLY_CONNECTED.value == "fully_connected"


class TestNetworkSystemDataclass:
    """Test NetworkSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with required parameters."""
        system = NetworkSystem(
            name="Test",
            network_type=NetworkType.SOCIAL,
            n_nodes=100,
            n_edges=200,
            connection_probability=0.04
        )
        assert system.name == "Test"
        assert system.n_nodes == 100

    def test_default_values(self):
        """Optional fields have defaults."""
        system = NetworkSystem(
            name="Test",
            network_type=NetworkType.SOCIAL,
            n_nodes=100,
            n_edges=200,
            connection_probability=0.5
        )
        assert system.bandwidth == 0.0
        assert system.snr == 1.0
        assert system.is_quantum is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_snr(self):
        """Very high SNR gives high capacity."""
        C = shannon_capacity(1e6, 1e6)
        assert C > shannon_capacity(1e6, 100)

    def test_very_low_connection(self):
        """Very low connection probability gives low theta."""
        theta = compute_percolation_theta(0.001, p_c=0.5)
        assert theta < 0.01

    def test_negative_connection(self):
        """Negative connection probability gives theta = 0."""
        theta = compute_percolation_theta(-0.1, p_c=0.5)
        assert theta == 0.0
