"""
Tests for new domain modules: education, mechanical, networks, cognition, social, chemistry.

These tests verify the theta mappings for creative domain extensions.
"""

import pytest
import numpy as np

from theta_calculator.domains import (
    # Education
    LearningSystem,
    compute_education_theta,
    compute_retention_theta,
    compute_learning_theta,
    ebbinghaus_retention,
    EDUCATION_SYSTEMS,
    # Mechanical Systems
    MechanicalSystem,
    compute_mechanical_theta,
    compute_engine_theta,
    carnot_efficiency,
    MECHANICAL_SYSTEMS,
    # Networks
    NetworkSystem,
    compute_network_theta,
    compute_shannon_theta,
    shannon_capacity,
    compute_percolation_theta,
    NETWORK_SYSTEMS,
    # Cognition
    CognitiveSystem,
    compute_cognition_theta,
    compute_phi_ratio,
    compute_criticality_theta,
    compute_working_memory_theta,
    COGNITIVE_SYSTEMS,
    # Social Systems
    SocialSystem,
    compute_social_theta,
    compute_epidemic_theta,
    compute_traffic_theta,
    herd_immunity_threshold,
    SOCIAL_SYSTEMS,
    # Chemistry
    QuantumMaterial,
    compute_chemistry_theta,
    compute_superconductor_theta,
    compute_bec_theta,
    bec_condensate_fraction,
    SUPERCONDUCTORS,
)


class TestEducationDomain:
    """Tests for education domain theta mappings."""

    def test_education_systems_exist(self):
        """Verify example education systems are defined."""
        assert len(EDUCATION_SYSTEMS) >= 5
        assert "cramming" in EDUCATION_SYSTEMS
        assert "spaced_repetition" in EDUCATION_SYSTEMS

    def test_retention_theta_range(self):
        """Retention theta should be in [0, 1]."""
        for strength in [1, 10, 100, 168]:
            result = compute_retention_theta(10, strength)
            assert 0 <= result.theta <= 1

    def test_long_term_memory_high_theta(self):
        """Strong memory (long decay constant) should have high theta."""
        result_strong = compute_retention_theta(10, strength=168)  # 1 week
        result_weak = compute_retention_theta(10, strength=1)  # 1 hour
        assert result_strong.theta > result_weak.theta

    def test_ebbinghaus_decay(self):
        """Ebbinghaus retention should decay over time."""
        retention_0 = ebbinghaus_retention(0, 10)
        retention_10 = ebbinghaus_retention(10, 10)
        retention_100 = ebbinghaus_retention(100, 10)
        assert retention_0 == pytest.approx(1.0)
        assert retention_10 < retention_0
        assert retention_100 < retention_10

    def test_learning_curve_theta(self):
        """Learning rate should map to theta."""
        result = compute_learning_theta(trials=10, initial_time=100, current_time=50)
        assert 0 <= result.theta <= 1
        assert result.learning_rate > 0

    def test_education_theta_ordering(self):
        """Better learning methods should have higher theta."""
        theta_cramming = compute_education_theta(EDUCATION_SYSTEMS["cramming"])
        theta_spaced = compute_education_theta(EDUCATION_SYSTEMS["spaced_repetition"])
        assert theta_spaced > theta_cramming


class TestMechanicalSystemsDomain:
    """Tests for mechanical systems domain theta mappings."""

    def test_mechanical_systems_exist(self):
        """Verify example mechanical systems are defined."""
        assert len(MECHANICAL_SYSTEMS) >= 5
        assert "car_engine" in MECHANICAL_SYSTEMS
        assert "ev_motor" in MECHANICAL_SYSTEMS

    def test_carnot_efficiency(self):
        """Carnot efficiency should be correct."""
        # T_hot = 500K, T_cold = 300K
        eta = carnot_efficiency(500, 300)
        assert eta == pytest.approx(0.4, rel=0.01)

    def test_carnot_requires_hot_greater_than_cold(self):
        """Carnot should raise if T_hot <= T_cold."""
        with pytest.raises(ValueError):
            carnot_efficiency(300, 500)

    def test_engine_theta(self):
        """Engine theta should be efficiency ratio."""
        # 25% actual, 50% Carnot = 0.5 theta
        theta = compute_engine_theta(0.25, T_hot=600, T_cold=300)
        expected_carnot = 0.5  # 1 - 300/600
        expected_theta = 0.25 / expected_carnot
        assert theta == pytest.approx(expected_theta, rel=0.01)

    def test_mechanical_theta_range(self):
        """All mechanical thetas should be in [0, 1]."""
        for name, system in MECHANICAL_SYSTEMS.items():
            theta = compute_mechanical_theta(system)
            assert 0 <= theta <= 1, f"{name} has theta {theta}"

    def test_high_efficiency_high_theta(self):
        """High efficiency systems should have high theta."""
        ev_motor = MECHANICAL_SYSTEMS["ev_motor"]
        car_engine = MECHANICAL_SYSTEMS["car_engine"]
        assert compute_mechanical_theta(ev_motor) > compute_mechanical_theta(car_engine)


class TestNetworksDomain:
    """Tests for networks domain theta mappings."""

    def test_network_systems_exist(self):
        """Verify example network systems are defined."""
        assert len(NETWORK_SYSTEMS) >= 5
        assert "wifi_home" in NETWORK_SYSTEMS

    def test_shannon_capacity(self):
        """Shannon capacity formula should be correct."""
        # C = B * log2(1 + SNR)
        capacity = shannon_capacity(bandwidth=1e6, snr=10)
        expected = 1e6 * np.log2(11)  # ~3.46 Mbps
        assert capacity == pytest.approx(expected, rel=0.01)

    def test_shannon_theta(self):
        """Shannon theta should be rate/capacity ratio."""
        capacity = shannon_capacity(1e6, 10)
        theta = compute_shannon_theta(0.5 * capacity, 1e6, 10)
        assert theta == pytest.approx(0.5, rel=0.01)

    def test_percolation_below_threshold(self):
        """Below percolation threshold, network is disconnected."""
        theta = compute_percolation_theta(p=0.3, p_c=0.5)
        assert theta < 1.0

    def test_percolation_above_threshold(self):
        """Above percolation threshold, giant component exists."""
        theta = compute_percolation_theta(p=0.8, p_c=0.5)
        assert theta > 0.5

    def test_network_theta_range(self):
        """All network thetas should be in [0, 1]."""
        for name, system in NETWORK_SYSTEMS.items():
            theta = compute_network_theta(system)
            assert 0 <= theta <= 1


class TestCognitionDomain:
    """Tests for cognition domain theta mappings."""

    def test_cognitive_systems_exist(self):
        """Verify example cognitive systems are defined."""
        assert len(COGNITIVE_SYSTEMS) >= 5
        assert "flow_state" in COGNITIVE_SYSTEMS
        assert "deep_sleep" in COGNITIVE_SYSTEMS

    def test_phi_ratio(self):
        """Phi ratio should be in [0, 1]."""
        theta = compute_phi_ratio(0.5, phi_max=1.0)
        assert theta == 0.5

    def test_criticality_at_critical(self):
        """At criticality (tau=1.5), theta should be 1."""
        theta = compute_criticality_theta(tau=1.5, tau_critical=1.5)
        assert theta == pytest.approx(1.0, rel=0.01)

    def test_criticality_away_from_critical(self):
        """Away from criticality, theta decreases."""
        theta_at = compute_criticality_theta(tau=1.5)
        theta_away = compute_criticality_theta(tau=2.0)
        assert theta_at > theta_away

    def test_working_memory(self):
        """Working memory theta should be load/capacity."""
        theta = compute_working_memory_theta(items=5, capacity=7)
        assert theta == pytest.approx(5/7, rel=0.01)

    def test_cognition_theta_range(self):
        """All cognition thetas should be in [0, 1]."""
        for name, system in COGNITIVE_SYSTEMS.items():
            theta = compute_cognition_theta(system)
            assert 0 <= theta <= 1

    def test_flow_state_high_theta(self):
        """Flow state should have high theta."""
        flow = COGNITIVE_SYSTEMS["flow_state"]
        sleep = COGNITIVE_SYSTEMS["deep_sleep"]
        theta_flow = compute_cognition_theta(flow)
        theta_sleep = compute_cognition_theta(sleep)
        assert theta_flow > theta_sleep


class TestSocialSystemsDomain:
    """Tests for social systems domain theta mappings."""

    def test_social_systems_exist(self):
        """Verify example social systems are defined."""
        assert len(SOCIAL_SYSTEMS) >= 3
        assert "polarized_society" in SOCIAL_SYSTEMS

    def test_epidemic_threshold(self):
        """R0 = 1 is the critical threshold."""
        theta_below = compute_epidemic_theta(R0=0.8)
        theta_at = compute_epidemic_theta(R0=1.0)
        theta_above = compute_epidemic_theta(R0=2.0)

        assert theta_below < 0.5  # Dying out
        assert theta_at == pytest.approx(0.5, rel=0.1)  # Critical
        assert theta_above > 0.5  # Spreading

    def test_herd_immunity(self):
        """Herd immunity threshold should be 1 - 1/R0."""
        hit = herd_immunity_threshold(R0=4.0)
        assert hit == pytest.approx(0.75, rel=0.01)

    def test_traffic_free_flow(self):
        """Low density should be free flow."""
        theta = compute_traffic_theta(density=0.1, critical_density=0.3)
        assert theta < 0.5

    def test_traffic_congestion(self):
        """High density should be congested."""
        theta = compute_traffic_theta(density=0.5, critical_density=0.3)
        assert theta > 0.9

    def test_social_theta_range(self):
        """All social thetas should be in [0, 1]."""
        for name, system in SOCIAL_SYSTEMS.items():
            theta = compute_social_theta(system)
            assert 0 <= theta <= 1


class TestChemistryDomain:
    """Tests for chemistry domain theta mappings."""

    def test_superconductors_exist(self):
        """Verify example superconductors are defined."""
        assert len(SUPERCONDUCTORS) >= 3
        assert "niobium" in SUPERCONDUCTORS
        assert "YBCO" in SUPERCONDUCTORS

    def test_superconductor_above_Tc(self):
        """Above Tc, superconductor is normal."""
        theta = compute_superconductor_theta(T=100, T_c=9.3)  # Nb at 100K
        assert theta < 0.2  # Normal state

    def test_superconductor_below_Tc(self):
        """Below Tc, superconductor is quantum."""
        theta = compute_superconductor_theta(T=4, T_c=9.3)  # Nb at 4K
        assert theta > 0.5  # Superconducting

    def test_bec_above_Tc(self):
        """Above BEC Tc, condensate fraction is 0."""
        fraction = bec_condensate_fraction(T=200e-9, T_c=170e-9)
        assert fraction == 0.0

    def test_bec_below_Tc(self):
        """Below BEC Tc, condensate fraction > 0."""
        fraction = bec_condensate_fraction(T=100e-9, T_c=170e-9)
        assert fraction > 0.5

    def test_chemistry_theta_range(self):
        """All chemistry thetas should be in [0, 1]."""
        for name, material in SUPERCONDUCTORS.items():
            theta = compute_chemistry_theta(material)
            assert 0 <= theta <= 1


class TestCrossDomainConsistency:
    """Tests for consistency across domains."""

    def test_all_domains_have_examples(self):
        """Each domain should have example systems."""
        assert len(EDUCATION_SYSTEMS) > 0
        assert len(MECHANICAL_SYSTEMS) > 0
        assert len(NETWORK_SYSTEMS) > 0
        assert len(COGNITIVE_SYSTEMS) > 0
        assert len(SOCIAL_SYSTEMS) > 0
        assert len(SUPERCONDUCTORS) > 0

    def test_theta_interpretation_consistent(self):
        """High theta should mean 'more quantum/optimal'."""
        # Education: spaced > cramming
        assert compute_education_theta(EDUCATION_SYSTEMS["spaced_repetition"]) > \
               compute_education_theta(EDUCATION_SYSTEMS["cramming"])

        # Cognition: flow > sleep
        assert compute_cognition_theta(COGNITIVE_SYSTEMS["flow_state"]) > \
               compute_cognition_theta(COGNITIVE_SYSTEMS["deep_sleep"])

        # Chemistry: below Tc > above Tc
        nb = SUPERCONDUCTORS["niobium"]
        theta_cold = compute_superconductor_theta(4.0, nb.critical_temperature)
        theta_hot = compute_superconductor_theta(100.0, nb.critical_temperature)
        assert theta_cold > theta_hot
