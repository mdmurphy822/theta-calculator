"""
Tests for Social Systems Domain Module

Tests cover:
- Opinion dynamics and polarization
- Epidemic spreading (SIR model)
- Urban scaling laws
- Traffic flow dynamics
- Theta range validation [0, 1]
"""

import pytest
import numpy as np

from theta_calculator.domains.social_systems import (
    SocialSystem,
    SocialPhase,
    EpidemicPhase,
    TrafficPhase,
    SIRState,
    compute_social_theta,
    compute_opinion_theta,
    compute_epidemic_theta,
    compute_urban_theta,
    compute_traffic_theta,
    compute_polarization,
    compute_R0,
    herd_immunity_threshold,
    urban_scaling_exponent,
    fundamental_diagram,
    classify_social_phase,
    classify_epidemic,
    classify_traffic,
    SOCIAL_SYSTEMS,
    EPIDEMIC_EXAMPLES,
    TRAFFIC_EXAMPLES,
)


class TestSocialSystemsExist:
    """Test that example social systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """SOCIAL_SYSTEMS dict should exist."""
        assert SOCIAL_SYSTEMS is not None
        assert isinstance(SOCIAL_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 6 systems."""
        assert len(SOCIAL_SYSTEMS) >= 6

    def test_key_systems_defined(self):
        """Key social systems should be defined."""
        expected = ["diverse_democracy", "polarized_society", "echo_chamber"]
        for name in expected:
            assert name in SOCIAL_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in SOCIAL_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "population")
            assert hasattr(system, "polarization")
            assert hasattr(system, "clustering")


class TestEpidemicExamples:
    """Test epidemic examples."""

    def test_epidemic_examples_exist(self):
        """EPIDEMIC_EXAMPLES dict should exist."""
        assert EPIDEMIC_EXAMPLES is not None
        assert len(EPIDEMIC_EXAMPLES) >= 4

    def test_epidemic_has_r0(self):
        """Each epidemic should have R0."""
        for name, epi in EPIDEMIC_EXAMPLES.items():
            assert "R0" in epi, f"{name} missing R0"
            assert epi["R0"] > 0


class TestTrafficExamples:
    """Test traffic examples."""

    def test_traffic_examples_exist(self):
        """TRAFFIC_EXAMPLES dict should exist."""
        assert TRAFFIC_EXAMPLES is not None
        assert len(TRAFFIC_EXAMPLES) >= 4

    def test_traffic_has_density(self):
        """Each traffic example should have density."""
        for name, traffic in TRAFFIC_EXAMPLES.items():
            assert "density" in traffic, f"{name} missing density"
            assert 0 <= traffic["density"] <= 1


class TestPolarization:
    """Test opinion polarization calculation."""

    def test_single_opinion(self):
        """Single opinion gives zero polarization."""
        pol = compute_polarization([0.5])
        assert pol == 0.0

    def test_consensus(self):
        """All same opinion gives low polarization."""
        opinions = [0.5] * 100
        pol = compute_polarization(opinions)
        assert pol < 0.1

    def test_bimodal(self):
        """Bimodal distribution gives high polarization."""
        opinions = [-1.0] * 50 + [1.0] * 50
        pol = compute_polarization(opinions)
        assert pol > 0.5

    def test_uniform(self):
        """Uniform distribution gives moderate polarization."""
        opinions = list(np.linspace(-1, 1, 100))
        pol = compute_polarization(opinions)
        assert 0.1 < pol < 0.5

    def test_polarization_range(self):
        """Polarization should be in [0, 1]."""
        for _ in range(10):
            opinions = list(np.random.uniform(-1, 1, 100))
            pol = compute_polarization(opinions)
            assert 0 <= pol <= 1


class TestOpinionTheta:
    """Test opinion dynamics theta."""

    def test_high_polarization_high_theta(self):
        """High polarization gives high theta."""
        theta = compute_opinion_theta(0.9)
        assert theta == 0.9

    def test_low_polarization_low_theta(self):
        """Low polarization gives low theta."""
        theta = compute_opinion_theta(0.1)
        assert theta == 0.1

    def test_theta_equals_polarization(self):
        """Theta directly maps to polarization."""
        for pol in [0.0, 0.3, 0.5, 0.8, 1.0]:
            theta = compute_opinion_theta(pol)
            assert theta == pol


class TestComputeR0:
    """Test basic reproduction number calculation."""

    def test_formula(self):
        """Verify R0 = (transmission * contacts) / recovery."""
        R0 = compute_R0(
            transmission_rate=0.1,
            recovery_rate=0.5,
            contacts_per_day=10
        )
        expected = (0.1 * 10) / 0.5
        assert R0 == pytest.approx(expected)

    def test_high_transmission(self):
        """High transmission gives high R0."""
        R0_low = compute_R0(0.05, 0.5, 10)
        R0_high = compute_R0(0.2, 0.5, 10)
        assert R0_high > R0_low

    def test_fast_recovery(self):
        """Faster recovery gives lower R0."""
        R0_slow = compute_R0(0.1, 0.2, 10)
        R0_fast = compute_R0(0.1, 1.0, 10)
        assert R0_fast < R0_slow

    def test_zero_recovery(self):
        """Zero recovery rate gives infinite R0."""
        R0 = compute_R0(0.1, 0.0, 10)
        assert R0 == float('inf')


class TestEpidemicTheta:
    """Test epidemic theta calculation."""

    def test_dying_out(self):
        """R0 < 1 gives low theta."""
        theta = compute_epidemic_theta(0.5)
        assert theta < 0.5

    def test_at_critical(self):
        """R0 = 1 gives theta = 0.5."""
        theta = compute_epidemic_theta(1.0)
        assert theta == pytest.approx(0.5)

    def test_spreading(self):
        """R0 > 1 gives theta > 0.5."""
        theta = compute_epidemic_theta(3.0)
        assert theta > 0.5

    def test_explosive(self):
        """High R0 gives high theta."""
        theta = compute_epidemic_theta(15.0)
        assert theta > 0.9

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for R0 in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0]:
            theta = compute_epidemic_theta(R0)
            assert 0 <= theta <= 1


class TestHerdImmunityThreshold:
    """Test herd immunity threshold calculation."""

    def test_r0_below_one(self):
        """R0 < 1 gives HIT = 0."""
        HIT = herd_immunity_threshold(0.8)
        assert HIT == 0.0

    def test_r0_equals_one(self):
        """R0 = 1 gives HIT = 0."""
        HIT = herd_immunity_threshold(1.0)
        assert HIT == 0.0

    def test_formula(self):
        """Verify HIT = 1 - 1/R0."""
        R0 = 4.0
        HIT = herd_immunity_threshold(R0)
        assert HIT == pytest.approx(1 - 1/R0)

    def test_measles(self):
        """Measles (R0=15) needs ~93% immunity."""
        HIT = herd_immunity_threshold(15.0)
        assert HIT == pytest.approx(0.933, rel=0.01)

    def test_covid(self):
        """COVID (R0=2.5) needs 60% immunity."""
        HIT = herd_immunity_threshold(2.5)
        assert HIT == pytest.approx(0.6)


class TestClassifyEpidemic:
    """Test epidemic phase classification."""

    def test_dying_out(self):
        """R0 < 0.9 -> DYING_OUT."""
        assert classify_epidemic(0.5) == EpidemicPhase.DYING_OUT
        assert classify_epidemic(0.8) == EpidemicPhase.DYING_OUT

    def test_critical(self):
        """R0 near 1 -> CRITICAL."""
        assert classify_epidemic(0.95) == EpidemicPhase.CRITICAL
        assert classify_epidemic(1.05) == EpidemicPhase.CRITICAL

    def test_spreading(self):
        """1.1 < R0 < 3 -> SPREADING."""
        assert classify_epidemic(1.5) == EpidemicPhase.SPREADING
        assert classify_epidemic(2.5) == EpidemicPhase.SPREADING

    def test_explosive(self):
        """R0 > 3 -> EXPLOSIVE."""
        assert classify_epidemic(5.0) == EpidemicPhase.EXPLOSIVE
        assert classify_epidemic(15.0) == EpidemicPhase.EXPLOSIVE


class TestUrbanScalingExponent:
    """Test urban scaling exponent calculation."""

    def test_insufficient_data(self):
        """Too few cities returns default 1.0."""
        beta = urban_scaling_exponent([100, 200], [10, 20])
        assert beta == 1.0

    def test_linear_scaling(self):
        """Linear relationship gives beta = 1."""
        sizes = [100, 1000, 10000]
        metrics = [10, 100, 1000]  # Perfect linear
        beta = urban_scaling_exponent(sizes, metrics)
        assert beta == pytest.approx(1.0, rel=0.1)

    def test_superlinear_scaling(self):
        """Superlinear gives beta > 1."""
        sizes = [100, 1000, 10000]
        # Y ~ N^1.15 (superlinear)
        metrics = [s ** 1.15 for s in sizes]
        beta = urban_scaling_exponent(sizes, metrics)
        assert beta > 1.0


class TestUrbanTheta:
    """Test urban scaling theta."""

    def test_linear_scaling(self):
        """Linear scaling (beta=1) gives theta = 0."""
        theta = compute_urban_theta(1.0)
        assert theta == 0.0

    def test_typical_superlinear(self):
        """Typical superlinear (beta=1.15) gives theta = 1."""
        theta = compute_urban_theta(1.15)
        assert theta == 1.0

    def test_sublinear(self):
        """Sublinear scaling (β=0.85) gives theta = 0.5."""
        theta = compute_urban_theta(0.85)
        # Infrastructure scaling: theta = (1 - 0.85) / 0.15 * 0.5 = 0.5
        assert theta == pytest.approx(0.5)


class TestFundamentalDiagram:
    """Test traffic fundamental diagram."""

    def test_zero_density(self):
        """Zero density gives zero flow."""
        flow = fundamental_diagram(0)
        assert flow == 0.0

    def test_full_density(self):
        """Full density gives zero flow (gridlock)."""
        flow = fundamental_diagram(1.0)
        assert flow == 0.0

    def test_free_flow(self):
        """Low density gives flow ~ density * v_max."""
        flow = fundamental_diagram(0.1, v_max=1.0, critical_density=0.3)
        assert flow == pytest.approx(0.1)

    def test_congested(self):
        """High density gives reduced flow."""
        flow = fundamental_diagram(0.5, v_max=1.0, critical_density=0.3)
        assert flow == pytest.approx(0.5)  # (1-0.5)*1

    def test_maximum_flow(self):
        """Maximum flow at critical density."""
        flow_low = fundamental_diagram(0.1, critical_density=0.3)
        flow_crit = fundamental_diagram(0.3, critical_density=0.3)
        flow_high = fundamental_diagram(0.5, critical_density=0.3)
        # Flow should be equal at 0.3 from either side
        assert flow_crit >= flow_low
        assert flow_crit >= flow_high


class TestTrafficTheta:
    """Test traffic flow theta."""

    def test_empty_road(self):
        """Zero density gives theta = 0."""
        theta = compute_traffic_theta(0, critical_density=0.3)
        assert theta == 0.0

    def test_at_critical(self):
        """At critical density, theta = 1."""
        theta = compute_traffic_theta(0.3, critical_density=0.3)
        assert theta == 1.0

    def test_below_critical(self):
        """Below critical, theta < 1."""
        theta = compute_traffic_theta(0.15, critical_density=0.3)
        assert theta == pytest.approx(0.5)

    def test_above_critical(self):
        """Above critical (congested), theta stays high."""
        theta = compute_traffic_theta(0.6, critical_density=0.3)
        assert theta > 0.9

    def test_all_examples_valid(self):
        """All traffic examples should have theta in [0, 1]."""
        for name, traffic in TRAFFIC_EXAMPLES.items():
            theta = compute_traffic_theta(traffic["density"])
            assert 0 <= theta <= 1, f"{name}: theta={theta}"


class TestClassifyTraffic:
    """Test traffic phase classification."""

    def test_free_flow(self):
        """Low density -> FREE_FLOW."""
        assert classify_traffic(0.1, rho_c=0.3) == TrafficPhase.FREE_FLOW
        assert classify_traffic(0.2, rho_c=0.3) == TrafficPhase.FREE_FLOW

    def test_synchronized(self):
        """Near critical -> SYNCHRONIZED."""
        assert classify_traffic(0.28, rho_c=0.3) == TrafficPhase.SYNCHRONIZED
        assert classify_traffic(0.32, rho_c=0.3) == TrafficPhase.SYNCHRONIZED

    def test_jammed(self):
        """High density -> JAMMED."""
        assert classify_traffic(0.5, rho_c=0.3) == TrafficPhase.JAMMED
        assert classify_traffic(0.8, rho_c=0.3) == TrafficPhase.JAMMED


class TestUnifiedSocialTheta:
    """Test unified social theta calculation."""

    def test_all_systems_valid_theta(self):
        """All social systems should have theta in [0, 1]."""
        for name, system in SOCIAL_SYSTEMS.items():
            theta = compute_social_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_echo_chamber_high_theta(self):
        """Echo chamber has high theta (collective behavior)."""
        echo = SOCIAL_SYSTEMS["echo_chamber"]
        theta = compute_social_theta(echo)
        assert theta > 0.8

    def test_diverse_democracy_low_theta(self):
        """Diverse democracy has low theta (individual behavior)."""
        diverse = SOCIAL_SYSTEMS["diverse_democracy"]
        theta = compute_social_theta(diverse)
        assert theta < 0.3


class TestClassifySocialPhase:
    """Test social phase classification."""

    def test_individual(self):
        """Low theta -> INDIVIDUAL."""
        assert classify_social_phase(0.1) == SocialPhase.INDIVIDUAL
        assert classify_social_phase(0.25) == SocialPhase.INDIVIDUAL

    def test_clustering(self):
        """Medium theta -> CLUSTERING."""
        assert classify_social_phase(0.4) == SocialPhase.CLUSTERING
        assert classify_social_phase(0.55) == SocialPhase.CLUSTERING

    def test_collective(self):
        """High theta -> COLLECTIVE."""
        assert classify_social_phase(0.7) == SocialPhase.COLLECTIVE
        assert classify_social_phase(0.85) == SocialPhase.COLLECTIVE

    def test_consensus(self):
        """Very high theta -> CONSENSUS."""
        assert classify_social_phase(0.92) == SocialPhase.CONSENSUS
        assert classify_social_phase(0.99) == SocialPhase.CONSENSUS


class TestEnums:
    """Test enum definitions."""

    def test_social_phases(self):
        """All social phases should be defined."""
        assert SocialPhase.INDIVIDUAL.value == "individual"
        assert SocialPhase.CLUSTERING.value == "clustering"
        assert SocialPhase.COLLECTIVE.value == "collective"
        assert SocialPhase.CONSENSUS.value == "consensus"

    def test_epidemic_phases(self):
        """All epidemic phases should be defined."""
        assert EpidemicPhase.DYING_OUT.value == "dying_out"
        assert EpidemicPhase.CRITICAL.value == "critical"
        assert EpidemicPhase.SPREADING.value == "spreading"
        assert EpidemicPhase.EXPLOSIVE.value == "explosive"

    def test_traffic_phases(self):
        """All traffic phases should be defined."""
        assert TrafficPhase.FREE_FLOW.value == "free_flow"
        assert TrafficPhase.SYNCHRONIZED.value == "synchronized"
        assert TrafficPhase.JAMMED.value == "jammed"


class TestDataclasses:
    """Test dataclass definitions."""

    def test_social_system_creation(self):
        """Should create SocialSystem."""
        system = SocialSystem(
            name="Test",
            population=1000,
            connectivity=50,
            polarization=0.5,
            clustering=0.4,
            cascade_probability=0.3
        )
        assert system.name == "Test"
        assert system.population == 1000

    def test_sir_state_creation(self):
        """Should create SIRState."""
        state = SIRState(
            susceptible=0.8,
            infected=0.1,
            recovered=0.1,
            R0=2.5,
            herd_immunity_threshold=0.6
        )
        assert state.R0 == 2.5
        assert state.susceptible + state.infected + state.recovered == pytest.approx(1.0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_polarization(self):
        """Zero polarization gives theta = 0."""
        theta = compute_opinion_theta(0.0)
        assert theta == 0.0

    def test_very_high_r0(self):
        """Very high R0 clips theta to 1."""
        theta = compute_epidemic_theta(100.0)
        assert theta == 1.0

    def test_zero_critical_density(self):
        """Zero critical density gives theta = 0."""
        theta = compute_traffic_theta(0.5, critical_density=0)
        assert theta == 0.0

    def test_empty_opinions(self):
        """Empty opinion list gives zero polarization."""
        pol = compute_polarization([])
        assert pol == 0.0
