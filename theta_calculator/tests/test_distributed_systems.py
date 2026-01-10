"""
Tests for Distributed Systems Domain Module

Tests cover:
- Consistency levels
- Availability calculations
- CAP theorem analysis
- Theta range validation [0, 1]
"""

import pytest

from theta_calculator.domains.distributed_systems import (
    DistributedSystem,
    ConsistencyLevel,
    PartitionTolerance,
    ScalabilityClass,
    ReplicationMode,
    compute_distributed_theta,
    compute_consistency_theta,
    compute_quorum_theta,
    compute_availability_theta,
    compute_latency_theta,
    compute_throughput_theta,
    compute_partition_theta,
    node_availability,
    classify_consistency,
    classify_scalability,
    DISTRIBUTED_SYSTEMS,
)


class TestDistributedSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """DISTRIBUTED_SYSTEMS dict should exist."""
        assert DISTRIBUTED_SYSTEMS is not None
        assert isinstance(DISTRIBUTED_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(DISTRIBUTED_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = ["paxos_raft", "cassandra_ap", "spanner_cp", "kafka"]
        for name in expected:
            assert name in DISTRIBUTED_SYSTEMS, f"Missing system: {name}"


class TestConsistencyTheta:
    """Test consistency theta calculation."""

    def test_eventual(self):
        """Eventual consistency gives low theta."""
        theta = compute_consistency_theta(ConsistencyLevel.EVENTUAL)
        assert theta == 0.25

    def test_linearizable(self):
        """Linearizable gives theta = 1."""
        theta = compute_consistency_theta(ConsistencyLevel.LINEARIZABLE)
        assert theta == 1.0

    def test_sequential(self):
        """Sequential gives high theta."""
        theta = compute_consistency_theta(ConsistencyLevel.SEQUENTIAL)
        assert theta == 0.75


class TestQuorumTheta:
    """Test quorum theta calculation."""

    def test_strong_quorum(self):
        """R + W > N gives theta > 0.5."""
        theta = compute_quorum_theta(3, 3, 5)  # 3 + 3 > 5
        assert theta > 0.5

    def test_weak_quorum(self):
        """R + W <= N gives theta <= 0.5."""
        theta = compute_quorum_theta(2, 2, 5)  # 2 + 2 < 5
        assert theta <= 0.5

    def test_zero_nodes(self):
        """Zero nodes gives theta = 0."""
        theta = compute_quorum_theta(1, 1, 0)
        assert theta == 0.0


class TestAvailabilityTheta:
    """Test availability theta calculation."""

    def test_perfect_availability(self):
        """Availability = 1 gives theta = 1."""
        theta = compute_availability_theta(1.0)
        assert theta == 1.0

    def test_zero_availability(self):
        """Availability = 0 gives theta = 0."""
        theta = compute_availability_theta(0)
        assert theta == 0.0

    def test_five_nines(self):
        """Five nines availability."""
        theta = compute_availability_theta(0.99999, 0.99999)
        assert theta == pytest.approx(1.0)

    def test_three_nines(self):
        """Three nines availability."""
        theta = compute_availability_theta(0.999, 0.99999)
        assert theta < 1.0
        assert theta > 0.5


class TestNodeAvailability:
    """Test node availability calculation."""

    def test_all_required(self):
        """All nodes required equals single node availability."""
        avail = node_availability(0.99, 3, 3)
        assert avail == pytest.approx(0.99**3)

    def test_any_sufficient(self):
        """Any 1 of n sufficient gives high availability."""
        avail = node_availability(0.99, 3, 1)
        # P(at least 1) = 1 - P(none) = 1 - 0.01^3
        assert avail == pytest.approx(1 - 0.01**3)


class TestLatencyTheta:
    """Test latency theta calculation."""

    def test_zero_latency(self):
        """Zero latency gives theta = 1."""
        theta = compute_latency_theta(0)
        assert theta == 1.0

    def test_target_latency(self):
        """Target latency gives theta = 1."""
        theta = compute_latency_theta(100, 100)
        assert theta == 1.0

    def test_high_latency(self):
        """High latency gives low theta."""
        theta = compute_latency_theta(1000, 100)
        assert theta == 0.1


class TestThroughputTheta:
    """Test throughput theta calculation."""

    def test_target_throughput(self):
        """Target throughput gives theta = 1."""
        theta = compute_throughput_theta(100000, 100000)
        assert theta == 1.0

    def test_zero_throughput(self):
        """Zero throughput gives theta = 0."""
        theta = compute_throughput_theta(0)
        assert theta == 0.0


class TestPartitionTheta:
    """Test partition tolerance theta calculation."""

    def test_cp_mode(self):
        """CP mode gives high theta."""
        theta = compute_partition_theta(PartitionTolerance.CAP_CP, 0.01)
        assert theta > 0.7

    def test_ap_mode(self):
        """AP mode gives moderate theta."""
        theta = compute_partition_theta(PartitionTolerance.CAP_AP, 0.01)
        assert theta > 0.5

    def test_always_partitioned(self):
        """Always partitioned CA gives theta = 0."""
        theta = compute_partition_theta(PartitionTolerance.CAP_CA, 1.0)
        assert theta == 0.0


class TestUnifiedDistributedTheta:
    """Test unified distributed system theta calculation."""

    def test_all_systems_valid_theta(self):
        """All systems should have theta in [0, 1]."""
        for name, system in DISTRIBUTED_SYSTEMS.items():
            theta = compute_distributed_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_spanner_high_consistency(self):
        """Spanner has linearizable consistency."""
        spanner = DISTRIBUTED_SYSTEMS["spanner_cp"]
        assert spanner.consistency == ConsistencyLevel.LINEARIZABLE


class TestClassifyConsistency:
    """Test consistency classification."""

    def test_eventual(self):
        """Low theta -> EVENTUAL."""
        result = classify_consistency(0.2)
        assert result == ConsistencyLevel.EVENTUAL

    def test_linearizable(self):
        """High theta -> LINEARIZABLE."""
        result = classify_consistency(0.95)
        assert result == ConsistencyLevel.LINEARIZABLE


class TestClassifyScalability:
    """Test scalability classification."""

    def test_elastic(self):
        """Elastic flag -> ELASTIC."""
        result = classify_scalability(True, True)
        assert result == ScalabilityClass.ELASTIC

    def test_horizontal(self):
        """Horizontal only -> HORIZONTAL."""
        result = classify_scalability(True, False)
        assert result == ScalabilityClass.HORIZONTAL

    def test_vertical(self):
        """Neither -> VERTICAL."""
        result = classify_scalability(False, False)
        assert result == ScalabilityClass.VERTICAL


class TestEnums:
    """Test enum definitions."""

    def test_consistency_levels(self):
        """All consistency levels defined."""
        assert ConsistencyLevel.EVENTUAL.value == "eventual"
        assert ConsistencyLevel.LINEARIZABLE.value == "linearizable"

    def test_partition_modes(self):
        """All partition modes defined."""
        assert PartitionTolerance.CAP_CP.value == "cp"
        assert PartitionTolerance.CAP_AP.value == "ap"

    def test_replication_modes(self):
        """All replication modes defined."""
        assert ReplicationMode.SYNCHRONOUS.value == "synchronous"
        assert ReplicationMode.ASYNCHRONOUS.value == "asynchronous"


class TestDistributedSystemDataclass:
    """Test DistributedSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with name only."""
        system = DistributedSystem(name="Test")
        assert system.name == "Test"
        assert system.node_count == 3
        assert system.availability == 0.99

    def test_custom_values(self):
        """Can set custom values."""
        system = DistributedSystem(
            name="Custom",
            node_count=10,
            consistency=ConsistencyLevel.SEQUENTIAL
        )
        assert system.node_count == 10
        assert system.consistency == ConsistencyLevel.SEQUENTIAL


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_latency(self):
        """Negative latency handled gracefully."""
        theta = compute_latency_theta(-10)
        assert theta == 1.0

    def test_zero_target_throughput(self):
        """Zero target gives theta = 0."""
        theta = compute_throughput_theta(1000, 0)
        assert theta == 0.0
