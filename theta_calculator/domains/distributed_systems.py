"""
Distributed Systems Domain Module

This module maps theta to distributed computing systems including
consensus, replication, and scalability.

Theta Mapping:
    theta -> 0: Weak consistency, low availability
    theta -> 1: Strong consistency, high availability
    theta = availability: System availability
    theta = 1 - latency/timeout: Response quality
    theta = consensus_nodes / total_nodes: Consensus strength

Key Features:
    - CAP theorem analysis
    - Consensus protocols (Paxos, Raft)
    - Replication and consistency
    - Scalability measures

References:
    @article{Brewer2000,
      author = {Brewer, Eric A.},
      title = {Towards robust distributed systems},
      journal = {PODC Keynote},
      year = {2000}
    }
    @article{Lamport1998,
      author = {Lamport, Leslie},
      title = {The part-time parliament},
      journal = {ACM Trans. Comput. Syst.},
      year = {1998}
    }
    @article{Ongaro2014,
      author = {Ongaro, Diego and Ousterhout, John},
      title = {In search of an understandable consensus algorithm},
      journal = {USENIX ATC},
      year = {2014}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

# Target latency in ms
TARGET_LATENCY_MS = 100.0

# Target availability (five nines)
TARGET_AVAILABILITY = 0.99999


# =============================================================================
# Enums for Classification
# =============================================================================

class ConsistencyLevel(Enum):
    """Classification of consistency level."""
    EVENTUAL = "eventual"            # Eventually consistent
    CAUSAL = "causal"                # Causal consistency
    SEQUENTIAL = "sequential"        # Sequential consistency
    LINEARIZABLE = "linearizable"    # Linearizable (strongest)


class PartitionTolerance(Enum):
    """CAP theorem partition handling."""
    CAP_CP = "cp"                    # Consistency + Partition tolerance
    CAP_AP = "ap"                    # Availability + Partition tolerance
    CAP_CA = "ca"                    # Consistency + Availability (no partitions)


class ScalabilityClass(Enum):
    """Classification of scalability."""
    VERTICAL = "vertical"            # Scale up
    HORIZONTAL = "horizontal"        # Scale out
    ELASTIC = "elastic"              # Dynamic scaling
    LIMITED = "limited"              # Limited scalability


class ReplicationMode(Enum):
    """Data replication mode."""
    SYNCHRONOUS = "synchronous"      # Sync replication
    ASYNCHRONOUS = "asynchronous"    # Async replication
    SEMI_SYNC = "semi_sync"          # Semi-synchronous
    CHAIN = "chain"                  # Chain replication


# =============================================================================
# Dataclass for Distributed Systems
# =============================================================================

@dataclass
class DistributedSystem:
    """
    A distributed computing system.

    Attributes:
        name: Descriptive name
        node_count: Number of nodes
        replication_factor: Data replication factor
        latency_ms: Average latency in milliseconds
        throughput_ops: Operations per second
        availability: System availability [0, 1]
        consistency: Consistency level
        partition_mode: CAP theorem choice
    """
    name: str
    node_count: int = 3
    replication_factor: int = 3
    latency_ms: float = 10.0
    throughput_ops: float = 10000.0
    availability: float = 0.99
    consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    partition_mode: PartitionTolerance = PartitionTolerance.CAP_AP


# =============================================================================
# Consistency Analysis
# =============================================================================

def compute_consistency_theta(
    consistency: ConsistencyLevel
) -> float:
    r"""
    Compute theta from consistency level.

    Stronger consistency = higher theta.

    Args:
        consistency: Consistency level enum

    Returns:
        theta in [0, 1]

    Reference: \cite{Brewer2000}
    """
    consistency_map = {
        ConsistencyLevel.EVENTUAL: 0.25,
        ConsistencyLevel.CAUSAL: 0.5,
        ConsistencyLevel.SEQUENTIAL: 0.75,
        ConsistencyLevel.LINEARIZABLE: 1.0,
    }
    return consistency_map.get(consistency, 0.5)


def compute_quorum_theta(
    read_quorum: int,
    write_quorum: int,
    total_nodes: int
) -> float:
    r"""
    Compute theta from quorum configuration.

    Strong quorum: R + W > N for consistency.

    Args:
        read_quorum: Read quorum size
        write_quorum: Write quorum size
        total_nodes: Total number of nodes

    Returns:
        theta in [0, 1]: 1 = strong quorum

    Reference: \cite{Lamport1998}
    """
    if total_nodes <= 0:
        return 0.0

    # Check if quorum is strong
    if read_quorum + write_quorum > total_nodes:
        overlap = (read_quorum + write_quorum - total_nodes) / total_nodes
        return np.clip(0.5 + 0.5 * overlap, 0.5, 1.0)
    else:
        # Weak quorum
        sum_ratio = (read_quorum + write_quorum) / (2 * total_nodes)
        return np.clip(sum_ratio, 0.0, 0.5)


# =============================================================================
# Availability Analysis
# =============================================================================

def compute_availability_theta(
    availability: float,
    target: float = TARGET_AVAILABILITY
) -> float:
    r"""
    Compute theta from system availability.

    Args:
        availability: System availability [0, 1]
        target: Target availability (e.g., five nines)

    Returns:
        theta in [0, 1]

    Reference: \cite{Brewer2000}
    """
    if availability <= 0:
        return 0.0
    if availability >= 1:
        return 1.0

    # Use log scale for nines
    nines_actual = -np.log10(1 - availability)
    nines_target = -np.log10(1 - target)

    if nines_target <= 0:
        return 1.0

    theta = nines_actual / nines_target
    return np.clip(theta, 0.0, 1.0)


def node_availability(
    single_node_availability: float,
    n_nodes: int,
    k_required: int
) -> float:
    r"""
    Compute system availability from node availability.

    System available if >= k out of n nodes available.

    Args:
        single_node_availability: Individual node availability
        n_nodes: Total number of nodes
        k_required: Minimum nodes required

    Returns:
        System availability [0, 1]
    """
    if k_required > n_nodes:
        return 0.0
    if k_required <= 0:
        return 1.0

    from math import comb
    p = single_node_availability
    system_avail = 0.0
    for i in range(k_required, n_nodes + 1):
        system_avail += comb(n_nodes, i) * (p**i) * ((1-p)**(n_nodes-i))

    return system_avail


# =============================================================================
# Latency Analysis
# =============================================================================

def compute_latency_theta(
    latency_ms: float,
    target_ms: float = TARGET_LATENCY_MS
) -> float:
    r"""
    Compute theta from latency.

    Lower latency = higher theta.

    Args:
        latency_ms: Actual latency in milliseconds
        target_ms: Target latency

    Returns:
        theta in [0, 1]
    """
    if latency_ms <= 0:
        return 1.0
    if target_ms <= 0:
        return 0.0

    theta = target_ms / latency_ms
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Throughput Analysis
# =============================================================================

def compute_throughput_theta(
    throughput: float,
    target: float = 100000.0
) -> float:
    r"""
    Compute theta from throughput.

    Higher throughput = higher theta.

    Args:
        throughput: Operations per second
        target: Target throughput

    Returns:
        theta in [0, 1]
    """
    if throughput <= 0:
        return 0.0
    if target <= 0:
        return 0.0

    theta = throughput / target
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Partition Tolerance
# =============================================================================

def compute_partition_theta(
    mode: PartitionTolerance,
    partition_probability: float = 0.01
) -> float:
    r"""
    Compute theta based on CAP choice and partition handling.

    Args:
        mode: CAP theorem choice
        partition_probability: Probability of network partition

    Returns:
        theta in [0, 1]

    Reference: \cite{Brewer2000}
    """
    if partition_probability >= 1:
        # Always partitioned
        if mode == PartitionTolerance.CAP_CA:
            return 0.0  # CA fails with partitions
        else:
            return 0.5

    # CP prioritizes consistency during partition
    # AP prioritizes availability during partition
    # CA assumes no partitions
    mode_weight = {
        PartitionTolerance.CAP_CP: 0.8,
        PartitionTolerance.CAP_AP: 0.7,
        PartitionTolerance.CAP_CA: 0.5,
    }
    base = mode_weight.get(mode, 0.5)

    # Adjust for partition probability
    theta = base * (1 - partition_probability)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Unified Distributed System Theta
# =============================================================================

def compute_distributed_theta(system: DistributedSystem) -> float:
    r"""
    Compute unified theta for distributed system.

    Args:
        system: DistributedSystem dataclass

    Returns:
        theta in [0, 1]
    """
    thetas = []

    # Consistency contribution
    theta_cons = compute_consistency_theta(system.consistency)
    thetas.append(theta_cons)

    # Availability contribution
    theta_avail = compute_availability_theta(system.availability)
    thetas.append(theta_avail)

    # Latency contribution
    theta_lat = compute_latency_theta(system.latency_ms)
    thetas.append(theta_lat)

    # Partition tolerance contribution
    theta_part = compute_partition_theta(system.partition_mode)
    thetas.append(theta_part)

    if not thetas:
        return 0.5

    return np.prod(thetas)**(1/len(thetas))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_consistency(theta: float) -> ConsistencyLevel:
    """
    Classify consistency level from theta.

    Args:
        theta: Consistency theta [0, 1]

    Returns:
        ConsistencyLevel enum
    """
    if theta < 0.3:
        return ConsistencyLevel.EVENTUAL
    elif theta < 0.6:
        return ConsistencyLevel.CAUSAL
    elif theta < 0.85:
        return ConsistencyLevel.SEQUENTIAL
    else:
        return ConsistencyLevel.LINEARIZABLE


def classify_scalability(
    can_horizontal: bool,
    is_elastic: bool
) -> ScalabilityClass:
    """
    Classify scalability type.

    Args:
        can_horizontal: Whether system can scale horizontally
        is_elastic: Whether scaling is automatic/elastic

    Returns:
        ScalabilityClass enum
    """
    if is_elastic:
        return ScalabilityClass.ELASTIC
    elif can_horizontal:
        return ScalabilityClass.HORIZONTAL
    else:
        return ScalabilityClass.VERTICAL


# =============================================================================
# Example Systems Dictionary
# =============================================================================

DISTRIBUTED_SYSTEMS: Dict[str, DistributedSystem] = {
    "paxos_raft": DistributedSystem(
        name="Paxos/Raft Consensus",
        node_count=5,
        replication_factor=5,
        latency_ms=5.0,
        availability=0.9999,
        consistency=ConsistencyLevel.LINEARIZABLE,
        partition_mode=PartitionTolerance.CAP_CP
    ),
    "cassandra_ap": DistributedSystem(
        name="Cassandra (AP mode)",
        node_count=12,
        replication_factor=3,
        latency_ms=2.0,
        throughput_ops=100000.0,
        availability=0.99999,
        consistency=ConsistencyLevel.EVENTUAL,
        partition_mode=PartitionTolerance.CAP_AP
    ),
    "spanner_cp": DistributedSystem(
        name="Google Spanner",
        node_count=1000,
        replication_factor=5,
        latency_ms=10.0,
        availability=0.99999,
        consistency=ConsistencyLevel.LINEARIZABLE,
        partition_mode=PartitionTolerance.CAP_CP
    ),
    "redis_cluster": DistributedSystem(
        name="Redis Cluster",
        node_count=6,
        replication_factor=2,
        latency_ms=0.5,
        throughput_ops=500000.0,
        availability=0.999,
        consistency=ConsistencyLevel.EVENTUAL,
        partition_mode=PartitionTolerance.CAP_AP
    ),
    "zookeeper": DistributedSystem(
        name="Apache ZooKeeper",
        node_count=5,
        replication_factor=5,
        latency_ms=2.0,
        availability=0.9999,
        consistency=ConsistencyLevel.SEQUENTIAL,
        partition_mode=PartitionTolerance.CAP_CP
    ),
    "dynamodb": DistributedSystem(
        name="Amazon DynamoDB",
        node_count=100,
        replication_factor=3,
        latency_ms=5.0,
        throughput_ops=50000.0,
        availability=0.99999,
        consistency=ConsistencyLevel.EVENTUAL,
        partition_mode=PartitionTolerance.CAP_AP
    ),
    "cockroachdb": DistributedSystem(
        name="CockroachDB",
        node_count=9,
        replication_factor=3,
        latency_ms=10.0,
        availability=0.9999,
        consistency=ConsistencyLevel.LINEARIZABLE,
        partition_mode=PartitionTolerance.CAP_CP
    ),
    "kafka": DistributedSystem(
        name="Apache Kafka",
        node_count=6,
        replication_factor=3,
        latency_ms=1.0,
        throughput_ops=1000000.0,
        availability=0.9999,
        consistency=ConsistencyLevel.CAUSAL,
        partition_mode=PartitionTolerance.CAP_AP
    ),
    "etcd": DistributedSystem(
        name="etcd",
        node_count=5,
        replication_factor=5,
        latency_ms=1.0,
        availability=0.9999,
        consistency=ConsistencyLevel.LINEARIZABLE,
        partition_mode=PartitionTolerance.CAP_CP
    ),
    "mongodb": DistributedSystem(
        name="MongoDB ReplicaSet",
        node_count=3,
        replication_factor=3,
        latency_ms=3.0,
        availability=0.999,
        consistency=ConsistencyLevel.CAUSAL,
        partition_mode=PartitionTolerance.CAP_CP
    ),
}


# Precomputed theta values
DISTRIBUTED_THETA_VALUES: Dict[str, float] = {
    name: compute_distributed_theta(system)
    for name, system in DISTRIBUTED_SYSTEMS.items()
}
