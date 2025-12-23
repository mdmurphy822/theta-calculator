"""
Networks Domain: Shannon Capacity, Percolation, and Quantum Key Distribution

This module implements theta as the communication/network efficiency parameter.

Key Insight: Networks exhibit phase transitions and fundamental limits:
- theta ~ 0: Disconnected/noisy (below percolation threshold)
- theta ~ 1: Fully connected/optimal (at Shannon limit)

Theta Maps To:
1. Shannon capacity: C_actual / C_max
2. Network percolation: p / p_c (connectivity threshold)
3. QKD security: key_rate_quantum / key_rate_classical

References (see BIBLIOGRAPHY.bib):
    \cite{Shannon1948} - A Mathematical Theory of Communication
    \cite{Stauffer1994} - Introduction to Percolation Theory
    \cite{GisinRibordy2002} - Quantum cryptography
    \cite{BarabasiAlbert1999} - Emergence of scaling in random networks
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class NetworkType(Enum):
    """Types of networks for theta analysis."""
    COMMUNICATION = "communication"  # Data transmission
    SOCIAL = "social"                # Social connections
    BIOLOGICAL = "biological"        # Neural, metabolic
    INFRASTRUCTURE = "infrastructure"  # Power, transport
    QUANTUM = "quantum"              # Quantum networks


class ConnectivityRegime(Enum):
    """Network connectivity regimes."""
    DISCONNECTED = "disconnected"  # theta < p_c
    CRITICAL = "critical"          # theta ~ p_c
    CONNECTED = "connected"        # theta > p_c
    FULLY_CONNECTED = "fully_connected"  # theta ~ 1


@dataclass
class NetworkSystem:
    """
    A network system for theta analysis.

    Attributes:
        name: Network identifier
        network_type: Type of network
        n_nodes: Number of nodes
        n_edges: Number of edges
        connection_probability: Edge probability p
        bandwidth: Channel bandwidth [Hz]
        snr: Signal-to-noise ratio
        is_quantum: Whether quantum protocols are used
    """
    name: str
    network_type: NetworkType
    n_nodes: int
    n_edges: int
    connection_probability: float
    bandwidth: float = 0.0
    snr: float = 1.0
    is_quantum: bool = False


# =============================================================================
# SHANNON CAPACITY
# =============================================================================

def shannon_capacity(bandwidth: float, snr: float) -> float:
    """
    Compute Shannon channel capacity.

    C = B * log₂(1 + SNR)

    This is the MAXIMUM data rate for reliable transmission
    over a noisy channel.

    Args:
        bandwidth: Channel bandwidth [Hz]
        snr: Signal-to-noise ratio (linear, not dB)

    Returns:
        Channel capacity [bits/s]

    Reference: \cite{Shannon1948}
    """
    if bandwidth <= 0 or snr < 0:
        return 0.0
    return bandwidth * np.log2(1 + snr)


def snr_from_db(snr_db: float) -> float:
    """Convert SNR from dB to linear scale."""
    return 10 ** (snr_db / 10)


def compute_shannon_theta(
    actual_rate: float,
    bandwidth: float,
    snr: float
) -> float:
    """
    Compute theta for communication channel.

    Theta = R_actual / C_Shannon

    Args:
        actual_rate: Achieved data rate [bits/s]
        bandwidth: Channel bandwidth [Hz]
        snr: Signal-to-noise ratio (linear)

    Returns:
        theta in [0, 1]

    Reference: \cite{Shannon1948}
    """
    capacity = shannon_capacity(bandwidth, snr)
    if capacity <= 0:
        return 0.0

    theta = actual_rate / capacity
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# NETWORK PERCOLATION
# =============================================================================

def percolation_threshold(network_type: str = "2d_square") -> float:
    """
    Get percolation threshold for different network types.

    The percolation threshold p_c is where a giant component emerges.

    Args:
        network_type: Type of network lattice

    Returns:
        Critical probability p_c

    Reference: \cite{Stauffer1994}
    """
    thresholds = {
        "2d_square": 0.5927,      # Site percolation
        "2d_triangular": 0.5,     # Exact
        "2d_honeycomb": 0.6962,
        "3d_cubic": 0.3116,
        "bethe_z3": 0.5,          # 1/(z-1) for Bethe lattice
        "bethe_z4": 0.333,
        "erdos_renyi": 1.0,       # p_c = 1/N for ER random graph
        "scale_free": 0.0,        # No threshold for γ < 3
    }
    return thresholds.get(network_type, 0.5)


def giant_component_fraction(p: float, p_c: float = 0.5) -> float:
    """
    Compute fraction of network in giant component.

    Above p_c:
    P∞ ~ (p - p_c)^β with β ≈ 0.4 (2D) or 0.41 (3D)

    Args:
        p: Connection probability
        p_c: Percolation threshold

    Returns:
        Fraction in giant component [0, 1]

    Reference: \cite{Stauffer1994}
    """
    if p <= p_c:
        return 0.0

    beta = 0.4  # 2D percolation exponent
    return ((p - p_c) / (1 - p_c)) ** beta


def compute_percolation_theta(
    p: float,
    p_c: float = 0.5
) -> float:
    """
    Compute theta for network percolation.

    Theta = p / p_c for p < p_c (subcritical)
    Theta = giant_component_fraction for p > p_c

    Args:
        p: Connection probability
        p_c: Percolation threshold

    Returns:
        theta in [0, 1]

    Reference: \cite{Stauffer1994}
    """
    if p <= 0:
        return 0.0

    if p < p_c:
        # Below threshold: theta = p/p_c
        theta = p / p_c
    else:
        # Above threshold: use giant component
        theta = p_c + (1 - p_c) * giant_component_fraction(p, p_c)

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# QUANTUM KEY DISTRIBUTION
# =============================================================================

@dataclass
class QKDResult:
    """
    Result of QKD analysis.

    Attributes:
        raw_key_rate: Raw key generation rate [bits/s]
        secure_key_rate: Secure key rate after privacy amplification
        qber: Quantum bit error rate
        security_parameter: Security level
        theta: Quantum advantage measure

    Reference: \cite{GisinRibordy2002}
    """
    raw_key_rate: float
    secure_key_rate: float
    qber: float
    security_parameter: float
    theta: float


def bb84_key_rate(
    pulse_rate: float,
    detection_efficiency: float,
    channel_loss: float,
    dark_count_rate: float,
    qber: float
) -> Tuple[float, float]:
    """
    Compute BB84 QKD key rate.

    Secure key rate:
    R = (1/2) * μ * η * (1 - 2*h(QBER))

    Where h is binary entropy.

    Args:
        pulse_rate: Laser pulse rate [Hz]
        detection_efficiency: Detector efficiency [0, 1]
        channel_loss: Channel transmission [0, 1]
        dark_count_rate: Dark counts per gate [0, 1]
        qber: Quantum bit error rate [0, 0.5]

    Returns:
        (raw_rate, secure_rate) in bits/s

    Reference: \cite{GisinRibordy2002}
    """
    # Mean photon number (weak coherent pulse)
    mu = 0.1

    # Transmission
    eta = detection_efficiency * channel_loss

    # Raw key rate
    raw_rate = 0.5 * pulse_rate * mu * eta

    # Binary entropy
    if qber <= 0:
        h_qber = 0.0
    elif qber >= 0.5:
        h_qber = 1.0
    else:
        h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)

    # Secure key rate (after error correction and privacy amplification)
    # For BB84, security requires QBER < 11%
    if qber > 0.11:
        secure_rate = 0.0
    else:
        secure_rate = raw_rate * (1 - 2 * h_qber)

    return raw_rate, max(0, secure_rate)


def compute_qkd_theta(
    secure_rate: float,
    classical_rate: float,
    max_qkd_advantage: float = 10.0
) -> float:
    """
    Compute theta for QKD vs classical cryptography.

    QKD provides information-theoretic security, classical doesn't.
    Theta measures the quantum advantage.

    Args:
        secure_rate: QKD secure key rate [bits/s]
        classical_rate: Comparable classical key rate [bits/s]
        max_qkd_advantage: Maximum expected advantage factor

    Returns:
        theta in [0, 1]

    Reference: \cite{GisinRibordy2002}
    """
    if classical_rate <= 0:
        return 1.0 if secure_rate > 0 else 0.0

    # QKD advantage: secure but slower
    # theta = 1 means QKD is viable (rate > threshold)
    # Consider security value in the ratio
    security_factor = 2.0  # Security worth 2x rate

    effective_rate = secure_rate * security_factor
    theta = effective_rate / (classical_rate + effective_rate)

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# SCALE-FREE NETWORKS
# =============================================================================

def barabasi_albert_degree_dist(k: int, m: int = 2) -> float:
    """
    Degree distribution for Barabasi-Albert scale-free network.

    P(k) ~ k^(-γ) with γ = 3

    Args:
        k: Degree
        m: Number of edges per new node

    Returns:
        Probability P(k)

    Reference: \cite{BarabasiAlbert1999}
    """
    if k < m:
        return 0.0
    return 2 * m ** 2 / k ** 3


def compute_scalefree_theta(
    gamma: float,
    gamma_min: float = 2.0,
    gamma_max: float = 4.0
) -> float:
    """
    Compute theta for scale-free network.

    Theta based on power-law exponent γ:
    - γ < 2: Ultra-small world (hub-dominated)
    - γ = 2.5-3: Typical (internet, WWW)
    - γ > 4: Random-like

    Args:
        gamma: Power-law exponent
        gamma_min: Minimum physical γ
        gamma_max: Maximum meaningful γ

    Returns:
        theta in [0, 1]

    Reference: \cite{BarabasiAlbert1999}
    """
    # Normalize gamma to [0, 1]
    # γ = 3 (BA model) maps to theta ~ 0.5
    theta = 1 - (gamma - gamma_min) / (gamma_max - gamma_min)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED CALCULATIONS
# =============================================================================

def compute_network_theta(system: NetworkSystem) -> float:
    """
    Compute unified theta for network system.

    Args:
        system: NetworkSystem to analyze

    Returns:
        theta in [0, 1]
    """
    if system.network_type == NetworkType.COMMUNICATION:
        # Use Shannon capacity
        if system.bandwidth > 0 and system.snr > 0:
            capacity = shannon_capacity(system.bandwidth, system.snr)
            # Assume 80% utilization typical
            return 0.8
        else:
            return system.connection_probability

    elif system.network_type == NetworkType.QUANTUM:
        # Use QKD advantage
        return 0.7 if system.is_quantum else 0.0

    else:
        # Use percolation threshold
        p_c = percolation_threshold("2d_square")
        return compute_percolation_theta(system.connection_probability, p_c)


def classify_connectivity(theta: float, p_c: float = 0.5) -> ConnectivityRegime:
    """Classify network connectivity from theta."""
    if theta < p_c - 0.1:
        return ConnectivityRegime.DISCONNECTED
    elif theta < p_c + 0.1:
        return ConnectivityRegime.CRITICAL
    elif theta < 0.95:
        return ConnectivityRegime.CONNECTED
    else:
        return ConnectivityRegime.FULLY_CONNECTED


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

NETWORK_SYSTEMS: Dict[str, NetworkSystem] = {
    "wifi_home": NetworkSystem(
        name="Home WiFi (802.11ac)",
        network_type=NetworkType.COMMUNICATION,
        n_nodes=10,
        n_edges=9,
        connection_probability=1.0,
        bandwidth=80e6,  # 80 MHz
        snr=snr_from_db(30),  # 30 dB typical
    ),
    "5g_cellular": NetworkSystem(
        name="5G mmWave Cell",
        network_type=NetworkType.COMMUNICATION,
        n_nodes=100,
        n_edges=99,
        connection_probability=0.9,
        bandwidth=400e6,  # 400 MHz
        snr=snr_from_db(20),
    ),
    "fiber_backbone": NetworkSystem(
        name="Fiber Backbone (100G)",
        network_type=NetworkType.COMMUNICATION,
        n_nodes=1000,
        n_edges=5000,
        connection_probability=0.95,
        bandwidth=100e9,  # 100 GHz optical
        snr=snr_from_db(25),
    ),
    "qkd_link": NetworkSystem(
        name="QKD Fiber Link",
        network_type=NetworkType.QUANTUM,
        n_nodes=2,
        n_edges=1,
        connection_probability=1.0,
        bandwidth=1e9,
        snr=100,
        is_quantum=True,
    ),
    "social_sparse": NetworkSystem(
        name="Sparse Social Network",
        network_type=NetworkType.SOCIAL,
        n_nodes=1000,
        n_edges=2000,  # ~4 friends avg
        connection_probability=0.004,
    ),
    "social_dense": NetworkSystem(
        name="Dense Community",
        network_type=NetworkType.SOCIAL,
        n_nodes=100,
        n_edges=1500,  # ~30 connections
        connection_probability=0.30,
    ),
    "power_grid": NetworkSystem(
        name="Power Grid (Regional)",
        network_type=NetworkType.INFRASTRUCTURE,
        n_nodes=500,
        n_edges=600,  # Sparse
        connection_probability=0.005,
    ),
    "internet_as": NetworkSystem(
        name="Internet AS-level",
        network_type=NetworkType.INFRASTRUCTURE,
        n_nodes=50000,
        n_edges=200000,
        connection_probability=0.0002,  # Scale-free
    ),
}


def network_theta_summary():
    """Print theta analysis for example networks."""
    print("=" * 70)
    print("NETWORK THETA ANALYSIS (Connectivity & Capacity)")
    print("=" * 70)
    print()

    # Communication networks
    print("COMMUNICATION NETWORKS:")
    print(f"{'Network':<25} {'BW [Hz]':>12} {'SNR [dB]':>10} {'C [bps]':>12}")
    print("-" * 60)

    for name, sys in NETWORK_SYSTEMS.items():
        if sys.network_type == NetworkType.COMMUNICATION:
            snr_db = 10 * np.log10(sys.snr)
            capacity = shannon_capacity(sys.bandwidth, sys.snr)
            print(f"{sys.name:<25} {sys.bandwidth:>12.2e} {snr_db:>10.1f} {capacity:>12.2e}")

    print()

    # Percolation analysis
    print("NETWORK CONNECTIVITY:")
    print(f"{'Network':<25} {'Nodes':>8} {'Edges':>8} {'p':>8} {'θ':>8}")
    print("-" * 60)

    for name, sys in NETWORK_SYSTEMS.items():
        theta = compute_network_theta(sys)
        print(f"{sys.name:<25} {sys.n_nodes:>8} {sys.n_edges:>8} "
              f"{sys.connection_probability:>8.4f} {theta:>8.3f}")


if __name__ == "__main__":
    network_theta_summary()
