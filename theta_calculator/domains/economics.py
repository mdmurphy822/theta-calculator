"""
Economics Domain: Ising Model of Financial Markets

This module implements theta as the quantum-classical interpolation parameter
for financial markets using the Ising model framework.

Key Insight: Markets exhibit phase transitions between:
- theta ~ 0: Efficient market (random walk, no correlations)
- theta ~ 1: Market crash/bubble (collective behavior, strong correlations)

The Ising model maps traders to spins:
- Spin up (+1): Buy decision
- Spin down (-1): Sell decision
- Coupling J_ij: Correlation between traders i and j
- Temperature T: Market volatility / uncertainty

References:
- Bornholdt (2001): Ising model of financial markets
- Krawiecki et al. (2002): Volatility clustering
- Phase Transitions in Financial Markets (ArXiv)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification based on theta."""
    EFFICIENT = "efficient"          # theta < 0.2: Random walk
    NORMAL = "normal"                # 0.2 <= theta < 0.5: Some correlations
    TRENDING = "trending"            # 0.5 <= theta < 0.8: Strong trends
    BUBBLE = "bubble"                # theta >= 0.8: Collective mania
    CRASH = "crash"                  # theta ~ 1.0 + negative sentiment


@dataclass
class MarketSystem:
    """
    A financial market system for theta analysis.

    Attributes:
        name: Market identifier
        n_traders: Number of market participants
        coupling_strength: Average interaction strength J
        temperature: Market temperature (volatility measure)
        order_parameter: Net market sentiment m in [-1, 1]
        volatility: Historical volatility
        correlation: Average trader correlation
    """
    name: str
    n_traders: int
    coupling_strength: float  # J in Ising model
    temperature: float        # T (volatility proxy)
    order_parameter: float    # m = <s_i> (net sentiment)
    volatility: Optional[float] = None
    correlation: Optional[float] = None

    @property
    def critical_temperature(self) -> float:
        """
        Critical temperature for phase transition.

        In mean-field Ising: T_c = J * z, where z is coordination number.
        For fully connected graph: T_c ~ J * N.
        """
        # Mean-field approximation
        return self.coupling_strength * np.sqrt(self.n_traders)

    @property
    def reduced_temperature(self) -> float:
        """t = (T - T_c) / T_c: Distance from critical point."""
        T_c = self.critical_temperature
        if T_c == 0:
            return float('inf')
        return (self.temperature - T_c) / T_c


@dataclass
class IsingMarket:
    """
    Full Ising model implementation for market dynamics.

    Hamiltonian: H = -∑_{i<j} J_ij s_i s_j - h ∑_i s_i

    Where:
    - s_i = ±1: Trader i's position (buy/sell)
    - J_ij: Coupling between traders (correlation)
    - h: External field (market trend/news)
    """
    n_spins: int
    coupling_matrix: np.ndarray  # J_ij
    external_field: float = 0.0  # h

    def energy(self, spins: np.ndarray) -> float:
        """Compute Hamiltonian for given spin configuration."""
        interaction = -0.5 * np.sum(self.coupling_matrix * np.outer(spins, spins))
        field = -self.external_field * np.sum(spins)
        return interaction + field

    def magnetization(self, spins: np.ndarray) -> float:
        """Order parameter m = (1/N) ∑_i s_i."""
        return np.mean(spins)

    def susceptibility(self, spins_history: List[np.ndarray]) -> float:
        """
        Magnetic susceptibility χ = N * (<m²> - <m>²) / T.

        High susceptibility near critical point indicates
        sensitivity to perturbations (market instability).
        """
        magnetizations = [self.magnetization(s) for s in spins_history]
        m_mean = np.mean(magnetizations)
        m2_mean = np.mean([m**2 for m in magnetizations])
        return self.n_spins * (m2_mean - m_mean**2)


# =============================================================================
# CRITICAL EXPONENTS (Universality Class)
# =============================================================================

# Mean-field (infinite range) critical exponents
MEAN_FIELD_EXPONENTS = {
    "beta": 0.5,    # m ~ |T - T_c|^beta (order parameter)
    "gamma": 1.0,   # chi ~ |T - T_c|^(-gamma) (susceptibility)
    "nu": 0.5,      # xi ~ |T - T_c|^(-nu) (correlation length)
    "delta": 3.0,   # m ~ h^(1/delta) at T = T_c
    "alpha": 0.0,   # C ~ |T - T_c|^(-alpha) (specific heat)
}

# 3D Ising critical exponents (finite-dimensional markets)
ISING_3D_EXPONENTS = {
    "beta": 0.326,
    "gamma": 1.237,
    "nu": 0.630,
    "delta": 4.789,
    "alpha": 0.110,
}


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_market_theta(market: MarketSystem) -> float:
    """
    Compute theta for a market system.

    Theta measures how far the market is from efficient (classical) behavior.

    Methods:
    1. Order parameter: |m| near 1 means collective behavior (high theta)
    2. Critical proximity: |t| near 0 means near phase transition (high theta)
    3. Susceptibility divergence: High chi means instability (high theta)

    Returns:
        theta in [0, 1] where:
        - 0 = efficient market (random walk)
        - 1 = collective behavior (crash/bubble)
    """
    # Method 1: Order parameter contribution
    # |m| = 0 means random (theta low), |m| = 1 means aligned (theta high)
    theta_order = abs(market.order_parameter)

    # Method 2: Critical proximity
    # Near T_c, correlations are long-range (quantum-like)
    t = market.reduced_temperature
    theta_critical = np.exp(-abs(t)) if t != float('inf') else 0.0

    # Method 3: Coupling strength
    # Strong coupling = correlated behavior
    J_normalized = min(market.coupling_strength / 1.0, 1.0)  # Normalize to ~1

    # Unified theta (weighted average)
    theta = 0.4 * theta_order + 0.4 * theta_critical + 0.2 * J_normalized

    return np.clip(theta, 0.0, 1.0)


def compute_coupling_from_correlation(correlation_matrix: np.ndarray) -> float:
    """
    Infer coupling strength J from observed correlation matrix.

    Using TAP (Thouless-Anderson-Palmer) approximation:
    J_ij ≈ -C_ij^(-1) for off-diagonal elements

    Args:
        correlation_matrix: Pairwise correlation between traders/assets

    Returns:
        Average coupling strength J
    """
    n = correlation_matrix.shape[0]
    # Regularize for inversion
    C_reg = correlation_matrix + 0.01 * np.eye(n)
    try:
        C_inv = np.linalg.inv(C_reg)
        # Off-diagonal elements give J
        J_matrix = -C_inv
        np.fill_diagonal(J_matrix, 0)
        return np.mean(np.abs(J_matrix))
    except np.linalg.LinAlgError:
        return 0.0


def detect_phase_transition(
    order_parameters: List[float],
    temperatures: List[float]
) -> Tuple[bool, Optional[float]]:
    """
    Detect phase transition from order parameter vs temperature data.

    Look for:
    1. Sudden change in order parameter
    2. Susceptibility peak
    3. Critical slowing down

    Returns:
        (detected, T_c) where T_c is estimated critical temperature
    """
    if len(order_parameters) < 5:
        return False, None

    # Compute derivative dm/dT
    dm_dT = np.gradient(order_parameters, temperatures)

    # Find maximum derivative (phase transition point)
    max_idx = np.argmax(np.abs(dm_dT))
    max_derivative = abs(dm_dT[max_idx])

    # Threshold for detection
    if max_derivative > 0.5:  # Significant change
        return True, temperatures[max_idx]

    return False, None


def classify_regime(theta: float, order_parameter: float) -> MarketRegime:
    """Classify market regime from theta and sentiment."""
    if theta < 0.2:
        return MarketRegime.EFFICIENT
    elif theta < 0.5:
        return MarketRegime.NORMAL
    elif theta < 0.8:
        return MarketRegime.TRENDING
    else:
        if order_parameter < -0.5:
            return MarketRegime.CRASH
        else:
            return MarketRegime.BUBBLE


# =============================================================================
# EXAMPLE MARKET SYSTEMS
# =============================================================================

ECONOMIC_SYSTEMS: Dict[str, MarketSystem] = {
    "efficient_market": MarketSystem(
        name="Efficient Market (EMH)",
        n_traders=10000,
        coupling_strength=0.01,
        temperature=1.0,
        order_parameter=0.0,
        volatility=0.15,
        correlation=0.0,
    ),
    "normal_trading": MarketSystem(
        name="Normal Trading Day",
        n_traders=10000,
        coupling_strength=0.1,
        temperature=0.8,
        order_parameter=0.1,
        volatility=0.20,
        correlation=0.2,
    ),
    "trending_market": MarketSystem(
        name="Trending Market",
        n_traders=10000,
        coupling_strength=0.3,
        temperature=0.5,
        order_parameter=0.5,
        volatility=0.25,
        correlation=0.5,
    ),
    "bubble_forming": MarketSystem(
        name="Bubble Formation",
        n_traders=10000,
        coupling_strength=0.8,
        temperature=0.3,
        order_parameter=0.8,
        volatility=0.35,
        correlation=0.7,
    ),
    "market_crash": MarketSystem(
        name="Market Crash",
        n_traders=10000,
        coupling_strength=1.0,
        temperature=0.1,
        order_parameter=-0.9,
        volatility=0.80,
        correlation=0.9,
    ),
    "flash_crash": MarketSystem(
        name="Flash Crash (2010)",
        n_traders=50000,
        coupling_strength=1.5,
        temperature=0.05,
        order_parameter=-0.95,
        volatility=2.0,
        correlation=0.95,
    ),
    "dotcom_bubble": MarketSystem(
        name="Dot-com Bubble (1999)",
        n_traders=100000,
        coupling_strength=1.2,
        temperature=0.2,
        order_parameter=0.85,
        volatility=0.40,
        correlation=0.8,
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def market_theta_summary():
    """Print theta analysis for all example markets."""
    print("=" * 70)
    print("MARKET THETA ANALYSIS (Ising Model Framework)")
    print("=" * 70)
    print()
    print(f"{'Market':<25} {'θ':>8} {'|m|':>8} {'T/T_c':>8} {'Regime':<15}")
    print("-" * 70)

    for name, market in ECONOMIC_SYSTEMS.items():
        theta = compute_market_theta(market)
        regime = classify_regime(theta, market.order_parameter)
        t_ratio = market.temperature / market.critical_temperature if market.critical_temperature > 0 else float('inf')

        print(f"{market.name:<25} {theta:>8.3f} {abs(market.order_parameter):>8.2f} "
              f"{t_ratio:>8.3f} {regime.value:<15}")


if __name__ == "__main__":
    market_theta_summary()
