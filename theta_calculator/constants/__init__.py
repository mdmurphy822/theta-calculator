"""
Constants module: Complete CODATA 2022 fundamental physical constants.

This module provides:
- FundamentalConstants: Core 12 constants from values.py (legacy)
- ALL_CONSTANTS: Complete 100+ CODATA 2022 constants
- ALL_CONNECTIONS: Mathematical relationships between constants
- PlanckUnits: Planck scale natural units

The constants form a self-consistent network where each can be derived
from others, proving theta emerges naturally from physics structure.
"""

from .values import FundamentalConstants, PhysicalConstant
from .planck_units import PlanckUnits
from .codata_2022 import (
    Constant,
    ConstantCategory,
    ALL_CONSTANTS,
    get_constants_by_category,
    get_exact_constants,
    get_derived_constants,
    get_constant_network,
    print_constant_summary,
    # Key individual constants
    c, h, h_bar, e_charge, k_B, N_A, G,
    alpha, mu_0, epsilon_0,
    m_e, m_p, m_n, m_mu, m_tau,
    a_0, R_inf, E_h, r_e, lambda_C,
    l_P, t_P, m_P, E_P, T_P,
    mu_B, mu_N, mu_e, mu_p, mu_n,
)
from .connections import (
    ConnectionType,
    ConstantConnection,
    ALL_CONNECTIONS,
    verify_all_connections,
    get_connection_network,
    get_theta_relevant_connections,
    print_connection_summary,
    generate_theta_boundary_report,
    compute_hawking_temperature,
)

__all__ = [
    # Legacy
    "FundamentalConstants",
    "PhysicalConstant",
    "PlanckUnits",
    # New CODATA 2022
    "Constant",
    "ConstantCategory",
    "ALL_CONSTANTS",
    "get_constants_by_category",
    "get_exact_constants",
    "get_derived_constants",
    "get_constant_network",
    "print_constant_summary",
    # Connections
    "ConnectionType",
    "ConstantConnection",
    "ALL_CONNECTIONS",
    "verify_all_connections",
    "get_connection_network",
    "get_theta_relevant_connections",
    "print_connection_summary",
    "generate_theta_boundary_report",
    "compute_hawking_temperature",
    # Key constants
    "c", "h", "h_bar", "e_charge", "k_B", "N_A", "G",
    "alpha", "mu_0", "epsilon_0",
    "m_e", "m_p", "m_n", "m_mu", "m_tau",
    "a_0", "R_inf", "E_h", "r_e", "lambda_C",
    "l_P", "t_P", "m_P", "E_P", "T_P",
    "mu_B", "mu_N", "mu_e", "mu_p", "mu_n",
]
