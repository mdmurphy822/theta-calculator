"""
Quantum Proofs Module

This module contains theta proofs based on fundamental quantum mechanics.

Proofs:
    - Uncertainty proofs: Heisenberg, Energy-Time, Entropic uncertainty

References (see BIBLIOGRAPHY.bib):
    \cite{Heisenberg1927} - Original uncertainty principle
    \cite{Ozawa2003} - Universally valid reformulation
    \cite{Deutsch1983} - Entropic uncertainty relations
"""

from .uncertainty_proofs import (
    UncertaintyProofs,
    HeisenbergResult,
    EnergyTimeResult,
    EntropicResult,
    compute_heisenberg_theta,
    compute_energy_time_theta,
    compute_entropic_theta,
)

__all__ = [
    "UncertaintyProofs",
    "HeisenbergResult",
    "EnergyTimeResult",
    "EntropicResult",
    "compute_heisenberg_theta",
    "compute_energy_time_theta",
    "compute_entropic_theta",
]
