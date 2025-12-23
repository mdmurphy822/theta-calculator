"""
Cosmology Proofs Module

This module contains theta proofs based on cosmological physics,
particularly vacuum energy and dark energy.

Proofs:
    - Vacuum energy: Cosmological constant problem
    - Dark energy equation of state

References (see BIBLIOGRAPHY.bib):
    \cite{Weinberg1989} - Cosmological constant problem
    \cite{Carroll2001} - Cosmological constant
    \cite{PlanckCollaboration2020} - Planck 2018 results
"""

from .vacuum_energy import (
    VacuumEnergyProofs,
    VacuumEnergyResult,
    DarkEnergyEOSResult,
    compute_vacuum_theta,
    compute_dark_energy_theta,
)

__all__ = [
    "VacuumEnergyProofs",
    "VacuumEnergyResult",
    "DarkEnergyEOSResult",
    "compute_vacuum_theta",
    "compute_dark_energy_theta",
]
