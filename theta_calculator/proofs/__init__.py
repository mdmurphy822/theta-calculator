r"""
Proofs module: Mathematical, numerical, and information-theoretic proofs.

This module contains theta derivations from fundamental physics principles.

Submodules:
    - quantum: Uncertainty principle proofs
    - gravity: Hawking radiation and holographic entropy proofs
    - cosmology: Vacuum energy and dark energy proofs
    - information: Bekenstein bound and Landauer limit (existing)

References (see BIBLIOGRAPHY.bib):
    \cite{Heisenberg1927} - Uncertainty principle
    \cite{Hawking1975} - Black hole thermodynamics
    \cite{RyuTakayanagi2006} - Holographic entropy
    \cite{Weinberg1989} - Cosmological constant problem
"""

from .unified import UnifiedThetaProof, UnifiedProofResult

# Import new proof modules
from .quantum import (
    UncertaintyProofs,
    HeisenbergResult,
    EnergyTimeResult,
    EntropicResult,
    compute_heisenberg_theta,
    compute_energy_time_theta,
    compute_entropic_theta,
)

from .gravity import (
    HawkingProofs,
    HawkingRadiationResult,
    PageTimeResult,
    AreaQuantizationResult,
    compute_hawking_theta,
    compute_page_time_theta,
    compute_area_quantization_theta,
    HolographicProofs,
    RyuTakayanagiResult,
    EntanglementWedgeResult,
    compute_rt_theta,
    compute_wedge_theta,
)

from .cosmology import (
    VacuumEnergyProofs,
    VacuumEnergyResult,
    DarkEnergyEOSResult,
    compute_vacuum_theta,
    compute_dark_energy_theta,
)

__all__ = [
    # Unified interface
    "UnifiedThetaProof",
    "UnifiedProofResult",
    # Uncertainty proofs
    "UncertaintyProofs",
    "HeisenbergResult",
    "EnergyTimeResult",
    "EntropicResult",
    "compute_heisenberg_theta",
    "compute_energy_time_theta",
    "compute_entropic_theta",
    # Hawking proofs
    "HawkingProofs",
    "HawkingRadiationResult",
    "PageTimeResult",
    "AreaQuantizationResult",
    "compute_hawking_theta",
    "compute_page_time_theta",
    "compute_area_quantization_theta",
    # Holographic proofs
    "HolographicProofs",
    "RyuTakayanagiResult",
    "EntanglementWedgeResult",
    "compute_rt_theta",
    "compute_wedge_theta",
    # Cosmology proofs
    "VacuumEnergyProofs",
    "VacuumEnergyResult",
    "DarkEnergyEOSResult",
    "compute_vacuum_theta",
    "compute_dark_energy_theta",
]
