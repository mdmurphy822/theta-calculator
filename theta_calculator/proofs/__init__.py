r"""
Proofs module: Mathematical, numerical, and information-theoretic proofs.

This module contains theta derivations from fundamental physics principles.

Submodules:
    - quantum: Uncertainty principle proofs
    - gravity: Hawking radiation and holographic entropy proofs
    - cosmology: Vacuum energy and dark energy proofs
    - information: Bekenstein bound and Landauer limit (existing)
    - cross_domain: Universal proofs across all scientific domains

References (see BIBLIOGRAPHY.bib):
    \cite{Heisenberg1927} - Uncertainty principle
    \cite{Hawking1975} - Black hole thermodynamics
    \cite{RyuTakayanagi2006} - Holographic entropy
    \cite{Weinberg1989} - Cosmological constant problem
    \cite{Wilson1971} - Renormalization group theory
    \cite{Bekenstein1981} - Information bounds
    \cite{Sornette2003} - Critical phenomena in markets
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

# Cross-domain unification proofs
from .cross_domain import (
    # Information unification
    InformationUnificationProof,
    CrossDomainBekenstein,
    CrossDomainLandauer,
    HolographicCorrespondence,
    compute_universal_information_theta,
    compute_channel_capacity_theta,
    compute_holographic_theta,
    verify_information_bounds,
    # Scale invariance
    ScaleInvarianceProof,
    UniversalityClass,
    CriticalExponent,
    compute_universal_exponents,
    compute_scaling_function,
    verify_universality_class,
    classify_universality_class,
    UNIVERSALITY_CLASSES,
    # Emergent correspondences
    EmergentCorrespondenceProof,
    DomainCorrespondence,
    PhaseTransitionMapping,
    compute_correspondence_theta,
    map_market_to_ferromagnet,
    map_neural_to_ising,
    map_bec_to_consensus,
    map_superconductor_to_flow,
    verify_correspondence,
    KNOWN_CORRESPONDENCES,
    # Renormalization proofs
    RenormalizationProof,
    RGFlowResult,
    FixedPoint,
    compute_rg_flow_theta,
    compute_beta_function,
    find_fixed_points,
    classify_flow_regime,
    verify_rg_consistency,
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
    # Cross-domain: Information unification
    "InformationUnificationProof",
    "CrossDomainBekenstein",
    "CrossDomainLandauer",
    "HolographicCorrespondence",
    "compute_universal_information_theta",
    "compute_channel_capacity_theta",
    "compute_holographic_theta",
    "verify_information_bounds",
    # Cross-domain: Scale invariance
    "ScaleInvarianceProof",
    "UniversalityClass",
    "CriticalExponent",
    "compute_universal_exponents",
    "compute_scaling_function",
    "verify_universality_class",
    "classify_universality_class",
    "UNIVERSALITY_CLASSES",
    # Cross-domain: Emergent correspondences
    "EmergentCorrespondenceProof",
    "DomainCorrespondence",
    "PhaseTransitionMapping",
    "compute_correspondence_theta",
    "map_market_to_ferromagnet",
    "map_neural_to_ising",
    "map_bec_to_consensus",
    "map_superconductor_to_flow",
    "verify_correspondence",
    "KNOWN_CORRESPONDENCES",
    # Cross-domain: Renormalization proofs
    "RenormalizationProof",
    "RGFlowResult",
    "FixedPoint",
    "compute_rg_flow_theta",
    "compute_beta_function",
    "find_fixed_points",
    "classify_flow_regime",
    "verify_rg_consistency",
]
