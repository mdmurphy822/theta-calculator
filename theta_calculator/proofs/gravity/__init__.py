r"""
Gravity Proofs Module

This module contains theta proofs based on gravitational physics,
particularly black hole thermodynamics and holography.

Proofs:
    - Hawking proofs: Radiation power, Page time, area quantization
    - Holographic proofs: Ryu-Takayanagi, entanglement wedge

References (see BIBLIOGRAPHY.bib):
    \cite{Hawking1975} - Particle creation by black holes
    \cite{Bekenstein1973} - Black holes and entropy
    \cite{RyuTakayanagi2006} - Holographic entanglement entropy
"""

from .hawking_proofs import (
    HawkingProofs,
    HawkingRadiationResult,
    PageTimeResult,
    AreaQuantizationResult,
    compute_hawking_theta,
    compute_page_time_theta,
    compute_area_quantization_theta,
)

from .holographic_proofs import (
    HolographicProofs,
    RyuTakayanagiResult,
    EntanglementWedgeResult,
    compute_rt_theta,
    compute_wedge_theta,
)

__all__ = [
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
]
