"""
Vacuum Energy and Dark Energy Proofs

This module implements theta derivations from cosmological observations,
particularly the cosmological constant and dark energy.

Key Insight: The cosmological constant problem arises from the mismatch:
- Quantum prediction: ρ_vac ~ ρ_Planck ~ 10^113 J/m³
- Observed value: ρ_Λ ~ 10^-9 J/m³

This 10^122 discrepancy is "the worst prediction in physics."

Theta Mapping:
- theta ~ 0: Classical cosmology (Λ = 0 or tuned)
- theta ~ 1: Quantum vacuum dominates (ρ_vac ~ ρ_Planck)

The observed small theta suggests unknown physics that suppresses
quantum contributions to vacuum energy.

References (see BIBLIOGRAPHY.bib):
    \cite{Weinberg1989} - The cosmological constant problem
    \cite{Carroll2001} - The cosmological constant
    \cite{PlanckCollaboration2020} - Planck 2018 cosmological parameters
    \cite{Riess1998} - Observational evidence for accelerating universe
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

# Physical constants (SI units)
HBAR = 1.054571817e-34  # [J·s]
C = 2.99792458e8  # [m/s]
G = 6.67430e-11  # [m³/kg/s²]
K_B = 1.380649e-23  # [J/K]

# Planck units
L_PLANCK = np.sqrt(HBAR * G / C**3)  # ~1.6e-35 m
M_PLANCK = np.sqrt(HBAR * C / G)  # ~2.2e-8 kg
T_PLANCK = np.sqrt(HBAR * C**5 / (G * K_B**2))  # ~1.4e32 K
RHO_PLANCK = C**5 / (HBAR * G**2)  # Planck density ~5.2e96 kg/m³

# Cosmological parameters (Planck 2018)
H_0 = 67.4e3 / 3.086e22  # Hubble constant [s⁻¹] (67.4 km/s/Mpc)
OMEGA_LAMBDA = 0.685  # Dark energy density parameter
OMEGA_MATTER = 0.315  # Matter density parameter
RHO_CRITICAL = 3 * H_0**2 / (8 * np.pi * G)  # Critical density [kg/m³]

# Observed cosmological constant
LAMBDA_OBS = 1.1056e-52  # [m⁻²] (from Planck)
RHO_LAMBDA_OBS = LAMBDA_OBS * C**2 / (8 * np.pi * G)  # [J/m³]


class DarkEnergyModel(Enum):
    """Models of dark energy."""
    COSMOLOGICAL_CONSTANT = "lambda"  # w = -1 exactly
    QUINTESSENCE = "quintessence"  # w > -1
    PHANTOM = "phantom"  # w < -1
    THAWING = "thawing"  # w evolves to -1
    FREEZING = "freezing"  # w evolves from -1


@dataclass
class VacuumEnergyResult:
    """
    Result of vacuum energy analysis.

    The cosmological constant Λ relates to vacuum energy:
    ρ_vac = Λc² / (8πG)

    The cosmological constant problem:
    ρ_quantum / ρ_observed ~ 10^122

    Attributes:
        rho_observed: Observed vacuum energy density [J/m³]
        rho_quantum: Quantum prediction (Planck scale) [J/m³]
        discrepancy: ρ_quantum / ρ_observed
        lambda_observed: Cosmological constant [m⁻²]
        theta: ρ_observed / ρ_quantum (incredibly small!)

    Reference: \cite{Weinberg1989}
    """
    rho_observed: float
    rho_quantum: float
    discrepancy: float
    lambda_observed: float
    theta: float
    suppression_factor: float  # 10^(-122)


@dataclass
class DarkEnergyEOSResult:
    """
    Result of dark energy equation of state analysis.

    The equation of state parameter w:
    P = w ρ c²

    For cosmological constant: w = -1 exactly
    Deviations indicate dynamical dark energy.

    Attributes:
        w: Equation of state parameter
        w_uncertainty: Measurement uncertainty in w
        deviation_from_lambda: |w - (-1)|
        theta: Quantum deviation measure
        model: Likely dark energy model

    Reference: \cite{PlanckCollaboration2020}
    """
    w: float
    w_uncertainty: float
    deviation_from_lambda: float
    theta: float
    model: DarkEnergyModel


def compute_vacuum_theta(
    rho_observed: Optional[float] = None,
    use_planck_cutoff: bool = True
) -> VacuumEnergyResult:
    """
    Compute theta from vacuum energy.

    Theta = ρ_observed / ρ_quantum

    This is the "cosmological constant problem" theta.
    The incredibly small value (10^-122) suggests unknown physics.

    Args:
        rho_observed: Observed vacuum energy [J/m³], default from Planck
        use_planck_cutoff: Use Planck scale for quantum prediction

    Returns:
        VacuumEnergyResult with vacuum energy analysis

    Note:
        This gives theta ~ 10^-122, the smallest non-zero physical ratio!

    Reference: \cite{Weinberg1989}
    """
    if rho_observed is None:
        rho_observed = RHO_LAMBDA_OBS

    # Quantum prediction: vacuum energy at Planck scale
    if use_planck_cutoff:
        # ρ_quantum = ρ_Planck × c² converts mass density to energy density
        # ρ_Planck = c⁵/(ℏG²) ≈ 5.16×10⁹⁶ kg/m³
        # ρ_Planck × c² ≈ 4.6×10¹¹³ J/m³
        rho_quantum = RHO_PLANCK * C**2
    else:
        # Lower cutoff (e.g., electroweak scale ~100 GeV)
        # ρ_vac = E_c⁴ / (ℏ³c⁵) ~ (100 GeV)⁴ / (ℏ³c⁵)
        E_EW = 100e9 * 1.6e-19  # 100 GeV in J
        rho_quantum = E_EW**4 / (HBAR**3 * C**5)

    # Discrepancy
    discrepancy = rho_quantum / rho_observed

    # Theta: ratio of observed to quantum
    theta = rho_observed / rho_quantum

    # Lambda from rho
    lambda_obs = 8 * np.pi * G * rho_observed / C**2

    return VacuumEnergyResult(
        rho_observed=rho_observed,
        rho_quantum=rho_quantum,
        discrepancy=discrepancy,
        lambda_observed=lambda_obs,
        theta=theta,
        suppression_factor=theta
    )


def compute_dark_energy_theta(
    w: float = -1.0,
    w_uncertainty: float = 0.03
) -> DarkEnergyEOSResult:
    """
    Compute theta from dark energy equation of state.

    Theta measures deviation from pure cosmological constant (w = -1).

    theta = |w + 1| / |w + 1|_max

    Where |w + 1|_max ~ 0.1 (phantom boundary concerns).

    Physical interpretation:
    - theta = 0: Pure Λ (w = -1 exactly)
    - theta > 0: Dynamical dark energy
    - theta ~ 1: Strong deviation, new physics

    Args:
        w: Equation of state parameter (P = wρc²)
        w_uncertainty: Measurement uncertainty

    Returns:
        DarkEnergyEOSResult with EOS analysis

    Reference: \cite{PlanckCollaboration2020}
    """
    # Deviation from cosmological constant
    deviation = abs(w + 1)

    # Classify model
    if w > -0.99:
        model = DarkEnergyModel.QUINTESSENCE
    elif w < -1.01:
        model = DarkEnergyModel.PHANTOM
    else:
        model = DarkEnergyModel.COSMOLOGICAL_CONSTANT

    # Theta: normalized deviation
    # Scale so that w = -0.9 or w = -1.1 gives theta ~ 1
    max_deviation = 0.1
    theta = deviation / max_deviation
    theta = np.clip(theta, 0.0, 1.0)

    return DarkEnergyEOSResult(
        w=w,
        w_uncertainty=w_uncertainty,
        deviation_from_lambda=deviation,
        theta=theta,
        model=model
    )


def hubble_tension_theta() -> Dict[str, float]:
    """
    Compute theta from Hubble tension.

    The Hubble tension is the discrepancy between:
    - Early universe: H_0 = 67.4 ± 0.5 km/s/Mpc (Planck CMB)
    - Late universe: H_0 = 73.0 ± 1.0 km/s/Mpc (local distance ladder)

    Difference: ~4-5 sigma

    Reference: \cite{Riess2019}
    """
    H_0_early = 67.4  # Planck
    H_0_late = 73.0   # SH0ES
    sigma = 0.5       # Planck uncertainty

    tension = abs(H_0_late - H_0_early)
    sigma_tension = tension / sigma

    # Theta: tension normalized to 5 sigma (discovery threshold)
    theta = sigma_tension / 5.0
    theta = np.clip(theta, 0.0, 1.0)

    return {
        "H_0_early": H_0_early,
        "H_0_late": H_0_late,
        "tension_km_s_Mpc": tension,
        "tension_sigma": sigma_tension,
        "theta": theta
    }


def cosmic_coincidence_theta() -> Dict[str, float]:
    """
    Compute theta from cosmic coincidence problem.

    Why is Ω_Λ ~ Ω_m ~ 0.7, 0.3 TODAY?

    Throughout cosmic history:
    - Early: Ω_Λ << Ω_m (matter dominated)
    - Late: Ω_Λ >> Ω_m (Λ dominated)
    - Now: Ω_Λ ~ Ω_m (coincidence!)

    This is a fine-tuning problem related to anthropic selection.

    Reference: \cite{Weinberg1989}
    """
    # Ratio of dark energy to matter
    ratio = OMEGA_LAMBDA / OMEGA_MATTER

    # Fine-tuning measure: how close are they?
    # If exactly equal, ratio = 1
    closeness = 1 - abs(ratio - 1) / (ratio + 1)

    # Theta: how coincidental is this?
    theta = closeness

    return {
        "omega_lambda": OMEGA_LAMBDA,
        "omega_matter": OMEGA_MATTER,
        "ratio": ratio,
        "coincidence_measure": closeness,
        "theta": theta
    }


class VacuumEnergyProofs:
    """
    Unified interface for vacuum energy theta calculations.

    Reference: \cite{Weinberg1989}
    """

    @staticmethod
    def cosmological_constant() -> Dict[str, Any]:
        """Compute theta from cosmological constant problem."""
        result = compute_vacuum_theta()
        return {
            "theta": result.theta,
            "proof_type": "cosmological_constant",
            "rho_observed": result.rho_observed,
            "rho_quantum": result.rho_quantum,
            "discrepancy": result.discrepancy,
            "confidence": 0.99,  # Well-measured
            "citation": "\\cite{Weinberg1989}"
        }

    @staticmethod
    def dark_energy_eos(w: float = -1.0) -> Dict[str, Any]:
        """Compute theta from dark energy equation of state."""
        result = compute_dark_energy_theta(w)
        return {
            "theta": result.theta,
            "proof_type": "dark_energy_eos",
            "w": result.w,
            "deviation": result.deviation_from_lambda,
            "model": result.model.value,
            "confidence": 0.90,
            "citation": "\\cite{PlanckCollaboration2020}"
        }

    @staticmethod
    def hubble_tension() -> Dict[str, Any]:
        """Compute theta from Hubble tension."""
        result = hubble_tension_theta()
        return {
            "theta": result["theta"],
            "proof_type": "hubble_tension",
            "tension_sigma": result["tension_sigma"],
            "H_0_early": result["H_0_early"],
            "H_0_late": result["H_0_late"],
            "confidence": 0.95,
            "citation": "\\cite{Riess2019}"
        }

    @staticmethod
    def cosmic_coincidence() -> Dict[str, Any]:
        """Compute theta from cosmic coincidence."""
        result = cosmic_coincidence_theta()
        return {
            "theta": result["theta"],
            "proof_type": "cosmic_coincidence",
            "omega_ratio": result["ratio"],
            "confidence": 0.85,
            "citation": "\\cite{Weinberg1989}"
        }


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

COSMOLOGY_EXAMPLES = {
    "observed_vacuum": {
        "description": "Observed cosmological constant",
        "rho": RHO_LAMBDA_OBS,
        "source": "Planck 2018",
    },
    "lambda_cdm": {
        "description": "ΛCDM model (w = -1)",
        "w": -1.0,
    },
    "quintessence": {
        "description": "Quintessence (w = -0.95)",
        "w": -0.95,
    },
    "phantom": {
        "description": "Phantom energy (w = -1.05)",
        "w": -1.05,
    },
}


def vacuum_energy_theta_summary():
    """Print theta analysis for cosmological vacuum energy."""
    print("=" * 70)
    print("VACUUM ENERGY / DARK ENERGY THETA ANALYSIS")
    print("=" * 70)
    print()

    # Cosmological constant problem
    result = compute_vacuum_theta()
    print("COSMOLOGICAL CONSTANT PROBLEM:")
    print(f"  ρ_observed: {result.rho_observed:.2e} J/m³")
    print(f"  ρ_quantum:  {result.rho_quantum:.2e} J/m³")
    print(f"  Discrepancy: {result.discrepancy:.2e}")
    print(f"  θ = {result.theta:.2e}")
    print(f"  (This is the 10^122 problem!)")
    print()

    # Dark energy EOS
    print("DARK ENERGY EQUATION OF STATE:")
    print(f"{'Model':<30} {'w':>10} {'|w+1|':>10} {'θ':>10}")
    print("-" * 60)

    for name, sys in COSMOLOGY_EXAMPLES.items():
        if "w" in sys:
            result = compute_dark_energy_theta(sys["w"])
            print(f"{sys['description']:<30} "
                  f"{sys['w']:>10.3f} "
                  f"{result.deviation_from_lambda:>10.3f} "
                  f"{result.theta:>10.3f}")

    print()

    # Hubble tension
    h_result = hubble_tension_theta()
    print(f"HUBBLE TENSION:")
    print(f"  H_0 (early): {h_result['H_0_early']:.1f} km/s/Mpc")
    print(f"  H_0 (late):  {h_result['H_0_late']:.1f} km/s/Mpc")
    print(f"  Tension: {h_result['tension_sigma']:.1f}σ")
    print(f"  θ = {h_result['theta']:.3f}")


if __name__ == "__main__":
    vacuum_energy_theta_summary()
