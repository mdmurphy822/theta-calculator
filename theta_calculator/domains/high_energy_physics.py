"""
High Energy Physics Domain Module

This module maps theta to high-energy physics systems including QCD,
lattice gauge theory, collider experiments, and particle phenomenology.

Theta Mapping:
    theta -> 0: Asymptotic freedom (high Q^2), weak coupling
    theta -> 1: Confinement (low Q^2), strong coupling
    theta = alpha_s(Q^2) / alpha_s(Lambda_QCD): Running coupling
    theta = a / a_c: Lattice spacing ratio (continuum limit)
    theta = sigma / sigma_0: Confinement string tension

Key Features:
    - QCD running coupling and asymptotic freedom
    - Lattice QCD spacing and continuum limit
    - Confinement and string tension
    - Chiral symmetry breaking
    - Quark-gluon plasma transitions
    - Collider kinematics and precision

References:
    @article{Gross1973,
      author = {Gross, David J. and Wilczek, Frank},
      title = {Ultraviolet behavior of non-abelian gauge theories},
      journal = {Phys. Rev. Lett.},
      volume = {30},
      pages = {1343},
      year = {1973}
    }
    @article{Politzer1973,
      author = {Politzer, H. David},
      title = {Reliable perturbative results for strong interactions?},
      journal = {Phys. Rev. Lett.},
      volume = {30},
      pages = {1346},
      year = {1973}
    }
    @article{Wilson1974,
      author = {Wilson, Kenneth G.},
      title = {Confinement of quarks},
      journal = {Phys. Rev. D},
      volume = {10},
      pages = {2445},
      year = {1974}
    }
    @article{Polyakov1987,
      author = {Polyakov, A. M.},
      title = {Gauge fields and strings},
      publisher = {Harwood Academic},
      year = {1987}
    }
    @article{PDG2022,
      author = {Particle Data Group},
      title = {Review of particle physics},
      journal = {Prog. Theor. Exp. Phys.},
      year = {2022}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


# =============================================================================
# Physical Constants
# =============================================================================

# QCD scale (GeV) - confinement threshold
LAMBDA_QCD = 0.217  # Lambda_MS-bar in GeV (PDG 2022)

# Strong coupling at Z mass
ALPHA_S_MZ = 0.1179  # alpha_s(M_Z) (PDG 2022)
M_Z = 91.1876  # Z boson mass (GeV)

# String tension
SIGMA_0 = 0.44  # GeV^2, typical hadronic string tension

# Quark masses (GeV) - MS-bar
M_UP = 2.16e-3
M_DOWN = 4.67e-3
M_STRANGE = 0.093
M_CHARM = 1.27
M_BOTTOM = 4.18
M_TOP = 172.76

# Lattice QCD typical values
A_TYPICAL = 0.1  # fm, typical lattice spacing
A_CONTINUUM = 0.01  # fm, approaching continuum limit

# QGP critical temperature (GeV)
T_QGP = 0.154  # GeV ~ 1.7e12 K


# =============================================================================
# Enums for Regime Classification
# =============================================================================

class ParticleRegime(Enum):
    """Classification of QCD coupling regimes based on theta."""
    ASYMPTOTIC = "asymptotic"        # theta < 0.1: Q >> Lambda_QCD
    PERTURBATIVE = "perturbative"    # 0.1 <= theta < 0.3: Perturbation theory valid
    TRANSITION = "transition"        # 0.3 <= theta < 0.6: Transition region
    CONFINED = "confined"            # 0.6 <= theta < 0.9: Confinement regime
    STRONGLY_COUPLED = "strongly_coupled"  # theta >= 0.9: Strong coupling


class LatticeRegime(Enum):
    """Classification of lattice QCD regimes."""
    CONTINUUM = "continuum"          # theta < 0.2: Near continuum limit
    FINE = "fine"                    # 0.2 <= theta < 0.4: Fine lattice
    MODERATE = "moderate"            # 0.4 <= theta < 0.6: Moderate spacing
    COARSE = "coarse"                # 0.6 <= theta < 0.8: Coarse lattice
    STRONG_COUPLING = "strong_coupling"  # theta >= 0.8: Strong coupling limit


class ExperimentType(Enum):
    """Classification of HEP experiment types."""
    COLLIDER = "collider"            # pp, e+e-, heavy ion collisions
    FIXED_TARGET = "fixed_target"    # Beam on target experiments
    RARE_DECAY = "rare_decay"        # B physics, kaon physics
    PRECISION = "precision"          # g-2, EDM, etc.
    HEAVY_ION = "heavy_ion"          # QGP studies


class ChiralRegime(Enum):
    """Classification of chiral symmetry breaking."""
    RESTORED = "restored"            # theta < 0.2: Chiral symmetry restored
    PARTIAL = "partial"              # 0.2 <= theta < 0.5: Partial breaking
    BROKEN = "broken"                # 0.5 <= theta < 0.8: Broken symmetry
    DEEPLY_BROKEN = "deeply_broken"  # theta >= 0.8: Deep chiral breaking


# =============================================================================
# Dataclass for HEP Systems
# =============================================================================

@dataclass
class HEPSystem:
    """
    A high-energy physics system characterized by energy scale and coupling.

    Attributes:
        name: Descriptive name
        energy_scale: Characteristic energy (GeV)
        coupling_alpha_s: Strong coupling at this scale
        lattice_spacing: Lattice spacing in fm (if applicable)
        temperature: System temperature in GeV (if applicable)
        quark_mass_ratio: m_s/m_ud ratio (for chiral analysis)
        experiment_type: Type of experiment
        n_colors: Number of QCD colors (default 3)
        n_flavors: Number of active quark flavors
    """
    name: str
    energy_scale: float  # GeV
    coupling_alpha_s: Optional[float] = None
    lattice_spacing: Optional[float] = None  # fm
    temperature: Optional[float] = None  # GeV
    quark_mass_ratio: Optional[float] = None  # m_s / m_ud
    experiment_type: ExperimentType = ExperimentType.COLLIDER
    n_colors: int = 3
    n_flavors: int = 6


# =============================================================================
# QCD Running Coupling
# =============================================================================

def beta_0(n_f: int = 5, n_c: int = 3) -> float:
    r"""
    QCD beta function leading coefficient.

    beta_0 = (11*N_c - 2*N_f) / (12*pi)

    Args:
        n_f: Number of active quark flavors
        n_c: Number of colors (default 3)

    Returns:
        Leading coefficient of beta function

    Reference: \cite{Gross1973}
    """
    return (11 * n_c - 2 * n_f) / (12 * np.pi)


def running_alpha_s(Q: float, n_f: int = 5) -> float:
    r"""
    QCD running coupling at scale Q.

    Uses 1-loop running:
    alpha_s(Q) = alpha_s(M_Z) / (1 + beta_0 * alpha_s(M_Z) * ln(Q^2/M_Z^2))

    Args:
        Q: Energy scale in GeV
        n_f: Number of active flavors

    Returns:
        Strong coupling alpha_s(Q)

    Reference: \cite{Gross1973}
    """
    if Q <= 0:
        return 1.0  # Non-perturbative

    b0 = beta_0(n_f)
    log_ratio = np.log(Q**2 / M_Z**2)
    denominator = 1 + b0 * ALPHA_S_MZ * log_ratio

    if denominator <= 0:
        return 1.0  # Landau pole reached

    return ALPHA_S_MZ / denominator


def compute_qcd_coupling_theta(
    Q: float,
    Q_ref: float = LAMBDA_QCD * 10,
    n_f: int = 5
) -> float:
    r"""
    Compute theta for QCD running coupling.

    Theta measures proximity to asymptotic freedom:
    theta = alpha_s(Q) / alpha_s(Q_ref)

    Args:
        Q: Energy scale in GeV
        Q_ref: Reference scale (default 10*Lambda_QCD)
        n_f: Number of active flavors

    Returns:
        theta in [0, 1]: 0 = asymptotic freedom, 1 = strong coupling

    Reference: \cite{Gross1973}
    """
    alpha_Q = running_alpha_s(Q, n_f)
    alpha_ref = running_alpha_s(Q_ref, n_f)

    if alpha_ref <= 0:
        return 0.0

    theta = alpha_Q / alpha_ref
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Lattice QCD
# =============================================================================

def lattice_beta(a: float, Lambda: float = LAMBDA_QCD) -> float:
    r"""
    Lattice coupling beta from lattice spacing.

    beta = 6 / g^2 where g is the bare coupling.
    For a ~ 0.1 fm, beta ~ 6.0

    Args:
        a: Lattice spacing in fm
        Lambda: QCD scale in GeV

    Returns:
        Lattice beta parameter

    Reference: \cite{Wilson1974}
    """
    if a <= 0:
        return float('inf')  # Continuum limit

    # Approximate: beta ~ 6 / g^2, where g^2 ~ 1 / ln(1/(a*Lambda))
    # Convert a from fm to GeV^-1: 1 fm = 5.068 GeV^-1
    a_gev = a * 5.068
    Lambda_inv = 1.0 / Lambda

    if a_gev >= Lambda_inv:
        return 5.0  # Strong coupling regime

    # 1-loop relation
    log_term = np.log(Lambda_inv / a_gev)
    if log_term <= 0:
        return 5.0

    beta = 6 * log_term / np.log(10)  # Approximate scaling
    return max(5.0, min(7.0, beta))


def compute_lattice_theta(
    a: float,
    a_continuum: float = A_CONTINUUM,
    a_coarse: float = 0.2
) -> float:
    r"""
    Compute theta for lattice spacing.

    Theta = 0: Continuum limit (a -> 0)
    Theta = 1: Strong coupling limit (large a)

    Args:
        a: Lattice spacing in fm
        a_continuum: Target continuum spacing
        a_coarse: Coarse lattice spacing

    Returns:
        theta in [0, 1]

    Reference: \cite{Wilson1974}
    """
    if a <= 0:
        return 0.0
    if a >= a_coarse:
        return 1.0

    # Linear interpolation in log space
    log_a = np.log(a)
    log_min = np.log(a_continuum)
    log_max = np.log(a_coarse)

    theta = (log_a - log_min) / (log_max - log_min)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Confinement and String Tension
# =============================================================================

def string_tension(sigma_0: float = SIGMA_0, T: float = 0.0) -> float:
    r"""
    Temperature-dependent string tension.

    sigma(T) = sigma_0 * (1 - T/T_c)^alpha for T < T_c
    sigma(T) = 0 for T >= T_c (deconfinement)

    Args:
        sigma_0: Zero-temperature string tension (GeV^2)
        T: Temperature in GeV

    Returns:
        String tension in GeV^2

    Reference: \cite{Polyakov1987}
    """
    if T >= T_QGP:
        return 0.0  # Deconfined

    if T <= 0:
        return sigma_0

    # Critical exponent ~0.5 for 3D universality
    alpha = 0.5
    return sigma_0 * (1 - T / T_QGP)**alpha


def compute_confinement_theta(
    T: float,
    sigma_0: float = SIGMA_0
) -> float:
    r"""
    Compute theta for confinement/deconfinement.

    Theta = 1: Full confinement (T = 0)
    Theta = 0: Deconfined (T >= T_c)

    Args:
        T: Temperature in GeV
        sigma_0: Zero-temperature string tension

    Returns:
        theta in [0, 1]

    Reference: \cite{Polyakov1987}
    """
    sigma_T = string_tension(sigma_0, T)

    if sigma_0 <= 0:
        return 0.0

    return np.clip(sigma_T / sigma_0, 0.0, 1.0)


# =============================================================================
# Chiral Symmetry Breaking
# =============================================================================

def chiral_condensate(T: float, sigma_0: float = 1.0) -> float:
    r"""
    Temperature-dependent chiral condensate.

    <qq>(T) = <qq>_0 * (1 - T/T_c)^beta for T < T_c

    Args:
        T: Temperature in GeV
        sigma_0: Zero-temperature condensate (normalized)

    Returns:
        Normalized chiral condensate

    Reference: \cite{PDG2022}
    """
    T_chiral = T_QGP * 0.95  # Slightly below QGP transition

    if T >= T_chiral:
        return 0.0

    if T <= 0:
        return sigma_0

    # 3D universality beta ~ 0.33
    beta = 0.33
    return sigma_0 * (1 - T / T_chiral)**beta


def compute_chiral_theta(
    m_s_over_m_ud: float,
    m_s_over_m_ud_phys: float = 27.3
) -> float:
    r"""
    Compute theta for chiral symmetry breaking.

    Physical quark mass ratio m_s/m_ud ~ 27.3.
    Theta measures deviation from chiral limit (m_q -> 0).

    Args:
        m_s_over_m_ud: Strange to light quark mass ratio
        m_s_over_m_ud_phys: Physical value

    Returns:
        theta in [0, 1]: 0 = chiral limit, 1 = physical masses

    Reference: \cite{PDG2022}
    """
    if m_s_over_m_ud_phys <= 0:
        return 0.0

    theta = m_s_over_m_ud / m_s_over_m_ud_phys
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Collider Kinematics
# =============================================================================

def cms_energy(beam_energy: float, is_collider: bool = True) -> float:
    r"""
    Center of mass energy.

    For collider: sqrt(s) = 2 * E_beam
    For fixed target: sqrt(s) = sqrt(2 * m * E_beam)

    Args:
        beam_energy: Beam energy in GeV
        is_collider: True for collider, False for fixed target

    Returns:
        Center of mass energy in GeV
    """
    if is_collider:
        return 2 * beam_energy
    else:
        m_proton = 0.938  # GeV
        return np.sqrt(2 * m_proton * beam_energy)


def compute_precision_theta(
    precision: float,
    precision_target: float = 1e-6
) -> float:
    r"""
    Compute theta for experimental precision.

    Theta = 1: Achieved target precision
    Theta = 0: Far from target

    Args:
        precision: Achieved relative precision
        precision_target: Target precision

    Returns:
        theta in [0, 1]
    """
    if precision <= 0 or precision_target <= 0:
        return 0.0

    # Log ratio for precision scaling
    theta = precision_target / precision
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Unified HEP Theta
# =============================================================================

def compute_hep_theta(system: HEPSystem) -> float:
    r"""
    Compute unified theta for HEP system.

    Combines multiple physics aspects:
    - Running coupling for collider/precision
    - Lattice spacing for lattice QCD
    - Confinement for thermal systems

    Args:
        system: HEPSystem dataclass

    Returns:
        theta in [0, 1]

    Reference: \cite{Gross1973}, \cite{Wilson1974}
    """
    thetas = []

    # Running coupling contribution
    if system.energy_scale > 0:
        theta_coupling = compute_qcd_coupling_theta(
            system.energy_scale,
            n_f=system.n_flavors
        )
        thetas.append(theta_coupling)

    # Lattice spacing contribution
    if system.lattice_spacing is not None and system.lattice_spacing > 0:
        theta_lattice = compute_lattice_theta(system.lattice_spacing)
        thetas.append(theta_lattice)

    # Temperature/confinement contribution
    if system.temperature is not None:
        theta_conf = compute_confinement_theta(system.temperature)
        thetas.append(theta_conf)

    # Chiral contribution
    if system.quark_mass_ratio is not None:
        theta_chiral = compute_chiral_theta(system.quark_mass_ratio)
        thetas.append(theta_chiral)

    if not thetas:
        return 0.5  # Default

    # Geometric mean of contributions
    return np.prod(thetas)**(1/len(thetas))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_particle_regime(theta: float) -> ParticleRegime:
    """
    Classify particle physics regime from theta.

    Args:
        theta: Theta value [0, 1]

    Returns:
        ParticleRegime enum
    """
    if theta < 0.1:
        return ParticleRegime.ASYMPTOTIC
    elif theta < 0.3:
        return ParticleRegime.PERTURBATIVE
    elif theta < 0.6:
        return ParticleRegime.TRANSITION
    elif theta < 0.9:
        return ParticleRegime.CONFINED
    else:
        return ParticleRegime.STRONGLY_COUPLED


def classify_lattice_regime(theta: float) -> LatticeRegime:
    """
    Classify lattice QCD regime from theta.

    Args:
        theta: Theta value [0, 1]

    Returns:
        LatticeRegime enum
    """
    if theta < 0.2:
        return LatticeRegime.CONTINUUM
    elif theta < 0.4:
        return LatticeRegime.FINE
    elif theta < 0.6:
        return LatticeRegime.MODERATE
    elif theta < 0.8:
        return LatticeRegime.COARSE
    else:
        return LatticeRegime.STRONG_COUPLING


def classify_chiral_regime(theta: float) -> ChiralRegime:
    """
    Classify chiral symmetry regime from theta.

    Args:
        theta: Theta value [0, 1]

    Returns:
        ChiralRegime enum
    """
    if theta < 0.2:
        return ChiralRegime.RESTORED
    elif theta < 0.5:
        return ChiralRegime.PARTIAL
    elif theta < 0.8:
        return ChiralRegime.BROKEN
    else:
        return ChiralRegime.DEEPLY_BROKEN


# =============================================================================
# Example Systems Dictionary
# =============================================================================

HEP_SYSTEMS: Dict[str, HEPSystem] = {
    "lhc_13tev": HEPSystem(
        name="LHC 13 TeV Collisions",
        energy_scale=13000.0,  # GeV
        coupling_alpha_s=0.085,
        experiment_type=ExperimentType.COLLIDER,
        n_flavors=6
    ),
    "lattice_physical": HEPSystem(
        name="Lattice QCD Physical Point",
        energy_scale=2.0,
        lattice_spacing=0.09,  # fm
        quark_mass_ratio=27.3,
        experiment_type=ExperimentType.PRECISION,
        n_flavors=4
    ),
    "lattice_coarse": HEPSystem(
        name="Lattice QCD Coarse",
        energy_scale=1.0,
        lattice_spacing=0.15,  # fm
        quark_mass_ratio=27.3,
        n_flavors=3
    ),
    "qgp_rhic": HEPSystem(
        name="RHIC QGP",
        energy_scale=200.0,  # per nucleon
        temperature=0.17,  # GeV ~ 2e12 K
        experiment_type=ExperimentType.HEAVY_ION,
        n_flavors=3
    ),
    "qgp_lhc": HEPSystem(
        name="LHC Heavy Ion QGP",
        energy_scale=2760.0,  # per nucleon
        temperature=0.30,  # GeV
        experiment_type=ExperimentType.HEAVY_ION,
        n_flavors=4
    ),
    "rare_b_decay": HEPSystem(
        name="B Meson Rare Decays",
        energy_scale=5.28,  # B meson mass
        coupling_alpha_s=0.22,
        experiment_type=ExperimentType.RARE_DECAY,
        n_flavors=5
    ),
    "charm_physics": HEPSystem(
        name="Charm Quark Physics",
        energy_scale=1.5,  # Near charm threshold
        coupling_alpha_s=0.35,
        experiment_type=ExperimentType.PRECISION,
        n_flavors=4
    ),
    "muon_g2": HEPSystem(
        name="Muon g-2 Precision",
        energy_scale=0.1,  # Low scale
        experiment_type=ExperimentType.PRECISION,
        n_flavors=3
    ),
    "hadron_spectroscopy": HEPSystem(
        name="Hadron Spectroscopy",
        energy_scale=1.0,  # Typical hadron mass
        coupling_alpha_s=0.4,
        quark_mass_ratio=27.3,
        n_flavors=3
    ),
    "deep_inelastic": HEPSystem(
        name="Deep Inelastic Scattering",
        energy_scale=100.0,  # High Q^2
        coupling_alpha_s=0.11,
        experiment_type=ExperimentType.FIXED_TARGET,
        n_flavors=5
    ),
}


# Precomputed theta values for example systems
HEP_THETA_VALUES: Dict[str, float] = {
    name: compute_hep_theta(system)
    for name, system in HEP_SYSTEMS.items()
}
