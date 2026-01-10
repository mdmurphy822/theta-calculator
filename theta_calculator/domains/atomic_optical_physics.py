"""
Atomic and Optical Physics Domain Module

This module maps theta to atomic physics and quantum optics systems including
cavity QED, laser cooling, atom-light interaction, and squeezed states.

Theta Mapping:
    theta -> 0: Classical/thermal/incoherent state
    theta -> 1: Quantum/coherent/entangled state
    theta = Omega * T2: Rabi coherence parameter
    theta = C / (1 + C): Cooperativity normalized
    theta = OD / OD_max: Optical depth ratio
    theta = 1 - V_shot / V_measured: Squeezing parameter

Key Features:
    - Rabi oscillations and coherence
    - Cavity QED cooperativity
    - Optical depth and light-matter coupling
    - Squeezed light below shot noise
    - Laser cooling and atom trapping
    - Atomic clock precision

References:
    @article{Haroche2006,
      author = {Haroche, Serge and Raimond, Jean-Michel},
      title = {Exploring the quantum: atoms, cavities, and photons},
      publisher = {Oxford University Press},
      year = {2006}
    }
    @article{Wineland1998,
      author = {Wineland, D. J. and Monroe, C. and Itano, W. M. and Leibfried, D.
                and King, B. E. and Meekhof, D. M.},
      title = {Experimental issues in coherent quantum-state manipulation of
               trapped atomic ions},
      journal = {J. Res. Natl. Inst. Stand. Technol.},
      year = {1998}
    }
    @article{Chu1998,
      author = {Chu, Steven},
      title = {Nobel Lecture: The manipulation of neutral particles},
      journal = {Rev. Mod. Phys.},
      year = {1998}
    }
    @article{Kimble1998,
      author = {Kimble, H. J.},
      title = {Strong interactions of single atoms and photons in cavity QED},
      journal = {Phys. Scr.},
      year = {1998}
    }
    @article{Caves1981,
      author = {Caves, Carlton M.},
      title = {Quantum-mechanical noise in an interferometer},
      journal = {Phys. Rev. D},
      year = {1981}
    }
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


# =============================================================================
# Physical Constants
# =============================================================================

HBAR = 1.054571817e-34      # Reduced Planck constant (J*s)
K_B = 1.380649e-23          # Boltzmann constant (J/K)
C_LIGHT = 299792458         # Speed of light (m/s)
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
M_E = 9.1093837015e-31      # Electron mass (kg)
A_BOHR = 5.29177210903e-11  # Bohr radius (m)

# Typical atomic transition parameters
GAMMA_RB87_D2 = 2 * np.pi * 6.065e6  # Rb-87 D2 line decay rate (rad/s)
LAMBDA_RB87_D2 = 780.241e-9  # Rb-87 D2 wavelength (m)

# Recoil temperature
T_RECOIL_RB87 = 361.96e-9  # K, recoil limit for Rb-87

# Doppler temperature
T_DOPPLER_RB87 = 145.57e-6  # K, Doppler limit for Rb-87


# =============================================================================
# Enums for Regime Classification
# =============================================================================

class AtomicRegime(Enum):
    """Classification of atomic coherence regimes based on theta."""
    INCOHERENT = "incoherent"        # theta < 0.1: Classical/thermal
    WEAK_COUPLING = "weak_coupling"   # 0.1 <= theta < 0.3: Weak coherence
    INTERMEDIATE = "intermediate"     # 0.3 <= theta < 0.6: Moderate coupling
    STRONG_COUPLING = "strong_coupling"  # 0.6 <= theta < 0.9: Strong coherence
    ULTRASTRONG = "ultrastrong"       # theta >= 0.9: Ultrastrong coupling


class OpticalPhase(Enum):
    """Classification of optical state."""
    THERMAL = "thermal"              # Thermal/chaotic light
    COHERENT = "coherent"            # Laser light, Poissonian
    SQUEEZED = "squeezed"            # Below shot noise in one quadrature
    ENTANGLED = "entangled"          # Non-classical correlations


class TrapType(Enum):
    """Type of atom trapping."""
    MAGNETO_OPTICAL = "mot"          # Magneto-optical trap
    OPTICAL_DIPOLE = "dipole"        # Optical dipole trap
    OPTICAL_LATTICE = "lattice"      # Optical lattice
    ION_TRAP = "ion"                 # Paul/Penning ion trap
    TWEEZER = "tweezer"              # Optical tweezer array
    MAGNETIC = "magnetic"            # Magnetic trap


class CoolingRegime(Enum):
    """Laser cooling regime."""
    THERMAL = "thermal"              # Above Doppler limit
    DOPPLER = "doppler"              # At Doppler limit
    SUB_DOPPLER = "sub_doppler"      # Below Doppler (polarization gradient)
    RECOIL = "recoil"                # Near recoil limit
    DEGENERATE = "degenerate"        # Quantum degenerate


# =============================================================================
# Dataclass for Atomic/Optical Systems
# =============================================================================

@dataclass
class AtomicOpticalSystem:
    """
    An atomic/optical physics system.

    Attributes:
        name: Descriptive name
        atom_species: Atom type (e.g., "Rb87", "Cs133")
        wavelength: Transition wavelength (m)
        linewidth: Natural linewidth (rad/s)
        atom_number: Number of atoms
        temperature: System temperature (K)
        coherence_time: T2 coherence time (s)
        rabi_frequency: Rabi frequency (rad/s)
        cooperativity: Cavity QED cooperativity C
        optical_depth: Optical depth OD
        trap_type: Type of trapping
        squeezing_db: Squeezing level in dB
    """
    name: str
    atom_species: str = "Rb87"
    wavelength: float = LAMBDA_RB87_D2
    linewidth: float = GAMMA_RB87_D2
    atom_number: int = 1000000
    temperature: float = 1e-6  # K
    coherence_time: Optional[float] = None  # s
    rabi_frequency: Optional[float] = None  # rad/s
    cooperativity: Optional[float] = None
    optical_depth: Optional[float] = None
    trap_type: TrapType = TrapType.MAGNETO_OPTICAL
    squeezing_db: float = 0.0  # dB below shot noise


# =============================================================================
# Rabi Oscillations and Coherence
# =============================================================================

def rabi_frequency(dipole_moment: float, E_field: float) -> float:
    r"""
    Calculate Rabi frequency for dipole coupling.

    Omega = d * E / hbar

    Args:
        dipole_moment: Electric dipole moment (C*m)
        E_field: Electric field amplitude (V/m)

    Returns:
        Rabi frequency in rad/s

    Reference: \cite{Haroche2006}
    """
    return dipole_moment * E_field / HBAR


def compute_rabi_theta(
    Omega: float,
    T2: float,
    Gamma: Optional[float] = None
) -> float:
    r"""
    Compute theta for Rabi oscillation coherence.

    Theta = Omega * T2 for weak driving
    For strong driving: theta = Omega / (Omega + Gamma)

    Args:
        Omega: Rabi frequency (rad/s)
        T2: Coherence time (s)
        Gamma: Decay rate (rad/s), optional

    Returns:
        theta in [0, 1]: 0 = incoherent, 1 = fully coherent

    Reference: \cite{Haroche2006}
    """
    if Omega <= 0:
        return 0.0

    if Gamma is not None and Gamma > 0:
        # Strong driving regime
        theta = Omega / (Omega + Gamma)
    else:
        # Coherence parameter
        theta = min(1.0, Omega * T2)

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Cavity QED
# =============================================================================

def cavity_cooperativity(
    g: float,
    kappa: float,
    gamma: float
) -> float:
    r"""
    Calculate cavity QED cooperativity.

    C = g^2 / (kappa * gamma)

    Args:
        g: Atom-cavity coupling rate (rad/s)
        kappa: Cavity decay rate (rad/s)
        gamma: Atomic decay rate (rad/s)

    Returns:
        Cooperativity C (dimensionless)

    Reference: \cite{Kimble1998}
    """
    if kappa <= 0 or gamma <= 0:
        return 0.0
    return g**2 / (kappa * gamma)


def compute_cooperativity_theta(
    C: float,
    C_strong: float = 1.0
) -> float:
    r"""
    Compute theta from cavity cooperativity.

    Theta = C / (1 + C) for normalized cooperativity
    C > 1 indicates strong coupling regime.

    Args:
        C: Cooperativity
        C_strong: Threshold for strong coupling

    Returns:
        theta in [0, 1]

    Reference: \cite{Kimble1998}
    """
    if C <= 0:
        return 0.0

    theta = C / (1 + C)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Optical Depth
# =============================================================================

def optical_depth(
    n_atoms: float,
    cross_section: float,
    length: float
) -> float:
    r"""
    Calculate optical depth.

    OD = n * sigma * L

    Args:
        n_atoms: Atom number density (m^-3)
        cross_section: Absorption cross section (m^2)
        length: Sample length (m)

    Returns:
        Optical depth (dimensionless)
    """
    return n_atoms * cross_section * length


def resonant_cross_section(wavelength: float) -> float:
    r"""
    Resonant absorption cross section.

    sigma = 3 * lambda^2 / (2 * pi)

    Args:
        wavelength: Transition wavelength (m)

    Returns:
        Cross section (m^2)
    """
    return 3 * wavelength**2 / (2 * np.pi)


def compute_optical_depth_theta(
    OD: float,
    OD_target: float = 100.0
) -> float:
    r"""
    Compute theta from optical depth.

    High OD enables strong atom-light coupling.

    Args:
        OD: Optical depth
        OD_target: Target OD for strong coupling

    Returns:
        theta in [0, 1]

    Reference: \cite{Haroche2006}
    """
    if OD <= 0:
        return 0.0

    theta = 1 - np.exp(-OD / OD_target)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Squeezed Light
# =============================================================================

def squeezing_to_linear(db: float) -> float:
    """
    Convert squeezing from dB to linear variance ratio.

    Args:
        db: Squeezing in dB (negative = below shot noise)

    Returns:
        Variance ratio (< 1 for squeezed)
    """
    return 10**(db / 10)


def shot_noise_variance(n_photons: float) -> float:
    r"""
    Shot noise variance for coherent state.

    Var = sqrt(N) for photon number N

    Args:
        n_photons: Mean photon number

    Returns:
        Shot noise standard deviation
    """
    return np.sqrt(max(0, n_photons))


def compute_squeezing_theta(
    squeezing_db: float,
    target_db: float = -15.0
) -> float:
    r"""
    Compute theta for squeezed light.

    Theta measures how close to target squeezing.

    Args:
        squeezing_db: Achieved squeezing (dB, negative = squeezed)
        target_db: Target squeezing (dB)

    Returns:
        theta in [0, 1]: 1 = achieved target squeezing

    Reference: \cite{Caves1981}
    """
    if squeezing_db >= 0:
        return 0.0  # No squeezing

    # More negative dB = more squeezing
    theta = squeezing_db / target_db
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Laser Cooling
# =============================================================================

def doppler_temperature(gamma: float, m: float) -> float:
    r"""
    Doppler cooling limit temperature.

    T_D = hbar * Gamma / (2 * k_B)

    Args:
        gamma: Natural linewidth (rad/s)
        m: Atom mass (kg)

    Returns:
        Doppler temperature (K)

    Reference: \cite{Chu1998}
    """
    return HBAR * gamma / (2 * K_B)


def recoil_temperature(wavelength: float, m: float) -> float:
    r"""
    Recoil limit temperature.

    T_r = hbar^2 * k^2 / (m * k_B)

    Args:
        wavelength: Laser wavelength (m)
        m: Atom mass (kg)

    Returns:
        Recoil temperature (K)

    Reference: \cite{Chu1998}
    """
    k = 2 * np.pi / wavelength
    return (HBAR * k)**2 / (m * K_B)


def compute_cooling_theta(
    T: float,
    T_doppler: float,
    T_recoil: float
) -> float:
    r"""
    Compute theta for laser cooling.

    Theta measures progress toward quantum degeneracy.
    T >> T_D: theta ~ 0 (thermal)
    T ~ T_D: theta ~ 0.5 (Doppler limited)
    T ~ T_r: theta ~ 1 (recoil limited)

    Args:
        T: Actual temperature (K)
        T_doppler: Doppler limit (K)
        T_recoil: Recoil limit (K)

    Returns:
        theta in [0, 1]

    Reference: \cite{Chu1998}
    """
    if T <= 0 or T_doppler <= 0:
        return 0.0

    if T >= T_doppler:
        # Above Doppler limit
        theta = T_doppler / T * 0.5
    elif T >= T_recoil:
        # Between Doppler and recoil
        log_ratio = np.log(T_doppler / T) / np.log(T_doppler / T_recoil)
        theta = 0.5 + 0.5 * log_ratio
    else:
        # Below recoil limit (BEC regime)
        theta = 1.0

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Atomic Clocks
# =============================================================================

def clock_stability(
    tau: float,
    T_interrogation: float,
    n_atoms: int,
    Q: float
) -> float:
    r"""
    Allan deviation for atomic clock.

    sigma_y = 1 / (Q * sqrt(N * tau / T))

    Args:
        tau: Averaging time (s)
        T_interrogation: Single interrogation time (s)
        n_atoms: Number of atoms
        Q: Quality factor (line Q)

    Returns:
        Fractional frequency stability

    Reference: \cite{Wineland1998}
    """
    if Q <= 0 or n_atoms <= 0 or T_interrogation <= 0:
        return 1.0

    return 1 / (Q * np.sqrt(n_atoms * tau / T_interrogation))


def compute_clock_theta(
    stability: float,
    stability_target: float = 1e-18
) -> float:
    r"""
    Compute theta for atomic clock precision.

    Args:
        stability: Achieved stability (fractional)
        stability_target: Target stability

    Returns:
        theta in [0, 1]: 1 = achieved target

    Reference: \cite{Wineland1998}
    """
    if stability <= 0 or stability_target <= 0:
        return 0.0

    theta = stability_target / stability
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# Unified Atomic/Optical Theta
# =============================================================================

def compute_atomic_optical_theta(system: AtomicOpticalSystem) -> float:
    r"""
    Compute unified theta for atomic/optical system.

    Combines multiple physics aspects:
    - Rabi coherence
    - Cavity cooperativity
    - Optical depth
    - Cooling (temperature vs limits)
    - Squeezing

    Args:
        system: AtomicOpticalSystem dataclass

    Returns:
        theta in [0, 1]

    Reference: \cite{Haroche2006}, \cite{Kimble1998}
    """
    thetas = []

    # Rabi coherence contribution
    if system.rabi_frequency is not None and system.coherence_time is not None:
        theta_rabi = compute_rabi_theta(
            system.rabi_frequency,
            system.coherence_time,
            system.linewidth
        )
        thetas.append(theta_rabi)

    # Cooperativity contribution
    if system.cooperativity is not None:
        theta_coop = compute_cooperativity_theta(system.cooperativity)
        thetas.append(theta_coop)

    # Optical depth contribution
    if system.optical_depth is not None:
        theta_od = compute_optical_depth_theta(system.optical_depth)
        thetas.append(theta_od)

    # Temperature/cooling contribution
    if system.temperature > 0:
        T_doppler = doppler_temperature(system.linewidth, 87 * 1.66e-27)
        T_recoil = recoil_temperature(system.wavelength, 87 * 1.66e-27)
        theta_cool = compute_cooling_theta(
            system.temperature, T_doppler, T_recoil
        )
        thetas.append(theta_cool)

    # Squeezing contribution
    if system.squeezing_db < 0:
        theta_squeeze = compute_squeezing_theta(system.squeezing_db)
        thetas.append(theta_squeeze)

    if not thetas:
        return 0.5  # Default

    # Geometric mean
    return np.prod(thetas)**(1/len(thetas))


# =============================================================================
# Classification Functions
# =============================================================================

def classify_atomic_regime(theta: float) -> AtomicRegime:
    """
    Classify atomic physics regime from theta.

    Args:
        theta: Theta value [0, 1]

    Returns:
        AtomicRegime enum
    """
    if theta < 0.1:
        return AtomicRegime.INCOHERENT
    elif theta < 0.3:
        return AtomicRegime.WEAK_COUPLING
    elif theta < 0.6:
        return AtomicRegime.INTERMEDIATE
    elif theta < 0.9:
        return AtomicRegime.STRONG_COUPLING
    else:
        return AtomicRegime.ULTRASTRONG


def classify_optical_phase(squeezing_db: float, is_entangled: bool = False) -> OpticalPhase:
    """
    Classify optical state from squeezing level.

    Args:
        squeezing_db: Squeezing in dB (negative = squeezed)
        is_entangled: Whether state is entangled

    Returns:
        OpticalPhase enum
    """
    if is_entangled:
        return OpticalPhase.ENTANGLED
    elif squeezing_db < -3:
        return OpticalPhase.SQUEEZED
    elif squeezing_db > 3:
        return OpticalPhase.THERMAL
    else:
        return OpticalPhase.COHERENT


def classify_cooling_regime(T: float, T_doppler: float, T_recoil: float) -> CoolingRegime:
    """
    Classify laser cooling regime.

    Args:
        T: Actual temperature (K)
        T_doppler: Doppler limit (K)
        T_recoil: Recoil limit (K)

    Returns:
        CoolingRegime enum
    """
    if T > 10 * T_doppler:
        return CoolingRegime.THERMAL
    elif T > 2 * T_doppler:
        return CoolingRegime.DOPPLER
    elif T > 10 * T_recoil:
        return CoolingRegime.SUB_DOPPLER
    elif T > T_recoil:
        return CoolingRegime.RECOIL
    else:
        return CoolingRegime.DEGENERATE


# =============================================================================
# Example Systems Dictionary
# =============================================================================

ATOMIC_OPTICAL_SYSTEMS: Dict[str, AtomicOpticalSystem] = {
    "rb87_mot": AtomicOpticalSystem(
        name="Rb-87 MOT",
        atom_species="Rb87",
        atom_number=1_000_000_000,
        temperature=150e-6,  # Doppler limit
        trap_type=TrapType.MAGNETO_OPTICAL
    ),
    "rb87_bec": AtomicOpticalSystem(
        name="Rb-87 BEC",
        atom_species="Rb87",
        atom_number=100_000,
        temperature=100e-9,  # 100 nK
        trap_type=TrapType.MAGNETIC,
        optical_depth=100.0
    ),
    "trapped_ion_clock": AtomicOpticalSystem(
        name="Al+ Optical Clock",
        atom_species="Al27",
        wavelength=267.4e-9,  # Clock transition
        linewidth=2 * np.pi * 8e-3,  # ~8 mHz natural linewidth
        atom_number=1,
        temperature=0.5e-3,  # 0.5 mK
        coherence_time=10.0,  # 10 s
        trap_type=TrapType.ION_TRAP
    ),
    "cavity_qed_cs": AtomicOpticalSystem(
        name="Cs Cavity QED",
        atom_species="Cs133",
        wavelength=852e-9,
        atom_number=1,
        cooperativity=100.0,  # Strong coupling
        rabi_frequency=2 * np.pi * 10e6,
        coherence_time=100e-6
    ),
    "squeezed_light_ligo": AtomicOpticalSystem(
        name="LIGO Squeezed Light",
        atom_species="N/A",
        wavelength=1064e-9,
        squeezing_db=-15.0,  # 15 dB squeezing
        atom_number=0
    ),
    "rydberg_array": AtomicOpticalSystem(
        name="Rydberg Tweezer Array",
        atom_species="Rb87",
        atom_number=256,
        temperature=10e-6,
        rabi_frequency=2 * np.pi * 1e6,
        coherence_time=100e-6,
        trap_type=TrapType.TWEEZER
    ),
    "lattice_clock_sr": AtomicOpticalSystem(
        name="Sr Optical Lattice Clock",
        atom_species="Sr87",
        wavelength=698e-9,  # Clock transition
        linewidth=2 * np.pi * 1e-3,  # mHz linewidth
        atom_number=10000,
        temperature=1e-6,
        coherence_time=1.0,
        trap_type=TrapType.OPTICAL_LATTICE
    ),
    "ultracold_fermi": AtomicOpticalSystem(
        name="Ultracold Fermi Gas",
        atom_species="Li6",
        wavelength=671e-9,
        atom_number=1_000_000,
        temperature=50e-9,  # 50 nK
        trap_type=TrapType.OPTICAL_DIPOLE
    ),
    "atom_interferometer": AtomicOpticalSystem(
        name="Atom Interferometer",
        atom_species="Rb87",
        atom_number=1_000_000,
        temperature=1e-6,
        rabi_frequency=2 * np.pi * 50e3,
        coherence_time=1.0
    ),
    "quantum_memory": AtomicOpticalSystem(
        name="Quantum Memory EIT",
        atom_species="Rb87",
        atom_number=100_000_000,
        optical_depth=200.0,
        coherence_time=1e-3
    ),
}


# Precomputed theta values
ATOMIC_OPTICAL_THETA_VALUES: Dict[str, float] = {
    name: compute_atomic_optical_theta(system)
    for name, system in ATOMIC_OPTICAL_SYSTEMS.items()
}
