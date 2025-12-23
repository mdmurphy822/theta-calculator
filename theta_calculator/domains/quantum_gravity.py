"""
Quantum Gravity Domain: Planck Scale Emergence as Theta

This module implements theta as the quantum-classical interpolation parameter
for quantum gravity, measuring how "quantum" spacetime is at different scales.

Key Insight: Gravity exhibits a spectrum between:
- theta ~ 0: Classical spacetime (smooth, continuous, GR valid)
- theta ~ 1: Planck scale (quantum geometry, discrete, GR breaks down)

The quantum gravity theta measures where a physical system sits on
the classical-to-quantum-geometric spectrum.

Key Mappings:
- Length scale: theta = l_P / L (smaller lengths → more quantum)
- Energy scale: theta = E / E_P (higher energies → more quantum)
- Curvature: theta = R / R_P (higher curvature → more quantum)
- Black hole mass: theta = m_P / M (smaller mass → more quantum)

References (see BIBLIOGRAPHY.bib):
    \\cite{Einstein1915GR} - Einstein field equations (foundation of GR)
    \\cite{Wheeler1957} - Quantum geometrodynamics, quantum foam
    \\cite{Wheeler1990ItFromBit} - "It from Bit" information-theoretic foundation
    \\cite{MisnerThorneWheeler1973} - Gravitation (definitive GR textbook)
    \\cite{Rovelli2004} - Quantum Gravity (Loop quantum gravity)
    \\cite{Ashtekar2004} - Background Independent Quantum Gravity
    \\cite{Thiemann2007} - Modern Canonical Quantum General Relativity
    \\cite{Immirzi1997} - Barbero-Immirzi parameter
    \\cite{Penrose1996} - Gravitational decoherence
    \\cite{Diosi1989} - Gravitational decoherence model
    \\cite{Bekenstein1973} - Black holes and entropy
    \\cite{Hawking1975} - Particle creation by black holes
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class SpacetimeRegime(Enum):
    """Classification of spacetime regimes."""
    CLASSICAL = "classical"                # theta < 0.001: GR perfectly valid
    SEMICLASSICAL = "semiclassical"        # 0.001 <= theta < 0.1: QFT in curved spacetime
    TRANS_PLANCKIAN = "trans_planckian"    # 0.1 <= theta < 0.5: Quantum corrections matter
    QUANTUM_FOAM = "quantum_foam"          # 0.5 <= theta < 0.9: Discrete structure emerges
    PLANCK_SCALE = "planck_scale"          # theta >= 0.9: Full quantum gravity


class QuantumGravityTheory(Enum):
    """Different approaches to quantum gravity."""
    LOOP = "loop_quantum_gravity"          # Discrete spacetime from spin networks
    STRING = "string_theory"               # Extended objects replace points
    CAUSAL_SET = "causal_set"              # Discrete causal structure
    ASYMPTOTIC_SAFETY = "asymptotic_safety"  # Non-perturbative UV fixed point
    EMERGENT = "emergent_gravity"          # Gravity as emergent phenomenon


@dataclass
class QuantumGravitySystem:
    """
    A physical system for quantum gravity theta analysis.

    Attributes:
        name: System identifier
        length_scale_m: Characteristic length in meters
        energy_ev: Characteristic energy in eV
        mass_kg: Characteristic mass in kg (if applicable)
        curvature_m2: Ricci curvature R in m^-2 (if applicable)
        theory: Relevant QG theory (if applicable)
        description: Physical description
    """
    name: str
    length_scale_m: float
    energy_ev: float
    mass_kg: Optional[float] = None
    curvature_m2: Optional[float] = None
    theory: Optional[QuantumGravityTheory] = None
    description: Optional[str] = None


# =============================================================================
# PLANCK UNITS (CODATA 2022)
# =============================================================================

# Fundamental constants
HBAR = 1.054571817e-34      # J·s
C = 299792458               # m/s
G = 6.67430e-11             # m³/(kg·s²)
K_B = 1.380649e-23          # J/K
EV_TO_J = 1.602176634e-19   # J/eV

# Planck units
L_PLANCK = np.sqrt(HBAR * G / C**3)        # 1.616255e-35 m
T_PLANCK = np.sqrt(HBAR * G / C**5)        # 5.391247e-44 s
M_PLANCK = np.sqrt(HBAR * C / G)           # 2.176434e-8 kg
E_PLANCK = M_PLANCK * C**2                 # 1.956e9 J
E_PLANCK_EV = E_PLANCK / EV_TO_J           # 1.22e28 eV
T_PLANCK_K = E_PLANCK / K_B                # 1.417e32 K

# Planck curvature (dimensional analysis: 1/L_P^2)
R_PLANCK = 1 / L_PLANCK**2                 # ~3.8e69 m^-2

# Loop Quantum Gravity: Minimum area gap
IMMIRZI_PARAMETER = 0.2375  # Barbero-Immirzi parameter (from black hole entropy)
A_MIN_LQG = 4 * np.pi * IMMIRZI_PARAMETER * np.sqrt(3) * L_PLANCK**2  # ~5.2e-70 m²


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_length_theta(system: QuantumGravitySystem) -> float:
    """
    Compute theta from length scale.

    Reference: Rovelli (2004), Ch. 1

    theta = l_P / L

    Smaller systems → closer to Planck length → higher theta.
    """
    if system.length_scale_m <= 0:
        return 1.0

    theta = L_PLANCK / system.length_scale_m
    return np.clip(theta, 0.0, 1.0)


def compute_energy_theta(system: QuantumGravitySystem) -> float:
    """
    Compute theta from energy scale.

    Reference: Penrose (1996)

    theta = E / E_P

    Higher energies → closer to Planck energy → higher theta.
    """
    theta = system.energy_ev / E_PLANCK_EV
    return np.clip(theta, 0.0, 1.0)


def compute_mass_theta(system: QuantumGravitySystem) -> float:
    """
    Compute theta from mass scale.

    Reference: Bekenstein (1973), Hawking (1975)

    theta measures closeness to Planck mass on log scale:
    - theta = 1 at M = m_P (Planck mass)
    - theta → 0 for M >> m_P or M << m_P

    This captures that quantum gravity effects are strongest at Planck scale,
    not for arbitrarily light particles (which are QM, not QG).
    """
    if system.mass_kg is None or system.mass_kg <= 0:
        return compute_energy_theta(system)

    # Logarithmic distance from Planck mass
    log_ratio = abs(np.log10(system.mass_kg / M_PLANCK))

    # theta = exp(-log_ratio) peaks at M = m_P
    # Scale factor of 2 gives reasonable falloff
    theta = np.exp(-log_ratio / 2)
    return np.clip(theta, 0.0, 1.0)


def compute_curvature_theta(system: QuantumGravitySystem) -> float:
    """
    Compute theta from spacetime curvature.

    Reference: Ashtekar & Lewandowski (2004)

    theta = R / R_P (or sqrt for gentler scaling)

    Higher curvature → stronger gravity → more quantum effects.
    """
    if system.curvature_m2 is None or system.curvature_m2 <= 0:
        return compute_length_theta(system)

    # Use sqrt for gentler scaling across many orders of magnitude
    theta = np.sqrt(system.curvature_m2 / R_PLANCK)
    return np.clip(theta, 0.0, 1.0)


def compute_quantum_gravity_theta(system: QuantumGravitySystem) -> float:
    """
    Compute unified theta for a quantum gravity system.

    For massive objects (especially black holes), mass_theta = m_P/M is used.
    For particle-like objects, length_theta = l_P/L is primary.
    Energy_theta is only used for collision/scattering processes.

    The key insight: smaller objects (relative to Planck scale) are more quantum.
    """
    theta_L = compute_length_theta(system)

    # For objects with mass (black holes, particles), use mass theta
    # This gives m_P/M, so smaller masses → higher theta (more quantum)
    if system.mass_kg is not None:
        theta_M = compute_mass_theta(system)
        # For massive objects, length and mass are the relevant scales
        # Energy theta would give E=Mc² which is misleading
        theta = max(theta_L, theta_M)
    else:
        # For collision processes or abstract scales, use energy
        theta_E = compute_energy_theta(system)
        theta = max(theta_L, theta_E)

    # Curvature adds quantum effects for curved spacetimes
    if system.curvature_m2 is not None:
        theta_R = compute_curvature_theta(system)
        theta = max(theta, theta_R)

    return np.clip(theta, 0.0, 1.0)


def classify_regime(theta: float) -> SpacetimeRegime:
    """Classify spacetime regime from theta."""
    if theta < 0.001:
        return SpacetimeRegime.CLASSICAL
    elif theta < 0.1:
        return SpacetimeRegime.SEMICLASSICAL
    elif theta < 0.5:
        return SpacetimeRegime.TRANS_PLANCKIAN
    elif theta < 0.9:
        return SpacetimeRegime.QUANTUM_FOAM
    else:
        return SpacetimeRegime.PLANCK_SCALE


# =============================================================================
# BLACK HOLE QUANTUM PROPERTIES
# =============================================================================

def schwarzschild_radius(mass_kg: float) -> float:
    """
    Schwarzschild radius: r_s = 2GM/c²

    Reference: Schwarzschild (1916)
    """
    return 2 * G * mass_kg / C**2


def hawking_temperature_k(mass_kg: float) -> float:
    """
    Hawking temperature: T_H = ℏc³/(8πGMk_B)

    Reference: Hawking (1975)
    """
    return HBAR * C**3 / (8 * np.pi * G * mass_kg * K_B)


def bekenstein_entropy(mass_kg: float) -> float:
    """
    Bekenstein-Hawking entropy: S = A/(4l_P²) = 4πGM²/(ℏc)

    Reference: Bekenstein (1973), Hawking (1975)

    Returns entropy in units of k_B.
    """
    r_s = schwarzschild_radius(mass_kg)
    area = 4 * np.pi * r_s**2
    return area / (4 * L_PLANCK**2)


def black_hole_evaporation_time_s(mass_kg: float) -> float:
    """
    Black hole evaporation time: t ~ 5120πG²M³/(ℏc⁴)

    Reference: Hawking (1975)
    """
    return 5120 * np.pi * G**2 * mass_kg**3 / (HBAR * C**4)


def black_hole_theta(mass_kg: float) -> float:
    """
    Compute theta for a black hole.

    Small BHs (near Planck mass) → theta ~ 1 (quantum)
    Large BHs (stellar, supermassive) → theta ~ 0 (classical)

    Uses logarithmic distance from Planck mass.
    """
    log_ratio = abs(np.log10(mass_kg / M_PLANCK))
    theta = np.exp(-log_ratio / 2)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# LOOP QUANTUM GRAVITY DISCRETENESS
# =============================================================================

def area_eigenvalue(j: float) -> float:
    """
    LQG area eigenvalue: A_j = 8πγl_P² √(j(j+1))

    Reference: Ashtekar & Lewandowski (2004)

    Args:
        j: Half-integer spin label (j = 1/2, 1, 3/2, ...)
    """
    return 8 * np.pi * IMMIRZI_PARAMETER * L_PLANCK**2 * np.sqrt(j * (j + 1))


def volume_eigenvalue_approx(j: float) -> float:
    """
    Approximate LQG volume eigenvalue scaling.

    Reference: Thiemann (2007)

    V ~ l_P³ * j^(3/2) for large j
    """
    return L_PLANCK**3 * j**1.5


def area_theta(area_m2: float) -> float:
    """
    Compute theta from area (LQG perspective).

    Areas near A_min are quantum (theta ~ 1).
    Large areas are classical (theta ~ 0).
    """
    return np.clip(A_MIN_LQG / area_m2, 0.0, 1.0)


# =============================================================================
# EXAMPLE QUANTUM GRAVITY SYSTEMS
# =============================================================================

# Mass scales for reference
M_ELECTRON = 9.1093837e-31      # kg
M_PROTON = 1.67262192e-27       # kg
M_SUN = 1.989e30                # kg
M_EARTH = 5.972e24              # kg

# Energy scales
EV_TEV = 1e12                   # 1 TeV in eV
EV_GEV = 1e9                    # 1 GeV in eV
EV_MEV = 1e6                    # 1 MeV in eV

QUANTUM_GRAVITY_SYSTEMS: Dict[str, QuantumGravitySystem] = {
    # Everyday classical systems
    "human_scale": QuantumGravitySystem(
        name="Human Scale (1 meter)",
        length_scale_m=1.0,
        energy_ev=0.025,  # Room temperature thermal energy
        description="Everyday human scale - completely classical",
    ),
    "electron": QuantumGravitySystem(
        name="Electron",
        length_scale_m=2.8e-15,  # Classical electron radius
        energy_ev=0.511e6,       # Electron rest mass
        mass_kg=M_ELECTRON,
        description="Elementary particle - QM but classical gravity",
    ),
    "proton": QuantumGravitySystem(
        name="Proton",
        length_scale_m=0.87e-15,  # Proton charge radius
        energy_ev=938.3e6,        # Proton rest mass
        mass_kg=M_PROTON,
        description="Composite particle - QCD but classical gravity",
    ),

    # High-energy experiments
    "lhc_collision": QuantumGravitySystem(
        name="LHC Collision (13.6 TeV)",
        length_scale_m=1e-19,     # Probed scale
        energy_ev=13.6e12,        # Center of mass energy
        description="Highest human-made energies - still far from Planck",
    ),
    "cosmic_ray_record": QuantumGravitySystem(
        name="Oh-My-God Particle (3e20 eV)",
        length_scale_m=1e-26,     # Effective scale
        energy_ev=3e20,           # Most energetic cosmic ray detected
        description="Highest energy particle ever observed",
    ),

    # Cosmological
    "inflation_energy": QuantumGravitySystem(
        name="Cosmic Inflation Energy Scale",
        length_scale_m=1e-29,     # Approximate GUT scale
        energy_ev=1e25,           # ~10^16 GeV GUT scale
        description="Energy scale of inflation - QG effects possible",
    ),
    "big_bang_singularity": QuantumGravitySystem(
        name="Big Bang (Planck Era)",
        length_scale_m=L_PLANCK,
        energy_ev=E_PLANCK_EV,
        curvature_m2=R_PLANCK,
        description="Initial singularity - full quantum gravity required",
    ),

    # Black holes
    "stellar_black_hole": QuantumGravitySystem(
        name="Stellar Black Hole (10 M☉)",
        length_scale_m=schwarzschild_radius(10 * M_SUN),  # ~30 km
        energy_ev=10 * M_SUN * C**2 / EV_TO_J,
        mass_kg=10 * M_SUN,
        description="Stellar mass BH - classical horizon",
    ),
    "sgr_a_star": QuantumGravitySystem(
        name="Sgr A* (4e6 M☉)",
        length_scale_m=schwarzschild_radius(4e6 * M_SUN),
        energy_ev=4e6 * M_SUN * C**2 / EV_TO_J,
        mass_kg=4e6 * M_SUN,
        description="Milky Way SMBH - very classical",
    ),
    "primordial_bh": QuantumGravitySystem(
        name="Primordial Black Hole (10^12 kg)",
        length_scale_m=schwarzschild_radius(1e12),
        energy_ev=1e12 * C**2 / EV_TO_J,
        mass_kg=1e12,
        description="Asteroid-mass PBH - evaporating via Hawking radiation",
    ),
    "planck_mass_bh": QuantumGravitySystem(
        name="Planck Mass Black Hole",
        length_scale_m=L_PLANCK,
        energy_ev=E_PLANCK_EV,
        mass_kg=M_PLANCK,
        curvature_m2=R_PLANCK,
        theory=QuantumGravityTheory.LOOP,
        description="Minimal black hole - full quantum gravity",
    ),

    # Theoretical constructs
    "string_scale": QuantumGravitySystem(
        name="String Theory Scale",
        length_scale_m=1e-34,  # Often ~ 10 * l_P
        energy_ev=1e27,       # ~ 0.1 E_P
        theory=QuantumGravityTheory.STRING,
        description="Characteristic string length scale",
    ),
    "lqg_spin_network": QuantumGravitySystem(
        name="LQG Spin Network Node",
        length_scale_m=L_PLANCK,
        energy_ev=E_PLANCK_EV,
        theory=QuantumGravityTheory.LOOP,
        description="Discrete quantum of space in loop QG",
    ),
    "causal_set_element": QuantumGravitySystem(
        name="Causal Set Spacetime Atom",
        length_scale_m=L_PLANCK,
        energy_ev=E_PLANCK_EV,
        theory=QuantumGravityTheory.CAUSAL_SET,
        description="Fundamental spacetime event in causal set theory",
    ),

    # Quantum effects at macroscopic scales
    "gravitational_wave_detector": QuantumGravitySystem(
        name="LIGO at Quantum Limit",
        length_scale_m=1e-19,  # Displacement sensitivity
        energy_ev=1e-15,       # ~femtojoule measurement
        description="Macroscopic quantum measurement of spacetime",
    ),
    "atom_interferometry": QuantumGravitySystem(
        name="Atom Interferometry Gravity Test",
        length_scale_m=1e-6,   # Atomic wavepacket
        energy_ev=1e-9,        # Cold atoms
        description="Testing QM-gravity interface at lab scale",
    ),
}


# =============================================================================
# CROSS-DOMAIN CONNECTIONS
# =============================================================================

def gravity_induced_decoherence_rate(mass_kg: float, superposition_size_m: float) -> float:
    """
    Penrose-Diósi gravitational decoherence rate.

    Reference: Penrose (1996)

    Δt ~ ℏ / E_G where E_G = Gm²/Δx

    Returns decoherence rate in Hz.
    """
    e_gravity = G * mass_kg**2 / superposition_size_m
    return e_gravity / HBAR


def gravitational_theta_for_superposition(mass_kg: float, superposition_size_m: float,
                                          coherence_time_s: float) -> float:
    """
    Compute theta for quantum superposition in gravitational field.

    If gravitational decoherence time < coherence time, gravity wins → lower theta.
    """
    decoherence_rate = gravity_induced_decoherence_rate(mass_kg, superposition_size_m)
    if decoherence_rate <= 0:
        return 1.0

    decoherence_time = 1 / decoherence_rate
    theta = decoherence_time / (decoherence_time + coherence_time_s)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def quantum_gravity_theta_summary():
    """Print theta analysis for quantum gravity systems."""
    print("=" * 90)
    print("QUANTUM GRAVITY THETA ANALYSIS")
    print("=" * 90)
    print()
    print(f"Planck length: {L_PLANCK:.3e} m")
    print(f"Planck energy: {E_PLANCK_EV:.3e} eV")
    print(f"Planck mass:   {M_PLANCK:.3e} kg")
    print()
    print(f"{'System':<35} {'θ':>10} {'L (m)':>12} {'E (eV)':>12} {'Regime':<18}")
    print("-" * 90)

    for name, system in QUANTUM_GRAVITY_SYSTEMS.items():
        theta = compute_quantum_gravity_theta(system)
        regime = classify_regime(theta)

        l_str = f"{system.length_scale_m:.2e}"
        e_str = f"{system.energy_ev:.2e}"

        print(f"{system.name:<35} {theta:>10.2e} {l_str:>12} {e_str:>12} {regime.value:<18}")

    print()
    print("Key: θ → 0 classical spacetime, θ → 1 Planck scale quantum geometry")
    print("     LQG minimum area gap: A_min ≈ 5.2 × 10^-70 m²")


if __name__ == "__main__":
    quantum_gravity_theta_summary()
