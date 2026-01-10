r"""
Physics Extended Domain: General Relativity, High-Energy Theory, and Phenomenology

This module implements theta as the physics regime interpolation parameter
for general relativity (gr-qc), high-energy theory (hep-th), and
high-energy phenomenology (hep-ph).

## Mapping Definition

This domain maps physical systems to theta via regime parameters:

**Inputs (Physical Analogs):**
- r_s/r -> Schwarzschild radius ratio (GR curvature)
- g_s -> String coupling constant (string theory)
- alpha' -> Regge slope (string tension)
- sqrt(s)/Lambda -> Energy/BSM scale ratio

**Theta Mapping:**
GR: theta = r_s / r (gravitational strength)
HEP-TH: theta = g / g_critical (coupling strength)
HEP-PH: theta = E / Lambda_BSM (new physics proximity)

**Interpretation:**
- theta -> 0: Weak-field / perturbative / SM regime
- theta -> 1: Strong-field / non-perturbative / BSM regime
- 0.1 < theta < 0.9: Transition regime (measurable deviations)

**Key Feature:** Physics exhibits regime transitions from
classical/perturbative to quantum/non-perturbative behavior.

**Important:** This is a PHYSICAL SCORE based on regime parameters.

References (see BIBLIOGRAPHY.bib):
    \cite{Einstein1915GR} - General Relativity
    \cite{Schwarzschild1916} - Schwarzschild solution
    \cite{Maldacena1999} - AdS/CFT correspondence
    \cite{Han2025LQG} - LQG entanglement entropy
    \cite{Brahma2023} - Gravitational entanglement probes
    \cite{LIGOCollaboration2025} - Gravitational wave catalog
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class GRRegime(Enum):
    """General relativity regime classification."""
    FLAT = "flat_spacetime"               # theta < 0.01: Minkowski
    WEAK_FIELD = "weak_field"             # 0.01 <= theta < 0.1: PPN valid
    STRONG_FIELD = "strong_field"         # 0.1 <= theta < 0.5: Full GR needed
    HORIZON = "near_horizon"              # theta >= 0.5: Near r_s


class HEPTheoryRegime(Enum):
    """High-energy theory regime classification."""
    PERTURBATIVE = "perturbative"         # g << 1
    WEAKLY_COUPLED = "weakly_coupled"     # g ~ 0.1
    MODERATELY_COUPLED = "moderately_coupled"  # g ~ 1
    STRONGLY_COUPLED = "strongly_coupled"  # g >> 1


class BSMScale(Enum):
    """Beyond Standard Model energy scale classification."""
    STANDARD_MODEL = "standard_model"     # E << 1 TeV
    ELECTROWEAK = "electroweak"           # E ~ 100 GeV - 1 TeV
    TEV_SCALE = "tev_scale"               # E ~ 1-10 TeV
    GUT_SCALE = "gut_scale"               # E ~ 10^16 GeV
    PLANCK_SCALE = "planck_scale"         # E ~ 10^19 GeV


class QuantumGravityRegime(Enum):
    """Quantum gravity regime classification."""
    SEMICLASSICAL = "semiclassical"       # Classical spacetime, quantum matter
    TRANSITION = "transition"             # Quantum corrections important
    PLANCKIAN = "planckian"               # Full quantum gravity needed
    HOLOGRAPHIC = "holographic"           # AdS/CFT regime


class SpacetimeType(Enum):
    """Types of spacetime geometries."""
    MINKOWSKI = "minkowski"               # Flat spacetime
    SCHWARZSCHILD = "schwarzschild"       # Spherically symmetric vacuum
    KERR = "kerr"                         # Rotating black hole
    DE_SITTER = "de_sitter"               # Positive cosmological constant
    ANTI_DE_SITTER = "anti_de_sitter"     # Negative cosmological constant
    FRIEDMANN = "friedmann"               # Cosmological (FRW)


@dataclass
class PhysicsExtendedSystem:
    """
    An extended physics system for theta analysis.

    Represents a physical system characterized by its domain (gr-qc, hep-th, hep-ph),
    length/energy scales, and coupling constants.

    Attributes:
        name: System identifier
        domain: Primary physics domain ("gr-qc", "hep-th", "hep-ph")
        length_scale: Characteristic length (meters)
        energy_scale: Characteristic energy (eV)
        coupling: Relevant coupling constant (dimensionless)
        curvature: Spacetime curvature (1/m^2) for GR
        mass: Central/source mass (kg) for GR
        temperature: System temperature (K) if relevant
        spacetime: Type of spacetime geometry
    """
    name: str
    domain: str  # "gr-qc", "hep-th", "hep-ph"
    length_scale: float
    energy_scale: float
    coupling: Optional[float] = None
    curvature: Optional[float] = None
    mass: Optional[float] = None
    temperature: Optional[float] = None
    spacetime: SpacetimeType = SpacetimeType.MINKOWSKI

    @property
    def is_relativistic(self) -> bool:
        """Check if length scale is comparable to Schwarzschild radius."""
        if self.mass is None:
            return False
        r_s = 2 * G * self.mass / C**2
        return self.length_scale < 10 * r_s

    @property
    def is_quantum_gravitational(self) -> bool:
        """Check if length scale is near Planck length."""
        return self.length_scale < 100 * L_PLANCK


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

G = 6.67430e-11           # m^3/(kg*s^2) - Newton's constant
C = 299792458             # m/s - Speed of light
HBAR = 1.054571817e-34    # J*s - Reduced Planck constant
K_B = 1.380649e-23        # J/K - Boltzmann constant

# Planck scale
M_PLANCK = 2.176434e-8    # kg - Planck mass
L_PLANCK = 1.616255e-35   # m - Planck length
T_PLANCK = 5.391247e-44   # s - Planck time
E_PLANCK_EV = 1.22e28     # eV - Planck energy

# Standard Model / BSM scales
E_ELECTROWEAK_EV = 246e9  # eV - Electroweak scale (~246 GeV)
E_GUT_EV = 1e25           # eV - GUT scale (~10^16 GeV)
E_LHC_MAX_EV = 13.6e12    # eV - LHC collision energy (~13.6 TeV)

# Solar system scales
M_SUN = 1.989e30          # kg
R_SCHWARZSCHILD_SUN = 2953  # m - Schwarzschild radius of Sun


def schwarzschild_radius(mass: float) -> float:
    """Compute Schwarzschild radius r_s = 2GM/c^2."""
    return 2 * G * mass / C**2


def hawking_temperature(mass: float) -> float:
    """Compute Hawking temperature T_H = hbar*c^3 / (8*pi*G*M*k_B)."""
    if mass <= 0:
        return float('inf')
    return HBAR * C**3 / (8 * np.pi * G * mass * K_B)


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_gr_theta(
    r: float,
    mass: float
) -> float:
    r"""
    Compute theta for general relativity regime.

    theta = r_s / r where r_s = 2GM/c^2

    This measures gravitational field strength:
    - theta ~ 0: Weak field (Newtonian limit)
    - theta ~ 1: Near horizon (strong field)

    Args:
        r: Radial distance from center (meters)
        mass: Central mass (kg)

    Returns:
        theta in [0, 1]

    Reference: \cite{Schwarzschild1916}
    """
    if r <= 0:
        return 1.0
    r_s = schwarzschild_radius(mass)
    theta = r_s / r
    return np.clip(theta, 0.0, 1.0)


def compute_curvature_theta(
    curvature: float,
    curvature_planck: float = 1.0 / L_PLANCK**2
) -> float:
    r"""
    Compute theta from spacetime curvature.

    theta = sqrt(R / R_planck)

    where R is Ricci scalar curvature and R_planck ~ 1/l_P^2.

    Args:
        curvature: Ricci curvature (1/m^2)
        curvature_planck: Planck curvature scale

    Returns:
        theta in [0, 1]
    """
    if curvature <= 0:
        return 0.0
    ratio = curvature / curvature_planck
    theta = np.sqrt(ratio)
    return np.clip(theta, 0.0, 1.0)


def compute_hep_th_theta(
    coupling: float,
    critical_coupling: float = 1.0
) -> float:
    r"""
    Compute theta for high-energy theory regime.

    theta = g / g_critical

    where g is the coupling constant and g_critical ~ 1 marks
    the perturbative/non-perturbative boundary.

    Args:
        coupling: Coupling constant (g_s for strings, g for gauge)
        critical_coupling: Critical coupling for transition

    Returns:
        theta in [0, 1]

    Reference: \cite{Maldacena1999}
    """
    if critical_coupling <= 0:
        return 1.0
    theta = coupling / critical_coupling
    return np.clip(theta, 0.0, 1.0)


def compute_hep_ph_theta(
    energy_ev: float,
    bsm_scale_ev: float = 1e13  # Default: 10 TeV
) -> float:
    r"""
    Compute theta for high-energy phenomenology.

    theta = sqrt(s) / Lambda_BSM

    where sqrt(s) is collision energy and Lambda_BSM is
    the new physics scale.

    Args:
        energy_ev: Collision/process energy (eV)
        bsm_scale_ev: BSM physics scale (eV)

    Returns:
        theta in [0, 1]
    """
    if bsm_scale_ev <= 0:
        return 1.0
    theta = energy_ev / bsm_scale_ev
    return np.clip(theta, 0.0, 1.0)


def compute_quantum_gravity_theta(
    length_scale: float,
    planck_length: float = L_PLANCK
) -> float:
    r"""
    Compute theta for quantum gravity regime.

    theta = l_P / L

    where l_P is Planck length and L is characteristic length.

    Args:
        length_scale: Characteristic length (meters)
        planck_length: Planck length

    Returns:
        theta in [0, 1]
    """
    if length_scale <= 0:
        return 1.0
    theta = planck_length / length_scale
    return np.clip(theta, 0.0, 1.0)


def compute_holographic_theta(
    area: float,
    entropy: float
) -> float:
    r"""
    Compute theta from holographic entropy bound.

    theta = S / S_max where S_max = A / (4 * l_P^2)

    Measures saturation of Bekenstein-Hawking entropy bound.

    Args:
        area: Boundary area (m^2)
        entropy: System entropy (dimensionless, in units of k_B)

    Returns:
        theta in [0, 1]

    Reference: \cite{RyuTakayanagi2006}
    """
    s_max = area / (4 * L_PLANCK**2)
    if s_max <= 0:
        return 0.0
    theta = entropy / s_max
    return np.clip(theta, 0.0, 1.0)


def compute_physics_extended_theta(system: PhysicsExtendedSystem) -> float:
    """
    Compute unified theta for an extended physics system.

    Routes to domain-specific calculation based on system.domain.

    Args:
        system: PhysicsExtendedSystem instance

    Returns:
        theta in [0, 1]
    """
    if system.domain == "gr-qc":
        if system.mass is not None and system.length_scale > 0:
            return compute_gr_theta(system.length_scale, system.mass)
        elif system.curvature is not None:
            return compute_curvature_theta(system.curvature)
        else:
            return compute_quantum_gravity_theta(system.length_scale)

    elif system.domain == "hep-th":
        if system.coupling is not None:
            return compute_hep_th_theta(system.coupling)
        else:
            # Use energy-based theta
            return compute_hep_ph_theta(system.energy_scale, E_PLANCK_EV)

    elif system.domain == "hep-ph":
        return compute_hep_ph_theta(system.energy_scale, 1e13)  # 10 TeV default

    else:
        # Default: energy scale relative to Planck
        return np.clip(system.energy_scale / E_PLANCK_EV, 0.0, 1.0)


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_gr_regime(theta: float) -> GRRegime:
    """Classify GR regime from theta value."""
    if theta < 0.01:
        return GRRegime.FLAT
    elif theta < 0.1:
        return GRRegime.WEAK_FIELD
    elif theta < 0.5:
        return GRRegime.STRONG_FIELD
    else:
        return GRRegime.HORIZON


def classify_hep_th_regime(theta: float) -> HEPTheoryRegime:
    """Classify HEP-TH regime from theta."""
    if theta < 0.1:
        return HEPTheoryRegime.PERTURBATIVE
    elif theta < 0.5:
        return HEPTheoryRegime.WEAKLY_COUPLED
    elif theta < 1.0:
        return HEPTheoryRegime.MODERATELY_COUPLED
    else:
        return HEPTheoryRegime.STRONGLY_COUPLED


def classify_bsm_scale(energy_ev: float) -> BSMScale:
    """Classify BSM scale from energy."""
    if energy_ev < 1e11:  # 100 GeV
        return BSMScale.STANDARD_MODEL
    elif energy_ev < 1e13:  # 10 TeV
        return BSMScale.ELECTROWEAK
    elif energy_ev < 1e16:  # 10 PeV
        return BSMScale.TEV_SCALE
    elif energy_ev < 1e27:  # 10^18 GeV
        return BSMScale.GUT_SCALE
    else:
        return BSMScale.PLANCK_SCALE


def classify_quantum_gravity_regime(theta: float) -> QuantumGravityRegime:
    """Classify quantum gravity regime from theta."""
    if theta < 0.01:
        return QuantumGravityRegime.SEMICLASSICAL
    elif theta < 0.1:
        return QuantumGravityRegime.TRANSITION
    elif theta < 0.5:
        return QuantumGravityRegime.PLANCKIAN
    else:
        return QuantumGravityRegime.HOLOGRAPHIC


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

PHYSICS_EXTENDED_SYSTEMS: Dict[str, PhysicsExtendedSystem] = {
    # General Relativity (gr-qc)
    "earth_surface": PhysicsExtendedSystem(
        name="Earth Surface (Weak GR)",
        domain="gr-qc",
        length_scale=6.371e6,           # Earth radius
        energy_scale=0.025,             # Thermal (room temp)
        mass=5.972e24,                  # Earth mass
        spacetime=SpacetimeType.SCHWARZSCHILD,
    ),
    "gps_satellite": PhysicsExtendedSystem(
        name="GPS Satellite",
        domain="gr-qc",
        length_scale=2.66e7,            # GPS orbit radius
        energy_scale=0.025,
        mass=5.972e24,
        spacetime=SpacetimeType.SCHWARZSCHILD,
    ),
    "sun_surface": PhysicsExtendedSystem(
        name="Sun Surface",
        domain="gr-qc",
        length_scale=6.96e8,            # Solar radius
        energy_scale=0.5,               # ~5800 K
        mass=M_SUN,
        spacetime=SpacetimeType.SCHWARZSCHILD,
    ),
    "neutron_star": PhysicsExtendedSystem(
        name="Neutron Star Surface",
        domain="gr-qc",
        length_scale=1e4,               # 10 km radius
        energy_scale=1e6,               # MeV scale
        mass=1.4 * M_SUN,
        spacetime=SpacetimeType.SCHWARZSCHILD,
    ),
    "black_hole_horizon": PhysicsExtendedSystem(
        name="Stellar Black Hole Horizon",
        domain="gr-qc",
        length_scale=3e4,               # ~10 solar mass BH
        energy_scale=1e9,
        mass=10 * M_SUN,
        spacetime=SpacetimeType.SCHWARZSCHILD,
    ),
    "ligo_merger": PhysicsExtendedSystem(
        name="LIGO Binary Merger",
        domain="gr-qc",
        length_scale=1e5,               # Merger separation
        energy_scale=1e12,              # Enormous
        mass=60 * M_SUN,                # Combined mass
        spacetime=SpacetimeType.KERR,
    ),
    # High-Energy Theory (hep-th)
    "weak_string": PhysicsExtendedSystem(
        name="Weakly Coupled String (g_s=0.1)",
        domain="hep-th",
        length_scale=1e-34,             # Near string scale
        energy_scale=1e27,              # String scale
        coupling=0.1,
    ),
    "strong_string": PhysicsExtendedSystem(
        name="Strongly Coupled String (g_s=1)",
        domain="hep-th",
        length_scale=1e-34,
        energy_scale=1e27,
        coupling=1.0,
    ),
    "ads_cft": PhysicsExtendedSystem(
        name="AdS/CFT (N=4 SYM)",
        domain="hep-th",
        length_scale=1e-35,
        energy_scale=1e28,
        coupling=0.5,                   # 't Hooft coupling
        spacetime=SpacetimeType.ANTI_DE_SITTER,
    ),
    "qcd_confinement": PhysicsExtendedSystem(
        name="QCD Confinement Scale",
        domain="hep-th",
        length_scale=1e-15,             # Proton size
        energy_scale=200e6,             # ~200 MeV
        coupling=1.0,                   # alpha_s ~ 1
    ),
    # High-Energy Phenomenology (hep-ph)
    "lhc_collision": PhysicsExtendedSystem(
        name="LHC 13.6 TeV",
        domain="hep-ph",
        length_scale=1e-19,             # Probed length
        energy_scale=13.6e12,           # 13.6 TeV
    ),
    "higgs_scale": PhysicsExtendedSystem(
        name="Higgs Discovery (~125 GeV)",
        domain="hep-ph",
        length_scale=1e-18,
        energy_scale=125e9,             # Higgs mass
    ),
    "top_quark": PhysicsExtendedSystem(
        name="Top Quark Production",
        domain="hep-ph",
        length_scale=1e-18,
        energy_scale=173e9,             # Top quark mass
    ),
    "future_collider": PhysicsExtendedSystem(
        name="Future 100 TeV Collider",
        domain="hep-ph",
        length_scale=1e-20,
        energy_scale=100e12,            # 100 TeV
    ),
    # Quantum Gravity
    "planck_regime": PhysicsExtendedSystem(
        name="Planck Scale",
        domain="gr-qc",
        length_scale=L_PLANCK,
        energy_scale=E_PLANCK_EV,
        curvature=1.0 / L_PLANCK**2,
    ),
    "lqg_spin_foam": PhysicsExtendedSystem(
        name="LQG Spin Foam",
        domain="gr-qc",
        length_scale=10 * L_PLANCK,
        energy_scale=0.1 * E_PLANCK_EV,
        curvature=1e68,                 # Near Planck
    ),
    # Cosmological
    "cmb_horizon": PhysicsExtendedSystem(
        name="CMB Horizon",
        domain="gr-qc",
        length_scale=4.4e26,            # Hubble radius
        energy_scale=2.4e-4,            # 2.7 K CMB
        temperature=2.725,
        spacetime=SpacetimeType.FRIEDMANN,
    ),
    "inflation_scale": PhysicsExtendedSystem(
        name="Inflation Energy Scale",
        domain="hep-th",
        length_scale=1e-26,             # Horizon during inflation
        energy_scale=1e25,              # GUT scale
        coupling=0.01,
    ),
}


# =============================================================================
# GRAVITATIONAL WAVE ANALYSIS
# =============================================================================

def analyze_gw_merger(
    m1_solar: float,
    m2_solar: float,
    separation_m: float
) -> Dict[str, float]:
    """
    Analyze gravitational wave binary merger.

    Returns theta and regime for the binary system.
    """
    m_total = (m1_solar + m2_solar) * M_SUN
    r_s = schwarzschild_radius(m_total)
    theta = compute_gr_theta(separation_m, m_total)

    return {
        "theta": theta,
        "schwarzschild_radius": r_s,
        "separation_over_rs": separation_m / r_s,
        "regime": classify_gr_regime(theta).value,
        "strong_field": theta > 0.1,
    }


def analyze_black_hole_thermodynamics(
    mass_solar: float
) -> Dict[str, float]:
    """
    Analyze black hole thermodynamic properties.
    """
    mass_kg = mass_solar * M_SUN
    r_s = schwarzschild_radius(mass_kg)
    T_hawking = hawking_temperature(mass_kg)
    area = 4 * np.pi * r_s**2
    entropy = area / (4 * L_PLANCK**2)

    return {
        "schwarzschild_radius_m": r_s,
        "hawking_temperature_K": T_hawking,
        "horizon_area_m2": area,
        "bekenstein_entropy": entropy,
        "evaporation_time_s": (mass_kg**3) * 5120 * np.pi * G**2 / (HBAR * C**4),
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def physics_extended_summary():
    """Print theta analysis for physics extended systems."""
    print("=" * 95)
    print("PHYSICS EXTENDED THETA ANALYSIS")
    print("=" * 95)
    print()
    print(f"{'System':<30} {'Domain':<8} {'theta':>8} {'Length':>12} "
          f"{'Energy':>12} {'Regime':<20}")
    print("-" * 95)

    for name, system in PHYSICS_EXTENDED_SYSTEMS.items():
        theta = compute_physics_extended_theta(system)

        # Format length scale
        L = system.length_scale
        if L >= 1e6:
            L_str = f"{L/1e6:.1f} km"
        elif L >= 1:
            L_str = f"{L:.1f} m"
        elif L >= 1e-3:
            L_str = f"{L*1e3:.1f} mm"
        elif L >= 1e-9:
            L_str = f"{L*1e9:.1f} nm"
        elif L >= 1e-15:
            L_str = f"{L*1e15:.1f} fm"
        else:
            L_str = f"{L:.1e} m"

        # Format energy scale
        E = system.energy_scale
        if E >= 1e18:
            E_str = f"{E/1e18:.1f} EeV"
        elif E >= 1e12:
            E_str = f"{E/1e12:.1f} TeV"
        elif E >= 1e9:
            E_str = f"{E/1e9:.1f} GeV"
        elif E >= 1e6:
            E_str = f"{E/1e6:.1f} MeV"
        elif E >= 1e3:
            E_str = f"{E/1e3:.1f} keV"
        else:
            E_str = f"{E:.1f} eV"

        # Get regime
        if system.domain == "gr-qc":
            regime = classify_gr_regime(theta).value
        elif system.domain == "hep-th":
            regime = classify_hep_th_regime(theta).value
        else:
            regime = classify_bsm_scale(system.energy_scale).value

        print(f"{system.name:<30} {system.domain:<8} {theta:>8.4f} {L_str:>12} "
              f"{E_str:>12} {regime:<20}")

    print()
    print("Key: GR theta = r_s/r measures gravitational field strength")
    print("     HEP theta measures approach to BSM or non-perturbative physics")


if __name__ == "__main__":
    physics_extended_summary()
