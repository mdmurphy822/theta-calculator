"""
Cosmology Domain: Theta Across Cosmic History

This module implements theta as the quantum-classical interpolation parameter
for the evolution of the universe from the Big Bang to the far future.

## Mapping Definition

This domain maps cosmic epochs to theta via Planck scale comparison:

**Inputs (Physical Analogs):**
- temperature → CMB/radiation temperature (K)
- energy → Characteristic energy scale (eV)
- time_since_bang → Time since Big Bang (seconds)
- hubble_parameter → Expansion rate H (km/s/Mpc)
- era → Cosmic era classification

**Theta Mapping:**
θ = (E / E_Planck)^n × f(T / T_Planck)

Where E_Planck ≈ 1.22 × 10^19 GeV, T_Planck ≈ 1.42 × 10^32 K

**Interpretation:**
- θ → 1: Planck era (quantum gravity, all forces unified, t < 10^-44 s)
- θ → 0: Present epoch (classical gravity, quantum only at small scales)

**Key Feature:** Cosmic evolution represents the universe's journey from
θ ≈ 1 (Big Bang) to θ → 0 (far future heat death).

**Important:** This is an ANALOGY SCORE comparing energy scales to Planck scale.

References (see BIBLIOGRAPHY.bib):
    \\cite{PlanckCollaboration2020} - Planck 2018 cosmological parameters
    \\cite{Weinberg1972} - Gravitation and Cosmology
    \\cite{Guth1981} - Inflationary universe
    \\cite{Peebles1966} - Big Bang nucleosynthesis
    \\cite{Riess1998} - Discovery of cosmic acceleration
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class CosmicEra(Enum):
    """Major eras in cosmic history."""
    PLANCK = "planck"              # 0 to 10^-44 s
    GUT = "gut"                    # 10^-44 to 10^-36 s
    ELECTROWEAK = "electroweak"    # 10^-36 to 10^-12 s
    QUARK = "quark"                # 10^-12 to 10^-6 s
    HADRON = "hadron"              # 10^-6 to 1 s
    NUCLEOSYNTHESIS = "nucleosynthesis"  # 1 to 300 s
    RADIATION = "radiation"        # 300 s to 47,000 yr
    MATTER = "matter"              # 47,000 yr to 9.8 Gyr
    DARK_ENERGY = "dark_energy"    # 9.8 Gyr to present
    FAR_FUTURE = "far_future"      # > 10^100 yr


@dataclass
class CosmicEpoch:
    """
    A cosmic epoch for theta analysis.

    Attributes:
        name: Epoch identifier
        era: Major cosmic era
        time: Time since Big Bang (seconds)
        temperature: Cosmic temperature (Kelvin)
        energy: Characteristic energy (eV)
        description: Physical description
        key_events: Notable events during this epoch
    """
    name: str
    era: CosmicEra
    time: float           # seconds since Big Bang
    temperature: float    # Kelvin
    energy: float         # eV
    description: str
    key_events: Optional[List[str]] = None


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

HBAR = 1.054571817e-34       # J*s
C = 299792458                # m/s
G = 6.67430e-11              # m^3/(kg*s^2)
K_B = 1.380649e-23           # J/K

# Planck units
T_PLANCK = 1.416784e32       # K (Planck temperature)
E_PLANCK = 1.956e9           # J (Planck energy)
E_PLANCK_EV = 1.22e28        # eV (Planck energy in eV)
T_PLANCK_TIME = 5.391e-44    # s (Planck time)

# Key energy scales (eV)
E_GUT = 1e25                 # GUT scale
E_ELECTROWEAK = 100e9        # Electroweak unification (~100 GeV)
E_QCD = 200e6                # QCD confinement scale (~200 MeV)
E_NUCLEON = 938e6            # Nucleon mass (~938 MeV)
E_ELECTRON = 511e3           # Electron mass (~511 keV)
E_BINDING = 13.6             # Hydrogen binding energy (eV)
E_CMB = 2.35e-4              # CMB photon energy today (~2.7K)


def temperature_to_energy_ev(T: float) -> float:
    """Convert temperature (K) to characteristic energy (eV)."""
    # E = k_B * T, then convert J to eV
    E_joules = K_B * T
    return E_joules / 1.602176634e-19


def energy_to_temperature(E_ev: float) -> float:
    """Convert energy (eV) to temperature (K)."""
    E_joules = E_ev * 1.602176634e-19
    return E_joules / K_B


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_cosmic_theta(epoch: CosmicEpoch) -> float:
    """
    Compute theta for a cosmic epoch.

    theta = E_epoch / E_Planck

    At Planck energy, theta = 1 (fully quantum gravitational)
    At low energies, theta -> 0 (classical)
    """
    theta = epoch.energy / E_PLANCK_EV
    return np.clip(theta, 0.0, 1.0)


def compute_thermal_theta(temperature: float) -> float:
    """
    Compute theta from cosmic temperature.

    theta = T / T_Planck
    """
    theta = temperature / T_PLANCK
    return np.clip(theta, 0.0, 1.0)


def compute_hubble_theta(hubble_time: float) -> float:
    """
    Compute theta from Hubble time (inverse Hubble parameter).

    theta = t_Planck / t_Hubble

    When Hubble time equals Planck time, theta = 1.
    """
    theta = T_PLANCK_TIME / hubble_time
    return np.clip(theta, 0.0, 1.0)


def compute_curvature_theta(curvature_radius: float) -> float:
    """
    Compute theta from spacetime curvature radius.

    theta = l_Planck / R_curvature

    When curvature radius equals Planck length, theta = 1.
    """
    L_PLANCK = 1.616255e-35  # m
    theta = L_PLANCK / curvature_radius
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# COSMIC TIMELINE
# =============================================================================

COSMIC_TIMELINE: Dict[str, CosmicEpoch] = {
    "planck_era": CosmicEpoch(
        name="Planck Era",
        era=CosmicEra.PLANCK,
        time=5.39e-44,
        temperature=1.42e32,
        energy=1.22e28,
        description="Quantum gravity dominates. All forces unified. Spacetime foam.",
        key_events=["Quantum fluctuations create spacetime", "All forces unified"]
    ),
    "gut_era": CosmicEpoch(
        name="Grand Unification Era",
        era=CosmicEra.GUT,
        time=1e-36,
        temperature=1e28,
        energy=1e25,
        description="Strong force separates. GUT phase transition.",
        key_events=["Strong force decouples", "Possible magnetic monopole production"]
    ),
    "inflation_end": CosmicEpoch(
        name="End of Inflation",
        era=CosmicEra.GUT,
        time=1e-32,
        temperature=1e27,
        energy=1e24,
        description="Cosmic inflation ends. Universe reheats.",
        key_events=["Exponential expansion ends", "Reheating creates particles"]
    ),
    "electroweak_era": CosmicEpoch(
        name="Electroweak Era",
        era=CosmicEra.ELECTROWEAK,
        time=1e-12,
        temperature=1e15,
        energy=100e9,
        description="Electromagnetic and weak forces unified.",
        key_events=["W and Z bosons massless", "Higgs field symmetric"]
    ),
    "electroweak_transition": CosmicEpoch(
        name="Electroweak Phase Transition",
        era=CosmicEra.ELECTROWEAK,
        time=1e-11,
        temperature=1.5e12,
        energy=100e9,
        description="Higgs mechanism activates. Particles acquire mass.",
        key_events=["Higgs field breaks symmetry", "W, Z bosons become massive"]
    ),
    "quark_epoch": CosmicEpoch(
        name="Quark Epoch",
        era=CosmicEra.QUARK,
        time=1e-6,
        temperature=1e12,
        energy=100e6,
        description="Quark-gluon plasma. Quarks move freely.",
        key_events=["Quark-gluon plasma", "Matter-antimatter asymmetry established"]
    ),
    "qcd_transition": CosmicEpoch(
        name="QCD Phase Transition",
        era=CosmicEra.HADRON,
        time=1e-5,
        temperature=2e12,
        energy=170e6,
        description="Quarks confined into hadrons (protons, neutrons).",
        key_events=["Quark confinement", "Hadrons form"]
    ),
    "hadron_epoch": CosmicEpoch(
        name="Hadron Epoch",
        era=CosmicEra.HADRON,
        time=1.0,
        temperature=1e10,
        energy=1e6,
        description="Hadrons dominate. Most antimatter annihilates.",
        key_events=["Hadron-antihadron annihilation", "Neutrino decoupling"]
    ),
    "nucleosynthesis": CosmicEpoch(
        name="Big Bang Nucleosynthesis",
        era=CosmicEra.NUCLEOSYNTHESIS,
        time=180.0,  # ~3 minutes
        temperature=1e9,
        energy=100e3,
        description="Light nuclei form: H, He, Li.",
        key_events=["Deuterium forms", "Helium-4 synthesis", "Li-7 production"]
    ),
    "nucleosynthesis_end": CosmicEpoch(
        name="End of Nucleosynthesis",
        era=CosmicEra.NUCLEOSYNTHESIS,
        time=1200.0,  # ~20 minutes
        temperature=3e8,
        energy=30e3,
        description="Nuclear reactions freeze out.",
        key_events=["~75% H, ~25% He by mass", "Trace Li, Be"]
    ),
    "matter_radiation_equality": CosmicEpoch(
        name="Matter-Radiation Equality",
        era=CosmicEra.RADIATION,
        time=1.5e12,  # ~47,000 years
        temperature=9000,
        energy=0.75,
        description="Matter density equals radiation density.",
        key_events=["Transition to matter-dominated era"]
    ),
    "recombination": CosmicEpoch(
        name="Recombination",
        era=CosmicEra.MATTER,
        time=1.2e13,  # ~380,000 years
        temperature=3000,
        energy=0.26,
        description="Electrons combine with nuclei. Universe becomes transparent.",
        key_events=["Neutral atoms form", "CMB released", "Universe becomes transparent"]
    ),
    "dark_ages": CosmicEpoch(
        name="Dark Ages",
        era=CosmicEra.MATTER,
        time=3e14,  # ~10 million years
        temperature=60,
        energy=5e-3,
        description="No stars yet. Universe cools in darkness.",
        key_events=["Density fluctuations grow", "Dark matter halos form"]
    ),
    "first_stars": CosmicEpoch(
        name="First Stars (Cosmic Dawn)",
        era=CosmicEra.MATTER,
        time=6e15,  # ~200 million years
        temperature=30,
        energy=2.5e-3,
        description="First Population III stars ignite.",
        key_events=["First star formation", "Reionization begins"]
    ),
    "reionization": CosmicEpoch(
        name="Reionization Complete",
        era=CosmicEra.MATTER,
        time=3e16,  # ~1 billion years
        temperature=10,
        energy=8e-4,
        description="UV from stars reionizes intergalactic medium.",
        key_events=["Universe reionized", "First galaxies form"]
    ),
    "present_day": CosmicEpoch(
        name="Present Day",
        era=CosmicEra.DARK_ENERGY,
        time=4.35e17,  # 13.8 billion years
        temperature=2.725,
        energy=2.35e-4,
        description="Dark energy dominated. Cosmic acceleration.",
        key_events=["Dark energy dominates", "Cosmic acceleration", "Life exists"]
    ),
    "solar_death": CosmicEpoch(
        name="Sun Becomes Red Giant",
        era=CosmicEra.DARK_ENERGY,
        time=1.6e17 + 4.35e17,  # 5 Gyr from now + current age
        temperature=2.0,
        energy=1.7e-4,
        description="Sun expands. Earth uninhabitable.",
        key_events=["Sun exhausts hydrogen", "Inner planets engulfed"]
    ),
    "stellar_era_end": CosmicEpoch(
        name="Last Stars Die",
        era=CosmicEra.FAR_FUTURE,
        time=1e21,  # 100 trillion years
        temperature=0.01,
        energy=1e-6,
        description="Star formation ends. Last red dwarfs fade.",
        key_events=["No new stars form", "Universe goes dark"]
    ),
    "black_hole_era": CosmicEpoch(
        name="Black Hole Era",
        era=CosmicEra.FAR_FUTURE,
        time=1e40,
        temperature=1e-20,
        energy=1e-24,
        description="Only black holes remain as organized structures.",
        key_events=["Proton decay (if it occurs)", "Black holes dominate"]
    ),
    "heat_death": CosmicEpoch(
        name="Heat Death",
        era=CosmicEra.FAR_FUTURE,
        time=1e100,
        temperature=1e-30,
        energy=1e-34,
        description="Maximum entropy. No thermodynamic free energy.",
        key_events=["Last black holes evaporate", "Thermal equilibrium"]
    ),
}


# =============================================================================
# PHASE TRANSITIONS
# =============================================================================

@dataclass
class PhaseTransition:
    """A cosmological phase transition."""
    name: str
    temperature: float      # K
    energy: float           # eV
    order: int              # 1st or 2nd order
    symmetry_broken: str    # Which symmetry breaks
    theta_before: float     # Theta just before transition
    theta_after: float      # Theta just after transition


PHASE_TRANSITIONS: Dict[str, PhaseTransition] = {
    "gut_transition": PhaseTransition(
        name="GUT Phase Transition",
        temperature=1e28,
        energy=1e25,
        order=1,
        symmetry_broken="SU(5) -> SU(3) x SU(2) x U(1)",
        theta_before=0.82,
        theta_after=0.82,
    ),
    "electroweak_transition": PhaseTransition(
        name="Electroweak Phase Transition",
        temperature=1.5e12,
        energy=100e9,
        order=2,  # Crossover in Standard Model
        symmetry_broken="SU(2) x U(1) -> U(1)_EM",
        theta_before=8.2e-19,
        theta_after=8.2e-19,
    ),
    "qcd_transition": PhaseTransition(
        name="QCD Phase Transition",
        temperature=1.7e12,
        energy=150e6,
        order=2,  # Crossover
        symmetry_broken="Chiral symmetry breaking",
        theta_before=1.2e-22,
        theta_after=1.2e-22,
    ),
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def theta_evolution() -> List[tuple]:
    """
    Compute theta as a function of cosmic time.

    Returns list of (time, theta, epoch_name) tuples.
    """
    evolution = []
    for name, epoch in COSMIC_TIMELINE.items():
        theta = compute_cosmic_theta(epoch)
        evolution.append((epoch.time, theta, epoch.name))
    return sorted(evolution, key=lambda x: x[0])


def orders_of_magnitude_summary() -> Dict[str, float]:
    """
    Summarize the range of theta across cosmic history.
    """
    thetas = [compute_cosmic_theta(epoch) for epoch in COSMIC_TIMELINE.values()]
    times = [epoch.time for epoch in COSMIC_TIMELINE.values()]

    return {
        "max_theta": max(thetas),
        "min_theta": min(thetas),
        "log_range": np.log10(max(thetas)) - np.log10(max(min(thetas), 1e-100)),
        "time_range_orders": np.log10(max(times)) - np.log10(min(times)),
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def cosmic_theta_summary():
    """Print theta analysis across cosmic history."""
    print("=" * 90)
    print("COSMIC THETA EVOLUTION: From Big Bang to Heat Death")
    print("=" * 90)
    print()
    print(f"{'Epoch':<30} {'Time':>15} {'Temp (K)':>12} {'Energy (eV)':>14} {'θ':>12}")
    print("-" * 90)

    for name, epoch in COSMIC_TIMELINE.items():
        theta = compute_cosmic_theta(epoch)

        # Format time
        if epoch.time < 1e-30:
            time_str = f"{epoch.time:.2e} s"
        elif epoch.time < 60:
            time_str = f"{epoch.time:.2g} s"
        elif epoch.time < 3600:
            time_str = f"{epoch.time/60:.1f} min"
        elif epoch.time < 86400:
            time_str = f"{epoch.time/3600:.1f} hr"
        elif epoch.time < 3.15e7:
            time_str = f"{epoch.time/86400:.1f} days"
        elif epoch.time < 3.15e16:
            time_str = f"{epoch.time/3.15e7:.2g} yr"
        else:
            time_str = f"{epoch.time/3.15e7:.2e} yr"

        # Format temperature
        if epoch.temperature >= 1e6:
            temp_str = f"{epoch.temperature:.2e}"
        else:
            temp_str = f"{epoch.temperature:.2g}"

        # Format energy
        if epoch.energy >= 1e6:
            energy_str = f"{epoch.energy:.2e}"
        else:
            energy_str = f"{epoch.energy:.2g}"

        # Format theta
        if theta >= 0.01:
            theta_str = f"{theta:.4f}"
        else:
            theta_str = f"{theta:.2e}"

        print(f"{epoch.name:<30} {time_str:>15} {temp_str:>12} {energy_str:>14} {theta_str:>12}")

    print()
    print("Key: θ → 1 at Planck scale (quantum gravity), θ → 0 today (classical)")
    print("     Universe spans ~32 orders of magnitude in theta!")


if __name__ == "__main__":
    cosmic_theta_summary()
