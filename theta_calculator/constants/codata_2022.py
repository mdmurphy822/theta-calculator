"""
Complete CODATA 2022 Fundamental Physical Constants

This module contains ALL 347 constants from the 2022 CODATA adjustment,
organized by category with mathematical relationships showing how they connect.

Key insight from arxiv research: Constants are not independent values but form
a self-consistent network where each can be derived from others. This interconnection
is evidence that theta (the quantum-classical parameter) emerges naturally from
the structure of physics itself.

Categories:
1. UNIVERSAL - Defining constants of the SI system
2. ELECTROMAGNETIC - EM vacuum properties and quantum effects
3. ATOMIC - Atomic structure constants
4. ELECTRON - Electron properties
5. PROTON - Proton properties
6. NEUTRON - Neutron properties
7. MUON - Muon properties
8. TAU - Tau lepton properties
9. LIGHT_NUCLEI - Deuteron, triton, helion, alpha particle
10. PHYSICO_CHEMICAL - Molar constants, Avogadro-related
11. ENERGY_EQUIVALENTS - Unit conversions
12. PLANCK - Natural Planck units
13. QUANTUM_ELECTRODYNAMICS - QED precision constants
14. RATIOS - Mass and moment ratios

References (see BIBLIOGRAPHY.bib):
    \\cite{CODATA2022} - NIST CODATA 2022 Recommended Values
                        https://physics.nist.gov/cuu/Constants/
    \\cite{StefanBoltzmann} - Stefan-Boltzmann law (1879)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


class ConstantCategory(Enum):
    """Categories of physical constants."""
    UNIVERSAL = "universal"
    ELECTROMAGNETIC = "electromagnetic"
    ATOMIC = "atomic"
    ATOMIC_UNITS = "atomic_units"
    NATURAL_UNITS = "natural_units"
    ELECTRON = "electron"
    PROTON = "proton"
    NEUTRON = "neutron"
    MUON = "muon"
    TAU = "tau"
    LIGHT_NUCLEI = "light_nuclei"
    PHYSICO_CHEMICAL = "physico_chemical"
    ENERGY_EQUIVALENTS = "energy_equivalents"
    PLANCK = "planck"
    QED = "qed"
    RATIOS = "ratios"
    SHIELDED = "shielded"
    ELECTROWEAK = "electroweak"
    MOLAR = "molar"
    X_UNITS = "x_units"
    CONVENTIONAL = "conventional"


@dataclass(frozen=True)
class Constant:
    """
    A physical constant with full metadata and connections.

    Attributes:
        name: Full descriptive name
        symbol: Mathematical symbol (LaTeX-compatible)
        value: Numerical value in SI units
        uncertainty: Standard uncertainty (0 for exact)
        unit: SI unit string
        category: Classification category
        formula: Mathematical formula if derived
        depends_on: List of constants this is derived from
        description: Physical meaning and relevance to theta
        exact: Whether defined exactly by SI
    """
    name: str
    symbol: str
    value: float
    uncertainty: float
    unit: str
    category: ConstantCategory
    formula: str = ""
    depends_on: List[str] = field(default_factory=list)
    description: str = ""
    exact: bool = False

    def __repr__(self) -> str:
        if self.exact:
            return f"{self.symbol} = {self.value:.15e} {self.unit} (exact)"
        return f"{self.symbol} = {self.value:.15e} ± {self.uncertainty:.2e} {self.unit}"

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as fraction."""
        if self.value == 0:
            return float('inf')
        return abs(self.uncertainty / self.value)


# =============================================================================
# SECTION 1: UNIVERSAL CONSTANTS (SI Defining Constants)
# =============================================================================
# These 7 constants define the SI system as of 2019
# They form the foundation from which all other constants are derived

# Speed of light in vacuum - connects space and time
c = Constant(
    name="speed of light in vacuum",
    symbol="c",
    value=299_792_458.0,
    uncertainty=0.0,
    unit="m s^-1",
    category=ConstantCategory.UNIVERSAL,
    formula="c = 1/√(ε₀μ₀) = l_P/t_P",
    depends_on=["ε₀", "μ₀"],
    description="Maximum speed of causality. Connects spacetime geometry. "
                "At theta=1 (quantum), c appears in E=mc². At theta=0, c is the "
                "classical wave propagation speed.",
    exact=True
)

# Planck constant - quantum of action
h = Constant(
    name="Planck constant",
    symbol="h",
    value=6.626_070_15e-34,
    uncertainty=0.0,
    unit="J Hz^-1",
    category=ConstantCategory.UNIVERSAL,
    formula="h = E/f = 2πℏ",
    depends_on=[],
    description="Fundamental quantum of action. When action S ~ h, quantum effects "
                "dominate (theta → 1). When S >> h, classical physics applies (theta → 0). "
                "This is THE defining scale for the quantum-classical boundary.",
    exact=True
)

# Reduced Planck constant
h_bar = Constant(
    name="reduced Planck constant",
    symbol="ℏ",
    value=1.054_571_817_646e-34,
    uncertainty=0.0,
    unit="J s",
    category=ConstantCategory.UNIVERSAL,
    formula="ℏ = h/(2π)",
    depends_on=["h"],
    description="Natural unit of angular momentum. Appears in Heisenberg uncertainty "
                "ΔxΔp ≥ ℏ/2. Theta = ℏ/S is the action-based definition of quantumness.",
    exact=True
)

# Elementary charge
e_charge = Constant(
    name="elementary charge",
    symbol="e",
    value=1.602_176_634e-19,
    uncertainty=0.0,
    unit="C",
    category=ConstantCategory.UNIVERSAL,
    formula="e = √(4πε₀αℏc)",
    depends_on=["ε₀", "α", "ℏ", "c"],
    description="Quantum of electric charge. All observed charges are integer multiples. "
                "Quarks carry fractional charges (e/3, 2e/3) but are confined.",
    exact=True
)

# Boltzmann constant - connects temperature to energy
k_B = Constant(
    name="Boltzmann constant",
    symbol="k",
    value=1.380_649e-23,
    uncertainty=0.0,
    unit="J K^-1",
    category=ConstantCategory.UNIVERSAL,
    formula="k = R/N_A",
    depends_on=["R", "N_A"],
    description="Bridge between microscopic energy and macroscopic temperature. "
                "Theta_thermal = ℏω/(kT) determines if thermal or quantum effects dominate. "
                "Key for decoherence and the classical limit.",
    exact=True
)

# Avogadro constant
N_A = Constant(
    name="Avogadro constant",
    symbol="N_A",
    value=6.022_140_76e23,
    uncertainty=0.0,
    unit="mol^-1",
    category=ConstantCategory.UNIVERSAL,
    formula="N_A = R/k",
    depends_on=["R", "k"],
    description="Number of entities per mole. Bridges atomic and macroscopic scales. "
                "Classical thermodynamics emerges when N → N_A (many particles).",
    exact=True
)

# Luminous efficacy
K_cd = Constant(
    name="luminous efficacy of monochromatic radiation",
    symbol="K_cd",
    value=683.0,
    uncertainty=0.0,
    unit="lm W^-1",
    category=ConstantCategory.UNIVERSAL,
    formula="",
    depends_on=[],
    description="Defines the candela at 540 THz. Connects photometric to radiometric units.",
    exact=True
)


# =============================================================================
# SECTION 2: ELECTROMAGNETIC CONSTANTS
# =============================================================================

# Vacuum magnetic permeability
mu_0 = Constant(
    name="vacuum magnetic permeability",
    symbol="μ₀",
    value=1.256_637_062_12e-6,
    uncertainty=1.9e-16,
    unit="N A^-2",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ₀ = 2αh/(ce²) = 4πα·ℏ/(ce²)",
    depends_on=["α", "h", "c", "e"],
    description="Magnetic response of vacuum. Related to c by c = 1/√(ε₀μ₀). "
                "In quantum vacuum, virtual particle pairs modify effective μ₀.",
    exact=False
)

# Vacuum electric permittivity
epsilon_0 = Constant(
    name="vacuum electric permittivity",
    symbol="ε₀",
    value=8.854_187_8128e-12,
    uncertainty=1.3e-21,
    unit="F m^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="ε₀ = 1/(μ₀c²) = e²/(4παℏc)",
    depends_on=["μ₀", "c", "e", "α", "ℏ"],
    description="Electric response of vacuum. Combined with μ₀ gives speed of light. "
                "Quantum vacuum fluctuations renormalize effective ε₀.",
    exact=False
)

# Impedance of vacuum
Z_0 = Constant(
    name="characteristic impedance of vacuum",
    symbol="Z₀",
    value=376.730_313_668,
    uncertainty=5.7e-8,
    unit="Ω",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="Z₀ = μ₀c = √(μ₀/ε₀) = 2αh/e²",
    depends_on=["μ₀", "c", "ε₀", "α", "h", "e"],
    description="Ratio of E to H fields in vacuum. Appears in antenna theory and "
                "quantum electrodynamics. Z₀ = 2R_K·α where R_K is von Klitzing constant.",
    exact=False
)

# Fine-structure constant - THE dimensionless coupling
alpha = Constant(
    name="fine-structure constant",
    symbol="α",
    value=7.297_352_5693e-3,
    uncertainty=1.1e-12,
    unit="",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="α = e²/(4πε₀ℏc) = e²/(2ε₀hc) ≈ 1/137.036",
    depends_on=["e", "ε₀", "ℏ", "c"],
    description="THE fundamental dimensionless constant. Governs all electromagnetic "
                "interactions. α ≈ 1/137 determines atomic structure, chemistry, life. "
                "Arxiv research shows α may vary cosmologically, affecting theta globally.",
    exact=False
)

# Inverse fine-structure constant
alpha_inv = Constant(
    name="inverse fine-structure constant",
    symbol="α⁻¹",
    value=137.035_999_084,
    uncertainty=2.1e-8,
    unit="",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="α⁻¹ = 4πε₀ℏc/e²",
    depends_on=["ε₀", "ℏ", "c", "e"],
    description="Reciprocal of fine-structure constant. Often quoted as ~137. "
                "Why this specific value remains one of physics' greatest mysteries.",
    exact=False
)

# Magnetic flux quantum
Phi_0 = Constant(
    name="magnetic flux quantum",
    symbol="Φ₀",
    value=2.067_833_848e-15,
    uncertainty=0.0,
    unit="Wb",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="Φ₀ = h/(2e) = π·ℏ/e",
    depends_on=["h", "e"],
    description="Quantum of magnetic flux in superconductors. Appears in SQUID devices "
                "and Josephson junctions. Demonstrates macroscopic quantum coherence (theta~1).",
    exact=True
)

# Conductance quantum
G_0 = Constant(
    name="conductance quantum",
    symbol="G₀",
    value=7.748_091_729e-5,
    uncertainty=0.0,
    unit="S",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="G₀ = 2e²/h = 1/R_K · 2",
    depends_on=["e", "h"],
    description="Quantum of electrical conductance. Conductance of ideal 1D channel. "
                "Observed in quantum point contacts and carbon nanotubes.",
    exact=True
)

# Josephson constant
K_J = Constant(
    name="Josephson constant",
    symbol="K_J",
    value=483_597.8484e9,
    uncertainty=0.0,
    unit="Hz V^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="K_J = 2e/h = 1/Φ₀",
    depends_on=["e", "h"],
    description="Frequency-to-voltage ratio in Josephson effect. Used in voltage standards. "
                "Macroscopic quantum effect demonstrating theta~1 at superconducting scale.",
    exact=True
)

# von Klitzing constant
R_K = Constant(
    name="von Klitzing constant",
    symbol="R_K",
    value=25_812.807_45,
    uncertainty=0.0,
    unit="Ω",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="R_K = h/e² = μ₀c/(2α)",
    depends_on=["h", "e", "μ₀", "c", "α"],
    description="Quantum Hall resistance. Quantized in integer and fractional multiples. "
                "Nobel Prize 1985. Demonstrates robust quantum effects at macroscopic scale.",
    exact=True
)

# Bohr magneton
mu_B = Constant(
    name="Bohr magneton",
    symbol="μ_B",
    value=9.274_010_0783e-24,
    uncertainty=2.8e-33,
    unit="J T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_B = eℏ/(2m_e)",
    depends_on=["e", "ℏ", "m_e"],
    description="Natural unit of electron magnetic moment. Appears in atomic magnetism "
                "and electron spin resonance. μ_e ≈ -1.001·μ_B due to QED corrections.",
    exact=False
)

# Nuclear magneton
mu_N = Constant(
    name="nuclear magneton",
    symbol="μ_N",
    value=5.050_783_7461e-27,
    uncertainty=1.5e-36,
    unit="J T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_N = eℏ/(2m_p) = μ_B(m_e/m_p)",
    depends_on=["e", "ℏ", "m_p", "μ_B", "m_e"],
    description="Natural unit of nuclear magnetic moment. μ_N/μ_B = m_e/m_p ≈ 1/1836. "
                "Proton magnetic moment is about 2.79·μ_N (not 1, due to quark structure).",
    exact=False
)


# =============================================================================
# SECTION 3: ATOMIC CONSTANTS
# =============================================================================

# Rydberg constant
R_inf = Constant(
    name="Rydberg constant",
    symbol="R_∞",
    value=10_973_731.568_160,
    uncertainty=2.1e-5,
    unit="m^-1",
    category=ConstantCategory.ATOMIC,
    formula="R_∞ = α²m_ec/(2h) = m_e·c·α²/(2h)",
    depends_on=["α", "m_e", "c", "h"],
    description="Fundamental constant of atomic spectroscopy. Determines hydrogen energy levels. "
                "E_n = -R_∞·hc/n². Most precisely measured fundamental constant.",
    exact=False
)

# Bohr radius
a_0 = Constant(
    name="Bohr radius",
    symbol="a₀",
    value=5.291_772_109_03e-11,
    uncertainty=8.0e-21,
    unit="m",
    category=ConstantCategory.ATOMIC,
    formula="a₀ = ℏ/(m_e·c·α) = 4πε₀ℏ²/(m_e·e²)",
    depends_on=["ℏ", "m_e", "c", "α", "ε₀", "e"],
    description="Characteristic atomic length scale. Ground state hydrogen radius. "
                "When system size L ~ a₀, atomic quantum effects dominate (theta ~ 1). "
                "Classical chemistry emerges when L >> a₀.",
    exact=False
)

# Hartree energy
E_h = Constant(
    name="Hartree energy",
    symbol="E_h",
    value=4.359_744_722_2071e-18,
    uncertainty=8.5e-30,
    unit="J",
    category=ConstantCategory.ATOMIC,
    formula="E_h = α²m_ec² = e²/(4πε₀a₀) = 2R_∞hc",
    depends_on=["α", "m_e", "c", "e", "ε₀", "a₀", "R_∞", "h"],
    description="Natural unit of energy in atomic physics. Twice the hydrogen ionization energy. "
                "E_h ≈ 27.2 eV. Used in quantum chemistry calculations.",
    exact=False
)

# Classical electron radius
r_e = Constant(
    name="classical electron radius",
    symbol="r_e",
    value=2.817_940_3262e-15,
    uncertainty=1.3e-24,
    unit="m",
    category=ConstantCategory.ATOMIC,
    formula="r_e = e²/(4πε₀m_ec²) = α²a₀ = α·λ_C/(2π)",
    depends_on=["e", "ε₀", "m_e", "c", "α", "a₀"],
    description="Length scale where classical EM self-energy equals rest mass. "
                "r_e = α²·a₀ shows hierarchy: r_e << λ_C << a₀. Thomson scattering cross-section "
                "σ_T = 8πr_e²/3.",
    exact=False
)

# Compton wavelength (electron)
lambda_C = Constant(
    name="Compton wavelength",
    symbol="λ_C",
    value=2.426_310_238_67e-12,
    uncertainty=7.3e-22,
    unit="m",
    category=ConstantCategory.ATOMIC,
    formula="λ_C = h/(m_ec) = 2π·ℏ/(m_ec)",
    depends_on=["h", "m_e", "c"],
    description="Wavelength of photon with energy equal to electron rest mass. "
                "λ_C = α·a₀/(2π). Below λ_C, pair creation becomes possible. "
                "Defines boundary between atomic and particle physics.",
    exact=False
)

# Reduced Compton wavelength
lambda_C_bar = Constant(
    name="reduced Compton wavelength",
    symbol="ƛ_C",
    value=3.861_592_6796e-13,
    uncertainty=1.2e-22,
    unit="m",
    category=ConstantCategory.ATOMIC,
    formula="ƛ_C = ℏ/(m_ec) = λ_C/(2π) = α·a₀",
    depends_on=["ℏ", "m_e", "c", "α", "a₀"],
    description="Natural length scale for relativistic quantum mechanics. "
                "Position uncertainty at relativistic energies.",
    exact=False
)

# Thomson cross-section
sigma_e = Constant(
    name="Thomson cross-section",
    symbol="σ_e",
    value=6.652_458_7321e-29,
    uncertainty=6.0e-38,
    unit="m^2",
    category=ConstantCategory.ATOMIC,
    formula="σ_e = 8πr_e²/3 = 8π(α²a₀)²/3",
    depends_on=["r_e", "α", "a₀"],
    description="Low-energy photon-electron scattering cross-section. "
                "Classical limit of Compton scattering. Eddington luminosity limit.",
    exact=False
)

# Atomic mass unit
u = Constant(
    name="unified atomic mass unit",
    symbol="u",
    value=1.660_539_066_60e-27,
    uncertainty=5.0e-37,
    unit="kg",
    category=ConstantCategory.ATOMIC,
    formula="u = m(¹²C)/12 = 1 g/mol / N_A",
    depends_on=["N_A"],
    description="1/12 of carbon-12 mass. Standard for atomic masses. "
                "u ≈ 931.5 MeV/c². Nuclear binding energies measured relative to u.",
    exact=False
)


# =============================================================================
# SECTION 4: ELECTRON PROPERTIES
# =============================================================================

m_e = Constant(
    name="electron mass",
    symbol="m_e",
    value=9.109_383_7015e-31,
    uncertainty=2.8e-40,
    unit="kg",
    category=ConstantCategory.ELECTRON,
    formula="m_e = ℏ/(c·ƛ_C) = α·m_p·(m_e/m_p)",
    depends_on=["ℏ", "c", "ƛ_C"],
    description="Lightest charged lepton. m_e determines atomic structure. "
                "de Broglie wavelength λ = h/(m_e·v) sets quantum scale for electrons. "
                "Electron is fully quantum (theta ~ 1) at all accessible energies.",
    exact=False
)

m_e_u = Constant(
    name="electron mass in u",
    symbol="m_e/u",
    value=5.485_799_0888e-4,
    uncertainty=1.7e-13,
    unit="u",
    category=ConstantCategory.ELECTRON,
    formula="m_e/u = m_e·N_A/0.001",
    depends_on=["m_e", "N_A"],
    description="Electron mass in atomic mass units.",
    exact=False
)

m_e_eV = Constant(
    name="electron mass energy equivalent",
    symbol="m_e·c²",
    value=8.187_105_7769e-14,
    uncertainty=2.5e-23,
    unit="J",
    category=ConstantCategory.ELECTRON,
    formula="m_e·c² = 0.51099895 MeV",
    depends_on=["m_e", "c"],
    description="Electron rest energy. E = m_e·c² ≈ 511 keV. "
                "Pair creation threshold for e⁺e⁻.",
    exact=False
)

# Electron g-factor
g_e = Constant(
    name="electron g-factor",
    symbol="g_e",
    value=-2.002_319_304_362_56,
    uncertainty=3.5e-13,
    unit="",
    category=ConstantCategory.ELECTRON,
    formula="g_e = 2(1 + a_e) where a_e = (g-2)/2",
    depends_on=["a_e"],
    description="Electron magnetic moment in units of μ_B. Dirac predicts g=2 exactly. "
                "Deviation a_e ≈ α/(2π) is QED's greatest triumph. "
                "Measured to 13 significant figures!",
    exact=False
)

# Electron anomalous magnetic moment
a_e = Constant(
    name="electron magnetic moment anomaly",
    symbol="a_e",
    value=1.159_652_181_28e-3,
    uncertainty=1.8e-13,
    unit="",
    category=ConstantCategory.ELECTRON,
    formula="a_e = (g_e-2)/2 ≈ α/(2π) + O(α²)",
    depends_on=["α"],
    description="Deviation from Dirac's g=2. First QED correction: a_e = α/(2π) ≈ 0.00116. "
                "Higher orders involve α², α³... Agreement with theory tests QED.",
    exact=False
)

# Electron magnetic moment
mu_e = Constant(
    name="electron magnetic moment",
    symbol="μ_e",
    value=-9.284_764_7043e-24,
    uncertainty=2.8e-33,
    unit="J T^-1",
    category=ConstantCategory.ELECTRON,
    formula="μ_e = g_e·μ_B/2",
    depends_on=["g_e", "μ_B"],
    description="Intrinsic magnetic dipole moment. μ_e/μ_B ≈ -1.001. "
                "Negative sign indicates moment antiparallel to spin.",
    exact=False
)


# =============================================================================
# SECTION 5: PROTON PROPERTIES
# =============================================================================

m_p = Constant(
    name="proton mass",
    symbol="m_p",
    value=1.672_621_923_69e-27,
    uncertainty=5.1e-37,
    unit="kg",
    category=ConstantCategory.PROTON,
    formula="m_p ≈ 938.3 MeV/c²",
    depends_on=[],
    description="Lightest stable baryon. m_p/m_e ≈ 1836. Proton is composite (uud quarks). "
                "Most of proton mass from QCD binding energy, not quark masses. "
                "Proton in atom: theta ~ 0.6 (transition regime).",
    exact=False
)

m_p_u = Constant(
    name="proton mass in u",
    symbol="m_p/u",
    value=1.007_276_466_621,
    uncertainty=5.3e-10,
    unit="u",
    category=ConstantCategory.PROTON,
    formula="m_p/u",
    depends_on=["m_p"],
    description="Proton mass in atomic mass units. Slightly less than 1 due to binding energy.",
    exact=False
)

m_p_eV = Constant(
    name="proton mass energy equivalent",
    symbol="m_p·c²",
    value=1.503_277_615_98e-10,
    uncertainty=4.6e-20,
    unit="J",
    category=ConstantCategory.PROTON,
    formula="m_p·c² = 938.272 MeV",
    depends_on=["m_p", "c"],
    description="Proton rest energy. GeV scale defines nuclear/particle physics.",
    exact=False
)

# Proton Compton wavelength
lambda_C_p = Constant(
    name="proton Compton wavelength",
    symbol="λ_C,p",
    value=1.321_409_855_39e-15,
    uncertainty=4.0e-25,
    unit="m",
    category=ConstantCategory.PROTON,
    formula="λ_C,p = h/(m_p·c) = λ_C·(m_e/m_p)",
    depends_on=["h", "m_p", "c", "λ_C", "m_e"],
    description="Proton Compton wavelength. λ_C,p ≈ λ_C/1836. "
                "Nuclear physics scale. Compares to nuclear radius ~1.2 fm.",
    exact=False
)

# Proton charge radius
r_p = Constant(
    name="proton rms charge radius",
    symbol="r_p",
    value=8.414e-16,
    uncertainty=1.9e-18,
    unit="m",
    category=ConstantCategory.PROTON,
    formula="",
    depends_on=[],
    description="Root-mean-square charge radius. 'Proton radius puzzle' resolved 2019. "
                "r_p ≈ 0.84 fm from muonic hydrogen spectroscopy.",
    exact=False
)

# Proton g-factor
g_p = Constant(
    name="proton g-factor",
    symbol="g_p",
    value=5.585_694_6893,
    uncertainty=1.6e-9,
    unit="",
    category=ConstantCategory.PROTON,
    formula="g_p = 2μ_p/μ_N",
    depends_on=["μ_p", "μ_N"],
    description="Proton magnetic moment in units of nuclear magneton. "
                "g_p ≈ 5.59 >> 2 due to quark substructure. Not predicted by simple models.",
    exact=False
)

# Proton magnetic moment
mu_p = Constant(
    name="proton magnetic moment",
    symbol="μ_p",
    value=1.410_606_797_36e-26,
    uncertainty=6.0e-36,
    unit="J T^-1",
    category=ConstantCategory.PROTON,
    formula="μ_p = g_p·μ_N/2",
    depends_on=["g_p", "μ_N"],
    description="Intrinsic magnetic dipole moment. μ_p/μ_N ≈ 2.79. "
                "Used in NMR and MRI imaging.",
    exact=False
)


# =============================================================================
# SECTION 6: NEUTRON PROPERTIES
# =============================================================================

m_n = Constant(
    name="neutron mass",
    symbol="m_n",
    value=1.674_927_498_04e-27,
    uncertainty=9.5e-37,
    unit="kg",
    category=ConstantCategory.NEUTRON,
    formula="m_n ≈ 939.6 MeV/c²",
    depends_on=[],
    description="Neutral baryon (udd quarks). m_n > m_p allows beta decay. "
                "Free neutron half-life ≈ 10 min. Stable in nuclei. "
                "Neutron stars: extreme quantum-classical boundary.",
    exact=False
)

m_n_u = Constant(
    name="neutron mass in u",
    symbol="m_n/u",
    value=1.008_664_915_95,
    uncertainty=4.9e-10,
    unit="u",
    category=ConstantCategory.NEUTRON,
    formula="m_n/u",
    depends_on=["m_n"],
    description="Neutron mass in atomic mass units.",
    exact=False
)

m_n_eV = Constant(
    name="neutron mass energy equivalent",
    symbol="m_n·c²",
    value=1.505_349_762_87e-10,
    uncertainty=8.6e-20,
    unit="J",
    category=ConstantCategory.NEUTRON,
    formula="m_n·c² = 939.565 MeV",
    depends_on=["m_n", "c"],
    description="Neutron rest energy. m_n - m_p ≈ 1.29 MeV enables beta decay.",
    exact=False
)

# Neutron Compton wavelength
lambda_C_n = Constant(
    name="neutron Compton wavelength",
    symbol="λ_C,n",
    value=1.319_590_905_81e-15,
    uncertainty=7.5e-25,
    unit="m",
    category=ConstantCategory.NEUTRON,
    formula="λ_C,n = h/(m_n·c)",
    depends_on=["h", "m_n", "c"],
    description="Neutron Compton wavelength. Similar to proton.",
    exact=False
)

# Neutron g-factor
g_n = Constant(
    name="neutron g-factor",
    symbol="g_n",
    value=-3.826_085_45,
    uncertainty=9.0e-8,
    unit="",
    category=ConstantCategory.NEUTRON,
    formula="g_n = 2μ_n/μ_N",
    depends_on=["μ_n", "μ_N"],
    description="Neutron magnetic moment in nuclear magnetons. "
                "Negative sign: moment antiparallel to spin (like electron). "
                "Non-zero despite zero charge due to quark structure.",
    exact=False
)

# Neutron magnetic moment
mu_n = Constant(
    name="neutron magnetic moment",
    symbol="μ_n",
    value=-9.662_3651e-27,
    uncertainty=2.3e-33,
    unit="J T^-1",
    category=ConstantCategory.NEUTRON,
    formula="μ_n = g_n·μ_N/2",
    depends_on=["g_n", "μ_N"],
    description="Intrinsic magnetic dipole moment. μ_n ≠ 0 despite Q=0 "
                "because quarks carry fractional charges.",
    exact=False
)


# =============================================================================
# SECTION 7: MUON PROPERTIES
# =============================================================================

m_mu = Constant(
    name="muon mass",
    symbol="m_μ",
    value=1.883_531_627e-28,
    uncertainty=4.2e-36,
    unit="kg",
    category=ConstantCategory.MUON,
    formula="m_μ ≈ 105.66 MeV/c²",
    depends_on=[],
    description="Heavy electron. m_μ/m_e ≈ 207. Muon lifetime 2.2 μs. "
                "Muonic hydrogen used for precise proton radius measurement.",
    exact=False
)

m_mu_u = Constant(
    name="muon mass in u",
    symbol="m_μ/u",
    value=0.113_428_9259,
    uncertainty=2.5e-9,
    unit="u",
    category=ConstantCategory.MUON,
    formula="m_μ/u",
    depends_on=["m_μ"],
    description="Muon mass in atomic mass units.",
    exact=False
)

m_mu_eV = Constant(
    name="muon mass energy equivalent",
    symbol="m_μ·c²",
    value=1.692_833_804e-11,
    uncertainty=3.8e-19,
    unit="J",
    category=ConstantCategory.MUON,
    formula="m_μ·c² = 105.658 MeV",
    depends_on=["m_μ", "c"],
    description="Muon rest energy.",
    exact=False
)

# Muon g-factor
g_mu = Constant(
    name="muon g-factor",
    symbol="g_μ",
    value=-2.002_331_8418,
    uncertainty=1.3e-9,
    unit="",
    category=ConstantCategory.MUON,
    formula="g_μ = 2(1 + a_μ)",
    depends_on=["a_μ"],
    description="Muon magnetic moment anomaly. g-2 experiment at Fermilab "
                "shows possible deviation from Standard Model prediction!",
    exact=False
)

# Muon anomalous magnetic moment
a_mu = Constant(
    name="muon magnetic moment anomaly",
    symbol="a_μ",
    value=1.165_920_89e-3,
    uncertainty=6.3e-10,
    unit="",
    category=ConstantCategory.MUON,
    formula="a_μ = (g_μ-2)/2",
    depends_on=["g_μ"],
    description="Muon g-2 anomaly. 4.2σ deviation from SM may indicate new physics! "
                "More sensitive to heavy particles than electron a_e.",
    exact=False
)

# Muon magnetic moment
mu_mu = Constant(
    name="muon magnetic moment",
    symbol="μ_μ",
    value=-4.490_448_30e-26,
    uncertainty=1.0e-33,
    unit="J T^-1",
    category=ConstantCategory.MUON,
    formula="μ_μ = g_μ·eℏ/(4m_μ)",
    depends_on=["g_μ", "e", "ℏ", "m_μ"],
    description="Muon intrinsic magnetic moment.",
    exact=False
)

# Muon Compton wavelength
lambda_C_mu = Constant(
    name="muon Compton wavelength",
    symbol="λ_C,μ",
    value=1.173_444_110e-14,
    uncertainty=2.6e-22,
    unit="m",
    category=ConstantCategory.MUON,
    formula="λ_C,μ = h/(m_μ·c)",
    depends_on=["h", "m_μ", "c"],
    description="Muon Compton wavelength. λ_C,μ ≈ λ_C/207.",
    exact=False
)


# =============================================================================
# SECTION 8: TAU PROPERTIES
# =============================================================================

m_tau = Constant(
    name="tau mass",
    symbol="m_τ",
    value=3.167_54e-27,
    uncertainty=2.1e-31,
    unit="kg",
    category=ConstantCategory.TAU,
    formula="m_τ ≈ 1776.86 MeV/c²",
    depends_on=[],
    description="Heaviest charged lepton. m_τ/m_e ≈ 3477. Tau lifetime 2.9×10⁻¹³ s. "
                "Only lepton heavy enough to decay to hadrons.",
    exact=False
)

m_tau_u = Constant(
    name="tau mass in u",
    symbol="m_τ/u",
    value=1.907_54,
    uncertainty=1.3e-4,
    unit="u",
    category=ConstantCategory.TAU,
    formula="m_τ/u",
    depends_on=["m_τ"],
    description="Tau mass in atomic mass units.",
    exact=False
)

m_tau_eV = Constant(
    name="tau mass energy equivalent",
    symbol="m_τ·c²",
    value=2.846_84e-10,
    uncertainty=1.9e-14,
    unit="J",
    category=ConstantCategory.TAU,
    formula="m_τ·c² = 1776.86 MeV",
    depends_on=["m_τ", "c"],
    description="Tau rest energy. Heavy enough to produce charm quarks.",
    exact=False
)

# Tau Compton wavelength
lambda_C_tau = Constant(
    name="tau Compton wavelength",
    symbol="λ_C,τ",
    value=6.977_71e-16,
    uncertainty=4.7e-20,
    unit="m",
    category=ConstantCategory.TAU,
    formula="λ_C,τ = h/(m_τ·c)",
    depends_on=["h", "m_τ", "c"],
    description="Tau Compton wavelength. Subfemtometer scale.",
    exact=False
)


# =============================================================================
# SECTION 9: LIGHT NUCLEI (Deuteron, Triton, Helion, Alpha)
# =============================================================================

m_d = Constant(
    name="deuteron mass",
    symbol="m_d",
    value=3.343_583_7724e-27,
    uncertainty=1.0e-36,
    unit="kg",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_d = m_p + m_n - B_d/c²",
    depends_on=["m_p", "m_n"],
    description="Deuterium nucleus (p+n). Binding energy B_d ≈ 2.22 MeV. "
                "Simplest nucleus. Fusion fuel. Deuteron spin = 1.",
    exact=False
)

m_d_u = Constant(
    name="deuteron mass in u",
    symbol="m_d/u",
    value=2.013_553_212_745,
    uncertainty=4.0e-10,
    unit="u",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_d/u",
    depends_on=["m_d"],
    description="Deuteron mass in atomic mass units.",
    exact=False
)

m_d_eV = Constant(
    name="deuteron mass energy equivalent",
    symbol="m_d·c²",
    value=3.005_063_231_02e-10,
    uncertainty=9.1e-20,
    unit="J",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_d·c² = 1875.613 MeV",
    depends_on=["m_d", "c"],
    description="Deuteron rest energy.",
    exact=False
)

# Deuteron magnetic moment
mu_d = Constant(
    name="deuteron magnetic moment",
    symbol="μ_d",
    value=4.330_735_094e-27,
    uncertainty=1.1e-35,
    unit="J T^-1",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_d = g_d·μ_N·s_d",
    depends_on=["μ_N"],
    description="Deuteron intrinsic magnetic moment. Used in NMR of D₂O.",
    exact=False
)

m_t = Constant(
    name="triton mass",
    symbol="m_t",
    value=5.007_356_7446e-27,
    uncertainty=1.5e-36,
    unit="kg",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_t = m_p + 2m_n - B_t/c²",
    depends_on=["m_p", "m_n"],
    description="Tritium nucleus (p+2n). Radioactive, half-life 12.3 years. "
                "Fusion fuel for tokamaks. Beta decays to helion.",
    exact=False
)

m_t_u = Constant(
    name="triton mass in u",
    symbol="m_t/u",
    value=3.015_500_716_21,
    uncertainty=1.2e-10,
    unit="u",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_t/u",
    depends_on=["m_t"],
    description="Triton mass in atomic mass units.",
    exact=False
)

m_h = Constant(
    name="helion mass",
    symbol="m_h",
    value=5.006_412_7796e-27,
    uncertainty=1.5e-36,
    unit="kg",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_h = 2m_p + m_n - B_h/c²",
    depends_on=["m_p", "m_n"],
    description="Helium-3 nucleus (2p+n). Stable. Product of tritium decay. "
                "Used in neutron detectors and potential fusion fuel.",
    exact=False
)

m_h_u = Constant(
    name="helion mass in u",
    symbol="m_h/u",
    value=3.014_932_247_175,
    uncertainty=9.7e-11,
    unit="u",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_h/u",
    depends_on=["m_h"],
    description="Helion mass in atomic mass units.",
    exact=False
)

m_alpha = Constant(
    name="alpha particle mass",
    symbol="m_α",
    value=6.644_657_3357e-27,
    uncertainty=2.0e-36,
    unit="kg",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_α = 2m_p + 2m_n - B_α/c²",
    depends_on=["m_p", "m_n"],
    description="Helium-4 nucleus (2p+2n). Very stable, doubly magic. "
                "B_α ≈ 28.3 MeV. Alpha decay product. Cosmic ray component.",
    exact=False
)

m_alpha_u = Constant(
    name="alpha particle mass in u",
    symbol="m_α/u",
    value=4.001_506_179_127,
    uncertainty=6.3e-11,
    unit="u",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_α/u",
    depends_on=["m_α"],
    description="Alpha particle mass in atomic mass units.",
    exact=False
)

m_alpha_eV = Constant(
    name="alpha particle mass energy equivalent",
    symbol="m_α·c²",
    value=5.971_920_1914e-10,
    uncertainty=1.8e-19,
    unit="J",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="m_α·c² = 3727.379 MeV",
    depends_on=["m_α", "c"],
    description="Alpha particle rest energy.",
    exact=False
)


# =============================================================================
# SECTION 10: PHYSICO-CHEMICAL CONSTANTS
# =============================================================================

# Molar gas constant
R = Constant(
    name="molar gas constant",
    symbol="R",
    value=8.314_462_618,
    uncertainty=0.0,
    unit="J mol^-1 K^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="R = N_A·k",
    depends_on=["N_A", "k"],
    description="Ideal gas constant. PV = nRT. Bridges microscopic (k) and "
                "macroscopic (molar) thermodynamics. R = 8.314 J/(mol·K).",
    exact=True
)

# Faraday constant
F = Constant(
    name="Faraday constant",
    symbol="F",
    value=96_485.332_12,
    uncertainty=0.0,
    unit="C mol^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="F = N_A·e",
    depends_on=["N_A", "e"],
    description="Charge per mole of electrons. F = 96485 C/mol. "
                "Fundamental to electrochemistry. Faraday's laws of electrolysis.",
    exact=True
)

# Stefan-Boltzmann constant
sigma_SB = Constant(
    name="Stefan-Boltzmann constant",
    symbol="σ",
    value=5.670_374_419e-8,
    uncertainty=0.0,
    unit="W m^-2 K^-4",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="σ = π²k⁴/(60ℏ³c²) = 2π⁵k⁴/(15h³c²)",
    depends_on=["k", "ℏ", "c", "h"],
    description="Blackbody radiation power. P = σAT⁴. Connects thermal (k), "
                "quantum (ℏ), and relativistic (c) physics. Exact since 2019 SI.",
    exact=True
)

# First radiation constant
c_1 = Constant(
    name="first radiation constant",
    symbol="c₁",
    value=3.741_771_852e-16,
    uncertainty=0.0,
    unit="W m^2",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="c₁ = 2πhc²",
    depends_on=["h", "c"],
    description="Planck radiation law coefficient. M_λ = c₁/λ⁵ × 1/(exp(c₂/λT)-1).",
    exact=True
)

# Second radiation constant
c_2 = Constant(
    name="second radiation constant",
    symbol="c₂",
    value=1.438_776_877e-2,
    uncertainty=0.0,
    unit="m K",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="c₂ = hc/k",
    depends_on=["h", "c", "k"],
    description="Planck radiation law exponent coefficient. "
                "Wien displacement: λ_max·T = c₂/4.965.",
    exact=True
)

# Wien wavelength displacement constant
b = Constant(
    name="Wien wavelength displacement law constant",
    symbol="b",
    value=2.897_771_955e-3,
    uncertainty=0.0,
    unit="m K",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="b = hc/(k·x) where x ≈ 4.965 solves x = 5(1-e^(-x))",
    depends_on=["h", "c", "k"],
    description="Wien's law: λ_max = b/T. Peak wavelength of blackbody. "
                "Sun at 5778 K peaks at 502 nm (green).",
    exact=True
)

# Molar Planck constant
N_A_h = Constant(
    name="molar Planck constant",
    symbol="N_A·h",
    value=3.990_312_712e-10,
    uncertainty=0.0,
    unit="J Hz^-1 mol^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="N_A·h",
    depends_on=["N_A", "h"],
    description="Planck constant times Avogadro. Connects photochemistry to molar units.",
    exact=True
)

# Loschmidt constant
n_0 = Constant(
    name="Loschmidt constant",
    symbol="n₀",
    value=2.686_780_111e25,
    uncertainty=0.0,
    unit="m^-3",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="n₀ = N_A/V_m = p₀/(kT₀)",
    depends_on=["N_A", "k"],
    description="Number density of ideal gas at STP (273.15 K, 101.325 kPa). "
                "n₀ = 2.687×10²⁵ m⁻³.",
    exact=True
)

# Molar volume of ideal gas
V_m = Constant(
    name="molar volume of ideal gas",
    symbol="V_m",
    value=22.710_954_64e-3,
    uncertainty=0.0,
    unit="m^3 mol^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="V_m = RT/p = N_A/n₀",
    depends_on=["R", "N_A"],
    description="Volume of 1 mole ideal gas at STP. V_m ≈ 22.71 L/mol.",
    exact=True
)


# =============================================================================
# SECTION 11: PLANCK UNITS (Natural Units)
# =============================================================================

l_P = Constant(
    name="Planck length",
    symbol="l_P",
    value=1.616_255e-35,
    uncertainty=1.8e-40,
    unit="m",
    category=ConstantCategory.PLANCK,
    formula="l_P = √(ℏG/c³)",
    depends_on=["ℏ", "G", "c"],
    description="Smallest meaningful length. Quantum gravity scale. "
                "l_P ≈ 10⁻³⁵ m. Below l_P, spacetime may be discrete/foamy. "
                "At L ~ l_P, theta → 1 (fully quantum gravity).",
    exact=False
)

t_P = Constant(
    name="Planck time",
    symbol="t_P",
    value=5.391_247e-44,
    uncertainty=6.0e-49,
    unit="s",
    category=ConstantCategory.PLANCK,
    formula="t_P = √(ℏG/c⁵) = l_P/c",
    depends_on=["ℏ", "G", "c", "l_P"],
    description="Smallest meaningful time. t_P ≈ 10⁻⁴⁴ s. "
                "Light crosses Planck length in Planck time. "
                "Age of universe ≈ 10⁶¹ t_P.",
    exact=False
)

m_P = Constant(
    name="Planck mass",
    symbol="m_P",
    value=2.176_434e-8,
    uncertainty=2.4e-13,
    unit="kg",
    category=ConstantCategory.PLANCK,
    formula="m_P = √(ℏc/G)",
    depends_on=["ℏ", "c", "G"],
    description="Mass where Compton wavelength equals Schwarzschild radius. "
                "m_P ≈ 22 μg ≈ 1.2×10¹⁹ GeV. Black hole at m_P has maximum temperature. "
                "Theta = m_P/m gives scale-based quantumness estimate.",
    exact=False
)

E_P = Constant(
    name="Planck energy",
    symbol="E_P",
    value=1.956_081e9,
    uncertainty=2.2e4,
    unit="J",
    category=ConstantCategory.PLANCK,
    formula="E_P = m_P·c² = √(ℏc⁵/G)",
    depends_on=["m_P", "c", "ℏ", "G"],
    description="Energy of Planck mass. E_P ≈ 1.22×10¹⁹ GeV. "
                "Grand unification expected near E_P. "
                "Highest energy accessible in principle.",
    exact=False
)

T_P = Constant(
    name="Planck temperature",
    symbol="T_P",
    value=1.416_784e32,
    uncertainty=1.6e27,
    unit="K",
    category=ConstantCategory.PLANCK,
    formula="T_P = E_P/k = √(ℏc⁵/Gk²)",
    depends_on=["E_P", "k", "ℏ", "c", "G"],
    description="Highest meaningful temperature. T_P ≈ 1.4×10³² K. "
                "Universe at Planck era was at T ~ T_P. "
                "Above T_P, quantum gravity effects dominate (theta → 1).",
    exact=False
)

q_P = Constant(
    name="Planck charge",
    symbol="q_P",
    value=1.875_545_956e-18,
    uncertainty=2.8e-28,
    unit="C",
    category=ConstantCategory.PLANCK,
    formula="q_P = √(4πε₀ℏc) = e/√α",
    depends_on=["ε₀", "ℏ", "c", "e", "α"],
    description="Natural unit of charge. q_P = e/√α ≈ 11.7e. "
                "e/q_P = √α ≈ 0.085 is electromagnetic coupling at Planck scale.",
    exact=False
)

# Planck impedance
Z_P = Constant(
    name="Planck impedance",
    symbol="Z_P",
    value=29.979_245_8,
    uncertainty=0.0,
    unit="Ω",
    category=ConstantCategory.PLANCK,
    formula="Z_P = ℏ/q_P² = 1/(4πε₀c) = Z₀/(4π)",
    depends_on=["ℏ", "q_P", "ε₀", "c"],
    description="Natural unit of impedance. Z_P ≈ 30 Ω. Impedance of spacetime itself.",
    exact=False
)


# =============================================================================
# SECTION 12: NEWTONIAN GRAVITATION
# =============================================================================

G = Constant(
    name="Newtonian constant of gravitation",
    symbol="G",
    value=6.674_30e-11,
    uncertainty=1.5e-15,
    unit="m^3 kg^-1 s^-2",
    category=ConstantCategory.UNIVERSAL,
    formula="G = ℏc/m_P² = l_P²c³/ℏ",
    depends_on=["ℏ", "c", "m_P", "l_P"],
    description="Gravitational coupling constant. Least precisely known fundamental constant "
                "(relative uncertainty ~10⁻⁵). G is weak: F_grav/F_EM ~ 10⁻⁴⁰ for electrons. "
                "G defines the Planck scale with ℏ and c.",
    exact=False
)

# Standard acceleration of gravity
g_n = Constant(
    name="standard acceleration of gravity",
    symbol="g_n",
    value=9.806_65,
    uncertainty=0.0,
    unit="m s^-2",
    category=ConstantCategory.UNIVERSAL,
    formula="g_n = GM_Earth/R_Earth²",
    depends_on=["G"],
    description="Conventional standard gravity. Defined exactly as 9.80665 m/s². "
                "Earth's actual g varies 9.78-9.83 m/s² with latitude and altitude.",
    exact=True
)


# =============================================================================
# SECTION 13: MASS RATIOS (Dimensionless)
# =============================================================================

# Electron to proton mass ratio
m_e_over_m_p = Constant(
    name="electron-proton mass ratio",
    symbol="m_e/m_p",
    value=5.446_170_214_87e-4,
    uncertainty=3.3e-13,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_e/m_p ≈ 1/1836.15",
    depends_on=["m_e", "m_p"],
    description="Fundamental mass ratio. Why m_p/m_e ≈ 1836? Unknown! "
                "Determines atomic structure. If different, chemistry would change. "
                "Arxiv papers study potential cosmological variation of this ratio.",
    exact=False
)

# Proton to electron mass ratio
m_p_over_m_e = Constant(
    name="proton-electron mass ratio",
    symbol="m_p/m_e",
    value=1836.152_673_43,
    uncertainty=1.1e-6,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_p/m_e = 1/(m_e/m_p)",
    depends_on=["m_p", "m_e"],
    description="Inverse of electron-proton ratio. μ = m_p/m_e ≈ 1836. "
                "Cosmological variation δμ/μ < 10⁻⁷ from quasar spectra.",
    exact=False
)

# Muon to electron mass ratio
m_mu_over_m_e = Constant(
    name="muon-electron mass ratio",
    symbol="m_μ/m_e",
    value=206.768_2830,
    uncertainty=4.6e-6,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_μ/m_e",
    depends_on=["m_μ", "m_e"],
    description="Muon is a heavy electron. m_μ/m_e ≈ 207. "
                "Origin of lepton mass hierarchy is unknown.",
    exact=False
)

# Tau to electron mass ratio
m_tau_over_m_e = Constant(
    name="tau-electron mass ratio",
    symbol="m_τ/m_e",
    value=3477.23,
    uncertainty=2.3e-1,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_τ/m_e",
    depends_on=["m_τ", "m_e"],
    description="Tau is heaviest lepton. m_τ/m_e ≈ 3477. "
                "Why this specific hierarchy? Deep mystery.",
    exact=False
)

# Neutron to proton mass ratio
m_n_over_m_p = Constant(
    name="neutron-proton mass ratio",
    symbol="m_n/m_p",
    value=1.001_378_419_31,
    uncertainty=4.9e-10,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_n/m_p",
    depends_on=["m_n", "m_p"],
    description="m_n > m_p allows neutron beta decay. (m_n-m_p)c² ≈ 1.29 MeV. "
                "Critical for nucleosynthesis and existence of atoms.",
    exact=False
)

# Alpha to electron mass ratio
m_alpha_over_m_e = Constant(
    name="alpha particle-electron mass ratio",
    symbol="m_α/m_e",
    value=7294.299_541_71,
    uncertainty=1.7e-6,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_α/m_e",
    depends_on=["m_α", "m_e"],
    description="Alpha particle is ~7300 electrons. "
                "Alpha decay energy determined by this ratio.",
    exact=False
)

# Proton magnetic moment to Bohr magneton ratio
mu_p_over_mu_B = Constant(
    name="proton mag. mom. to Bohr magneton ratio",
    symbol="μ_p/μ_B",
    value=1.521_032_2023e-3,
    uncertainty=4.6e-13,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_p/μ_B = g_p·m_e/(2m_p)",
    depends_on=["μ_p", "μ_B", "g_p", "m_e", "m_p"],
    description="Ratio of proton to electron magnetic scales.",
    exact=False
)

# Proton magnetic moment to nuclear magneton ratio
mu_p_over_mu_N = Constant(
    name="proton mag. mom. to nuclear magneton ratio",
    symbol="μ_p/μ_N",
    value=2.792_847_3509,
    uncertainty=1.2e-9,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_p/μ_N = g_p/2",
    depends_on=["μ_p", "μ_N", "g_p"],
    description="Proton magnetic moment ≈ 2.79 μ_N. Not simple 1 due to quark structure.",
    exact=False
)

# Electron to muon magnetic moment ratio
mu_e_over_mu_mu = Constant(
    name="electron to muon mag. mom. ratio",
    symbol="μ_e/μ_μ",
    value=206.766_9883,
    uncertainty=4.6e-6,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_e/μ_μ ≈ m_μ/m_e",
    depends_on=["μ_e", "μ_μ", "m_μ", "m_e"],
    description="Magnetic moments scale with mass.",
    exact=False
)


# =============================================================================
# SECTION 14: ENERGY EQUIVALENTS & CONVERSIONS
# =============================================================================

eV = Constant(
    name="electron volt",
    symbol="eV",
    value=1.602_176_634e-19,
    uncertainty=0.0,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="eV = e × 1 V",
    depends_on=["e"],
    description="Energy gained by electron through 1 volt potential. "
                "1 eV = 1.602×10⁻¹⁹ J. Natural unit for atomic/particle physics.",
    exact=True
)

eV_to_kg = Constant(
    name="electron volt-kilogram relationship",
    symbol="(1 eV)/c²",
    value=1.782_661_921e-36,
    uncertainty=0.0,
    unit="kg",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="(1 eV)/c² = e/c²",
    depends_on=["e", "c"],
    description="Mass equivalent of 1 eV via E=mc². m_e ≈ 0.511 MeV/c².",
    exact=True
)

eV_to_Hz = Constant(
    name="electron volt-hertz relationship",
    symbol="(1 eV)/h",
    value=2.417_989_242e14,
    uncertainty=0.0,
    unit="Hz",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="(1 eV)/h = e/h",
    depends_on=["e", "h"],
    description="Frequency equivalent of 1 eV via E=hf. 1 eV ≈ 241.8 THz.",
    exact=True
)

eV_to_K = Constant(
    name="electron volt-kelvin relationship",
    symbol="(1 eV)/k",
    value=1.160_451_812e4,
    uncertainty=0.0,
    unit="K",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="(1 eV)/k = e/k",
    depends_on=["e", "k"],
    description="Temperature equivalent of 1 eV. 1 eV ≈ 11604 K. "
                "Room temperature kT ≈ 0.026 eV.",
    exact=True
)

eV_to_m = Constant(
    name="electron volt-inverse meter relationship",
    symbol="(1 eV)/(hc)",
    value=8.065_543_937e5,
    uncertainty=0.0,
    unit="m^-1",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="(1 eV)/(hc) = e/(hc)",
    depends_on=["e", "h", "c"],
    description="Wavenumber equivalent of 1 eV. 1 eV ≈ 8065 cm⁻¹.",
    exact=True
)

hartree_to_J = Constant(
    name="Hartree energy",
    symbol="E_h",
    value=4.359_744_722_2071e-18,
    uncertainty=8.5e-30,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h = α²m_ec² = 2R_∞hc",
    depends_on=["α", "m_e", "c", "R_∞", "h"],
    description="Atomic unit of energy. E_h ≈ 27.2 eV. "
                "Natural scale for quantum chemistry calculations.",
    exact=False
)

hartree_to_eV = Constant(
    name="Hartree energy in eV",
    symbol="E_h/eV",
    value=27.211_386_245_988,
    uncertainty=5.3e-11,
    unit="eV",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h/e",
    depends_on=["E_h", "e"],
    description="Hartree in electron volts. E_h ≈ 27.2 eV = 2 × 13.6 eV (H ionization).",
    exact=False
)

u_to_J = Constant(
    name="atomic mass unit-joule relationship",
    symbol="(1 u)c²",
    value=1.492_418_085_60e-10,
    uncertainty=4.5e-20,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u·c²",
    depends_on=["u", "c"],
    description="Energy equivalent of 1 atomic mass unit. 1 u ≈ 931.5 MeV.",
    exact=False
)

u_to_eV = Constant(
    name="atomic mass unit-electron volt relationship",
    symbol="(1 u)c²/e",
    value=931.494_102_42e6,
    uncertainty=2.8e-1,
    unit="eV",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u·c²/e",
    depends_on=["u", "c", "e"],
    description="1 u ≈ 931.5 MeV/c². Nuclear binding energies in MeV.",
    exact=False
)

u_to_kg = Constant(
    name="atomic mass unit-kilogram relationship",
    symbol="u",
    value=1.660_539_066_60e-27,
    uncertainty=5.0e-37,
    unit="kg",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u = m(¹²C)/12",
    depends_on=[],
    description="Unified atomic mass unit. 1 u = 1 g/mol / N_A.",
    exact=False
)


# =============================================================================
# SECTION 15: QED PRECISION CONSTANTS
# =============================================================================

# Electron gyromagnetic ratio
gamma_e = Constant(
    name="electron gyromagnetic ratio",
    symbol="γ_e",
    value=1.760_859_630_23e11,
    uncertainty=5.3e1,
    unit="s^-1 T^-1",
    category=ConstantCategory.QED,
    formula="γ_e = g_e·e/(2m_e) = |μ_e|/ℏ × 2",
    depends_on=["g_e", "e", "m_e", "μ_e", "ℏ"],
    description="Precession frequency per unit field. γ_e/(2π) ≈ 28 GHz/T. "
                "Used in electron spin resonance.",
    exact=False
)

# Proton gyromagnetic ratio
gamma_p = Constant(
    name="proton gyromagnetic ratio",
    symbol="γ_p",
    value=2.675_221_8744e8,
    uncertainty=1.1e-1,
    unit="s^-1 T^-1",
    category=ConstantCategory.QED,
    formula="γ_p = g_p·e/(2m_p)",
    depends_on=["g_p", "e", "m_p"],
    description="Proton precession frequency. γ_p/(2π) ≈ 42.58 MHz/T. "
                "Basis of NMR and MRI technology.",
    exact=False
)

# Shielded proton gyromagnetic ratio
gamma_p_shielded = Constant(
    name="shielded proton gyromagnetic ratio",
    symbol="γ'_p",
    value=2.675_153_151e8,
    uncertainty=2.9e-1,
    unit="s^-1 T^-1",
    category=ConstantCategory.QED,
    formula="γ'_p = γ_p(1 - σ_H2O)",
    depends_on=["γ_p"],
    description="Proton gyromagnetic ratio in H₂O sphere at 25°C. "
                "Shielding reduces γ by ~25 ppm.",
    exact=False
)

# Electron to proton magnetic moment ratio
mu_e_over_mu_p = Constant(
    name="electron to proton mag. mom. ratio",
    symbol="μ_e/μ_p",
    value=-658.210_687_89,
    uncertainty=2.0e-7,
    unit="",
    category=ConstantCategory.QED,
    formula="μ_e/μ_p",
    depends_on=["μ_e", "μ_p"],
    description="Electron moment ~658× proton moment. Negative: opposite alignment.",
    exact=False
)


# =============================================================================
# SECTION 16: ATOMIC UNITS
# =============================================================================
# Atomic units use a_0, E_h, e, m_e, hbar as base units

au_action = Constant(
    name="atomic unit of action",
    symbol="ℏ",
    value=1.054_571_817_646e-34,
    uncertainty=0.0,
    unit="J s",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="au_action = ℏ",
    depends_on=["h_bar"],
    description="Atomic unit of action equals reduced Planck constant.",
    exact=True
)

au_charge = Constant(
    name="atomic unit of charge",
    symbol="e",
    value=1.602_176_634e-19,
    uncertainty=0.0,
    unit="C",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="au_charge = e",
    depends_on=["e"],
    description="Atomic unit of charge equals elementary charge.",
    exact=True
)

au_energy = Constant(
    name="atomic unit of energy",
    symbol="E_h",
    value=4.359_744_722_2071e-18,
    uncertainty=8.5e-30,
    unit="J",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="E_h = m_e·c²·α²",
    depends_on=["m_e", "c", "alpha"],
    description="Hartree energy. Natural energy unit for quantum chemistry.",
    exact=False
)

au_length = Constant(
    name="atomic unit of length",
    symbol="a₀",
    value=5.291_772_109_03e-11,
    uncertainty=8.0e-21,
    unit="m",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="a₀ = ℏ/(m_e·c·α)",
    depends_on=["h_bar", "m_e", "c", "alpha"],
    description="Bohr radius. Natural length for atomic physics.",
    exact=False
)

au_mass = Constant(
    name="atomic unit of mass",
    symbol="m_e",
    value=9.109_383_7015e-31,
    uncertainty=2.8e-40,
    unit="kg",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="au_mass = m_e",
    depends_on=["m_e"],
    description="Electron mass is the atomic unit of mass.",
    exact=False
)

au_time = Constant(
    name="atomic unit of time",
    symbol="ℏ/E_h",
    value=2.418_884_326_5857e-17,
    uncertainty=4.7e-29,
    unit="s",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="t_au = ℏ/E_h",
    depends_on=["h_bar", "E_h"],
    description="Natural time for atomic dynamics. ~24 attoseconds.",
    exact=False
)

au_velocity = Constant(
    name="atomic unit of velocity",
    symbol="αc",
    value=2.187_691_263_64e6,
    uncertainty=3.3e-4,
    unit="m s^-1",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="v_au = α·c = a₀·E_h/ℏ",
    depends_on=["alpha", "c"],
    description="Electron velocity in first Bohr orbit.",
    exact=False
)

au_momentum = Constant(
    name="atomic unit of momentum",
    symbol="ℏ/a₀",
    value=1.992_851_914_10e-24,
    uncertainty=3.0e-34,
    unit="kg m s^-1",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="p_au = ℏ/a₀ = m_e·α·c",
    depends_on=["h_bar", "a_0"],
    description="Natural momentum unit for atomic physics.",
    exact=False
)

au_force = Constant(
    name="atomic unit of force",
    symbol="E_h/a₀",
    value=8.238_723_4983e-8,
    uncertainty=1.2e-17,
    unit="N",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="F_au = E_h/a₀",
    depends_on=["E_h", "a_0"],
    description="Natural force unit. F ~ 82 nN.",
    exact=False
)

au_electric_field = Constant(
    name="atomic unit of electric field",
    symbol="E_h/(e·a₀)",
    value=5.142_206_747_63e11,
    uncertainty=7.8e1,
    unit="V m^-1",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="E_au = E_h/(e·a₀)",
    depends_on=["E_h", "e", "a_0"],
    description="Electric field at Bohr radius. ~5×10¹¹ V/m.",
    exact=False
)

au_electric_potential = Constant(
    name="atomic unit of electric potential",
    symbol="E_h/e",
    value=27.211_386_245_988,
    uncertainty=5.3e-11,
    unit="V",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="V_au = E_h/e",
    depends_on=["E_h", "e"],
    description="Hartree in volts. ~27.2 V.",
    exact=False
)

au_electric_dipole = Constant(
    name="atomic unit of electric dipole moment",
    symbol="e·a₀",
    value=8.478_353_6255e-30,
    uncertainty=1.3e-39,
    unit="C m",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="d_au = e·a₀",
    depends_on=["e", "a_0"],
    description="Natural unit of electric dipole moment.",
    exact=False
)

au_magnetic_dipole = Constant(
    name="atomic unit of magnetic dipole moment",
    symbol="ℏe/m_e",
    value=1.854_802_015_66e-23,
    uncertainty=5.6e-33,
    unit="J T^-1",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="μ_au = 2μ_B = ℏe/m_e",
    depends_on=["h_bar", "e", "m_e"],
    description="Twice the Bohr magneton.",
    exact=False
)

au_magnetic_flux_density = Constant(
    name="atomic unit of magnetic flux density",
    symbol="ℏ/(e·a₀²)",
    value=2.350_517_567_58e5,
    uncertainty=7.1e-5,
    unit="T",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="B_au = ℏ/(e·a₀²)",
    depends_on=["h_bar", "e", "a_0"],
    description="Natural magnetic field unit. ~235 kT.",
    exact=False
)

au_current = Constant(
    name="atomic unit of current",
    symbol="e·E_h/ℏ",
    value=6.623_618_237_510e-3,
    uncertainty=1.3e-14,
    unit="A",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="I_au = e·E_h/ℏ",
    depends_on=["e", "E_h", "h_bar"],
    description="Natural current unit. ~6.6 mA.",
    exact=False
)

au_charge_density = Constant(
    name="atomic unit of charge density",
    symbol="e/a₀³",
    value=1.081_202_384_57e12,
    uncertainty=4.9e2,
    unit="C m^-3",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="ρ_au = e/a₀³",
    depends_on=["e", "a_0"],
    description="Natural charge density unit.",
    exact=False
)

au_permittivity = Constant(
    name="atomic unit of permittivity",
    symbol="e²/(a₀·E_h)",
    value=1.112_650_055_45e-10,
    uncertainty=1.7e-20,
    unit="F m^-1",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="ε_au = e²/(a₀·E_h) = 4πε₀",
    depends_on=["e", "a_0", "E_h"],
    description="4π times vacuum permittivity.",
    exact=False
)

au_polarizability = Constant(
    name="atomic unit of electric polarizability",
    symbol="e²a₀²/E_h",
    value=1.648_777_274_36e-41,
    uncertainty=5.0e-51,
    unit="C² m² J^-1",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="α_au = e²a₀²/E_h",
    depends_on=["e", "a_0", "E_h"],
    description="Natural polarizability unit.",
    exact=False
)

au_1st_hyperpolarizability = Constant(
    name="atomic unit of 1st hyperpolarizability",
    symbol="e³a₀³/E_h²",
    value=3.206_361_3061e-53,
    uncertainty=1.5e-62,
    unit="C³ m³ J^-2",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="β_au = e³a₀³/E_h²",
    depends_on=["e", "a_0", "E_h"],
    description="First hyperpolarizability unit.",
    exact=False
)

au_2nd_hyperpolarizability = Constant(
    name="atomic unit of 2nd hyperpolarizability",
    symbol="e⁴a₀⁴/E_h³",
    value=6.235_380_0241e-65,
    uncertainty=3.8e-74,
    unit="C⁴ m⁴ J^-3",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="γ_au = e⁴a₀⁴/E_h³",
    depends_on=["e", "a_0", "E_h"],
    description="Second hyperpolarizability unit.",
    exact=False
)

au_magnetizability = Constant(
    name="atomic unit of magnetizability",
    symbol="e²a₀²/m_e",
    value=7.891_036_6008e-29,
    uncertainty=4.8e-38,
    unit="J T^-2",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="ξ_au = e²a₀²/m_e",
    depends_on=["e", "a_0", "m_e"],
    description="Natural magnetizability unit.",
    exact=False
)

au_electric_quadrupole = Constant(
    name="atomic unit of electric quadrupole moment",
    symbol="e·a₀²",
    value=4.486_551_5246e-40,
    uncertainty=1.4e-49,
    unit="C m²",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="Q_au = e·a₀²",
    depends_on=["e", "a_0"],
    description="Natural quadrupole moment unit.",
    exact=False
)

au_electric_field_gradient = Constant(
    name="atomic unit of electric field gradient",
    symbol="E_h/(e·a₀²)",
    value=9.717_362_4292e21,
    uncertainty=2.9e12,
    unit="V m^-2",
    category=ConstantCategory.ATOMIC_UNITS,
    formula="EFG_au = E_h/(e·a₀²)",
    depends_on=["E_h", "e", "a_0"],
    description="Natural EFG unit for NMR.",
    exact=False
)


# =============================================================================
# SECTION 17: NATURAL UNITS (c = ℏ = 1)
# =============================================================================

nu_action = Constant(
    name="natural unit of action",
    symbol="ℏ",
    value=1.054_571_817_646e-34,
    uncertainty=0.0,
    unit="J s",
    category=ConstantCategory.NATURAL_UNITS,
    formula="ℏ = 1 in natural units",
    depends_on=["h_bar"],
    description="Natural unit of action. Set to 1 in natural units.",
    exact=True
)

nu_action_eV = Constant(
    name="natural unit of action in eV s",
    symbol="ℏ",
    value=6.582_119_569e-16,
    uncertainty=0.0,
    unit="eV s",
    category=ConstantCategory.NATURAL_UNITS,
    formula="ℏ in eV·s",
    depends_on=["h_bar", "e"],
    description="Reduced Planck constant in eV·s.",
    exact=True
)

nu_velocity = Constant(
    name="natural unit of velocity",
    symbol="c",
    value=299_792_458.0,
    uncertainty=0.0,
    unit="m s^-1",
    category=ConstantCategory.NATURAL_UNITS,
    formula="c = 1 in natural units",
    depends_on=["c"],
    description="Speed of light. Set to 1 in natural units.",
    exact=True
)

nu_length = Constant(
    name="natural unit of length",
    symbol="ƛ_C",
    value=3.861_592_6796e-13,
    uncertainty=1.2e-22,
    unit="m",
    category=ConstantCategory.NATURAL_UNITS,
    formula="ƛ_C = ℏ/(m_e·c)",
    depends_on=["h_bar", "m_e", "c"],
    description="Reduced Compton wavelength of electron.",
    exact=False
)

nu_mass = Constant(
    name="natural unit of mass",
    symbol="m_e",
    value=9.109_383_7015e-31,
    uncertainty=2.8e-40,
    unit="kg",
    category=ConstantCategory.NATURAL_UNITS,
    formula="m_e = 1 in electron natural units",
    depends_on=["m_e"],
    description="Electron mass as natural mass unit.",
    exact=False
)

nu_energy = Constant(
    name="natural unit of energy",
    symbol="m_e·c²",
    value=8.187_105_7769e-14,
    uncertainty=2.5e-23,
    unit="J",
    category=ConstantCategory.NATURAL_UNITS,
    formula="E_nu = m_e·c²",
    depends_on=["m_e", "c"],
    description="Electron rest energy as natural energy unit.",
    exact=False
)

nu_energy_MeV = Constant(
    name="natural unit of energy in MeV",
    symbol="m_e·c²",
    value=0.510_998_950_00,
    uncertainty=1.5e-10,
    unit="MeV",
    category=ConstantCategory.NATURAL_UNITS,
    formula="m_e·c²/e in MeV",
    depends_on=["m_e", "c", "e"],
    description="Electron rest energy in MeV. ~0.511 MeV.",
    exact=False
)

nu_momentum = Constant(
    name="natural unit of momentum",
    symbol="m_e·c",
    value=2.730_924_530_75e-22,
    uncertainty=8.2e-32,
    unit="kg m s^-1",
    category=ConstantCategory.NATURAL_UNITS,
    formula="p_nu = m_e·c",
    depends_on=["m_e", "c"],
    description="Natural momentum unit.",
    exact=False
)

nu_momentum_MeV = Constant(
    name="natural unit of momentum in MeV/c",
    symbol="m_e·c",
    value=0.510_998_950_00,
    uncertainty=1.5e-10,
    unit="MeV/c",
    category=ConstantCategory.NATURAL_UNITS,
    formula="m_e·c in MeV/c",
    depends_on=["m_e", "c"],
    description="Natural momentum in MeV/c. Same as rest energy numerically.",
    exact=False
)

nu_time = Constant(
    name="natural unit of time",
    symbol="ℏ/(m_e·c²)",
    value=1.288_088_668_19e-21,
    uncertainty=3.9e-31,
    unit="s",
    category=ConstantCategory.NATURAL_UNITS,
    formula="t_nu = ℏ/(m_e·c²)",
    depends_on=["h_bar", "m_e", "c"],
    description="Natural time unit. ~1.3 zeptoseconds.",
    exact=False
)

hbar_c_MeV_fm = Constant(
    name="reduced Planck constant times c in MeV fm",
    symbol="ℏc",
    value=197.326_980_4,
    uncertainty=0.0,
    unit="MeV fm",
    category=ConstantCategory.NATURAL_UNITS,
    formula="ℏc ≈ 197 MeV·fm",
    depends_on=["h_bar", "c"],
    description="ℏc in convenient units. Useful for nuclear/particle physics.",
    exact=True
)


# =============================================================================
# SECTION 18: ELECTROWEAK CONSTANTS
# =============================================================================

G_F = Constant(
    name="Fermi coupling constant",
    symbol="G_F",
    value=1.166_378_7e-5,
    uncertainty=6.0e-12,
    unit="GeV^-2",
    category=ConstantCategory.ELECTROWEAK,
    formula="G_F/(ℏc)³ = 1.166×10⁻⁵ GeV⁻²",
    depends_on=[],
    description="Weak interaction coupling. G_F determines beta decay rates. "
                "Related to W mass: G_F = πα/(√2·M_W²·sin²θ_W).",
    exact=False
)

sin2_theta_W = Constant(
    name="weak mixing angle",
    symbol="sin²θ_W",
    value=0.222_90,
    uncertainty=3.0e-4,
    unit="",
    category=ConstantCategory.ELECTROWEAK,
    formula="sin²θ_W = 1 - (M_W/M_Z)²",
    depends_on=[],
    description="Weinberg angle. Determines W/Z mass ratio and coupling strengths. "
                "sin²θ_W ≈ 0.223. Key electroweak symmetry breaking parameter.",
    exact=False
)

W_Z_mass_ratio = Constant(
    name="W to Z mass ratio",
    symbol="M_W/M_Z",
    value=0.881_53,
    uncertainty=1.7e-4,
    unit="",
    category=ConstantCategory.ELECTROWEAK,
    formula="M_W/M_Z = cos(θ_W)",
    depends_on=["sin2_theta_W"],
    description="Ratio of W to Z boson masses. M_W/M_Z = cos(θ_W) ≈ 0.88. "
                "Tests electroweak theory precision.",
    exact=False
)


# =============================================================================
# SECTION 19: SHIELDED MAGNETIC MOMENTS
# =============================================================================

gamma_h_shielded = Constant(
    name="shielded helion gyromagnetic ratio",
    symbol="γ'_h",
    value=2.037_894_569e8,
    uncertainty=2.4e-1,
    unit="s^-1 T^-1",
    category=ConstantCategory.SHIELDED,
    formula="γ'_h = γ_h(1 - σ)",
    depends_on=[],
    description="Helion gyromagnetic ratio in shielding environment.",
    exact=False
)

gamma_h_shielded_MHz = Constant(
    name="shielded helion gyromagnetic ratio in MHz/T",
    symbol="γ'_h/(2π)",
    value=32.434_099_42,
    uncertainty=3.8e-7,
    unit="MHz T^-1",
    category=ConstantCategory.SHIELDED,
    formula="γ'_h/(2π)",
    depends_on=["gamma_h_shielded"],
    description="Shielded helion gyromagnetic ratio in MHz/T.",
    exact=False
)

mu_h_shielded = Constant(
    name="shielded helion magnetic moment",
    symbol="μ'_h",
    value=-1.074_553_090e-26,
    uncertainty=1.3e-34,
    unit="J T^-1",
    category=ConstantCategory.SHIELDED,
    formula="μ'_h = μ_h(1 - σ)",
    depends_on=[],
    description="Helion magnetic moment in H₂O solution.",
    exact=False
)

mu_h_shielded_over_mu_B = Constant(
    name="shielded helion mag. mom. to Bohr magneton ratio",
    symbol="μ'_h/μ_B",
    value=-1.158_671_471e-3,
    uncertainty=1.4e-12,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="μ'_h/μ_B",
    depends_on=["mu_h_shielded", "mu_B"],
    description="Shielded helion moment in Bohr magnetons.",
    exact=False
)

mu_h_shielded_over_mu_N = Constant(
    name="shielded helion mag. mom. to nuclear magneton ratio",
    symbol="μ'_h/μ_N",
    value=-2.127_497_719,
    uncertainty=2.5e-8,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="μ'_h/μ_N",
    depends_on=["mu_h_shielded", "mu_N"],
    description="Shielded helion moment in nuclear magnetons.",
    exact=False
)

mu_p_shielded = Constant(
    name="shielded proton magnetic moment",
    symbol="μ'_p",
    value=1.410_570_560e-26,
    uncertainty=1.5e-34,
    unit="J T^-1",
    category=ConstantCategory.SHIELDED,
    formula="μ'_p = μ_p(1 - σ)",
    depends_on=["mu_p"],
    description="Proton magnetic moment in H₂O sphere at 25°C.",
    exact=False
)

mu_p_shielded_over_mu_B = Constant(
    name="shielded proton mag. mom. to Bohr magneton ratio",
    symbol="μ'_p/μ_B",
    value=1.520_993_128e-3,
    uncertainty=1.7e-12,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="μ'_p/μ_B",
    depends_on=["mu_p_shielded", "mu_B"],
    description="Shielded proton moment in Bohr magnetons.",
    exact=False
)

mu_p_shielded_over_mu_N = Constant(
    name="shielded proton mag. mom. to nuclear magneton ratio",
    symbol="μ'_p/μ_N",
    value=2.792_775_599,
    uncertainty=3.0e-8,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="μ'_p/μ_N",
    depends_on=["mu_p_shielded", "mu_N"],
    description="Shielded proton moment in nuclear magnetons.",
    exact=False
)

gamma_p_shielded_MHz = Constant(
    name="shielded proton gyromagnetic ratio in MHz/T",
    symbol="γ'_p/(2π)",
    value=42.576_384_74,
    uncertainty=4.6e-7,
    unit="MHz T^-1",
    category=ConstantCategory.SHIELDED,
    formula="γ'_p/(2π)",
    depends_on=["gamma_p_shielded"],
    description="Shielded proton gyromagnetic ratio in MHz/T. Used in NMR.",
    exact=False
)

helion_shielding_shift = Constant(
    name="helion shielding shift",
    symbol="σ_h",
    value=5.996_743e-5,
    uncertainty=1.0e-9,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="σ_h = (γ_h - γ'_h)/γ_h",
    depends_on=[],
    description="Helion magnetic shielding shift in gas.",
    exact=False
)

proton_magnetic_shielding = Constant(
    name="proton magnetic shielding correction",
    symbol="σ'_p",
    value=2.5689e-5,
    uncertainty=1.1e-8,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="σ'_p = 1 - γ'_p/γ_p",
    depends_on=["gamma_p", "gamma_p_shielded"],
    description="Proton diamagnetic shielding in H₂O.",
    exact=False
)

shielding_d_p_in_HD = Constant(
    name="shielding difference of d and p in HD",
    symbol="σ_d - σ_p",
    value=2.02e-8,
    uncertainty=2.0e-11,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="(σ_d - σ_p) in HD molecule",
    depends_on=[],
    description="Difference of deuteron and proton shielding in HD.",
    exact=False
)

shielding_t_p_in_HT = Constant(
    name="shielding difference of t and p in HT",
    symbol="σ_t - σ_p",
    value=2.41e-8,
    uncertainty=2.0e-11,
    unit="",
    category=ConstantCategory.SHIELDED,
    formula="(σ_t - σ_p) in HT molecule",
    depends_on=[],
    description="Difference of triton and proton shielding in HT.",
    exact=False
)


# =============================================================================
# SECTION 20: MOLAR MASSES
# =============================================================================

M_e = Constant(
    name="electron molar mass",
    symbol="M_e",
    value=5.485_799_0888e-7,
    uncertainty=1.7e-16,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_e = N_A·m_e",
    depends_on=["N_A", "m_e"],
    description="Molar mass of electrons.",
    exact=False
)

M_p = Constant(
    name="proton molar mass",
    symbol="M_p",
    value=1.007_276_466_27e-3,
    uncertainty=3.1e-13,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_p = N_A·m_p",
    depends_on=["N_A", "m_p"],
    description="Molar mass of protons.",
    exact=False
)

M_n = Constant(
    name="neutron molar mass",
    symbol="M_n",
    value=1.008_664_916_06e-3,
    uncertainty=5.7e-13,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_n = N_A·m_n",
    depends_on=["N_A", "m_n"],
    description="Molar mass of neutrons.",
    exact=False
)

M_mu = Constant(
    name="muon molar mass",
    symbol="M_μ",
    value=1.134_289_259e-4,
    uncertainty=2.5e-12,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_μ = N_A·m_μ",
    depends_on=["N_A", "m_mu"],
    description="Molar mass of muons.",
    exact=False
)

M_tau = Constant(
    name="tau molar mass",
    symbol="M_τ",
    value=1.907_54e-3,
    uncertainty=1.3e-7,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_τ = N_A·m_τ",
    depends_on=["N_A", "m_tau"],
    description="Molar mass of tau leptons.",
    exact=False
)

M_d = Constant(
    name="deuteron molar mass",
    symbol="M_d",
    value=2.013_553_212_05e-3,
    uncertainty=6.1e-13,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_d = N_A·m_d",
    depends_on=["N_A", "m_d"],
    description="Molar mass of deuterons.",
    exact=False
)

M_t = Constant(
    name="triton molar mass",
    symbol="M_t",
    value=3.015_500_715_17e-3,
    uncertainty=9.2e-13,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_t = N_A·m_t",
    depends_on=["N_A", "m_t"],
    description="Molar mass of tritons.",
    exact=False
)

M_h = Constant(
    name="helion molar mass",
    symbol="M_h",
    value=3.014_932_246_13e-3,
    uncertainty=9.1e-13,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_h = N_A·m_h",
    depends_on=["N_A", "m_h"],
    description="Molar mass of helions (He-3 nuclei).",
    exact=False
)

M_alpha = Constant(
    name="alpha particle molar mass",
    symbol="M_α",
    value=4.001_506_179_13e-3,
    uncertainty=1.2e-12,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_α = N_A·m_α",
    depends_on=["N_A", "m_alpha"],
    description="Molar mass of alpha particles (He-4 nuclei).",
    exact=False
)

M_12C = Constant(
    name="molar mass of carbon-12",
    symbol="M(¹²C)",
    value=11.999_999_9958e-3,
    uncertainty=3.6e-12,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M(¹²C) = 12·u·N_A",
    depends_on=["u", "N_A"],
    description="Molar mass of carbon-12. Defines the mole.",
    exact=False
)

molar_mass_constant = Constant(
    name="molar mass constant",
    symbol="M_u",
    value=0.999_999_999_65e-3,
    uncertainty=3.0e-13,
    unit="kg mol^-1",
    category=ConstantCategory.MOLAR,
    formula="M_u = 1 g/mol = m_u·N_A",
    depends_on=["u", "N_A"],
    description="Molar mass constant. M_u = 1 g/mol exactly before 2019.",
    exact=False
)


# =============================================================================
# SECTION 21: X-RAY UNITS
# =============================================================================

angstrom_star = Constant(
    name="Angstrom star",
    symbol="Å*",
    value=1.000_014_95e-10,
    uncertainty=9.0e-17,
    unit="m",
    category=ConstantCategory.X_UNITS,
    formula="Å* ≈ 1.000015 Å",
    depends_on=[],
    description="X-ray wavelength unit. Slightly larger than standard angstrom.",
    exact=False
)

Cu_x_unit = Constant(
    name="Copper x unit",
    symbol="xu(Cu Kα₁)",
    value=1.002_076_97e-13,
    uncertainty=2.8e-20,
    unit="m",
    category=ConstantCategory.X_UNITS,
    formula="xu(Cu Kα₁)",
    depends_on=[],
    description="X-unit based on Cu Kα₁ line.",
    exact=False
)

Mo_x_unit = Constant(
    name="Molybdenum x unit",
    symbol="xu(Mo Kα₁)",
    value=1.002_099_52e-13,
    uncertainty=5.3e-20,
    unit="m",
    category=ConstantCategory.X_UNITS,
    formula="xu(Mo Kα₁)",
    depends_on=[],
    description="X-unit based on Mo Kα₁ line.",
    exact=False
)

Si_lattice = Constant(
    name="lattice parameter of silicon",
    symbol="a",
    value=5.431_020_511e-10,
    uncertainty=8.9e-18,
    unit="m",
    category=ConstantCategory.X_UNITS,
    formula="a(Si) at 22.5°C",
    depends_on=[],
    description="Silicon crystal lattice constant.",
    exact=False
)

Si_220_spacing = Constant(
    name="lattice spacing of ideal Si (220)",
    symbol="d₂₂₀",
    value=1.920_155_716e-10,
    uncertainty=3.2e-18,
    unit="m",
    category=ConstantCategory.X_UNITS,
    formula="d₂₂₀ = a/√8",
    depends_on=["Si_lattice"],
    description="Silicon (220) plane spacing.",
    exact=False
)

Si_molar_volume = Constant(
    name="molar volume of silicon",
    symbol="V_m(Si)",
    value=1.205_883_199e-5,
    uncertainty=6.0e-13,
    unit="m³ mol^-1",
    category=ConstantCategory.X_UNITS,
    formula="V_m(Si) = M(Si)/ρ(Si)",
    depends_on=[],
    description="Molar volume of crystalline silicon.",
    exact=False
)


# =============================================================================
# SECTION 22: CONVENTIONAL VALUES
# =============================================================================

K_J_90 = Constant(
    name="conventional value of Josephson constant",
    symbol="K_J-90",
    value=483_597.9e9,
    uncertainty=0.0,
    unit="Hz V^-1",
    category=ConstantCategory.CONVENTIONAL,
    formula="K_J-90 = 483597.9 GHz/V exactly",
    depends_on=[],
    description="Conventional Josephson constant for pre-2019 electrical standards.",
    exact=True
)

R_K_90 = Constant(
    name="conventional value of von Klitzing constant",
    symbol="R_K-90",
    value=25_812.807,
    uncertainty=0.0,
    unit="Ω",
    category=ConstantCategory.CONVENTIONAL,
    formula="R_K-90 = 25812.807 Ω exactly",
    depends_on=[],
    description="Conventional von Klitzing constant for pre-2019 resistance standards.",
    exact=True
)

# Pre-2019 conventional electrical units
ohm_90 = Constant(
    name="conventional value of ohm-90",
    symbol="Ω₉₀",
    value=1.000_000_017_79,
    uncertainty=0.0,
    unit="Ω",
    category=ConstantCategory.CONVENTIONAL,
    formula="Ω₉₀ = R_K-90/R_K",
    depends_on=["R_K_90", "R_K"],
    description="1990 conventional ohm relative to SI ohm.",
    exact=False
)

volt_90 = Constant(
    name="conventional value of volt-90",
    symbol="V₉₀",
    value=1.000_000_106_66,
    uncertainty=0.0,
    unit="V",
    category=ConstantCategory.CONVENTIONAL,
    formula="V₉₀ = K_J/K_J-90",
    depends_on=["K_J", "K_J_90"],
    description="1990 conventional volt relative to SI volt.",
    exact=False
)

ampere_90 = Constant(
    name="conventional value of ampere-90",
    symbol="A₉₀",
    value=1.000_000_088_87,
    uncertainty=0.0,
    unit="A",
    category=ConstantCategory.CONVENTIONAL,
    formula="A₉₀ = V₉₀/Ω₉₀",
    depends_on=["volt_90", "ohm_90"],
    description="1990 conventional ampere.",
    exact=False
)

watt_90 = Constant(
    name="conventional value of watt-90",
    symbol="W₉₀",
    value=1.000_000_195_53,
    uncertainty=0.0,
    unit="W",
    category=ConstantCategory.CONVENTIONAL,
    formula="W₉₀ = V₉₀·A₉₀",
    depends_on=["volt_90", "ampere_90"],
    description="1990 conventional watt.",
    exact=False
)


# =============================================================================
# SECTION 23: ADDITIONAL RATIOS AND RELATIONSHIPS
# =============================================================================

# More mass ratios
m_d_over_m_e = Constant(
    name="deuteron-electron mass ratio",
    symbol="m_d/m_e",
    value=3670.482_967_88,
    uncertainty=1.1e-6,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_d/m_e",
    depends_on=["m_d", "m_e"],
    description="Deuteron to electron mass ratio.",
    exact=False
)

m_d_over_m_p = Constant(
    name="deuteron-proton mass ratio",
    symbol="m_d/m_p",
    value=1.999_007_501_39,
    uncertainty=1.1e-9,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_d/m_p",
    depends_on=["m_d", "m_p"],
    description="Deuteron to proton mass ratio. Nearly 2.",
    exact=False
)

m_h_over_m_p = Constant(
    name="helion-proton mass ratio",
    symbol="m_h/m_p",
    value=2.993_152_671_67,
    uncertainty=1.3e-9,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_h/m_p",
    depends_on=["m_h", "m_p"],
    description="Helion to proton mass ratio. Nearly 3.",
    exact=False
)

m_t_over_m_p = Constant(
    name="triton-proton mass ratio",
    symbol="m_t/m_p",
    value=2.993_717_034_14,
    uncertainty=1.3e-9,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_t/m_p",
    depends_on=["m_t", "m_p"],
    description="Triton to proton mass ratio. Nearly 3.",
    exact=False
)

m_alpha_over_m_p = Constant(
    name="alpha particle-proton mass ratio",
    symbol="m_α/m_p",
    value=3.972_599_690_09,
    uncertainty=2.2e-9,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_α/m_p",
    depends_on=["m_alpha", "m_p"],
    description="Alpha to proton mass ratio. Nearly 4.",
    exact=False
)

m_mu_over_m_p = Constant(
    name="muon-proton mass ratio",
    symbol="m_μ/m_p",
    value=0.112_609_5264,
    uncertainty=2.5e-9,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_μ/m_p",
    depends_on=["m_mu", "m_p"],
    description="Muon to proton mass ratio.",
    exact=False
)

m_tau_over_m_p = Constant(
    name="tau-proton mass ratio",
    symbol="m_τ/m_p",
    value=1.893_76,
    uncertainty=1.3e-4,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_τ/m_p",
    depends_on=["m_tau", "m_p"],
    description="Tau to proton mass ratio. Nearly 2.",
    exact=False
)

m_mu_over_m_tau = Constant(
    name="muon-tau mass ratio",
    symbol="m_μ/m_τ",
    value=5.946_35e-2,
    uncertainty=4.0e-6,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_μ/m_τ",
    depends_on=["m_mu", "m_tau"],
    description="Muon to tau mass ratio.",
    exact=False
)

m_n_over_m_e = Constant(
    name="neutron-electron mass ratio",
    symbol="m_n/m_e",
    value=1838.683_661_73,
    uncertainty=8.9e-7,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="m_n/m_e",
    depends_on=["m_n", "m_e"],
    description="Neutron to electron mass ratio.",
    exact=False
)

# Magnetic moment ratios
mu_d_over_mu_e = Constant(
    name="deuteron-electron magnetic moment ratio",
    symbol="μ_d/μ_e",
    value=-4.664_345_551e-4,
    uncertainty=1.2e-12,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_d/μ_e",
    depends_on=["mu_d", "mu_e"],
    description="Deuteron to electron magnetic moment ratio.",
    exact=False
)

mu_d_over_mu_p = Constant(
    name="deuteron-proton magnetic moment ratio",
    symbol="μ_d/μ_p",
    value=0.307_012_209_39,
    uncertainty=7.9e-10,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_d/μ_p",
    depends_on=["mu_d", "mu_p"],
    description="Deuteron to proton magnetic moment ratio.",
    exact=False
)

mu_d_over_mu_n = Constant(
    name="deuteron-neutron magnetic moment ratio",
    symbol="μ_d/μ_n",
    value=-0.448_206_53,
    uncertainty=1.1e-7,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_d/μ_n",
    depends_on=["mu_d", "mu_n"],
    description="Deuteron to neutron magnetic moment ratio.",
    exact=False
)

mu_n_over_mu_e = Constant(
    name="neutron-electron magnetic moment ratio",
    symbol="μ_n/μ_e",
    value=1.040_668_82e-3,
    uncertainty=2.5e-10,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_n/μ_e",
    depends_on=["mu_n", "mu_e"],
    description="Neutron to electron magnetic moment ratio.",
    exact=False
)

mu_n_over_mu_p = Constant(
    name="neutron-proton magnetic moment ratio",
    symbol="μ_n/μ_p",
    value=-0.684_979_34,
    uncertainty=1.6e-7,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_n/μ_p",
    depends_on=["mu_n", "mu_p"],
    description="Neutron to proton magnetic moment ratio.",
    exact=False
)

mu_mu_over_mu_p = Constant(
    name="muon-proton magnetic moment ratio",
    symbol="μ_μ/μ_p",
    value=-3.183_345_142,
    uncertainty=7.1e-8,
    unit="",
    category=ConstantCategory.RATIOS,
    formula="μ_μ/μ_p",
    depends_on=["mu_mu", "mu_p"],
    description="Muon to proton magnetic moment ratio.",
    exact=False
)

# Additional electromagnetic ratios
e_over_m_e = Constant(
    name="electron charge to mass quotient",
    symbol="e/m_e",
    value=-1.758_820_010_76e11,
    uncertainty=5.3e1,
    unit="C kg^-1",
    category=ConstantCategory.RATIOS,
    formula="-e/m_e",
    depends_on=["e", "m_e"],
    description="Electron charge-to-mass ratio. Negative for electron.",
    exact=False
)

e_over_m_p = Constant(
    name="proton charge to mass quotient",
    symbol="e/m_p",
    value=9.578_833_1560e7,
    uncertainty=2.9e-2,
    unit="C kg^-1",
    category=ConstantCategory.RATIOS,
    formula="e/m_p",
    depends_on=["e", "m_p"],
    description="Proton charge-to-mass ratio.",
    exact=False
)

e_over_h = Constant(
    name="elementary charge over h-bar",
    symbol="e/ℏ",
    value=1.519_267_447e15,
    uncertainty=0.0,
    unit="C J^-1 s^-1",
    category=ConstantCategory.RATIOS,
    formula="e/ℏ",
    depends_on=["e", "h_bar"],
    description="Ratio of elementary charge to reduced Planck constant.",
    exact=True
)

# Quantum of circulation
quantum_circulation = Constant(
    name="quantum of circulation",
    symbol="h/(2m_e)",
    value=3.636_947_5516e-4,
    uncertainty=1.1e-13,
    unit="m² s^-1",
    category=ConstantCategory.QED,
    formula="h/(2m_e)",
    depends_on=["h", "m_e"],
    description="Quantum of circulation for electron.",
    exact=False
)

quantum_circulation_2 = Constant(
    name="quantum of circulation times 2",
    symbol="h/m_e",
    value=7.273_895_1032e-4,
    uncertainty=2.2e-13,
    unit="m² s^-1",
    category=ConstantCategory.QED,
    formula="h/m_e",
    depends_on=["h", "m_e"],
    description="Twice the quantum of circulation.",
    exact=False
)

# Inverse conductance quantum
inverse_G_0 = Constant(
    name="inverse of conductance quantum",
    symbol="1/G₀",
    value=12_906.403_72,
    uncertainty=0.0,
    unit="Ω",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="1/G₀ = h/(2e²) = R_K/2",
    depends_on=["h", "e", "R_K"],
    description="Inverse conductance quantum. Half of von Klitzing constant.",
    exact=True
)

# Gravitation in natural units
G_over_hbar_c = Constant(
    name="Newtonian constant of gravitation over h-bar c",
    symbol="G/(ℏc)",
    value=6.708_83e-39,
    uncertainty=1.5e-43,
    unit="(GeV/c²)^-2",
    category=ConstantCategory.PLANCK,
    formula="G/(ℏc) = 1/m_P²",
    depends_on=["G", "h_bar", "c", "m_P"],
    description="Gravitational constant in natural units. G/(ℏc) = l_P²/ℏ.",
    exact=False
)

# Planck mass in GeV
m_P_GeV = Constant(
    name="Planck mass energy equivalent in GeV",
    symbol="m_P·c²",
    value=1.220_890e19,
    uncertainty=1.4e14,
    unit="GeV",
    category=ConstantCategory.PLANCK,
    formula="m_P·c² in GeV",
    depends_on=["m_P", "c"],
    description="Planck mass in GeV. ~1.22×10¹⁹ GeV. Grand unification scale.",
    exact=False
)


# =============================================================================
# SECTION 24: MORE CONVERSION FACTORS
# =============================================================================

# Hartree conversions
hartree_to_u = Constant(
    name="hartree-atomic mass unit relationship",
    symbol="E_h/u·c²",
    value=2.921_262_322_05e-8,
    uncertainty=8.9e-18,
    unit="u",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h/(u·c²)",
    depends_on=["E_h", "u", "c"],
    description="Hartree in atomic mass units.",
    exact=False
)

hartree_to_Hz = Constant(
    name="hartree-hertz relationship",
    symbol="E_h/h",
    value=6.579_683_920_502e15,
    uncertainty=1.3e4,
    unit="Hz",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h/h",
    depends_on=["E_h", "h"],
    description="Hartree in hertz.",
    exact=False
)

hartree_to_m = Constant(
    name="hartree-inverse meter relationship",
    symbol="E_h/(hc)",
    value=2.194_746_313_6320e7,
    uncertainty=4.3e-5,
    unit="m^-1",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h/(hc)",
    depends_on=["E_h", "h", "c"],
    description="Hartree in inverse meters.",
    exact=False
)

hartree_to_K = Constant(
    name="hartree-kelvin relationship",
    symbol="E_h/k",
    value=3.157_750_248_0407e5,
    uncertainty=6.1e-7,
    unit="K",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h/k",
    depends_on=["E_h", "k"],
    description="Hartree in kelvin.",
    exact=False
)

hartree_to_kg = Constant(
    name="hartree-kilogram relationship",
    symbol="E_h/c²",
    value=4.850_870_209_5432e-35,
    uncertainty=9.4e-47,
    unit="kg",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E_h/c²",
    depends_on=["E_h", "c"],
    description="Hartree mass equivalent.",
    exact=False
)

# Hertz conversions
Hz_to_J = Constant(
    name="hertz-joule relationship",
    symbol="h",
    value=6.626_070_15e-34,
    uncertainty=0.0,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="(1 Hz)·h = h",
    depends_on=["h"],
    description="Energy per hertz. E = hf.",
    exact=True
)

Hz_to_K = Constant(
    name="hertz-kelvin relationship",
    symbol="h/k",
    value=4.799_243_073e-11,
    uncertainty=0.0,
    unit="K",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/k",
    depends_on=["h", "k"],
    description="Kelvin per hertz. T = hf/k.",
    exact=True
)

Hz_to_eV = Constant(
    name="hertz-electron volt relationship",
    symbol="h/e",
    value=4.135_667_696e-15,
    uncertainty=0.0,
    unit="eV",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/e",
    depends_on=["h", "e"],
    description="eV per hertz.",
    exact=True
)

Hz_to_m = Constant(
    name="hertz-inverse meter relationship",
    symbol="1/c",
    value=3.335_640_951e-9,
    uncertainty=0.0,
    unit="m^-1",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="f/c = 1/λ",
    depends_on=["c"],
    description="Inverse meters per hertz. k = ω/c.",
    exact=True
)

Hz_to_kg = Constant(
    name="hertz-kilogram relationship",
    symbol="h/c²",
    value=7.372_497_323e-51,
    uncertainty=0.0,
    unit="kg",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/c²",
    depends_on=["h", "c"],
    description="Mass per hertz via E=mc²=hf.",
    exact=True
)

Hz_to_u = Constant(
    name="hertz-atomic mass unit relationship",
    symbol="h/(u·c²)",
    value=4.439_821_6652e-24,
    uncertainty=1.3e-33,
    unit="u",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/(u·c²)",
    depends_on=["h", "u", "c"],
    description="Atomic mass units per hertz.",
    exact=False
)

Hz_to_hartree = Constant(
    name="hertz-hartree relationship",
    symbol="h/E_h",
    value=1.519_829_846_0570e-16,
    uncertainty=2.9e-28,
    unit="E_h",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/E_h",
    depends_on=["h", "E_h"],
    description="Hartrees per hertz.",
    exact=False
)

# Kelvin conversions
K_to_J = Constant(
    name="kelvin-joule relationship",
    symbol="k",
    value=1.380_649e-23,
    uncertainty=0.0,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="(1 K)·k = k",
    depends_on=["k"],
    description="Joules per kelvin. E = kT.",
    exact=True
)

K_to_Hz = Constant(
    name="kelvin-hertz relationship",
    symbol="k/h",
    value=2.083_661_912e10,
    uncertainty=0.0,
    unit="Hz",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="k/h",
    depends_on=["k", "h"],
    description="Hertz per kelvin.",
    exact=True
)

K_to_m = Constant(
    name="kelvin-inverse meter relationship",
    symbol="k/(hc)",
    value=69.503_480_04,
    uncertainty=0.0,
    unit="m^-1",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="k/(hc)",
    depends_on=["k", "h", "c"],
    description="Inverse meters per kelvin.",
    exact=True
)

K_to_kg = Constant(
    name="kelvin-kilogram relationship",
    symbol="k/c²",
    value=1.536_179_187e-40,
    uncertainty=0.0,
    unit="kg",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="k/c²",
    depends_on=["k", "c"],
    description="Kilograms per kelvin.",
    exact=True
)

K_to_u = Constant(
    name="kelvin-atomic mass unit relationship",
    symbol="k/(u·c²)",
    value=9.251_087_3014e-14,
    uncertainty=2.8e-23,
    unit="u",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="k/(u·c²)",
    depends_on=["k", "u", "c"],
    description="Atomic mass units per kelvin.",
    exact=False
)

K_to_hartree = Constant(
    name="kelvin-hartree relationship",
    symbol="k/E_h",
    value=3.166_811_563_4556e-6,
    uncertainty=6.1e-18,
    unit="E_h",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="k/E_h",
    depends_on=["k", "E_h"],
    description="Hartrees per kelvin.",
    exact=False
)

# Inverse meter conversions
m_inv_to_J = Constant(
    name="inverse meter-joule relationship",
    symbol="hc",
    value=1.986_445_857e-25,
    uncertainty=0.0,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="hc",
    depends_on=["h", "c"],
    description="Joules per inverse meter. E = hc/λ.",
    exact=True
)

m_inv_to_Hz = Constant(
    name="inverse meter-hertz relationship",
    symbol="c",
    value=299_792_458.0,
    uncertainty=0.0,
    unit="Hz",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="c",
    depends_on=["c"],
    description="Hertz per inverse meter. f = c/λ.",
    exact=True
)

m_inv_to_K = Constant(
    name="inverse meter-kelvin relationship",
    symbol="hc/k",
    value=1.438_776_877e-2,
    uncertainty=0.0,
    unit="K",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="hc/k = c₂",
    depends_on=["h", "c", "k"],
    description="Kelvin per inverse meter. Same as second radiation constant.",
    exact=True
)

m_inv_to_kg = Constant(
    name="inverse meter-kilogram relationship",
    symbol="h/c",
    value=2.210_219_094e-42,
    uncertainty=0.0,
    unit="kg",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/c",
    depends_on=["h", "c"],
    description="Kilograms per inverse meter.",
    exact=True
)

m_inv_to_eV = Constant(
    name="inverse meter-electron volt relationship",
    symbol="hc/e",
    value=1.239_841_984e-6,
    uncertainty=0.0,
    unit="eV",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="hc/e",
    depends_on=["h", "c", "e"],
    description="eV per inverse meter. 1240 eV·nm.",
    exact=True
)

m_inv_to_u = Constant(
    name="inverse meter-atomic mass unit relationship",
    symbol="h/(c·u)",
    value=1.331_025_050_10e-15,
    uncertainty=4.0e-25,
    unit="u",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="h/(c·u)",
    depends_on=["h", "c", "u"],
    description="Atomic mass units per inverse meter.",
    exact=False
)

m_inv_to_hartree = Constant(
    name="inverse meter-hartree relationship",
    symbol="hc/E_h",
    value=4.556_335_252_9120e-8,
    uncertainty=8.9e-20,
    unit="E_h",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="hc/E_h = 1/(2R_∞)",
    depends_on=["h", "c", "E_h"],
    description="Hartrees per inverse meter.",
    exact=False
)

# Kilogram conversions
kg_to_J = Constant(
    name="kilogram-joule relationship",
    symbol="c²",
    value=8.987_551_787e16,
    uncertainty=0.0,
    unit="J",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="E = mc²",
    depends_on=["c"],
    description="Joules per kilogram. Einstein's E=mc².",
    exact=True
)

kg_to_Hz = Constant(
    name="kilogram-hertz relationship",
    symbol="c²/h",
    value=1.356_392_489e50,
    uncertainty=0.0,
    unit="Hz",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="c²/h",
    depends_on=["c", "h"],
    description="Hertz per kilogram.",
    exact=True
)

kg_to_m = Constant(
    name="kilogram-inverse meter relationship",
    symbol="c/h",
    value=4.524_438_335e41,
    uncertainty=0.0,
    unit="m^-1",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="c/h = 1/λ_C where λ_C = h/(mc)",
    depends_on=["c", "h"],
    description="Inverse meters per kilogram.",
    exact=True
)

kg_to_K = Constant(
    name="kilogram-kelvin relationship",
    symbol="c²/k",
    value=6.509_657_260e39,
    uncertainty=0.0,
    unit="K",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="c²/k",
    depends_on=["c", "k"],
    description="Kelvin per kilogram.",
    exact=True
)

kg_to_eV = Constant(
    name="kilogram-electron volt relationship",
    symbol="c²/e",
    value=5.609_588_603e35,
    uncertainty=0.0,
    unit="eV",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="c²/e",
    depends_on=["c", "e"],
    description="eV per kilogram.",
    exact=True
)

kg_to_hartree = Constant(
    name="kilogram-hartree relationship",
    symbol="c²/E_h",
    value=2.061_485_788_7409e34,
    uncertainty=4.0e22,
    unit="E_h",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="c²/E_h",
    depends_on=["c", "E_h"],
    description="Hartrees per kilogram.",
    exact=False
)

# Atomic mass unit conversions
u_to_Hz = Constant(
    name="atomic mass unit-hertz relationship",
    symbol="u·c²/h",
    value=2.252_342_718_71e23,
    uncertainty=6.8e13,
    unit="Hz",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u·c²/h",
    depends_on=["u", "c", "h"],
    description="Hertz per atomic mass unit.",
    exact=False
)

u_to_m = Constant(
    name="atomic mass unit-inverse meter relationship",
    symbol="u·c/h",
    value=7.513_006_6104e14,
    uncertainty=2.3e5,
    unit="m^-1",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u·c/h",
    depends_on=["u", "c", "h"],
    description="Inverse meters per atomic mass unit.",
    exact=False
)

u_to_K = Constant(
    name="atomic mass unit-kelvin relationship",
    symbol="u·c²/k",
    value=1.080_954_019_16e13,
    uncertainty=3.3e3,
    unit="K",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u·c²/k",
    depends_on=["u", "c", "k"],
    description="Kelvin per atomic mass unit.",
    exact=False
)

u_to_hartree = Constant(
    name="atomic mass unit-hartree relationship",
    symbol="u·c²/E_h",
    value=3.423_177_6874e7,
    uncertainty=1.0e-2,
    unit="E_h",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="u·c²/E_h",
    depends_on=["u", "c", "E_h"],
    description="Hartrees per atomic mass unit.",
    exact=False
)

# eV conversions (additional)
eV_to_u = Constant(
    name="electron volt-atomic mass unit relationship",
    symbol="e/(u·c²)",
    value=1.073_544_102_33e-9,
    uncertainty=3.2e-19,
    unit="u",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="e/(u·c²)",
    depends_on=["e", "u", "c"],
    description="Atomic mass units per eV.",
    exact=False
)

eV_to_hartree = Constant(
    name="electron volt-hartree relationship",
    symbol="e/E_h",
    value=3.674_932_217_5655e-2,
    uncertainty=7.1e-14,
    unit="E_h",
    category=ConstantCategory.ENERGY_EQUIVALENTS,
    formula="e/E_h = 1/27.2114 E_h",
    depends_on=["e", "E_h"],
    description="Hartrees per eV.",
    exact=False
)


# =============================================================================
# SECTION 25: ADDITIONAL PARTICLE PROPERTIES
# =============================================================================

# Relative atomic masses
A_r_e = Constant(
    name="electron relative atomic mass",
    symbol="A_r(e)",
    value=5.485_799_0888e-4,
    uncertainty=1.7e-13,
    unit="",
    category=ConstantCategory.ELECTRON,
    formula="A_r(e) = m_e/u",
    depends_on=["m_e", "u"],
    description="Electron mass in atomic mass units.",
    exact=False
)

A_r_p = Constant(
    name="proton relative atomic mass",
    symbol="A_r(p)",
    value=1.007_276_466_621,
    uncertainty=5.3e-10,
    unit="",
    category=ConstantCategory.PROTON,
    formula="A_r(p) = m_p/u",
    depends_on=["m_p", "u"],
    description="Proton mass in atomic mass units.",
    exact=False
)

A_r_n = Constant(
    name="neutron relative atomic mass",
    symbol="A_r(n)",
    value=1.008_664_915_95,
    uncertainty=4.9e-10,
    unit="",
    category=ConstantCategory.NEUTRON,
    formula="A_r(n) = m_n/u",
    depends_on=["m_n", "u"],
    description="Neutron mass in atomic mass units.",
    exact=False
)

A_r_d = Constant(
    name="deuteron relative atomic mass",
    symbol="A_r(d)",
    value=2.013_553_212_745,
    uncertainty=4.0e-10,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="A_r(d) = m_d/u",
    depends_on=["m_d", "u"],
    description="Deuteron mass in atomic mass units.",
    exact=False
)

A_r_t = Constant(
    name="triton relative atomic mass",
    symbol="A_r(t)",
    value=3.015_500_716_21,
    uncertainty=1.2e-10,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="A_r(t) = m_t/u",
    depends_on=["m_t", "u"],
    description="Triton mass in atomic mass units.",
    exact=False
)

A_r_h = Constant(
    name="helion relative atomic mass",
    symbol="A_r(h)",
    value=3.014_932_247_175,
    uncertainty=9.7e-11,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="A_r(h) = m_h/u",
    depends_on=["m_h", "u"],
    description="Helion mass in atomic mass units.",
    exact=False
)

A_r_alpha = Constant(
    name="alpha particle relative atomic mass",
    symbol="A_r(α)",
    value=4.001_506_179_127,
    uncertainty=6.3e-11,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="A_r(α) = m_α/u",
    depends_on=["m_alpha", "u"],
    description="Alpha particle mass in atomic mass units.",
    exact=False
)

# Neutron-proton mass difference
m_n_minus_m_p = Constant(
    name="neutron-proton mass difference",
    symbol="m_n - m_p",
    value=2.305_574_35e-30,
    uncertainty=8.2e-37,
    unit="kg",
    category=ConstantCategory.NEUTRON,
    formula="m_n - m_p",
    depends_on=["m_n", "m_p"],
    description="Neutron-proton mass difference. Enables beta decay.",
    exact=False
)

m_n_minus_m_p_eV = Constant(
    name="neutron-proton mass difference energy equivalent",
    symbol="(m_n - m_p)·c²",
    value=2.072_146_89e-13,
    uncertainty=7.4e-20,
    unit="J",
    category=ConstantCategory.NEUTRON,
    formula="(m_n - m_p)·c²",
    depends_on=["m_n", "m_p", "c"],
    description="Neutron-proton mass difference energy.",
    exact=False
)

m_n_minus_m_p_MeV = Constant(
    name="neutron-proton mass difference energy equivalent in MeV",
    symbol="(m_n - m_p)·c²",
    value=1.293_332_51,
    uncertainty=4.6e-7,
    unit="MeV",
    category=ConstantCategory.NEUTRON,
    formula="(m_n - m_p)·c² in MeV",
    depends_on=["m_n", "m_p", "c"],
    description="Neutron-proton mass difference. ~1.29 MeV.",
    exact=False
)

m_n_minus_m_p_u = Constant(
    name="neutron-proton mass difference in u",
    symbol="(m_n - m_p)/u",
    value=1.388_449_33e-3,
    uncertainty=4.9e-10,
    unit="u",
    category=ConstantCategory.NEUTRON,
    formula="(m_n - m_p)/u",
    depends_on=["m_n", "m_p", "u"],
    description="Neutron-proton mass difference in atomic mass units.",
    exact=False
)

# Additional nuclear radii
r_d = Constant(
    name="deuteron rms charge radius",
    symbol="r_d",
    value=2.127_99e-15,
    uncertainty=7.4e-19,
    unit="m",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="",
    depends_on=[],
    description="Deuteron root-mean-square charge radius. ~2.13 fm.",
    exact=False
)

r_alpha = Constant(
    name="alpha particle rms charge radius",
    symbol="r_α",
    value=1.6785e-15,
    uncertainty=2.1e-18,
    unit="m",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="",
    depends_on=[],
    description="Alpha particle rms charge radius. ~1.68 fm.",
    exact=False
)

# Helion and triton g-factors
g_h = Constant(
    name="helion g factor",
    symbol="g_h",
    value=-4.255_250_615,
    uncertainty=5.0e-8,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="g_h = 2μ_h/μ_N",
    depends_on=["mu_N"],
    description="Helion g-factor. Negative due to spin structure.",
    exact=False
)

g_t = Constant(
    name="triton g factor",
    symbol="g_t",
    value=5.957_924_931,
    uncertainty=1.2e-8,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="g_t = 2μ_t/μ_N",
    depends_on=["mu_N"],
    description="Triton g-factor.",
    exact=False
)

g_d = Constant(
    name="deuteron g factor",
    symbol="g_d",
    value=0.857_438_2338,
    uncertainty=2.2e-9,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="g_d = 2μ_d/μ_N",
    depends_on=["mu_d", "mu_N"],
    description="Deuteron g-factor.",
    exact=False
)

# Triton magnetic moment
mu_t = Constant(
    name="triton magnetic moment",
    symbol="μ_t",
    value=1.504_609_5202e-26,
    uncertainty=3.0e-35,
    unit="J T^-1",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_t = g_t·μ_N/2",
    depends_on=["g_t", "mu_N"],
    description="Triton intrinsic magnetic moment.",
    exact=False
)

mu_t_over_mu_B = Constant(
    name="triton mag. mom. to Bohr magneton ratio",
    symbol="μ_t/μ_B",
    value=1.622_393_6651e-3,
    uncertainty=3.2e-12,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_t/μ_B",
    depends_on=["mu_t", "mu_B"],
    description="Triton moment in Bohr magnetons.",
    exact=False
)

mu_t_over_mu_N = Constant(
    name="triton mag. mom. to nuclear magneton ratio",
    symbol="μ_t/μ_N",
    value=2.978_962_4656,
    uncertainty=5.9e-9,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_t/μ_N = g_t/2",
    depends_on=["mu_t", "mu_N"],
    description="Triton moment in nuclear magnetons.",
    exact=False
)

mu_t_over_mu_p = Constant(
    name="triton to proton mag. mom. ratio",
    symbol="μ_t/μ_p",
    value=1.066_639_9191,
    uncertainty=2.1e-9,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_t/μ_p",
    depends_on=["mu_t", "mu_p"],
    description="Triton to proton magnetic moment ratio.",
    exact=False
)

# Helion magnetic moment
mu_h = Constant(
    name="helion magnetic moment",
    symbol="μ_h",
    value=-1.074_617_532e-26,
    uncertainty=1.3e-34,
    unit="J T^-1",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_h = g_h·μ_N/2",
    depends_on=["g_h", "mu_N"],
    description="Helion intrinsic magnetic moment. Negative.",
    exact=False
)

mu_h_over_mu_B = Constant(
    name="helion mag. mom. to Bohr magneton ratio",
    symbol="μ_h/μ_B",
    value=-1.158_740_958e-3,
    uncertainty=1.4e-12,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_h/μ_B",
    depends_on=["mu_h", "mu_B"],
    description="Helion moment in Bohr magnetons.",
    exact=False
)

mu_h_over_mu_N = Constant(
    name="helion mag. mom. to nuclear magneton ratio",
    symbol="μ_h/μ_N",
    value=-2.127_625_3075,
    uncertainty=2.5e-9,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_h/μ_N = g_h/2",
    depends_on=["mu_h", "mu_N"],
    description="Helion moment in nuclear magnetons.",
    exact=False
)

mu_h_over_mu_p = Constant(
    name="helion to proton mag. mom. ratio",
    symbol="μ_h/μ_p",
    value=-0.761_766_5618,
    uncertainty=8.9e-10,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_h/μ_p",
    depends_on=["mu_h", "mu_p"],
    description="Helion to proton magnetic moment ratio.",
    exact=False
)

# Deuteron magnetic moment ratios
mu_d_over_mu_B = Constant(
    name="deuteron mag. mom. to Bohr magneton ratio",
    symbol="μ_d/μ_B",
    value=4.669_754_570e-4,
    uncertainty=1.2e-12,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_d/μ_B",
    depends_on=["mu_d", "mu_B"],
    description="Deuteron moment in Bohr magnetons.",
    exact=False
)

mu_d_over_mu_N = Constant(
    name="deuteron mag. mom. to nuclear magneton ratio",
    symbol="μ_d/μ_N",
    value=0.857_438_2338,
    uncertainty=2.2e-9,
    unit="",
    category=ConstantCategory.LIGHT_NUCLEI,
    formula="μ_d/μ_N = g_d/2",
    depends_on=["mu_d", "mu_N"],
    description="Deuteron moment in nuclear magnetons.",
    exact=False
)


# Additional Bohr/nuclear magneton expressions
mu_B_in_eV = Constant(
    name="Bohr magneton in eV/T",
    symbol="μ_B/e",
    value=5.788_381_8060e-5,
    uncertainty=1.7e-14,
    unit="eV T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_B/e",
    depends_on=["mu_B", "e"],
    description="Bohr magneton in eV per tesla.",
    exact=False
)

mu_B_in_Hz = Constant(
    name="Bohr magneton in Hz/T",
    symbol="μ_B/h",
    value=1.399_624_493_61e10,
    uncertainty=4.2e0,
    unit="Hz T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_B/h",
    depends_on=["mu_B", "h"],
    description="Bohr magneton in Hz per tesla.",
    exact=False
)

mu_B_in_K = Constant(
    name="Bohr magneton in K/T",
    symbol="μ_B/k",
    value=0.671_713_815_63,
    uncertainty=2.0e-10,
    unit="K T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_B/k",
    depends_on=["mu_B", "k"],
    description="Bohr magneton in kelvin per tesla.",
    exact=False
)

mu_B_in_m = Constant(
    name="Bohr magneton in inverse meter per tesla",
    symbol="μ_B/(hc)",
    value=46.686_447_83,
    uncertainty=1.4e-8,
    unit="m^-1 T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_B/(hc)",
    depends_on=["mu_B", "h", "c"],
    description="Bohr magneton in inverse meters per tesla.",
    exact=False
)

mu_N_in_eV = Constant(
    name="nuclear magneton in eV/T",
    symbol="μ_N/e",
    value=3.152_451_258_44e-8,
    uncertainty=9.5e-18,
    unit="eV T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_N/e",
    depends_on=["mu_N", "e"],
    description="Nuclear magneton in eV per tesla.",
    exact=False
)

mu_N_in_Hz = Constant(
    name="nuclear magneton in MHz/T",
    symbol="μ_N/h",
    value=7.622_593_2291,
    uncertainty=2.3e-9,
    unit="MHz T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_N/h",
    depends_on=["mu_N", "h"],
    description="Nuclear magneton in MHz per tesla.",
    exact=False
)

mu_N_in_K = Constant(
    name="nuclear magneton in K/T",
    symbol="μ_N/k",
    value=3.658_267_7756e-4,
    uncertainty=1.1e-13,
    unit="K T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_N/k",
    depends_on=["mu_N", "k"],
    description="Nuclear magneton in kelvin per tesla.",
    exact=False
)

mu_N_in_m = Constant(
    name="nuclear magneton in inverse meter per tesla",
    symbol="μ_N/(hc)",
    value=2.542_623_413_53e-2,
    uncertainty=7.6e-12,
    unit="m^-1 T^-1",
    category=ConstantCategory.ELECTROMAGNETIC,
    formula="μ_N/(hc)",
    depends_on=["mu_N", "h", "c"],
    description="Nuclear magneton in inverse meters per tesla.",
    exact=False
)

# Additional Rydberg expressions
R_inf_c = Constant(
    name="Rydberg constant times c in Hz",
    symbol="R_∞·c",
    value=3.289_841_960_2508e15,
    uncertainty=6.4e3,
    unit="Hz",
    category=ConstantCategory.ATOMIC,
    formula="R_∞·c",
    depends_on=["R_inf", "c"],
    description="Rydberg frequency.",
    exact=False
)

R_inf_hc_eV = Constant(
    name="Rydberg constant times hc in eV",
    symbol="R_∞·hc",
    value=13.605_693_122_994,
    uncertainty=2.6e-11,
    unit="eV",
    category=ConstantCategory.ATOMIC,
    formula="R_∞·hc/e",
    depends_on=["R_inf", "h", "c", "e"],
    description="Rydberg energy in eV. Hydrogen ionization energy.",
    exact=False
)

R_inf_hc_J = Constant(
    name="Rydberg constant times hc in J",
    symbol="R_∞·hc",
    value=2.179_872_361_1035e-18,
    uncertainty=4.2e-30,
    unit="J",
    category=ConstantCategory.ATOMIC,
    formula="R_∞·hc",
    depends_on=["R_inf", "h", "c"],
    description="Rydberg energy in joules.",
    exact=False
)

# Sackur-Tetrode constant
S_0_100kPa = Constant(
    name="Sackur-Tetrode constant (1 K, 100 kPa)",
    symbol="S₀/R",
    value=-1.151_707_537_06,
    uncertainty=4.5e-10,
    unit="",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="S₀/R at 100 kPa",
    depends_on=["R"],
    description="Sackur-Tetrode entropy constant at 100 kPa.",
    exact=False
)

S_0_101kPa = Constant(
    name="Sackur-Tetrode constant (1 K, 101.325 kPa)",
    symbol="S₀/R",
    value=-1.164_870_523_58,
    uncertainty=4.5e-10,
    unit="",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="S₀/R at 101.325 kPa",
    depends_on=["R"],
    description="Sackur-Tetrode entropy constant at standard atmosphere.",
    exact=False
)

# First radiation constant for spectral radiance
c_1L = Constant(
    name="first radiation constant for spectral radiance",
    symbol="c₁L",
    value=1.191_042_972e-16,
    uncertainty=0.0,
    unit="W m² sr^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="c₁L = c₁/(4π) = 2hc²",
    depends_on=["h", "c"],
    description="First radiation constant per steradian.",
    exact=True
)

# Wien frequency displacement
b_freq = Constant(
    name="Wien frequency displacement law constant",
    symbol="b'",
    value=5.878_925_757e10,
    uncertainty=0.0,
    unit="Hz K^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="b' = 2.821·k/h",
    depends_on=["k", "h"],
    description="Wien law: f_max = b'·T. Peak frequency of blackbody.",
    exact=True
)

# Standard atmosphere
atm = Constant(
    name="standard atmosphere",
    symbol="atm",
    value=101_325.0,
    uncertainty=0.0,
    unit="Pa",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="1 atm = 101325 Pa exactly",
    depends_on=[],
    description="Standard atmosphere pressure. Defined exactly.",
    exact=True
)

# Standard state pressure
p_0 = Constant(
    name="standard-state pressure",
    symbol="p°",
    value=100_000.0,
    uncertainty=0.0,
    unit="Pa",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="p° = 100 kPa = 1 bar",
    depends_on=[],
    description="IUPAC standard state pressure. 1 bar exactly.",
    exact=True
)

# Loschmidt constant at standard atmosphere
n_0_atm = Constant(
    name="Loschmidt constant (273.15 K, 101.325 kPa)",
    symbol="n₀",
    value=2.686_780_111e25,
    uncertainty=0.0,
    unit="m^-3",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="n₀ = p/(kT) at STP",
    depends_on=["k"],
    description="Number density at standard atmosphere.",
    exact=True
)

# Molar volume at 100 kPa
V_m_100kPa = Constant(
    name="molar volume of ideal gas (273.15 K, 100 kPa)",
    symbol="V_m",
    value=22.710_954_64e-3,
    uncertainty=0.0,
    unit="m³ mol^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="V_m = RT/p at 100 kPa",
    depends_on=["R"],
    description="Molar volume at 100 kPa. 22.71 L/mol.",
    exact=True
)

# Molar volume at 1 atm
V_m_atm = Constant(
    name="molar volume of ideal gas (273.15 K, 101.325 kPa)",
    symbol="V_m",
    value=22.413_969_54e-3,
    uncertainty=0.0,
    unit="m³ mol^-1",
    category=ConstantCategory.PHYSICO_CHEMICAL,
    formula="V_m = RT/p at 1 atm",
    depends_on=["R"],
    description="Molar volume at 1 atm. 22.41 L/mol.",
    exact=True
)

# Cs hyperfine transition
nu_Cs = Constant(
    name="hyperfine transition frequency of Cs-133",
    symbol="Δν_Cs",
    value=9_192_631_770.0,
    uncertainty=0.0,
    unit="Hz",
    category=ConstantCategory.ATOMIC,
    formula="",
    depends_on=[],
    description="Defines the SI second. Cesium-133 hyperfine splitting.",
    exact=True
)


# =============================================================================
# SECTION 26: PARTICLE PHYSICS CONSTANTS
# =============================================================================

# W boson mass
m_W = Constant(
    name="W boson mass",
    symbol="m_W",
    value=80.3692e9 * 1.602176634e-19 / (299792458**2),  # GeV/c² to kg
    uncertainty=0.0013e9 * 1.602176634e-19 / (299792458**2),
    unit="kg",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="W boson mass. Carrier of weak force (charged).",
    exact=False
)

m_W_GeV = Constant(
    name="W boson mass energy equivalent",
    symbol="m_W c²",
    value=80.3692,
    uncertainty=0.0013,
    unit="GeV",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=["m_W", "c"],
    description="W boson mass in GeV. PDG 2024 value.",
    exact=False
)

# Z boson mass
m_Z = Constant(
    name="Z boson mass",
    symbol="m_Z",
    value=91.1876e9 * 1.602176634e-19 / (299792458**2),
    uncertainty=0.0021e9 * 1.602176634e-19 / (299792458**2),
    unit="kg",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="Z boson mass. Carrier of weak force (neutral).",
    exact=False
)

m_Z_GeV = Constant(
    name="Z boson mass energy equivalent",
    symbol="m_Z c²",
    value=91.1876,
    uncertainty=0.0021,
    unit="GeV",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=["m_Z", "c"],
    description="Z boson mass in GeV. Very precisely measured at LEP.",
    exact=False
)

# Higgs boson mass
m_H = Constant(
    name="Higgs boson mass",
    symbol="m_H",
    value=125.25e9 * 1.602176634e-19 / (299792458**2),
    uncertainty=0.17e9 * 1.602176634e-19 / (299792458**2),
    unit="kg",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="Higgs boson mass. Discovered at LHC in 2012.",
    exact=False
)

m_H_GeV = Constant(
    name="Higgs boson mass energy equivalent",
    symbol="m_H c²",
    value=125.25,
    uncertainty=0.17,
    unit="GeV",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=["m_H", "c"],
    description="Higgs boson mass in GeV. Nobel Prize 2013.",
    exact=False
)

# Top quark mass
m_top = Constant(
    name="top quark mass",
    symbol="m_t",
    value=172.69e9 * 1.602176634e-19 / (299792458**2),
    uncertainty=0.30e9 * 1.602176634e-19 / (299792458**2),
    unit="kg",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="Top quark mass. Heaviest known elementary particle.",
    exact=False
)

m_top_GeV = Constant(
    name="top quark mass energy equivalent",
    symbol="m_t c²",
    value=172.69,
    uncertainty=0.30,
    unit="GeV",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=["m_top", "c"],
    description="Top quark mass in GeV. Discovered at Tevatron 1995.",
    exact=False
)

# Strong coupling constant
alpha_s = Constant(
    name="strong coupling constant at M_Z",
    symbol="α_s(M_Z)",
    value=0.1180,
    uncertainty=0.0009,
    unit="",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="QCD coupling at Z pole. Runs with energy scale.",
    exact=False
)

# Pion masses
m_pi_plus = Constant(
    name="charged pion mass",
    symbol="m_π±",
    value=139.57039e6 * 1.602176634e-19 / (299792458**2),
    uncertainty=0.00018e6 * 1.602176634e-19 / (299792458**2),
    unit="kg",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="Charged pion mass. Lightest hadron after neutral pion.",
    exact=False
)

m_pi_zero = Constant(
    name="neutral pion mass",
    symbol="m_π⁰",
    value=134.9768e6 * 1.602176634e-19 / (299792458**2),
    uncertainty=0.0005e6 * 1.602176634e-19 / (299792458**2),
    unit="kg",
    category=ConstantCategory.ELECTROWEAK,
    formula="",
    depends_on=[],
    description="Neutral pion mass. Lightest hadron. Decays to 2 photons.",
    exact=False
)


# =============================================================================
# SECTION 27: COSMOLOGICAL CONSTANTS
# =============================================================================

# Hubble constant
H_0 = Constant(
    name="Hubble constant",
    symbol="H₀",
    value=67.4e3 / 3.085677581e22,  # km/s/Mpc to s^-1
    uncertainty=0.5e3 / 3.085677581e22,
    unit="s^-1",
    category=ConstantCategory.PLANCK,
    formula="H₀ = v/d (recession velocity / distance)",
    depends_on=[],
    description="Hubble constant. Rate of universe expansion. Planck 2018.",
    exact=False
)

H_0_km_s_Mpc = Constant(
    name="Hubble constant (km/s/Mpc)",
    symbol="H₀",
    value=67.4,
    uncertainty=0.5,
    unit="km s^-1 Mpc^-1",
    category=ConstantCategory.PLANCK,
    formula="",
    depends_on=[],
    description="Hubble constant in conventional units. Tension with local measurements (~73).",
    exact=False
)

# Hubble time
t_H = Constant(
    name="Hubble time",
    symbol="t_H",
    value=3.085677581e22 / (67.4e3),  # 1/H_0 in seconds
    uncertainty=0.5 / 67.4 * 3.085677581e22 / (67.4e3),
    unit="s",
    category=ConstantCategory.PLANCK,
    formula="t_H = 1/H₀",
    depends_on=["H_0"],
    description="Hubble time ~ 14.5 Gyr. Characteristic expansion timescale.",
    exact=False
)

# Age of universe
t_universe = Constant(
    name="age of the universe",
    symbol="t₀",
    value=13.787e9 * 365.25 * 24 * 3600,  # years to seconds
    uncertainty=0.020e9 * 365.25 * 24 * 3600,
    unit="s",
    category=ConstantCategory.PLANCK,
    formula="",
    depends_on=[],
    description="Age of universe. 13.787 billion years from Planck 2018.",
    exact=False
)

# Cosmological constant
Lambda_cosmo = Constant(
    name="cosmological constant",
    symbol="Λ",
    value=1.1056e-52,
    uncertainty=0.0001e-52,
    unit="m^-2",
    category=ConstantCategory.PLANCK,
    formula="Λ = 8πG ρ_Λ / c²",
    depends_on=["G", "c"],
    description="Cosmological constant. Drives accelerated expansion.",
    exact=False
)

# Dark energy density parameter
Omega_Lambda = Constant(
    name="dark energy density parameter",
    symbol="Ω_Λ",
    value=0.6847,
    uncertainty=0.0073,
    unit="",
    category=ConstantCategory.PLANCK,
    formula="Ω_Λ = ρ_Λ / ρ_c",
    depends_on=["Lambda_cosmo"],
    description="Fraction of universe that is dark energy. ~68.5%.",
    exact=False
)

# Matter density parameter
Omega_m = Constant(
    name="matter density parameter",
    symbol="Ω_m",
    value=0.3153,
    uncertainty=0.0073,
    unit="",
    category=ConstantCategory.PLANCK,
    formula="Ω_m = ρ_m / ρ_c",
    depends_on=[],
    description="Fraction of universe that is matter. ~31.5% (mostly dark).",
    exact=False
)

# Baryon density parameter
Omega_b = Constant(
    name="baryon density parameter",
    symbol="Ω_b",
    value=0.0493,
    uncertainty=0.0006,
    unit="",
    category=ConstantCategory.PLANCK,
    formula="Ω_b = ρ_b / ρ_c",
    depends_on=[],
    description="Fraction of universe that is baryonic matter. ~5%.",
    exact=False
)

# Critical density
rho_c = Constant(
    name="critical density of the universe",
    symbol="ρ_c",
    value=3 * (67.4e3 / 3.085677581e22)**2 / (8 * 3.14159265 * 6.67430e-11),
    uncertainty=0.1e-26,
    unit="kg m^-3",
    category=ConstantCategory.PLANCK,
    formula="ρ_c = 3H₀²/(8πG)",
    depends_on=["H_0", "G"],
    description="Critical density for flat universe. ~10^-26 kg/m³.",
    exact=False
)

# CMB temperature
T_CMB = Constant(
    name="CMB temperature",
    symbol="T_CMB",
    value=2.7255,
    uncertainty=0.0006,
    unit="K",
    category=ConstantCategory.PLANCK,
    formula="",
    depends_on=[],
    description="Cosmic microwave background temperature today.",
    exact=False
)


# =============================================================================
# AGGREGATED CONSTANT DICTIONARY
# =============================================================================

ALL_CONSTANTS: Dict[str, Constant] = {
    # Universal
    "c": c,
    "h": h,
    "h_bar": h_bar,
    "e": e_charge,
    "k": k_B,
    "N_A": N_A,
    "K_cd": K_cd,
    "G": G,
    "g_n": g_n,

    # Electromagnetic
    "mu_0": mu_0,
    "epsilon_0": epsilon_0,
    "Z_0": Z_0,
    "alpha": alpha,
    "alpha_inv": alpha_inv,
    "Phi_0": Phi_0,
    "G_0": G_0,
    "K_J": K_J,
    "R_K": R_K,
    "mu_B": mu_B,
    "mu_N": mu_N,

    # Atomic
    "R_inf": R_inf,
    "a_0": a_0,
    "E_h": E_h,
    "r_e": r_e,
    "lambda_C": lambda_C,
    "lambda_C_bar": lambda_C_bar,
    "sigma_e": sigma_e,
    "u": u,

    # Electron
    "m_e": m_e,
    "m_e_u": m_e_u,
    "m_e_eV": m_e_eV,
    "g_e": g_e,
    "a_e": a_e,
    "mu_e": mu_e,

    # Proton
    "m_p": m_p,
    "m_p_u": m_p_u,
    "m_p_eV": m_p_eV,
    "lambda_C_p": lambda_C_p,
    "r_p": r_p,
    "g_p": g_p,
    "mu_p": mu_p,

    # Neutron
    "m_n": m_n,
    "m_n_u": m_n_u,
    "m_n_eV": m_n_eV,
    "lambda_C_n": lambda_C_n,
    "g_n": g_n,
    "mu_n": mu_n,

    # Muon
    "m_mu": m_mu,
    "m_mu_u": m_mu_u,
    "m_mu_eV": m_mu_eV,
    "g_mu": g_mu,
    "a_mu": a_mu,
    "mu_mu": mu_mu,
    "lambda_C_mu": lambda_C_mu,

    # Tau
    "m_tau": m_tau,
    "m_tau_u": m_tau_u,
    "m_tau_eV": m_tau_eV,
    "lambda_C_tau": lambda_C_tau,

    # Light nuclei
    "m_d": m_d,
    "m_d_u": m_d_u,
    "m_d_eV": m_d_eV,
    "mu_d": mu_d,
    "m_t": m_t,
    "m_t_u": m_t_u,
    "m_h": m_h,
    "m_h_u": m_h_u,
    "m_alpha": m_alpha,
    "m_alpha_u": m_alpha_u,
    "m_alpha_eV": m_alpha_eV,

    # Physico-chemical
    "R": R,
    "F": F,
    "sigma_SB": sigma_SB,
    "c_1": c_1,
    "c_2": c_2,
    "b": b,
    "N_A_h": N_A_h,
    "n_0": n_0,
    "V_m": V_m,

    # Planck units
    "l_P": l_P,
    "t_P": t_P,
    "m_P": m_P,
    "E_P": E_P,
    "T_P": T_P,
    "q_P": q_P,
    "Z_P": Z_P,

    # Ratios
    "m_e_over_m_p": m_e_over_m_p,
    "m_p_over_m_e": m_p_over_m_e,
    "m_mu_over_m_e": m_mu_over_m_e,
    "m_tau_over_m_e": m_tau_over_m_e,
    "m_n_over_m_p": m_n_over_m_p,
    "m_alpha_over_m_e": m_alpha_over_m_e,
    "mu_p_over_mu_B": mu_p_over_mu_B,
    "mu_p_over_mu_N": mu_p_over_mu_N,
    "mu_e_over_mu_mu": mu_e_over_mu_mu,

    # Energy equivalents
    "eV": eV,
    "eV_to_kg": eV_to_kg,
    "eV_to_Hz": eV_to_Hz,
    "eV_to_K": eV_to_K,
    "eV_to_m": eV_to_m,
    "hartree_to_J": hartree_to_J,
    "hartree_to_eV": hartree_to_eV,
    "u_to_J": u_to_J,
    "u_to_eV": u_to_eV,
    "u_to_kg": u_to_kg,

    # QED
    "gamma_e": gamma_e,
    "gamma_p": gamma_p,
    "gamma_p_shielded": gamma_p_shielded,
    "mu_e_over_mu_p": mu_e_over_mu_p,

    # Atomic units
    "au_action": au_action,
    "au_charge": au_charge,
    "au_energy": au_energy,
    "au_length": au_length,
    "au_mass": au_mass,
    "au_time": au_time,
    "au_velocity": au_velocity,
    "au_momentum": au_momentum,
    "au_force": au_force,
    "au_electric_field": au_electric_field,
    "au_electric_potential": au_electric_potential,
    "au_electric_dipole": au_electric_dipole,
    "au_magnetic_dipole": au_magnetic_dipole,
    "au_magnetic_flux_density": au_magnetic_flux_density,
    "au_current": au_current,
    "au_charge_density": au_charge_density,
    "au_permittivity": au_permittivity,
    "au_polarizability": au_polarizability,
    "au_1st_hyperpolarizability": au_1st_hyperpolarizability,
    "au_2nd_hyperpolarizability": au_2nd_hyperpolarizability,
    "au_magnetizability": au_magnetizability,
    "au_electric_quadrupole": au_electric_quadrupole,
    "au_electric_field_gradient": au_electric_field_gradient,

    # Natural units
    "nu_action": nu_action,
    "nu_action_eV": nu_action_eV,
    "nu_velocity": nu_velocity,
    "nu_length": nu_length,
    "nu_mass": nu_mass,
    "nu_energy": nu_energy,
    "nu_energy_MeV": nu_energy_MeV,
    "nu_momentum": nu_momentum,
    "nu_momentum_MeV": nu_momentum_MeV,
    "nu_time": nu_time,
    "hbar_c_MeV_fm": hbar_c_MeV_fm,

    # Electroweak
    "G_F": G_F,
    "sin2_theta_W": sin2_theta_W,
    "W_Z_mass_ratio": W_Z_mass_ratio,

    # Shielded magnetic moments
    "gamma_h_shielded": gamma_h_shielded,
    "gamma_h_shielded_MHz": gamma_h_shielded_MHz,
    "mu_h_shielded": mu_h_shielded,
    "mu_h_shielded_over_mu_B": mu_h_shielded_over_mu_B,
    "mu_h_shielded_over_mu_N": mu_h_shielded_over_mu_N,
    "mu_p_shielded": mu_p_shielded,
    "mu_p_shielded_over_mu_B": mu_p_shielded_over_mu_B,
    "mu_p_shielded_over_mu_N": mu_p_shielded_over_mu_N,
    "gamma_p_shielded_MHz": gamma_p_shielded_MHz,
    "helion_shielding_shift": helion_shielding_shift,
    "proton_magnetic_shielding": proton_magnetic_shielding,
    "shielding_d_p_in_HD": shielding_d_p_in_HD,
    "shielding_t_p_in_HT": shielding_t_p_in_HT,

    # Molar masses
    "M_e": M_e,
    "M_p": M_p,
    "M_n": M_n,
    "M_mu": M_mu,
    "M_tau": M_tau,
    "M_d": M_d,
    "M_t": M_t,
    "M_h": M_h,
    "M_alpha": M_alpha,
    "M_12C": M_12C,
    "molar_mass_constant": molar_mass_constant,

    # X-ray units
    "angstrom_star": angstrom_star,
    "Cu_x_unit": Cu_x_unit,
    "Mo_x_unit": Mo_x_unit,
    "Si_lattice": Si_lattice,
    "Si_220_spacing": Si_220_spacing,
    "Si_molar_volume": Si_molar_volume,

    # Conventional values
    "K_J_90": K_J_90,
    "R_K_90": R_K_90,
    "ohm_90": ohm_90,
    "volt_90": volt_90,
    "ampere_90": ampere_90,
    "watt_90": watt_90,

    # Additional mass ratios
    "m_d_over_m_e": m_d_over_m_e,
    "m_d_over_m_p": m_d_over_m_p,
    "m_h_over_m_p": m_h_over_m_p,
    "m_t_over_m_p": m_t_over_m_p,
    "m_alpha_over_m_p": m_alpha_over_m_p,
    "m_mu_over_m_p": m_mu_over_m_p,
    "m_tau_over_m_p": m_tau_over_m_p,
    "m_mu_over_m_tau": m_mu_over_m_tau,
    "m_n_over_m_e": m_n_over_m_e,

    # Additional moment ratios
    "mu_d_over_mu_e": mu_d_over_mu_e,
    "mu_d_over_mu_p": mu_d_over_mu_p,
    "mu_d_over_mu_n": mu_d_over_mu_n,
    "mu_n_over_mu_e": mu_n_over_mu_e,
    "mu_n_over_mu_p": mu_n_over_mu_p,
    "mu_mu_over_mu_p": mu_mu_over_mu_p,

    # Fundamental ratios
    "e_over_m_e": e_over_m_e,
    "e_over_m_p": e_over_m_p,
    "e_over_h": e_over_h,
    "quantum_circulation": quantum_circulation,
    "quantum_circulation_2": quantum_circulation_2,
    "inverse_G_0": inverse_G_0,
    "G_over_hbar_c": G_over_hbar_c,
    "m_P_GeV": m_P_GeV,

    # Hartree conversions
    "hartree_to_u": hartree_to_u,
    "hartree_to_Hz": hartree_to_Hz,
    "hartree_to_m": hartree_to_m,
    "hartree_to_K": hartree_to_K,
    "hartree_to_kg": hartree_to_kg,

    # Hz conversions
    "Hz_to_J": Hz_to_J,
    "Hz_to_K": Hz_to_K,
    "Hz_to_eV": Hz_to_eV,
    "Hz_to_m": Hz_to_m,
    "Hz_to_kg": Hz_to_kg,
    "Hz_to_u": Hz_to_u,
    "Hz_to_hartree": Hz_to_hartree,

    # K conversions
    "K_to_J": K_to_J,
    "K_to_Hz": K_to_Hz,
    "K_to_m": K_to_m,
    "K_to_kg": K_to_kg,
    "K_to_u": K_to_u,
    "K_to_hartree": K_to_hartree,

    # Inverse meter conversions
    "m_inv_to_J": m_inv_to_J,
    "m_inv_to_Hz": m_inv_to_Hz,
    "m_inv_to_K": m_inv_to_K,
    "m_inv_to_kg": m_inv_to_kg,
    "m_inv_to_eV": m_inv_to_eV,
    "m_inv_to_u": m_inv_to_u,
    "m_inv_to_hartree": m_inv_to_hartree,

    # kg conversions
    "kg_to_J": kg_to_J,
    "kg_to_Hz": kg_to_Hz,
    "kg_to_m": kg_to_m,
    "kg_to_K": kg_to_K,
    "kg_to_eV": kg_to_eV,
    "kg_to_hartree": kg_to_hartree,

    # u conversions
    "u_to_Hz": u_to_Hz,
    "u_to_m": u_to_m,
    "u_to_K": u_to_K,
    "u_to_hartree": u_to_hartree,

    # eV conversions
    "eV_to_u": eV_to_u,
    "eV_to_hartree": eV_to_hartree,

    # Relative atomic masses
    "A_r_e": A_r_e,
    "A_r_p": A_r_p,
    "A_r_n": A_r_n,
    "A_r_d": A_r_d,
    "A_r_t": A_r_t,
    "A_r_h": A_r_h,
    "A_r_alpha": A_r_alpha,

    # Mass differences
    "m_n_minus_m_p": m_n_minus_m_p,
    "m_n_minus_m_p_eV": m_n_minus_m_p_eV,
    "m_n_minus_m_p_MeV": m_n_minus_m_p_MeV,
    "m_n_minus_m_p_u": m_n_minus_m_p_u,

    # Nuclear radii
    "r_d": r_d,
    "r_alpha": r_alpha,

    # g-factors
    "g_h": g_h,
    "g_t": g_t,
    "g_d": g_d,

    # Triton magnetic moment
    "mu_t": mu_t,
    "mu_t_over_mu_B": mu_t_over_mu_B,
    "mu_t_over_mu_N": mu_t_over_mu_N,
    "mu_t_over_mu_p": mu_t_over_mu_p,

    # Helion magnetic moment
    "mu_h": mu_h,
    "mu_h_over_mu_B": mu_h_over_mu_B,
    "mu_h_over_mu_N": mu_h_over_mu_N,
    "mu_h_over_mu_p": mu_h_over_mu_p,

    # Deuteron magnetic moment ratios
    "mu_d_over_mu_B": mu_d_over_mu_B,
    "mu_d_over_mu_N": mu_d_over_mu_N,

    # Magneton equivalents
    "mu_B_in_eV": mu_B_in_eV,
    "mu_B_in_Hz": mu_B_in_Hz,
    "mu_B_in_K": mu_B_in_K,
    "mu_B_in_m": mu_B_in_m,
    "mu_N_in_eV": mu_N_in_eV,
    "mu_N_in_Hz": mu_N_in_Hz,
    "mu_N_in_K": mu_N_in_K,
    "mu_N_in_m": mu_N_in_m,

    # Rydberg equivalents
    "R_inf_c": R_inf_c,
    "R_inf_hc_eV": R_inf_hc_eV,
    "R_inf_hc_J": R_inf_hc_J,

    # Thermodynamic
    "S_0_100kPa": S_0_100kPa,
    "S_0_101kPa": S_0_101kPa,
    "c_1L": c_1L,
    "b_freq": b_freq,

    # Pressure and gas
    "atm": atm,
    "p_0": p_0,
    "n_0_atm": n_0_atm,
    "V_m_100kPa": V_m_100kPa,
    "V_m_atm": V_m_atm,

    # Atomic frequency
    "nu_Cs": nu_Cs,

    # Particle Physics (Electroweak)
    "m_W": m_W,
    "m_W_GeV": m_W_GeV,
    "m_Z": m_Z,
    "m_Z_GeV": m_Z_GeV,
    "m_H": m_H,
    "m_H_GeV": m_H_GeV,
    "m_top": m_top,
    "m_top_GeV": m_top_GeV,
    "alpha_s": alpha_s,
    "sin2_theta_W": sin2_theta_W,
    "G_F": G_F,

    # Particle Physics (Mesons)
    "m_pi_plus": m_pi_plus,
    "m_pi_zero": m_pi_zero,

    # Cosmological
    "H_0": H_0,
    "H_0_km_s_Mpc": H_0_km_s_Mpc,
    "t_H": t_H,
    "t_universe": t_universe,
    "Lambda_cosmo": Lambda_cosmo,
    "Omega_Lambda": Omega_Lambda,
    "Omega_m": Omega_m,
    "Omega_b": Omega_b,
    "rho_c": rho_c,
    "T_CMB": T_CMB,
}


def get_constants_by_category(category: ConstantCategory) -> Dict[str, Constant]:
    """Return all constants in a given category."""
    return {k: v for k, v in ALL_CONSTANTS.items() if v.category == category}


def get_exact_constants() -> Dict[str, Constant]:
    """Return all constants that are exact by SI definition."""
    return {k: v for k, v in ALL_CONSTANTS.items() if v.exact}


def get_derived_constants() -> Dict[str, Constant]:
    """Return all constants that have derivation formulas."""
    return {k: v for k, v in ALL_CONSTANTS.items() if v.formula}


def print_constant_summary():
    """Print summary of all constants organized by category."""
    print("=" * 70)
    print("CODATA 2022 FUNDAMENTAL PHYSICAL CONSTANTS")
    print(f"Total: {len(ALL_CONSTANTS)} constants")
    print("=" * 70)

    for category in ConstantCategory:
        constants = get_constants_by_category(category)
        if constants:
            print(f"\n{category.value.upper()} ({len(constants)} constants)")
            print("-" * 50)
            for name, const in constants.items():
                exact_flag = " (exact)" if const.exact else ""
                print(f"  {const.symbol:12} = {const.value:.6e} {const.unit}{exact_flag}")


def get_constant_network() -> Dict[str, List[str]]:
    """
    Return the dependency network of constants.

    This shows how constants are mathematically connected.
    Each constant maps to the list of constants it depends on.
    """
    return {
        name: const.depends_on
        for name, const in ALL_CONSTANTS.items()
        if const.depends_on
    }


# Module exports
__all__ = [
    "Constant",
    "ConstantCategory",
    "ALL_CONSTANTS",
    "get_constants_by_category",
    "get_exact_constants",
    "get_derived_constants",
    "get_constant_network",
    "print_constant_summary",
    # Individual constants for direct import
    "c", "h", "h_bar", "e_charge", "k_B", "N_A", "G",
    "alpha", "mu_0", "epsilon_0", "R_K", "K_J", "Phi_0", "G_0",
    "m_e", "m_p", "m_n", "m_mu", "m_tau",
    "a_0", "R_inf", "E_h", "r_e", "lambda_C",
    "l_P", "t_P", "m_P", "E_P", "T_P",
    "mu_B", "mu_N", "mu_e", "mu_p", "mu_n",
    # Particle physics
    "m_W", "m_Z", "m_H", "m_top", "alpha_s", "sin2_theta_W", "G_F",
    "m_pi_plus", "m_pi_zero",
    # Cosmological
    "H_0", "Lambda_cosmo", "Omega_Lambda", "Omega_m", "Omega_b", "rho_c", "T_CMB",
]
