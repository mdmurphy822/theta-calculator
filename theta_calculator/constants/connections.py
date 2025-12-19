"""
Mathematical Connections Between Physical Constants

This module defines the network of mathematical relationships that connect
all fundamental physical constants. These connections demonstrate that
constants are not independent values but form a self-consistent system.

Key Insight from Arxiv Research:
The interconnection of constants proves that theta (the quantum-classical
interpolation parameter) emerges naturally from physics itself. Constants
define boundary conditions where different descriptions apply.

Connection Types:
1. EXACT - Mathematical identities (e.g., h_bar = h/2pi)
2. DERIVED - Computed from more fundamental constants
3. BOOTSTRAP - Circular relationships proving consistency
4. THETA_BOUNDARY - Defines quantum-classical transition scale

Reference: Arxiv papers on "Fundamental Constants in Physics and Their Time Variation"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple
from enum import Enum

from .codata_2022 import ALL_CONSTANTS, Constant


class ConnectionType(Enum):
    """Types of mathematical connections between constants."""
    EXACT = "exact"           # Definitional identity
    DERIVED = "derived"       # Computed from others
    BOOTSTRAP = "bootstrap"   # Circular verification
    THETA_BOUNDARY = "theta"  # Defines quantum-classical scale
    RATIO = "ratio"           # Dimensionless ratio
    CONVERSION = "conversion" # Unit conversion


@dataclass
class ConstantConnection:
    """
    A mathematical relationship between constants.

    Attributes:
        name: Name of this relationship
        formula: Human-readable formula
        latex: LaTeX formula
        inputs: List of input constant names
        output: Output constant name
        compute: Function to compute output from inputs
        connection_type: Type of connection
        description: Physical meaning
        theta_relevance: How this relates to quantum-classical boundary
    """
    name: str
    formula: str
    latex: str
    inputs: List[str]
    output: str
    compute: Callable[..., float]
    connection_type: ConnectionType
    description: str
    theta_relevance: str = ""

    def verify(self, tolerance: float = 1e-9) -> Tuple[bool, float]:
        """
        Verify this connection holds within tolerance.

        Returns:
            Tuple of (is_valid, relative_error)
        """
        try:
            input_values = [ALL_CONSTANTS[name].value for name in self.inputs]
            computed = self.compute(*input_values)
            expected = ALL_CONSTANTS[self.output].value

            if expected == 0:
                error = abs(computed)
                return error < tolerance, error

            error = abs(computed - expected) / abs(expected)
            return error < tolerance, error
        except KeyError as e:
            return False, float('inf')


# =============================================================================
# TIER 1: SI DEFINING RELATIONSHIPS (Exact by definition)
# =============================================================================

connection_h_bar_from_h = ConstantConnection(
    name="Reduced Planck from Planck",
    formula="h_bar = h / (2*pi)",
    latex=r"\hbar = \frac{h}{2\pi}",
    inputs=["h"],
    output="h_bar",
    compute=lambda h: h / (2 * np.pi),
    connection_type=ConnectionType.EXACT,
    description="Reduced Planck constant from Planck constant.",
    theta_relevance="h_bar is THE quantum scale. Action S ~ h_bar means theta -> 1"
)

connection_R_from_N_A_k = ConstantConnection(
    name="Molar gas constant",
    formula="R = N_A * k",
    latex=r"R = N_A \cdot k",
    inputs=["N_A", "k"],
    output="R",
    compute=lambda N_A, k: N_A * k,
    connection_type=ConnectionType.EXACT,
    description="Molar gas constant bridges atomic (k) and molar (R) scales.",
    theta_relevance="At N ~ N_A particles, quantum fluctuations average out (theta -> 0)"
)

connection_F_from_N_A_e = ConstantConnection(
    name="Faraday constant",
    formula="F = N_A * e",
    latex=r"F = N_A \cdot e",
    inputs=["N_A", "e"],
    output="F",
    compute=lambda N_A, e: N_A * e,
    connection_type=ConnectionType.EXACT,
    description="Faraday constant is charge per mole.",
    theta_relevance="Electrochemistry is macroscopic limit of quantum charge"
)


# =============================================================================
# TIER 2: ELECTROMAGNETIC RELATIONSHIPS
# =============================================================================

connection_c_from_epsilon_mu = ConstantConnection(
    name="Speed of light from EM constants",
    formula="c = 1 / sqrt(epsilon_0 * mu_0)",
    latex=r"c = \frac{1}{\sqrt{\epsilon_0 \mu_0}}",
    inputs=["epsilon_0", "mu_0"],
    output="c",
    compute=lambda eps, mu: 1.0 / np.sqrt(eps * mu),
    connection_type=ConnectionType.BOOTSTRAP,
    description="Speed of light emerges from vacuum electromagnetic properties. "
                "This shows c is not independent but determined by spacetime structure.",
    theta_relevance="c defines the causality boundary. Relativistic (v~c) effects modify theta"
)

connection_alpha_from_fundamentals = ConstantConnection(
    name="Fine structure constant",
    formula="alpha = e^2 / (4*pi*epsilon_0*h_bar*c)",
    latex=r"\alpha = \frac{e^2}{4\pi\epsilon_0\hbar c}",
    inputs=["e", "epsilon_0", "h_bar", "c"],
    output="alpha",
    compute=lambda e, eps, hbar, c: e**2 / (4 * np.pi * eps * hbar * c),
    connection_type=ConnectionType.DERIVED,
    description="THE fundamental dimensionless constant. alpha ~ 1/137 governs "
                "all electromagnetic interactions. Origin unknown - key mystery!",
    theta_relevance="alpha determines atomic structure. Systems with E ~ alpha^2 * m_e * c^2 "
                    "are at the quantum-classical boundary for atoms"
)

connection_Z_0_from_mu_c = ConstantConnection(
    name="Vacuum impedance",
    formula="Z_0 = mu_0 * c",
    latex=r"Z_0 = \mu_0 c",
    inputs=["mu_0", "c"],
    output="Z_0",
    compute=lambda mu, c: mu * c,
    connection_type=ConnectionType.DERIVED,
    description="Impedance of free space. Z_0 = sqrt(mu_0/epsilon_0) ~ 377 ohms.",
    theta_relevance="Z_0 appears in quantum electrodynamics vacuum fluctuations"
)

connection_R_K_from_h_e = ConstantConnection(
    name="von Klitzing constant",
    formula="R_K = h / e^2",
    latex=r"R_K = \frac{h}{e^2}",
    inputs=["h", "e"],
    output="R_K",
    compute=lambda h, e: h / e**2,
    connection_type=ConnectionType.EXACT,
    description="Quantum of resistance. Quantized Hall resistance R_H = R_K/n. "
                "Nobel Prize 1985 for quantum Hall effect.",
    theta_relevance="R_K quantization is macroscopic quantum effect (theta ~ 1 at 2D electron gas)"
)

connection_K_J_from_e_h = ConstantConnection(
    name="Josephson constant",
    formula="K_J = 2*e / h",
    latex=r"K_J = \frac{2e}{h}",
    inputs=["e", "h"],
    output="K_J",
    compute=lambda e, h: 2 * e / h,
    connection_type=ConnectionType.EXACT,
    description="Josephson frequency-voltage ratio. V = hf/(2e) in superconductors.",
    theta_relevance="Josephson effect is macroscopic quantum coherence (theta ~ 1)"
)

connection_Phi_0_from_h_e = ConstantConnection(
    name="Magnetic flux quantum",
    formula="Phi_0 = h / (2*e)",
    latex=r"\Phi_0 = \frac{h}{2e}",
    inputs=["h", "e"],
    output="Phi_0",
    compute=lambda h, e: h / (2 * e),
    connection_type=ConnectionType.EXACT,
    description="Quantum of magnetic flux in superconductors. Basis of SQUIDs.",
    theta_relevance="Flux quantization demonstrates quantum coherence at macro scale"
)

connection_G_0_from_e_h = ConstantConnection(
    name="Conductance quantum",
    formula="G_0 = 2*e^2 / h",
    latex=r"G_0 = \frac{2e^2}{h}",
    inputs=["e", "h"],
    output="G_0",
    compute=lambda e, h: 2 * e**2 / h,
    connection_type=ConnectionType.EXACT,
    description="Quantum of conductance. Conductance of ideal 1D channel.",
    theta_relevance="Quantized conductance in nanoscale systems (theta ~ 1)"
)

connection_mu_B_from_e_hbar_me = ConstantConnection(
    name="Bohr magneton",
    formula="mu_B = e * h_bar / (2 * m_e)",
    latex=r"\mu_B = \frac{e\hbar}{2m_e}",
    inputs=["e", "h_bar", "m_e"],
    output="mu_B",
    compute=lambda e, hbar, me: e * hbar / (2 * me),
    connection_type=ConnectionType.DERIVED,
    description="Natural unit of electron magnetic moment. Dirac predicts mu_e = mu_B.",
    theta_relevance="mu_B sets scale for magnetic quantum effects in atoms"
)

connection_mu_N_from_e_hbar_mp = ConstantConnection(
    name="Nuclear magneton",
    formula="mu_N = e * h_bar / (2 * m_p)",
    latex=r"\mu_N = \frac{e\hbar}{2m_p}",
    inputs=["e", "h_bar", "m_p"],
    output="mu_N",
    compute=lambda e, hbar, mp: e * hbar / (2 * mp),
    connection_type=ConnectionType.DERIVED,
    description="Natural unit of nuclear magnetic moment. mu_N = mu_B * m_e/m_p.",
    theta_relevance="Nuclear magnetic moments measured in mu_N determine NMR/MRI"
)


# =============================================================================
# TIER 3: ATOMIC SCALE RELATIONSHIPS
# =============================================================================

connection_a_0_from_fundamentals = ConstantConnection(
    name="Bohr radius",
    formula="a_0 = h_bar / (m_e * c * alpha)",
    latex=r"a_0 = \frac{\hbar}{m_e c \alpha}",
    inputs=["h_bar", "m_e", "c", "alpha"],
    output="a_0",
    compute=lambda hbar, me, c, alpha: hbar / (me * c * alpha),
    connection_type=ConnectionType.DERIVED,
    description="Characteristic atomic length. Ground state H radius. "
                "a_0 = 5.29e-11 m sets the scale for chemistry.",
    theta_relevance="At length L ~ a_0, atomic quantum effects dominate (theta ~ 1). "
                    "Classical limit when L >> a_0"
)

connection_R_inf_from_fundamentals = ConstantConnection(
    name="Rydberg constant",
    formula="R_inf = alpha^2 * m_e * c / (2 * h)",
    latex=r"R_\infty = \frac{\alpha^2 m_e c}{2h}",
    inputs=["alpha", "m_e", "c", "h"],
    output="R_inf",
    compute=lambda alpha, me, c, h: alpha**2 * me * c / (2 * h),
    connection_type=ConnectionType.DERIVED,
    description="Fundamental constant of atomic spectroscopy. Most precisely "
                "measured constant. E_n = -R_inf * hc / n^2 for hydrogen.",
    theta_relevance="Rydberg sets energy scale for atomic quantum mechanics"
)

connection_E_h_from_alpha_me_c = ConstantConnection(
    name="Hartree energy",
    formula="E_h = alpha^2 * m_e * c^2",
    latex=r"E_h = \alpha^2 m_e c^2",
    inputs=["alpha", "m_e", "c"],
    output="E_h",
    compute=lambda alpha, me, c: alpha**2 * me * c**2,
    connection_type=ConnectionType.DERIVED,
    description="Natural energy unit for quantum chemistry. E_h ~ 27.2 eV. "
                "Twice the hydrogen ionization energy.",
    theta_relevance="E_h is the characteristic energy for atomic binding"
)

connection_lambda_C_from_h_me_c = ConstantConnection(
    name="Electron Compton wavelength",
    formula="lambda_C = h / (m_e * c)",
    latex=r"\lambda_C = \frac{h}{m_e c}",
    inputs=["h", "m_e", "c"],
    output="lambda_C",
    compute=lambda h, me, c: h / (me * c),
    connection_type=ConnectionType.DERIVED,
    description="Wavelength of photon with energy = m_e c^2. Below lambda_C, "
                "pair creation possible. Defines relativistic QM scale.",
    theta_relevance="At wavelengths ~ lambda_C, pair creation becomes possible (theta = 1)"
)

connection_r_e_from_alpha_a_0 = ConstantConnection(
    name="Classical electron radius",
    formula="r_e = alpha^2 * a_0",
    latex=r"r_e = \alpha^2 a_0",
    inputs=["alpha", "a_0"],
    output="r_e",
    compute=lambda alpha, a0: alpha**2 * a0,
    connection_type=ConnectionType.DERIVED,
    description="Length where classical EM energy equals rest mass. "
                "r_e << lambda_C << a_0 shows scale hierarchy.",
    theta_relevance="r_e is deep quantum scale (theta = 1)"
)

connection_sigma_e_from_r_e = ConstantConnection(
    name="Thomson cross-section",
    formula="sigma_e = 8 * pi * r_e^2 / 3",
    latex=r"\sigma_e = \frac{8\pi r_e^2}{3}",
    inputs=["r_e"],
    output="sigma_e",
    compute=lambda re: 8 * np.pi * re**2 / 3,
    connection_type=ConnectionType.DERIVED,
    description="Low-energy photon-electron scattering. Classical limit of Compton.",
    theta_relevance="Thomson scattering is classical limit (theta ~ 0) of Compton"
)


# =============================================================================
# TIER 4: PLANCK SCALE RELATIONSHIPS
# =============================================================================

connection_l_P_from_hbar_G_c = ConstantConnection(
    name="Planck length",
    formula="l_P = sqrt(h_bar * G / c^3)",
    latex=r"l_P = \sqrt{\frac{\hbar G}{c^3}}",
    inputs=["h_bar", "G", "c"],
    output="l_P",
    compute=lambda hbar, G, c: np.sqrt(hbar * G / c**3),
    connection_type=ConnectionType.DERIVED,
    description="Smallest meaningful length. At l_P, quantum gravity dominates. "
                "l_P ~ 1.6e-35 m. Below this, spacetime may be discrete.",
    theta_relevance="At L ~ l_P, theta = 1 (fully quantum gravity). "
                    "Theta = l_P / L gives scale-based quantumness"
)

connection_t_P_from_hbar_G_c = ConstantConnection(
    name="Planck time",
    formula="t_P = sqrt(h_bar * G / c^5)",
    latex=r"t_P = \sqrt{\frac{\hbar G}{c^5}}",
    inputs=["h_bar", "G", "c"],
    output="t_P",
    compute=lambda hbar, G, c: np.sqrt(hbar * G / c**5),
    connection_type=ConnectionType.DERIVED,
    description="Smallest meaningful time. t_P = l_P / c ~ 5.4e-44 s. "
                "Universe age ~ 10^61 Planck times.",
    theta_relevance="Planck era of universe had theta ~ 1"
)

connection_m_P_from_hbar_c_G = ConstantConnection(
    name="Planck mass",
    formula="m_P = sqrt(h_bar * c / G)",
    latex=r"m_P = \sqrt{\frac{\hbar c}{G}}",
    inputs=["h_bar", "c", "G"],
    output="m_P",
    compute=lambda hbar, c, G: np.sqrt(hbar * c / G),
    connection_type=ConnectionType.DERIVED,
    description="Mass where Compton wavelength = Schwarzschild radius. "
                "m_P ~ 22 micrograms ~ 1.2e19 GeV. Black hole at m_P has max temp.",
    theta_relevance="Theta = m_P / m gives mass-based quantumness estimate"
)

connection_E_P_from_m_P_c = ConstantConnection(
    name="Planck energy",
    formula="E_P = m_P * c^2",
    latex=r"E_P = m_P c^2",
    inputs=["m_P", "c"],
    output="E_P",
    compute=lambda mP, c: mP * c**2,
    connection_type=ConnectionType.DERIVED,
    description="Energy of Planck mass. E_P ~ 1.22e19 GeV. "
                "Grand unification expected near this scale.",
    theta_relevance="At E ~ E_P, all forces unify (theta = 1 for all interactions)"
)

connection_T_P_from_E_P_k = ConstantConnection(
    name="Planck temperature",
    formula="T_P = E_P / k",
    latex=r"T_P = \frac{E_P}{k}",
    inputs=["E_P", "k"],
    output="T_P",
    compute=lambda EP, k: EP / k,
    connection_type=ConnectionType.DERIVED,
    description="Highest meaningful temperature. T_P ~ 1.4e32 K. "
                "Above T_P, quantum gravity dominates thermal physics.",
    theta_relevance="At T ~ T_P, thermal energy equals quantum gravity scale"
)

connection_G_from_Planck = ConstantConnection(
    name="G from Planck units",
    formula="G = h_bar * c / m_P^2",
    latex=r"G = \frac{\hbar c}{m_P^2}",
    inputs=["h_bar", "c", "m_P"],
    output="G",
    compute=lambda hbar, c, mP: hbar * c / mP**2,
    connection_type=ConnectionType.BOOTSTRAP,
    description="Gravitational constant from Planck mass. Shows G is determined "
                "by quantum (h_bar) and relativistic (c) scales.",
    theta_relevance="G defines gravitational quantum-classical boundary"
)


# =============================================================================
# TIER 5: THERMAL/STATISTICAL RELATIONSHIPS
# =============================================================================

connection_sigma_SB_from_fundamentals = ConstantConnection(
    name="Stefan-Boltzmann constant",
    formula="sigma = pi^2 * k^4 / (60 * h_bar^3 * c^2)",
    latex=r"\sigma = \frac{\pi^2 k^4}{60 \hbar^3 c^2}",
    inputs=["k", "h_bar", "c"],
    output="sigma_SB",
    compute=lambda k, hbar, c: np.pi**2 * k**4 / (60 * hbar**3 * c**2),
    connection_type=ConnectionType.DERIVED,
    description="Blackbody radiation: P = sigma * T^4. Connects thermal (k), "
                "quantum (h_bar), and relativistic (c) physics in one formula.",
    theta_relevance="sigma shows how quantum (h_bar) effects appear in thermal radiation"
)

connection_c_2_from_h_c_k = ConstantConnection(
    name="Second radiation constant",
    formula="c_2 = h * c / k",
    latex=r"c_2 = \frac{hc}{k}",
    inputs=["h", "c", "k"],
    output="c_2",
    compute=lambda h, c, k: h * c / k,
    connection_type=ConnectionType.DERIVED,
    description="Planck radiation exponent coefficient. Appears in exp(c_2/(lambda*T)).",
    theta_relevance="c_2 sets scale where quantum (hc) meets thermal (kT)"
)


# =============================================================================
# TIER 6: MASS RATIO RELATIONSHIPS
# =============================================================================

connection_m_e_over_m_p = ConstantConnection(
    name="Electron-proton mass ratio",
    formula="m_e_over_m_p = m_e / m_p",
    latex=r"\frac{m_e}{m_p}",
    inputs=["m_e", "m_p"],
    output="m_e_over_m_p",
    compute=lambda me, mp: me / mp,
    connection_type=ConnectionType.RATIO,
    description="m_e/m_p ~ 1/1836. Why this value? Unknown! Determines atomic "
                "structure. Arxiv shows potential cosmological variation.",
    theta_relevance="This ratio determines the quantum-classical boundary for atoms"
)

connection_m_p_over_m_e = ConstantConnection(
    name="Proton-electron mass ratio",
    formula="m_p_over_m_e = m_p / m_e",
    latex=r"\frac{m_p}{m_e}",
    inputs=["m_p", "m_e"],
    output="m_p_over_m_e",
    compute=lambda mp, me: mp / me,
    connection_type=ConnectionType.RATIO,
    description="mu = m_p/m_e ~ 1836. Cosmological bounds: delta_mu/mu < 10^-7.",
    theta_relevance="Proton is 1836x heavier, so less quantum (lower theta)"
)

connection_mu_N_over_mu_B = ConstantConnection(
    name="Nuclear to Bohr magneton ratio",
    formula="mu_N / mu_B = m_e / m_p",
    latex=r"\frac{\mu_N}{\mu_B} = \frac{m_e}{m_p}",
    inputs=["m_e", "m_p"],
    output="m_e_over_m_p",
    compute=lambda me, mp: me / mp,
    connection_type=ConnectionType.RATIO,
    description="Nuclear magneton is 1836x smaller than Bohr magneton.",
    theta_relevance="Magnetic moments scale with mass, affecting decoherence"
)


# =============================================================================
# TIER 7: ATOMIC UNIT CONNECTIONS
# =============================================================================

connection_au_length_from_a0 = ConstantConnection(
    name="Atomic unit of length from Bohr radius",
    formula="au_length = a_0",
    latex=r"a_0 = \frac{4\pi\epsilon_0\hbar^2}{m_e e^2}",
    inputs=["a_0"],
    output="au_length",
    compute=lambda a0: a0,
    connection_type=ConnectionType.EXACT,
    description="Atomic unit of length is exactly the Bohr radius.",
    theta_relevance="Sets the quantum scale for atomic physics (theta ~ 1)"
)

connection_au_energy_from_Eh = ConstantConnection(
    name="Atomic unit of energy from Hartree",
    formula="au_energy = E_h",
    latex=r"E_h = \alpha^2 m_e c^2",
    inputs=["E_h"],
    output="au_energy",
    compute=lambda Eh: Eh,
    connection_type=ConnectionType.EXACT,
    description="Atomic unit of energy is exactly the Hartree.",
    theta_relevance="E_h ~ 27.2 eV defines atomic energy scale"
)

connection_au_time = ConstantConnection(
    name="Atomic unit of time from h_bar and E_h",
    formula="au_time = h_bar / E_h",
    latex=r"\tau_{au} = \frac{\hbar}{E_h}",
    inputs=["h_bar", "E_h"],
    output="au_time",
    compute=lambda hbar, Eh: hbar / Eh,
    connection_type=ConnectionType.DERIVED,
    description="Atomic timescale: ~24 attoseconds.",
    theta_relevance="Sets quantum timescale for electron dynamics"
)

connection_au_velocity = ConstantConnection(
    name="Atomic unit of velocity from alpha and c",
    formula="au_velocity = alpha * c",
    latex=r"v_{au} = \alpha c",
    inputs=["alpha", "c"],
    output="au_velocity",
    compute=lambda alpha, c: alpha * c,
    connection_type=ConnectionType.DERIVED,
    description="Electron velocity in first Bohr orbit ~ c/137.",
    theta_relevance="Shows relativistic corrections are O(alpha^2)"
)

# =============================================================================
# TIER 8: ADDITIONAL MASS RATIO CONNECTIONS
# =============================================================================

connection_m_mu_over_m_e = ConstantConnection(
    name="Muon-electron mass ratio",
    formula="m_mu_over_m_e = m_mu / m_e",
    latex=r"\frac{m_\mu}{m_e}",
    inputs=["m_mu", "m_e"],
    output="m_mu_over_m_e",
    compute=lambda mmu, me: mmu / me,
    connection_type=ConnectionType.RATIO,
    description="m_mu/m_e ~ 207. Muon is heavy electron, more classical.",
    theta_relevance="Heavier muon has lower theta than electron"
)

connection_m_tau_over_m_e = ConstantConnection(
    name="Tau-electron mass ratio",
    formula="m_tau_over_m_e = m_tau / m_e",
    latex=r"\frac{m_\tau}{m_e}",
    inputs=["m_tau", "m_e"],
    output="m_tau_over_m_e",
    compute=lambda mtau, me: mtau / me,
    connection_type=ConnectionType.RATIO,
    description="m_tau/m_e ~ 3477. Tau is heaviest lepton.",
    theta_relevance="Tau has even lower theta, decays very quickly"
)

connection_m_n_over_m_p = ConstantConnection(
    name="Neutron-proton mass ratio",
    formula="m_n_over_m_p = m_n / m_p",
    latex=r"\frac{m_n}{m_p}",
    inputs=["m_n", "m_p"],
    output="m_n_over_m_p",
    compute=lambda mn, mp: mn / mp,
    connection_type=ConnectionType.RATIO,
    description="m_n/m_p ~ 1.00137. Critical for nuclear stability!",
    theta_relevance="Tiny mass difference enables neutron decay and nucleosynthesis"
)

# =============================================================================
# TIER 9: MAGNETIC MOMENT CONNECTIONS
# =============================================================================

connection_mu_p_over_mu_N = ConstantConnection(
    name="Proton magnetic moment in nuclear magnetons",
    formula="mu_p_over_mu_N = mu_p / mu_N",
    latex=r"\frac{\mu_p}{\mu_N}",
    inputs=["mu_p", "mu_N"],
    output="mu_p_over_mu_N",
    compute=lambda mup, muN: mup / muN,
    connection_type=ConnectionType.RATIO,
    description="g_p ~ 5.59. NOT 2! Shows proton is not point-like.",
    theta_relevance="Anomalous moment reveals internal quark structure"
)

connection_mu_e_over_mu_B = ConstantConnection(
    name="Electron magnetic moment anomaly",
    formula="mu_e / mu_B = g_e / 2",
    latex=r"\frac{\mu_e}{\mu_B} = \frac{g_e}{2}",
    inputs=["g_e"],
    output="mu_e_over_mu_B",
    compute=lambda ge: ge / 2,
    connection_type=ConnectionType.RATIO,
    description="Electron g-factor from QED: g_e = 2(1 + a_e). Most precise QED prediction!",
    theta_relevance="QED correction a_e ~ alpha/(2pi) measures vacuum fluctuations"
)

# =============================================================================
# TIER 10: ENERGY CONVERSION CONNECTIONS
# =============================================================================

connection_eV_to_Hz = ConstantConnection(
    name="Electron volt to frequency",
    formula="eV_to_Hz = e / h",
    latex=r"\frac{e}{h} = 2.418 \times 10^{14} \text{ Hz/eV}",
    inputs=["e", "h"],
    output="eV_to_Hz",
    compute=lambda e, h: e / h,
    connection_type=ConnectionType.CONVERSION,
    description="E = hf, so f = E/h. 1 eV corresponds to 241.8 THz.",
    theta_relevance="Links particle energy to wave frequency (wave-particle duality)"
)

connection_eV_to_K = ConstantConnection(
    name="Electron volt to kelvin",
    formula="eV_to_K = e / k",
    latex=r"\frac{e}{k} = 11604.5 \text{ K/eV}",
    inputs=["e", "k"],
    output="eV_to_K",
    compute=lambda e, k: e / k,
    connection_type=ConnectionType.CONVERSION,
    description="E = kT, so T = E/k. 1 eV ~ 11,600 K.",
    theta_relevance="Connects particle energy to thermal energy (quantum vs thermal)"
)

connection_hartree_to_eV = ConstantConnection(
    name="Hartree to electron volt",
    formula="hartree_to_eV = E_h / e",
    latex=r"\frac{E_h}{e} = 27.21 \text{ eV}",
    inputs=["E_h", "e"],
    output="hartree_to_eV",
    compute=lambda Eh, e: Eh / e,
    connection_type=ConnectionType.CONVERSION,
    description="1 Hartree = 27.2 eV = 2 Rydberg.",
    theta_relevance="Atomic unit of energy in particle physics units"
)

# =============================================================================
# TIER 11: COMPTON WAVELENGTH CONNECTIONS
# =============================================================================

connection_lambda_C_p = ConstantConnection(
    name="Proton Compton wavelength",
    formula="lambda_C_p = h / (m_p * c)",
    latex=r"\lambda_{C,p} = \frac{h}{m_p c}",
    inputs=["h", "m_p", "c"],
    output="lambda_C_p",
    compute=lambda h, mp, c: h / (mp * c),
    connection_type=ConnectionType.DERIVED,
    description="Proton Compton wavelength ~ 1.32 fm. Sets nuclear scale.",
    theta_relevance="Quantum scale for proton ~ 10^-15 m"
)

connection_lambda_C_n = ConstantConnection(
    name="Neutron Compton wavelength",
    formula="lambda_C_n = h / (m_n * c)",
    latex=r"\lambda_{C,n} = \frac{h}{m_n c}",
    inputs=["h", "m_n", "c"],
    output="lambda_C_n",
    compute=lambda h, mn, c: h / (mn * c),
    connection_type=ConnectionType.DERIVED,
    description="Neutron Compton wavelength ~ 1.32 fm.",
    theta_relevance="Nearly identical to proton due to similar masses"
)

# =============================================================================
# TIER 12: QUANTUM CIRCULATION AND SPECIAL RATIOS
# =============================================================================

connection_quantum_circulation = ConstantConnection(
    name="Quantum of circulation",
    formula="h / (2 * m_e)",
    latex=r"\frac{h}{2m_e}",
    inputs=["h", "m_e"],
    output="quantum_circulation",
    compute=lambda h, me: h / (2 * me),
    connection_type=ConnectionType.DERIVED,
    description="Fundamental quantum of circulation in superfluids.",
    theta_relevance="Quantized vortices are macroscopic quantum phenomena"
)

connection_e_over_m_e = ConstantConnection(
    name="Electron charge to mass quotient",
    formula="e_over_m_e = -e / m_e",
    latex=r"-\frac{e}{m_e}",
    inputs=["e", "m_e"],
    output="e_over_m_e",
    compute=lambda e, me: -e / me,
    connection_type=ConnectionType.RATIO,
    description="Electron charge to mass quotient (negative by convention). First measured by Thomson.",
    theta_relevance="Determines electron cyclotron frequency in magnetic fields"
)

connection_inverse_G_0 = ConstantConnection(
    name="Inverse conductance quantum",
    formula="1 / G_0 = h / (2*e^2)",
    latex=r"\frac{1}{G_0} = \frac{h}{2e^2}",
    inputs=["h", "e"],
    output="inverse_G_0",
    compute=lambda h, e: h / (2 * e**2),
    connection_type=ConnectionType.DERIVED,
    description="Quantum of resistance ~ 12.9 kOhm.",
    theta_relevance="Appears in quantum Hall effect, atomic-scale electronics"
)

# =============================================================================
# SPECIAL: HAWKING TEMPERATURE (Unifies all physics)
# =============================================================================

def compute_hawking_temperature(hbar: float, c: float, G: float, k: float, M: float) -> float:
    """Compute Hawking temperature for black hole of mass M."""
    return hbar * c**3 / (8 * np.pi * G * M * k)


# This is special because it involves a parameter (mass M)
# We define it separately for black hole physics
HAWKING_FORMULA = r"T_H = \frac{\hbar c^3}{8\pi G M k}"
HAWKING_DESCRIPTION = (
    "Hawking temperature unifies quantum (h_bar), relativistic (c), "
    "gravitational (G), and thermal (k) physics in one formula. "
    "Small black holes are hot (quantum), large ones cold (classical)."
)
HAWKING_THETA_RELEVANCE = (
    "At the event horizon, theta -> 1 (quantum). Hawking radiation is the "
    "quantum effect of black holes. T_H determines black hole evaporation."
)


# =============================================================================
# MASTER CONNECTION REGISTRY
# =============================================================================

ALL_CONNECTIONS: Dict[str, ConstantConnection] = {
    # Tier 1: SI definitions
    "h_bar_from_h": connection_h_bar_from_h,
    "R_from_N_A_k": connection_R_from_N_A_k,
    "F_from_N_A_e": connection_F_from_N_A_e,

    # Tier 2: Electromagnetic
    "c_from_epsilon_mu": connection_c_from_epsilon_mu,
    "alpha_from_fundamentals": connection_alpha_from_fundamentals,
    "Z_0_from_mu_c": connection_Z_0_from_mu_c,
    "R_K_from_h_e": connection_R_K_from_h_e,
    "K_J_from_e_h": connection_K_J_from_e_h,
    "Phi_0_from_h_e": connection_Phi_0_from_h_e,
    "G_0_from_e_h": connection_G_0_from_e_h,
    "mu_B_from_e_hbar_me": connection_mu_B_from_e_hbar_me,
    "mu_N_from_e_hbar_mp": connection_mu_N_from_e_hbar_mp,

    # Tier 3: Atomic
    "a_0_from_fundamentals": connection_a_0_from_fundamentals,
    "R_inf_from_fundamentals": connection_R_inf_from_fundamentals,
    "E_h_from_alpha_me_c": connection_E_h_from_alpha_me_c,
    "lambda_C_from_h_me_c": connection_lambda_C_from_h_me_c,
    "r_e_from_alpha_a_0": connection_r_e_from_alpha_a_0,
    "sigma_e_from_r_e": connection_sigma_e_from_r_e,

    # Tier 4: Planck
    "l_P_from_hbar_G_c": connection_l_P_from_hbar_G_c,
    "t_P_from_hbar_G_c": connection_t_P_from_hbar_G_c,
    "m_P_from_hbar_c_G": connection_m_P_from_hbar_c_G,
    "E_P_from_m_P_c": connection_E_P_from_m_P_c,
    "T_P_from_E_P_k": connection_T_P_from_E_P_k,
    "G_from_Planck": connection_G_from_Planck,

    # Tier 5: Thermal
    "sigma_SB_from_fundamentals": connection_sigma_SB_from_fundamentals,
    "c_2_from_h_c_k": connection_c_2_from_h_c_k,

    # Tier 6: Ratios
    "m_e_over_m_p": connection_m_e_over_m_p,
    "m_p_over_m_e": connection_m_p_over_m_e,

    # Tier 7: Atomic units
    "au_length_from_a0": connection_au_length_from_a0,
    "au_energy_from_Eh": connection_au_energy_from_Eh,
    "au_time": connection_au_time,
    "au_velocity": connection_au_velocity,

    # Tier 8: Additional mass ratios
    "m_mu_over_m_e": connection_m_mu_over_m_e,
    "m_tau_over_m_e": connection_m_tau_over_m_e,
    "m_n_over_m_p": connection_m_n_over_m_p,

    # Tier 9: Magnetic moments
    "mu_p_over_mu_N": connection_mu_p_over_mu_N,

    # Tier 10: Energy conversions
    "eV_to_Hz": connection_eV_to_Hz,
    "eV_to_K": connection_eV_to_K,
    "hartree_to_eV": connection_hartree_to_eV,

    # Tier 11: Compton wavelengths
    "lambda_C_p": connection_lambda_C_p,
    "lambda_C_n": connection_lambda_C_n,

    # Tier 12: Quantum circulation and special
    "quantum_circulation": connection_quantum_circulation,
    "e_over_m_e": connection_e_over_m_e,
    "inverse_G_0": connection_inverse_G_0,
}


def verify_all_connections(tolerance: float = 1e-8) -> Dict[str, Tuple[bool, float]]:
    """
    Verify all mathematical connections hold.

    Returns:
        Dict mapping connection name to (is_valid, relative_error)
    """
    results = {}
    for name, conn in ALL_CONNECTIONS.items():
        results[name] = conn.verify(tolerance)
    return results


def get_connection_network() -> Dict[str, List[str]]:
    """
    Return the dependency graph of constants.

    Maps each derived constant to the constants it depends on.
    """
    network = {}
    for name, conn in ALL_CONNECTIONS.items():
        if conn.output not in network:
            network[conn.output] = []
        network[conn.output].extend(conn.inputs)
    return network


def get_theta_relevant_connections() -> Dict[str, ConstantConnection]:
    """Return connections that directly relate to quantum-classical boundary."""
    return {
        name: conn for name, conn in ALL_CONNECTIONS.items()
        if conn.theta_relevance
    }


def print_connection_summary():
    """Print summary of all connections with verification status."""
    print("=" * 70)
    print("MATHEMATICAL CONNECTIONS BETWEEN CONSTANTS")
    print("=" * 70)

    results = verify_all_connections()

    for conn_type in ConnectionType:
        conns = {k: v for k, v in ALL_CONNECTIONS.items() if v.connection_type == conn_type}
        if conns:
            print(f"\n{conn_type.value.upper()} CONNECTIONS")
            print("-" * 50)
            for name, conn in conns.items():
                valid, error = results[name]
                status = "PASS" if valid else "FAIL"
                print(f"  [{status}] {conn.formula}")
                print(f"         Error: {error:.2e}")


def generate_theta_boundary_report() -> str:
    """Generate report on how constants define quantum-classical boundaries."""
    lines = [
        "=" * 70,
        "THETA BOUNDARIES FROM FUNDAMENTAL CONSTANTS",
        "=" * 70,
        "",
        "Each constant defines a scale where quantum and classical meet:",
        "",
    ]

    theta_conns = get_theta_relevant_connections()
    for name, conn in theta_conns.items():
        lines.append(f"  {conn.formula}")
        lines.append(f"    -> {conn.theta_relevance}")
        lines.append("")

    lines.extend([
        "-" * 70,
        "KEY INSIGHT: Theta emerges from the ratio of quantum to classical scales.",
        "The interconnection of constants proves this is not arbitrary but",
        "fundamental to the structure of physics.",
        "=" * 70,
    ])

    return "\n".join(lines)


# Module exports
__all__ = [
    "ConnectionType",
    "ConstantConnection",
    "ALL_CONNECTIONS",
    "verify_all_connections",
    "get_connection_network",
    "get_theta_relevant_connections",
    "print_connection_summary",
    "generate_theta_boundary_report",
    "compute_hawking_temperature",
    "HAWKING_FORMULA",
    "HAWKING_DESCRIPTION",
]
