"""
Electromagnetic Spectrum Domain: Light as Theta

This module implements theta as the quantum-classical interpolation parameter
for electromagnetic radiation, mapping wavelength/frequency to the transition
from classical wave behavior to quantum particle behavior.

Key Insight: The quantum-classical transition for photons is governed by
photon statistics from black-body radiation physics:
- E << kT: Many photons per mode -> classical field (Rayleigh-Jeans regime)
- E >> kT: Few photons per mode -> quantum discreteness (Wien regime)

This is exactly the physics that launched quantum mechanics when Planck
resolved the ultraviolet catastrophe in 1900.

Theta Formula:
    theta = E / (E + kT)

Where:
- E = hf = hc/lambda (photon energy)
- kT = thermal energy at reference temperature

Properties:
- theta -> 0 as E -> 0 (radio waves: classical)
- theta = 0.5 when E = kT (thermal crossover)
- theta -> 1 as E -> infinity (gamma rays: fully quantum)

References (see BIBLIOGRAPHY.bib):
    \\cite{Planck1901} - Planck's law and quantum hypothesis
    \\cite{Einstein1905Photoelectric} - Photoelectric effect
    \\cite{StefanBoltzmann} - Stefan-Boltzmann law
    \\cite{CODATA2022} - Fundamental constants
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class EMBand(Enum):
    """Electromagnetic spectrum bands."""
    RADIO = "radio"              # > 1 mm (< 300 GHz)
    MICROWAVE = "microwave"      # 1 mm - 1 m (300 MHz - 300 GHz)
    INFRARED = "infrared"        # 700 nm - 1 mm
    VISIBLE = "visible"          # 380 - 700 nm
    ULTRAVIOLET = "ultraviolet"  # 10 - 380 nm
    X_RAY = "x_ray"              # 0.01 - 10 nm
    GAMMA = "gamma"              # < 0.01 nm


class PhotonRegime(Enum):
    """Classification of photon behavior regime."""
    CLASSICAL = "classical"          # theta < 0.1: wave optics valid
    SEMICLASSICAL = "semiclassical"  # 0.1 <= theta < 0.5: transitional
    QUANTUM = "quantum"              # 0.5 <= theta < 0.99: particle-like
    DEEP_QUANTUM = "deep_quantum"    # theta >= 0.99: fully quantum


# =============================================================================
# PHYSICAL CONSTANTS (CODATA 2022)
# =============================================================================

C = 299792458                    # Speed of light (m/s)
H = 6.62607015e-34               # Planck constant (J*s)
HBAR = 1.054571817e-34           # Reduced Planck constant (J*s)
K_B = 1.380649e-23               # Boltzmann constant (J/K)
EV_TO_J = 1.602176634e-19        # eV to Joules conversion

# Derived constants
HC_EV_NM = 1239.84193            # hc in eV*nm (for E = hc/lambda)
KT_300K_EV = K_B * 300 / EV_TO_J  # kT at 300K in eV (~0.02585 eV)


@dataclass
class PhotonSystem:
    """
    A photon/EM radiation system for theta analysis.

    Attributes:
        name: Human-readable name
        wavelength_m: Wavelength in meters
        band: EM spectrum band classification
        description: Physical description or application

    Note: frequency_hz and energy_ev are computed from wavelength.
    """
    name: str
    wavelength_m: float
    band: EMBand
    description: Optional[str] = None

    @property
    def frequency_hz(self) -> float:
        """Frequency in Hz: f = c / lambda"""
        return C / self.wavelength_m

    @property
    def energy_j(self) -> float:
        """Photon energy in Joules: E = hf"""
        return H * self.frequency_hz

    @property
    def energy_ev(self) -> float:
        """Photon energy in eV"""
        return self.energy_j / EV_TO_J

    @property
    def wavelength_nm(self) -> float:
        """Wavelength in nanometers"""
        return self.wavelength_m * 1e9


# =============================================================================
# THETA COMPUTATION
# =============================================================================

def compute_em_theta(
    system: PhotonSystem,
    temperature_k: float = 300.0
) -> float:
    """
    Compute theta for electromagnetic radiation.

    theta = E / (E + kT)

    This formula captures the quantum-classical crossover from black-body
    radiation physics:
    - When E << kT: theta -> 0 (classical, many photons per mode)
    - When E = kT: theta = 0.5 (crossover point)
    - When E >> kT: theta -> 1 (quantum, single photon regime)

    Args:
        system: PhotonSystem to analyze
        temperature_k: Reference temperature in Kelvin (default 300K)

    Returns:
        Theta in [0, 1]

    Reference: \\cite{Planck1901}
    """
    # Thermal energy kT in eV
    kT_ev = K_B * temperature_k / EV_TO_J

    # Photon energy
    E_ev = system.energy_ev

    # Theta formula
    theta = E_ev / (E_ev + kT_ev)

    return np.clip(theta, 0.0, 1.0)


def compute_wavelength_theta(
    wavelength_m: float,
    temperature_k: float = 300.0
) -> float:
    """
    Compute theta directly from wavelength.

    Convenience function when you don't need a full PhotonSystem.

    Args:
        wavelength_m: Wavelength in meters
        temperature_k: Reference temperature in Kelvin

    Returns:
        Theta in [0, 1]
    """
    # E = hc/lambda in eV
    E_ev = HC_EV_NM / (wavelength_m * 1e9)  # Convert m to nm

    # kT in eV
    kT_ev = K_B * temperature_k / EV_TO_J

    theta = E_ev / (E_ev + kT_ev)
    return np.clip(theta, 0.0, 1.0)


def compute_frequency_theta(
    frequency_hz: float,
    temperature_k: float = 300.0
) -> float:
    """
    Compute theta directly from frequency.

    Args:
        frequency_hz: Frequency in Hz
        temperature_k: Reference temperature in Kelvin

    Returns:
        Theta in [0, 1]
    """
    # E = hf in eV
    E_ev = H * frequency_hz / EV_TO_J

    # kT in eV
    kT_ev = K_B * temperature_k / EV_TO_J

    theta = E_ev / (E_ev + kT_ev)
    return np.clip(theta, 0.0, 1.0)


def classify_photon_regime(theta: float) -> PhotonRegime:
    """Classify photon behavior regime from theta."""
    if theta < 0.1:
        return PhotonRegime.CLASSICAL
    elif theta < 0.5:
        return PhotonRegime.SEMICLASSICAL
    elif theta < 0.99:
        return PhotonRegime.QUANTUM
    else:
        return PhotonRegime.DEEP_QUANTUM


def classify_em_band(wavelength_m: float) -> EMBand:
    """Classify wavelength into EM spectrum band."""
    wavelength_nm = wavelength_m * 1e9

    if wavelength_nm < 0.01:
        return EMBand.GAMMA
    elif wavelength_nm < 10:
        return EMBand.X_RAY
    elif wavelength_nm < 380:
        return EMBand.ULTRAVIOLET
    elif wavelength_nm < 700:
        return EMBand.VISIBLE
    elif wavelength_nm < 1e6:  # 1 mm
        return EMBand.INFRARED
    elif wavelength_nm < 1e9:  # 1 m
        return EMBand.MICROWAVE
    else:
        return EMBand.RADIO


# =============================================================================
# WIEN AND PLANCK FUNCTIONS
# =============================================================================

def wien_displacement_wavelength(temperature_k: float) -> float:
    """
    Wien's displacement law: lambda_max = b / T

    Returns wavelength of peak black-body emission in meters.

    Reference: \\cite{StefanBoltzmann}
    """
    WIEN_B = 2.897771955e-3  # Wien displacement constant (m*K)
    return WIEN_B / temperature_k


def planck_spectral_radiance(wavelength_m: float, temperature_k: float) -> float:
    """
    Planck's law for spectral radiance.

    B(lambda, T) = (2hc^2/lambda^5) / (exp(hc/(lambda*kT)) - 1)

    Returns spectral radiance in W/(m^2 * sr * m).

    Reference: \\cite{Planck1901}
    """
    numerator = 2 * H * C**2 / wavelength_m**5
    exponent = H * C / (wavelength_m * K_B * temperature_k)

    # Avoid overflow for very short wavelengths
    if exponent > 700:
        return 0.0

    denominator = np.exp(exponent) - 1
    return numerator / denominator


def photon_occupation_number(energy_ev: float, temperature_k: float) -> float:
    """
    Bose-Einstein distribution for photon occupation number.

    n = 1 / (exp(E/kT) - 1)

    This is the average number of photons per mode at energy E.
    - n >> 1: Classical regime (many photons)
    - n ~ 1: Quantum regime (few photons)
    - n << 1: Deep quantum (rare events)

    Reference: \\cite{Planck1901}
    """
    kT_ev = K_B * temperature_k / EV_TO_J
    exponent = energy_ev / kT_ev

    if exponent > 700:
        return 0.0

    return 1.0 / (np.exp(exponent) - 1)


# =============================================================================
# EXAMPLE SYSTEMS: THE ELECTROMAGNETIC SPECTRUM
# =============================================================================

EM_SPECTRUM: Dict[str, PhotonSystem] = {
    # Radio waves
    "am_radio": PhotonSystem(
        name="AM Radio (1 MHz)",
        wavelength_m=300.0,
        band=EMBand.RADIO,
        description="Broadcast AM radio, highly classical wave behavior"
    ),
    "fm_radio": PhotonSystem(
        name="FM Radio (100 MHz)",
        wavelength_m=3.0,
        band=EMBand.RADIO,
        description="Broadcast FM radio"
    ),
    "shortwave": PhotonSystem(
        name="Shortwave Radio (10 MHz)",
        wavelength_m=30.0,
        band=EMBand.RADIO,
        description="International broadcasting, ionospheric reflection"
    ),

    # Microwaves
    "wifi_2ghz": PhotonSystem(
        name="WiFi 2.4 GHz",
        wavelength_m=0.125,
        band=EMBand.MICROWAVE,
        description="Wireless networking frequency"
    ),
    "wifi_5ghz": PhotonSystem(
        name="WiFi 5 GHz",
        wavelength_m=0.06,
        band=EMBand.MICROWAVE,
        description="Higher bandwidth wireless"
    ),
    "microwave_oven": PhotonSystem(
        name="Microwave Oven (2.45 GHz)",
        wavelength_m=0.122,
        band=EMBand.MICROWAVE,
        description="Resonant with water molecules"
    ),
    "cmb": PhotonSystem(
        name="Cosmic Microwave Background",
        wavelength_m=1.9e-3,
        band=EMBand.MICROWAVE,
        description="Relic radiation from Big Bang, T=2.725K peak"
    ),
    "5g_mmwave": PhotonSystem(
        name="5G mmWave (30 GHz)",
        wavelength_m=0.01,
        band=EMBand.MICROWAVE,
        description="High-bandwidth 5G cellular"
    ),

    # Infrared
    "far_infrared": PhotonSystem(
        name="Far Infrared (100 um)",
        wavelength_m=100e-6,
        band=EMBand.INFRARED,
        description="Thermal emission from cool objects"
    ),
    "thermal_ir": PhotonSystem(
        name="Thermal IR (10 um)",
        wavelength_m=10e-6,
        band=EMBand.INFRARED,
        description="Peak human body emission, thermal imaging"
    ),
    "near_ir": PhotonSystem(
        name="Near IR (1 um)",
        wavelength_m=1e-6,
        band=EMBand.INFRARED,
        description="Fiber optics, night vision"
    ),
    "ir_remote": PhotonSystem(
        name="IR Remote (940 nm)",
        wavelength_m=940e-9,
        band=EMBand.INFRARED,
        description="TV remote controls"
    ),

    # Visible light
    "red_light": PhotonSystem(
        name="Red Light (700 nm)",
        wavelength_m=700e-9,
        band=EMBand.VISIBLE,
        description="Longest visible wavelength"
    ),
    "orange_light": PhotonSystem(
        name="Orange Light (600 nm)",
        wavelength_m=600e-9,
        band=EMBand.VISIBLE,
        description="Sodium lamp emission"
    ),
    "yellow_light": PhotonSystem(
        name="Yellow Light (580 nm)",
        wavelength_m=580e-9,
        band=EMBand.VISIBLE,
        description="Peak human eye sensitivity"
    ),
    "green_light": PhotonSystem(
        name="Green Light (550 nm)",
        wavelength_m=550e-9,
        band=EMBand.VISIBLE,
        description="Maximum human eye sensitivity"
    ),
    "blue_light": PhotonSystem(
        name="Blue Light (450 nm)",
        wavelength_m=450e-9,
        band=EMBand.VISIBLE,
        description="Short visible wavelength"
    ),
    "violet_light": PhotonSystem(
        name="Violet Light (400 nm)",
        wavelength_m=400e-9,
        band=EMBand.VISIBLE,
        description="Shortest visible wavelength"
    ),

    # Ultraviolet
    "uva": PhotonSystem(
        name="UVA (350 nm)",
        wavelength_m=350e-9,
        band=EMBand.ULTRAVIOLET,
        description="Near UV, causes tanning"
    ),
    "uvb": PhotonSystem(
        name="UVB (300 nm)",
        wavelength_m=300e-9,
        band=EMBand.ULTRAVIOLET,
        description="Causes sunburn, vitamin D synthesis"
    ),
    "uvc": PhotonSystem(
        name="UVC (250 nm)",
        wavelength_m=250e-9,
        band=EMBand.ULTRAVIOLET,
        description="Germicidal UV, absorbed by ozone"
    ),
    "extreme_uv": PhotonSystem(
        name="Extreme UV (100 nm)",
        wavelength_m=100e-9,
        band=EMBand.ULTRAVIOLET,
        description="EUV lithography for semiconductors"
    ),

    # X-rays
    "soft_xray": PhotonSystem(
        name="Soft X-ray (10 nm)",
        wavelength_m=10e-9,
        band=EMBand.X_RAY,
        description="X-ray microscopy, synchrotron light"
    ),
    "medical_xray": PhotonSystem(
        name="Medical X-ray (0.1 nm)",
        wavelength_m=0.1e-9,
        band=EMBand.X_RAY,
        description="Diagnostic radiography, ~12 keV"
    ),
    "hard_xray": PhotonSystem(
        name="Hard X-ray (0.01 nm)",
        wavelength_m=0.01e-9,
        band=EMBand.X_RAY,
        description="Crystal diffraction, ~120 keV"
    ),

    # Gamma rays
    "gamma_nuclear": PhotonSystem(
        name="Nuclear Gamma (0.001 nm)",
        wavelength_m=0.001e-9,
        band=EMBand.GAMMA,
        description="Nuclear decay, ~1.2 MeV"
    ),
    "gamma_medical": PhotonSystem(
        name="Medical Gamma (Tc-99m)",
        wavelength_m=8.8e-12,
        band=EMBand.GAMMA,
        description="SPECT imaging, 140 keV"
    ),
    "gamma_cosmic": PhotonSystem(
        name="Cosmic Gamma Ray",
        wavelength_m=1e-15,
        band=EMBand.GAMMA,
        description="High-energy astrophysical sources, ~1 GeV"
    ),
}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def em_theta_summary():
    """Print theta analysis across the electromagnetic spectrum."""
    print("=" * 95)
    print("ELECTROMAGNETIC SPECTRUM THETA ANALYSIS")
    print("Formula: theta = E / (E + kT) at T = 300K")
    print("=" * 95)
    print()
    print(f"{'System':<30} {'Wavelength':>12} {'Energy (eV)':>14} {'theta':>10} {'Regime':<15}")
    print("-" * 95)

    # Sort by wavelength (longest to shortest = lowest to highest theta)
    sorted_systems = sorted(
        EM_SPECTRUM.items(),
        key=lambda x: -x[1].wavelength_m
    )

    for name, system in sorted_systems:
        theta = compute_em_theta(system)
        regime = classify_photon_regime(theta)

        # Format wavelength appropriately
        wl = system.wavelength_m
        if wl >= 1:
            wl_str = f"{wl:.1f} m"
        elif wl >= 1e-3:
            wl_str = f"{wl*1e3:.1f} mm"
        elif wl >= 1e-6:
            wl_str = f"{wl*1e6:.1f} um"
        elif wl >= 1e-9:
            wl_str = f"{wl*1e9:.1f} nm"
        else:
            wl_str = f"{wl*1e12:.3f} pm"

        # Format energy
        E = system.energy_ev
        if E >= 1e6:
            e_str = f"{E/1e6:.2f} MeV"
        elif E >= 1e3:
            e_str = f"{E/1e3:.2f} keV"
        elif E >= 1:
            e_str = f"{E:.2f} eV"
        else:
            e_str = f"{E:.2e} eV"

        # Format theta
        if theta >= 0.001:
            theta_str = f"{theta:.4f}"
        else:
            theta_str = f"{theta:.2e}"

        print(f"{system.name:<30} {wl_str:>12} {e_str:>14} {theta_str:>10} {regime.value:<15}")

    print()
    print("Key: theta -> 0 (classical waves), theta -> 1 (quantum particles)")
    print("     Crossover at theta = 0.5 corresponds to E = kT")


if __name__ == "__main__":
    em_theta_summary()
