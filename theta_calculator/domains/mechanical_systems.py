"""
Mechanical Systems Domain: Engines, Motors, and Energy Conversion

This module implements theta as the efficiency parameter for mechanical
and thermodynamic systems - "how things work."

Key Insight: Mechanical systems are bounded by fundamental limits:
- theta ~ 0: Wasteful operation (low efficiency)
- theta ~ 1: Ideal operation (Carnot, reversible)

Theta Maps To:
1. Heat engines: η_actual / η_Carnot
2. Electric motors: P_out / P_in
3. Batteries: E_actual / E_Nernst
4. Damped oscillators: c / c_critical

References (see BIBLIOGRAPHY.bib):
    \cite{Carnot1824} - Reflections on the Motive Power of Fire
    \cite{Callen1985} - Thermodynamics and an Introduction to Thermostatistics
    \cite{Newman2004} - Electrochemical Systems
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class SystemType(Enum):
    """Types of mechanical/thermodynamic systems."""
    HEAT_ENGINE = "heat_engine"
    REFRIGERATOR = "refrigerator"
    HEAT_PUMP = "heat_pump"
    ELECTRIC_MOTOR = "electric_motor"
    GENERATOR = "generator"
    BATTERY = "battery"
    OSCILLATOR = "oscillator"
    TRANSMISSION = "transmission"


class EfficiencyRegime(Enum):
    """Efficiency regime classification."""
    WASTEFUL = "wasteful"       # theta < 0.3
    TYPICAL = "typical"         # 0.3 <= theta < 0.6
    EFFICIENT = "efficient"     # 0.6 <= theta < 0.9
    NEAR_IDEAL = "near_ideal"   # theta >= 0.9


@dataclass
class MechanicalSystem:
    """
    A mechanical system for theta analysis.

    Attributes:
        name: System identifier
        system_type: Type of system
        input_power: Power input [W]
        output_power: Useful power output [W]
        efficiency: Actual efficiency [0, 1]
        theoretical_max: Maximum possible efficiency [0, 1]
        temperature_hot: Hot reservoir temperature [K]
        temperature_cold: Cold reservoir temperature [K]
    """
    name: str
    system_type: SystemType
    input_power: float
    output_power: float
    efficiency: float
    theoretical_max: float
    temperature_hot: Optional[float] = None
    temperature_cold: Optional[float] = None


# =============================================================================
# HEAT ENGINE CALCULATIONS
# =============================================================================

def carnot_efficiency(T_hot: float, T_cold: float) -> float:
    """
    Compute Carnot efficiency for heat engine.

    η_Carnot = 1 - T_cold / T_hot

    This is the MAXIMUM possible efficiency for any heat engine
    operating between temperatures T_hot and T_cold.

    Args:
        T_hot: Hot reservoir temperature [K]
        T_cold: Cold reservoir temperature [K]

    Returns:
        Carnot efficiency in [0, 1]

    Reference: \cite{Carnot1824}
    """
    if T_hot <= T_cold:
        raise ValueError("T_hot must be greater than T_cold")
    if T_hot <= 0 or T_cold <= 0:
        raise ValueError("Temperatures must be positive (Kelvin)")

    return 1 - T_cold / T_hot


def otto_efficiency(compression_ratio: float, gamma: float = 1.4) -> float:
    """
    Compute Otto cycle efficiency (gasoline engine).

    η_Otto = 1 - r^(1-γ)

    Where r is compression ratio and γ = Cp/Cv.

    Args:
        compression_ratio: V_max / V_min (typically 8-12)
        gamma: Heat capacity ratio (1.4 for air)

    Returns:
        Otto efficiency

    Reference: \cite{Callen1985}
    """
    if compression_ratio <= 1:
        raise ValueError("Compression ratio must be > 1")
    return 1 - compression_ratio ** (1 - gamma)


def diesel_efficiency(
    compression_ratio: float,
    cutoff_ratio: float,
    gamma: float = 1.4
) -> float:
    """
    Compute Diesel cycle efficiency.

    η_Diesel = 1 - (1/r^(γ-1)) * (ρ^γ - 1) / (γ(ρ - 1))

    Where r is compression ratio and ρ is cutoff ratio.

    Args:
        compression_ratio: V_max / V_min (typically 14-25)
        cutoff_ratio: Volume ratio at end of heat addition
        gamma: Heat capacity ratio

    Returns:
        Diesel efficiency

    Reference: \cite{Callen1985}
    """
    if compression_ratio <= 1 or cutoff_ratio <= 1:
        raise ValueError("Ratios must be > 1")

    r = compression_ratio
    rho = cutoff_ratio

    factor1 = 1 / r ** (gamma - 1)
    factor2 = (rho ** gamma - 1) / (gamma * (rho - 1))

    return 1 - factor1 * factor2


def compute_engine_theta(
    actual_efficiency: float,
    T_hot: float,
    T_cold: float
) -> float:
    """
    Compute theta for heat engine.

    Theta = η_actual / η_Carnot

    Args:
        actual_efficiency: Measured efficiency
        T_hot: Hot reservoir temperature [K]
        T_cold: Cold reservoir temperature [K]

    Returns:
        theta in [0, 1]

    Reference: \cite{Carnot1824}
    """
    eta_carnot = carnot_efficiency(T_hot, T_cold)

    if eta_carnot <= 0:
        return 0.0

    theta = actual_efficiency / eta_carnot
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ELECTRIC MOTOR/GENERATOR
# =============================================================================

@dataclass
class MotorEfficiency:
    """
    Electric motor efficiency analysis.

    Losses in motors:
    - Copper losses: I²R in windings
    - Iron losses: Hysteresis and eddy currents
    - Mechanical losses: Friction, windage
    - Stray losses: Other electromagnetic losses

    Attributes:
        input_power: Electrical input [W]
        output_power: Mechanical output [W]
        efficiency: Overall efficiency
        copper_loss: Resistive losses [W]
        iron_loss: Core losses [W]
        mechanical_loss: Friction losses [W]
        theta: Efficiency as theta

    Reference: \cite{Chapman2012}
    """
    input_power: float
    output_power: float
    efficiency: float
    copper_loss: float
    iron_loss: float
    mechanical_loss: float
    theta: float


def compute_motor_theta(
    input_power: float,
    output_power: float,
    max_efficiency: float = 0.98  # Best motors
) -> float:
    """
    Compute theta for electric motor.

    Theta = η_actual / η_max

    Modern motors can achieve 95%+ efficiency.

    Args:
        input_power: Electrical power in [W]
        output_power: Mechanical power out [W]
        max_efficiency: Maximum achievable efficiency

    Returns:
        theta in [0, 1]
    """
    if input_power <= 0:
        return 0.0

    efficiency = output_power / input_power
    theta = efficiency / max_efficiency
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# BATTERY / ELECTROCHEMISTRY
# =============================================================================

def nernst_potential(
    E_standard: float,
    temperature: float,
    n_electrons: int,
    activity_ratio: float
) -> float:
    """
    Compute Nernst potential for electrochemical cell.

    E = E° - (RT/nF) * ln(Q)

    Args:
        E_standard: Standard electrode potential [V]
        temperature: Temperature [K]
        n_electrons: Number of electrons transferred
        activity_ratio: Q = products/reactants activity ratio

    Returns:
        Cell potential [V]

    Reference: \cite{Newman2004}
    """
    R = 8.314  # Gas constant [J/mol/K]
    F = 96485  # Faraday constant [C/mol]

    return E_standard - (R * temperature / (n_electrons * F)) * np.log(activity_ratio)


def compute_battery_theta(
    actual_voltage: float,
    open_circuit_voltage: float,
    internal_resistance: float = 0.0,
    current: float = 0.0
) -> float:
    """
    Compute theta for battery.

    Theta = V_actual / V_OCV = (V_OCV - IR) / V_OCV

    Under load, voltage drops due to internal resistance.
    theta = 1 means no internal losses (ideal battery).

    Args:
        actual_voltage: Terminal voltage under load [V]
        open_circuit_voltage: OCV (thermodynamic limit) [V]
        internal_resistance: Internal resistance [Ω]
        current: Discharge current [A]

    Returns:
        theta in [0, 1]
    """
    if open_circuit_voltage <= 0:
        return 0.0

    if internal_resistance > 0 and current > 0:
        # Voltage under load
        V_load = open_circuit_voltage - internal_resistance * current
        actual_voltage = max(V_load, 0)

    theta = actual_voltage / open_circuit_voltage
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# DAMPED OSCILLATOR
# =============================================================================

def critical_damping(mass: float, spring_constant: float) -> float:
    """
    Compute critical damping coefficient.

    c_critical = 2 * sqrt(k * m)

    At critical damping, system returns to equilibrium
    in minimum time without oscillation.

    Args:
        mass: Mass [kg]
        spring_constant: Spring constant [N/m]

    Returns:
        Critical damping coefficient [kg/s]

    Reference: \cite{Thornton2004}
    """
    return 2 * np.sqrt(spring_constant * mass)


def compute_damping_theta(
    damping: float,
    mass: float,
    spring_constant: float
) -> float:
    """
    Compute theta for damped oscillator.

    Theta = c / c_critical (damping ratio ζ)

    - theta < 1: Underdamped (oscillates)
    - theta = 1: Critical damping (optimal return)
    - theta > 1: Overdamped (slow return)

    For automotive suspension, theta ≈ 0.3-0.7 is typical.

    Args:
        damping: Damping coefficient [kg/s]
        mass: Mass [kg]
        spring_constant: Spring constant [N/m]

    Returns:
        Damping ratio (can be > 1)
    """
    c_crit = critical_damping(mass, spring_constant)
    if c_crit <= 0:
        return 0.0
    return damping / c_crit


# =============================================================================
# UNIFIED THETA CALCULATION
# =============================================================================

def compute_mechanical_theta(system: MechanicalSystem) -> float:
    """
    Compute unified theta for mechanical system.

    Args:
        system: MechanicalSystem to analyze

    Returns:
        theta in [0, 1]
    """
    if system.theoretical_max <= 0:
        return 0.0

    theta = system.efficiency / system.theoretical_max
    return np.clip(theta, 0.0, 1.0)


def classify_efficiency(theta: float) -> EfficiencyRegime:
    """Classify efficiency regime from theta."""
    if theta < 0.3:
        return EfficiencyRegime.WASTEFUL
    elif theta < 0.6:
        return EfficiencyRegime.TYPICAL
    elif theta < 0.9:
        return EfficiencyRegime.EFFICIENT
    else:
        return EfficiencyRegime.NEAR_IDEAL


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

MECHANICAL_SYSTEMS: Dict[str, MechanicalSystem] = {
    "car_engine": MechanicalSystem(
        name="Car Engine (Otto cycle)",
        system_type=SystemType.HEAT_ENGINE,
        input_power=100000,  # 100 kW fuel input
        output_power=25000,  # 25 kW output
        efficiency=0.25,
        theoretical_max=0.56,  # η_Otto for r=10
        temperature_hot=2500,
        temperature_cold=300,
    ),
    "diesel_truck": MechanicalSystem(
        name="Diesel Truck Engine",
        system_type=SystemType.HEAT_ENGINE,
        input_power=300000,
        output_power=120000,
        efficiency=0.40,
        theoretical_max=0.65,  # η_Diesel for r=20
        temperature_hot=2200,
        temperature_cold=300,
    ),
    "power_plant": MechanicalSystem(
        name="Combined Cycle Power Plant",
        system_type=SystemType.HEAT_ENGINE,
        input_power=1e9,
        output_power=6e8,
        efficiency=0.60,
        theoretical_max=0.75,  # With CCGT
        temperature_hot=1500,
        temperature_cold=300,
    ),
    "ev_motor": MechanicalSystem(
        name="EV Induction Motor",
        system_type=SystemType.ELECTRIC_MOTOR,
        input_power=150000,
        output_power=142500,
        efficiency=0.95,
        theoretical_max=0.98,
    ),
    "industrial_motor": MechanicalSystem(
        name="Industrial Motor (IE4)",
        system_type=SystemType.ELECTRIC_MOTOR,
        input_power=7500,
        output_power=7125,
        efficiency=0.95,
        theoretical_max=0.98,
    ),
    "lithium_battery": MechanicalSystem(
        name="Li-ion Battery",
        system_type=SystemType.BATTERY,
        input_power=1000,  # Charging
        output_power=950,  # Discharging
        efficiency=0.95,
        theoretical_max=0.99,  # OCV ratio
    ),
    "lead_acid_battery": MechanicalSystem(
        name="Lead-Acid Battery",
        system_type=SystemType.BATTERY,
        input_power=1000,
        output_power=800,
        efficiency=0.80,
        theoretical_max=0.95,
    ),
    "car_suspension": MechanicalSystem(
        name="Car Suspension",
        system_type=SystemType.OSCILLATOR,
        input_power=0,
        output_power=0,
        efficiency=0.5,  # ζ = 0.5 typical
        theoretical_max=1.0,  # ζ = 1 (critical)
    ),
    "manual_transmission": MechanicalSystem(
        name="Manual Transmission",
        system_type=SystemType.TRANSMISSION,
        input_power=100000,
        output_power=97000,
        efficiency=0.97,
        theoretical_max=0.99,
    ),
    "automatic_transmission": MechanicalSystem(
        name="Automatic (Torque Converter)",
        system_type=SystemType.TRANSMISSION,
        input_power=100000,
        output_power=85000,
        efficiency=0.85,
        theoretical_max=0.98,
    ),
}


def mechanical_theta_summary():
    """Print theta analysis for example mechanical systems."""
    print("=" * 70)
    print("MECHANICAL SYSTEMS THETA ANALYSIS (Efficiency)")
    print("=" * 70)
    print()
    print(f"{'System':<35} {'η':>8} {'η_max':>8} {'θ':>8} {'Regime':<12}")
    print("-" * 70)

    for name, system in MECHANICAL_SYSTEMS.items():
        theta = compute_mechanical_theta(system)
        regime = classify_efficiency(theta)
        print(f"{system.name:<35} "
              f"{system.efficiency:>8.2f} "
              f"{system.theoretical_max:>8.2f} "
              f"{theta:>8.3f} "
              f"{regime.value:<12}")

    print()
    print("Key: θ = η_actual / η_theoretical")
    print("     θ = 1 means operating at fundamental limit")


if __name__ == "__main__":
    mechanical_theta_summary()
