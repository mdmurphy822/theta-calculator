"""
Theta Interpolation Engine: Core computations for theta values.

This module implements multiple methods to compute theta, the fundamental
parameter that interpolates between quantum (θ=1) and classical (θ=0)
descriptions.

Each method approaches theta from a different physical principle:
1. Action ratio: θ = ℏ/S (quantum when action ≈ Planck constant)
2. Thermal ratio: θ = ℏω/(kT) (quantum when thermal energy < quantum energy)
3. Scale ratio: θ from Planck scale comparisons
4. Decoherence: θ = exp(-t/t_D) (exponential decay to classical)
5. Unified: Weighted combination of all methods

The convergence of these independent methods demonstrates that theta
is a real, measurable property of physical systems.
"""

import numpy as np
from typing import Callable, Optional, Dict
from dataclasses import dataclass

from .theta_state import ThetaState, PhysicalSystem, Regime
from ..constants.values import FundamentalConstants as FC
from ..constants.planck_units import PlanckUnits


@dataclass
class ThetaCalculator:
    """
    Core engine for computing theta values for physical systems.

    Theta emerges from the ratio of quantum to classical characteristic scales:

        θ = f(S_quantum / S_classical)

    where S represents relevant characteristic scales (action, length, time, etc.)

    This class provides multiple independent methods to compute theta,
    all of which should converge for a consistent physical description.
    """

    # Default weights for unified computation
    default_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.default_weights is None:
            self.default_weights = {
                "action_ratio": 0.30,
                "thermal_ratio": 0.20,
                "scale_ratio": 0.20,
                "decoherence": 0.30,
            }

    def compute_action_theta(self, system: PhysicalSystem) -> ThetaState:
        """
        Compute theta from action ratio: θ = ℏ/S

        The quantum/classical boundary is characterized by action ~ ℏ.
        This is the most fundamental definition of theta.

        When S ≈ ℏ (action at quantum scale): θ → 1 (quantum)
        When S >> ℏ (action much larger):     θ → 0 (classical)

        Physical basis:
        - Heisenberg uncertainty: ΔxΔp ≥ ℏ/2
        - Path integral: phases exp(iS/ℏ) random when S >> ℏ
        - Bohr-Sommerfeld: quantization when S ~ nℏ

        Args:
            system: Physical system with known or estimable action

        Returns:
            ThetaState with action-based theta value
        """
        h_bar = FC.h_bar.value

        # Get or estimate action
        action = system.estimate_action()

        # Compute theta
        if action <= h_bar:
            theta = 1.0  # Pure quantum
        else:
            theta = h_bar / action

        return ThetaState(
            theta=theta,
            system=system,
            quantum_scale=h_bar,
            classical_scale=action,
            proof_method="action_ratio",
            components={"action_ratio": theta},
            confidence=0.9 if system.action is not None else 0.7,
            validation_notes=f"S = {action:.2e} J·s, ℏ = {h_bar:.2e} J·s"
        )

    def compute_thermal_theta(self, system: PhysicalSystem) -> ThetaState:
        """
        Compute theta from thermal/quantum energy ratio.

        Quantum effects dominate when quantum energy exceeds thermal energy:
            E_quantum = ℏω >> kT = E_thermal

        θ = ℏω / (kT)  clamped to [0, 1]

        where ω is estimated from the system's characteristic energy.

        Physical basis:
        - At low T: quantum zero-point motion dominates
        - At high T: thermal fluctuations dominate
        - Crossover at ℏω ~ kT

        Examples:
        - Room temperature: kT ≈ 0.026 eV
        - Atomic transitions: ℏω ≈ few eV → θ large
        - Phonons in solids: depends on Debye temperature

        Args:
            system: Physical system with temperature

        Returns:
            ThetaState with thermal-based theta value
        """
        h_bar = FC.h_bar.value
        k = FC.k.value

        # Estimate characteristic frequency from energy
        omega = system.energy / h_bar

        # Thermal energy
        E_thermal = k * system.temperature

        # Quantum energy at this frequency
        E_quantum = h_bar * omega

        # Compute theta
        if E_thermal == 0:
            theta = 1.0  # Absolute zero → pure quantum
        else:
            ratio = E_quantum / E_thermal
            theta = min(1.0, ratio / (1 + ratio))  # Smooth sigmoid-like function

        return ThetaState(
            theta=theta,
            system=system,
            quantum_scale=E_quantum,
            classical_scale=E_thermal,
            proof_method="thermal_ratio",
            components={"thermal_ratio": theta},
            confidence=0.8,
            validation_notes=f"E_q = {E_quantum:.2e} J, E_th = {E_thermal:.2e} J"
        )

    def compute_scale_theta(self, system: PhysicalSystem) -> ThetaState:
        """
        Compute theta from length/mass scale ratios to Planck scale.

        Systems close to Planck scale are quantum; large systems are classical.

        θ_length = l_P / L  (1 at Planck scale, 0 for large systems)
        θ_mass = m_P / m    (1 at Planck mass, 0 for heavy systems)

        Combined via geometric mean: θ = √(θ_length × θ_mass)

        Physical basis:
        - Planck scale is where quantum and gravity are equal
        - de Broglie wavelength λ = h/(mv) → quantum when λ ~ L
        - Schwarzschild radius r_s = 2GM/c² → classical when r_s << L

        Args:
            system: Physical system with mass and length scale

        Returns:
            ThetaState with scale-based theta value
        """
        l_P = PlanckUnits.planck_length()
        m_P = PlanckUnits.planck_mass()

        # Length contribution (quantum when L ~ l_P)
        if system.length_scale > 0:
            theta_length = min(1.0, l_P / system.length_scale)
        else:
            theta_length = 1.0

        # Mass contribution (complex relationship)
        # Light particles are quantum, but massive particles can also be quantum
        # if well-isolated. Use de Broglie comparison instead.
        if system.mass > 0:
            # Compare de Broglie wavelength to system size
            lambda_dB = system.de_broglie_wavelength
            if lambda_dB is not None and lambda_dB > 0:
                theta_dB = min(1.0, lambda_dB / system.length_scale)
            else:
                # Fallback to Planck mass ratio
                theta_dB = min(1.0, m_P / system.mass) if system.mass > m_P else 1.0
        else:
            theta_dB = 1.0

        # Geometric mean of contributions
        theta = np.sqrt(theta_length * theta_dB)

        return ThetaState(
            theta=theta,
            system=system,
            quantum_scale=l_P,
            classical_scale=system.length_scale,
            proof_method="scale_ratio",
            components={
                "length_scale": theta_length,
                "wavelength_scale": theta_dB,
            },
            confidence=0.7,
            validation_notes=f"L = {system.length_scale:.2e} m, l_P = {l_P:.2e} m"
        )

    def compute_decoherence_theta(
        self,
        system: PhysicalSystem,
        environment_coupling: Optional[float] = None,
        observation_time: Optional[float] = None
    ) -> ThetaState:
        """
        Compute theta from decoherence dynamics.

        Decoherence is the mechanism by which quantum systems become classical
        through interaction with their environment.

        Decoherence time: t_D = ℏ / E_coupling
        θ = exp(-t_observation / t_D)

        As t → ∞: θ → 0 (classical due to decoherence)
        As t → 0: θ → 1 (quantum, not yet decohered)

        Physical basis:
        - Environment entangles with system
        - Off-diagonal density matrix elements decay
        - Pointer states (classical states) are selected

        Args:
            system: Physical system
            environment_coupling: Coupling energy to environment (J)
                                 If None, estimated from kT
            observation_time: Time over which system is observed (s)
                            If None, estimated from length/c

        Returns:
            ThetaState with decoherence-based theta value
        """
        h_bar = FC.h_bar.value
        k = FC.k.value
        c = FC.c.value

        # Estimate environment coupling if not provided
        if environment_coupling is None:
            # Thermal bath coupling ~ kT
            environment_coupling = k * system.temperature

        # Estimate observation time if not provided
        if observation_time is None:
            if system.characteristic_time is not None:
                observation_time = system.characteristic_time
            else:
                # Light-crossing time as minimal timescale
                observation_time = system.length_scale / c

        # Decoherence time
        if environment_coupling > 0:
            t_D = h_bar / environment_coupling
        else:
            t_D = float('inf')  # No decoherence

        # Theta decays exponentially
        if t_D == float('inf'):
            theta = 1.0
        else:
            theta = np.exp(-observation_time / t_D)

        return ThetaState(
            theta=theta,
            system=system,
            quantum_scale=t_D,
            classical_scale=observation_time,
            proof_method="decoherence",
            components={"decoherence": theta},
            confidence=0.85 if environment_coupling is not None else 0.6,
            validation_notes=f"t_D = {t_D:.2e} s, t_obs = {observation_time:.2e} s"
        )

    def compute_unified_theta(
        self,
        system: PhysicalSystem,
        weights: Optional[Dict[str, float]] = None
    ) -> ThetaState:
        """
        Compute unified theta using weighted combination of all methods.

        This is the primary theta computation, combining evidence from
        multiple independent physical principles.

        θ_unified = Σ_i w_i × θ_i

        The convergence of independent methods validates that theta is
        a real property of the physical system.

        Args:
            system: Physical system to analyze
            weights: Custom weights for each method. Keys:
                    - action_ratio
                    - thermal_ratio
                    - scale_ratio
                    - decoherence
                    Must sum to 1.0. If None, uses default weights.

        Returns:
            ThetaState with unified theta value and all components
        """
        if weights is None:
            weights = self.default_weights.copy()

        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Compute individual thetas
        action_state = self.compute_action_theta(system)
        thermal_state = self.compute_thermal_theta(system)
        scale_state = self.compute_scale_theta(system)
        decoherence_state = self.compute_decoherence_theta(system)

        # Weighted combination
        theta = (
            weights.get("action_ratio", 0) * action_state.theta +
            weights.get("thermal_ratio", 0) * thermal_state.theta +
            weights.get("scale_ratio", 0) * scale_state.theta +
            weights.get("decoherence", 0) * decoherence_state.theta
        )

        # Collect all components
        components = {
            "action_ratio": action_state.theta,
            "thermal_ratio": thermal_state.theta,
            "scale_ratio": scale_state.theta,
            "decoherence": decoherence_state.theta,
        }

        # Compute uncertainty from spread
        thetas = list(components.values())
        theta_std = np.std(thetas)

        # Compute confidence from agreement
        cv = theta_std / max(0.01, np.mean(thetas))  # Coefficient of variation
        confidence = max(0, 1 - cv)

        return ThetaState(
            theta=theta,
            theta_uncertainty=theta_std,
            system=system,
            proof_method="unified",
            components=components,
            confidence=confidence,
            validation_notes=(
                f"Components: action={action_state.theta:.3f}, "
                f"thermal={thermal_state.theta:.3f}, "
                f"scale={scale_state.theta:.3f}, "
                f"decoherence={decoherence_state.theta:.3f}"
            )
        )

    def compute_all_methods(
        self,
        system: PhysicalSystem
    ) -> Dict[str, ThetaState]:
        """
        Compute theta using all available methods.

        Returns a dictionary with results from each method for comparison.

        Args:
            system: Physical system to analyze

        Returns:
            Dict mapping method name to ThetaState
        """
        return {
            "action_ratio": self.compute_action_theta(system),
            "thermal_ratio": self.compute_thermal_theta(system),
            "scale_ratio": self.compute_scale_theta(system),
            "decoherence": self.compute_decoherence_theta(system),
            "unified": self.compute_unified_theta(system),
        }

    def analyze_convergence(
        self,
        system: PhysicalSystem
    ) -> Dict[str, float]:
        """
        Analyze how well different theta methods converge.

        Good convergence (low spread) indicates theta is well-defined.
        Poor convergence may indicate transition regime or missing physics.

        Args:
            system: Physical system to analyze

        Returns:
            Dict with convergence statistics
        """
        results = self.compute_all_methods(system)

        # Exclude unified from convergence analysis
        thetas = [
            results["action_ratio"].theta,
            results["thermal_ratio"].theta,
            results["scale_ratio"].theta,
            results["decoherence"].theta,
        ]

        return {
            "mean": float(np.mean(thetas)),
            "std": float(np.std(thetas)),
            "min": float(np.min(thetas)),
            "max": float(np.max(thetas)),
            "range": float(np.max(thetas) - np.min(thetas)),
            "coefficient_of_variation": float(np.std(thetas) / max(0.01, np.mean(thetas))),
        }


def theta_interpolation_function(
    theta: float,
    quantum_description: Callable,
    classical_description: Callable,
    *args, **kwargs
) -> float:
    """
    Generic interpolation between quantum and classical descriptions.

    This is THE fundamental operation showing theta's role:

        result = θ × quantum_description(...) + (1-θ) × classical_description(...)

    Theta smoothly connects the two limiting descriptions.

    Args:
        theta: The theta value (0 to 1)
        quantum_description: Function returning quantum prediction
        classical_description: Function returning classical prediction
        *args, **kwargs: Arguments passed to both functions

    Returns:
        Interpolated result

    Example:
        # Energy of harmonic oscillator
        def quantum_energy(n, omega):
            return h_bar * omega * (n + 0.5)  # Zero-point energy

        def classical_energy(n, omega):
            return n * h_bar * omega  # No zero-point energy

        # Interpolated energy
        E = theta_interpolation_function(
            theta=0.7,
            quantum_description=quantum_energy,
            classical_description=classical_energy,
            n=0, omega=1e15
        )
    """
    q_val = quantum_description(*args, **kwargs)
    c_val = classical_description(*args, **kwargs)
    return theta * q_val + (1 - theta) * c_val


def estimate_theta_quick(
    mass: float,
    length: float,
    temperature: float
) -> float:
    """
    Quick theta estimate without creating full PhysicalSystem.

    Useful for rapid surveys or interactive exploration.

    Args:
        mass: Mass in kg
        length: Length scale in meters
        temperature: Temperature in Kelvin

    Returns:
        Estimated theta value
    """
    h_bar = FC.h_bar.value
    k = FC.k.value
    c = FC.c.value
    l_P = PlanckUnits.planck_length()

    # Quick estimates
    action = mass * c**2 * (length / c)  # E × t
    theta_action = min(1.0, h_bar / action) if action > 0 else 1.0

    theta_scale = min(1.0, l_P / length) if length > 0 else 1.0

    E_thermal = k * temperature
    theta_thermal = min(1.0, h_bar * c / (length * E_thermal)) if E_thermal > 0 else 1.0

    # Simple average
    return (theta_action + theta_scale + theta_thermal) / 3


# Convenience function for common use case
def compute_theta(system: PhysicalSystem) -> ThetaState:
    """
    Compute theta for a physical system using unified method.

    This is the recommended entry point for theta calculations.

    Args:
        system: Physical system to analyze

    Returns:
        ThetaState with unified theta value
    """
    calculator = ThetaCalculator()
    return calculator.compute_unified_theta(system)
