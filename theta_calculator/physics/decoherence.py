"""
Decoherence: The mechanism by which quantum systems become classical.

Decoherence explains why macroscopic objects don't exhibit quantum behavior
despite being made of quantum particles. Environmental interaction causes
quantum superpositions to decay into classical mixtures.

Key concepts:
- Pointer states: Classical-like states that survive decoherence
- Einselection: Environment-induced superselection
- Quantum Darwinism: Classical states survive through redundancy

Decoherence provides a physical mechanism for the quantum-classical transition,
directly computing how theta decays from 1 (quantum) to 0 (classical).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from ..constants.values import FundamentalConstants as FC
from ..core.theta_state import ThetaState, PhysicalSystem


@dataclass
class DecoherenceResult:
    """
    Result of decoherence calculation.

    Attributes:
        decoherence_time: Time scale for quantum-to-classical transition (s)
        decoherence_rate: Rate of coherence loss (1/s)
        coherence_remaining: Fraction of quantum coherence remaining
        theta: Theta value from decoherence analysis
        pointer_state_count: Estimated number of pointer states
        observation_time: Time over which system was observed
        explanation: Human-readable interpretation
    """
    decoherence_time: float
    decoherence_rate: float
    coherence_remaining: float
    theta: float
    pointer_state_count: Optional[int]
    observation_time: float
    explanation: str


class DecoherenceCalculator:
    """
    Calculates decoherence effects and derives theta from them.

    Decoherence is the physical process that explains the quantum-classical
    transition. When a quantum system interacts with its environment:
    1. Entanglement spreads quantum information to environment
    2. Off-diagonal density matrix elements decay exponentially
    3. System appears classical (no interference)

    The decoherence time t_D determines how fast theta decays:
        θ(t) = exp(-t/t_D)

    For most macroscopic objects at room temperature, t_D is incredibly
    short (femtoseconds or less), explaining classical behavior.
    """

    def __init__(self):
        self.h_bar = FC.h_bar.value
        self.k = FC.k.value
        self.c = FC.c.value

    def thermal_decoherence_time(
        self,
        mass: float,
        temperature: float,
        separation: float
    ) -> float:
        """
        Compute decoherence time from thermal photon scattering.

        For a superposition of positions separated by Δx in a thermal
        bath at temperature T:

            t_D = ℏ² / (Λ_th² × kT × m)

        where Λ_th = ℏ / √(2mkT) is the thermal de Broglie wavelength.

        Simplified for spatial superposition:
            t_D ≈ ℏ / (kT) × (Λ_th / Δx)²

        Args:
            mass: Particle mass in kg
            temperature: Environment temperature in K
            separation: Superposition separation in meters

        Returns:
            Decoherence time in seconds
        """
        if temperature == 0:
            return float('inf')  # No thermal decoherence at T=0

        # Thermal de Broglie wavelength
        lambda_th = self.h_bar / np.sqrt(2 * mass * self.k * temperature)

        # Decoherence time
        # This is a simplified model; actual rates depend on environment details
        t_D = (self.h_bar / (self.k * temperature)) * (lambda_th / separation)**2

        return t_D

    def collisional_decoherence_time(
        self,
        mass: float,
        cross_section: float,
        gas_density: float,
        temperature: float
    ) -> float:
        """
        Compute decoherence time from gas molecule collisions.

        When a quantum system collides with gas molecules, each
        collision can "measure" the position.

            t_D = 1 / (n × σ × v_th)

        where n is gas density, σ is cross-section, v_th is thermal velocity.

        Args:
            mass: System mass in kg
            cross_section: Scattering cross-section in m²
            gas_density: Number density of gas molecules (m⁻³)
            temperature: Temperature in K

        Returns:
            Decoherence time in seconds
        """
        if gas_density == 0 or temperature == 0:
            return float('inf')

        # Typical gas molecule mass (nitrogen)
        m_gas = 28 * 1.66e-27  # kg

        # Thermal velocity
        v_th = np.sqrt(8 * self.k * temperature / (np.pi * m_gas))

        # Collision rate
        collision_rate = gas_density * cross_section * v_th

        if collision_rate == 0:
            return float('inf')

        return 1.0 / collision_rate

    def gravitational_decoherence_time(
        self,
        mass: float,
        separation: float
    ) -> float:
        """
        Compute decoherence time from gravitational self-interaction.

        Penrose's proposal: gravity causes objective collapse when
        superposed masses have significant gravitational self-energy difference.

            t_D ≈ ℏ / E_G

        where E_G is the gravitational self-energy difference.

        For a spherical mass split into two locations:
            E_G ≈ G m² / R (simplified)

        Args:
            mass: System mass in kg
            separation: Superposition separation in meters

        Returns:
            Decoherence time in seconds (Penrose estimate)
        """
        G = FC.G.value

        # Gravitational self-energy difference
        E_G = G * mass**2 / separation

        if E_G == 0:
            return float('inf')

        return self.h_bar / E_G

    def compute_total_decoherence_time(
        self,
        system: PhysicalSystem,
        superposition_size: Optional[float] = None
    ) -> Tuple[float, dict]:
        """
        Compute total decoherence time from all mechanisms.

        Combines thermal, collisional, and gravitational decoherence.
        The fastest mechanism dominates.

            1/t_D_total = 1/t_D_thermal + 1/t_D_collision + 1/t_D_gravity

        Args:
            system: Physical system
            superposition_size: Size of superposition. If None, uses length_scale.

        Returns:
            Tuple of (total decoherence time, dict of individual times)
        """
        if superposition_size is None:
            superposition_size = system.length_scale

        # Thermal decoherence
        t_thermal = self.thermal_decoherence_time(
            system.mass, system.temperature, superposition_size
        )

        # Collisional (assume air at STP for macroscopic objects)
        # Air density ~ 2.5e25 molecules/m³, cross-section ~ 1e-19 m²
        if system.mass > 1e-15:  # Macroscopic
            t_collision = self.collisional_decoherence_time(
                system.mass, 1e-19, 2.5e25, system.temperature
            )
        else:
            t_collision = float('inf')

        # Gravitational (Penrose mechanism)
        t_gravity = self.gravitational_decoherence_time(
            system.mass, superposition_size
        )

        # Combine rates (add rates, not times)
        rates = []
        if t_thermal != float('inf'):
            rates.append(1/t_thermal)
        if t_collision != float('inf'):
            rates.append(1/t_collision)
        if t_gravity != float('inf'):
            rates.append(1/t_gravity)

        if rates:
            t_total = 1.0 / sum(rates)
        else:
            t_total = float('inf')

        return t_total, {
            'thermal': t_thermal,
            'collisional': t_collision,
            'gravitational': t_gravity,
            'total': t_total,
        }

    def theta_from_decoherence(
        self,
        system: PhysicalSystem,
        observation_time: Optional[float] = None
    ) -> DecoherenceResult:
        """
        Compute theta from decoherence dynamics.

        θ = exp(-t_obs / t_D)

        This directly shows how quantum coherence (theta) decays
        over time due to environmental interaction.

        Args:
            system: Physical system
            observation_time: Time since preparation. If None, estimated.

        Returns:
            DecoherenceResult with theta and all decoherence details
        """
        # Get observation time
        if observation_time is None:
            if system.characteristic_time is not None:
                observation_time = system.characteristic_time
            else:
                observation_time = system.length_scale / self.c

        # Compute decoherence times
        t_D, time_breakdown = self.compute_total_decoherence_time(system)

        # Compute theta
        if t_D == float('inf'):
            theta = 1.0
            coherence = 1.0
        else:
            theta = np.exp(-observation_time / t_D)
            coherence = theta

        # Decoherence rate
        rate = 1.0 / t_D if t_D != float('inf') else 0.0

        # Estimate pointer states (rough: ~1 classical state per thermal energy)
        pointer_count = None
        if system.number_of_particles:
            pointer_count = system.number_of_particles

        # Generate explanation
        dominant = "thermal"
        min_time = min(time_breakdown['thermal'], time_breakdown['collisional'],
                      time_breakdown['gravitational'])
        if min_time == time_breakdown['collisional']:
            dominant = "collisional"
        elif min_time == time_breakdown['gravitational']:
            dominant = "gravitational"

        if theta > 0.9:
            explanation = (
                f"System maintains quantum coherence (θ={theta:.3f}). "
                f"Decoherence time t_D = {t_D:.2e} s >> observation time. "
                f"Quantum superposition survives."
            )
        elif theta > 0.1:
            explanation = (
                f"Partial decoherence (θ={theta:.3f}). "
                f"Decoherence time t_D = {t_D:.2e} s ≈ observation time. "
                f"System in quantum-classical transition."
            )
        else:
            explanation = (
                f"Strong decoherence (θ={theta:.3f}). "
                f"Decoherence time t_D = {t_D:.2e} s << observation time. "
                f"Dominant mechanism: {dominant}. "
                f"System effectively classical."
            )

        return DecoherenceResult(
            decoherence_time=t_D,
            decoherence_rate=rate,
            coherence_remaining=coherence,
            theta=theta,
            pointer_state_count=pointer_count,
            observation_time=observation_time,
            explanation=explanation
        )

    def to_theta_state(self, system: PhysicalSystem) -> ThetaState:
        """
        Create ThetaState from decoherence analysis.

        Args:
            system: Physical system

        Returns:
            ThetaState with decoherence-derived theta
        """
        result = self.theta_from_decoherence(system)

        return ThetaState(
            theta=result.theta,
            system=system,
            proof_method="decoherence_dynamics",
            quantum_scale=result.decoherence_time,
            classical_scale=result.observation_time,
            components={
                "decoherence": result.theta,
                "coherence_remaining": result.coherence_remaining,
            },
            confidence=0.85,
            validation_notes=result.explanation
        )
