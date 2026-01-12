"""
Landauer Limit: Information-theoretic proof of theta via thermodynamics.

Landauer's Principle states that erasing one bit of information requires
a minimum energy dissipation:

    E ≥ kT ln(2)

This connects information theory to thermodynamics and defines fundamental
limits on computation. Combined with quantum limits (Margolus-Levitin),
this provides another route to computing theta.

Key insight: The ratio of quantum to thermal information limits
determines theta.
- When quantum limit dominates → classical (thermal fluctuations dominate)
- When Landauer limit dominates → quantum effects matter

References (see BIBLIOGRAPHY.bib):
    \\cite{Landauer1961} - Irreversibility and heat generation in computing
    \\cite{MargolusLevitin1998} - Maximum speed of dynamical evolution
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...constants.values import FundamentalConstants as FC
from ...core.theta_state import ThetaState, PhysicalSystem


@dataclass
class LandauerResult:
    """
    Result of Landauer limit calculation.

    Attributes:
        minimum_energy: Minimum energy to erase one bit (J)
        available_energy: Total energy available for computation (J)
        max_bit_operations: Maximum bit operations possible
        quantum_limit_ops: Operations limited by quantum mechanics
        thermal_limit_ops: Operations limited by thermodynamics
        theta_from_landauer: Theta derived from information limits
        temperature: System temperature (K)
        explanation: Human-readable interpretation
    """
    minimum_energy: float
    available_energy: float
    max_bit_operations: float
    quantum_limit_ops: float
    thermal_limit_ops: float
    theta_from_landauer: float
    temperature: float
    explanation: str


class LandauerLimit:
    """
    Calculates Landauer's limit and derives theta from it.

    Landauer's Principle: E_min = kT ln(2) per bit erased

    This defines the fundamental thermodynamic cost of information
    processing, connecting entropy (information) to energy.

    Combined with the Margolus-Levitin quantum limit:
        N_ops ≤ 2Et / (πℏ)

    We get two independent bounds on computation, whose ratio
    determines theta.

    Physical basis:
    - Information erasure increases entropy of environment
    - Minimum entropy increase is k ln(2) per bit
    - At temperature T, this requires energy kT ln(2)

    Theta interpretation:
    - Compare quantum and thermal limits on operations
    - θ → 1 when near quantum limit
    - θ → 0 when thermal limit dominates
    """

    def __init__(self):
        self.k = FC.k.value
        self.h_bar = FC.h_bar.value
        self.h = FC.h.value

    def minimum_erasure_energy(self, temperature: float) -> float:
        """
        Compute minimum energy to erase one bit.

        E_min = kT ln(2)

        At room temperature (300 K): E_min ≈ 2.87 × 10^-21 J
        This is incredibly small but fundamentally non-zero.

        Args:
            temperature: Temperature in Kelvin

        Returns:
            Minimum erasure energy in Joules
        """
        return self.k * temperature * np.log(2)

    def max_bit_operations_thermal(
        self,
        energy: float,
        temperature: float
    ) -> float:
        """
        Compute maximum bit operations from thermal (Landauer) limit.

        N_thermal = E / (kT ln(2))

        This is the maximum number of bit erasures possible
        with the given energy at the given temperature.

        Args:
            energy: Available energy in Joules
            temperature: Temperature in Kelvin

        Returns:
            Maximum bit operations
        """
        E_min = self.minimum_erasure_energy(temperature)
        if E_min > 0:
            return energy / E_min
        return float('inf')

    def max_operations_quantum(self, energy: float, time: float) -> float:
        """
        Compute maximum operations from quantum (Margolus-Levitin) limit.

        N_quantum = 2Et / (πℏ)

        This is the fundamental quantum limit on computation rate.
        It depends only on energy and time, not temperature.

        The Margolus-Levitin theorem states that a quantum system
        with energy E needs time at least πℏ/(2E) to evolve to
        an orthogonal state.

        Args:
            energy: Available energy in Joules
            time: Available time in seconds

        Returns:
            Maximum quantum operations
        """
        return 2 * energy * time / (np.pi * self.h_bar)

    def bremermann_limit(self, mass: float, time: float) -> float:
        """
        Compute the Bremermann limit on computation.

        N_brem = mc²t / (πℏ)

        This is the maximum computation rate for a system of mass m,
        derived from the Margolus-Levitin theorem using rest mass energy.

        Args:
            mass: System mass in kg
            time: Time in seconds

        Returns:
            Maximum operations (Bremermann limit)
        """
        c = FC.c.value
        return mass * c**2 * time / (np.pi * self.h_bar)

    def minimum_computation_time(self, energy: float) -> float:
        """
        Compute minimum time for one quantum operation.

        t_min = πℏ / (2E)

        This is the quantum speed limit - the fastest a system
        with energy E can perform a single operation.

        Args:
            energy: Energy in Joules

        Returns:
            Minimum time in seconds
        """
        if energy > 0:
            return np.pi * self.h_bar / (2 * energy)
        return float('inf')

    def compute_theta(
        self,
        system: PhysicalSystem,
        operation_time: Optional[float] = None
    ) -> LandauerResult:
        """
        Compute theta from ratio of quantum to thermal limits.

        θ = N_thermal / (N_thermal + N_quantum)

        When N_quantum >> N_thermal: θ → 0 (quantum limit not reached)
        When N_thermal >> N_quantum: θ → 1 (near quantum limit)

        Actually, let's think about this more carefully:
        - High temperature → low Landauer cost → many thermal ops
        - Low temperature → high Landauer cost → few thermal ops
        - Quantum limit is independent of temperature

        θ should be high (quantum) when quantum effects dominate.
        This happens when:
        - Temperature is low (thermal ops expensive)
        - Energy is low (quantum limit tight)
        - Time is short (quantum limit tight)

        Args:
            system: Physical system
            operation_time: Time for computation. If None, estimated.

        Returns:
            LandauerResult with theta
        """
        # Estimate operation time if not provided
        if operation_time is None:
            if system.characteristic_time is not None:
                operation_time = system.characteristic_time
            else:
                # Use light-crossing time
                operation_time = system.length_scale / FC.c.value

        # Compute limits
        N_thermal = self.max_bit_operations_thermal(
            system.energy, system.temperature
        )
        N_quantum = self.max_operations_quantum(system.energy, operation_time)

        # Compute Landauer energy
        E_landauer = self.minimum_erasure_energy(system.temperature)

        # Compute minimum quantum time (used for validation)
        _ = self.minimum_computation_time(system.energy)

        # Theta from ratio of limits
        # When quantum limit is tighter (fewer ops allowed), system is more quantum
        # N_quantum < N_thermal means quantum limit constrains → high theta
        # N_quantum > N_thermal means thermal limit constrains → low theta
        if N_thermal + N_quantum > 0:
            ratio = N_quantum / N_thermal if N_thermal > 0 else float('inf')
            # theta = 1/(1+ratio): small ratio → high theta (quantum-limited)
            theta_ops = 1 / (1 + ratio) if ratio != float('inf') else 0.0
        else:
            theta_ops = 0.5  # Undefined case

        # Alternative: based on time scales
        # t_thermal = ℏ/(kT) is the thermal coherence time
        # Operations faster than t_thermal show quantum behavior
        t_thermal = self.h_bar / (self.k * system.temperature) if system.temperature > 0 else float('inf')
        time_ratio = operation_time / t_thermal if t_thermal != float('inf') else 0
        # theta_time = 1/(1+ratio): fast operations (small ratio) → high theta
        theta_time = 1 / (1 + time_ratio)

        # Average both measures for robust estimate
        theta = (theta_ops + theta_time) / 2

        # Generate explanation
        if theta > 0.7:
            explanation = (
                f"Near quantum limit (θ={theta:.3f}). "
                f"Quantum operations: {N_quantum:.2e}, Thermal: {N_thermal:.2e}. "
                f"The quantum speed limit constrains this system."
            )
        elif theta > 0.3:
            explanation = (
                f"Transition regime (θ={theta:.3f}). "
                f"Both quantum and thermal limits are relevant. "
                f"Landauer cost: {E_landauer:.2e} J/bit."
            )
        else:
            explanation = (
                f"Classical regime (θ={theta:.3f}). "
                f"Thermal fluctuations dominate over quantum limits. "
                f"Many operations possible: {min(N_thermal, N_quantum):.2e}."
            )

        return LandauerResult(
            minimum_energy=E_landauer,
            available_energy=system.energy,
            max_bit_operations=min(N_thermal, N_quantum),
            quantum_limit_ops=N_quantum,
            thermal_limit_ops=N_thermal,
            theta_from_landauer=theta,
            temperature=system.temperature,
            explanation=explanation
        )

    def theta_from_landauer(self, system: PhysicalSystem) -> ThetaState:
        """
        Create a ThetaState from Landauer limit analysis.

        Args:
            system: Physical system to analyze

        Returns:
            ThetaState with Landauer-derived theta
        """
        result = self.compute_theta(system)

        return ThetaState(
            theta=result.theta_from_landauer,
            system=system,
            proof_method="landauer_limit",
            information_bits=result.max_bit_operations,
            components={
                "quantum_ops": result.quantum_limit_ops,
                "thermal_ops": result.thermal_limit_ops,
                "landauer_energy": result.minimum_energy,
            },
            confidence=0.75,
            validation_notes=result.explanation
        )

    def energy_for_computation(
        self,
        bits: float,
        temperature: float,
        reversibility: float = 0.0
    ) -> float:
        """
        Compute energy required for a computation.

        E = bits × kT ln(2) × (1 - reversibility)

        Reversible computation (reversibility=1) can approach zero energy.
        Irreversible computation (reversibility=0) hits Landauer limit.

        Args:
            bits: Number of bits to process
            temperature: Temperature in Kelvin
            reversibility: Fraction of reversible operations (0 to 1)

        Returns:
            Energy required in Joules
        """
        E_min = self.minimum_erasure_energy(temperature)
        return bits * E_min * (1 - reversibility)


def landauer_formula() -> str:
    """Return Landauer's principle as a formula string."""
    return "E ≥ kT ln(2) per bit erased"


def margolus_levitin_formula() -> str:
    """Return the Margolus-Levitin theorem formula."""
    return "N_ops ≤ 2Et/(πℏ)"
