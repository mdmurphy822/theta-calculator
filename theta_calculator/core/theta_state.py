"""
ThetaState: Core data structures for representing quantum-classical states.

Theta (θ) is the fundamental interpolation parameter between:
- θ = 1: Fully quantum description (superposition, entanglement, interference)
- θ = 0: Fully classical description (definite states, deterministic evolution)

For any physical observable O:
    O = θ × O_quantum + (1-θ) × O_classical

This module defines the data structures that hold theta values and
physical system parameters throughout the proof computation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np


class Regime(Enum):
    """
    Physical regime classification based on theta value.

    The regime determines which physical description is appropriate:
    - QUANTUM: Use full quantum mechanics (θ > 0.99)
    - CLASSICAL: Use classical physics (θ < 0.01)
    - TRANSITION: Both descriptions contribute (0.01 ≤ θ ≤ 0.99)
    - PLANCK: At Planck scale, quantum gravity required (special case)
    - UNDEFINED: Theta not yet computed or invalid
    """
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    TRANSITION = "transition"
    PLANCK = "planck"
    UNDEFINED = "undefined"

    def __str__(self) -> str:
        descriptions = {
            "quantum": "Quantum regime (θ → 1): Superposition and interference dominate",
            "classical": "Classical regime (θ → 0): Definite states and trajectories",
            "transition": "Transition regime (0 < θ < 1): Both descriptions needed",
            "planck": "Planck regime: Quantum gravity effects, θ undefined",
            "undefined": "Regime not determined",
        }
        return descriptions.get(self.value, self.value)


@dataclass
class PhysicalSystem:
    """
    Represents a physical system with measurable properties.

    These properties determine the system's theta value through
    various computation methods (action ratio, decoherence, etc.)

    Attributes:
        name: Descriptive name for the system
        mass: Total mass in kg
        length_scale: Characteristic length in meters
        energy: Total or characteristic energy in Joules
        temperature: Temperature in Kelvin (for thermal theta)
        action: Action in J·s if known (S = ∫L dt)
        entropy: Entropy in J/K if known
        characteristic_time: Characteristic timescale in seconds
        number_of_particles: Particle count for statistical systems
        metadata: Additional system-specific data
    """
    name: str
    mass: float
    length_scale: float
    energy: float
    temperature: float

    # Optional parameters
    action: Optional[float] = None
    entropy: Optional[float] = None
    characteristic_time: Optional[float] = None
    number_of_particles: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate system parameters."""
        if self.mass < 0:
            raise ValueError(f"Mass must be non-negative, got {self.mass}")
        if self.length_scale <= 0:
            raise ValueError(f"Length scale must be positive, got {self.length_scale}")
        if self.energy < 0:
            raise ValueError(f"Energy must be non-negative, got {self.energy}")
        if self.temperature < 0:
            raise ValueError(f"Temperature must be non-negative, got {self.temperature}")

    @property
    def rest_energy(self) -> float:
        """Rest mass energy E = mc²."""
        from ..constants.values import c
        return self.mass * c**2

    @property
    def de_broglie_wavelength(self) -> Optional[float]:
        """
        de Broglie wavelength λ = h/p.

        Returns None if momentum can't be estimated.
        """
        from ..constants.values import h
        # Estimate momentum from energy (non-relativistic)
        if self.mass > 0 and self.energy > 0:
            # p = √(2mE) for kinetic energy
            kinetic = max(0, self.energy - self.rest_energy)
            if kinetic > 0:
                p = np.sqrt(2 * self.mass * kinetic)
                return h / p
        return None

    @property
    def compton_wavelength(self) -> float:
        """Compton wavelength λ_C = h/(mc)."""
        from ..constants.values import h, c
        if self.mass > 0:
            return h / (self.mass * c)
        return float('inf')

    @property
    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius r_s = 2GM/c²."""
        from ..constants.values import G, c
        return 2 * G * self.mass / c**2

    def estimate_action(self) -> float:
        """
        Estimate the characteristic action S = E × t.

        Uses characteristic_time if provided, otherwise estimates
        from length_scale / c.
        """
        from ..constants.values import c
        if self.action is not None:
            return self.action

        if self.characteristic_time is not None:
            return self.energy * self.characteristic_time

        # Estimate time from light-crossing time
        t_cross = self.length_scale / c
        return self.energy * t_cross

    def __repr__(self) -> str:
        return (
            f"PhysicalSystem(name='{self.name}', "
            f"mass={self.mass:.2e} kg, "
            f"L={self.length_scale:.2e} m, "
            f"E={self.energy:.2e} J, "
            f"T={self.temperature:.1f} K)"
        )


@dataclass
class ThetaState:
    """
    Central data structure representing a theta state.

    Theta (θ) represents the interpolation between quantum and classical:
    - θ = 0: Fully classical description
    - θ = 1: Fully quantum description

    Theta is computed from the ratio of quantum to classical
    characteristic scales of a physical system.

    Attributes:
        theta: The theta value (0.0 to 1.0)
        theta_uncertainty: Uncertainty in theta computation
        system: The physical system this state describes
        regime: Classification (QUANTUM, CLASSICAL, TRANSITION)
        components: Individual theta contributions from each method
        quantum_scale: Characteristic quantum scale used
        classical_scale: Characteristic classical scale used
        information_bits: Information content if computed
        entropy_normalized: S/S_max if computed
        proof_method: Which method computed this theta
        confidence: Confidence in the theta value (0-1)
        is_validated: Whether consistency checks passed
        validation_notes: Notes from validation process
    """
    theta: float
    theta_uncertainty: float = 0.0
    system: Optional[PhysicalSystem] = None
    regime: Regime = Regime.UNDEFINED
    components: Dict[str, float] = field(default_factory=dict)
    quantum_scale: Optional[float] = None
    classical_scale: Optional[float] = None
    information_bits: Optional[float] = None
    entropy_normalized: Optional[float] = None
    proof_method: Optional[str] = None
    confidence: float = 0.0
    is_validated: bool = False
    validation_notes: str = ""

    def __post_init__(self):
        """Validate and classify the theta state."""
        # Clamp theta to valid range [0, 1]
        self.theta = max(0.0, min(1.0, self.theta))

        # Auto-classify regime if not set
        if self.regime == Regime.UNDEFINED:
            self._classify_regime()

    def _classify_regime(self):
        """Classify the regime based on theta value."""
        if self.theta > 0.99:
            self.regime = Regime.QUANTUM
        elif self.theta < 0.01:
            self.regime = Regime.CLASSICAL
        else:
            self.regime = Regime.TRANSITION

    @property
    def is_quantum(self) -> bool:
        """True if system is in quantum regime."""
        return self.regime == Regime.QUANTUM

    @property
    def is_classical(self) -> bool:
        """True if system is in classical regime."""
        return self.regime == Regime.CLASSICAL

    @property
    def is_transitional(self) -> bool:
        """True if system is in transition regime."""
        return self.regime == Regime.TRANSITION

    @property
    def quantum_fraction(self) -> float:
        """Percentage of system that is quantum."""
        return self.theta * 100

    @property
    def classical_fraction(self) -> float:
        """Percentage of system that is classical."""
        return (1 - self.theta) * 100

    def interpolate(self, quantum_value: float, classical_value: float) -> float:
        """
        Interpolate between quantum and classical values using theta.

        This is THE fundamental operation:
            result = θ × quantum_value + (1-θ) × classical_value

        Args:
            quantum_value: The value in pure quantum description
            classical_value: The value in pure classical description

        Returns:
            Interpolated value based on theta
        """
        return self.theta * quantum_value + (1 - self.theta) * classical_value

    def gradient_to(self, other: 'ThetaState') -> float:
        """
        Compute the theta gradient to another state.

        Args:
            other: Another ThetaState

        Returns:
            Change in theta (other.theta - self.theta)
        """
        return other.theta - self.theta

    def merge_with(
        self,
        other: 'ThetaState',
        weight_self: float = 0.5
    ) -> 'ThetaState':
        """
        Merge this state with another using weighted average.

        Args:
            other: Another ThetaState to merge with
            weight_self: Weight for this state (0-1)

        Returns:
            New ThetaState with merged theta value
        """
        weight_other = 1.0 - weight_self
        merged_theta = weight_self * self.theta + weight_other * other.theta
        merged_uncertainty = np.sqrt(
            (weight_self * self.theta_uncertainty)**2 +
            (weight_other * other.theta_uncertainty)**2
        )

        # Merge components
        merged_components = {}
        for key in set(self.components.keys()) | set(other.components.keys()):
            v1 = self.components.get(key, self.theta)
            v2 = other.components.get(key, other.theta)
            merged_components[key] = weight_self * v1 + weight_other * v2

        return ThetaState(
            theta=merged_theta,
            theta_uncertainty=merged_uncertainty,
            system=self.system,
            components=merged_components,
            proof_method=f"merged({self.proof_method}, {other.proof_method})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'theta': self.theta,
            'theta_uncertainty': self.theta_uncertainty,
            'regime': self.regime.value,
            'components': self.components,
            'quantum_scale': self.quantum_scale,
            'classical_scale': self.classical_scale,
            'proof_method': self.proof_method,
            'confidence': self.confidence,
            'is_validated': self.is_validated,
        }

    def __repr__(self) -> str:
        return (
            f"ThetaState(θ={self.theta:.4f} ± {self.theta_uncertainty:.4f}, "
            f"regime={self.regime.value}, "
            f"method={self.proof_method})"
        )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Theta State Summary",
            f"=" * 40,
            f"θ = {self.theta:.6f} ± {self.theta_uncertainty:.6f}",
            f"Regime: {self.regime}",
            f"Quantum fraction: {self.quantum_fraction:.1f}%",
            f"Classical fraction: {self.classical_fraction:.1f}%",
        ]

        if self.system:
            lines.append(f"System: {self.system.name}")

        if self.components:
            lines.append(f"\nComponent contributions:")
            for method, value in self.components.items():
                lines.append(f"  {method}: {value:.4f}")

        if self.proof_method:
            lines.append(f"\nProof method: {self.proof_method}")
            lines.append(f"Confidence: {self.confidence:.1%}")

        return "\n".join(lines)


@dataclass
class ThetaTrajectory:
    """
    Represents evolution of theta over a parameter or time.

    Used to track how theta changes as system parameters vary,
    identifying phase transitions where theta changes rapidly.

    Attributes:
        states: List of ThetaState objects along the trajectory
        parameter_name: Name of the varying parameter (e.g., "temperature")
        parameter_values: Values of the parameter at each state
    """
    states: List[ThetaState]
    parameter_name: str
    parameter_values: List[float]

    def __post_init__(self):
        """Validate trajectory consistency."""
        if len(self.states) != len(self.parameter_values):
            raise ValueError(
                f"Mismatch: {len(self.states)} states but "
                f"{len(self.parameter_values)} parameter values"
            )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> ThetaState:
        return self.states[index]

    @property
    def thetas(self) -> np.ndarray:
        """Array of theta values along trajectory."""
        return np.array([s.theta for s in self.states])

    def find_transitions(self, threshold: float = 0.1) -> List[int]:
        """
        Find indices where theta changes rapidly (phase transitions).

        A phase transition is where |dθ/dp| is large, indicating
        rapid change from quantum to classical (or vice versa).

        Args:
            threshold: Minimum theta change to count as transition

        Returns:
            List of indices where transitions occur
        """
        transitions = []
        for i in range(1, len(self.states)):
            gradient = abs(self.states[i].theta - self.states[i-1].theta)
            if gradient > threshold:
                transitions.append(i)
        return transitions

    def mean_theta(self) -> float:
        """Average theta across trajectory."""
        return float(np.mean(self.thetas))

    def theta_variance(self) -> float:
        """Variance in theta across trajectory."""
        return float(np.var(self.thetas))

    def gradient(self) -> np.ndarray:
        """Compute dθ/dp along the trajectory."""
        return np.gradient(self.thetas, self.parameter_values)

    def find_critical_point(self) -> Optional[int]:
        """
        Find the critical point where |dθ/dp| is maximum.

        This is the most significant phase transition point.

        Returns:
            Index of critical point, or None if gradient is flat
        """
        grad = np.abs(self.gradient())
        if np.max(grad) > 0.01:  # Significant gradient
            return int(np.argmax(grad))
        return None

    def interpolate_at(self, parameter_value: float) -> ThetaState:
        """
        Interpolate theta at an arbitrary parameter value.

        Args:
            parameter_value: The parameter value to interpolate at

        Returns:
            Interpolated ThetaState
        """
        theta_interp = np.interp(
            parameter_value,
            self.parameter_values,
            self.thetas
        )
        return ThetaState(
            theta=theta_interp,
            proof_method=f"interpolated_at_{self.parameter_name}={parameter_value}"
        )


# Pre-defined example systems for testing
EXAMPLE_SYSTEMS = {
    "electron": PhysicalSystem(
        name="electron",
        mass=9.109e-31,
        length_scale=2.818e-15,  # Classical electron radius
        energy=8.187e-14,        # Rest mass energy
        temperature=300.0,
        metadata={"type": "fundamental_particle"}
    ),
    "proton": PhysicalSystem(
        name="proton",
        mass=1.673e-27,
        length_scale=8.8e-16,    # Charge radius
        energy=1.503e-10,        # Rest mass energy
        temperature=300.0,
        metadata={"type": "composite_particle"}
    ),
    "hydrogen_atom": PhysicalSystem(
        name="hydrogen_atom",
        mass=1.674e-27,
        length_scale=5.29e-11,   # Bohr radius
        energy=2.18e-18,         # Ionization energy
        temperature=300.0,
        metadata={"type": "atom"}
    ),
    "water_molecule": PhysicalSystem(
        name="water_molecule",
        mass=2.99e-26,
        length_scale=2.75e-10,   # Molecular diameter
        energy=7.6e-19,          # Bond energy
        temperature=300.0,
        number_of_particles=3,
        metadata={"type": "molecule"}
    ),
    "virus": PhysicalSystem(
        name="virus",
        mass=1e-18,
        length_scale=1e-7,       # ~100 nm
        energy=1e-16,
        temperature=300.0,
        number_of_particles=int(1e6),
        metadata={"type": "biological"}
    ),
    "human_cell": PhysicalSystem(
        name="human_cell",
        mass=1e-12,
        length_scale=1e-5,       # ~10 μm
        energy=1e-13,
        temperature=310.0,
        number_of_particles=int(1e14),
        metadata={"type": "biological"}
    ),
    "baseball": PhysicalSystem(
        name="baseball",
        mass=0.145,
        length_scale=0.074,      # Diameter
        energy=100.0,            # ~100 J kinetic energy
        temperature=300.0,
        number_of_particles=int(1e25),
        metadata={"type": "macroscopic"}
    ),
    "human": PhysicalSystem(
        name="human",
        mass=70.0,
        length_scale=1.7,
        energy=1e7,              # Daily metabolic energy
        temperature=310.0,
        number_of_particles=int(7e27),
        metadata={"type": "biological"}
    ),
    "earth": PhysicalSystem(
        name="earth",
        mass=5.972e24,
        length_scale=6.371e6,    # Radius
        energy=2.24e32,          # Gravitational binding energy
        temperature=288.0,       # Surface average
        metadata={"type": "astronomical"}
    ),
    "stellar_black_hole": PhysicalSystem(
        name="stellar_black_hole",
        mass=1e31,               # ~5 solar masses
        length_scale=1.5e4,      # Schwarzschild radius
        energy=9e47,             # Rest mass energy
        temperature=1e-8,        # Hawking temperature (very cold)
        metadata={"type": "black_hole"}
    ),
}
