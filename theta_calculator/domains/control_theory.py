"""
Control Theory Domain: Stability and Feedback as Theta

This module implements theta as the quantum-classical interpolation parameter
for control systems, mapping stability margins to the theta framework.

Key Insight: Control systems exhibit a spectrum between:
- theta ~ 0: Unstable (diverging, no control, noise-dominated)
- theta ~ 1: Perfectly stable (robust, fault-tolerant, precise)

The control-theoretic theta measures how "controlled" a system is,
analogous to how quantum theta measures how "quantum" a system is.

Key Mappings:
- Gain margin → theta (higher margin = more stable = higher theta)
- Phase margin → theta (45-60 deg optimal, 0 deg = instability)
- Feedback strength → theta (stronger feedback = higher theta)
- Noise rejection → theta (better rejection = higher theta)

References:
- Astrom & Murray (2010): Feedback Systems
- Ogata (2010): Modern Control Engineering
- Doyle, Francis, Tannenbaum (1992): Feedback Control Theory
- Skogestad & Postlethwaite (2005): Multivariable Feedback Control
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class StabilityRegime(Enum):
    """Classification of control system stability regimes."""
    UNSTABLE = "unstable"              # theta < 0.1: System diverges
    MARGINALLY_STABLE = "marginal"     # 0.1 <= theta < 0.3: Near instability
    STABLE = "stable"                  # 0.3 <= theta < 0.7: Controlled
    ROBUST = "robust"                  # 0.7 <= theta < 0.9: Good margins
    OPTIMAL = "optimal"                # theta >= 0.9: Near-perfect control


class ControllerType(Enum):
    """Types of control strategies."""
    OPEN_LOOP = "open_loop"            # No feedback
    PROPORTIONAL = "proportional"      # P control
    PID = "pid"                        # PID control
    STATE_FEEDBACK = "state_feedback"  # Full state feedback
    OPTIMAL = "optimal"                # LQR/LQG
    ROBUST = "robust"                  # H-infinity
    ADAPTIVE = "adaptive"              # Self-tuning
    QUANTUM = "quantum"                # Quantum error correction as control


@dataclass
class ControlSystem:
    """
    A control system for theta analysis.

    Attributes:
        name: System identifier
        controller_type: Type of control strategy
        gain_margin_db: Gain margin in dB (0 = marginally stable)
        phase_margin_deg: Phase margin in degrees (0 = marginally stable)
        bandwidth_hz: Closed-loop bandwidth in Hz
        settling_time: Time to reach steady state (seconds)
        overshoot_pct: Percent overshoot
        noise_rejection_db: Noise attenuation in dB
        poles_real_parts: Real parts of closed-loop poles (all < 0 for stability)
    """
    name: str
    controller_type: ControllerType
    gain_margin_db: float           # Gain margin (dB), >0 for stable
    phase_margin_deg: float         # Phase margin (degrees), >0 for stable
    bandwidth_hz: Optional[float] = None
    settling_time: Optional[float] = None
    overshoot_pct: Optional[float] = None
    noise_rejection_db: Optional[float] = None
    poles_real_parts: Optional[List[float]] = None

    @property
    def is_stable(self) -> bool:
        """Check if system is stable (positive margins)."""
        return self.gain_margin_db > 0 and self.phase_margin_deg > 0

    @property
    def gain_margin_linear(self) -> float:
        """Convert gain margin from dB to linear scale."""
        return 10 ** (self.gain_margin_db / 20)


# =============================================================================
# THETA CALCULATION
# =============================================================================

def compute_gain_margin_theta(system: ControlSystem) -> float:
    """
    Compute theta from gain margin.

    Gain margin > 6 dB is considered robust.
    Gain margin < 0 dB means unstable.

    theta = 1 - exp(-GM_dB / 6)  for GM > 0
    theta = 0                    for GM <= 0
    """
    if system.gain_margin_db <= 0:
        return 0.0

    # 6 dB is a typical "good" gain margin
    theta = 1.0 - np.exp(-system.gain_margin_db / 6.0)
    return np.clip(theta, 0.0, 1.0)


def compute_phase_margin_theta(system: ControlSystem) -> float:
    """
    Compute theta from phase margin.

    Phase margin of 45-60 degrees is optimal.
    Phase margin < 0 means unstable.
    Phase margin > 90 is very robust.

    theta = PM / 90  (capped at 1)
    """
    if system.phase_margin_deg <= 0:
        return 0.0

    theta = system.phase_margin_deg / 90.0
    return np.clip(theta, 0.0, 1.0)


def compute_pole_theta(system: ControlSystem) -> float:
    """
    Compute theta from pole locations.

    All poles must have negative real parts for stability.
    More negative = more stable = higher theta.

    theta = 1 - exp(max_real_part)
    """
    if system.poles_real_parts is None:
        return 0.5  # Unknown, assume moderate

    max_real = max(system.poles_real_parts)

    if max_real >= 0:
        return 0.0  # Unstable or marginally stable

    # More negative poles = higher theta
    theta = 1.0 - np.exp(max_real)
    return np.clip(theta, 0.0, 1.0)


def compute_control_theta(system: ControlSystem) -> float:
    """
    Compute unified theta for a control system.

    Combines gain margin, phase margin, and pole locations.
    All must be good for high overall theta.
    """
    theta_gm = compute_gain_margin_theta(system)
    theta_pm = compute_phase_margin_theta(system)
    theta_pole = compute_pole_theta(system)

    # Geometric mean: all factors must be good
    theta = (theta_gm * theta_pm * theta_pole) ** (1/3)

    return np.clip(theta, 0.0, 1.0)


def classify_stability(theta: float) -> StabilityRegime:
    """Classify stability regime from theta."""
    if theta < 0.1:
        return StabilityRegime.UNSTABLE
    elif theta < 0.3:
        return StabilityRegime.MARGINALLY_STABLE
    elif theta < 0.7:
        return StabilityRegime.STABLE
    elif theta < 0.9:
        return StabilityRegime.ROBUST
    else:
        return StabilityRegime.OPTIMAL


# =============================================================================
# CONTROL METRICS
# =============================================================================

def bode_stability_margins(
    gain_crossover_hz: float,
    phase_crossover_hz: float,
    gain_at_phase_crossover_db: float,
    phase_at_gain_crossover_deg: float
) -> Dict[str, float]:
    """
    Compute stability margins from Bode plot data.

    Reference: Astrom & Murray (2010), Ch. 9

    Args:
        gain_crossover_hz: Frequency where |G| = 1 (0 dB)
        phase_crossover_hz: Frequency where phase = -180 deg
        gain_at_phase_crossover_db: |G| at phase crossover in dB
        phase_at_gain_crossover_deg: Phase at gain crossover in degrees

    Returns:
        Dictionary with gain_margin_db and phase_margin_deg
    """
    gain_margin_db = -gain_at_phase_crossover_db
    phase_margin_deg = 180 + phase_at_gain_crossover_deg

    return {
        "gain_margin_db": gain_margin_db,
        "phase_margin_deg": phase_margin_deg,
        "gain_crossover_hz": gain_crossover_hz,
        "phase_crossover_hz": phase_crossover_hz,
    }


def settling_time_theta(settling_time: float, time_constant: float) -> float:
    """
    Compute theta from settling time relative to natural time constant.

    Faster settling (relative to open-loop) = better control = higher theta.

    theta = 1 - settling_time / (4 * time_constant)
    (4*tau is typical uncontrolled settling time)
    """
    if time_constant <= 0:
        return 0.5

    ratio = settling_time / (4 * time_constant)
    theta = 1.0 - ratio
    return np.clip(theta, 0.0, 1.0)


def overshoot_theta(overshoot_pct: float) -> float:
    """
    Compute theta from overshoot.

    No overshoot (critically damped) = theta = 1
    Large overshoot = theta -> 0
    Optimal is often 5-10% overshoot.

    theta = exp(-overshoot / 20)
    """
    if overshoot_pct <= 0:
        return 1.0

    theta = np.exp(-overshoot_pct / 20.0)
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# EXAMPLE CONTROL SYSTEMS
# =============================================================================

CONTROL_SYSTEMS: Dict[str, ControlSystem] = {
    "open_loop": ControlSystem(
        name="Open Loop (No Feedback)",
        controller_type=ControllerType.OPEN_LOOP,
        gain_margin_db=-float('inf'),
        phase_margin_deg=0,
        poles_real_parts=[0.1],  # Unstable
    ),
    "thermostat_simple": ControlSystem(
        name="Simple Thermostat (On/Off)",
        controller_type=ControllerType.PROPORTIONAL,
        gain_margin_db=6,
        phase_margin_deg=30,
        settling_time=300,  # 5 minutes
        overshoot_pct=20,
        poles_real_parts=[-0.01, -0.01],
    ),
    "pid_tuned": ControlSystem(
        name="Well-Tuned PID Controller",
        controller_type=ControllerType.PID,
        gain_margin_db=12,
        phase_margin_deg=60,
        settling_time=2.0,
        overshoot_pct=5,
        noise_rejection_db=40,
        poles_real_parts=[-2, -1.5, -1],
    ),
    "inverted_pendulum": ControlSystem(
        name="Inverted Pendulum (Balanced)",
        controller_type=ControllerType.STATE_FEEDBACK,
        gain_margin_db=8,
        phase_margin_deg=45,
        settling_time=0.5,
        overshoot_pct=10,
        poles_real_parts=[-5, -4, -3, -2],
    ),
    "spacecraft_attitude": ControlSystem(
        name="Spacecraft Attitude Control",
        controller_type=ControllerType.ROBUST,
        gain_margin_db=15,
        phase_margin_deg=70,
        settling_time=10,
        overshoot_pct=2,
        noise_rejection_db=60,
        poles_real_parts=[-0.5, -0.4, -0.3],
    ),
    "lqr_optimal": ControlSystem(
        name="LQR Optimal Controller",
        controller_type=ControllerType.OPTIMAL,
        gain_margin_db=float('inf'),  # LQR guarantees infinite gain margin
        phase_margin_deg=60,
        settling_time=1.0,
        overshoot_pct=0,
        poles_real_parts=[-3, -2, -1],
    ),
    "h_infinity": ControlSystem(
        name="H-infinity Robust Controller",
        controller_type=ControllerType.ROBUST,
        gain_margin_db=20,
        phase_margin_deg=75,
        settling_time=1.5,
        overshoot_pct=1,
        noise_rejection_db=80,
        poles_real_parts=[-4, -3, -2, -1],
    ),
    "quantum_error_correction": ControlSystem(
        name="Quantum Error Correction",
        controller_type=ControllerType.QUANTUM,
        gain_margin_db=25,  # Very robust to small errors
        phase_margin_deg=85,
        settling_time=1e-6,  # Microsecond timescale
        overshoot_pct=0,
        noise_rejection_db=100,  # Exponential error suppression
        poles_real_parts=[-1e6, -1e5, -1e4],
    ),
    "marginally_stable": ControlSystem(
        name="Marginally Stable System",
        controller_type=ControllerType.PROPORTIONAL,
        gain_margin_db=2,
        phase_margin_deg=10,
        settling_time=20,
        overshoot_pct=50,
        poles_real_parts=[-0.1, -0.05],
    ),
    "neural_feedback": ControlSystem(
        name="Neural Feedback Loop (Homeostasis)",
        controller_type=ControllerType.ADAPTIVE,
        gain_margin_db=10,
        phase_margin_deg=50,
        settling_time=60,  # Minutes for physiological systems
        overshoot_pct=15,
        poles_real_parts=[-0.02, -0.01, -0.005],
    ),
}


# =============================================================================
# NYQUIST STABILITY ANALYSIS
# =============================================================================

def nyquist_encirclements(
    poles_rhp: int,
    zeros_rhp: int,
    encirclements_ccw: int
) -> Dict[str, any]:
    """
    Analyze stability using Nyquist criterion.

    Reference: Doyle, Francis, Tannenbaum (1992)

    Nyquist Criterion: N = P - Z
    where:
    - N = number of CCW encirclements of -1
    - P = number of RHP poles of open-loop
    - Z = number of RHP poles of closed-loop

    For stability: Z = 0, so need N = P
    """
    z_rhp = poles_rhp - encirclements_ccw
    is_stable = (z_rhp == 0)

    return {
        "poles_rhp_open_loop": poles_rhp,
        "zeros_rhp_open_loop": zeros_rhp,
        "encirclements_ccw": encirclements_ccw,
        "poles_rhp_closed_loop": z_rhp,
        "is_stable": is_stable,
        "theta": 1.0 if is_stable else 0.0,
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def control_theory_theta_summary():
    """Print theta analysis for control systems."""
    print("=" * 80)
    print("CONTROL THEORY THETA ANALYSIS")
    print("=" * 80)
    print()
    print(f"{'System':<35} {'θ':>8} {'GM (dB)':>10} {'PM (deg)':>10} {'Regime':<15}")
    print("-" * 80)

    for name, system in CONTROL_SYSTEMS.items():
        theta = compute_control_theta(system)
        regime = classify_stability(theta)

        gm_str = f"{system.gain_margin_db:.1f}" if np.isfinite(system.gain_margin_db) else "inf"
        pm_str = f"{system.phase_margin_deg:.1f}"

        print(f"{system.name:<35} {theta:>8.3f} {gm_str:>10} {pm_str:>10} {regime.value:<15}")

    print()
    print("Key: θ → 1 for robust control, θ → 0 for instability")
    print("     LQR guarantees infinite gain margin (θ = 1 for gain)")


if __name__ == "__main__":
    control_theory_theta_summary()
