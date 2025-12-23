# Theta Calculator API Reference

## Overview

The Theta Calculator computes θ (theta), the quantum-classical interpolation parameter:
- θ = 0: Classical limit (deterministic, independent, macroscopic)
- θ = 1: Quantum limit (coherent, entangled, microscopic)

## Installation

```python
pip install theta-calculator
```

## Quick Start

```python
from theta_calculator import PhysicalSystem, UnifiedThetaProof

# Create a physical system
electron = PhysicalSystem(
    name="electron",
    mass=9.109e-31,          # kg
    length_scale=2.818e-15,  # m (classical electron radius)
    energy=8.187e-14,        # J (rest energy)
    temperature=300.0        # K
)

# Prove theta exists
proof = UnifiedThetaProof()
result = proof.prove_theta_exists(electron)
print(f"Theta: {result.theta:.4f}")  # Output: Theta: 0.9723
```

---

## Core Module (`theta_calculator.core`)

### PhysicalSystem

Represents a physical system with properties relevant to theta calculation.

```python
from theta_calculator import PhysicalSystem

system = PhysicalSystem(
    name: str,              # System identifier
    mass: float,            # Mass in kg
    length_scale: float,    # Characteristic length in m
    energy: float,          # Characteristic energy in J
    temperature: float = 300.0,  # Temperature in K
    time_scale: float = None,    # Optional time scale in s
)
```

**Properties:**
- `rest_energy` - E = mc² in Joules
- `schwarzschild_radius` - r_s = 2GM/c² in meters
- `estimate_action()` - Estimated characteristic action in J·s

### ThetaState

Represents the quantum-classical state with a theta value.

```python
from theta_calculator import ThetaState, Regime

state = ThetaState(
    theta: float,           # Value in [0, 1]
    proof_method: str,      # How theta was computed
    system: PhysicalSystem = None,
    components: dict = None,
)
```

**Properties:**
- `regime` - Returns `Regime.QUANTUM`, `Regime.CLASSICAL`, or `Regime.TRANSITION`
- `quantum_fraction` - Same as theta
- `classical_fraction` - Equal to 1 - theta

**Methods:**
- `interpolate(classical_val, quantum_val)` - Interpolate between values
- `merge_with(other, weight)` - Combine with another ThetaState
- `gradient_to(other)` - Compute gradient toward another state

### ThetaCalculator

Computes theta using multiple methodologies.

```python
from theta_calculator import ThetaCalculator, PhysicalSystem

calc = ThetaCalculator()
system = PhysicalSystem(name="test", mass=1e-30, length_scale=1e-10, energy=1e-20)

# Individual methods
action_state = calc.compute_action_theta(system)    # θ = ℏ/S
thermal_state = calc.compute_thermal_theta(system)  # θ = E_q/E_th
scale_state = calc.compute_scale_theta(system)      # θ = L_P/L
decoherence_state = calc.compute_decoherence_theta(system)

# Unified (combines all methods)
unified_state = calc.compute_unified_theta(system)

# Get all results
all_results = calc.compute_all_methods(system)

# Analyze consistency
stats = calc.analyze_convergence(system)
```

---

## Constants Module (`theta_calculator.constants`)

### FundamentalConstants

CODATA 2022 physical constants.

```python
from theta_calculator import FundamentalConstants

# Access constants
c = FundamentalConstants.c       # Speed of light: 299792458 m/s (exact)
h = FundamentalConstants.h       # Planck constant: 6.62607015e-34 J·s (exact)
hbar = FundamentalConstants.hbar # Reduced Planck: 1.054571817e-34 J·s
k_B = FundamentalConstants.k_B   # Boltzmann: 1.380649e-23 J/K (exact)
G = FundamentalConstants.G       # Gravitational: 6.67430e-11 m³/(kg·s²)
alpha = FundamentalConstants.alpha  # Fine structure: 7.2973525693e-3

# Get all constants
all_constants = FundamentalConstants.get_all()
exact_only = FundamentalConstants.get_exact()

# Verify consistency
relationships = FundamentalConstants.verify_relationships()
```

### PlanckUnits

Natural units where ℏ = c = G = k_B = 1.

```python
from theta_calculator import PlanckUnits

# Planck units
l_P = PlanckUnits.length      # 1.616255e-35 m
t_P = PlanckUnits.time        # 5.391247e-44 s
m_P = PlanckUnits.mass        # 2.176434e-8 kg
E_P = PlanckUnits.energy      # 1.9561e9 J
T_P = PlanckUnits.temperature # 1.416784e32 K

# Conversions
planck_lengths = PlanckUnits.in_planck_units(1e-35, "length")  # ~0.619
si_length = PlanckUnits.from_planck_units(1.0, "length")       # 1.616e-35 m

# Verify relationships
checks = PlanckUnits.verify_planck_relationships()
```

---

## Proofs Module (`theta_calculator.proofs`)

### UnifiedThetaProof

Combines multiple proof methodologies to establish theta.

```python
from theta_calculator import UnifiedThetaProof, PhysicalSystem

proof = UnifiedThetaProof()
system = PhysicalSystem(name="electron", mass=9.1e-31,
                        length_scale=2.8e-15, energy=8.2e-14)

# Prove theta exists
result = proof.prove_theta_exists(system)

# Access results
print(result.theta)           # Final theta value
print(result.regime)          # Regime classification
print(result.is_valid)        # True if methods agree
print(result.proof_agreement) # Agreement score [0,1]
print(result.theta_values)    # Dict of individual method results
print(result.summary)         # Human-readable summary
print(result.validation_notes) # List of validation notes

# Compare multiple systems
systems = [electron, proton, baseball]
comparison = proof.compare_systems(systems)
```

### BekensteinBound

Information-theoretic bound on entropy.

```python
from theta_calculator.proofs.information.bekenstein_bound import BekensteinBound

bound = BekensteinBound()

# Compute maximum entropy
S_nats = bound.compute_bound_nats(radius=1.0, energy=1e9)
S_bits = bound.compute_bound_bits(radius=1.0, energy=1e9)

# Black hole entropy
S_bh = bound.black_hole_entropy_bits(mass=1e30)

# Theta from Bekenstein bound
state = bound.theta_from_bekenstein(system)
```

### LandauerLimit

Thermodynamic limit on computation.

```python
from theta_calculator.proofs.information.landauer_limit import LandauerLimit

landauer = LandauerLimit()

# Minimum erasure energy
E_min = landauer.minimum_erasure_energy(temperature=300.0)

# Quantum computation limit
N_ops = landauer.max_operations_quantum(energy=1e-15, time=1e-9)

# Theta from Landauer limit
state = landauer.theta_from_landauer(system)
```

---

## Domains Module (`theta_calculator.domains`)

### Economics

Market theta using Ising model framework.

```python
from theta_calculator.domains import (
    MarketSystem, compute_market_theta, ECONOMIC_SYSTEMS
)

# Use predefined system
crash = ECONOMIC_SYSTEMS["market_crash"]
theta = compute_market_theta(crash)  # High theta (correlated)

# Create custom market
market = MarketSystem(
    name="Custom Market",
    n_traders=1000,
    coupling_strength=1.5,    # Trader interactions
    temperature=1.0,          # Market volatility
    order_parameter=0.8,      # Sentiment (-1 to 1)
    correlation=0.7,          # Stock correlation
)
theta = compute_market_theta(market)
```

**Available systems:** `efficient_market`, `normal_trading`, `trending_market`, `bubble_forming`, `market_crash`, `flash_crash`, `dotcom_bubble`

### Information

Entropy-based theta calculation.

```python
from theta_calculator.domains import (
    InformationSystem, compute_information_theta,
    compute_shannon_entropy, compute_von_neumann_entropy,
    INFORMATION_SYSTEMS
)

# Shannon entropy
probs = [0.5, 0.5]
H = compute_shannon_entropy(probs)  # 1.0 bits

# Von Neumann entropy
import numpy as np
rho = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed qubit
S = compute_von_neumann_entropy(rho)  # 1.0 bits

# Use predefined system
mixed = INFORMATION_SYSTEMS["mixed_qubit"]
theta = compute_information_theta(mixed)  # ~1.0 (high entropy)
```

**Available systems:** `pure_qubit`, `mixed_qubit`, `thermal_hot`, `thermal_cold`, `bell_reduced`, `fair_coin`, `biased_coin`, `deterministic`, `uniform_die`, `8_level_mixed`

### Game Theory

Quantum game entanglement parameter.

```python
from theta_calculator.domains import (
    QuantumGame, compute_entanglement_theta, GAME_SYSTEMS
)

# Use predefined games
classical = GAME_SYSTEMS["classical_pd"]
quantum = GAME_SYSTEMS["quantum_pd"]

theta_classical = compute_entanglement_theta(classical)  # 0.0
theta_quantum = compute_entanglement_theta(quantum)      # 1.0
```

**Available systems:** `classical_pd`, `partial_quantum_pd`, `quantum_pd`, `classical_chicken`, `quantum_chicken`, `classical_bos`, `quantum_bos`

### Complex Systems

Phase transition and criticality.

```python
from theta_calculator.domains import (
    ComplexSystem, CriticalExponents, compute_complex_theta,
    COMPLEX_SYSTEMS, compute_order_parameter
)

# Use predefined system
critical = COMPLEX_SYSTEMS["ferromagnet_critical"]
theta = compute_complex_theta(critical)  # ~1.0 (at phase transition)

# Compute order parameter
m = compute_order_parameter(
    temperature=900,
    T_c=1000,
    exponents=CriticalExponents(alpha=0, beta=0.5, gamma=1, delta=3, nu=0.5, eta=0)
)
```

**Available systems:** `ferromagnet_hot`, `ferromagnet_critical`, `ferromagnet_cold`, `opinion_polarized`, `opinion_diverse`, `epidemic_spreading`, `neural_criticality`, `civil_unrest`

### Quantum Computing

Error threshold and coherence.

```python
from theta_calculator.domains import (
    QubitSystem, compute_quantum_computing_theta,
    compute_coherence_theta, QUANTUM_HARDWARE
)

# Use real quantum hardware specs
willow = QUANTUM_HARDWARE["google_willow"]
theta = compute_quantum_computing_theta(willow)  # High (below threshold)

# Create custom system
qubits = QubitSystem(
    name="Custom",
    qubit_type="transmon",
    n_qubits=100,
    T1=100e-6,         # 100 μs relaxation
    T2=50e-6,          # 50 μs dephasing
    gate_time=20e-9,   # 20 ns gates
    error_rate=0.001,  # 0.1% error
    error_threshold=0.01,  # 1% threshold
)
```

**Available systems:** `google_sycamore`, `google_willow`, `ibm_heron`, `ionq_forte`, `quantinuum_h2`, `nv_center_lab`, `neutral_atom_quera`, `noisy_classical`

### Quantum Biology

Biological quantum coherence.

```python
from theta_calculator.domains import (
    BiologicalSystem, compute_quantum_bio_theta, BIOLOGICAL_SYSTEMS
)

# Photosynthesis coherence
fmo = BIOLOGICAL_SYSTEMS["fmo_complex"]
theta = compute_quantum_bio_theta(fmo)

# Bird navigation
compass = BIOLOGICAL_SYSTEMS["avian_compass"]
theta = compute_quantum_bio_theta(compass)
```

**Available systems:** `fmo_complex`, `photosystem_ii`, `avian_compass`, `enzyme_catalysis`, `olfaction`, `dna_proton_transfer`

### Cosmology

Cosmic theta evolution.

```python
from theta_calculator.domains import (
    CosmicEpoch, compute_cosmic_theta, COSMIC_TIMELINE
)

# Planck era (maximum theta)
planck = COSMIC_TIMELINE["planck_era"]
theta = compute_cosmic_theta(planck)  # ~1.0

# Present day (classical)
now = COSMIC_TIMELINE["present_day"]
theta = compute_cosmic_theta(now)  # ~10^-33
```

**Available epochs:** `planck_era`, `gut_era`, `electroweak_era`, `quark_era`, `nucleosynthesis`, `recombination`, `first_stars`, `galaxy_formation`, `present_day`, `heat_death`

### Control Theory

Stability margins and feedback.

```python
from theta_calculator.domains import (
    ControlSystem, compute_control_theta, CONTROL_SYSTEMS
)

# LQR optimal control
lqr = CONTROL_SYSTEMS["lqr_optimal"]
theta = compute_control_theta(lqr)  # High (robust)

# Marginally stable
marginal = CONTROL_SYSTEMS["marginally_stable"]
theta = compute_control_theta(marginal)  # Low
```

**Available systems:** `open_loop`, `thermostat_simple`, `pid_tuned`, `inverted_pendulum`, `spacecraft_attitude`, `lqr_optimal`, `h_infinity`, `quantum_error_correction`, `marginally_stable`, `neural_feedback`

### Nonlinear Dynamics

Chaos and bifurcations.

```python
from theta_calculator.domains import (
    DynamicalSystem, compute_dynamics_theta, DYNAMICAL_SYSTEMS,
    logistic_map, logistic_lyapunov
)

# Chaotic logistic map
chaotic = DYNAMICAL_SYSTEMS["logistic_chaotic"]
theta = compute_dynamics_theta(chaotic)  # High (chaotic)

# Compute Lyapunov exponent
lyap = logistic_lyapunov(r=4.0)  # Positive = chaotic
```

**Available systems:** `logistic_stable`, `logistic_period2`, `logistic_edge_of_chaos`, `logistic_chaotic`, `lorenz_attractor`, `lorenz_stable`, `double_pendulum`, `henon_map`, `cardiac_normal`, `cardiac_fibrillation`, `brain_criticality`

### Quantum Gravity

Planck-scale phenomena.

```python
from theta_calculator.domains import (
    QuantumGravitySystem, compute_quantum_gravity_theta,
    black_hole_theta, hawking_temperature_k,
    QUANTUM_GRAVITY_SYSTEMS, L_PLANCK, M_PLANCK
)

# Planck-scale black hole
planck_bh = QUANTUM_GRAVITY_SYSTEMS["planck_mass_bh"]
theta = compute_quantum_gravity_theta(planck_bh)  # ~1.0

# Stellar black hole
stellar = QUANTUM_GRAVITY_SYSTEMS["stellar_bh"]
theta = compute_quantum_gravity_theta(stellar)  # ~0

# Hawking temperature
T = hawking_temperature_k(mass=M_PLANCK)  # Planck temperature
```

**Available systems:** `planck_mass_bh`, `stellar_bh`, `sgr_a_star`, `human_scale`, `lhc_collision`, `big_bang`, `inflation`

### Universal Interface

Cross-domain theta computation.

```python
from theta_calculator.domains import UniversalTheta, cross_domain_comparison

# Compute theta for any system
from theta_calculator.domains import ECONOMIC_SYSTEMS, QUANTUM_HARDWARE

result = UniversalTheta.compute(ECONOMIC_SYSTEMS["market_crash"])
print(result.theta)   # Theta value
print(result.domain)  # DomainType.ECONOMICS
print(result.regime)  # "crash/bubble"

# Compare across all domains
comparison = cross_domain_comparison()
for domain, systems in comparison.items():
    print(f"\n{domain}:")
    for name, theta in systems.items():
        print(f"  {name}: {theta:.3f}")
```

---

## CLI Commands

```bash
# Prove theta for a system
python -m theta_calculator prove --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300

# Show physical constants
python -m theta_calculator constants --show-planck

# Explain a system
python -m theta_calculator explain --system electron

# Quick calculation
python -m theta_calculator quick --mass 0.145 --length 0.074 --temp 300

# Compare example systems
python -m theta_calculator compare

# List all domains
python -m theta_calculator domains

# Analyze domain system
python -m theta_calculator domain -d quantum_gravity -s planck_mass_bh

# Cross-domain comparison
python -m theta_calculator crossdomain
```

---

## Key Formulas

| Method | Formula | Interpretation |
|--------|---------|----------------|
| Action Ratio | θ = ℏ/S | Quantum when S ≈ ℏ |
| Thermal Ratio | θ = E_q/E_th | Quantum when E_q > k_B T |
| Scale Ratio | θ = L_P/L | Quantum when L ≈ L_P |
| Decoherence | θ = τ_coh/τ_dyn | Quantum when coherence persists |
| Information | θ = S/S_max | Mixed when entropy is high |
| Entanglement | θ = 2γ/π | Quantum when γ = π/2 |

---

## Version

- **Current Version:** 0.1.0
- **Python:** ≥3.8
- **Dependencies:** numpy

## References

All citations reference entries in `BIBLIOGRAPHY.bib`.

### Fundamental Constants
- \cite{CODATA2022} - NIST CODATA 2022 Recommended Values

### Information Theory
- \cite{Bekenstein1981} - Bekenstein bound on entropy
- \cite{Landauer1961} - Landauer limit on computation
- \cite{MargoluLevitin1998} - Margolus-Levitin quantum speed limit
- \cite{Shannon1948} - Shannon entropy
- \cite{VonNeumann1932} - von Neumann entropy

### Black Hole Thermodynamics
- \cite{Hawking1974} - Hawking temperature
- \cite{Bekenstein1973} - Bekenstein-Hawking entropy
- \cite{Schwarzschild1916} - Schwarzschild radius

### Quantum Computing
- \cite{Shor1996} - Fault-tolerant quantum computation
- \cite{Kitaev2003} - Topological quantum error correction
- \cite{GoogleQuantum2024} - Experimental error correction threshold
- \cite{Preskill2018} - NISQ computing framework
- \cite{Fowler2012} - Surface code architecture

### Quantum Biology
- \cite{Engel2007} - Quantum coherence in photosynthesis
- \cite{Ritz2000} - Radical pair magnetoreception
- \cite{Klinman2013} - Hydrogen tunneling in enzymes
- \cite{Lowdin1963} - Proton tunneling in DNA
- \cite{Turin1996} - Vibration theory of olfaction

### Game Theory
- \cite{Eisert1999} - Quantum games and strategies
- \cite{Meyer1999} - Quantum strategies
- \cite{Du2002} - Experimental quantum games

### Complex Systems
- \cite{WilsonKogut1974} - Renormalization group theory
- \cite{Stanley1971} - Phase transitions and critical phenomena
- \cite{BakTangWiesenfeld1987} - Self-organized criticality

### Cosmology
- \cite{Planck2020} - Cosmological parameters
- \cite{Weinberg1972} - Gravitation and cosmology
- \cite{Guth1981} - Inflationary universe

### Control Theory
- \cite{Astrom2010} - Feedback systems
- \cite{Ogata2010} - Modern control engineering
- \cite{Doyle1992} - Feedback control theory

### Nonlinear Dynamics
- \cite{Strogatz2015} - Nonlinear dynamics and chaos
- \cite{Feigenbaum1978} - Feigenbaum constants
- \cite{Lorenz1963} - Deterministic chaos

### Quantum Gravity
- \cite{Rovelli2004} - Loop quantum gravity
- \cite{Ashtekar2004} - Background independent quantum gravity
- \cite{Thiemann2007} - Modern canonical quantum GR
- \cite{Immirzi1997} - Barbero-Immirzi parameter

### Economics
- \cite{Bornholdt2001} - Ising model for markets
- \cite{Krawiecki2002} - Volatility clustering

### Quantum Decoherence
- \cite{Zurek2003} - Quantum Darwinism
- \cite{Penrose1996} - Gravitational decoherence
- \cite{JoosZeh1985} - Environmental decoherence
