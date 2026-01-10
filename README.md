# Theta Calculator

Computes a dimensionless parameter **θ** that estimates where a system sits on a classical ↔ quantum-like continuum.

- **θ ≈ 0**: classical-like (deterministic, well-localized, large action)
- **θ ≈ 1**: quantum-like (coherent/superposed, action near ℏ)
- **0 < θ < 1**: transition regime (crossover scales)

## Installation

```bash
pip install theta-calc
```

Or for development:
```bash
git clone <repo>
pip install -e .[dev]
```

## Quick Start

```bash
# Compare theta across systems
theta-calc compare

# Score a custom system
theta-calc quick --mass 9.1e-31 --length 2.8e-15 --temp 300
```

---

## Golden Demos (< 30 seconds each)

### 1. Market Bubble Detection
```bash
# Normal trading vs flash crash
theta-calc domain -d economics -s normal_trading
# θ ≈ 0.41 (classical-leaning: independent traders)

theta-calc domain -d economics -s flash_crash
# θ ≈ 0.73 (quantum-like: correlated herding behavior)
```

### 2. Qubit Coherence Threshold
```bash
# Compare quantum computing technologies
theta-calc domain -d quantum_computing -s google_willow
# θ ≈ 0.90+ (highly coherent, below error threshold)

theta-calc domain -d quantum_computing -s early_superconducting
# θ ≈ 0.15 (noisy, classical-like)
```

### 3. Critical Point Detection
```bash
# Ferromagnet near/far from phase transition
theta-calc domain -d complex_systems -s ferromagnet_critical
# θ ≈ 1.0 (at critical point: diverging correlations)

theta-calc domain -d complex_systems -s ferromagnet_cold
# θ ≈ 0.37 (ordered phase: stable, predictable)
```

### JSON Output for Dashboards
```bash
theta-calc compare --json | jq '.systems.electron'
# {"theta": 0.87, "regime": "transition"}
```

---

## What is Theta?

**Theta (θ)** is a dimensionless parameter that compares a system's characteristic action scale to Planck's constant (ℏ):

```
θ = ℏ / S    (Planck action / system action)
```

**S** (system action) is estimated from user-provided physical parameters:
- Mass × velocity × length scale
- Or: Energy × characteristic time
- The calculator uses multiple estimation methods and takes a weighted average

This is an *operational scoring model*, not a new physical constant.

---

## What Theta is NOT

- θ is **not** an officially recognized physical constant
- θ is **not** claiming to replace quantum mechanics or general relativity
- θ is **not** a measurement device—it's a computed index under modeling assumptions
- θ values across domains are **analogy scores**, not literal quantum parameters

---

## Usage

### Compare theta across physical systems

```bash
theta-calc compare
```

Output:
```
THETA COMPARISON ACROSS PHYSICAL SYSTEMS
============================================================
System                         Theta Regime
------------------------------------------------------------
electron                    0.869991 quantum-like
hydrogen_atom               0.770084 quantum-like
water_molecule              0.769286 quantum-like
proton                      0.607457 transition
stellar_black_hole          0.508157 transition
virus                       0.422613 transition
human_cell                  0.232905 classical-like
baseball                    0.130017 classical-like
human                       0.130001 classical-like
earth                       0.130000 classical-like
```

### Score theta for a custom system

```bash
theta-calc score --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300
```

### Display fundamental constants

```bash
theta-calc constants --show-planck --show-bootstrap
```

### Get detailed explanation for a system

```bash
theta-calc explain --system electron --format markdown
```

### Cross-domain analysis

```bash
# List all domains
theta-calc domains

# Analyze specific domain system
theta-calc domain -d economics -s market_crash
theta-calc domain -d quantum_computing -s google_willow

# Cross-domain comparison
theta-calc crossdomain
```

---

## Validation Methods

Five independent approaches that should yield consistent θ values:

### 1. Action Ratio
θ = ℏ/S (quantum when action ≈ Planck constant)

### 2. Thermal Ratio
θ = (ℏω)/(kT) (quantum vs thermal energy)

### 3. Scale Ratio
Compare Planck/de Broglie wavelength to system size

### 4. Decoherence
θ = exp(-t/t_D) (coherence decay rate)

### 5. Unified
Weighted combination with confidence scoring

---

## Fundamental Constants Used

| Constant | Symbol | Value | Role |
|----------|--------|-------|------|
| Speed of light | c | 299,792,458 m/s | Relativistic limit |
| Planck constant | ℏ | 1.054571817 × 10⁻³⁴ J·s | Quantum of action |
| Gravitational constant | G | 6.67430 × 10⁻¹¹ m³/kg·s² | Gravitational coupling |
| Boltzmann constant | k_B | 1.380649 × 10⁻²³ J/K | Thermal energy scale |
| Elementary charge | e | 1.602176634 × 10⁻¹⁹ C | Charge quantum |
| Fine structure constant | α | 1/137.035999... | EM coupling strength |

The library includes **308 CODATA 2022 constants** including particle physics (W, Z, Higgs masses) and cosmological parameters (H₀, Λ, Ω).

---

## Cross-Domain Extensions

The theta framework extends to **35 domains** exhibiting classical-quantum-like transitions:

### Physics & Foundations
| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Physics (core) | Planets, baseballs | Electrons, photons |
| Quantum Computing | Noisy/decoherent | Coherent qubits |
| Quantum Biology | Classical chemistry | Coherent transfer |
| Quantum Gravity | Macroscopic | Planck scale |
| Quantum Foundations | Classical limit | Quantum superposition |
| Cosmology | Present day | Planck era |
| Condensed Matter | Disordered/insulating | Ordered/topological |
| High Energy Physics | Perturbative QCD | Confinement |
| Atomic/Optical | Incoherent | Strong coupling/BEC |
| Physics Extended | Weak field GR | Planck scale |

### Mathematics
| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Advanced Mathematics | Trivial topology | Complex homotopy |
| Pure Mathematics | Low genus | Modular forms |
| Applied Mathematics | Ill-conditioned | Spectral convergence |
| Category Theory | Set-theoretic | ∞-categories |

### Information & Computing
| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Information Theory | Pure/deterministic | Maximally mixed |
| Signal Processing | Low SNR | Optimal coding |
| Distributed Systems | Eventual consistency | Strong consistency |
| Information Systems | Poor retrieval | Perfect precision |
| Cybersecurity | Compromised | Zero-trust fortified |
| AI/ML | Overfitting | Perfect generalization |

### Complex & Social Systems
| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Economics | Efficient markets | Crashes, bubbles |
| Complex Systems | Disordered phase | Critical point |
| Social Systems | Fragmented | Consensus |
| Networks | Disconnected | Giant component |
| Game Theory | Classical Nash | Entangled strategies |

### Cognitive & Biological
| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Cognition | Fragmented | Integrated consciousness |
| Cognitive Neuroscience | Inattentive | Full metacognition |
| Semantic Structure | Incoherent | Formal ontology |
| Recursive Learning | Static algorithm | Self-improving |

### Engineering & Applied
| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Control Theory | Unstable | Optimal control |
| Nonlinear Dynamics | Fixed point | Chaotic attractor |
| Mechanical Systems | Inefficient | Carnot limit |
| Education | Rote memorization | Deep understanding |
| Work-Life Balance | Burnout | Flow state |
| UX/Accessibility | Inaccessible | WCAG AAA |
| Chemistry | Normal materials | Superconductors/BEC |

> **Note**: Cross-domain θ values are analogy scores produced by domain-specific mappings (see `domains/`), not literal quantum parameters.

---

## Physics Hypothesis (Speculative)

The following is a *hypothesis*, not established physics:

Theta may represent a fundamental interpolation parameter between classical and quantum descriptions—the "bridge" that smoothly connects:

- **θ → 0**: General Relativity (spacetime curvature, determinism)
- **θ → 1**: Quantum Field Theory (superposition, entanglement)
- **0 < θ < 1**: The transition regime where both descriptions contribute

The interpolation formula:
```
O = θ × O_quantum + (1-θ) × O_classical
```

This remains speculative until peer-reviewed validation.

---

## Project Structure

```
theta_calculator/
├── core/                    # Core computational engine
│   ├── theta_state.py       # Data structures
│   └── interpolation.py     # 5 theta calculation methods
├── constants/               # Physical constants (CODATA 2022)
│   ├── codata_2022.py       # 308 constants
│   ├── values.py            # Core values
│   └── planck_units.py      # Planck scale
├── domains/                 # 35 cross-domain extensions
│   ├── # Physics & Foundations (10)
│   ├── quantum_computing.py # Error thresholds, decoherence
│   ├── quantum_biology.py   # Photosynthesis, enzyme tunneling
│   ├── quantum_gravity.py   # Planck-scale physics
│   ├── quantum_foundations.py # Decoherence, measurement
│   ├── cosmology.py         # Big Bang to heat death
│   ├── condensed_matter.py  # Phase transitions, topological
│   ├── high_energy_physics.py # QCD, lattice gauge theory
│   ├── atomic_optical_physics.py # BEC, cavity QED
│   ├── physics_extended.py  # GR, holography
│   ├── # Mathematics (4)
│   ├── advanced_mathematics.py # Topology, geometry
│   ├── pure_mathematics.py  # Algebraic geometry
│   ├── applied_mathematics.py # PDE, optimization
│   ├── category_theory.py   # Functors, topoi
│   ├── # Information & Computing (6)
│   ├── information.py       # Shannon vs von Neumann entropy
│   ├── signal_processing.py # SNR, compression
│   ├── distributed_systems.py # CAP theorem
│   ├── information_systems.py # IR, graphics, code
│   ├── cybersecurity.py     # Attack surface, MTTD
│   ├── ai_ml.py             # Generalization, attention
│   ├── # Complex & Social (5)
│   ├── economics.py         # Ising model markets
│   ├── complex_systems.py   # Critical phenomena
│   ├── social_systems.py    # Opinion dynamics, epidemics
│   ├── networks.py          # Percolation, QKD
│   ├── game_theory.py       # Quantum game entanglement
│   ├── # Cognitive & Biological (4)
│   ├── cognition.py         # IIT, neural criticality
│   ├── cognitive_neuro.py   # Attention, metacognition
│   ├── semantic_structure.py # Text coherence
│   ├── recursive_learning.py # Meta-learning
│   ├── # Engineering & Applied (7)
│   ├── control_theory.py    # Stability margins
│   ├── nonlinear_dynamics.py # Lyapunov, chaos
│   ├── mechanical_systems.py # Efficiency, damping
│   ├── education.py         # Learning curves
│   ├── work_life_balance.py # Burnout, JD-R
│   ├── ux_accessibility.py  # WCAG, SUS
│   ├── chemistry.py         # Superconductors, BEC
│   └── universal.py         # Cross-domain framework
├── proofs/                  # Validation framework
│   ├── unified.py
│   ├── mathematical/
│   ├── information/
│   └── cross_domain/        # Universal proofs
├── visualization/           # Plotting and explanation
├── tests/                   # 1,966 tests across 43 files
└── cli.py                   # Command-line interface
```

---

## License

MIT

---

## References

This project uses established physics and computational foundations. All citations reference entries in `BIBLIOGRAPHY.bib`.

### Fundamental Constants
- CODATA 2022 values: \cite{CODATA2022}

### Black Hole Thermodynamics
- Hawking radiation: \cite{Hawking1974}, \cite{Hawking1975}
- Bekenstein entropy: \cite{Bekenstein1973}
- Schwarzschild metric: \cite{Schwarzschild1916}

### Information Theory
- Shannon entropy: \cite{Shannon1948}
- von Neumann entropy: \cite{VonNeumann1932}
- Landauer limit: \cite{Landauer1961}
- Bekenstein bound: \cite{Bekenstein1981}
- Margolus-Levitin limit: \cite{MargoluLevitin1998}

### Quantum Mechanics & Decoherence
- Quantum Darwinism: \cite{Zurek2003}
- Gravitational decoherence: \cite{Penrose1996}
- Environmental decoherence: \cite{JoosZeh1985}

### Foundational Physics
- Planck constant: \cite{Planck1901}
- Special relativity: \cite{Einstein1905SR}
