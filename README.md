# Theta Calculator

A scoring framework that estimates the "quantum-likeness" of physical systems using action-scale ratios.

## Quick Start

```bash
pip install -r requirements.txt

# Compare theta across systems
python3 -m theta_calculator compare

# Score a custom system
python3 -m theta_calculator quick --mass 9.1e-31 --length 2.8e-15 --temp 300
```

---

## What is Theta?

**Theta (θ)** is a dimensionless parameter that compares a system's characteristic action scale to Planck's constant (ℏ):

```
θ = ℏ / S    (Planck action / system action)
```

- **θ → 0**: Classical regime (large action, deterministic behavior)
- **θ → 1**: Quantum regime (action ≈ ℏ, probabilistic behavior)
- **0 < θ < 1**: Transition regime

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
python3 -m theta_calculator compare
```

Output:
```
THETA COMPARISON ACROSS PHYSICAL SYSTEMS
============================================================
System                         Theta Regime
------------------------------------------------------------
electron                    0.869991 transition
proton                      0.607457 transition
hydrogen_atom               0.770084 transition
water_molecule              0.769286 transition
virus                       0.422613 transition
human_cell                  0.232905 transition
baseball                    0.130017 transition
human                       0.130001 transition
earth                       0.130000 transition
stellar_black_hole          0.508157 transition
```

### Prove theta for a custom system

```bash
python3 -m theta_calculator prove --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300
```

### Display fundamental constants

```bash
python3 -m theta_calculator constants --show-planck --show-bootstrap
```

### Get detailed explanation for a system

```bash
python3 -m theta_calculator explain --system electron --format markdown
```

### Cross-domain analysis

```bash
# List all domains
python3 -m theta_calculator domains

# Analyze specific domain system
python3 -m theta_calculator domain -d economics -s market_crash
python3 -m theta_calculator domain -d quantum_computing -s google_willow

# Cross-domain comparison
python3 -m theta_calculator crossdomain
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

The theta framework extends to other domains exhibiting classical-quantum-like transitions:

| Domain | θ → 0 | θ → 1 |
|--------|-------|-------|
| Physics | Planets, baseballs | Electrons, photons |
| Economics | Efficient markets | Crashes, bubbles |
| Information | Pure/deterministic | Maximally mixed |
| Game Theory | Classical Nash | Entangled strategies |
| Complex Systems | Disordered phase | Critical point |
| Quantum Computing | Noisy/decoherent | Coherent qubits |

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
├── domains/                 # Cross-domain extensions
│   ├── economics.py         # Ising model markets
│   ├── information.py       # Shannon vs von Neumann entropy
│   ├── game_theory.py       # Quantum game entanglement
│   ├── complex_systems.py   # Critical phenomena
│   ├── quantum_computing.py # Error thresholds
│   └── universal.py         # Cross-domain framework
├── proofs/                  # Validation framework
│   ├── unified.py
│   ├── mathematical/
│   └── information/
├── visualization/           # Plotting and explanation
└── cli.py                   # Command-line interface
```

---

## License

MIT
