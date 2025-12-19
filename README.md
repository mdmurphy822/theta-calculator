# Theta Calculator

A Python library that mathematically and computationally proves the existence of **theta (θ)** as the universal quantum-classical interpolation parameter across physics, economics, information theory, game theory, complex systems, and quantum computing.

## What is Theta?

Theta represents the degree to which a system exhibits quantum versus classical behavior:

- **θ = 1**: Quantum limit (coherent, correlated, entangled)
- **θ = 0**: Classical limit (deterministic, independent, random)
- **0 < θ < 1**: Transition regime (mixed behavior)

This concept is **universal** - the same mathematical structure appears across all domains:

| Domain | θ = 0 (Classical) | θ = 1 (Quantum) |
|--------|-------------------|-----------------|
| Physics | Planets, baseballs | Electrons, photons |
| Economics | Efficient markets | Crashes, bubbles |
| Information | Pure/deterministic | Maximally mixed |
| Game Theory | Classical Nash | Entangled strategies |
| Complex Systems | Disordered phase | Critical point |
| Quantum Computing | Noisy/decoherent | Coherent qubits |

For any observable O:
```
O = θ × O_quantum + (1-θ) × O_classical
```

## Installation

```bash
cd theta_calculator
pip install -r requirements.txt
```

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

### Quick theta estimate

```bash
python3 -m theta_calculator quick --mass 0.145 --length 0.074 --temp 300
```

### Multi-Domain Theta

**List all domains and their example systems:**
```bash
python3 -m theta_calculator domains
```

**Analyze a specific domain system:**
```bash
# Market crash (economics)
python3 -m theta_calculator domain -d economics -s market_crash

# Google Willow quantum chip
python3 -m theta_calculator domain -d quantum_computing -s google_willow

# Ferromagnet at critical temperature
python3 -m theta_calculator domain -d complex_systems -s ferromagnet_critical
```

**Cross-domain comparison:**
```bash
python3 -m theta_calculator crossdomain
```

Output:
```
UNIVERSAL THETA: Cross-Domain Comparison
================================================================================
ECONOMICS
--------------------------------------------------
  flash_crash                    0.727 [██████████████░░░░░░]
  market_crash                   0.707 [██████████████░░░░░░]
  efficient_market               0.402 [████████░░░░░░░░░░░░]

QUANTUM COMPUTING
--------------------------------------------------
  quantinuum_h2                  0.983 [███████████████████░]
  google_willow                  0.925 [██████████████████░░]
  noisy_classical                0.000 [░░░░░░░░░░░░░░░░░░░░]
```

## The Proof

Theta is proven through three independent methodologies that converge:

### 1. Numerical Proof
Five computational methods that should agree:
- **Action Ratio**: θ = ℏ/S (quantum when action ≈ Planck constant)
- **Thermal Ratio**: θ = (ℏω)/(kT) (quantum vs thermal energy)
- **Scale Ratio**: Compare Planck/de Broglie wavelength to system size
- **Decoherence**: θ = exp(-t/t_D) (coherence decay rate)
- **Unified**: Weighted combination with confidence scoring

### 2. Mathematical Proof (Constant Bootstrap)
Demonstrates that fundamental constants recursively define each other:
- c = 1/√(ε₀μ₀)
- α = e²/(4πε₀ℏc)
- G = ℏc/m_P²
- l_P/t_P = c

### 3. Information-Theoretic Proof
- **Bekenstein Bound**: Maximum entropy in volume with given energy
- **Landauer Limit**: Thermodynamic computation limits

## Project Structure

```
theta_calculator/
├── core/                    # Core computational engine
│   ├── theta_state.py       # Data structures
│   └── interpolation.py     # 5 theta calculation methods
├── constants/               # Physical constants (CODATA 2022)
│   ├── codata_2022.py       # 308 constants incl. particle physics & cosmology
│   ├── values.py            # Core values
│   └── planck_units.py      # Planck scale
├── domains/                 # Multi-domain theta framework
│   ├── economics.py         # Ising model markets
│   ├── information.py       # Shannon vs von Neumann entropy
│   ├── game_theory.py       # Quantum game entanglement
│   ├── complex_systems.py   # Critical phenomena
│   ├── quantum_computing.py # Error thresholds
│   └── universal.py         # Cross-domain unification
├── proofs/                  # Three-part proof framework
│   ├── unified.py
│   ├── mathematical/
│   └── information/
├── visualization/           # Plotting and explanation
└── cli.py                   # Command-line interface
```

## Constants

The library includes **308 CODATA 2022 fundamental constants** including:

- **Core constants**: c, h, ℏ, e, k_B, N_A, G
- **Electromagnetic**: α, μ₀, ε₀, Z₀, Φ₀
- **Particle masses**: electron, proton, neutron, muon, tau
- **Planck units**: l_P, t_P, m_P, E_P, T_P
- **Particle physics**: W, Z, Higgs masses, α_s, sin²θ_W
- **Cosmological**: H₀, Λ, Ω_Λ, Ω_m, ρ_c, T_CMB

## Arxiiv Knowledge Repo

The `arxiiv-knowledge-repo.tar.gz` file (tracked via Git LFS) contains downloaded research papers organized by category:
- `papers/physics/` - Quantum mechanics, thermodynamics
- `papers/math/` - Mathematical foundations
- `papers/ai_ml/` - Machine learning applications
- `papers/other/` - Related research

## License

MIT
