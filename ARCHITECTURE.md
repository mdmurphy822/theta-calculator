# Theta Calculator Architecture

## Core Concept

The Theta Calculator implements θ (theta) as the **quantum-classical interpolation parameter**:

```
θ = ℏ / S
```

Where:
- ℏ (h-bar) is the reduced Planck constant (≈ 1.054 × 10⁻³⁴ J·s)
- S is the characteristic action of the system

**Key insight:** When S ≈ ℏ, quantum effects dominate (θ → 1). When S >> ℏ, classical behavior emerges (θ → 0).

---

## Module Structure

```
theta_calculator/
├── __init__.py           # Public API exports
├── __main__.py           # CLI entry point
├── cli.py                # Command-line interface
│
├── core/                 # Core computation
│   ├── theta_state.py    # ThetaState, PhysicalSystem, Regime
│   └── interpolation.py  # ThetaCalculator methods
│
├── constants/            # Physical constants
│   ├── values.py         # FundamentalConstants
│   ├── planck_units.py   # PlanckUnits
│   ├── codata_2022.py    # CODATA 2022 values
│   └── connections.py    # Constant relationships
│
├── proofs/               # Proof methodologies
│   ├── unified.py        # UnifiedThetaProof
│   ├── information/      # Information-theoretic proofs
│   │   ├── bekenstein_bound.py
│   │   └── landauer_limit.py
│   └── mathematical/     # Mathematical derivations
│       └── constant_bootstrap.py
│
├── domains/              # Multi-domain theta
│   ├── economics.py      # Market phase transitions
│   ├── information.py    # Entropy measures
│   ├── game_theory.py    # Quantum games
│   ├── complex_systems.py # Critical phenomena
│   ├── quantum_computing.py # Error thresholds
│   ├── quantum_biology.py # Biological coherence
│   ├── cosmology.py      # Cosmic evolution
│   ├── control_theory.py # Stability margins
│   ├── nonlinear_dynamics.py # Chaos theory
│   ├── quantum_gravity.py # Planck-scale physics
│   └── universal.py      # Cross-domain unification
│
├── physics/              # Physical models
│   ├── decoherence.py    # Decoherence models
│   └── black_holes.py    # Black hole thermodynamics
│
├── visualization/        # Output formatting
│   ├── theta_landscape.py
│   └── proof_narrator.py
│
└── tests/                # Test suite (425 tests)
```

---

## Data Flow

```
┌─────────────────────┐
│   PhysicalSystem    │  Input: mass, length, energy, temperature
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ThetaCalculator   │  Applies multiple methods
│                     │
│  ┌───────────────┐  │
│  │ action_theta  │──┼──► θ = ℏ/S
│  ├───────────────┤  │
│  │ thermal_theta │──┼──► θ = E_q/E_th
│  ├───────────────┤  │
│  │ scale_theta   │──┼──► θ = L_P/L
│  ├───────────────┤  │
│  │ decoherence   │──┼──► θ = τ_coh/τ_dyn
│  └───────────────┘  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     ThetaState      │  Output: theta ∈ [0,1], regime, method
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Regime Class     │
│                     │
│  θ > 0.7  → QUANTUM │
│  θ < 0.3  → CLASSICAL│
│  else     → TRANSITION│
└─────────────────────┘
```

---

## Domain Architecture

Each domain module follows the same pattern:

```python
# 1. System dataclass
@dataclass
class DomainSystem:
    name: str
    # Domain-specific parameters
    ...

# 2. Compute theta function
def compute_domain_theta(system: DomainSystem) -> float:
    """Returns theta in [0, 1]"""
    ...

# 3. Predefined systems dictionary
DOMAIN_SYSTEMS: Dict[str, DomainSystem] = {
    "example_low_theta": DomainSystem(...),
    "example_high_theta": DomainSystem(...),
}

# 4. Classification function (optional)
def classify_regime(theta: float) -> DomainRegime:
    """Map theta to domain-specific regime"""
    ...
```

### Domain-Theta Mappings

| Domain | θ = 0 (Classical) | θ = 1 (Quantum) |
|--------|-------------------|-----------------|
| Physics | S >> ℏ | S ≈ ℏ |
| Economics | Efficient market | Crash/bubble |
| Information | Pure state | Maximally mixed |
| Game Theory | Nash equilibrium | Entangled strategies |
| Complex Systems | Disordered | Critical point |
| Quantum Computing | Noisy/decoherent | Coherent |
| Quantum Biology | Classical chemistry | Coherent transfer |
| Cosmology | Present day | Planck era |
| Control Theory | Unstable | Optimal |
| Nonlinear Dynamics | Fixed point | Chaotic |
| Quantum Gravity | Macroscopic | Planck scale |

---

## Proof Architecture

The `UnifiedThetaProof` combines three independent methodologies:

```
┌──────────────────────────────────────────────────────┐
│                 UnifiedThetaProof                     │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────┐  ┌─────────────────┐            │
│  │  Mathematical   │  │   Numerical     │            │
│  │                 │  │                 │            │
│  │ constant_boot-  │  │ ThetaCalculator │            │
│  │ strap.py        │  │ (4 methods)     │            │
│  └────────┬────────┘  └────────┬────────┘            │
│           │                    │                      │
│           └────────┬───────────┘                      │
│                    │                                  │
│           ┌────────┴────────┐                        │
│           │   Information   │                        │
│           │                 │                        │
│           │ Bekenstein +    │                        │
│           │ Landauer limits │                        │
│           └────────┬────────┘                        │
│                    │                                  │
│                    ▼                                  │
│  ┌─────────────────────────────────────────────────┐ │
│  │            UnifiedProofResult                   │ │
│  │  - theta: float                                 │ │
│  │  - theta_values: Dict[str, float]              │ │
│  │  - proof_agreement: float                      │ │
│  │  - is_valid: bool                              │ │
│  │  - regime: Regime                              │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Validation:** The proof is considered valid when all methodologies agree within tolerance (typically σ < 0.1).

---

## Constants Architecture \cite{CODATA2022}

Physical constants use CODATA 2022 values with SI/2019 exact definitions:

```
┌─────────────────────────────────────────────┐
│          FundamentalConstants               │
├─────────────────────────────────────────────┤
│ EXACT (SI-2019 defined):                   │
│   c = 299,792,458 m/s                      │
│   h = 6.62607015 × 10⁻³⁴ J·s              │
│   k_B = 1.380649 × 10⁻²³ J/K              │
│   e = 1.602176634 × 10⁻¹⁹ C               │
│                                             │
│ MEASURED (CODATA 2022):                    │
│   G = 6.67430 × 10⁻¹¹ m³/(kg·s²)          │
│   α = 7.2973525693 × 10⁻³                  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              PlanckUnits                    │
├─────────────────────────────────────────────┤
│ Derived from: ℏ, c, G, k_B                 │
│                                             │
│   l_P = √(ℏG/c³) = 1.616 × 10⁻³⁵ m       │
│   t_P = l_P/c = 5.391 × 10⁻⁴⁴ s           │
│   m_P = √(ℏc/G) = 2.176 × 10⁻⁸ kg         │
│   E_P = m_P c² = 1.956 × 10⁹ J            │
│   T_P = E_P/k_B = 1.417 × 10³² K          │
└─────────────────────────────────────────────┘
```

---

## CLI Architecture

```
python -m theta_calculator [COMMAND] [OPTIONS]

Commands:
┌──────────────┬────────────────────────────────────────┐
│ prove        │ Full proof for a system                │
│ constants    │ Display physical constants             │
│ explain      │ Explain theta for named system         │
│ quick        │ Fast calculation (action ratio only)   │
│ compare      │ Compare example systems                │
│ domains      │ List available domains                 │
│ domain       │ Analyze specific domain system         │
│ crossdomain  │ Cross-domain comparison                │
└──────────────┴────────────────────────────────────────┘
```

---

## Extension Guide

### Adding a New Domain

1. **Create the module:** `theta_calculator/domains/new_domain.py`

```python
from dataclasses import dataclass
from typing import Dict
from enum import Enum

class NewDomainRegime(Enum):
    LOW = "low"
    HIGH = "high"

@dataclass
class NewDomainSystem:
    name: str
    # Add domain-specific parameters
    parameter1: float
    parameter2: float

def compute_new_domain_theta(system: NewDomainSystem) -> float:
    """Compute theta for new domain."""
    # Map domain parameters to theta ∈ [0, 1]
    # Return 0 for classical-like, 1 for quantum-like
    return ...

def classify_regime(theta: float) -> NewDomainRegime:
    if theta < 0.5:
        return NewDomainRegime.LOW
    return NewDomainRegime.HIGH

NEW_DOMAIN_SYSTEMS: Dict[str, NewDomainSystem] = {
    "example_low": NewDomainSystem(name="Low", parameter1=0.1, parameter2=0.2),
    "example_high": NewDomainSystem(name="High", parameter1=0.9, parameter2=0.8),
}
```

2. **Export in `domains/__init__.py`:**

```python
from .new_domain import (
    NewDomainSystem,
    compute_new_domain_theta,
    NEW_DOMAIN_SYSTEMS,
)
```

3. **Add to CLI in `cli.py`:** Update the domain listing and analysis commands.

4. **Create tests:** `tests/test_new_domain.py`

---

## Design Decisions

### Why Multiple Theta Methods?

Different methods access different aspects of "quantumness":

1. **Action ratio (ℏ/S):** Direct quantum-classical boundary
2. **Thermal ratio:** Competition between quantum and thermal fluctuations
3. **Scale ratio:** Comparison to Planck length
4. **Decoherence:** Persistence of quantum coherence

Agreement across methods strengthens the proof.

### Why Domain Extensions?

Theta as a universal parameter appears across physics:
- **Economics:** Ising model maps traders to spins \cite{Bornholdt2001}
- **Biology:** Quantum coherence in warm systems \cite{Engel2007}
- **Computing:** Error correction thresholds \cite{Shor1996}

These aren't analogies—they're exact mathematical mappings.

### Why CODATA 2022?

The 2019 SI revision made ℏ, c, k_B exact by definition. This eliminates uncertainty propagation in Planck unit calculations.

---

## Testing Strategy

- **Unit tests:** Each function tested in isolation
- **Integration tests:** End-to-end proof verification
- **Consistency tests:** Methods should agree for standard systems
- **Domain tests:** All predefined systems produce valid theta

Current coverage: **425 tests** across 14 test files.

---

## Performance Considerations

- All calculations are O(1) or O(n) where n is small
- No matrix inversions or iterative solvers in critical path
- Domain systems use precomputed values where possible
- Suitable for interactive CLI use

---

## Future Directions

1. **GPU acceleration** for Monte Carlo theta estimation
2. **Uncertainty propagation** through all calculations
3. **Interactive visualization** of theta landscapes
4. **Machine learning** for theta prediction from experimental data
5. **Additional domains:** Finance, epidemiology, neural networks

---

## References

All citations reference entries in `BIBLIOGRAPHY.bib`.

- \cite{CODATA2022} - NIST CODATA 2022 fundamental constants
- \cite{Bornholdt2001} - Ising model for financial markets
- \cite{Engel2007} - Quantum coherence in photosynthesis
- \cite{Shor1996} - Fault-tolerant quantum computation
- \cite{Hawking1974} - Black hole thermodynamics
- \cite{Bekenstein1981} - Bekenstein entropy bound
- \cite{Landauer1961} - Landauer limit on computation
