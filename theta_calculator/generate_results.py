#!/usr/bin/env python3
"""
Generate PRECALCULATED_RESULTS.md with improved formatting.

This script computes theta for all domain systems and generates
a markdown file with:
- Scientific notation for actual theta values
- Log10 scale for easy comparison
- Domain statistics (range, counts by regime)
"""

import math
import sys
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np


def format_theta(theta: float) -> Tuple[str, str]:
    """
    Format theta with scientific notation and log scale.

    Returns:
        (theta_str, log_str): Formatted theta and log10(theta)
    """
    if theta is None or math.isnan(theta):
        return "nan", "nan"
    if theta <= 0:
        return "0", "-inf"
    if theta >= 1.0:
        return "1.0000", "0.0"

    log_theta = math.log10(theta)

    # Use different formatting based on magnitude
    if theta >= 0.001:
        theta_str = f"{theta:.4f}"
    else:
        theta_str = f"{theta:.2e}"

    log_str = f"{log_theta:.1f}"

    return theta_str, log_str


def classify_regime(theta: float) -> str:
    """Classify theta into regime categories."""
    if theta is None or math.isnan(theta):
        return "Error"
    if theta >= 0.5:
        return "Quantum"
    elif theta >= 0.1:
        return "Transition"
    elif theta >= 0.001:
        return "Semiclassical"
    else:
        return "Classical"


def compute_domain_stats(thetas: list) -> Dict[str, Any]:
    """Compute statistics for a domain's theta values."""
    valid = [t for t in thetas if t is not None and not math.isnan(t) and t > 0]

    if not valid:
        return {"min": None, "max": None, "range_orders": 0, "counts": {}}

    counts = {"Quantum": 0, "Transition": 0, "Semiclassical": 0, "Classical": 0, "Error": 0}
    for t in thetas:
        regime = classify_regime(t)
        counts[regime] = counts.get(regime, 0) + 1

    return {
        "min": min(valid),
        "max": max(valid),
        "range_orders": math.log10(max(valid)) - math.log10(min(valid)) if min(valid) > 0 else 0,
        "counts": counts
    }


def get_theta_value(result) -> float:
    """Extract theta value from various return types."""
    if isinstance(result, dict):
        # Some domains return dict with 'composite' or 'theta' key
        if 'composite' in result:
            return float(result['composite'])
        elif 'theta' in result:
            return float(result['theta'])
        else:
            # Try first numeric value
            for v in result.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
    elif isinstance(result, (int, float)):
        return float(result)
    return float('nan')


# Domain configurations: (systems_dict_name, compute_func_name, system_key_for_compute)
DOMAIN_CONFIGS = [
    ("Electromagnetic Spectrum", "EM_SPECTRUM", "compute_em_theta", "electromagnetic"),
    ("Economics", "ECONOMIC_SYSTEMS", "compute_market_theta", "economics"),
    ("Information Theory", "INFORMATION_SYSTEMS", "compute_information_theta", "information"),
    ("Game Theory", "GAME_SYSTEMS", "compute_entanglement_theta", "game_theory"),
    ("Complex Systems", "COMPLEX_SYSTEMS", "compute_complex_theta", "complex_systems"),
    ("Quantum Computing", "QUANTUM_HARDWARE", "compute_quantum_computing_theta", "quantum_computing"),
    ("Quantum Biology", "BIOLOGICAL_SYSTEMS", "compute_quantum_bio_theta", "quantum_biology"),
    ("Cosmology", "COSMIC_TIMELINE", "compute_cosmic_theta", "cosmology"),
    ("Control Theory", "CONTROL_SYSTEMS", "compute_control_theta", "control_theory"),
    ("Nonlinear Dynamics", "DYNAMICAL_SYSTEMS", "compute_dynamics_theta", "nonlinear_dynamics"),
    ("Quantum Gravity", "QUANTUM_GRAVITY_SYSTEMS", "compute_quantum_gravity_theta", "quantum_gravity"),
    ("Education", "EDUCATION_SYSTEMS", "compute_education_theta", "education"),
    ("Mechanical Systems", "MECHANICAL_SYSTEMS", "compute_mechanical_theta", "mechanical_systems"),
    ("Networks", "NETWORK_SYSTEMS", "compute_network_theta", "networks"),
    ("Cognition", "COGNITIVE_SYSTEMS", "compute_cognition_theta", "cognition"),
    ("Social Systems", "SOCIAL_SYSTEMS", "compute_social_theta", "social_systems"),
    ("Chemistry", "SUPERCONDUCTORS", "compute_chemistry_theta", "chemistry"),
    ("Work-Life Balance", "WORK_LIFE_SYSTEMS", "compute_work_life_theta", "work_life_balance"),
    ("Cybersecurity", "SECURITY_SYSTEMS", "compute_security_theta", "cybersecurity"),
    ("AI/ML", "ML_SYSTEMS", "compute_ml_theta", "ai_ml"),
    ("Category Theory", "CATEGORY_SYSTEMS", "compute_category_theta", "category_theory"),
    ("Semantic Structure", "SEMANTIC_SYSTEMS", "compute_semantic_theta", "semantic_structure"),
    ("Recursive Learning", "RECURSIVE_SYSTEMS", "compute_recursive_theta", "recursive_learning"),
    ("Quantum Foundations", "QUANTUM_FOUNDATIONS_SYSTEMS", "compute_quantum_foundations_theta", "quantum_foundations"),
    ("Cognitive Neuroscience", "COGNITIVE_NEURO_SYSTEMS", "compute_cognitive_neuro_theta", "cognitive_neuro"),
    ("Physics Extended", "PHYSICS_EXTENDED_SYSTEMS", "compute_physics_extended_theta", "physics_extended"),
    ("Condensed Matter", "CONDENSED_MATTER_SYSTEMS", "compute_condensed_matter_theta", "condensed_matter"),
    ("Information Systems", "INFORMATION_SYSTEMS", "compute_information_system_theta", "information_systems"),
    ("UX/Accessibility", "UX_ACCESSIBILITY_SYSTEMS", "compute_ux_accessibility_theta", "ux_accessibility"),
    ("Advanced Mathematics", "MATHEMATICAL_SYSTEMS", "compute_math_theta", "advanced_mathematics"),
    ("High Energy Physics", "HEP_SYSTEMS", "compute_hep_theta", "high_energy_physics"),
    ("Atomic/Optical Physics", "ATOMIC_OPTICAL_SYSTEMS", "compute_atomic_optical_theta", "atomic_optical_physics"),
    ("Pure Mathematics", "PURE_MATH_SYSTEMS", "compute_pure_math_theta", "pure_mathematics"),
    ("Applied Mathematics", "APPLIED_MATH_SYSTEMS", "compute_applied_math_theta", "applied_mathematics"),
    ("Distributed Systems", "DISTRIBUTED_SYSTEMS", "compute_distributed_theta", "distributed_systems"),
    ("Signal Processing", "SIGNAL_SYSTEMS", "compute_signal_theta", "signal_processing"),
]


def generate_domain_section(domain_name: str, systems_dict: Dict, compute_func, module) -> str:
    """Generate markdown section for a domain."""
    lines = []
    thetas = []
    rows = []

    for name, system in systems_dict.items():
        try:
            result = compute_func(system)
            theta = get_theta_value(result)
            thetas.append(theta)

            theta_str, log_str = format_theta(theta)
            regime = classify_regime(theta)

            rows.append((name, theta_str, log_str, regime))
        except Exception as e:
            thetas.append(float('nan'))
            rows.append((name, "*error*", "-", f"Error: {str(e)[:20]}"))

    # Sort by theta descending (quantum systems first)
    rows.sort(key=lambda x: float('-inf') if x[1] == '*error*' or x[1] == 'nan'
              else -float(x[1].replace('e', 'E')))

    # Header
    lines.append(f"## {domain_name} ({len(systems_dict)} systems)")
    lines.append("")
    lines.append("| System | θ | log₁₀(θ) | Regime |")
    lines.append("|--------|---|----------|--------|")

    for name, theta_str, log_str, regime in rows:
        lines.append(f"| {name} | {theta_str} | {log_str} | {regime} |")

    # Statistics
    stats = compute_domain_stats(thetas)
    lines.append("")
    if stats["max"] is not None:
        max_str, _ = format_theta(stats["max"])
        min_str, _ = format_theta(stats["min"])
        lines.append(f"**Range:** {min_str} to {max_str} ({stats['range_orders']:.0f} orders of magnitude)")

        regime_summary = ", ".join(f"{k}: {v}" for k, v in stats["counts"].items() if v > 0)
        lines.append(f"**Regimes:** {regime_summary}")

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate the PRECALCULATED_RESULTS.md file."""
    output_lines = []

    # Header
    output_lines.append("# Precalculated Theta Results")
    output_lines.append("")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    output_lines.append("## Format")
    output_lines.append("")
    output_lines.append("Each table shows:")
    output_lines.append("- **θ**: The theta value (scientific notation for small values)")
    output_lines.append("- **log₁₀(θ)**: Log scale for easy comparison across orders of magnitude")
    output_lines.append("- **Regime**: Classification based on theta value")
    output_lines.append("  - Quantum (θ ≥ 0.5): Strong quantum/correlated effects")
    output_lines.append("  - Transition (0.1 ≤ θ < 0.5): Intermediate regime")
    output_lines.append("  - Semiclassical (0.001 ≤ θ < 0.1): Weak quantum effects")
    output_lines.append("  - Classical (θ < 0.001): Classical/random behavior")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")

    # Process each domain
    for domain_name, systems_name, func_name, module_name in DOMAIN_CONFIGS:
        try:
            # Dynamic import
            module = __import__(f"theta_calculator.domains.{module_name}", fromlist=[systems_name, func_name])
            systems_dict = getattr(module, systems_name)
            compute_func = getattr(module, func_name)

            section = generate_domain_section(domain_name, systems_dict, compute_func, module)
            output_lines.append(section)
            print(f"✓ {domain_name}: {len(systems_dict)} systems")
        except Exception as e:
            output_lines.append(f"## {domain_name}")
            output_lines.append("")
            output_lines.append(f"*Error loading domain: {e}*")
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")
            print(f"✗ {domain_name}: {e}")

    # Write output
    output_path = "/home/bacon/Desktop/Theta Proof/PRECALCULATED_RESULTS.md"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nGenerated: {output_path}")


if __name__ == "__main__":
    main()
