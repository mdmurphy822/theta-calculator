#!/usr/bin/env python3
"""
Theta Calculator CLI: Command-line interface for theta computations.

Usage:
    python -m theta_calculator score --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300
    python -m theta_calculator constants --show-planck
    python -m theta_calculator explain --system electron
    python -m theta_calculator landscape --output theta.png
    python -m theta_calculator compare --json
"""

import argparse
import json
import sys
from typing import Any, Dict

from .core.theta_state import PhysicalSystem, EXAMPLE_SYSTEMS
from .proofs.unified import UnifiedThetaProof, score_theta
from .constants.values import FundamentalConstants
from .constants.planck_units import PlanckUnits
from .proofs.mathematical.constant_bootstrap import ConstantBootstrap
from .visualization.proof_narrator import ProofNarrator
from .visualization.theta_landscape import ThetaLandscapePlotter

# Domain imports
from .domains import (
    ECONOMIC_SYSTEMS, INFORMATION_SYSTEMS, GAME_SYSTEMS,
    COMPLEX_SYSTEMS, QUANTUM_HARDWARE,
    compute_market_theta, compute_information_theta,
    compute_entanglement_theta, compute_complex_theta,
    compute_quantum_computing_theta, cross_domain_comparison,
)
from .domains.quantum_biology import (
    BIOLOGICAL_SYSTEMS, compute_quantum_bio_theta
)
from .domains.cosmology import (
    COSMIC_TIMELINE, compute_cosmic_theta
)
from .domains.control_theory import (
    CONTROL_SYSTEMS, compute_control_theta
)
from .domains.nonlinear_dynamics import (
    DYNAMICAL_SYSTEMS, compute_dynamics_theta
)
from .domains.quantum_gravity import (
    QUANTUM_GRAVITY_SYSTEMS, compute_quantum_gravity_theta
)


def format_json(data: Dict[str, Any], indent: int = 2) -> str:
    """Format data as JSON string."""
    return json.dumps(data, indent=indent, default=str)


def result_to_dict(result) -> Dict[str, Any]:
    """Convert UnifiedProofResult to JSON-serializable dict."""
    return {
        "theta": result.theta,
        "regime": result.regime.value,
        "method_agreement": result.method_agreement,
        "consistency_score": result.consistency_score,
        "theta_values": result.theta_values,
        "is_valid": result.is_valid,
        "validation_notes": result.validation_notes,
    }


def cmd_score(args):
    """Execute score/estimation command."""
    system = PhysicalSystem(
        name=args.name,
        mass=args.mass,
        length_scale=args.length,
        energy=args.energy,
        temperature=args.temp
    )

    estimator = UnifiedThetaProof()
    result = estimator.compute_theta(system)

    if getattr(args, 'json', False):
        output = result_to_dict(result)
        output["system"] = {
            "name": args.name,
            "mass": args.mass,
            "length": args.length,
            "energy": args.energy,
            "temperature": args.temp,
        }
        print(format_json(output))
    else:
        print(result.summary)
        print()
        if args.verbose:
            print(result.detailed_explanation)

    return 0 if result.is_valid else 1


def cmd_constants(args):
    """Display fundamental constants."""
    print("=" * 60)
    print("FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2022)")
    print("=" * 60)

    for name, const in FundamentalConstants.get_all().items():
        print(f"\n{const.name} ({const.symbol})")
        print(f"  Value: {const.value:.10e} {const.unit}")
        if const.uncertainty > 0:
            print(f"  Uncertainty: {const.uncertainty:.1e}")
        else:
            print("  Uncertainty: exact")

    if args.show_planck:
        print("\n" + "=" * 60)
        print("PLANCK UNITS")
        print("=" * 60)

        for name, value in PlanckUnits.get_all().items():
            print(f"  {name} = {value:.4e}")

    if args.show_bootstrap:
        print("\n" + "=" * 60)
        print("CONSTANT BOOTSTRAP RELATIONSHIPS")
        print("=" * 60)

        bootstrap = ConstantBootstrap()
        for const, deps in bootstrap.get_dependency_graph().items():
            print(f"  {const} <- {', '.join(deps)}")

        print("\nVerification:")
        for name, ok in bootstrap.verify_all_bootstraps().items():
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}")

    return 0


def cmd_explain(args):
    """Generate proof explanation for a known system."""
    # Get predefined system
    if args.system in EXAMPLE_SYSTEMS:
        system = EXAMPLE_SYSTEMS[args.system]
    else:
        print(f"Unknown system: {args.system}")
        print(f"Available: {', '.join(EXAMPLE_SYSTEMS.keys())}")
        return 1

    # Run estimation
    result = score_theta(system)

    # Narrate
    narrator = ProofNarrator()
    steps = narrator.narrate_proof(result)

    if args.format == "markdown":
        print(narrator.to_markdown(steps))
    elif args.format == "latex":
        print(narrator.to_latex(steps))
    else:
        print(narrator.to_text(steps))

    return 0


def cmd_landscape(args):
    """Generate theta landscape visualization code."""
    plotter = ThetaLandscapePlotter()
    landscape = plotter.mass_length_landscape(temperature=args.temp)

    print(plotter.landscape_summary(landscape))
    print()

    if args.code:
        print("MATPLOTLIB CODE:")
        print("-" * 40)
        print(plotter.to_matplotlib_data(landscape))

    return 0


def cmd_quick(args):
    """Quick theta calculation."""
    from .core.interpolation import estimate_theta_quick

    theta = estimate_theta_quick(args.mass, args.length, args.temp)

    if theta > 0.9:
        regime = "QUANTUM"
    elif theta < 0.1:
        regime = "CLASSICAL"
    else:
        regime = "TRANSITION"

    if getattr(args, 'json', False):
        output = {
            "theta": theta,
            "regime": regime,
            "system": {
                "mass": args.mass,
                "length": args.length,
                "temperature": args.temp,
            }
        }
        print(format_json(output))
    else:
        print("Quick theta estimate:")
        print(f"  Mass: {args.mass:.2e} kg")
        print(f"  Length: {args.length:.2e} m")
        print(f"  Temperature: {args.temp:.1f} K")
        print(f"  θ ≈ {theta:.6f}")
        print(f"  Regime: {regime}")

    return 0


def cmd_compare(args):
    """Compare theta across multiple systems."""
    estimator = UnifiedThetaProof()

    results = {}
    for name, system in EXAMPLE_SYSTEMS.items():
        result = estimator.compute_theta(system)
        results[name] = {
            "theta": result.theta,
            "regime": result.regime.value,
        }

    if getattr(args, 'json', False):
        print(format_json({"systems": results}))
    else:
        print("THETA COMPARISON ACROSS PHYSICAL SYSTEMS")
        print("=" * 60)
        print(f"{'System':<25} {'Theta':>10} {'Regime':<15}")
        print("-" * 60)
        for name, data in results.items():
            print(f"{name:<25} {data['theta']:>10.6f} {data['regime']:<15}")

    return 0


def cmd_domains(args):
    """List all available domains and their example systems."""
    domains_info = {
        "economics": ("Market phase transitions (Ising model)", ECONOMIC_SYSTEMS),
        "information": ("Shannon vs von Neumann entropy", INFORMATION_SYSTEMS),
        "game_theory": ("Quantum game entanglement", GAME_SYSTEMS),
        "complex_systems": ("Critical phenomena and phase transitions", COMPLEX_SYSTEMS),
        "quantum_computing": ("Error thresholds and coherence", QUANTUM_HARDWARE),
        "quantum_biology": ("Quantum effects in living systems", BIOLOGICAL_SYSTEMS),
        "cosmology": ("Theta across cosmic history", COSMIC_TIMELINE),
        "control_theory": ("Stability margins and feedback loops", CONTROL_SYSTEMS),
        "nonlinear_dynamics": ("Chaos, bifurcations, and edge of chaos", DYNAMICAL_SYSTEMS),
        "quantum_gravity": ("Planck scale emergence and spacetime", QUANTUM_GRAVITY_SYSTEMS),
    }

    print("=" * 70)
    print("THETA DOMAINS: Quantum-Classical Interpolation Across Fields")
    print("=" * 70)
    print()
    print("θ = 0: Classical limit (deterministic, independent, random)")
    print("θ = 1: Quantum limit (coherent, correlated, entangled)")
    print()

    for domain, (description, systems) in domains_info.items():
        print(f"\n{domain.upper().replace('_', ' ')}")
        print(f"  {description}")
        print(f"  Systems: {', '.join(systems.keys())}")

    if args.verbose:
        print("\n" + "=" * 70)
        print("DETAILED THETA VALUES")
        print("=" * 70)
        comparison = cross_domain_comparison()
        for domain, systems in comparison.items():
            print(f"\n{domain.upper().replace('_', ' ')}")
            print("-" * 50)
            for name, theta in sorted(systems.items(), key=lambda x: -x[1]):
                bar = "█" * int(theta * 20) + "░" * (20 - int(theta * 20))
                print(f"  {name:<30} {theta:.3f} [{bar}]")

    return 0


def cmd_domain(args):
    """Compute theta for a specific domain system."""
    domain = args.domain
    system_name = args.system

    # Map domain to systems and compute function
    domain_map = {
        "economics": (ECONOMIC_SYSTEMS, compute_market_theta, "Coupling", "Temperature"),
        "information": (INFORMATION_SYSTEMS, compute_information_theta, "Entropy", "Purity"),
        "game_theory": (GAME_SYSTEMS, compute_entanglement_theta, "Entanglement γ", ""),
        "complex_systems": (COMPLEX_SYSTEMS, compute_complex_theta, "Order param", "Reduced T"),
        "quantum_computing": (QUANTUM_HARDWARE, compute_quantum_computing_theta, "Error rate", "T1"),
        "quantum_biology": (BIOLOGICAL_SYSTEMS, compute_quantum_bio_theta, "Coherence", "Mechanism"),
        "cosmology": (COSMIC_TIMELINE, compute_cosmic_theta, "Temperature", "Energy"),
        "control_theory": (CONTROL_SYSTEMS, compute_control_theta, "Gain margin", "Phase margin"),
        "nonlinear_dynamics": (DYNAMICAL_SYSTEMS, compute_dynamics_theta, "Lyapunov exp", "Attractor"),
        "quantum_gravity": (QUANTUM_GRAVITY_SYSTEMS, compute_quantum_gravity_theta, "Length scale", "Energy"),
    }

    if domain not in domain_map:
        print(f"Unknown domain: {domain}")
        print(f"Available: {', '.join(domain_map.keys())}")
        return 1

    systems, compute_fn, param1, param2 = domain_map[domain]

    if system_name not in systems:
        print(f"Unknown system: {system_name}")
        print(f"Available in {domain}: {', '.join(systems.keys())}")
        return 1

    system = systems[system_name]
    theta = compute_fn(system)

    # Determine regime
    if theta < 0.1:
        regime = "Classical"
    elif theta > 0.9:
        regime = "Quantum"
    elif theta > 0.5:
        regime = "Quantum-leaning"
    else:
        regime = "Classical-leaning"

    print(f"\n{domain.upper().replace('_', ' ')} THETA ANALYSIS")
    print("=" * 50)
    print(f"System: {system.name}")
    print(f"θ = {theta:.4f}")
    print(f"Regime: {regime}")
    print()

    # Show domain-specific details
    if domain == "economics":
        print(f"Coupling strength: {system.coupling_strength:.4f}")
        print(f"Temperature: {system.temperature:.2f}")
        print(f"Order parameter: {system.order_parameter:.3f}")
    elif domain == "information":
        print(f"Dimension: {system.dimension}")
        if system.purity:
            print(f"Purity: {system.purity:.4f}")
    elif domain == "game_theory":
        print(f"Entanglement γ: {system.gamma:.4f}")
        print(f"Game type: {system.game_type.value}")
    elif domain == "complex_systems":
        print(f"Order parameter: {system.order_parameter:.3f}")
        print(f"Reduced temperature: {system.reduced_temperature:.4f}")
    elif domain == "quantum_computing":
        print(f"Error rate: {system.error_rate:.2e}")
        print(f"T1 coherence: {system.T1*1e6:.1f} μs")
        print(f"Below threshold: {system.is_below_threshold}")
    elif domain == "quantum_biology":
        print(f"Mechanism: {system.mechanism.value}")
        print(f"Coherence time: {system.coherence_time:.2e} s")
        print(f"Organism: {system.organism}")
    elif domain == "cosmology":
        print(f"Temperature: {system.temperature:.2e} K")
        print(f"Energy: {system.energy:.2e} eV")
        print(f"Era: {system.era.value}")
    elif domain == "control_theory":
        print(f"Controller: {system.controller_type.value}")
        print(f"Gain margin: {system.gain_margin_db:.1f} dB")
        print(f"Phase margin: {system.phase_margin_deg:.1f} deg")
        print(f"Stable: {system.is_stable}")
    elif domain == "nonlinear_dynamics":
        print(f"Attractor type: {system.attractor_type.value}")
        print(f"Max Lyapunov: {system.max_lyapunov:.3f}")
        print(f"Dimension: {system.dimension}D")
        print(f"Chaotic: {system.is_chaotic}")
    elif domain == "quantum_gravity":
        print(f"Length scale: {system.length_scale_m:.2e} m")
        print(f"Energy scale: {system.energy_ev:.2e} eV")
        if system.mass_kg:
            print(f"Mass: {system.mass_kg:.2e} kg")

    return 0


def cmd_crossdomain(args):
    """Cross-domain theta comparison."""
    print("=" * 80)
    print("UNIVERSAL THETA: Cross-Domain Comparison")
    print("=" * 80)
    print()
    print("Theta represents the same concept across all domains:")
    print("  θ = 0: Classical (deterministic, independent, random)")
    print("  θ = 1: Quantum (coherent, correlated, entangled)")
    print()

    comparison = cross_domain_comparison()

    for domain, systems in comparison.items():
        print(f"\n{domain.upper().replace('_', ' ')}")
        print("-" * 50)
        for name, theta in sorted(systems.items(), key=lambda x: -x[1]):
            bar = "█" * int(theta * 20) + "░" * (20 - int(theta * 20))
            print(f"  {name:<30} {theta:.3f} [{bar}]")

    # Print isomorphism table
    print()
    print("=" * 70)
    print("ISOMORPHISM TABLE")
    print("=" * 70)
    print()
    print(f"{'Domain':<20} {'θ=0 (Classical)':<25} {'θ=1 (Quantum)':<25}")
    print("-" * 70)
    print(f"{'Physics':<20} {'Planets, baseballs':<25} {'Electrons, photons':<25}")
    print(f"{'Economics':<20} {'Efficient markets':<25} {'Crashes, bubbles':<25}")
    print(f"{'Information':<20} {'Pure/deterministic':<25} {'Maximally mixed':<25}")
    print(f"{'Game Theory':<20} {'Classical Nash':<25} {'Entangled strategies':<25}")
    print(f"{'Complex Systems':<20} {'Disordered phase':<25} {'Critical point':<25}")
    print(f"{'Quantum Computing':<20} {'Noisy/decoherent':<25} {'Coherent qubits':<25}")
    print(f"{'Control Theory':<20} {'Unstable system':<25} {'Perfectly stable':<25}")
    print(f"{'Nonlinear Dynamics':<20} {'Periodic/ordered':<25} {'Chaotic':<25}")
    print(f"{'Quantum Gravity':<20} {'Smooth spacetime':<25} {'Planck foam':<25}")

    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="theta-calc",
        description="Theta Calculator: Estimate where systems sit on the quantum-classical continuum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  theta-calc score --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300
  theta-calc constants --show-planck --show-bootstrap
  theta-calc explain --system electron --format markdown
  theta-calc quick --mass 0.145 --length 0.074 --temp 300
  theta-calc compare
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # score command
    score_parser = subparsers.add_parser("score", help="Score/estimate theta for a physical system")
    score_parser.add_argument("--mass", type=float, required=True, help="System mass in kg")
    score_parser.add_argument("--length", type=float, required=True, help="Length scale in m")
    score_parser.add_argument("--energy", type=float, required=True, help="Energy in J")
    score_parser.add_argument("--temp", type=float, default=300.0, help="Temperature in K")
    score_parser.add_argument("--name", type=str, default="system", help="System name")
    score_parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    score_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # constants command
    constants_parser = subparsers.add_parser("constants", help="Display fundamental constants")
    constants_parser.add_argument("--show-planck", action="store_true", help="Show Planck units")
    constants_parser.add_argument("--show-bootstrap", action="store_true", help="Show bootstrap relationships")

    # explain command
    explain_parser = subparsers.add_parser("explain", help="Generate proof explanation")
    explain_parser.add_argument(
        "--system", type=str, required=True,
        choices=list(EXAMPLE_SYSTEMS.keys()),
        help="System to explain"
    )
    explain_parser.add_argument(
        "--format", type=str, default="text",
        choices=["text", "markdown", "latex"],
        help="Output format"
    )

    # landscape command
    landscape_parser = subparsers.add_parser("landscape", help="Generate theta landscape")
    landscape_parser.add_argument("--temp", type=float, default=300.0, help="Temperature in K")
    landscape_parser.add_argument("--code", action="store_true", help="Output matplotlib code")
    landscape_parser.add_argument("--output", type=str, default=None, help="Output filename")

    # quick command
    quick_parser = subparsers.add_parser("quick", help="Quick theta estimate")
    quick_parser.add_argument("--mass", type=float, required=True, help="Mass in kg")
    quick_parser.add_argument("--length", type=float, required=True, help="Length in m")
    quick_parser.add_argument("--temp", type=float, default=300.0, help="Temperature in K")
    quick_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare theta across systems")
    compare_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # domains command
    domains_parser = subparsers.add_parser("domains", help="List all theta domains")
    domains_parser.add_argument("--verbose", "-v", action="store_true",
                                help="Show detailed theta values")

    # domain command
    domain_parser = subparsers.add_parser("domain", help="Compute theta for a domain system")
    domain_parser.add_argument("--domain", "-d", type=str, required=True,
                               choices=["economics", "information", "game_theory",
                                        "complex_systems", "quantum_computing",
                                        "quantum_biology", "cosmology",
                                        "control_theory", "nonlinear_dynamics",
                                        "quantum_gravity"],
                               help="Domain to analyze")
    domain_parser.add_argument("--system", "-s", type=str, required=True,
                               help="System name within the domain")

    # crossdomain command
    subparsers.add_parser("crossdomain",
                          help="Cross-domain theta comparison")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "score": cmd_score,
        "constants": cmd_constants,
        "explain": cmd_explain,
        "landscape": cmd_landscape,
        "quick": cmd_quick,
        "compare": cmd_compare,
        "domains": cmd_domains,
        "domain": cmd_domain,
        "crossdomain": cmd_crossdomain,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
