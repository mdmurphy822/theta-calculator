#!/usr/bin/env python3
"""
Theta Calculator CLI: Command-line interface for theta computations.

Usage:
    python -m theta_calculator prove --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300
    python -m theta_calculator constants --show-planck
    python -m theta_calculator explain --system electron
    python -m theta_calculator landscape --output theta.png
"""

import argparse
import sys
from typing import Optional

from .core.theta_state import PhysicalSystem, EXAMPLE_SYSTEMS
from .proofs.unified import UnifiedThetaProof, prove_theta
from .constants.values import FundamentalConstants
from .constants.planck_units import PlanckUnits
from .proofs.mathematical.constant_bootstrap import ConstantBootstrap
from .visualization.proof_narrator import ProofNarrator
from .visualization.theta_landscape import ThetaLandscapePlotter


def cmd_prove(args):
    """Execute proof command."""
    system = PhysicalSystem(
        name=args.name,
        mass=args.mass,
        length_scale=args.length,
        energy=args.energy,
        temperature=args.temp
    )

    proof = UnifiedThetaProof()
    result = proof.prove_theta_exists(system)

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
            print(f"  Uncertainty: exact")

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

    # Run proof
    result = prove_theta(system)

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

    print(f"Quick theta estimate:")
    print(f"  Mass: {args.mass:.2e} kg")
    print(f"  Length: {args.length:.2e} m")
    print(f"  Temperature: {args.temp:.1f} K")
    print(f"  θ ≈ {theta:.6f}")

    if theta > 0.9:
        print("  Regime: QUANTUM")
    elif theta < 0.1:
        print("  Regime: CLASSICAL")
    else:
        print("  Regime: TRANSITION")

    return 0


def cmd_compare(args):
    """Compare theta across multiple systems."""
    proof = UnifiedThetaProof()

    print("THETA COMPARISON ACROSS PHYSICAL SYSTEMS")
    print("=" * 60)
    print(f"{'System':<25} {'Theta':>10} {'Regime':<15}")
    print("-" * 60)

    for name, system in EXAMPLE_SYSTEMS.items():
        result = proof.prove_theta_exists(system)
        print(f"{name:<25} {result.theta:>10.6f} {result.regime.value:<15}")

    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="theta_calculator",
        description="Theta Calculator: Prove theta exists as the quantum-classical gradient",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  theta_calculator prove --mass 9.1e-31 --length 2.8e-15 --energy 8.2e-14 --temp 300
  theta_calculator constants --show-planck --show-bootstrap
  theta_calculator explain --system electron --format markdown
  theta_calculator quick --mass 0.145 --length 0.074 --temp 300
  theta_calculator compare
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # prove command
    prove_parser = subparsers.add_parser("prove", help="Prove theta for a physical system")
    prove_parser.add_argument("--mass", type=float, required=True, help="System mass in kg")
    prove_parser.add_argument("--length", type=float, required=True, help="Length scale in m")
    prove_parser.add_argument("--energy", type=float, required=True, help="Energy in J")
    prove_parser.add_argument("--temp", type=float, default=300.0, help="Temperature in K")
    prove_parser.add_argument("--name", type=str, default="system", help="System name")
    prove_parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")

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

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare theta across systems")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "prove": cmd_prove,
        "constants": cmd_constants,
        "explain": cmd_explain,
        "landscape": cmd_landscape,
        "quick": cmd_quick,
        "compare": cmd_compare,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
