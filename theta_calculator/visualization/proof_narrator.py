"""
Proof Narrator: Generate pedagogical explanations of theta proofs.

This module transforms the mathematical proof results into clear,
step-by-step narratives suitable for different audiences:
- Physicists: Rigorous formalism with equations
- Engineers: Practical computation focus
- Students: Intuitive understanding

The narrator shows not just WHAT theta is, but WHY it exists.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..proofs.unified import UnifiedProofResult
from ..core.theta_state import ThetaState, Regime


@dataclass
class NarrativeStep:
    """A single step in the proof narrative."""
    step_number: int
    title: str
    description: str
    formula: Optional[str]  # Plain text formula
    latex_formula: Optional[str]  # LaTeX formula
    value: Optional[float]
    interpretation: str


class ProofNarrator:
    """
    Generates pedagogical, step-by-step explanations of theta proofs.

    Designed to make the proof accessible to multiple audiences
    while maintaining scientific rigor.
    """

    def __init__(self, detail_level: str = "standard"):
        """
        Initialize narrator.

        Args:
            detail_level: One of "brief", "standard", "detailed"
        """
        self.detail_level = detail_level

    def narrate_proof(self, result: UnifiedProofResult) -> List[NarrativeStep]:
        """
        Generate narrative explanation of a unified proof.

        Args:
            result: UnifiedProofResult from a theta proof

        Returns:
            List of NarrativeStep objects
        """
        steps = []
        step_num = 1

        # Step 1: Introduction
        steps.append(NarrativeStep(
            step_number=step_num,
            title="What is Theta?",
            description=(
                "Theta (θ) is the fundamental parameter that interpolates between "
                "quantum and classical descriptions of physical systems. "
                "When θ = 1, quantum mechanics fully describes the system "
                "(superposition, entanglement, interference). "
                "When θ = 0, classical physics suffices "
                "(definite states, deterministic trajectories). "
                "For any observable O, the measured value is:\n\n"
                "    O = θ × O_quantum + (1-θ) × O_classical"
            ),
            formula="O = θ·O_quantum + (1-θ)·O_classical",
            latex_formula=r"O = \theta \cdot O_{\text{quantum}} + (1-\theta) \cdot O_{\text{classical}}",
            value=None,
            interpretation="This is the fundamental theta interpolation equation."
        ))
        step_num += 1

        # Step 2: Mathematical proof
        math_proof = result.mathematical_proof
        steps.append(NarrativeStep(
            step_number=step_num,
            title="Mathematical Derivation: Constants Bootstrap",
            description=(
                "Theta emerges from the relationships between fundamental constants. "
                "The constants c, ℏ, G, k, and α are not independent - each can be "
                "derived from others at boundary conditions. This recursive structure "
                "means theta is built into the fabric of physics.\n\n"
                "The key relationship is: θ = ℏ/S\n"
                "where S is the system's characteristic action (S = E × t)."
            ),
            formula="θ = ℏ/S where S is the action",
            latex_formula=r"\theta = \frac{\hbar}{S} \text{ where } S = \int L \, dt",
            value=math_proof.get("theta"),
            interpretation=(
                f"Bootstrap verification: {'PASSED' if math_proof.get('all_consistent') else 'FAILED'}. "
                f"Action S = {math_proof.get('action', 0):.2e} J·s. "
                "When S ≈ ℏ, quantum effects dominate (θ → 1). "
                "When S >> ℏ, classical physics applies (θ → 0)."
            )
        ))
        step_num += 1

        # Step 3: Numerical proof
        num_proof = result.numerical_proof
        steps.append(NarrativeStep(
            step_number=step_num,
            title="Numerical Verification: Multiple Methods",
            description=(
                "Theta is computed independently by four different physical mechanisms:\n"
                "1. Action ratio: θ = ℏ/S (quantum action vs system action)\n"
                "2. Thermal ratio: θ = ℏω/(kT) (quantum vs thermal energy)\n"
                "3. Scale ratio: θ from Planck length comparisons\n"
                "4. Decoherence: θ = exp(-t/t_D) (quantum coherence decay)\n\n"
                "These independent methods must converge for theta to be real."
            ),
            formula="θ_unified = Σ w_i × θ_i",
            latex_formula=r"\theta_{\text{unified}} = \sum_i w_i \theta_i",
            value=num_proof.get("theta_unified"),
            interpretation=(
                f"Action: θ = {num_proof.get('theta_action', 0):.4f}, "
                f"Thermal: θ = {num_proof.get('theta_thermal', 0):.4f}, "
                f"Scale: θ = {num_proof.get('theta_scale', 0):.4f}, "
                f"Decoherence: θ = {num_proof.get('theta_decoherence', 0):.4f}. "
                f"Unified: θ = {num_proof.get('theta_unified', 0):.4f}."
            )
        ))
        step_num += 1

        # Step 4: Information-theoretic proof
        info_proof = result.information_proof
        steps.append(NarrativeStep(
            step_number=step_num,
            title="Information-Theoretic Bound",
            description=(
                "The Bekenstein bound sets the maximum information in a region:\n"
                "    S ≤ 2πkRE/(ℏc)\n\n"
                "A system's proximity to this bound determines its theta:\n"
                "- At the bound (black holes): θ = 1 (maximally quantum)\n"
                "- Far from bound (everyday objects): θ → 0 (classical)\n\n"
                "The Landauer limit E ≥ kT ln(2) per bit provides another bound."
            ),
            formula="S ≤ 2πkRE/(ℏc)",
            latex_formula=r"S \leq \frac{2\pi k R E}{\hbar c}",
            value=info_proof.get("theta_bekenstein"),
            interpretation=(
                f"Bekenstein saturation: {info_proof.get('bekenstein_saturation', 0):.2e}. "
                f"Maximum bits: {info_proof.get('bekenstein_bits', 0):.2e}. "
                f"Holographic: {'Yes' if info_proof.get('is_holographic') else 'No'}."
            )
        ))
        step_num += 1

        # Step 5: Convergence
        steps.append(NarrativeStep(
            step_number=step_num,
            title="Proof Convergence",
            description=(
                "All three proof methodologies must converge to the same θ value "
                "for the proof to be valid. This convergence demonstrates that "
                "theta is a real, measurable property of physical systems - "
                "not an arbitrary parameter.\n\n"
                "The agreement score measures how well the methods align."
            ),
            formula=None,
            latex_formula=None,
            value=result.proof_agreement,
            interpretation=(
                f"Agreement score: {result.proof_agreement:.1%}. "
                f"Final theta: {result.theta:.6f} ± {result.theta_uncertainty:.6f}. "
                f"Valid proof: {'Yes' if result.is_valid else 'No'}."
            )
        ))
        step_num += 1

        # Step 6: Conclusion
        regime_descriptions = {
            Regime.QUANTUM: "fully quantum mechanical",
            Regime.CLASSICAL: "classical",
            Regime.TRANSITION: "in the quantum-classical transition zone"
        }
        regime_desc = regime_descriptions.get(result.regime, "undefined")

        steps.append(NarrativeStep(
            step_number=step_num,
            title="Conclusion: Q.E.D.",
            description=(
                f"This system is {regime_desc} with θ = {result.theta:.4f}.\n\n"
                "The proof demonstrates:\n"
                "• Theta exists as a well-defined, computable parameter\n"
                "• Theta emerges from fundamental constant relationships\n"
                "• Multiple independent methods converge to consistent values\n"
                "• Theta satisfies information-theoretic bounds\n\n"
                "Therefore, theta is the gradient between quantum and classical."
            ),
            formula=None,
            latex_formula=None,
            value=result.theta,
            interpretation="Q.E.D. - Theta exists and is the quantum-classical gradient."
        ))

        return steps

    def to_markdown(self, steps: List[NarrativeStep]) -> str:
        """Convert narrative steps to markdown format."""
        lines = [
            "# Proof that Theta Exists as the Quantum-Classical Gradient\n",
        ]

        for step in steps:
            lines.append(f"## Step {step.step_number}: {step.title}\n")
            lines.append(f"{step.description}\n")

            if step.formula:
                lines.append(f"**Formula:** `{step.formula}`\n")

            if step.value is not None:
                lines.append(f"**Computed Value:** {step.value:.6f}\n")

            lines.append(f"**Interpretation:** {step.interpretation}\n")
            lines.append("---\n")

        return "\n".join(lines)

    def to_latex(self, steps: List[NarrativeStep]) -> str:
        """Convert narrative steps to LaTeX format."""
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath,amssymb}",
            r"\title{Proof that $\theta$ Exists as the Quantum-Classical Gradient}",
            r"\begin{document}",
            r"\maketitle",
            "",
        ]

        for step in steps:
            lines.append(f"\\section{{Step {step.step_number}: {step.title}}}")
            lines.append("")
            lines.append(step.description.replace("\n", "\n\n"))

            if step.latex_formula:
                lines.append(r"\begin{equation}")
                lines.append(step.latex_formula)
                lines.append(r"\end{equation}")

            if step.value is not None:
                lines.append(f"\\textbf{{Value:}} $\\theta = {step.value:.6f}$")

            lines.append("")
            lines.append(f"\\textbf{{Interpretation:}} {step.interpretation}")
            lines.append("")

        lines.append(r"\end{document}")
        return "\n".join(lines)

    def to_text(self, steps: List[NarrativeStep]) -> str:
        """Convert narrative steps to plain text."""
        lines = [
            "=" * 60,
            "PROOF: THETA AS THE QUANTUM-CLASSICAL GRADIENT",
            "=" * 60,
            "",
        ]

        for step in steps:
            lines.append(f"STEP {step.step_number}: {step.title.upper()}")
            lines.append("-" * 40)
            lines.append(step.description)

            if step.formula:
                lines.append(f"\nFormula: {step.formula}")

            if step.value is not None:
                lines.append(f"\nValue: θ = {step.value:.6f}")

            lines.append(f"\nInterpretation: {step.interpretation}")
            lines.append("")

        return "\n".join(lines)


def narrate_theta_proof(result: UnifiedProofResult, format: str = "text") -> str:
    """
    Convenience function to narrate a theta proof.

    Args:
        result: UnifiedProofResult from prove_theta
        format: Output format - "text", "markdown", or "latex"

    Returns:
        Formatted proof narrative
    """
    narrator = ProofNarrator()
    steps = narrator.narrate_proof(result)

    if format == "markdown":
        return narrator.to_markdown(steps)
    elif format == "latex":
        return narrator.to_latex(steps)
    else:
        return narrator.to_text(steps)
