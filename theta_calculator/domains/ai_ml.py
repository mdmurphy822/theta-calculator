r"""
AI/Machine Learning Domain: Learning Dynamics as Theta

This module implements theta as the learning quality parameter
using frameworks from statistical learning theory and deep learning.

Key Insight: ML systems exhibit phase transitions between:
- theta ~ 0: Underfitting/overfitting (poor generalization)
- theta ~ 1: Optimal learning (robust generalization)

Theta Maps To:
1. Generalization Gap: 1 - |train_loss - test_loss| / train_loss
2. Convergence Quality: stability of training trajectory
3. Attention Focus: entropy of attention distributions
4. Gradient Health: ratio of useful to vanishing/exploding gradients
5. Robustness: accuracy under perturbation

Learning Regimes:
- UNDERFITTING (theta < 0.3): High bias, model too simple
- LEARNING (0.3 <= theta < 0.6): Active improvement, healthy gradients
- OPTIMAL (0.6 <= theta < 0.8): Good generalization, stable training
- OVERFITTING (theta >= 0.8 AND gap high): Low bias, high variance

Physical Analogy:
Training dynamics follow a phase transition similar to crystallization.
Below a critical "learning temperature" (learning rate), the loss
landscape freezes into local minima. At the critical point, the
model transitions between memorization and generalization.

The attention mechanism in transformers creates long-range correlations
similar to critical systems, enabling information flow across
the entire sequence.

References (see BIBLIOGRAPHY.bib):
    \cite{Vaswani2017} - Attention Is All You Need
    \cite{Zhang2017} - Understanding deep learning requires rethinking generalization
    \cite{Hochreiter1997} - Long Short-Term Memory
    \cite{Goodfellow2016} - Deep Learning
    \cite{Srivastava2014} - Dropout: A simple way to prevent overfitting
    \cite{Ioffe2015} - Batch Normalization
    \cite{He2016} - Deep Residual Learning
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class LearningRegime(Enum):
    """Learning regime states based on theta and generalization."""
    UNDERFITTING = "underfitting"    # theta < 0.3
    LEARNING = "learning"            # 0.3 <= theta < 0.6
    OPTIMAL = "optimal"              # 0.6 <= theta < 0.8
    OVERFITTING = "overfitting"      # theta >= 0.8 with high gap


class ModelArchitecture(Enum):
    """Common neural network architectures."""
    LINEAR = "linear"
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GNN = "gnn"
    VAE = "vae"
    GAN = "gan"


class AttentionType(Enum):
    """Types of attention mechanisms."""
    NONE = "none"
    ADDITIVE = "additive"           # Bahdanau attention
    MULTIPLICATIVE = "multiplicative"  # Luong attention
    SELF_ATTENTION = "self_attention"  # Transformer
    MULTI_HEAD = "multi_head"       # Multi-head self-attention
    SPARSE = "sparse"               # Sparse attention (Longformer, BigBird)


@dataclass
class MLSystem:
    """
    A machine learning system for theta analysis.

    Attributes:
        name: System identifier
        architecture: Model architecture type
        n_parameters: Number of trainable parameters
        train_loss: Training set loss
        val_loss: Validation set loss
        train_accuracy: Training accuracy [0, 1]
        val_accuracy: Validation accuracy [0, 1]
        gradient_norm: Average gradient L2 norm
        attention_entropy: Entropy of attention weights (if applicable)
        epochs_trained: Number of training epochs
        learning_rate: Current learning rate
        regularization: L2 regularization strength
        dropout_rate: Dropout probability [0, 1]
    """
    name: str
    architecture: ModelArchitecture
    n_parameters: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    gradient_norm: float
    attention_entropy: Optional[float] = None  # Only for attention models
    epochs_trained: int = 0
    learning_rate: float = 0.001
    regularization: float = 0.0
    dropout_rate: float = 0.0

    @property
    def generalization_gap(self) -> float:
        """Gap between training and validation performance."""
        return abs(self.train_accuracy - self.val_accuracy)

    @property
    def loss_gap(self) -> float:
        """Gap between training and validation loss."""
        if self.train_loss == 0:
            return 0.0
        return abs(self.val_loss - self.train_loss) / self.train_loss


# =============================================================================
# GENERALIZATION THETA
# =============================================================================

def compute_generalization_theta(
    train_acc: float,
    val_acc: float,
    tolerance: float = 0.1
) -> float:
    r"""
    Compute theta from generalization gap.

    Small gap + high accuracy = high theta (good generalization).
    Large gap = low theta (overfitting or underfitting).

    Args:
        train_acc: Training accuracy [0, 1]
        val_acc: Validation accuracy [0, 1]
        tolerance: Acceptable gap threshold

    Returns:
        theta in [0, 1]

    Reference: \cite{Zhang2017} - Rethinking generalization
    """
    gap = abs(train_acc - val_acc)

    # Base theta from validation accuracy
    base_theta = val_acc

    # Penalty for large generalization gap
    gap_penalty = gap / tolerance if gap > tolerance else gap / (2 * tolerance)

    theta = base_theta * (1 - gap_penalty)
    return np.clip(theta, 0.0, 1.0)


def compute_loss_ratio_theta(
    train_loss: float,
    val_loss: float
) -> float:
    r"""
    Compute theta from loss ratio.

    train_loss ~ val_loss -> high theta
    val_loss >> train_loss -> low theta (overfitting)
    train_loss >> val_loss -> suspicious (data leakage)

    Args:
        train_loss: Training loss
        val_loss: Validation loss

    Returns:
        theta in [0, 1]
    """
    if train_loss <= 0:
        return 0.0

    ratio = val_loss / train_loss

    # Optimal: ratio close to 1
    if ratio < 1:
        # Val loss lower than train (unusual but possible)
        theta = ratio
    else:
        # Val loss higher than train (normal)
        theta = 1 / ratio

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# CONVERGENCE THETA
# =============================================================================

def compute_convergence_theta(
    loss_history: List[float],
    window: int = 10
) -> float:
    r"""
    Compute theta from training convergence quality.

    Stable, decreasing loss = high theta.
    Oscillating or diverging loss = low theta.

    Args:
        loss_history: List of loss values over training
        window: Window size for variance calculation

    Returns:
        theta in [0, 1]

    Reference: \cite{Goodfellow2016} - Deep Learning optimization
    """
    if len(loss_history) < window:
        return 0.5  # Not enough data

    recent = loss_history[-window:]
    earlier = loss_history[-2*window:-window] if len(loss_history) >= 2*window else loss_history[:window]

    # Check if loss is decreasing
    if np.mean(recent) >= np.mean(earlier):
        decreasing = 0.3  # Not decreasing
    else:
        decrease_rate = (np.mean(earlier) - np.mean(recent)) / np.mean(earlier)
        decreasing = min(1.0, decrease_rate * 10)

    # Check stability (low variance)
    variance = np.var(recent) / (np.mean(recent) ** 2 + 1e-8)
    stability = 1 / (1 + variance * 100)

    theta = 0.5 * decreasing + 0.5 * stability
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# GRADIENT HEALTH THETA
# =============================================================================

def compute_gradient_theta(
    gradient_norm: float,
    optimal_range: Tuple[float, float] = (0.01, 10.0)
) -> float:
    r"""
    Compute theta from gradient health.

    Gradients in optimal range = high theta.
    Vanishing (< 0.01) or exploding (> 100) = low theta.

    Args:
        gradient_norm: L2 norm of gradients
        optimal_range: (min, max) for healthy gradients

    Returns:
        theta in [0, 1]

    Reference: \cite{Hochreiter1997} - LSTM (vanishing gradients)
    """
    min_norm, max_norm = optimal_range

    if gradient_norm < min_norm:
        # Vanishing gradients
        theta = gradient_norm / min_norm
    elif gradient_norm > max_norm:
        # Exploding gradients
        theta = max_norm / gradient_norm
    else:
        # Healthy range
        # Peak at geometric mean
        optimal = np.sqrt(min_norm * max_norm)
        distance = abs(np.log(gradient_norm) - np.log(optimal))
        range_log = np.log(max_norm) - np.log(min_norm)
        theta = 1 - distance / (range_log / 2)

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ATTENTION THETA
# =============================================================================

def compute_attention_entropy(
    attention_weights: np.ndarray
) -> float:
    r"""
    Compute entropy of attention distribution.

    Low entropy = focused attention (specific tokens matter)
    High entropy = diffuse attention (uniform distribution)

    Args:
        attention_weights: Attention probabilities [seq_len] or [heads, seq_len]

    Returns:
        Entropy in nats

    Reference: \cite{Vaswani2017} - Attention mechanism
    """
    # Flatten if multi-head
    weights = attention_weights.flatten()
    weights = weights / (weights.sum() + 1e-8)

    # Shannon entropy
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    return entropy


def compute_attention_theta(
    attention_entropy: float,
    max_entropy: float
) -> float:
    r"""
    Compute theta from attention entropy.

    Moderate entropy = high theta (selective but not too focused).
    Very low or very high entropy = lower theta.

    Args:
        attention_entropy: Computed attention entropy
        max_entropy: Maximum possible entropy (log(seq_len))

    Returns:
        theta in [0, 1]

    Reference: \cite{Vaswani2017} - Transformer attention
    """
    if max_entropy <= 0:
        return 0.5

    normalized = attention_entropy / max_entropy

    # Optimal around 0.3-0.7 (neither too focused nor too diffuse)
    # Gaussian penalty centered at 0.5
    optimal = 0.5
    width = 0.3
    theta = np.exp(-((normalized - optimal) ** 2) / (2 * width ** 2))

    return np.clip(theta, 0.0, 1.0)


def attention_sparsity(
    attention_weights: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Compute sparsity of attention (fraction of weights above threshold).

    Args:
        attention_weights: Attention probabilities
        threshold: Cutoff for "significant" attention

    Returns:
        Sparsity ratio [0, 1] (1 = all sparse, 0 = all dense)
    """
    weights = attention_weights.flatten()
    n_significant = np.sum(weights > threshold)
    sparsity = 1 - (n_significant / len(weights))
    return sparsity


# =============================================================================
# REGULARIZATION THETA
# =============================================================================

def compute_regularization_theta(
    dropout_rate: float,
    l2_strength: float,
    batch_norm: bool = False
) -> float:
    r"""
    Compute theta from regularization strength.

    Appropriate regularization = high theta.
    No regularization (overfit risk) = lower theta.
    Too much regularization (underfit risk) = lower theta.

    Args:
        dropout_rate: Dropout probability [0, 1]
        l2_strength: L2 weight decay coefficient
        batch_norm: Whether batch normalization is used

    Returns:
        theta in [0, 1]

    References:
        \cite{Srivastava2014} - Dropout
        \cite{Ioffe2015} - Batch Normalization
    """
    # Optimal dropout around 0.3-0.5
    dropout_optimal = 0.4
    dropout_theta = 1 - abs(dropout_rate - dropout_optimal) / dropout_optimal

    # Optimal L2 around 1e-4 to 1e-2
    l2_optimal = 1e-3
    if l2_strength > 0:
        l2_ratio = np.log10(l2_strength) - np.log10(l2_optimal)
        l2_theta = 1 / (1 + abs(l2_ratio))
    else:
        l2_theta = 0.3  # No regularization

    # Batch norm bonus
    bn_bonus = 0.1 if batch_norm else 0.0

    theta = 0.4 * dropout_theta + 0.4 * l2_theta + 0.2 + bn_bonus
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# MODEL CAPACITY THETA
# =============================================================================

def compute_capacity_theta(
    n_parameters: int,
    n_samples: int,
    architecture: ModelArchitecture
) -> float:
    r"""
    Compute theta from model capacity vs data ratio.

    Rule of thumb: need ~10-100 samples per parameter for simple models,
    but transformers can generalize with fewer due to pretraining.

    Args:
        n_parameters: Number of model parameters
        n_samples: Number of training samples
        architecture: Model architecture type

    Returns:
        theta in [0, 1]
    """
    # Samples per parameter ratio
    ratio = n_samples / (n_parameters + 1)

    # Architecture-specific optimal ratios
    optimal_ratios = {
        ModelArchitecture.LINEAR: 10.0,
        ModelArchitecture.MLP: 5.0,
        ModelArchitecture.CNN: 1.0,
        ModelArchitecture.RNN: 2.0,
        ModelArchitecture.LSTM: 2.0,
        ModelArchitecture.TRANSFORMER: 0.1,  # Can work with fewer due to attention
        ModelArchitecture.GNN: 1.0,
        ModelArchitecture.VAE: 2.0,
        ModelArchitecture.GAN: 2.0,
    }

    optimal = optimal_ratios.get(architecture, 1.0)

    if ratio < optimal:
        # Potentially overparameterized
        theta = ratio / optimal
    else:
        # Good or underparameterized
        theta = 1.0 - 0.5 * min(1.0, (ratio - optimal) / (10 * optimal))

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ROBUSTNESS THETA
# =============================================================================

def compute_robustness_theta(
    clean_accuracy: float,
    adversarial_accuracy: float,
    noise_accuracy: float
) -> float:
    r"""
    Compute theta from model robustness.

    Robust to perturbations = high theta.
    Brittle to small changes = low theta.

    Args:
        clean_accuracy: Accuracy on clean test set
        adversarial_accuracy: Accuracy under adversarial attack
        noise_accuracy: Accuracy with input noise

    Returns:
        theta in [0, 1]

    Reference: \cite{Goodfellow2014} - Adversarial Examples
    """
    # Base from clean accuracy
    base = clean_accuracy

    # Robustness ratio
    adv_ratio = adversarial_accuracy / (clean_accuracy + 1e-8)
    noise_ratio = noise_accuracy / (clean_accuracy + 1e-8)

    robustness = 0.5 * adv_ratio + 0.5 * noise_ratio

    theta = base * robustness
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# TRANSFER LEARNING THETA
# =============================================================================

def compute_transfer_theta(
    source_accuracy: float,
    target_accuracy: float,
    fine_tune_epochs: int
) -> float:
    r"""
    Compute theta from transfer learning effectiveness.

    Good transfer = high theta (knowledge transfers well).
    Poor transfer = low theta (domains too different).

    Args:
        source_accuracy: Accuracy on source domain
        target_accuracy: Accuracy on target domain after transfer
        fine_tune_epochs: Number of fine-tuning epochs

    Returns:
        theta in [0, 1]
    """
    # Transfer ratio
    if source_accuracy > 0:
        transfer_ratio = target_accuracy / source_accuracy
    else:
        transfer_ratio = 0.0

    # Efficiency bonus (fewer epochs = better transfer)
    efficiency = 1 / (1 + fine_tune_epochs / 10)

    theta = 0.7 * transfer_ratio + 0.3 * efficiency
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED ML THETA
# =============================================================================

def compute_ml_theta(system: MLSystem) -> float:
    """
    Compute unified theta for ML system.

    Combines:
    - Generalization quality (30%)
    - Loss ratio (20%)
    - Gradient health (20%)
    - Attention focus if applicable (15%)
    - Regularization (15%)

    Args:
        system: MLSystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Generalization
    theta_gen = compute_generalization_theta(
        system.train_accuracy,
        system.val_accuracy
    )

    # Loss ratio
    theta_loss = compute_loss_ratio_theta(
        system.train_loss,
        system.val_loss
    )

    # Gradient health
    theta_grad = compute_gradient_theta(system.gradient_norm)

    # Attention (if applicable)
    if system.attention_entropy is not None and system.architecture in [
        ModelArchitecture.TRANSFORMER,
        ModelArchitecture.LSTM,
        ModelArchitecture.RNN
    ]:
        # Estimate max entropy for typical sequence length
        max_entropy = np.log(512)  # Typical seq length
        theta_attn = compute_attention_theta(system.attention_entropy, max_entropy)
        attention_weight = 0.15
    else:
        theta_attn = 0.5
        attention_weight = 0.0

    # Regularization
    theta_reg = compute_regularization_theta(
        system.dropout_rate,
        system.regularization,
        batch_norm=True
    )

    # Weighted combination
    gen_weight = 0.30
    loss_weight = 0.20
    grad_weight = 0.20
    reg_weight = 0.15 + (0.15 - attention_weight)

    theta = (
        gen_weight * theta_gen +
        loss_weight * theta_loss +
        grad_weight * theta_grad +
        attention_weight * theta_attn +
        reg_weight * theta_reg
    )

    return np.clip(theta, 0.0, 1.0)


def classify_learning_regime(
    theta: float,
    generalization_gap: float = 0.0
) -> LearningRegime:
    """
    Classify learning regime from theta and generalization gap.

    Note: High theta with high gap indicates overfitting.
    """
    if theta < 0.3:
        return LearningRegime.UNDERFITTING
    elif theta < 0.6:
        return LearningRegime.LEARNING
    elif theta < 0.8:
        if generalization_gap > 0.15:
            return LearningRegime.OVERFITTING
        return LearningRegime.OPTIMAL
    else:
        if generalization_gap > 0.1:
            return LearningRegime.OVERFITTING
        return LearningRegime.OPTIMAL


# =============================================================================
# SCALING LAWS
# =============================================================================

def scaling_law_theta(
    n_parameters: float,
    n_tokens: float,
    compute_budget: float
) -> float:
    r"""
    Compute theta based on neural scaling laws.

    Optimal allocation follows Chinchilla scaling:
    N_opt ~ C^0.5, D_opt ~ C^0.5

    Args:
        n_parameters: Number of parameters (N)
        n_tokens: Number of training tokens (D)
        compute_budget: Compute in FLOPs (C ~ 6ND)

    Returns:
        theta in [0, 1] indicating scaling efficiency

    Reference: \cite{Hoffmann2022} - Training Compute-Optimal LLMs
    """
    # Chinchilla optimal: N = D (roughly)
    ratio = n_parameters / (n_tokens + 1)

    # Optimal ratio is around 1 for compute-optimal training
    optimal_ratio = 1.0

    # Distance from optimal
    if ratio < optimal_ratio:
        # Undertrained (need more tokens)
        theta = ratio / optimal_ratio
    else:
        # Overtrained (need more parameters or less data)
        theta = optimal_ratio / ratio

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

ML_SYSTEMS: Dict[str, MLSystem] = {
    "linear_regression": MLSystem(
        name="Simple Linear Regression",
        architecture=ModelArchitecture.LINEAR,
        n_parameters=100,
        train_loss=0.5,
        val_loss=0.52,
        train_accuracy=0.65,
        val_accuracy=0.63,
        gradient_norm=0.1,
        attention_entropy=None,
        epochs_trained=100,
        learning_rate=0.01,
        regularization=0.0,
        dropout_rate=0.0,
    ),
    "overfit_mlp": MLSystem(
        name="Overfitting MLP (no regularization)",
        architecture=ModelArchitecture.MLP,
        n_parameters=100000,
        train_loss=0.01,
        val_loss=0.8,
        train_accuracy=0.99,
        val_accuracy=0.65,
        gradient_norm=0.5,
        attention_entropy=None,
        epochs_trained=500,
        learning_rate=0.001,
        regularization=0.0,
        dropout_rate=0.0,
    ),
    "regularized_mlp": MLSystem(
        name="Regularized MLP",
        architecture=ModelArchitecture.MLP,
        n_parameters=100000,
        train_loss=0.15,
        val_loss=0.18,
        train_accuracy=0.92,
        val_accuracy=0.88,
        gradient_norm=0.3,
        attention_entropy=None,
        epochs_trained=200,
        learning_rate=0.001,
        regularization=0.001,
        dropout_rate=0.3,
    ),
    "resnet50": MLSystem(
        name="ResNet-50 (ImageNet)",
        architecture=ModelArchitecture.CNN,
        n_parameters=25_000_000,
        train_loss=0.8,
        val_loss=0.9,
        train_accuracy=0.85,
        val_accuracy=0.76,
        gradient_norm=1.0,
        attention_entropy=None,
        epochs_trained=90,
        learning_rate=0.1,
        regularization=0.0001,
        dropout_rate=0.0,
    ),
    "bert_base": MLSystem(
        name="BERT Base (Fine-tuned)",
        architecture=ModelArchitecture.TRANSFORMER,
        n_parameters=110_000_000,
        train_loss=0.2,
        val_loss=0.25,
        train_accuracy=0.95,
        val_accuracy=0.91,
        gradient_norm=0.8,
        attention_entropy=3.5,
        epochs_trained=3,
        learning_rate=2e-5,
        regularization=0.01,
        dropout_rate=0.1,
    ),
    "gpt2": MLSystem(
        name="GPT-2 (Language Modeling)",
        architecture=ModelArchitecture.TRANSFORMER,
        n_parameters=1_500_000_000,
        train_loss=2.5,
        val_loss=2.7,
        train_accuracy=0.45,  # Perplexity-based
        val_accuracy=0.42,
        gradient_norm=1.2,
        attention_entropy=4.0,
        epochs_trained=1,
        learning_rate=1e-4,
        regularization=0.0,
        dropout_rate=0.1,
    ),
    "gpt4_scale": MLSystem(
        name="GPT-4 Scale Model",
        architecture=ModelArchitecture.TRANSFORMER,
        n_parameters=1_000_000_000_000,  # Estimated
        train_loss=1.8,
        val_loss=1.85,
        train_accuracy=0.65,
        val_accuracy=0.63,
        gradient_norm=0.5,
        attention_entropy=4.5,
        epochs_trained=1,
        learning_rate=1e-5,
        regularization=0.0,
        dropout_rate=0.0,
    ),
    "lstm_sentiment": MLSystem(
        name="LSTM Sentiment Analysis",
        architecture=ModelArchitecture.LSTM,
        n_parameters=5_000_000,
        train_loss=0.3,
        val_loss=0.35,
        train_accuracy=0.88,
        val_accuracy=0.85,
        gradient_norm=0.2,
        attention_entropy=2.0,
        epochs_trained=20,
        learning_rate=0.001,
        regularization=0.0001,
        dropout_rate=0.5,
    ),
    "vae_mnist": MLSystem(
        name="VAE on MNIST",
        architecture=ModelArchitecture.VAE,
        n_parameters=500_000,
        train_loss=100.0,  # ELBO
        val_loss=105.0,
        train_accuracy=0.98,  # Reconstruction
        val_accuracy=0.96,
        gradient_norm=0.4,
        attention_entropy=None,
        epochs_trained=50,
        learning_rate=0.001,
        regularization=0.0,
        dropout_rate=0.0,
    ),
    "diverging_training": MLSystem(
        name="Diverging Training (bad LR)",
        architecture=ModelArchitecture.MLP,
        n_parameters=10000,
        train_loss=float('inf'),
        val_loss=float('inf'),
        train_accuracy=0.1,
        val_accuracy=0.1,
        gradient_norm=1000.0,  # Exploding
        attention_entropy=None,
        epochs_trained=10,
        learning_rate=10.0,  # Way too high
        regularization=0.0,
        dropout_rate=0.0,
    ),
}


def ml_theta_summary():
    """Print theta analysis for example ML systems."""
    print("=" * 90)
    print("AI/ML THETA ANALYSIS (Learning Dynamics)")
    print("=" * 90)
    print()
    print(f"{'System':<35} {'Arch':<12} {'Train':>8} {'Val':>8} "
          f"{'Gap':>6} {'θ':>8} {'Regime':<12}")
    print("-" * 90)

    for name, system in ML_SYSTEMS.items():
        theta = compute_ml_theta(system)
        gap = system.generalization_gap
        regime = classify_learning_regime(theta, gap)
        arch = system.architecture.value[:10]

        train_str = f"{system.train_accuracy:.2f}"
        val_str = f"{system.val_accuracy:.2f}"

        print(f"{system.name:<35} "
              f"{arch:<12} "
              f"{train_str:>8} "
              f"{val_str:>8} "
              f"{gap:>6.2f} "
              f"{theta:>8.3f} "
              f"{regime.value:<12}")

    print()
    print("Key: θ combines generalization, loss ratio, gradient health, attention")
    print("     Optimal regime has θ ~ 0.6-0.8 with low generalization gap")


if __name__ == "__main__":
    ml_theta_summary()
