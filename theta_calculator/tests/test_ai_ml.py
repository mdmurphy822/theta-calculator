"""
Tests for AI/Machine Learning Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Learning regime classification
- Component theta calculations
- Edge cases (diverging training, extreme values)
"""

import pytest
import numpy as np

from theta_calculator.domains.ai_ml import (
    ModelArchitecture,
    LearningRegime,
    compute_generalization_theta,
    compute_loss_ratio_theta,
    compute_convergence_theta,
    compute_gradient_theta,
    compute_attention_entropy,
    compute_attention_theta,
    compute_regularization_theta,
    compute_capacity_theta,
    compute_robustness_theta,
    compute_transfer_theta,
    compute_ml_theta,
    classify_learning_regime,
    scaling_law_theta,
    attention_sparsity,
    ML_SYSTEMS,
)


class TestMLSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """ML_SYSTEMS dict should exist."""
        assert ML_SYSTEMS is not None
        assert isinstance(ML_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(ML_SYSTEMS) >= 5

    def test_system_names(self):
        """Key systems should be defined."""
        expected = [
            "linear_regression",
            "overfit_mlp",
            "bert_base",
            "gpt2",
        ]
        for name in expected:
            assert name in ML_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in ML_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "architecture")
            assert hasattr(system, "n_parameters")
            assert hasattr(system, "train_loss")
            assert hasattr(system, "val_loss")
            assert hasattr(system, "train_accuracy")
            assert hasattr(system, "val_accuracy")
            assert hasattr(system, "gradient_norm")


class TestGeneralizationTheta:
    """Test generalization theta calculation."""

    def test_perfect_generalization(self):
        """Same train/val accuracy -> high theta."""
        theta = compute_generalization_theta(0.9, 0.9)
        assert theta > 0.8

    def test_overfitting(self):
        """Large gap -> lower theta."""
        theta = compute_generalization_theta(0.99, 0.6)
        assert theta < 0.5

    def test_underfitting(self):
        """Low accuracy (both) -> low theta."""
        theta = compute_generalization_theta(0.3, 0.3)
        assert theta < 0.4

    def test_zero_accuracy(self):
        """Zero accuracy -> theta = 0."""
        theta = compute_generalization_theta(0.0, 0.0)
        assert theta == 0.0


class TestLossRatioTheta:
    """Test loss ratio theta calculation."""

    def test_equal_loss(self):
        """Same train/val loss -> theta = 1.0."""
        theta = compute_loss_ratio_theta(0.5, 0.5)
        assert theta == pytest.approx(1.0)

    def test_overfitting_loss(self):
        """Val loss >> train loss -> low theta."""
        theta = compute_loss_ratio_theta(0.1, 1.0)
        assert theta < 0.2

    def test_zero_train_loss(self):
        """Zero train loss -> theta = 0."""
        theta = compute_loss_ratio_theta(0.0, 0.5)
        assert theta == 0.0


class TestConvergenceTheta:
    """Test convergence theta calculation."""

    def test_decreasing_loss(self):
        """Decreasing loss -> high theta."""
        history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15,
                   0.12, 0.11, 0.105, 0.102, 0.101, 0.1005, 0.1002, 0.1001, 0.1, 0.1]
        theta = compute_convergence_theta(history)
        assert theta > 0.5

    def test_oscillating_loss(self):
        """Oscillating loss -> lower theta."""
        history = [0.5, 0.8, 0.4, 0.9, 0.3, 1.0, 0.5, 0.7, 0.4, 0.8,
                   0.5, 0.9, 0.4, 0.8, 0.5, 0.7, 0.6, 0.8, 0.5, 0.7]
        theta = compute_convergence_theta(history)
        assert theta < 0.7

    def test_short_history(self):
        """Short history -> default theta."""
        history = [1.0, 0.9, 0.8]
        theta = compute_convergence_theta(history)
        assert theta == 0.5


class TestGradientTheta:
    """Test gradient health theta calculation."""

    def test_healthy_gradient(self):
        """Gradient in optimal range -> reasonable theta."""
        theta = compute_gradient_theta(1.0)
        assert theta > 0.6  # 1.0 is at optimal_norm boundary

    def test_vanishing_gradient(self):
        """Very small gradient -> low theta."""
        theta = compute_gradient_theta(0.0001)
        assert theta < 0.1

    def test_exploding_gradient(self):
        """Very large gradient -> low theta."""
        theta = compute_gradient_theta(1000.0)
        assert theta < 0.1


class TestAttentionTheta:
    """Test attention mechanism theta calculation."""

    def test_attention_entropy(self):
        """Uniform attention -> high entropy."""
        uniform = np.ones(100) / 100
        entropy = compute_attention_entropy(uniform)
        assert entropy > 4.0  # log(100) ~ 4.6

    def test_focused_attention(self):
        """Focused attention -> low entropy."""
        focused = np.zeros(100)
        focused[0] = 1.0
        entropy = compute_attention_entropy(focused)
        assert entropy < 0.1

    def test_attention_theta_optimal(self):
        """Moderate entropy -> high theta."""
        max_entropy = np.log(100)
        theta = compute_attention_theta(max_entropy * 0.5, max_entropy)
        assert theta > 0.8

    def test_attention_theta_extreme(self):
        """Extreme entropy (0 or max) -> lower theta."""
        max_entropy = np.log(100)
        theta_low = compute_attention_theta(0.0, max_entropy)
        theta_high = compute_attention_theta(max_entropy, max_entropy)
        assert theta_low < 0.6
        assert theta_high < 0.6


class TestAttentionSparsity:
    """Test attention sparsity calculation."""

    def test_fully_sparse(self):
        """All weights below threshold -> sparsity = 1."""
        weights = np.ones(100) * 0.001
        sparsity = attention_sparsity(weights, threshold=0.1)
        assert sparsity == 1.0

    def test_fully_dense(self):
        """All weights above threshold -> sparsity = 0."""
        weights = np.ones(100) * 0.5
        sparsity = attention_sparsity(weights, threshold=0.1)
        assert sparsity == 0.0


class TestRegularizationTheta:
    """Test regularization theta calculation."""

    def test_optimal_regularization(self):
        """Optimal dropout and L2 -> high theta."""
        theta = compute_regularization_theta(0.4, 0.001, batch_norm=True)
        assert theta > 0.7

    def test_no_regularization(self):
        """No regularization -> lower theta."""
        theta = compute_regularization_theta(0.0, 0.0, batch_norm=False)
        assert theta < 0.5

    def test_excessive_dropout(self):
        """Very high dropout -> lower theta."""
        theta = compute_regularization_theta(0.9, 0.001)
        assert theta < 0.7


class TestCapacityTheta:
    """Test model capacity theta calculation."""

    def test_balanced_capacity(self):
        """Good parameter/sample ratio -> high theta."""
        theta = compute_capacity_theta(
            n_parameters=1000,
            n_samples=10000,
            architecture=ModelArchitecture.MLP
        )
        assert theta > 0.5

    def test_overparameterized(self):
        """Too many parameters -> lower theta."""
        theta = compute_capacity_theta(
            n_parameters=1_000_000,
            n_samples=1000,
            architecture=ModelArchitecture.LINEAR
        )
        assert theta < 0.3


class TestRobustnessTheta:
    """Test robustness theta calculation."""

    def test_robust_model(self):
        """High accuracy under attack -> high theta."""
        theta = compute_robustness_theta(
            clean_accuracy=0.95,
            adversarial_accuracy=0.85,
            noise_accuracy=0.90
        )
        assert theta > 0.8

    def test_brittle_model(self):
        """Low accuracy under attack -> low theta."""
        theta = compute_robustness_theta(
            clean_accuracy=0.95,
            adversarial_accuracy=0.10,
            noise_accuracy=0.20
        )
        assert theta < 0.2


class TestTransferTheta:
    """Test transfer learning theta calculation."""

    def test_good_transfer(self):
        """High target accuracy -> high theta."""
        theta = compute_transfer_theta(
            source_accuracy=0.9,
            target_accuracy=0.85,
            fine_tune_epochs=3
        )
        assert theta > 0.7

    def test_poor_transfer(self):
        """Low target accuracy -> low theta."""
        theta = compute_transfer_theta(
            source_accuracy=0.9,
            target_accuracy=0.3,
            fine_tune_epochs=100
        )
        assert theta < 0.4


class TestUnifiedMLTheta:
    """Test unified ML theta calculation."""

    def test_all_systems_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in ML_SYSTEMS.items():
            if system.train_loss != float('inf'):  # Skip diverging
                theta = compute_ml_theta(system)
                assert 0 <= theta <= 1, f"{name} has invalid theta: {theta}"

    def test_overfit_system_characteristics(self):
        """Overfitting system should have moderate theta."""
        system = ML_SYSTEMS["overfit_mlp"]
        theta = compute_ml_theta(system)
        # Overfit has good training but poor generalization
        assert 0.2 < theta < 0.6

    def test_bert_high_theta(self):
        """Well-trained BERT should have high theta."""
        system = ML_SYSTEMS["bert_base"]
        theta = compute_ml_theta(system)
        assert theta > 0.6

    def test_ordering_preserved(self):
        """Better models should generally have higher theta."""
        theta_linear = compute_ml_theta(ML_SYSTEMS["linear_regression"])
        theta_bert = compute_ml_theta(ML_SYSTEMS["bert_base"])
        assert theta_linear < theta_bert


class TestLearningRegimeClassification:
    """Test learning regime classification."""

    def test_underfitting(self):
        assert classify_learning_regime(0.2) == LearningRegime.UNDERFITTING

    def test_learning(self):
        assert classify_learning_regime(0.45) == LearningRegime.LEARNING

    def test_optimal(self):
        assert classify_learning_regime(0.7, generalization_gap=0.05) == LearningRegime.OPTIMAL

    def test_overfitting_with_gap(self):
        """High theta but high gap -> overfitting."""
        regime = classify_learning_regime(0.75, generalization_gap=0.2)
        assert regime == LearningRegime.OVERFITTING


class TestScalingLawTheta:
    """Test neural scaling law theta calculation."""

    def test_optimal_scaling(self):
        """Chinchilla-optimal scaling -> high theta."""
        theta = scaling_law_theta(
            n_parameters=1e9,
            n_tokens=1e9,
            compute_budget=6e18
        )
        assert theta > 0.8

    def test_undertrained(self):
        """Too few tokens -> lower theta."""
        theta = scaling_law_theta(
            n_parameters=1e9,
            n_tokens=1e6,
            compute_budget=6e15
        )
        assert theta < 0.5

    def test_overtrained(self):
        """Too many tokens -> lower theta."""
        theta = scaling_law_theta(
            n_parameters=1e6,
            n_tokens=1e12,
            compute_budget=6e18
        )
        assert theta < 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_theta_bounds(self):
        """All theta functions should return values in [0, 1]."""
        # Generalization
        assert 0 <= compute_generalization_theta(-0.1, 1.5) <= 1
        assert 0 <= compute_generalization_theta(1.0, 0.0) <= 1

        # Loss ratio
        assert 0 <= compute_loss_ratio_theta(0.0, 1.0) <= 1
        assert 0 <= compute_loss_ratio_theta(1.0, 0.0) <= 1

        # Gradient
        assert 0 <= compute_gradient_theta(-1.0) <= 1
        assert 0 <= compute_gradient_theta(1e10) <= 1

        # Attention
        assert 0 <= compute_attention_theta(-1.0, 5.0) <= 1
        assert 0 <= compute_attention_theta(100.0, 5.0) <= 1

    def test_diverging_system(self):
        """Diverging training should have low theta."""
        system = ML_SYSTEMS["diverging_training"]
        # This system has inf loss, so we need to handle gracefully
        # The gradient is 1000 (exploding) which will give low theta
        theta_grad = compute_gradient_theta(system.gradient_norm)
        assert theta_grad < 0.1


class TestDocstrings:
    """Test that functions have proper documentation."""

    def test_module_docstring(self):
        """Module should have docstring with citations."""
        import theta_calculator.domains.ai_ml as module
        assert module.__doc__ is not None
        assert "\\cite{" in module.__doc__

    def test_function_docstrings(self):
        """Key functions should have docstrings."""
        functions = [
            compute_generalization_theta,
            compute_gradient_theta,
            compute_ml_theta,
            classify_learning_regime,
        ]
        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
