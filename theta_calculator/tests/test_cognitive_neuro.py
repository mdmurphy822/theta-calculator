"""
Tests for Cognitive Neuroscience Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Attention state classification
- Component theta calculations
- Working memory analysis
"""

import pytest

from theta_calculator.domains.cognitive_neuro import (
    NeuroCognitiveSystem,
    AttentionState,
    MetacognitiveLevel,
    WorkingMemoryPhase,
    ExecutiveMode,
    compute_attention_theta,
    compute_metacognition_theta,
    compute_wm_theta,
    compute_executive_theta,
    compute_arousal_theta,
    compute_global_workspace_theta,
    compute_cognitive_neuro_theta,
    classify_attention_state,
    classify_metacognition_level,
    classify_wm_phase,
    classify_executive_mode,
    COGNITIVE_NEURO_SYSTEMS,
    COWAN_LIMIT,
    MILLER_LIMIT,
)


class TestCognitiveNeuroSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """COGNITIVE_NEURO_SYSTEMS dict should exist."""
        assert COGNITIVE_NEURO_SYSTEMS is not None
        assert isinstance(COGNITIVE_NEURO_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(COGNITIVE_NEURO_SYSTEMS) >= 5

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = [
            "vigilant",
            "flow_state",
            "cognitive_overload",
            "fatigue",
        ]
        for name in expected:
            assert name in COGNITIVE_NEURO_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in COGNITIVE_NEURO_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "attention_resources")
            assert hasattr(system, "attention_demand")
            assert hasattr(system, "d_prime")
            assert hasattr(system, "meta_d_prime")
            assert hasattr(system, "wm_items")
            assert hasattr(system, "wm_capacity")
            assert hasattr(system, "prefrontal_activation")
            assert hasattr(system, "recurrent_connectivity")


class TestAttentionTheta:
    """Test attention theta calculation."""

    def test_high_resources_low_demand(self):
        """High resources / low demand -> theta = 1.0."""
        theta = compute_attention_theta(1.0, 0.1)
        assert theta == 1.0

    def test_low_resources_high_demand(self):
        """Low resources / high demand -> low theta."""
        theta = compute_attention_theta(0.1, 1.0)
        assert theta == pytest.approx(0.1)

    def test_matched_resources_demand(self):
        """Equal resources and demand -> theta = 1.0."""
        theta = compute_attention_theta(0.5, 0.5)
        assert theta == 1.0

    def test_zero_demand(self):
        """Zero demand -> theta = 1.0."""
        theta = compute_attention_theta(0.5, 0.0)
        assert theta == 1.0

    def test_theta_always_in_range(self):
        """Theta should always be in [0, 1]."""
        test_cases = [
            (0.0, 1.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (2.0, 1.0),  # Over-resourced
        ]
        for resources, demand in test_cases:
            theta = compute_attention_theta(resources, demand)
            assert 0.0 <= theta <= 1.0


class TestMetacognitionTheta:
    """Test metacognition theta calculation."""

    def test_perfect_metacognition(self):
        """meta-d' = d' -> theta = 1.0."""
        theta = compute_metacognition_theta(2.5, 2.5)
        assert theta == 1.0

    def test_no_metacognition(self):
        """meta-d' = 0 -> theta = 0.0."""
        theta = compute_metacognition_theta(0.0, 2.5)
        assert theta == 0.0

    def test_partial_metacognition(self):
        """meta-d' = 0.5 * d' -> theta = 0.5."""
        theta = compute_metacognition_theta(1.25, 2.5)
        assert theta == pytest.approx(0.5)

    def test_zero_d_prime(self):
        """d' = 0 -> theta = 0.0."""
        theta = compute_metacognition_theta(1.0, 0.0)
        assert theta == 0.0


class TestWorkingMemoryTheta:
    """Test working memory theta calculation."""

    def test_optimal_load(self):
        """4 items (Cowan's limit) -> high theta."""
        theta = compute_wm_theta(4, 7)
        assert theta > 0.5  # Near optimal

    def test_empty_wm(self):
        """0 items -> lower theta (underutilized)."""
        theta = compute_wm_theta(0, 7)
        # Deviation from optimal (4) = 4
        # theta = 1 - 4/7 ~ 0.43
        assert theta < 1.0

    def test_overloaded_wm(self):
        """Items > capacity -> low theta."""
        theta = compute_wm_theta(10, 6)
        assert theta < 0.3

    def test_zero_capacity(self):
        """Zero capacity -> theta = 0.0."""
        theta = compute_wm_theta(3, 0)
        assert theta == 0.0


class TestExecutiveTheta:
    """Test executive function theta calculation."""

    def test_high_pfc_high_recurrent(self):
        """Both high -> theta = 1.0."""
        theta = compute_executive_theta(1.0, 1.0)
        assert theta == 1.0

    def test_low_pfc_low_recurrent(self):
        """Both low -> low theta."""
        theta = compute_executive_theta(0.1, 0.1)
        assert theta == pytest.approx(0.1)

    def test_geometric_mean(self):
        """sqrt(0.5 * 0.5) = 0.5."""
        theta = compute_executive_theta(0.5, 0.5)
        assert theta == pytest.approx(0.5)

    def test_asymmetric(self):
        """High PFC but low recurrent -> moderate theta."""
        theta = compute_executive_theta(0.9, 0.1)
        assert theta == pytest.approx(0.3, rel=0.1)


class TestArousalTheta:
    """Test arousal (Yerkes-Dodson) theta calculation."""

    def test_optimal_arousal(self):
        """Optimal arousal (0.6) -> high theta."""
        theta = compute_arousal_theta(0.6)
        assert theta > 0.9

    def test_low_arousal(self):
        """Very low arousal -> lower theta."""
        theta = compute_arousal_theta(0.1)
        assert theta < 0.5

    def test_high_arousal(self):
        """Very high arousal -> lower theta."""
        theta = compute_arousal_theta(1.0)
        assert theta < 0.5


class TestGlobalWorkspaceTheta:
    """Test Global Workspace ignition theta."""

    def test_subliminal(self):
        """Below threshold -> theta = 0."""
        theta = compute_global_workspace_theta(0.2, threshold=0.3)
        assert theta == 0.0

    def test_at_threshold(self):
        """At threshold -> theta = 0."""
        theta = compute_global_workspace_theta(0.3, threshold=0.3)
        assert theta == 0.0

    def test_above_threshold(self):
        """Above threshold -> positive theta."""
        theta = compute_global_workspace_theta(0.5, threshold=0.3)
        assert theta > 0.0

    def test_full_activation(self):
        """Full activation -> theta = 1."""
        theta = compute_global_workspace_theta(1.0, threshold=0.3)
        assert theta == 1.0


class TestUnifiedTheta:
    """Test unified theta calculation for systems."""

    def test_all_systems_have_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in COGNITIVE_NEURO_SYSTEMS.items():
            theta = compute_cognitive_neuro_theta(system)
            assert 0.0 <= theta <= 1.0, f"{name}: theta={theta}"

    def test_flow_state_high_theta(self):
        """Flow state should have high theta."""
        flow = COGNITIVE_NEURO_SYSTEMS["flow_state"]
        theta = compute_cognitive_neuro_theta(flow)
        assert theta > 0.6

    def test_overload_low_theta(self):
        """Cognitive overload should have low theta."""
        overload = COGNITIVE_NEURO_SYSTEMS["cognitive_overload"]
        theta = compute_cognitive_neuro_theta(overload)
        assert theta < 0.5

    def test_vigilant_higher_than_fatigue(self):
        """Vigilant state should have higher theta than fatigue."""
        vigilant = COGNITIVE_NEURO_SYSTEMS["vigilant"]
        fatigue = COGNITIVE_NEURO_SYSTEMS["fatigue"]
        assert compute_cognitive_neuro_theta(vigilant) > compute_cognitive_neuro_theta(fatigue)


class TestAttentionStateClassification:
    """Test attention state classification."""

    def test_inattentional(self):
        """theta < 0.2 -> INATTENTIONAL."""
        assert classify_attention_state(0.1) == AttentionState.INATTENTIONAL

    def test_divided(self):
        """0.2 <= theta < 0.5 -> DIVIDED."""
        assert classify_attention_state(0.3) == AttentionState.DIVIDED

    def test_selective(self):
        """0.5 <= theta < 0.8 -> SELECTIVE."""
        assert classify_attention_state(0.6) == AttentionState.SELECTIVE

    def test_focused(self):
        """theta >= 0.8 -> FOCUSED."""
        assert classify_attention_state(0.9) == AttentionState.FOCUSED


class TestMetacognitionClassification:
    """Test metacognition level classification."""

    def test_absent(self):
        """theta < 0.1 -> ABSENT."""
        assert classify_metacognition_level(0.05) == MetacognitiveLevel.ABSENT

    def test_implicit(self):
        """0.1 <= theta < 0.5 -> IMPLICIT."""
        assert classify_metacognition_level(0.3) == MetacognitiveLevel.IMPLICIT

    def test_explicit(self):
        """0.5 <= theta < 0.9 -> EXPLICIT."""
        assert classify_metacognition_level(0.7) == MetacognitiveLevel.EXPLICIT

    def test_reflective(self):
        """theta >= 0.9 -> REFLECTIVE."""
        assert classify_metacognition_level(0.95) == MetacognitiveLevel.REFLECTIVE


class TestWMPhaseClassification:
    """Test working memory phase classification."""

    def test_underload(self):
        """Low items -> UNDERLOAD."""
        assert classify_wm_phase(1, 7) == WorkingMemoryPhase.UNDERLOAD

    def test_optimal(self):
        """Moderate items -> OPTIMAL."""
        assert classify_wm_phase(4, 7) == WorkingMemoryPhase.OPTIMAL

    def test_near_capacity(self):
        """High items -> NEAR_CAPACITY."""
        assert classify_wm_phase(6, 7) == WorkingMemoryPhase.NEAR_CAPACITY

    def test_overload(self):
        """Items > capacity -> OVERLOAD."""
        assert classify_wm_phase(10, 7) == WorkingMemoryPhase.OVERLOAD


class TestExecutiveModeClassification:
    """Test executive mode classification."""

    def test_automatic(self):
        """theta < 0.25 -> AUTOMATIC."""
        assert classify_executive_mode(0.1) == ExecutiveMode.AUTOMATIC

    def test_routine(self):
        """0.25 <= theta < 0.5 -> ROUTINE."""
        assert classify_executive_mode(0.4) == ExecutiveMode.ROUTINE

    def test_adaptive(self):
        """0.5 <= theta < 0.75 -> ADAPTIVE."""
        assert classify_executive_mode(0.6) == ExecutiveMode.ADAPTIVE

    def test_deliberate(self):
        """theta >= 0.75 -> DELIBERATE."""
        assert classify_executive_mode(0.9) == ExecutiveMode.DELIBERATE


class TestConstants:
    """Test cognitive constants."""

    def test_cowan_limit(self):
        """Cowan's limit should be 4."""
        assert COWAN_LIMIT == 4

    def test_miller_limit(self):
        """Miller's number should be 7."""
        assert MILLER_LIMIT == 7


class TestSystemDataclass:
    """Test NeuroCognitiveSystem dataclass."""

    def test_create_minimal_system(self):
        """Should create system with required parameters."""
        system = NeuroCognitiveSystem(
            name="Test",
            attention_resources=0.8,
            attention_demand=0.5,
            d_prime=2.0,
            meta_d_prime=1.5,
            wm_items=4,
            wm_capacity=7,
            prefrontal_activation=0.7,
            recurrent_connectivity=0.6,
        )
        assert system.name == "Test"
        assert system.attention_resources == 0.8

    def test_metacognitive_efficiency_property(self):
        """metacognitive_efficiency should be meta-d'/d'."""
        system = NeuroCognitiveSystem(
            name="Test",
            attention_resources=0.8,
            attention_demand=0.5,
            d_prime=2.0,
            meta_d_prime=1.5,
            wm_items=4,
            wm_capacity=7,
            prefrontal_activation=0.7,
            recurrent_connectivity=0.6,
        )
        assert system.metacognitive_efficiency == pytest.approx(0.75)

    def test_attention_ratio_property(self):
        """attention_ratio should be resources/demand."""
        system = NeuroCognitiveSystem(
            name="Test",
            attention_resources=0.8,
            attention_demand=0.5,
            d_prime=2.0,
            meta_d_prime=1.5,
            wm_items=4,
            wm_capacity=7,
            prefrontal_activation=0.7,
            recurrent_connectivity=0.6,
        )
        # Capped at 1.0
        assert system.attention_ratio == 1.0

    def test_wm_load_ratio_property(self):
        """wm_load_ratio should be items/capacity."""
        system = NeuroCognitiveSystem(
            name="Test",
            attention_resources=0.8,
            attention_demand=0.5,
            d_prime=2.0,
            meta_d_prime=1.5,
            wm_items=4,
            wm_capacity=7,
            prefrontal_activation=0.7,
            recurrent_connectivity=0.6,
        )
        assert system.wm_load_ratio == pytest.approx(4/7)
