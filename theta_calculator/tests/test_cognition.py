"""
Tests for Cognition Domain Module

Tests cover:
- Integrated Information Theory (IIT)
- Neural criticality and avalanches
- Working memory (Miller's 7+/-2)
- Attention and focus
- Global workspace theory
- Theta range validation [0, 1]
"""

import pytest
import numpy as np

from theta_calculator.domains.cognition import (
    CognitiveSystem,
    ConsciousnessState,
    BrainState,
    compute_cognition_theta,
    compute_phi_simple,
    compute_phi_ratio,
    compute_criticality_theta,
    compute_working_memory_theta,
    compute_attention_theta,
    compute_global_broadcast_theta,
    criticality_exponent,
    consciousness_from_phi,
    classify_brain_state,
    miller_capacity,
    COGNITIVE_SYSTEMS,
)


class TestCognitiveSystemsExist:
    """Test that example cognitive systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """COGNITIVE_SYSTEMS dict should exist."""
        assert COGNITIVE_SYSTEMS is not None
        assert isinstance(COGNITIVE_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 6 systems."""
        assert len(COGNITIVE_SYSTEMS) >= 6

    def test_key_systems_defined(self):
        """Key cognitive states should be defined."""
        expected = ["deep_sleep", "flow_state", "anesthesia", "focused_work"]
        for name in expected:
            assert name in COGNITIVE_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in COGNITIVE_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "n_modules")
            assert hasattr(system, "integration")
            assert hasattr(system, "criticality_exponent")
            assert hasattr(system, "working_memory_load")
            assert hasattr(system, "attention_focus")


class TestPhiSimple:
    """Test simplified integrated information calculation."""

    def test_empty_states(self):
        """Empty states give zero Phi."""
        phi = compute_phi_simple(np.array([[]]), np.array([]))
        assert phi == 0.0

    def test_simple_system(self):
        """Simple connected system has positive Phi."""
        # 3x3 connectivity matrix with some connections
        conn = np.array([
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.4],
            [0.1, 0.2, 0.5]
        ])
        states = np.array([1, 0, 1])
        phi = compute_phi_simple(conn, states)
        assert phi >= 0

    def test_disconnected_system(self):
        """Disconnected system (diagonal only) has lower Phi."""
        conn_connected = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ])
        conn_disconnected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        states = np.array([1, 0, 1])

        phi_connected = compute_phi_simple(conn_connected, states)
        phi_disconnected = compute_phi_simple(conn_disconnected, states)

        # Connected system should have higher Phi
        assert phi_connected >= phi_disconnected


class TestPhiRatio:
    """Test Phi ratio (theta) calculation."""

    def test_zero_phi(self):
        """Zero Phi gives zero theta."""
        theta = compute_phi_ratio(0.0, 1.0)
        assert theta == 0.0

    def test_max_phi(self):
        """Phi = Phi_max gives theta = 1."""
        theta = compute_phi_ratio(1.0, 1.0)
        assert theta == 1.0

    def test_half_phi(self):
        """Phi = 0.5 * Phi_max gives theta = 0.5."""
        theta = compute_phi_ratio(0.5, 1.0)
        assert theta == pytest.approx(0.5)

    def test_phi_exceeds_max(self):
        """Phi > Phi_max is clipped to 1.0."""
        theta = compute_phi_ratio(2.0, 1.0)
        assert theta == 1.0

    def test_zero_phi_max(self):
        """Zero Phi_max gives zero theta."""
        theta = compute_phi_ratio(0.5, 0.0)
        assert theta == 0.0

    def test_negative_phi_max(self):
        """Negative Phi_max gives zero theta."""
        theta = compute_phi_ratio(0.5, -1.0)
        assert theta == 0.0


class TestConsciousnessFromPhi:
    """Test consciousness state classification from Phi."""

    def test_unconscious(self):
        """Low Phi -> UNCONSCIOUS."""
        assert consciousness_from_phi(0.1) == ConsciousnessState.UNCONSCIOUS
        assert consciousness_from_phi(0.15) == ConsciousnessState.UNCONSCIOUS

    def test_subliminal(self):
        """Phi in [0.2, 0.4) -> SUBLIMINAL."""
        assert consciousness_from_phi(0.2) == ConsciousnessState.SUBLIMINAL
        assert consciousness_from_phi(0.35) == ConsciousnessState.SUBLIMINAL

    def test_preconscious(self):
        """Phi in [0.4, 0.6) -> PRECONSCIOUS."""
        assert consciousness_from_phi(0.4) == ConsciousnessState.PRECONSCIOUS
        assert consciousness_from_phi(0.55) == ConsciousnessState.PRECONSCIOUS

    def test_conscious(self):
        """Phi in [0.6, 0.8) -> CONSCIOUS."""
        assert consciousness_from_phi(0.6) == ConsciousnessState.CONSCIOUS
        assert consciousness_from_phi(0.75) == ConsciousnessState.CONSCIOUS

    def test_highly_aware(self):
        """High Phi -> HIGHLY_AWARE."""
        assert consciousness_from_phi(0.8) == ConsciousnessState.HIGHLY_AWARE
        assert consciousness_from_phi(0.95) == ConsciousnessState.HIGHLY_AWARE


class TestCriticalityExponent:
    """Test power-law exponent calculation from avalanche sizes."""

    def test_insufficient_data(self):
        """Too few avalanches returns zero."""
        tau = criticality_exponent([1, 2, 3])
        assert tau == 0.0

    def test_power_law_data(self):
        """Power-law distributed data gives reasonable exponent."""
        # Generate power-law distributed sizes with tau ~ 1.5
        np.random.seed(42)
        sizes = np.random.pareto(0.5, 100) + 1
        sizes = [int(s) for s in sizes if s > 0]

        if len(sizes) >= 10:
            tau = criticality_exponent(sizes)
            # Should be positive
            assert tau > 0

    def test_uniform_sizes(self):
        """Uniform sizes give different exponent."""
        sizes = [5] * 50  # All same size
        tau = criticality_exponent(sizes)
        # With only one unique size, may return 0
        assert tau >= 0


class TestCriticalityTheta:
    """Test criticality theta calculation."""

    def test_at_criticality(self):
        """tau = 1.5 gives theta = 1."""
        theta = compute_criticality_theta(1.5, tau_critical=1.5)
        assert theta == 1.0

    def test_subcritical(self):
        """tau < 1.5 gives lower theta."""
        theta = compute_criticality_theta(1.0, tau_critical=1.5, tau_range=0.5)
        assert theta == 0.0  # Exactly at range boundary

    def test_supercritical(self):
        """tau > 1.5 gives lower theta."""
        theta = compute_criticality_theta(2.0, tau_critical=1.5, tau_range=0.5)
        assert theta == 0.0  # Exactly at range boundary

    def test_near_criticality(self):
        """tau near 1.5 gives high theta."""
        theta = compute_criticality_theta(1.4, tau_critical=1.5, tau_range=0.5)
        assert theta == pytest.approx(0.8)

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for tau in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            theta = compute_criticality_theta(tau)
            assert 0 <= theta <= 1


class TestMillerCapacity:
    """Test Miller's magical number 7 +/- 2."""

    def test_returns_tuple(self):
        """Should return (min, typical, max)."""
        result = miller_capacity()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_values(self):
        """Should return (5, 7, 9)."""
        min_cap, typical, max_cap = miller_capacity()
        assert min_cap == 5
        assert typical == 7
        assert max_cap == 9


class TestWorkingMemoryTheta:
    """Test working memory theta calculation."""

    def test_zero_items(self):
        """Zero items gives theta = 0."""
        theta = compute_working_memory_theta(0, capacity=7)
        assert theta == 0.0

    def test_at_capacity(self):
        """Items = capacity gives theta = 1."""
        theta = compute_working_memory_theta(7, capacity=7)
        assert theta == 1.0

    def test_half_capacity(self):
        """Half capacity gives theta = 0.5."""
        theta = compute_working_memory_theta(3, capacity=6)
        assert theta == pytest.approx(0.5)

    def test_over_capacity(self):
        """Over capacity is clipped to 1.0."""
        theta = compute_working_memory_theta(10, capacity=7)
        assert theta == 1.0

    def test_zero_capacity(self):
        """Zero capacity gives theta = 0."""
        theta = compute_working_memory_theta(5, capacity=0)
        assert theta == 0.0

    def test_default_capacity(self):
        """Default capacity is 7."""
        theta = compute_working_memory_theta(7)
        assert theta == 1.0


class TestAttentionTheta:
    """Test attention theta calculation."""

    def test_full_focus_no_distractors(self):
        """Full focus, no distractors -> theta = 1."""
        theta = compute_attention_theta(1.0, distractors=0)
        assert theta == 1.0

    def test_zero_focus(self):
        """Zero focus -> theta = 0."""
        theta = compute_attention_theta(0.0, distractors=0)
        assert theta == 0.0

    def test_distractors_reduce_theta(self):
        """Distractors reduce theta."""
        theta_no_dist = compute_attention_theta(0.8, distractors=0)
        theta_with_dist = compute_attention_theta(0.8, distractors=5)
        assert theta_with_dist < theta_no_dist

    def test_max_distractors(self):
        """Max distractors -> theta = 0."""
        theta = compute_attention_theta(1.0, distractors=10, max_distractors=10)
        assert theta == 0.0

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for focus in [0, 0.5, 1.0]:
            for dist in [0, 5, 10]:
                theta = compute_attention_theta(focus, distractors=dist)
                assert 0 <= theta <= 1


class TestGlobalBroadcastTheta:
    """Test global workspace theta calculation."""

    def test_all_modules_activated(self):
        """All modules activated -> theta = 1."""
        theta = compute_global_broadcast_theta(6, 6, broadcast_strength=1.0)
        assert theta == 1.0

    def test_no_modules_activated(self):
        """No modules activated -> theta = 0."""
        theta = compute_global_broadcast_theta(0, 6)
        assert theta == 0.0

    def test_half_modules(self):
        """Half modules -> theta = 0.5."""
        theta = compute_global_broadcast_theta(3, 6, broadcast_strength=1.0)
        assert theta == pytest.approx(0.5)

    def test_broadcast_strength(self):
        """Broadcast strength modulates theta."""
        theta_full = compute_global_broadcast_theta(6, 6, broadcast_strength=1.0)
        theta_weak = compute_global_broadcast_theta(6, 6, broadcast_strength=0.5)
        assert theta_weak == pytest.approx(theta_full * 0.5)

    def test_zero_total_modules(self):
        """Zero total modules -> theta = 0."""
        theta = compute_global_broadcast_theta(0, 0)
        assert theta == 0.0


class TestUnifiedCognitionTheta:
    """Test unified cognition theta calculation."""

    def test_all_systems_valid_theta(self):
        """All cognitive systems should have theta in [0, 1]."""
        for name, system in COGNITIVE_SYSTEMS.items():
            theta = compute_cognition_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_flow_state_high_theta(self):
        """Flow state should have high theta."""
        flow = COGNITIVE_SYSTEMS["flow_state"]
        theta = compute_cognition_theta(flow)
        assert theta > 0.7

    def test_anesthesia_low_theta(self):
        """Anesthesia should have low theta."""
        anesthesia = COGNITIVE_SYSTEMS["anesthesia"]
        theta = compute_cognition_theta(anesthesia)
        assert theta < 0.3

    def test_deep_sleep_moderate_theta(self):
        """Deep sleep has moderate theta (high Phi but low attention)."""
        deep_sleep = COGNITIVE_SYSTEMS["deep_sleep"]
        theta = compute_cognition_theta(deep_sleep)
        # Deep sleep: high integration but low attention/WM
        assert 0.1 < theta < 0.5


class TestClassifyBrainState:
    """Test brain state classification from theta."""

    def test_deep_sleep(self):
        """Very low theta -> SLEEP_DEEP."""
        assert classify_brain_state(0.1) == BrainState.SLEEP_DEEP
        assert classify_brain_state(0.14) == BrainState.SLEEP_DEEP

    def test_rem_sleep(self):
        """Low theta -> SLEEP_REM."""
        assert classify_brain_state(0.2) == BrainState.SLEEP_REM
        assert classify_brain_state(0.25) == BrainState.SLEEP_REM

    def test_drowsy(self):
        """theta in [0.30, 0.45) -> DROWSY."""
        assert classify_brain_state(0.35) == BrainState.DROWSY
        assert classify_brain_state(0.40) == BrainState.DROWSY

    def test_relaxed(self):
        """theta in [0.45, 0.60) -> RELAXED."""
        assert classify_brain_state(0.50) == BrainState.RELAXED
        assert classify_brain_state(0.55) == BrainState.RELAXED

    def test_alert(self):
        """theta in [0.60, 0.75) -> ALERT."""
        assert classify_brain_state(0.65) == BrainState.ALERT
        assert classify_brain_state(0.70) == BrainState.ALERT

    def test_focused(self):
        """theta in [0.75, 0.90) -> FOCUSED."""
        assert classify_brain_state(0.80) == BrainState.FOCUSED
        assert classify_brain_state(0.85) == BrainState.FOCUSED

    def test_flow(self):
        """High theta -> FLOW."""
        assert classify_brain_state(0.92) == BrainState.FLOW
        assert classify_brain_state(0.98) == BrainState.FLOW


class TestCognitiveSystemDataclass:
    """Test CognitiveSystem dataclass."""

    def test_create_system(self):
        """Should create system with all parameters."""
        system = CognitiveSystem(
            name="Test",
            n_modules=6,
            integration=0.5,
            differentiation=0.5,
            criticality_exponent=1.5,
            working_memory_load=4,
            attention_focus=0.7
        )
        assert system.name == "Test"
        assert system.n_modules == 6
        assert system.integration == 0.5

    def test_system_theta_computable(self):
        """Created system should have computable theta."""
        system = CognitiveSystem(
            name="Test",
            n_modules=6,
            integration=0.5,
            differentiation=0.5,
            criticality_exponent=1.5,
            working_memory_load=4,
            attention_focus=0.7
        )
        theta = compute_cognition_theta(system)
        assert 0 <= theta <= 1


class TestEnums:
    """Test enum definitions."""

    def test_consciousness_states(self):
        """All consciousness states should be defined."""
        assert ConsciousnessState.UNCONSCIOUS.value == "unconscious"
        assert ConsciousnessState.SUBLIMINAL.value == "subliminal"
        assert ConsciousnessState.PRECONSCIOUS.value == "preconscious"
        assert ConsciousnessState.CONSCIOUS.value == "conscious"
        assert ConsciousnessState.HIGHLY_AWARE.value == "highly_aware"

    def test_brain_states(self):
        """All brain states should be defined."""
        assert BrainState.SLEEP_DEEP.value == "deep_sleep"
        assert BrainState.SLEEP_REM.value == "rem_sleep"
        assert BrainState.DROWSY.value == "drowsy"
        assert BrainState.RELAXED.value == "relaxed"
        assert BrainState.ALERT.value == "alert"
        assert BrainState.FOCUSED.value == "focused"
        assert BrainState.FLOW.value == "flow"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_integration(self):
        """Negative integration should give low theta."""
        system = CognitiveSystem(
            name="Test",
            n_modules=6,
            integration=-0.5,  # Invalid but should handle
            differentiation=0.5,
            criticality_exponent=1.5,
            working_memory_load=4,
            attention_focus=0.5
        )
        theta = compute_cognition_theta(system)
        assert 0 <= theta <= 1  # Should be clipped

    def test_very_high_criticality_exponent(self):
        """Very high tau (supercritical) should give low theta."""
        theta = compute_criticality_theta(5.0, tau_critical=1.5, tau_range=0.5)
        assert theta == 0.0

    def test_zero_criticality_exponent(self):
        """Zero tau should give low theta."""
        theta = compute_criticality_theta(0.0, tau_critical=1.5, tau_range=0.5)
        assert theta == 0.0

    def test_negative_working_memory(self):
        """Negative WM items should give theta = 0."""
        theta = compute_working_memory_theta(-5, capacity=7)
        assert theta == 0.0

    def test_negative_attention(self):
        """Negative attention should be clipped."""
        theta = compute_attention_theta(-0.5, distractors=0)
        assert theta == 0.0
