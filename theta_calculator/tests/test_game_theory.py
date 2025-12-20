"""
Tests for game_theory module.

Tests quantum game theta calculations using entanglement parameter
for strategic games like Prisoner's Dilemma.
"""

import pytest
import numpy as np

from theta_calculator.domains.game_theory import (
    GAME_SYSTEMS,
    QuantumGame,
    GameType,
    StrategyType,
    compute_entanglement_theta,
    compute_quantum_advantage,
    entangling_gate,
    strategy_operator,
    classical_payoff,
    quantum_payoff,
    prisoners_dilemma_payoff,
    find_classical_nash,
    COOPERATE,
    DEFECT,
    QUANTUM_STRATEGY,
    PD_PAYOFF,
    CHICKEN_PAYOFF,
    BOS_PAYOFF,
)


class TestGameSystems:
    """Test the predefined game systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "classical_pd", "partial_quantum_pd", "quantum_pd",
            "classical_chicken", "quantum_chicken",
            "classical_bos", "quantum_bos"
        ]
        for name in expected:
            assert name in GAME_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, game in GAME_SYSTEMS.items():
            assert isinstance(game, QuantumGame)
            assert game.name
            assert isinstance(game.game_type, GameType)
            assert 0 <= game.gamma <= np.pi / 2
            assert game.n_players == 2
            assert game.n_strategies == 2


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, game in GAME_SYSTEMS.items():
            theta = compute_entanglement_theta(game)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_classical_game_zero_theta(self):
        """Classical games (gamma=0) should have theta=0."""
        classical = GAME_SYSTEMS["classical_pd"]
        theta = compute_entanglement_theta(classical)
        assert theta == 0.0, f"Classical should have zero theta: {theta}"

    def test_quantum_game_one_theta(self):
        """Full quantum games (gamma=pi/2) should have theta=1."""
        quantum = GAME_SYSTEMS["quantum_pd"]
        theta = compute_entanglement_theta(quantum)
        assert abs(theta - 1.0) < 1e-10, f"Quantum should have theta=1: {theta}"

    def test_partial_quantum_half_theta(self):
        """Partial quantum (gamma=pi/4) should have theta=0.5."""
        partial = GAME_SYSTEMS["partial_quantum_pd"]
        theta = compute_entanglement_theta(partial)
        assert abs(theta - 0.5) < 1e-10, f"Partial should have theta=0.5: {theta}"

    def test_theta_formula(self):
        """Test theta = 2*gamma/pi formula."""
        for name, game in GAME_SYSTEMS.items():
            theta = compute_entanglement_theta(game)
            expected = 2 * game.gamma / np.pi
            assert abs(theta - expected) < 1e-10, f"{name} theta mismatch"


class TestQuantumGameProperties:
    """Test QuantumGame properties."""

    def test_is_classical_property(self):
        """Test is_classical property."""
        classical = GAME_SYSTEMS["classical_pd"]
        quantum = GAME_SYSTEMS["quantum_pd"]
        assert classical.is_classical
        assert not quantum.is_classical

    def test_is_fully_quantum_property(self):
        """Test is_fully_quantum property."""
        classical = GAME_SYSTEMS["classical_pd"]
        quantum = GAME_SYSTEMS["quantum_pd"]
        assert not classical.is_fully_quantum
        assert quantum.is_fully_quantum


class TestEntanglingGate:
    """Test the entangling gate J(gamma)."""

    def test_identity_at_zero(self):
        """J(0) should be identity."""
        J = entangling_gate(0.0)
        identity = np.eye(4)
        assert np.allclose(J, identity, atol=1e-10)

    def test_unitarity(self):
        """J(gamma) should be unitary for all gamma."""
        for gamma in [0, np.pi/4, np.pi/2]:
            J = entangling_gate(gamma)
            # J * J^dagger should be identity
            product = J @ J.conj().T
            assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_max_entanglement(self):
        """J(pi/2) should create maximal entanglement."""
        J = entangling_gate(np.pi / 2)
        # Apply to |00> state
        psi = J @ np.array([1, 0, 0, 0])
        # Result should be (|00> + i|11>)/sqrt(2)
        expected = np.array([1, 0, 0, 1j]) / np.sqrt(2)
        assert np.allclose(psi, expected, atol=1e-10)


class TestStrategyOperator:
    """Test strategy operator construction."""

    def test_cooperate_is_identity(self):
        """U(0,0) should be identity."""
        assert np.allclose(COOPERATE, np.eye(2), atol=1e-10)

    def test_defect_is_sigma_x(self):
        """U(pi,0) should be sigma_x (bit flip)."""
        sigma_x = np.array([[0, 1], [1, 0]])
        assert np.allclose(np.abs(DEFECT), np.abs(sigma_x), atol=1e-10)

    def test_unitarity(self):
        """All strategy operators should be unitary."""
        for theta in [0, np.pi/4, np.pi/2, np.pi]:
            for phi in [0, np.pi/4, np.pi/2]:
                U = strategy_operator(theta, phi)
                product = U @ U.conj().T
                assert np.allclose(product, np.eye(2), atol=1e-10)


class TestPrisonersDilemmaPayoff:
    """Test Prisoner's Dilemma payoff function."""

    def test_mutual_cooperation(self):
        """Both cooperate: (3, 3)."""
        payoff = prisoners_dilemma_payoff(True, True)
        assert payoff == (3, 3)

    def test_mutual_defection(self):
        """Both defect: (1, 1)."""
        payoff = prisoners_dilemma_payoff(False, False)
        assert payoff == (1, 1)

    def test_sucker_payoff(self):
        """A cooperates, B defects: (0, 5)."""
        payoff = prisoners_dilemma_payoff(True, False)
        assert payoff == (0, 5)

    def test_temptation_payoff(self):
        """A defects, B cooperates: (5, 0)."""
        payoff = prisoners_dilemma_payoff(False, True)
        assert payoff == (5, 0)


class TestClassicalPayoff:
    """Test classical payoff calculation."""

    def test_pd_payoff_matrix(self):
        """Test using PD payoff matrix directly."""
        # (Cooperate, Cooperate) = (3, 3)
        p_a, p_b = classical_payoff(PD_PAYOFF, 0, 0)
        assert p_a == 3 and p_b == 3

        # (Defect, Defect) = (1, 1)
        p_a, p_b = classical_payoff(PD_PAYOFF, 1, 1)
        assert p_a == 1 and p_b == 1


class TestQuantumPayoff:
    """Test quantum payoff calculation."""

    def test_classical_strategies_in_classical_game(self):
        """Classical strategies in gamma=0 should give classical payoffs."""
        # Both defect in classical game
        p_a, p_b = quantum_payoff(PD_PAYOFF, DEFECT, DEFECT, 0.0)
        assert abs(p_a - 1.0) < 0.1 and abs(p_b - 1.0) < 0.1

    def test_quantum_strategy_advantage(self):
        """Quantum strategies should give better payoff at gamma=pi/2."""
        # Both use quantum strategy in fully quantum game
        p_a, p_b = quantum_payoff(PD_PAYOFF, QUANTUM_STRATEGY, QUANTUM_STRATEGY, np.pi/2)
        # Should achieve cooperation-like payoff (3, 3)
        assert p_a > 1.5 and p_b > 1.5  # Better than Nash (1, 1)


class TestQuantumAdvantage:
    """Test quantum advantage calculation."""

    def test_classical_game_advantage(self):
        """Classical game with quantum strategy still achieves cooperation."""
        classical = GAME_SYSTEMS["classical_pd"]
        advantage = compute_quantum_advantage(classical)
        # Even at gamma=0, quantum strategy can achieve (3,3)
        # because the strategy operators still work on the state
        assert advantage >= 1.0

    def test_quantum_game_has_advantage(self):
        """Full quantum game should have advantage > 1."""
        quantum = GAME_SYSTEMS["quantum_pd"]
        advantage = compute_quantum_advantage(quantum)
        # Quantum can achieve (3,3), so advantage = 3
        assert advantage > 1.5


class TestNashEquilibrium:
    """Test Nash equilibrium finding."""

    def test_pd_nash_is_defect(self):
        """Prisoner's Dilemma Nash equilibrium is mutual defection."""
        nash = find_classical_nash(PD_PAYOFF)
        assert (1, 1) in nash  # Both defect

    def test_bos_has_two_nash(self):
        """Battle of Sexes has two pure Nash equilibria."""
        nash = find_classical_nash(BOS_PAYOFF)
        # Should have (0,0) and (1,1) - both at same event
        assert len(nash) == 2
        assert (0, 0) in nash
        assert (1, 1) in nash


class TestPayoffMatrices:
    """Test predefined payoff matrices."""

    def test_pd_payoff_shape(self):
        """PD payoff matrix should be (2, 2, 2)."""
        assert PD_PAYOFF.shape == (2, 2, 2)

    def test_chicken_payoff_shape(self):
        """Chicken payoff matrix should be (2, 2, 2)."""
        assert CHICKEN_PAYOFF.shape == (2, 2, 2)

    def test_bos_payoff_shape(self):
        """BOS payoff matrix should be (2, 2, 2)."""
        assert BOS_PAYOFF.shape == (2, 2, 2)

    def test_chicken_crash_payoff(self):
        """Chicken: both straight (crash) should be (-10, -10)."""
        assert CHICKEN_PAYOFF[1, 1, 0] == -10
        assert CHICKEN_PAYOFF[1, 1, 1] == -10


class TestGameTypes:
    """Test game type assignments."""

    def test_pd_games_typed_correctly(self):
        """PD games should have correct type."""
        assert GAME_SYSTEMS["classical_pd"].game_type == GameType.PRISONERS_DILEMMA
        assert GAME_SYSTEMS["quantum_pd"].game_type == GameType.PRISONERS_DILEMMA

    def test_chicken_games_typed_correctly(self):
        """Chicken games should have correct type."""
        assert GAME_SYSTEMS["classical_chicken"].game_type == GameType.CHICKEN
        assert GAME_SYSTEMS["quantum_chicken"].game_type == GameType.CHICKEN

    def test_bos_games_typed_correctly(self):
        """BOS games should have correct type."""
        assert GAME_SYSTEMS["classical_bos"].game_type == GameType.BATTLE_OF_SEXES
        assert GAME_SYSTEMS["quantum_bos"].game_type == GameType.BATTLE_OF_SEXES


class TestThetaOrdering:
    """Test that theta increases with gamma."""

    def test_pd_theta_ordering(self):
        """Theta should increase: classical < partial < quantum."""
        classical = GAME_SYSTEMS["classical_pd"]
        partial = GAME_SYSTEMS["partial_quantum_pd"]
        quantum = GAME_SYSTEMS["quantum_pd"]

        theta_classical = compute_entanglement_theta(classical)
        theta_partial = compute_entanglement_theta(partial)
        theta_quantum = compute_entanglement_theta(quantum)

        assert theta_classical < theta_partial < theta_quantum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
