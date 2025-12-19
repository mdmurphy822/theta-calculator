"""Tests for ThetaState and PhysicalSystem data classes."""

import numpy as np
import pytest

from theta_calculator.core.theta_state import (
    ThetaState, PhysicalSystem, Regime, ThetaTrajectory, EXAMPLE_SYSTEMS
)


class TestPhysicalSystem:
    """Test PhysicalSystem data class."""

    def test_create_system(self):
        """Should create a valid physical system."""
        system = PhysicalSystem(
            name="test",
            mass=1.0,
            length_scale=1.0,
            energy=1.0,
            temperature=300.0
        )
        assert system.name == "test"
        assert system.mass == 1.0
        assert system.length_scale == 1.0

    def test_negative_mass_raises(self):
        """Negative mass should raise ValueError."""
        with pytest.raises(ValueError):
            PhysicalSystem(
                name="invalid",
                mass=-1.0,
                length_scale=1.0,
                energy=1.0,
                temperature=300.0
            )

    def test_negative_length_raises(self):
        """Non-positive length should raise ValueError."""
        with pytest.raises(ValueError):
            PhysicalSystem(
                name="invalid",
                mass=1.0,
                length_scale=0.0,
                energy=1.0,
                temperature=300.0
            )

    def test_rest_energy(self):
        """Rest energy should be mc²."""
        from theta_calculator.constants.values import c
        system = PhysicalSystem(
            name="test",
            mass=1.0,
            length_scale=1.0,
            energy=1.0,
            temperature=300.0
        )
        assert np.isclose(system.rest_energy, c**2)

    def test_schwarzschild_radius(self):
        """Schwarzschild radius should be 2GM/c²."""
        from theta_calculator.constants.values import G, c
        system = PhysicalSystem(
            name="test",
            mass=1e30,  # ~solar mass
            length_scale=1.0,
            energy=1.0,
            temperature=300.0
        )
        expected = 2 * G * 1e30 / c**2
        assert np.isclose(system.schwarzschild_radius, expected, rtol=1e-6)

    def test_estimate_action(self):
        """Action estimate should work."""
        from theta_calculator.constants.values import c
        system = PhysicalSystem(
            name="test",
            mass=1.0,
            length_scale=1.0,
            energy=1e10,
            temperature=300.0
        )
        action = system.estimate_action()
        # S = E × t ≈ E × L/c
        expected = 1e10 * (1.0 / c)
        assert np.isclose(action, expected, rtol=1e-6)


class TestThetaState:
    """Test ThetaState data class."""

    def test_theta_clamping_high(self):
        """Theta > 1 should be clamped to 1."""
        state = ThetaState(theta=1.5)
        assert state.theta == 1.0

    def test_theta_clamping_low(self):
        """Theta < 0 should be clamped to 0."""
        state = ThetaState(theta=-0.5)
        assert state.theta == 0.0

    def test_regime_quantum(self):
        """Theta > 0.99 should classify as QUANTUM."""
        state = ThetaState(theta=0.999)
        assert state.regime == Regime.QUANTUM
        assert state.is_quantum

    def test_regime_classical(self):
        """Theta < 0.01 should classify as CLASSICAL."""
        state = ThetaState(theta=0.001)
        assert state.regime == Regime.CLASSICAL
        assert state.is_classical

    def test_regime_transition(self):
        """0.01 < theta < 0.99 should classify as TRANSITION."""
        state = ThetaState(theta=0.5)
        assert state.regime == Regime.TRANSITION
        assert state.is_transitional

    def test_interpolate(self):
        """Interpolation should work correctly."""
        state = ThetaState(theta=0.3)
        result = state.interpolate(quantum_value=10.0, classical_value=0.0)
        assert np.isclose(result, 3.0)

    def test_interpolate_midpoint(self):
        """θ=0.5 should give midpoint."""
        state = ThetaState(theta=0.5)
        result = state.interpolate(quantum_value=100.0, classical_value=0.0)
        assert np.isclose(result, 50.0)

    def test_quantum_fraction(self):
        """Quantum fraction should be theta * 100."""
        state = ThetaState(theta=0.75)
        assert np.isclose(state.quantum_fraction, 75.0)

    def test_classical_fraction(self):
        """Classical fraction should be (1-theta) * 100."""
        state = ThetaState(theta=0.75)
        assert np.isclose(state.classical_fraction, 25.0)

    def test_gradient_to(self):
        """Gradient should compute difference in theta."""
        state1 = ThetaState(theta=0.3)
        state2 = ThetaState(theta=0.7)
        assert np.isclose(state1.gradient_to(state2), 0.4)

    def test_merge_with(self):
        """Merging states should compute weighted average."""
        state1 = ThetaState(theta=0.2)
        state2 = ThetaState(theta=0.8)
        merged = state1.merge_with(state2, weight_self=0.5)
        assert np.isclose(merged.theta, 0.5)

    def test_to_dict(self):
        """to_dict should return dictionary."""
        state = ThetaState(theta=0.5, proof_method="test")
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d['theta'] == 0.5
        assert d['proof_method'] == "test"


class TestThetaTrajectory:
    """Test ThetaTrajectory data class."""

    def test_create_trajectory(self):
        """Should create valid trajectory."""
        states = [ThetaState(theta=t) for t in [0.1, 0.3, 0.5, 0.7, 0.9]]
        params = [1.0, 2.0, 3.0, 4.0, 5.0]
        trajectory = ThetaTrajectory(
            states=states,
            parameter_name="time",
            parameter_values=params
        )
        assert len(trajectory) == 5

    def test_mismatched_length_raises(self):
        """Mismatched state and parameter counts should raise."""
        states = [ThetaState(theta=0.5)]
        params = [1.0, 2.0]
        with pytest.raises(ValueError):
            ThetaTrajectory(states=states, parameter_name="x", parameter_values=params)

    def test_thetas_property(self):
        """thetas property should return numpy array."""
        states = [ThetaState(theta=t) for t in [0.1, 0.5, 0.9]]
        trajectory = ThetaTrajectory(
            states=states,
            parameter_name="x",
            parameter_values=[1, 2, 3]
        )
        thetas = trajectory.thetas
        assert isinstance(thetas, np.ndarray)
        assert np.isclose(thetas[0], 0.1)

    def test_find_transitions(self):
        """Should find points where theta changes rapidly."""
        # Create trajectory with sharp transition
        states = [ThetaState(theta=t) for t in [0.1, 0.1, 0.9, 0.9]]
        trajectory = ThetaTrajectory(
            states=states,
            parameter_name="x",
            parameter_values=[1, 2, 3, 4]
        )
        transitions = trajectory.find_transitions(threshold=0.5)
        assert 2 in transitions  # Transition at index 2

    def test_mean_theta(self):
        """Should compute mean theta."""
        states = [ThetaState(theta=t) for t in [0.2, 0.4, 0.6]]
        trajectory = ThetaTrajectory(
            states=states,
            parameter_name="x",
            parameter_values=[1, 2, 3]
        )
        assert np.isclose(trajectory.mean_theta(), 0.4)


class TestExampleSystems:
    """Test predefined example systems."""

    def test_example_systems_exist(self):
        """EXAMPLE_SYSTEMS should contain standard systems."""
        assert "electron" in EXAMPLE_SYSTEMS
        assert "proton" in EXAMPLE_SYSTEMS
        assert "hydrogen_atom" in EXAMPLE_SYSTEMS
        assert "baseball" in EXAMPLE_SYSTEMS
        assert "human" in EXAMPLE_SYSTEMS

    def test_electron_mass(self):
        """Electron mass should be approximately correct."""
        electron = EXAMPLE_SYSTEMS["electron"]
        assert np.isclose(electron.mass, 9.109e-31, rtol=1e-2)

    def test_all_systems_valid(self):
        """All example systems should be valid PhysicalSystem objects."""
        for name, system in EXAMPLE_SYSTEMS.items():
            assert isinstance(system, PhysicalSystem)
            assert system.mass > 0
            assert system.length_scale > 0
            assert system.energy > 0
            assert system.temperature >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
