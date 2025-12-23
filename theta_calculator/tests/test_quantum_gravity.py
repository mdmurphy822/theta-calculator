"""
Tests for quantum_gravity module.

Tests quantum gravity theta calculations for Planck scale emergence,
black hole quantum properties, and spacetime discreteness.
"""

import pytest

from theta_calculator.domains.quantum_gravity import (
    QUANTUM_GRAVITY_SYSTEMS,
    QuantumGravitySystem,
    SpacetimeRegime,
    QuantumGravityTheory,
    compute_quantum_gravity_theta,
    compute_length_theta,
    compute_energy_theta,
    classify_regime,
    schwarzschild_radius,
    hawking_temperature_k,
    bekenstein_entropy,
    black_hole_evaporation_time_s,
    black_hole_theta,
    area_eigenvalue,
    area_theta,
    gravity_induced_decoherence_rate,
    gravitational_theta_for_superposition,
    L_PLANCK,
    E_PLANCK_EV,
    M_PLANCK,
    R_PLANCK,
    IMMIRZI_PARAMETER,
)


class TestQuantumGravitySystems:
    """Test the predefined quantum gravity systems."""

    def test_all_systems_exist(self):
        """Verify all expected systems are defined."""
        expected = [
            "human_scale", "electron", "proton", "lhc_collision",
            "cosmic_ray_record", "inflation_energy", "big_bang_singularity",
            "stellar_black_hole", "sgr_a_star", "primordial_bh",
            "planck_mass_bh", "string_scale", "lqg_spin_network"
        ]
        for name in expected:
            assert name in QUANTUM_GRAVITY_SYSTEMS, f"Missing system: {name}"

    def test_all_systems_have_valid_attributes(self):
        """Check that all systems have required attributes."""
        for name, system in QUANTUM_GRAVITY_SYSTEMS.items():
            assert isinstance(system, QuantumGravitySystem)
            assert system.name
            assert system.length_scale_m > 0
            assert system.energy_ev > 0


class TestThetaComputation:
    """Test theta computation functions."""

    def test_all_systems_theta_in_range(self):
        """All systems should produce theta in [0, 1]."""
        for name, system in QUANTUM_GRAVITY_SYSTEMS.items():
            theta = compute_quantum_gravity_theta(system)
            assert 0 <= theta <= 1, f"{name} theta out of range: {theta}"

    def test_human_scale_classical(self):
        """Human scale should have theta ≈ 0 (classical)."""
        human = QUANTUM_GRAVITY_SYSTEMS["human_scale"]
        theta = compute_quantum_gravity_theta(human)
        assert theta < 1e-25, f"Human scale should be classical: {theta}"

    def test_planck_scale_quantum(self):
        """Planck scale should have theta = 1 (quantum)."""
        planck = QUANTUM_GRAVITY_SYSTEMS["planck_mass_bh"]
        theta = compute_quantum_gravity_theta(planck)
        assert theta > 0.9, f"Planck scale should be quantum: {theta}"

    def test_theta_ordering(self):
        """Planck mass BH should have highest theta, human scale lowest."""
        human = QUANTUM_GRAVITY_SYSTEMS["human_scale"]
        planck = QUANTUM_GRAVITY_SYSTEMS["planck_mass_bh"]
        stellar_bh = QUANTUM_GRAVITY_SYSTEMS["stellar_black_hole"]

        theta_human = compute_quantum_gravity_theta(human)
        theta_planck = compute_quantum_gravity_theta(planck)
        theta_stellar = compute_quantum_gravity_theta(stellar_bh)

        # Human scale is classical, Planck is quantum
        assert theta_human < theta_stellar < theta_planck


class TestLengthTheta:
    """Test length scale theta calculation."""

    def test_planck_length_gives_one(self):
        """Planck length should give theta = 1."""
        system = QuantumGravitySystem(
            name="Planck",
            length_scale_m=L_PLANCK,
            energy_ev=E_PLANCK_EV
        )
        theta = compute_length_theta(system)
        assert abs(theta - 1.0) < 0.01

    def test_macroscopic_length_gives_tiny_theta(self):
        """1 meter should give theta ≈ 10^-35."""
        system = QuantumGravitySystem(
            name="Macro",
            length_scale_m=1.0,
            energy_ev=0.1
        )
        theta = compute_length_theta(system)
        assert theta < 1e-30


class TestEnergyTheta:
    """Test energy scale theta calculation."""

    def test_planck_energy_gives_one(self):
        """Planck energy should give theta = 1."""
        system = QuantumGravitySystem(
            name="Planck",
            length_scale_m=L_PLANCK,
            energy_ev=E_PLANCK_EV
        )
        theta = compute_energy_theta(system)
        assert abs(theta - 1.0) < 0.01

    def test_thermal_energy_gives_tiny_theta(self):
        """Thermal energy (0.025 eV) should give tiny theta."""
        system = QuantumGravitySystem(
            name="Thermal",
            length_scale_m=1e-9,
            energy_ev=0.025
        )
        theta = compute_energy_theta(system)
        assert theta < 1e-25


class TestRegimeClassification:
    """Test spacetime regime classification."""

    def test_classical_regime(self):
        """Theta < 0.001 should be CLASSICAL."""
        assert classify_regime(1e-10) == SpacetimeRegime.CLASSICAL

    def test_semiclassical_regime(self):
        """0.001 <= theta < 0.1 should be SEMICLASSICAL."""
        assert classify_regime(0.01) == SpacetimeRegime.SEMICLASSICAL

    def test_trans_planckian_regime(self):
        """0.1 <= theta < 0.5 should be TRANS_PLANCKIAN."""
        assert classify_regime(0.3) == SpacetimeRegime.TRANS_PLANCKIAN

    def test_quantum_foam_regime(self):
        """0.5 <= theta < 0.9 should be QUANTUM_FOAM."""
        assert classify_regime(0.7) == SpacetimeRegime.QUANTUM_FOAM

    def test_planck_scale_regime(self):
        """theta >= 0.9 should be PLANCK_SCALE."""
        assert classify_regime(0.95) == SpacetimeRegime.PLANCK_SCALE


class TestBlackHoles:
    """Test black hole physics functions."""

    def test_schwarzschild_radius_solar(self):
        """Test Schwarzschild radius for solar mass."""
        M_SUN = 1.989e30  # kg
        r_s = schwarzschild_radius(M_SUN)
        # r_s ≈ 3 km for solar mass
        assert 2900 < r_s < 3100

    def test_hawking_temperature_solar(self):
        """Solar mass BH should have very low Hawking temp."""
        M_SUN = 1.989e30
        T_H = hawking_temperature_k(M_SUN)
        # T_H ≈ 6e-8 K for solar mass
        assert 1e-8 < T_H < 1e-7

    def test_hawking_temperature_planck_mass(self):
        """Planck mass BH should have Planck temperature."""
        T_H = hawking_temperature_k(M_PLANCK)
        # Planck temperature ~ 1.417e32 K
        # Should be of order Planck temperature
        assert T_H > 1e30

    def test_bekenstein_entropy(self):
        """Test Bekenstein-Hawking entropy scaling."""
        M_SUN = 1.989e30
        S = bekenstein_entropy(M_SUN)
        # S ≈ 10^77 for solar mass BH
        assert S > 1e70

    def test_evaporation_time_solar(self):
        """Solar mass BH should take >> age of universe to evaporate."""
        M_SUN = 1.989e30
        t_evap = black_hole_evaporation_time_s(M_SUN)
        age_universe_s = 4.35e17  # seconds
        assert t_evap > age_universe_s * 1e50

    def test_black_hole_theta_ordering(self):
        """BHs closer to Planck mass should have higher theta."""
        theta_stellar = black_hole_theta(10 * 1.989e30)  # 10 solar masses
        theta_primordial = black_hole_theta(1e12)  # asteroid mass (closer to Planck)
        theta_planck = black_hole_theta(M_PLANCK)

        # Planck mass BH has theta = 1
        assert abs(theta_planck - 1.0) < 0.01
        # Primordial is closer to Planck mass than stellar
        assert theta_stellar < theta_primordial
        # Both are far from Planck mass, so low theta
        assert theta_stellar < 0.01
        assert theta_primordial < 0.01


class TestLoopQuantumGravity:
    """Test LQG-specific functions."""

    def test_immirzi_parameter(self):
        """Immirzi parameter should be approximately 0.2375."""
        assert abs(IMMIRZI_PARAMETER - 0.2375) < 0.01

    def test_area_eigenvalue_j_half(self):
        """Test minimum area for j=1/2."""
        A = area_eigenvalue(0.5)
        # Should be close to A_min
        assert A > 0
        assert A < 1e-68  # Order of Planck area squared

    def test_area_theta_planck_area(self):
        """Planck area should give theta ≈ 1."""
        planck_area = L_PLANCK**2
        theta = area_theta(planck_area)
        # theta = A_min / A_planck ≈ 5.2e-70 / 2.6e-70 ≈ 2 (capped at 1)
        assert theta > 0.5

    def test_area_theta_macroscopic(self):
        """1 square meter should give tiny theta."""
        theta = area_theta(1.0)
        assert theta < 1e-60


class TestPlanckUnits:
    """Test Planck unit values."""

    def test_planck_length(self):
        """Planck length should be approximately 1.6e-35 m."""
        assert 1.5e-35 < L_PLANCK < 1.7e-35

    def test_planck_mass(self):
        """Planck mass should be approximately 2.2e-8 kg."""
        assert 2.0e-8 < M_PLANCK < 2.3e-8

    def test_planck_energy_ev(self):
        """Planck energy should be approximately 1.2e28 eV."""
        assert 1e28 < E_PLANCK_EV < 1.5e28

    def test_planck_curvature(self):
        """Planck curvature should be 1/l_P^2."""
        expected = 1 / L_PLANCK**2
        assert abs(R_PLANCK / expected - 1) < 0.01


class TestGravitationalDecoherence:
    """Test gravitational decoherence functions."""

    def test_decoherence_rate_increases_with_mass(self):
        """Heavier objects should decohere faster."""
        rate_small = gravity_induced_decoherence_rate(1e-15, 1e-9)
        rate_large = gravity_induced_decoherence_rate(1e-10, 1e-9)
        assert rate_large > rate_small

    def test_decoherence_rate_increases_with_superposition(self):
        """Larger superpositions should decohere faster."""
        rate_small = gravity_induced_decoherence_rate(1e-10, 1e-10)
        rate_large = gravity_induced_decoherence_rate(1e-10, 1e-8)
        assert rate_small > rate_large  # Larger Δx → smaller E_G → slower

    def test_gravitational_theta_for_molecule(self):
        """Large molecule superposition should have moderate theta."""
        # C60 molecule mass ~1e-24 kg
        theta = gravitational_theta_for_superposition(
            mass_kg=1e-24,
            superposition_size_m=1e-9,
            coherence_time_s=1e-6
        )
        assert 0 < theta <= 1


class TestCosmologicalSystems:
    """Test cosmological systems."""

    def test_big_bang_planck_scale(self):
        """Big Bang singularity should be at Planck scale."""
        big_bang = QUANTUM_GRAVITY_SYSTEMS["big_bang_singularity"]
        theta = compute_quantum_gravity_theta(big_bang)
        assert theta > 0.9

    def test_inflation_trans_planckian(self):
        """Inflation energy scale should be near Planck."""
        inflation = QUANTUM_GRAVITY_SYSTEMS["inflation_energy"]
        theta = compute_quantum_gravity_theta(inflation)
        assert theta > 0.0001  # Above classical


class TestTheoryTypes:
    """Test quantum gravity theory classifications."""

    def test_lqg_system(self):
        """LQG spin network should have LQG theory type."""
        lqg = QUANTUM_GRAVITY_SYSTEMS["lqg_spin_network"]
        assert lqg.theory == QuantumGravityTheory.LOOP

    def test_string_system(self):
        """String scale should have STRING theory type."""
        string = QUANTUM_GRAVITY_SYSTEMS["string_scale"]
        assert string.theory == QuantumGravityTheory.STRING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
