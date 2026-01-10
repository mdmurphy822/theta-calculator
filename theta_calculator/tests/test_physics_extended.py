"""
Tests for Physics Extended Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- GR regime classification
- Component theta calculations
- Physical constants
"""

import pytest
import numpy as np

from theta_calculator.domains.physics_extended import (
    PhysicsExtendedSystem,
    GRRegime,
    HEPTheoryRegime,
    BSMScale,
    QuantumGravityRegime,
    SpacetimeType,
    compute_gr_theta,
    compute_hep_th_theta,
    compute_hep_ph_theta,
    compute_quantum_gravity_theta,
    compute_holographic_theta,
    compute_physics_extended_theta,
    classify_gr_regime,
    classify_hep_th_regime,
    classify_bsm_scale,
    schwarzschild_radius,
    hawking_temperature,
    PHYSICS_EXTENDED_SYSTEMS,
    G,
    C,
    HBAR,
    M_SUN,
    L_PLANCK,
    E_PLANCK_EV,
)


class TestPhysicsExtendedSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """PHYSICS_EXTENDED_SYSTEMS dict should exist."""
        assert PHYSICS_EXTENDED_SYSTEMS is not None
        assert isinstance(PHYSICS_EXTENDED_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(PHYSICS_EXTENDED_SYSTEMS) >= 5

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = [
            "earth_surface",
            "lhc_collision",
            "black_hole_horizon",
            "planck_regime",
        ]
        for name in expected:
            assert name in PHYSICS_EXTENDED_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in PHYSICS_EXTENDED_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "domain")
            assert hasattr(system, "length_scale")
            assert hasattr(system, "energy_scale")


class TestGRTheta:
    """Test general relativity theta calculation."""

    def test_far_from_mass_low_theta(self):
        """Far from massive object -> low theta."""
        # Earth at 1 AU from Sun
        theta = compute_gr_theta(1.5e11, M_SUN)
        assert theta < 0.01

    def test_near_horizon_high_theta(self):
        """Near black hole horizon -> high theta."""
        # At 1.5 * r_s
        r_s = schwarzschild_radius(10 * M_SUN)
        theta = compute_gr_theta(1.5 * r_s, 10 * M_SUN)
        assert theta > 0.5

    def test_at_horizon_theta_one(self):
        """At Schwarzschild radius -> theta = 1."""
        r_s = schwarzschild_radius(M_SUN)
        theta = compute_gr_theta(r_s, M_SUN)
        assert theta == 1.0

    def test_zero_radius_theta_one(self):
        """r = 0 -> theta = 1.0."""
        theta = compute_gr_theta(0.0, M_SUN)
        assert theta == 1.0

    def test_theta_always_in_range(self):
        """Theta should always be in [0, 1]."""
        test_cases = [
            (1e10, M_SUN),
            (1e3, 10 * M_SUN),
            (1e20, M_SUN),
        ]
        for r, m in test_cases:
            theta = compute_gr_theta(r, m)
            assert 0.0 <= theta <= 1.0


class TestHEPThTheta:
    """Test high-energy theory theta calculation."""

    def test_weak_coupling_low_theta(self):
        """g << 1 -> low theta."""
        theta = compute_hep_th_theta(0.1, critical_coupling=1.0)
        assert theta == pytest.approx(0.1)

    def test_strong_coupling_high_theta(self):
        """g ~ 1 -> theta = 1."""
        theta = compute_hep_th_theta(1.0, critical_coupling=1.0)
        assert theta == 1.0

    def test_supercritical_capped(self):
        """g > g_c -> theta capped at 1."""
        theta = compute_hep_th_theta(2.0, critical_coupling=1.0)
        assert theta == 1.0


class TestHEPPhTheta:
    """Test high-energy phenomenology theta calculation."""

    def test_low_energy_low_theta(self):
        """E << Lambda_BSM -> low theta."""
        theta = compute_hep_ph_theta(1e9, bsm_scale_ev=1e13)  # 1 GeV vs 10 TeV
        assert theta < 0.01

    def test_at_bsm_scale_theta_one(self):
        """E = Lambda_BSM -> theta = 1."""
        theta = compute_hep_ph_theta(1e13, bsm_scale_ev=1e13)
        assert theta == 1.0

    def test_lhc_energy(self):
        """LHC energy should give moderate theta."""
        theta = compute_hep_ph_theta(13.6e12, bsm_scale_ev=1e13)  # 13.6 TeV
        assert theta > 0.5


class TestQuantumGravityTheta:
    """Test quantum gravity theta calculation."""

    def test_macroscopic_low_theta(self):
        """Large length scale -> low theta."""
        theta = compute_quantum_gravity_theta(1.0)  # 1 meter
        assert theta < 1e-30

    def test_planck_scale_high_theta(self):
        """Planck length -> theta = 1."""
        theta = compute_quantum_gravity_theta(L_PLANCK)
        assert theta == 1.0

    def test_near_planck(self):
        """10 * Planck length -> theta = 0.1."""
        theta = compute_quantum_gravity_theta(10 * L_PLANCK)
        assert theta == pytest.approx(0.1)


class TestHolographicTheta:
    """Test holographic entropy theta."""

    def test_saturated_bound(self):
        """S = S_max -> theta = 1."""
        area = 4 * np.pi * (1e4)**2  # 10 km radius sphere
        s_max = area / (4 * L_PLANCK**2)
        theta = compute_holographic_theta(area, s_max)
        assert theta == 1.0

    def test_low_entropy(self):
        """S << S_max -> low theta."""
        area = 1.0  # 1 m^2
        entropy = 1e10  # Much less than Bekenstein bound
        theta = compute_holographic_theta(area, entropy)
        assert theta < 1e-50


class TestUnifiedTheta:
    """Test unified theta calculation for systems."""

    def test_all_systems_have_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in PHYSICS_EXTENDED_SYSTEMS.items():
            theta = compute_physics_extended_theta(system)
            assert 0.0 <= theta <= 1.0, f"{name}: theta={theta}"

    def test_earth_low_theta(self):
        """Earth surface should have low GR theta."""
        earth = PHYSICS_EXTENDED_SYSTEMS["earth_surface"]
        theta = compute_physics_extended_theta(earth)
        assert theta < 0.001

    def test_black_hole_high_theta(self):
        """Black hole horizon should have high theta."""
        bh = PHYSICS_EXTENDED_SYSTEMS["black_hole_horizon"]
        theta = compute_physics_extended_theta(bh)
        assert theta > 0.5


class TestGRRegimeClassification:
    """Test GR regime classification."""

    def test_flat(self):
        """theta < 0.01 -> FLAT."""
        assert classify_gr_regime(0.001) == GRRegime.FLAT

    def test_weak_field(self):
        """0.01 <= theta < 0.1 -> WEAK_FIELD."""
        assert classify_gr_regime(0.05) == GRRegime.WEAK_FIELD

    def test_strong_field(self):
        """0.1 <= theta < 0.5 -> STRONG_FIELD."""
        assert classify_gr_regime(0.3) == GRRegime.STRONG_FIELD

    def test_horizon(self):
        """theta >= 0.5 -> HORIZON."""
        assert classify_gr_regime(0.8) == GRRegime.HORIZON


class TestHEPThRegimeClassification:
    """Test HEP-TH regime classification."""

    def test_perturbative(self):
        """theta < 0.1 -> PERTURBATIVE."""
        assert classify_hep_th_regime(0.05) == HEPTheoryRegime.PERTURBATIVE

    def test_weakly_coupled(self):
        """0.1 <= theta < 0.5 -> WEAKLY_COUPLED."""
        assert classify_hep_th_regime(0.3) == HEPTheoryRegime.WEAKLY_COUPLED

    def test_moderately_coupled(self):
        """0.5 <= theta < 1.0 -> MODERATELY_COUPLED."""
        assert classify_hep_th_regime(0.7) == HEPTheoryRegime.MODERATELY_COUPLED

    def test_strongly_coupled(self):
        """theta >= 1.0 -> STRONGLY_COUPLED."""
        assert classify_hep_th_regime(1.5) == HEPTheoryRegime.STRONGLY_COUPLED


class TestBSMScaleClassification:
    """Test BSM scale classification."""

    def test_standard_model(self):
        """E < 100 GeV -> STANDARD_MODEL."""
        assert classify_bsm_scale(10e9) == BSMScale.STANDARD_MODEL

    def test_electroweak(self):
        """100 GeV - 10 TeV -> ELECTROWEAK."""
        assert classify_bsm_scale(1e12) == BSMScale.ELECTROWEAK

    def test_tev_scale(self):
        """10 TeV - 10 PeV -> TEV_SCALE."""
        assert classify_bsm_scale(1e14) == BSMScale.TEV_SCALE

    def test_gut_scale(self):
        """~10^16 GeV -> GUT_SCALE."""
        assert classify_bsm_scale(1e25) == BSMScale.GUT_SCALE

    def test_planck_scale(self):
        """~10^19 GeV -> PLANCK_SCALE."""
        assert classify_bsm_scale(1e28) == BSMScale.PLANCK_SCALE


class TestPhysicalConstants:
    """Test that physical constants are correct."""

    def test_gravitational_constant(self):
        """G should be Newton's constant."""
        assert G == pytest.approx(6.67430e-11, rel=1e-4)

    def test_speed_of_light(self):
        """C should be speed of light."""
        assert C == 299792458

    def test_planck_constant(self):
        """HBAR should be reduced Planck constant."""
        assert HBAR == pytest.approx(1.054571817e-34, rel=1e-6)

    def test_planck_length(self):
        """L_PLANCK should be Planck length."""
        assert L_PLANCK == pytest.approx(1.616255e-35, rel=1e-4)

    def test_solar_mass(self):
        """M_SUN should be solar mass."""
        assert M_SUN == pytest.approx(1.989e30, rel=1e-2)


class TestSchwarzschildRadius:
    """Test Schwarzschild radius calculation."""

    def test_solar_mass(self):
        """r_s(Sun) ~ 3 km."""
        r_s = schwarzschild_radius(M_SUN)
        assert r_s == pytest.approx(2953, rel=0.01)

    def test_earth_mass(self):
        """r_s(Earth) ~ 9 mm."""
        r_s = schwarzschild_radius(5.972e24)
        assert r_s == pytest.approx(0.00887, rel=0.01)


class TestHawkingTemperature:
    """Test Hawking temperature calculation."""

    def test_solar_mass_bh(self):
        """T_H(M_sun) ~ 60 nK."""
        T = hawking_temperature(M_SUN)
        assert T == pytest.approx(6.17e-8, rel=0.1)

    def test_small_bh_hotter(self):
        """Smaller black holes are hotter."""
        T_small = hawking_temperature(1e10)  # 10 billion kg
        T_large = hawking_temperature(M_SUN)
        assert T_small > T_large

    def test_zero_mass_infinite(self):
        """Zero mass -> infinite temperature."""
        T = hawking_temperature(0)
        assert T == float('inf')


class TestSystemDataclass:
    """Test PhysicsExtendedSystem dataclass."""

    def test_create_minimal_system(self):
        """Should create system with required parameters."""
        system = PhysicsExtendedSystem(
            name="Test",
            domain="gr-qc",
            length_scale=1e6,
            energy_scale=1e9,
        )
        assert system.name == "Test"
        assert system.domain == "gr-qc"

    def test_is_relativistic_property(self):
        """is_relativistic should check length vs r_s."""
        # Near black hole
        system = PhysicsExtendedSystem(
            name="Near BH",
            domain="gr-qc",
            length_scale=1e4,  # 10 km
            energy_scale=1e9,
            mass=10 * M_SUN,
        )
        assert system.is_relativistic is True

        # Far from mass
        system2 = PhysicsExtendedSystem(
            name="Far",
            domain="gr-qc",
            length_scale=1e12,  # 1000 km
            energy_scale=1e9,
            mass=M_SUN,
        )
        assert system2.is_relativistic is False

    def test_is_quantum_gravitational_property(self):
        """is_quantum_gravitational should check length vs Planck."""
        # Planck scale
        system = PhysicsExtendedSystem(
            name="Planck",
            domain="gr-qc",
            length_scale=L_PLANCK,
            energy_scale=E_PLANCK_EV,
        )
        assert system.is_quantum_gravitational is True

        # Macroscopic
        system2 = PhysicsExtendedSystem(
            name="Macro",
            domain="gr-qc",
            length_scale=1.0,
            energy_scale=1.0,
        )
        assert system2.is_quantum_gravitational is False


class TestEnums:
    """Test enum definitions."""

    def test_spacetime_types(self):
        """All spacetime types should be defined."""
        assert SpacetimeType.MINKOWSKI.value == "minkowski"
        assert SpacetimeType.SCHWARZSCHILD.value == "schwarzschild"
        assert SpacetimeType.KERR.value == "kerr"
        assert SpacetimeType.DE_SITTER.value == "de_sitter"
        assert SpacetimeType.FRIEDMANN.value == "friedmann"

    def test_qg_regimes(self):
        """All quantum gravity regimes should be defined."""
        assert QuantumGravityRegime.SEMICLASSICAL.value == "semiclassical"
        assert QuantumGravityRegime.PLANCKIAN.value == "planckian"
