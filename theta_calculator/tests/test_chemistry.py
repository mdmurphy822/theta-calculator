"""
Tests for Chemistry Domain Module

Tests cover:
- Superconductor systems and BCS theory
- Bose-Einstein condensation
- Quantum dot confinement
- Superfluidity
- Theta range validation [0, 1]
- Physical constants
"""

import pytest
import numpy as np

from theta_calculator.domains.chemistry import (
    QuantumMaterial,
    MaterialPhase,
    SuperconductorType,
    compute_chemistry_theta,
    compute_superconductor_theta,
    compute_bec_theta,
    compute_quantum_dot_theta,
    compute_superfluid_theta,
    bcs_gap_zero_temp,
    bcs_gap_temperature,
    bec_critical_temperature,
    bec_condensate_fraction,
    quantum_dot_confinement_energy,
    lambda_transition_exponent,
    ginzburg_landau_kappa,
    classify_superconductor,
    classify_material_phase,
    SUPERCONDUCTORS,
    BEC_EXAMPLES,
    QUANTUM_DOT_EXAMPLES,
    K_B,
    HBAR,
    M_ELECTRON,
    E_CHARGE,
)


class TestSuperconductorsExist:
    """Test that example superconductors are properly defined."""

    def test_superconductors_dictionary_exists(self):
        """SUPERCONDUCTORS dict should exist."""
        assert SUPERCONDUCTORS is not None
        assert isinstance(SUPERCONDUCTORS, dict)

    def test_minimum_superconductors_count(self):
        """Should have at least 5 superconductors."""
        assert len(SUPERCONDUCTORS) >= 5

    def test_key_superconductors_defined(self):
        """Key superconductors should be defined."""
        expected = ["aluminum", "niobium", "YBCO", "MgB2", "lead"]
        for name in expected:
            assert name in SUPERCONDUCTORS, f"Missing superconductor: {name}"

    def test_superconductor_attributes(self):
        """Superconductors should have required attributes."""
        for name, mat in SUPERCONDUCTORS.items():
            assert hasattr(mat, "name"), f"{name} missing 'name'"
            assert hasattr(mat, "critical_temperature")
            assert hasattr(mat, "current_temperature")
            assert mat.critical_temperature > 0, f"{name}: T_c must be positive"


class TestBECExamples:
    """Test BEC examples."""

    def test_bec_examples_exist(self):
        """BEC_EXAMPLES dict should exist."""
        assert BEC_EXAMPLES is not None
        assert isinstance(BEC_EXAMPLES, dict)

    def test_bec_examples_have_required_keys(self):
        """Each BEC example should have T and T_c."""
        for name, bec in BEC_EXAMPLES.items():
            assert "T" in bec, f"{name} missing 'T'"
            assert "T_c" in bec, f"{name} missing 'T_c'"


class TestQuantumDotExamples:
    """Test quantum dot examples."""

    def test_qd_examples_exist(self):
        """QUANTUM_DOT_EXAMPLES dict should exist."""
        assert QUANTUM_DOT_EXAMPLES is not None
        assert isinstance(QUANTUM_DOT_EXAMPLES, dict)

    def test_qd_examples_have_required_keys(self):
        """Each QD example should have size and T."""
        for name, qd in QUANTUM_DOT_EXAMPLES.items():
            assert "size" in qd, f"{name} missing 'size'"
            assert "T" in qd, f"{name} missing 'T'"


class TestBCSGapZeroTemp:
    """Test zero-temperature BCS gap calculation."""

    def test_typical_tc(self):
        """Test BCS gap for typical T_c."""
        T_c = 9.3  # Niobium
        Delta_0 = bcs_gap_zero_temp(T_c)
        # Delta_0 = 1.76 * k_B * T_c
        expected = 1.76 * K_B * T_c
        assert Delta_0 == pytest.approx(expected)

    def test_high_tc(self):
        """Test BCS gap for high-T_c material."""
        T_c = 92.0  # YBCO
        Delta_0 = bcs_gap_zero_temp(T_c)
        assert Delta_0 > bcs_gap_zero_temp(9.3)

    def test_zero_tc(self):
        """T_c = 0 gives zero gap."""
        assert bcs_gap_zero_temp(0) == 0.0


class TestBCSGapTemperature:
    """Test temperature-dependent BCS gap."""

    def test_zero_temp(self):
        """At T=0, gap equals Delta_0."""
        T_c = 9.3
        Delta_0 = bcs_gap_zero_temp(T_c)
        gap = bcs_gap_temperature(0, T_c, Delta_0)
        assert gap == Delta_0

    def test_at_tc(self):
        """At T=T_c, gap vanishes."""
        T_c = 9.3
        Delta_0 = bcs_gap_zero_temp(T_c)
        gap = bcs_gap_temperature(T_c, T_c, Delta_0)
        assert gap == 0.0

    def test_above_tc(self):
        """Above T_c, gap is zero."""
        T_c = 9.3
        Delta_0 = bcs_gap_zero_temp(T_c)
        gap = bcs_gap_temperature(15.0, T_c, Delta_0)
        assert gap == 0.0

    def test_intermediate_temp(self):
        """Intermediate T gives intermediate gap."""
        T_c = 9.3
        Delta_0 = bcs_gap_zero_temp(T_c)
        gap = bcs_gap_temperature(4.2, T_c, Delta_0)
        assert 0 < gap < Delta_0


class TestSuperconductorTheta:
    """Test superconductor theta calculation."""

    def test_zero_temp(self):
        """At T=0, theta = 1."""
        theta = compute_superconductor_theta(0, 9.3)
        assert theta == 1.0

    def test_at_tc(self):
        """At T=T_c, theta approaches 1 (gap ratio)."""
        theta = compute_superconductor_theta(9.3, 9.3)
        # At exactly T_c, we're in normal state with ratio = 1
        assert theta == pytest.approx(1.0)

    def test_above_tc(self):
        """Above T_c, theta = T_c/T < 1."""
        T = 18.6  # 2 * T_c
        T_c = 9.3
        theta = compute_superconductor_theta(T, T_c)
        assert theta == pytest.approx(0.5)

    def test_far_above_tc(self):
        """Far above T_c, theta approaches 0."""
        theta = compute_superconductor_theta(1000, 9.3)
        assert theta < 0.02

    def test_theta_in_range(self):
        """Theta should always be in [0, 1]."""
        for T in [0, 1, 5, 10, 50, 100, 300]:
            theta = compute_superconductor_theta(T, 9.3)
            assert 0 <= theta <= 1

    def test_all_superconductors_valid_theta(self):
        """All example superconductors should have theta in [0, 1]."""
        for name, mat in SUPERCONDUCTORS.items():
            theta = compute_superconductor_theta(
                mat.current_temperature,
                mat.critical_temperature
            )
            assert 0 <= theta <= 1, f"{name}: theta={theta}"


class TestGinzburgLandauKappa:
    """Test Ginzburg-Landau parameter."""

    def test_type_I(self):
        """Aluminum is Type I (kappa < 1/sqrt(2))."""
        al = SUPERCONDUCTORS["aluminum"]
        kappa = ginzburg_landau_kappa(al.coherence_length, al.penetration_depth)
        threshold = 1 / np.sqrt(2)
        assert kappa < threshold

    def test_type_II(self):
        """YBCO is Type II (kappa > 1/sqrt(2))."""
        ybco = SUPERCONDUCTORS["YBCO"]
        kappa = ginzburg_landau_kappa(ybco.coherence_length, ybco.penetration_depth)
        threshold = 1 / np.sqrt(2)
        assert kappa > threshold

    def test_zero_coherence_length(self):
        """Zero coherence length gives infinity."""
        kappa = ginzburg_landau_kappa(0, 100e-9)
        assert kappa == float('inf')


class TestClassifySuperconductor:
    """Test superconductor type classification."""

    def test_type_I(self):
        """kappa < 1/sqrt(2) -> TYPE_I."""
        kappa = 0.1
        assert classify_superconductor(kappa) == SuperconductorType.TYPE_I

    def test_type_II(self):
        """kappa > 1/sqrt(2) -> TYPE_II."""
        kappa = 10.0
        assert classify_superconductor(kappa) == SuperconductorType.TYPE_II

    def test_boundary(self):
        """kappa = 1/sqrt(2) is boundary (goes to TYPE_II)."""
        kappa = 1 / np.sqrt(2)
        # At boundary, we get TYPE_II (not strictly less than)
        assert classify_superconductor(kappa) == SuperconductorType.TYPE_II


class TestBECCriticalTemperature:
    """Test BEC critical temperature calculation."""

    def test_typical_density(self):
        """Test T_c for typical atomic BEC density."""
        n = 1e19  # atoms/m^3
        mass = 87 * 1.66e-27  # Rb-87 mass
        T_c = bec_critical_temperature(n, mass)
        # Should be in microKelvin range
        assert 1e-9 < T_c < 1e-5

    def test_higher_density_higher_tc(self):
        """Higher density gives higher T_c."""
        mass = 87 * 1.66e-27
        T_c_low = bec_critical_temperature(1e18, mass)
        T_c_high = bec_critical_temperature(1e20, mass)
        assert T_c_high > T_c_low

    def test_lighter_mass_higher_tc(self):
        """Lighter mass gives higher T_c."""
        n = 1e19
        T_c_heavy = bec_critical_temperature(n, 87 * 1.66e-27)
        T_c_light = bec_critical_temperature(n, 23 * 1.66e-27)  # Na-23
        assert T_c_light > T_c_heavy


class TestBECCondensateFraction:
    """Test BEC condensate fraction calculation."""

    def test_zero_temp(self):
        """At T=0, fraction = 1."""
        fraction = bec_condensate_fraction(0, 170e-9)
        assert fraction == 1.0

    def test_at_tc(self):
        """At T=T_c, fraction = 0."""
        fraction = bec_condensate_fraction(170e-9, 170e-9)
        assert fraction == 0.0

    def test_above_tc(self):
        """Above T_c, fraction = 0."""
        fraction = bec_condensate_fraction(500e-9, 170e-9)
        assert fraction == 0.0

    def test_intermediate(self):
        """Intermediate T gives intermediate fraction."""
        fraction = bec_condensate_fraction(100e-9, 170e-9)
        assert 0 < fraction < 1


class TestBECTheta:
    """Test BEC theta calculation."""

    def test_all_bec_examples_valid_theta(self):
        """All BEC examples should have theta in [0, 1]."""
        for name, bec in BEC_EXAMPLES.items():
            theta = compute_bec_theta(bec["T"], bec["T_c"])
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_cold_bec_high_theta(self):
        """Cold BEC (T << T_c) should have high theta."""
        theta = compute_bec_theta(10e-9, 170e-9)
        assert theta > 0.9

    def test_warm_bec_low_theta(self):
        """Warm thermal cloud (T > T_c) should have zero theta."""
        theta = compute_bec_theta(500e-9, 170e-9)
        assert theta == 0.0


class TestQuantumDotConfinementEnergy:
    """Test quantum dot confinement energy."""

    def test_smaller_dot_higher_energy(self):
        """Smaller dot has higher confinement energy."""
        E_small = quantum_dot_confinement_energy(5e-9)
        E_large = quantum_dot_confinement_energy(20e-9)
        assert E_small > E_large

    def test_positive_energy(self):
        """Energy should be positive."""
        E = quantum_dot_confinement_energy(10e-9)
        assert E > 0

    def test_inverse_square_scaling(self):
        """Energy scales as 1/L^2."""
        E_1 = quantum_dot_confinement_energy(10e-9)
        E_2 = quantum_dot_confinement_energy(20e-9)
        # E_1 / E_2 = (L_2 / L_1)^2 = 4
        ratio = E_1 / E_2
        assert ratio == pytest.approx(4.0)


class TestQuantumDotTheta:
    """Test quantum dot theta calculation."""

    def test_zero_temp(self):
        """At T=0, theta = 1."""
        theta = compute_quantum_dot_theta(10e-9, 0)
        assert theta == 1.0

    def test_small_dot_cold(self):
        """Small dot at low T has high theta."""
        theta = compute_quantum_dot_theta(5e-9, 4.2)
        assert theta > 0.5

    def test_large_dot_warm(self):
        """Large dot at room temp has low theta."""
        theta = compute_quantum_dot_theta(50e-9, 300)
        assert theta < 0.5

    def test_all_qd_examples_valid_theta(self):
        """All QD examples should have theta in [0, 1]."""
        for name, qd in QUANTUM_DOT_EXAMPLES.items():
            theta = compute_quantum_dot_theta(qd["size"], qd["T"])
            assert 0 <= theta <= 1, f"{name}: theta={theta}"


class TestLambdaTransition:
    """Test superfluid lambda transition."""

    def test_zero_temp(self):
        """At T=0, superfluid fraction = 1."""
        fraction = lambda_transition_exponent(0, 2.17)
        assert fraction == 1.0

    def test_at_lambda(self):
        """At T=T_lambda, superfluid fraction = 0."""
        fraction = lambda_transition_exponent(2.17, 2.17)
        assert fraction == 0.0

    def test_above_lambda(self):
        """Above T_lambda, superfluid fraction = 0."""
        fraction = lambda_transition_exponent(4.0, 2.17)
        assert fraction == 0.0

    def test_intermediate(self):
        """Intermediate T gives intermediate fraction."""
        fraction = lambda_transition_exponent(1.5, 2.17)
        assert 0 < fraction < 1


class TestSuperfluidTheta:
    """Test superfluid theta calculation."""

    def test_default_t_lambda(self):
        """Default T_lambda is 2.17 K."""
        theta_explicit = compute_superfluid_theta(1.0, 2.17)
        theta_default = compute_superfluid_theta(1.0)
        assert theta_explicit == theta_default

    def test_theta_in_range(self):
        """Theta should be in [0, 1]."""
        for T in [0, 0.5, 1.0, 1.5, 2.0, 2.17, 3.0, 4.0]:
            theta = compute_superfluid_theta(T)
            assert 0 <= theta <= 1


class TestUnifiedChemistryTheta:
    """Test unified chemistry theta calculation."""

    def test_all_superconductors_valid_theta(self):
        """All superconductors should have theta in [0, 1]."""
        for name, mat in SUPERCONDUCTORS.items():
            theta = compute_chemistry_theta(mat)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_cold_superconductor_high_theta(self):
        """Superconductor at T << T_c has high theta."""
        mat = QuantumMaterial(
            name="Test",
            critical_temperature=10.0,
            current_temperature=1.0
        )
        theta = compute_chemistry_theta(mat)
        assert theta > 0.8

    def test_warm_material_lower_theta(self):
        """Material at T > T_c has lower theta."""
        mat = QuantumMaterial(
            name="Test",
            critical_temperature=10.0,
            current_temperature=20.0
        )
        theta = compute_chemistry_theta(mat)
        assert theta < 1.0


class TestClassifyMaterialPhase:
    """Test material phase classification."""

    def test_normal(self):
        """Low theta -> NORMAL."""
        assert classify_material_phase(0.1) == MaterialPhase.NORMAL
        assert classify_material_phase(0.2) == MaterialPhase.NORMAL

    def test_near_transition(self):
        """Medium theta -> NEAR_TRANSITION."""
        assert classify_material_phase(0.4) == MaterialPhase.NEAR_TRANSITION
        assert classify_material_phase(0.6) == MaterialPhase.NEAR_TRANSITION

    def test_quantum(self):
        """High theta -> QUANTUM."""
        assert classify_material_phase(0.85) == MaterialPhase.QUANTUM
        assert classify_material_phase(0.95) == MaterialPhase.QUANTUM

    def test_boundary_030(self):
        """theta = 0.3 is boundary (NEAR_TRANSITION)."""
        assert classify_material_phase(0.3) == MaterialPhase.NEAR_TRANSITION

    def test_boundary_080(self):
        """theta = 0.8 is boundary (QUANTUM)."""
        assert classify_material_phase(0.8) == MaterialPhase.QUANTUM


class TestPhysicalConstants:
    """Test physical constants."""

    def test_boltzmann(self):
        """Boltzmann constant."""
        assert K_B == pytest.approx(1.380649e-23, rel=1e-6)

    def test_hbar(self):
        """Reduced Planck constant."""
        assert HBAR == pytest.approx(1.054571817e-34, rel=1e-6)

    def test_electron_mass(self):
        """Electron mass."""
        assert M_ELECTRON == pytest.approx(9.10938e-31, rel=1e-4)

    def test_electron_charge(self):
        """Electron charge."""
        assert E_CHARGE == pytest.approx(1.602176634e-19, rel=1e-6)


class TestQuantumMaterialDataclass:
    """Test QuantumMaterial dataclass."""

    def test_create_minimal(self):
        """Should create material with required parameters."""
        mat = QuantumMaterial(
            name="Test",
            critical_temperature=10.0,
            current_temperature=4.0
        )
        assert mat.name == "Test"
        assert mat.critical_temperature == 10.0

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        mat = QuantumMaterial(
            name="Test",
            critical_temperature=10.0,
            current_temperature=4.0
        )
        assert mat.gap_energy is None
        assert mat.coherence_length is None
        assert mat.penetration_depth is None

    def test_full_creation(self):
        """Should create material with all parameters."""
        mat = QuantumMaterial(
            name="Full Test",
            critical_temperature=9.3,
            current_temperature=4.2,
            gap_energy=1.5e-3,
            coherence_length=38e-9,
            penetration_depth=39e-9
        )
        assert mat.gap_energy == 1.5e-3
        assert mat.coherence_length == 38e-9


class TestEnums:
    """Test enum definitions."""

    def test_material_phases(self):
        """All material phases should be defined."""
        assert MaterialPhase.NORMAL.value == "normal"
        assert MaterialPhase.NEAR_TRANSITION.value == "near_transition"
        assert MaterialPhase.QUANTUM.value == "quantum"

    def test_superconductor_types(self):
        """All superconductor types should be defined."""
        assert SuperconductorType.TYPE_I.value == "type_I"
        assert SuperconductorType.TYPE_II.value == "type_II"
        assert SuperconductorType.HIGH_TC.value == "high_Tc"
        assert SuperconductorType.CONVENTIONAL.value == "conventional"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_temperature(self):
        """Negative temperature should return edge value."""
        theta = compute_superconductor_theta(-5, 9.3)
        assert theta == 1.0

    def test_very_small_temperature(self):
        """Very small temperature should give theta ~ 1."""
        theta = compute_superconductor_theta(1e-10, 9.3)
        assert theta > 0.99

    def test_very_large_temperature(self):
        """Very large temperature should give theta ~ 0."""
        theta = compute_superconductor_theta(1e6, 9.3)
        assert theta < 0.01

    def test_tc_equals_t(self):
        """When T = T_c, theta should be defined."""
        theta = compute_superconductor_theta(9.3, 9.3)
        assert 0 <= theta <= 1

    def test_negative_bec_temp(self):
        """Negative BEC temperature gives fraction = 1."""
        fraction = bec_condensate_fraction(-100, 170e-9)
        assert fraction == 1.0
