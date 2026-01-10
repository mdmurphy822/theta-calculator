"""
Tests for Advanced Mathematics Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Topological calculations
- Geometric and dynamical classifications
- Mathematical constants
"""

import pytest

from theta_calculator.domains.advanced_mathematics import (
    MathematicalSystem,
    TopologicalPhase,
    GeometricRegime,
    IntegrabilityLevel,
    SymmetryType,
    ManifoldType,
    compute_topological_theta,
    compute_euler_theta,
    compute_curvature_theta,
    compute_integrability_theta,
    compute_symmetry_theta,
    compute_homotopy_theta,
    compute_math_theta,
    classify_topological_phase,
    classify_geometric_regime,
    classify_integrability,
    classify_symmetry,
    MATHEMATICAL_SYSTEMS,
    EULER_MASCHERONI,
    FEIGENBAUM_DELTA,
    APERY_CONSTANT,
    EULER_CHAR_S2,
    EULER_CHAR_T2,
)


class TestMathematicalSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """MATHEMATICAL_SYSTEMS dict should exist."""
        assert MATHEMATICAL_SYSTEMS is not None
        assert isinstance(MATHEMATICAL_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(MATHEMATICAL_SYSTEMS) >= 5

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = [
            "sphere_s2",
            "torus_t2",
            "harmonic_oscillator",
            "three_body_problem",
        ]
        for name in expected:
            assert name in MATHEMATICAL_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in MATHEMATICAL_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "domain")
            assert hasattr(system, "dimension")


class TestTopologicalTheta:
    """Test topological theta calculation."""

    def test_trivial_topology(self):
        """R^n has minimal topology -> low theta."""
        # R^n: b0=1, rest 0
        theta = compute_topological_theta([1, 0, 0, 0], dimension=3)
        assert theta < 0.2

    def test_sphere_topology(self):
        """S^n has b0=bn=1 -> moderate theta."""
        theta = compute_topological_theta([1, 0, 1], dimension=2)
        assert 0.1 < theta < 0.6

    def test_torus_topology(self):
        """T^2 has [1,2,1] -> higher theta."""
        theta = compute_topological_theta([1, 2, 1], dimension=2)
        assert theta > 0.5

    def test_empty_betti(self):
        """Empty Betti numbers -> theta = 0."""
        theta = compute_topological_theta([], dimension=2)
        assert theta == 0.0

    def test_high_dimension_torus(self):
        """T^n has 2^n Betti sum -> theta = 1."""
        # T^3: [1, 3, 3, 1] sum = 8 = 2^3
        theta = compute_topological_theta([1, 3, 3, 1], dimension=3)
        assert theta == 1.0


class TestEulerTheta:
    """Test Euler characteristic theta calculation."""

    def test_sphere_euler(self):
        """S^2 has chi=2 (expected for even dim) -> low theta."""
        theta = compute_euler_theta(2, dimension=2)
        assert theta < 0.2

    def test_torus_euler(self):
        """T^2 has chi=0 (different from expected) -> higher theta."""
        theta = compute_euler_theta(0, dimension=2)
        assert theta > 0.5

    def test_exotic_euler(self):
        """Large deviation -> high theta."""
        theta = compute_euler_theta(-200, dimension=6)  # CY3
        assert theta > 0.9


class TestCurvatureTheta:
    """Test curvature theta calculation."""

    def test_flat_space(self):
        """K = 0 -> theta = 0."""
        theta = compute_curvature_theta(0.0)
        assert theta == 0.0

    def test_unit_positive(self):
        """K = 1 (sphere) -> moderate theta."""
        theta = compute_curvature_theta(1.0)
        assert theta == pytest.approx(0.5)

    def test_negative_curvature(self):
        """K < 0 (hyperbolic) uses |K|."""
        theta_pos = compute_curvature_theta(1.0)
        theta_neg = compute_curvature_theta(-1.0)
        assert theta_pos == theta_neg

    def test_high_curvature(self):
        """High |K| -> theta ~ 1."""
        theta = compute_curvature_theta(100.0)
        assert theta > 0.99


class TestIntegrabilityTheta:
    """Test integrability theta calculation."""

    def test_completely_integrable(self):
        """n_conserved = n_dof -> theta = 0."""
        theta = compute_integrability_theta(3, 3)
        assert theta == 0.0

    def test_no_conserved(self):
        """No conserved quantities -> theta = 1."""
        theta = compute_integrability_theta(0, 3)
        assert theta == 1.0

    def test_partially_integrable(self):
        """Some conserved -> intermediate theta."""
        theta = compute_integrability_theta(1, 3)
        assert theta == pytest.approx(2/3)

    def test_zero_dof(self):
        """Zero DOF -> theta = 0.5 (default)."""
        theta = compute_integrability_theta(1, 0)
        assert theta == 0.5


class TestSymmetryTheta:
    """Test symmetry theta calculation."""

    def test_maximally_symmetric(self):
        """Full isometry group -> high theta."""
        # S^2: SO(3) has dim 3, max for dim 2 is 3
        theta = compute_symmetry_theta(3, 2)
        assert theta == 1.0

    def test_no_symmetry(self):
        """No continuous symmetry -> theta = 0."""
        theta = compute_symmetry_theta(0, 3)
        assert theta == 0.0

    def test_partial_symmetry(self):
        """Some symmetry -> intermediate theta."""
        theta = compute_symmetry_theta(1, 3)  # 1 out of 6 max
        assert 0.0 < theta < 0.3


class TestHomotopyTheta:
    """Test homotopy theta calculation."""

    def test_simply_connected(self):
        """pi_1 = 0 -> low theta."""
        theta = compute_homotopy_theta(0)
        assert theta == 0.0

    def test_infinite_pi1(self):
        """Large pi_1 rank -> high theta."""
        theta = compute_homotopy_theta(10)
        assert theta > 0.5

    def test_higher_homotopy(self):
        """Higher homotopy complexity increases theta."""
        theta_low = compute_homotopy_theta(1, higher_homotopy_complexity=0)
        theta_high = compute_homotopy_theta(1, higher_homotopy_complexity=5)
        assert theta_high > theta_low


class TestUnifiedTheta:
    """Test unified theta calculation for systems."""

    def test_all_systems_have_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in MATHEMATICAL_SYSTEMS.items():
            theta = compute_math_theta(system)
            assert 0.0 <= theta <= 1.0, f"{name}: theta={theta}"

    def test_integrable_low_theta(self):
        """Integrable systems should have low theta."""
        oscillator = MATHEMATICAL_SYSTEMS["harmonic_oscillator"]
        theta = compute_math_theta(oscillator)
        assert theta < 0.5  # Integrable implies low non-integrability theta

    def test_chaotic_high_theta(self):
        """Chaotic systems should have high theta."""
        lorenz = MATHEMATICAL_SYSTEMS["lorenz_system"]
        theta = compute_math_theta(lorenz)
        assert theta >= 0.6  # Chaotic = high non-integrability

    def test_torus_moderate_theta(self):
        """Torus has interesting but not extreme topology."""
        torus = MATHEMATICAL_SYSTEMS["torus_t2"]
        theta = compute_math_theta(torus)
        assert 0.3 < theta < 0.8


class TestTopologicalPhaseClassification:
    """Test topological phase classification."""

    def test_trivial(self):
        """pi_1=0, no higher -> TRIVIAL."""
        assert classify_topological_phase(0, False) == TopologicalPhase.TRIVIAL

    def test_simply_connected(self):
        """pi_1=0 but higher homotopy -> SIMPLY_CONNECTED."""
        assert classify_topological_phase(0, True) == TopologicalPhase.SIMPLY_CONNECTED

    def test_non_trivial_pi1(self):
        """pi_1 != 0 -> NON_TRIVIAL_PI1."""
        assert classify_topological_phase(2, False) == TopologicalPhase.NON_TRIVIAL_PI1

    def test_higher_homotopy(self):
        """Both pi_1 and higher -> HIGHER_HOMOTOPY."""
        assert classify_topological_phase(1, True) == TopologicalPhase.HIGHER_HOMOTOPY


class TestGeometricRegimeClassification:
    """Test geometric regime classification."""

    def test_flat(self):
        """K ~ 0 -> FLAT."""
        assert classify_geometric_regime(0.0) == GeometricRegime.FLAT
        assert classify_geometric_regime(1e-10) == GeometricRegime.FLAT

    def test_positive(self):
        """K > 0 -> POSITIVE_CURVED."""
        assert classify_geometric_regime(1.0) == GeometricRegime.POSITIVE_CURVED

    def test_negative(self):
        """K < 0 -> NEGATIVE_CURVED."""
        assert classify_geometric_regime(-1.0) == GeometricRegime.NEGATIVE_CURVED


class TestIntegrabilityClassification:
    """Test integrability classification."""

    def test_complete(self):
        """n_conserved >= n_dof -> COMPLETELY_INTEGRABLE."""
        assert classify_integrability(3, 3) == IntegrabilityLevel.COMPLETELY_INTEGRABLE
        assert classify_integrability(5, 3) == IntegrabilityLevel.COMPLETELY_INTEGRABLE

    def test_partial(self):
        """0.5 <= ratio < 1 -> PARTIALLY_INTEGRABLE."""
        assert classify_integrability(2, 3) == IntegrabilityLevel.PARTIALLY_INTEGRABLE

    def test_chaotic(self):
        """ratio < 0.5 -> CHAOTIC."""
        assert classify_integrability(1, 3) == IntegrabilityLevel.CHAOTIC
        assert classify_integrability(0, 3) == IntegrabilityLevel.CHAOTIC


class TestSymmetryClassification:
    """Test symmetry type classification."""

    def test_discrete(self):
        """dim=0 -> DISCRETE."""
        assert classify_symmetry(0) == SymmetryType.DISCRETE

    def test_continuous(self):
        """dim>0 -> CONTINUOUS."""
        assert classify_symmetry(3) == SymmetryType.CONTINUOUS

    def test_gauge(self):
        """Local symmetry -> GAUGE."""
        assert classify_symmetry(3, is_local=True) == SymmetryType.GAUGE

    def test_susy(self):
        """SUSY takes precedence."""
        assert classify_symmetry(3, has_susy=True) == SymmetryType.SUPERSYMMETRIC


class TestConstants:
    """Test mathematical constants."""

    def test_euler_mascheroni(self):
        """Euler-Mascheroni constant."""
        assert EULER_MASCHERONI == pytest.approx(0.5772156649, rel=1e-6)

    def test_feigenbaum_delta(self):
        """Feigenbaum delta constant."""
        assert FEIGENBAUM_DELTA == pytest.approx(4.6692016091, rel=1e-6)

    def test_apery(self):
        """Apery's constant zeta(3)."""
        assert APERY_CONSTANT == pytest.approx(1.2020569032, rel=1e-6)

    def test_euler_char_s2(self):
        """Euler characteristic of S^2 = 2."""
        assert EULER_CHAR_S2 == 2

    def test_euler_char_t2(self):
        """Euler characteristic of T^2 = 0."""
        assert EULER_CHAR_T2 == 0


class TestSystemDataclass:
    """Test MathematicalSystem dataclass."""

    def test_create_minimal_system(self):
        """Should create system with required parameters."""
        system = MathematicalSystem(
            name="Test",
            domain="topology",
            dimension=3,
        )
        assert system.name == "Test"
        assert system.dimension == 3

    def test_total_betti(self):
        """total_betti should sum Betti numbers."""
        system = MathematicalSystem(
            name="Test",
            domain="topology",
            dimension=2,
            betti_numbers=[1, 2, 1],
        )
        assert system.total_betti == 4

    def test_total_betti_none(self):
        """total_betti with no Betti numbers -> 0."""
        system = MathematicalSystem(
            name="Test",
            domain="topology",
            dimension=2,
        )
        assert system.total_betti == 0

    def test_is_simply_connected(self):
        """is_simply_connected checks pi_1 rank."""
        simply = MathematicalSystem(
            name="Simply",
            domain="topology",
            dimension=2,
            fundamental_group_rank=0,
        )
        not_simply = MathematicalSystem(
            name="Not Simply",
            domain="topology",
            dimension=2,
            fundamental_group_rank=2,
        )
        assert simply.is_simply_connected is True
        assert not_simply.is_simply_connected is False

    def test_integrability_ratio(self):
        """integrability_ratio should be conserved/dof."""
        system = MathematicalSystem(
            name="Test",
            domain="dynamics",
            dimension=4,
            n_conserved=2,
            n_degrees_of_freedom=4,
        )
        assert system.integrability_ratio == pytest.approx(0.5)

    def test_is_integrable(self):
        """is_integrable checks if Liouville integrable."""
        integrable = MathematicalSystem(
            name="Integrable",
            domain="dynamics",
            dimension=4,
            n_conserved=3,
            n_degrees_of_freedom=3,
        )
        not_integrable = MathematicalSystem(
            name="Not Integrable",
            domain="dynamics",
            dimension=4,
            n_conserved=1,
            n_degrees_of_freedom=3,
        )
        assert integrable.is_integrable is True
        assert not_integrable.is_integrable is False


class TestEnums:
    """Test enum definitions."""

    def test_topological_phases(self):
        """All topological phases should be defined."""
        assert TopologicalPhase.TRIVIAL.value == "trivial"
        assert TopologicalPhase.SIMPLY_CONNECTED.value == "simply_connected"
        assert TopologicalPhase.NON_TRIVIAL_PI1.value == "non_trivial_pi1"
        assert TopologicalPhase.HIGHER_HOMOTOPY.value == "higher_homotopy"

    def test_geometric_regimes(self):
        """All geometric regimes should be defined."""
        assert GeometricRegime.FLAT.value == "flat"
        assert GeometricRegime.POSITIVE_CURVED.value == "positive"
        assert GeometricRegime.NEGATIVE_CURVED.value == "negative"
        assert GeometricRegime.MIXED.value == "mixed"

    def test_integrability_levels(self):
        """All integrability levels should be defined."""
        assert IntegrabilityLevel.COMPLETELY_INTEGRABLE.value == "complete"
        assert IntegrabilityLevel.PARTIALLY_INTEGRABLE.value == "partial"
        assert IntegrabilityLevel.CHAOTIC.value == "chaotic"

    def test_symmetry_types(self):
        """All symmetry types should be defined."""
        assert SymmetryType.DISCRETE.value == "discrete"
        assert SymmetryType.CONTINUOUS.value == "continuous"
        assert SymmetryType.GAUGE.value == "gauge"
        assert SymmetryType.SUPERSYMMETRIC.value == "supersymmetric"

    def test_manifold_types(self):
        """All manifold types should be defined."""
        assert ManifoldType.EUCLIDEAN.value == "euclidean"
        assert ManifoldType.SPHERICAL.value == "spherical"
        assert ManifoldType.HYPERBOLIC.value == "hyperbolic"
        assert ManifoldType.CALABI_YAU.value == "calabi_yau"
