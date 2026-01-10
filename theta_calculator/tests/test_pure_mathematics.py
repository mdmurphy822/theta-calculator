"""
Tests for Pure Mathematics Domain Module

Tests cover:
- Algebraic geometry (genus, singularities)
- Representation theory (dimension, rank)
- Functional analysis (spectral gap, condition number)
- Combinatorics (Ramsey bounds)
- Theta range validation [0, 1]
"""

import pytest

from theta_calculator.domains.pure_mathematics import (
    PureMathSystem,
    AlgebraicComplexity,
    RepresentationType,
    FunctionalClass,
    CombinatorialPhase,
    compute_pure_math_theta,
    compute_genus_theta,
    compute_singularity_theta,
    compute_rank_theta,
    compute_dimension_theta,
    compute_spectral_theta,
    compute_condition_theta,
    compute_ramsey_theta,
    genus_from_degree,
    euler_characteristic,
    dimension_formula_sl2,
    spectral_gap,
    condition_number,
    ramsey_lower_bound,
    ramsey_upper_bound,
    chromatic_number_estimate,
    classify_algebraic_complexity,
    classify_representation,
    classify_functional_space,
    classify_combinatorial_phase,
    PURE_MATH_SYSTEMS,
    GENUS_MAX_TYPICAL,
)


class TestPureMathSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """PURE_MATH_SYSTEMS dict should exist."""
        assert PURE_MATH_SYSTEMS is not None
        assert isinstance(PURE_MATH_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 8 systems."""
        assert len(PURE_MATH_SYSTEMS) >= 8

    def test_key_systems_defined(self):
        """Key systems should be defined."""
        expected = ["elliptic_curve", "k3_surface", "sl2r_reps", "hilbert_l2"]
        for name in expected:
            assert name in PURE_MATH_SYSTEMS, f"Missing system: {name}"


class TestGenusFromDegree:
    """Test genus calculation for plane curves."""

    def test_degree_1(self):
        """Degree 1 (line) has genus 0."""
        assert genus_from_degree(1) == 0

    def test_degree_2(self):
        """Degree 2 (conic) has genus 0."""
        assert genus_from_degree(2) == 0

    def test_degree_3(self):
        """Degree 3 (cubic) has genus 1."""
        assert genus_from_degree(3) == 1

    def test_degree_4(self):
        """Degree 4 (quartic) has genus 3."""
        assert genus_from_degree(4) == 3


class TestEulerCharacteristic:
    """Test Euler characteristic calculation."""

    def test_sphere(self):
        """Sphere (genus 0) has chi = 2."""
        assert euler_characteristic(0) == 2

    def test_torus(self):
        """Torus (genus 1) has chi = 0."""
        assert euler_characteristic(1) == 0

    def test_genus_2(self):
        """Genus 2 surface has chi = -2."""
        assert euler_characteristic(2) == -2


class TestGenusTheta:
    """Test genus theta calculation."""

    def test_genus_zero(self):
        """Genus 0 gives theta = 0."""
        theta = compute_genus_theta(0)
        assert theta == 0.0

    def test_max_genus(self):
        """Max genus gives theta = 1."""
        theta = compute_genus_theta(GENUS_MAX_TYPICAL)
        assert theta == 1.0

    def test_half_genus(self):
        """Half max genus gives theta = 0.5."""
        theta = compute_genus_theta(GENUS_MAX_TYPICAL // 2, GENUS_MAX_TYPICAL)
        assert theta == pytest.approx(0.5, rel=0.1)


class TestSingularityTheta:
    """Test singularity theta calculation."""

    def test_no_singularities(self):
        """No singularities gives theta = 0."""
        theta = compute_singularity_theta(0, 4)
        assert theta == 0.0

    def test_many_singularities(self):
        """Many singularities gives higher theta."""
        theta = compute_singularity_theta(10, 4)
        assert theta > 0.5


class TestDimensionFormulaSL2:
    """Test SL(2) representation dimension."""

    def test_trivial(self):
        """Weight 0 gives dimension 1."""
        assert dimension_formula_sl2(0) == 1

    def test_standard(self):
        """Weight 1 gives dimension 2."""
        assert dimension_formula_sl2(1) == 2

    def test_adjoint(self):
        """Weight 2 gives dimension 3."""
        assert dimension_formula_sl2(2) == 3


class TestRankTheta:
    """Test rank theta calculation."""

    def test_rank_zero(self):
        """Rank 0 gives theta = 0."""
        theta = compute_rank_theta(0)
        assert theta == 0.0

    def test_rank_max(self):
        """Rank = max gives theta = 1."""
        theta = compute_rank_theta(100, 100)
        assert theta == 1.0


class TestDimensionTheta:
    """Test dimension theta calculation."""

    def test_dim_one(self):
        """Dimension 1 gives theta = 0."""
        theta = compute_dimension_theta(1)
        assert theta == 0.0

    def test_dim_max(self):
        """Dimension = max gives theta = 1."""
        theta = compute_dimension_theta(1000, 1000)
        assert theta == 1.0


class TestSpectralGap:
    """Test spectral gap calculation."""

    def test_two_eigenvalues(self):
        """Gap between two eigenvalues."""
        gap = spectral_gap([0.0, 0.5, 1.0])
        assert gap == 0.5

    def test_single_eigenvalue(self):
        """Single eigenvalue gives gap = 0."""
        gap = spectral_gap([1.0])
        assert gap == 0.0

    def test_unsorted_input(self):
        """Handles unsorted input."""
        gap = spectral_gap([1.0, 0.0, 0.5])
        assert gap == 0.5


class TestConditionNumber:
    """Test condition number calculation."""

    def test_well_conditioned(self):
        """Equal eigenvalues give kappa = 1."""
        kappa = condition_number([1.0, 1.0, 1.0])
        assert kappa == 1.0

    def test_ill_conditioned(self):
        """Large spread gives high kappa."""
        kappa = condition_number([0.01, 1.0])
        assert kappa == 100.0

    def test_singular(self):
        """Zero eigenvalue gives infinite kappa."""
        kappa = condition_number([0.0, 1.0])
        assert kappa == float('inf')


class TestSpectralTheta:
    """Test spectral theta calculation."""

    def test_zero_gap(self):
        """Zero gap gives theta = 0."""
        theta = compute_spectral_theta(0)
        assert theta == 0.0

    def test_target_gap(self):
        """Target gap gives theta = 1."""
        theta = compute_spectral_theta(1.0, 1.0)
        assert theta == 1.0


class TestConditionTheta:
    """Test condition theta calculation."""

    def test_perfectly_conditioned(self):
        """kappa = 1 gives theta = 1."""
        theta = compute_condition_theta(1.0)
        assert theta == 1.0

    def test_singular(self):
        """kappa = inf gives theta = 0."""
        theta = compute_condition_theta(float('inf'))
        assert theta == 0.0


class TestRamseyBounds:
    """Test Ramsey number bounds."""

    def test_r33(self):
        """R(3,3) = 6 is exact."""
        lower = ramsey_lower_bound(3, 3)
        upper = ramsey_upper_bound(3, 3)
        assert lower <= 6 <= upper

    def test_trivial(self):
        """R(1,s) = 1."""
        assert ramsey_lower_bound(1, 5) == 1
        assert ramsey_upper_bound(1, 5) == 1


class TestChromaticNumber:
    """Test chromatic number estimate."""

    def test_complete_graph(self):
        """Complete graph has chi = n."""
        # K_5: 5 vertices, 10 edges
        chi = chromatic_number_estimate(5, 10)
        assert chi == 5

    def test_empty_graph(self):
        """Empty graph has chi = 1."""
        chi = chromatic_number_estimate(5, 0)
        assert chi == 1


class TestRamseyTheta:
    """Test Ramsey theta calculation."""

    def test_exact_known(self):
        """Exact value gives theta = 1."""
        theta = compute_ramsey_theta(6, 6, 6)
        assert theta == 1.0

    def test_large_gap(self):
        """Large gap gives low theta."""
        theta = compute_ramsey_theta(45, 43, 48)
        assert theta < 1.0
        assert theta > 0.0


class TestUnifiedPureMathTheta:
    """Test unified pure math theta calculation."""

    def test_all_systems_valid_theta(self):
        """All systems should have theta in [0, 1]."""
        for name, system in PURE_MATH_SYSTEMS.items():
            theta = compute_pure_math_theta(system)
            assert 0 <= theta <= 1, f"{name}: theta={theta}"

    def test_elliptic_curve_moderate(self):
        """Elliptic curve has moderate theta."""
        curve = PURE_MATH_SYSTEMS["elliptic_curve"]
        theta = compute_pure_math_theta(curve)
        assert 0 < theta < 1


class TestClassifyAlgebraicComplexity:
    """Test algebraic complexity classification."""

    def test_trivial(self):
        """Genus 0, no singularities -> TRIVIAL."""
        result = classify_algebraic_complexity(0, 0)
        assert result == AlgebraicComplexity.TRIVIAL

    def test_singular(self):
        """With singularities -> SINGULAR."""
        result = classify_algebraic_complexity(1, 5)
        assert result == AlgebraicComplexity.SINGULAR

    def test_smooth(self):
        """Genus 1, no singularities -> SMOOTH."""
        result = classify_algebraic_complexity(1, 0)
        assert result == AlgebraicComplexity.SMOOTH


class TestClassifyRepresentation:
    """Test representation classification."""

    def test_unitary(self):
        """Unitary flag -> UNITARY."""
        result = classify_representation(10, is_unitary=True)
        assert result == RepresentationType.UNITARY

    def test_irreducible(self):
        """Dimension 1 -> IRREDUCIBLE."""
        result = classify_representation(1)
        assert result == RepresentationType.IRREDUCIBLE

    def test_finite_dim(self):
        """Finite dimension -> FINITE_DIM."""
        result = classify_representation(100)
        assert result == RepresentationType.FINITE_DIM


class TestClassifyFunctionalSpace:
    """Test functional space classification."""

    def test_hilbert(self):
        """Inner product + complete -> HILBERT."""
        result = classify_functional_space(True, True)
        assert result == FunctionalClass.HILBERT

    def test_banach(self):
        """No inner product + complete -> BANACH."""
        result = classify_functional_space(False, True)
        assert result == FunctionalClass.BANACH


class TestClassifyCombinatorialPhase:
    """Test combinatorial phase classification."""

    def test_polynomial(self):
        """Low theta -> POLYNOMIAL."""
        result = classify_combinatorial_phase(0.1)
        assert result == CombinatorialPhase.POLYNOMIAL

    def test_tower(self):
        """High theta -> TOWER."""
        result = classify_combinatorial_phase(0.9)
        assert result == CombinatorialPhase.TOWER


class TestEnums:
    """Test enum definitions."""

    def test_algebraic_complexity(self):
        """All algebraic complexity values defined."""
        assert AlgebraicComplexity.TRIVIAL.value == "trivial"
        assert AlgebraicComplexity.SINGULAR.value == "singular"

    def test_representation_type(self):
        """All representation types defined."""
        assert RepresentationType.FINITE_DIM.value == "finite_dim"
        assert RepresentationType.UNITARY.value == "unitary"


class TestPureMathSystemDataclass:
    """Test PureMathSystem dataclass."""

    def test_create_minimal(self):
        """Should create system with name only."""
        system = PureMathSystem(name="Test")
        assert system.name == "Test"
        assert system.dimension == 1
        assert system.genus == 0

    def test_custom_values(self):
        """Can set custom values."""
        system = PureMathSystem(
            name="Custom",
            dimension=5,
            genus=3,
            spectral_gap=0.5
        )
        assert system.genus == 3
        assert system.spectral_gap == 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_genus_max(self):
        """Zero genus_max gives theta = 0."""
        theta = compute_genus_theta(5, 0)
        assert theta == 0.0

    def test_negative_genus(self):
        """Negative genus gives theta = 0."""
        theta = compute_genus_theta(-1)
        assert theta == 0.0

    def test_empty_eigenvalue_list(self):
        """Empty list gives default behavior."""
        kappa = condition_number([])
        assert kappa == float('inf')
