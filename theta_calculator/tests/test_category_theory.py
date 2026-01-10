"""
Tests for Category Theory Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Abstraction level classification
- Component theta calculations
- Mathematical correctness
"""


from theta_calculator.domains.category_theory import (
    AbstractionLevel,
    compute_abstraction_theta,
    compute_functoriality_theta,
    compute_naturality_theta,
    compute_universality_theta,
    compute_coherence_theta,
    compute_yoneda_theta,
    compute_adjunction_theta,
    compute_enriched_theta,
    compute_topos_theta,
    compute_category_theta,
    classify_abstraction_regime,
    CATEGORY_SYSTEMS,
)


class TestCategorySystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """CATEGORY_SYSTEMS dict should exist."""
        assert CATEGORY_SYSTEMS is not None
        assert isinstance(CATEGORY_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(CATEGORY_SYSTEMS) >= 5

    def test_system_names(self):
        """Key systems should be defined."""
        expected = [
            "set_function",
            "functor_category",
            "adjoint_pair",
            "elementary_topos",
        ]
        for name in expected:
            assert name in CATEGORY_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in CATEGORY_SYSTEMS.items():
            assert hasattr(system, "name"), f"{name} missing 'name'"
            assert hasattr(system, "abstraction_level")
            assert hasattr(system, "n_objects")
            assert hasattr(system, "n_morphisms")
            assert hasattr(system, "functoriality")
            assert hasattr(system, "naturality_satisfaction")


class TestAbstractionTheta:
    """Test abstraction level theta calculation."""

    def test_set_level_low_theta(self):
        """Set level -> low theta."""
        theta = compute_abstraction_theta(AbstractionLevel.SET)
        assert theta < 0.2

    def test_category_level_medium_theta(self):
        """Category level -> medium theta."""
        theta = compute_abstraction_theta(AbstractionLevel.CATEGORY)
        assert 0.3 < theta < 0.5

    def test_topos_level_high_theta(self):
        """Topos level -> high theta."""
        theta = compute_abstraction_theta(AbstractionLevel.TOPOS)
        assert theta > 0.9

    def test_ordering_preserved(self):
        """Higher abstraction -> higher theta."""
        theta_set = compute_abstraction_theta(AbstractionLevel.SET)
        theta_cat = compute_abstraction_theta(AbstractionLevel.CATEGORY)
        theta_topos = compute_abstraction_theta(AbstractionLevel.TOPOS)
        assert theta_set < theta_cat < theta_topos


class TestFunctorialityTheta:
    """Test functoriality theta calculation."""

    def test_perfect_functor(self):
        """All compositions preserved -> theta = 1.0."""
        theta = compute_functoriality_theta(100, 100, identity_preserved=True)
        assert theta == 1.0

    def test_no_preservation(self):
        """Nothing preserved -> low theta."""
        theta = compute_functoriality_theta(0, 100, identity_preserved=False)
        assert theta == 0.0

    def test_partial_preservation(self):
        """Partial preservation -> medium theta."""
        theta = compute_functoriality_theta(50, 100, identity_preserved=True)
        assert 0.4 < theta < 0.7

    def test_identity_contribution(self):
        """Identity preservation should contribute 20%."""
        theta_with = compute_functoriality_theta(0, 0, identity_preserved=True)
        theta_without = compute_functoriality_theta(0, 0, identity_preserved=False)
        assert theta_with > theta_without


class TestNaturalityTheta:
    """Test naturality theta calculation."""

    def test_fully_natural(self):
        """All squares commute -> theta = 1.0."""
        theta = compute_naturality_theta(100, 100)
        assert theta == 1.0

    def test_not_natural(self):
        """No squares commute -> theta = 0.0."""
        theta = compute_naturality_theta(0, 100)
        assert theta == 0.0

    def test_vacuous_naturality(self):
        """No squares to check -> theta = 1.0."""
        theta = compute_naturality_theta(0, 0)
        assert theta == 1.0


class TestUniversalityTheta:
    """Test universality theta calculation."""

    def test_universal_property_satisfied(self):
        """Distance = 0 -> theta = 1.0."""
        theta = compute_universality_theta(0.0)
        assert theta == 1.0

    def test_no_universal_property(self):
        """Distance = inf -> theta = 0.0."""
        theta = compute_universality_theta(float('inf'))
        assert theta == 0.0

    def test_near_universal(self):
        """Small distance -> high theta."""
        theta = compute_universality_theta(0.1)
        assert theta > 0.9

    def test_far_from_universal(self):
        """Large distance -> low theta."""
        theta = compute_universality_theta(10.0)
        assert theta < 0.1


class TestCoherenceTheta:
    """Test coherence theta calculation."""

    def test_all_coherent(self):
        """All conditions satisfied -> theta = 1.0."""
        theta = compute_coherence_theta(5, 5)
        assert theta == 1.0

    def test_none_coherent(self):
        """No conditions satisfied -> theta = 0.0."""
        theta = compute_coherence_theta(0, 5)
        assert theta == 0.0

    def test_no_conditions(self):
        """No conditions required -> theta = 1.0."""
        theta = compute_coherence_theta(0, 0)
        assert theta == 1.0


class TestYonedaTheta:
    """Test Yoneda lemma theta calculation."""

    def test_fully_representable(self):
        """All representable, faithful -> high theta."""
        theta = compute_yoneda_theta(1.0, embedding_faithful=True)
        assert theta == 1.0

    def test_nothing_representable(self):
        """Nothing representable -> low theta."""
        theta = compute_yoneda_theta(0.0, embedding_faithful=False)
        assert theta == 0.0

    def test_faithful_bonus(self):
        """Faithful embedding should boost theta."""
        theta_faithful = compute_yoneda_theta(0.5, embedding_faithful=True)
        theta_unfaithful = compute_yoneda_theta(0.5, embedding_faithful=False)
        assert theta_faithful > theta_unfaithful


class TestAdjunctionTheta:
    """Test adjunction theta calculation."""

    def test_perfect_adjunction(self):
        """All conditions met -> theta = 1.0."""
        theta = compute_adjunction_theta(
            unit_natural=True,
            counit_natural=True,
            triangle_left=True,
            triangle_right=True
        )
        assert theta == 1.0

    def test_no_adjunction(self):
        """No conditions met -> theta = 0.0."""
        theta = compute_adjunction_theta(
            unit_natural=False,
            counit_natural=False,
            triangle_left=False,
            triangle_right=False
        )
        assert theta == 0.0

    def test_partial_adjunction(self):
        """Some conditions met -> medium theta."""
        theta = compute_adjunction_theta(
            unit_natural=True,
            counit_natural=True,
            triangle_left=False,
            triangle_right=False
        )
        assert theta == 0.5


class TestEnrichedTheta:
    """Test enriched category theta calculation."""

    def test_properly_enriched(self):
        """Proper enrichment -> high theta."""
        theta = compute_enriched_theta(
            AbstractionLevel.CATEGORY,
            hom_objects_defined=True,
            composition_enriched=True
        )
        assert theta > 0.5

    def test_set_enriched(self):
        """Set-enriched (ordinary) -> lower theta."""
        theta = compute_enriched_theta(
            AbstractionLevel.SET,
            hom_objects_defined=True,
            composition_enriched=True
        )
        assert theta < 0.3


class TestToposTheta:
    """Test topos theta calculation."""

    def test_elementary_topos(self):
        """All topos properties -> theta = 1.0."""
        theta = compute_topos_theta(
            has_finite_limits=True,
            has_exponentials=True,
            has_subobject_classifier=True,
            is_grothendieck=True
        )
        assert theta == 1.0

    def test_no_topos(self):
        """No topos properties -> theta = 0.0."""
        theta = compute_topos_theta(
            has_finite_limits=False,
            has_exponentials=False,
            has_subobject_classifier=False,
            is_grothendieck=False
        )
        assert theta == 0.0

    def test_partial_topos(self):
        """Some properties -> medium theta."""
        theta = compute_topos_theta(
            has_finite_limits=True,
            has_exponentials=True,
            has_subobject_classifier=False,
            is_grothendieck=False
        )
        assert 0.4 < theta < 0.6


class TestUnifiedCategoryTheta:
    """Test unified category theta calculation."""

    def test_all_systems_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in CATEGORY_SYSTEMS.items():
            theta = compute_category_theta(system)
            assert 0 <= theta <= 1, f"{name} has invalid theta: {theta}"

    def test_set_function_low_theta(self):
        """Set function should have low theta."""
        system = CATEGORY_SYSTEMS["set_function"]
        theta = compute_category_theta(system)
        assert theta < 0.3

    def test_topos_high_theta(self):
        """Elementary topos should have high theta."""
        system = CATEGORY_SYSTEMS["elementary_topos"]
        theta = compute_category_theta(system)
        assert theta > 0.8

    def test_ordering_preserved(self):
        """Higher abstraction should generally have higher theta."""
        theta_set = compute_category_theta(CATEGORY_SYSTEMS["set_function"])
        theta_functor = compute_category_theta(CATEGORY_SYSTEMS["functor_category"])
        theta_topos = compute_category_theta(CATEGORY_SYSTEMS["elementary_topos"])

        assert theta_set < theta_functor < theta_topos


class TestAbstractionRegimeClassification:
    """Test abstraction regime classification."""

    def test_set_theoretic(self):
        assert classify_abstraction_regime(0.1) == "set_theoretic"

    def test_algebraic(self):
        assert classify_abstraction_regime(0.3) == "algebraic"

    def test_categorical(self):
        assert classify_abstraction_regime(0.5) == "categorical"

    def test_higher(self):
        assert classify_abstraction_regime(0.7) == "higher"

    def test_homotopical(self):
        assert classify_abstraction_regime(0.9) == "homotopical"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_theta_bounds(self):
        """All theta functions should return values in [0, 1]."""
        # Abstraction
        for level in AbstractionLevel:
            theta = compute_abstraction_theta(level)
            assert 0 <= theta <= 1

        # Functoriality
        assert 0 <= compute_functoriality_theta(-1, 100) <= 1
        assert 0 <= compute_functoriality_theta(200, 100) <= 1

        # Naturality
        assert 0 <= compute_naturality_theta(-1, 100) <= 1
        assert 0 <= compute_naturality_theta(200, 100) <= 1

        # Universality
        assert 0 <= compute_universality_theta(-1.0) <= 1
        assert 0 <= compute_universality_theta(1e10) <= 1

        # Coherence
        assert 0 <= compute_coherence_theta(-1, 5) <= 1
        assert 0 <= compute_coherence_theta(10, 5) <= 1


class TestDocstrings:
    """Test that functions have proper documentation."""

    def test_module_docstring(self):
        """Module should have docstring with citations."""
        import theta_calculator.domains.category_theory as module
        assert module.__doc__ is not None
        assert "\\cite{" in module.__doc__

    def test_function_docstrings(self):
        """Key functions should have docstrings."""
        functions = [
            compute_abstraction_theta,
            compute_functoriality_theta,
            compute_category_theta,
            classify_abstraction_regime,
        ]
        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
