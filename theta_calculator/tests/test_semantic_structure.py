"""
Tests for Semantic Structure Domain Module

Tests cover:
- System existence and attributes
- Theta range validation [0, 1]
- Semantic coherence classification
- Component theta calculations
"""

import pytest
import numpy as np

from theta_calculator.domains.semantic_structure import (
    SemanticCoherence,
    compute_coherence_theta,
    compute_lsa_coherence,
    compute_hierarchy_theta,
    compute_connectivity_theta,
    compute_graph_connectivity,
    compute_disambiguation_theta,
    compute_grounding_theta,
    compute_schema_theta,
    compute_ontology_theta,
    compute_linked_data_theta,
    compute_discourse_theta,
    compute_semantic_theta,
    classify_semantic_coherence,
    word_sense_ambiguity,
    SEMANTIC_SYSTEMS,
)


class TestSemanticSystemsExist:
    """Test that example systems are properly defined."""

    def test_systems_dictionary_exists(self):
        """SEMANTIC_SYSTEMS dict should exist."""
        assert SEMANTIC_SYSTEMS is not None
        assert isinstance(SEMANTIC_SYSTEMS, dict)

    def test_minimum_systems_count(self):
        """Should have at least 5 example systems."""
        assert len(SEMANTIC_SYSTEMS) >= 5

    def test_system_names(self):
        """Key systems should be defined."""
        expected = [
            "random_text",
            "raw_text",
            "knowledge_graph",
            "formal_ontology",
        ]
        for name in expected:
            assert name in SEMANTIC_SYSTEMS, f"Missing system: {name}"

    def test_system_attributes(self):
        """Systems should have required attributes."""
        for name, system in SEMANTIC_SYSTEMS.items():
            assert hasattr(system, "name")
            assert hasattr(system, "structure_format")
            assert hasattr(system, "coherence_score")
            assert hasattr(system, "connectivity")
            assert hasattr(system, "ambiguity_score")


class TestCoherenceTheta:
    """Test coherence theta calculation."""

    def test_high_coherence(self):
        """High coherence -> high theta."""
        theta = compute_coherence_theta(0.9)
        assert theta == pytest.approx(0.9)

    def test_low_coherence(self):
        """Low coherence -> low theta."""
        theta = compute_coherence_theta(0.1)
        assert theta == pytest.approx(0.1)

    def test_clamping(self):
        """Values outside [0,1] should be clamped."""
        assert compute_coherence_theta(1.5) == 1.0
        assert compute_coherence_theta(-0.5) == 0.0


class TestLSACoherence:
    """Test LSA coherence from similarity matrix."""

    def test_perfect_coherence(self):
        """All similarities = 1 -> coherence = 1."""
        matrix = np.ones((5, 5))
        coherence = compute_lsa_coherence(matrix)
        assert coherence == pytest.approx(1.0)

    def test_no_coherence(self):
        """All similarities = 0 -> coherence = 0."""
        matrix = np.zeros((5, 5))
        coherence = compute_lsa_coherence(matrix)
        assert coherence == pytest.approx(0.0)

    def test_single_sentence(self):
        """Single sentence -> trivially coherent."""
        matrix = np.array([[1.0]])
        coherence = compute_lsa_coherence(matrix)
        assert coherence == 1.0


class TestHierarchyTheta:
    """Test hierarchy depth theta calculation."""

    def test_no_hierarchy(self):
        """Depth 0 -> theta = 0."""
        theta = compute_hierarchy_theta(0)
        assert theta == pytest.approx(0.0)

    def test_max_hierarchy(self):
        """Max depth -> theta = 1."""
        theta = compute_hierarchy_theta(10, max_depth=10)
        assert theta == pytest.approx(1.0)

    def test_partial_hierarchy(self):
        """Partial depth -> medium theta."""
        theta = compute_hierarchy_theta(5, max_depth=10)
        assert 0.4 < theta < 0.9


class TestConnectivityTheta:
    """Test connectivity theta calculation."""

    def test_full_connectivity(self):
        """Full connectivity -> theta = 1."""
        theta = compute_connectivity_theta(1.0)
        assert theta == 1.0

    def test_no_connectivity(self):
        """No connectivity -> theta = 0."""
        theta = compute_connectivity_theta(0.0)
        assert theta == 0.0


class TestGraphConnectivity:
    """Test graph connectivity estimation."""

    def test_single_component(self):
        """Single component -> high connectivity."""
        conn = compute_graph_connectivity(100, 200, 1)
        assert conn > 0.5

    def test_many_components(self):
        """Many components -> lower connectivity."""
        conn = compute_graph_connectivity(100, 50, 20)
        assert conn < 0.3

    def test_empty_graph(self):
        """Empty graph -> connectivity = 0."""
        conn = compute_graph_connectivity(0, 0, 0)
        assert conn == 0.0


class TestDisambiguationTheta:
    """Test disambiguation theta calculation."""

    def test_no_ambiguity(self):
        """No ambiguity -> theta = 1."""
        theta = compute_disambiguation_theta(0.0)
        assert theta == 1.0

    def test_high_ambiguity(self):
        """High ambiguity -> low theta."""
        theta = compute_disambiguation_theta(0.9)
        assert theta == pytest.approx(0.1)


class TestWordSenseAmbiguity:
    """Test word sense ambiguity calculation."""

    def test_unambiguous(self):
        """All words have 1 sense -> ambiguity = 0."""
        ambiguity = word_sense_ambiguity([1, 1, 1, 1])
        assert ambiguity == 0.0

    def test_highly_ambiguous(self):
        """Many senses per word -> high ambiguity."""
        ambiguity = word_sense_ambiguity([10, 10, 10])
        assert ambiguity > 0.8


class TestGroundingTheta:
    """Test grounding theta calculation."""

    def test_fully_grounded(self):
        """All concepts grounded -> theta = 1."""
        theta = compute_grounding_theta(100, 100)
        assert theta == 1.0

    def test_ungrounded(self):
        """No concepts grounded -> theta = 0."""
        theta = compute_grounding_theta(0, 100)
        assert theta == 0.0


class TestSchemaTheta:
    """Test schema completeness theta calculation."""

    def test_complete_schema(self):
        """All slots filled -> theta = 1."""
        theta = compute_schema_theta(10, 10)
        assert theta == 1.0

    def test_empty_schema(self):
        """No slots filled -> theta = 0."""
        theta = compute_schema_theta(0, 10)
        assert theta == 0.0

    def test_optional_slots(self):
        """Optional slots contribute less."""
        theta1 = compute_schema_theta(5, 5, 0, 5)
        theta2 = compute_schema_theta(5, 5, 5, 5)
        assert theta1 < theta2


class TestOntologyTheta:
    """Test ontology quality theta calculation."""

    def test_complete_ontology(self):
        """All properties -> theta = 1."""
        theta = compute_ontology_theta(
            has_taxonomy=True,
            has_properties=True,
            has_constraints=True,
            has_rules=True,
            is_consistent=True
        )
        assert theta == 1.0

    def test_empty_ontology(self):
        """No properties -> theta = 0."""
        theta = compute_ontology_theta(
            has_taxonomy=False,
            has_properties=False,
            has_constraints=False,
            has_rules=False,
            is_consistent=False
        )
        assert theta == 0.0


class TestLinkedDataTheta:
    """Test Linked Open Data theta calculation."""

    def test_well_linked(self):
        """Many external links -> high theta."""
        theta = compute_linked_data_theta(
            internal_links=500,
            external_links=100,
            n_entities=100,
            uses_uris=True
        )
        assert theta > 0.7

    def test_isolated_data(self):
        """No links -> low theta."""
        theta = compute_linked_data_theta(
            internal_links=0,
            external_links=0,
            n_entities=100,
            uses_uris=False
        )
        assert theta == 0.0


class TestDiscourseTheta:
    """Test discourse structure theta calculation."""

    def test_well_structured(self):
        """Many RST relations -> high theta."""
        theta = compute_discourse_theta(
            rst_relations=9,
            n_sentences=10,
            has_nucleus=True
        )
        assert theta > 0.8

    def test_unstructured(self):
        """No RST relations -> low theta."""
        theta = compute_discourse_theta(
            rst_relations=0,
            n_sentences=10,
            has_nucleus=False
        )
        assert theta == 0.0


class TestUnifiedSemanticTheta:
    """Test unified semantic theta calculation."""

    def test_all_systems_valid_theta(self):
        """All example systems should have theta in [0, 1]."""
        for name, system in SEMANTIC_SYSTEMS.items():
            theta = compute_semantic_theta(system)
            assert 0 <= theta <= 1, f"{name} has invalid theta: {theta}"

    def test_random_text_low_theta(self):
        """Random text should have very low theta."""
        system = SEMANTIC_SYSTEMS["random_text"]
        theta = compute_semantic_theta(system)
        assert theta < 0.2

    def test_ontology_high_theta(self):
        """Formal ontology should have high theta."""
        system = SEMANTIC_SYSTEMS["formal_ontology"]
        theta = compute_semantic_theta(system)
        assert theta > 0.8

    def test_ordering_preserved(self):
        """More structure -> higher theta."""
        theta_random = compute_semantic_theta(SEMANTIC_SYSTEMS["random_text"])
        theta_raw = compute_semantic_theta(SEMANTIC_SYSTEMS["raw_text"])
        theta_kg = compute_semantic_theta(SEMANTIC_SYSTEMS["knowledge_graph"])
        theta_ont = compute_semantic_theta(SEMANTIC_SYSTEMS["formal_ontology"])

        assert theta_random < theta_raw < theta_kg < theta_ont


class TestSemanticCoherenceClassification:
    """Test semantic coherence classification."""

    def test_incoherent(self):
        assert classify_semantic_coherence(0.1) == SemanticCoherence.INCOHERENT

    def test_fragmented(self):
        assert classify_semantic_coherence(0.3) == SemanticCoherence.FRAGMENTED

    def test_partial(self):
        assert classify_semantic_coherence(0.5) == SemanticCoherence.PARTIAL

    def test_coherent(self):
        assert classify_semantic_coherence(0.7) == SemanticCoherence.COHERENT

    def test_unified(self):
        assert classify_semantic_coherence(0.9) == SemanticCoherence.UNIFIED


class TestDocstrings:
    """Test that functions have proper documentation."""

    def test_module_docstring(self):
        """Module should have docstring with citations."""
        import theta_calculator.domains.semantic_structure as module
        assert module.__doc__ is not None
        assert "\\cite{" in module.__doc__
