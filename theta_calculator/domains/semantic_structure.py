r"""
Semantic Structure Domain: Information Organization as Theta

This module implements theta as the semantic coherence parameter
using frameworks from linguistics, knowledge representation, and NLP.

Key Insight: Information systems exhibit phase transitions between:
- theta ~ 0: Incoherent/unstructured (flat text, ambiguous)
- theta ~ 1: Coherent/structured (knowledge graphs, ontologies)

Theta Maps To:
1. Coherence: LSA/embedding similarity across document
2. Hierarchy Depth: Levels of semantic structure
3. Connectivity: Graph connectivity of knowledge representation
4. Disambiguation: Resolution of semantic ambiguity
5. Grounding: Connection to real-world referents

Semantic Regimes:
- INCOHERENT (theta < 0.2): Random text, no structure
- FRAGMENTED (0.2 <= theta < 0.4): Basic structure, weak connections
- PARTIAL (0.4 <= theta < 0.6): Moderate coherence, some hierarchy
- COHERENT (0.6 <= theta < 0.8): Strong structure, clear relationships
- UNIFIED (theta >= 0.8): Full semantic graph, rich ontology

Physical Analogy:
Semantic structure undergoes a phase transition similar to
percolation in statistical physics. Below a critical "connection
density", information exists as isolated fragments. Above threshold,
a giant connected component emerges, enabling global meaning.

The coherence of text can be modeled as correlation functions in
a statistical field theory of meaning, where topics create
long-range correlations.

References (see BIBLIOGRAPHY.bib):
    \cite{Landauer1997} - Latent Semantic Analysis
    \cite{Mikolov2013} - Word2Vec embeddings
    \cite{Devlin2019} - BERT contextual embeddings
    \cite{Gruber1993} - Ontology definition
    \cite{Bizer2009} - Linked Open Data
    \cite{Mann1988} - Rhetorical Structure Theory
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class SemanticCoherence(Enum):
    """Semantic coherence states based on theta."""
    INCOHERENT = "incoherent"      # theta < 0.2
    FRAGMENTED = "fragmented"      # 0.2 <= theta < 0.4
    PARTIAL = "partial"            # 0.4 <= theta < 0.6
    COHERENT = "coherent"          # 0.6 <= theta < 0.8
    UNIFIED = "unified"            # theta >= 0.8


class StructureFormat(Enum):
    """Types of semantic structure."""
    FLAT_TEXT = "flat_text"        # Unstructured prose
    TAGGED = "tagged"              # HTML/XML tags
    HIERARCHICAL = "hierarchical"  # Document outline
    RELATIONAL = "relational"      # Entity-relation triples
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Full KG
    ONTOLOGY = "ontology"          # Formal ontology (OWL)


class EmbeddingType(Enum):
    """Types of semantic embeddings."""
    BAG_OF_WORDS = "bow"           # Count-based
    TFIDF = "tfidf"                # Term frequency
    LSA = "lsa"                    # Latent semantic analysis
    WORD2VEC = "word2vec"          # Static embeddings
    GLOVE = "glove"                # Global vectors
    BERT = "bert"                  # Contextual embeddings
    GPT = "gpt"                    # Autoregressive LM embeddings


@dataclass
class SemanticSystem:
    """
    A semantic system for theta analysis.

    Attributes:
        name: System identifier
        structure_format: Type of semantic structure
        n_documents: Number of documents/passages
        n_entities: Number of named entities
        n_relations: Number of semantic relations
        n_concepts: Number of distinct concepts
        coherence_score: LSA/embedding coherence [0, 1]
        hierarchy_depth: Maximum depth of structure
        connectivity: Graph connectivity ratio [0, 1]
        ambiguity_score: Semantic ambiguity [0, 1] (lower is better)
        grounding_ratio: Fraction of grounded concepts [0, 1]
        schema_completeness: Schema slot filling ratio [0, 1]
    """
    name: str
    structure_format: StructureFormat
    n_documents: int
    n_entities: int
    n_relations: int
    n_concepts: int
    coherence_score: float
    hierarchy_depth: int
    connectivity: float
    ambiguity_score: float
    grounding_ratio: float
    schema_completeness: float = 0.0

    @property
    def relation_density(self) -> float:
        """Average relations per entity."""
        if self.n_entities == 0:
            return 0.0
        return self.n_relations / self.n_entities


# =============================================================================
# COHERENCE THETA
# =============================================================================

def compute_coherence_theta(
    coherence_score: float
) -> float:
    r"""
    Compute theta from semantic coherence score.

    Coherence measures how well text/concepts fit together.
    Typically computed via:
    - LSA: cosine similarity of adjacent sentences
    - Embeddings: average pairwise similarity

    Args:
        coherence_score: Pre-computed coherence [0, 1]

    Returns:
        theta in [0, 1]

    Reference: \cite{Landauer1997} - LSA for coherence
    """
    return np.clip(coherence_score, 0.0, 1.0)


def compute_lsa_coherence(
    similarity_matrix: np.ndarray
) -> float:
    r"""
    Compute coherence from LSA similarity matrix.

    Coherence = average of adjacent sentence similarities.

    Args:
        similarity_matrix: N x N matrix of sentence similarities

    Returns:
        Coherence score [0, 1]

    Reference: \cite{Landauer1997} - Latent Semantic Analysis
    """
    n = similarity_matrix.shape[0]
    if n < 2:
        return 1.0

    # Adjacent similarities (local coherence)
    local_coherence = np.mean([
        similarity_matrix[i, i+1]
        for i in range(n-1)
    ])

    # Global coherence (average all pairs)
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    global_coherence = np.mean(upper_tri)

    # Weight local more heavily
    coherence = 0.7 * local_coherence + 0.3 * global_coherence
    return np.clip(coherence, 0.0, 1.0)


# =============================================================================
# HIERARCHY THETA
# =============================================================================

def compute_hierarchy_theta(
    depth: int,
    max_depth: int = 10
) -> float:
    r"""
    Compute theta from hierarchy depth.

    Deeper structure = higher theta.
    Optimal depth is task-dependent, but generally 3-7 levels.

    Args:
        depth: Current hierarchy depth
        max_depth: Maximum expected depth

    Returns:
        theta in [0, 1]
    """
    if max_depth <= 0:
        return 0.0

    # Logarithmic scaling (diminishing returns)
    normalized = np.log1p(depth) / np.log1p(max_depth)
    return np.clip(normalized, 0.0, 1.0)


# =============================================================================
# CONNECTIVITY THETA
# =============================================================================

def compute_connectivity_theta(
    connectivity: float
) -> float:
    r"""
    Compute theta from knowledge graph connectivity.

    Connectivity = fraction of nodes in largest connected component.
    Above percolation threshold, giant component emerges.

    Args:
        connectivity: Fraction in giant component [0, 1]

    Returns:
        theta in [0, 1]

    Reference: Percolation theory analogy
    """
    return np.clip(connectivity, 0.0, 1.0)


def compute_graph_connectivity(
    n_nodes: int,
    n_edges: int,
    n_components: int
) -> float:
    """
    Estimate connectivity from graph statistics.

    Args:
        n_nodes: Number of nodes
        n_edges: Number of edges
        n_components: Number of connected components

    Returns:
        Estimated connectivity [0, 1]
    """
    if n_nodes == 0:
        return 0.0

    # Approximate: more edges and fewer components = higher connectivity
    # Perfect connectivity: 1 component
    component_factor = 1.0 / n_components

    # Edge density
    max_edges = n_nodes * (n_nodes - 1) / 2
    edge_density = n_edges / max_edges if max_edges > 0 else 0

    connectivity = 0.6 * component_factor + 0.4 * min(1.0, edge_density * 10)
    return np.clip(connectivity, 0.0, 1.0)


# =============================================================================
# DISAMBIGUATION THETA
# =============================================================================

def compute_disambiguation_theta(
    ambiguity_score: float
) -> float:
    r"""
    Compute theta from disambiguation quality.

    Lower ambiguity = higher theta.
    Ambiguity measured by:
    - Word sense disambiguation accuracy
    - Entity linking precision
    - Coreference resolution F1

    Args:
        ambiguity_score: Remaining ambiguity [0, 1] (lower is better)

    Returns:
        theta in [0, 1]
    """
    return np.clip(1 - ambiguity_score, 0.0, 1.0)


def word_sense_ambiguity(
    n_senses_per_word: List[int]
) -> float:
    """
    Compute average word sense ambiguity.

    Args:
        n_senses_per_word: Number of possible senses for each word

    Returns:
        Ambiguity score [0, 1]
    """
    if not n_senses_per_word:
        return 0.0

    # More senses = more ambiguity
    avg_senses = np.mean(n_senses_per_word)

    # Normalize: 1 sense = 0 ambiguity, 10+ senses = high ambiguity
    ambiguity = 1 - (1.0 / avg_senses)
    return np.clip(ambiguity, 0.0, 1.0)


# =============================================================================
# GROUNDING THETA
# =============================================================================

def compute_grounding_theta(
    grounded: int,
    total: int
) -> float:
    r"""
    Compute theta from concept grounding.

    Grounding connects abstract concepts to real-world referents.
    Higher grounding = more interpretable semantics.

    Args:
        grounded: Number of grounded concepts
        total: Total concepts

    Returns:
        theta in [0, 1]

    Reference: Symbol grounding problem
    """
    if total == 0:
        return 0.0

    return np.clip(grounded / total, 0.0, 1.0)


# =============================================================================
# SCHEMA COMPLETENESS THETA
# =============================================================================

def compute_schema_theta(
    filled_slots: int,
    required_slots: int,
    optional_filled: int = 0,
    optional_total: int = 0
) -> float:
    r"""
    Compute theta from schema slot filling.

    Schema completeness measures how well information
    fills expected semantic frames/slots.

    Args:
        filled_slots: Number of required slots filled
        required_slots: Total required slots
        optional_filled: Optional slots filled
        optional_total: Total optional slots

    Returns:
        theta in [0, 1]
    """
    if required_slots == 0:
        return 1.0 if optional_total == 0 else optional_filled / optional_total

    required_ratio = filled_slots / required_slots

    if optional_total > 0:
        optional_ratio = optional_filled / optional_total
        theta = 0.8 * required_ratio + 0.2 * optional_ratio
    else:
        theta = required_ratio

    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# ONTOLOGY QUALITY THETA
# =============================================================================

def compute_ontology_theta(
    has_taxonomy: bool,
    has_properties: bool,
    has_constraints: bool,
    has_rules: bool,
    is_consistent: bool
) -> float:
    r"""
    Compute theta from ontology quality.

    A full ontology has:
    - Taxonomy (class hierarchy)
    - Properties (attributes, relations)
    - Constraints (cardinality, domain/range)
    - Rules (inference rules)
    - Consistency (no contradictions)

    Args:
        has_taxonomy: Has class hierarchy
        has_properties: Has defined properties
        has_constraints: Has formal constraints
        has_rules: Has inference rules
        is_consistent: Passes consistency check

    Returns:
        theta in [0, 1]

    Reference: \cite{Gruber1993} - Ontology definition
    """
    score = 0.0

    if has_taxonomy:
        score += 0.20
    if has_properties:
        score += 0.20
    if has_constraints:
        score += 0.20
    if has_rules:
        score += 0.15
    if is_consistent:
        score += 0.25

    return score


# =============================================================================
# LINKED DATA THETA
# =============================================================================

def compute_linked_data_theta(
    internal_links: int,
    external_links: int,
    n_entities: int,
    uses_uris: bool = True
) -> float:
    r"""
    Compute theta from Linked Open Data principles.

    5-star Linked Data:
    1. Available on the web
    2. Machine-readable
    3. Non-proprietary format
    4. Uses URIs
    5. Links to other data

    Args:
        internal_links: Links within dataset
        external_links: Links to external datasets
        n_entities: Number of entities
        uses_uris: Whether URIs are used for identification

    Returns:
        theta in [0, 1]

    Reference: \cite{Bizer2009} - Linked Open Data
    """
    if n_entities == 0:
        return 0.0

    # Internal link density
    internal_ratio = min(1.0, internal_links / (n_entities * 5))

    # External connectivity (key for LOD)
    external_ratio = min(1.0, external_links / n_entities)

    # URI usage is fundamental
    uri_bonus = 0.2 if uses_uris else 0.0

    theta = 0.3 * internal_ratio + 0.5 * external_ratio + uri_bonus
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# DISCOURSE COHERENCE THETA
# =============================================================================

def compute_discourse_theta(
    rst_relations: int,
    n_sentences: int,
    has_nucleus: bool = True
) -> float:
    r"""
    Compute theta from discourse structure.

    Rhetorical Structure Theory (RST) analyzes:
    - Relations between text spans (cause, contrast, etc.)
    - Nucleus (main content) vs satellite (supporting)

    Args:
        rst_relations: Number of identified RST relations
        n_sentences: Number of sentences
        has_nucleus: Whether clear nucleus is identified

    Returns:
        theta in [0, 1]

    Reference: \cite{Mann1988} - Rhetorical Structure Theory
    """
    if n_sentences <= 1:
        return 1.0  # Trivially coherent

    # Expected: roughly 1 relation per sentence pair
    expected_relations = n_sentences - 1
    relation_ratio = min(1.0, rst_relations / expected_relations)

    # Nucleus identification is important
    nucleus_bonus = 0.15 if has_nucleus else 0.0

    theta = 0.85 * relation_ratio + nucleus_bonus
    return np.clip(theta, 0.0, 1.0)


# =============================================================================
# UNIFIED SEMANTIC THETA
# =============================================================================

def compute_semantic_theta(system: SemanticSystem) -> float:
    """
    Compute unified theta for semantic system.

    Combines:
    - Coherence (25%)
    - Connectivity (20%)
    - Disambiguation (20%)
    - Grounding (15%)
    - Hierarchy (10%)
    - Schema completeness (10%)

    Args:
        system: SemanticSystem to analyze

    Returns:
        theta in [0, 1]
    """
    # Coherence
    theta_coh = compute_coherence_theta(system.coherence_score)

    # Connectivity
    theta_conn = compute_connectivity_theta(system.connectivity)

    # Disambiguation (invert ambiguity)
    theta_disamb = compute_disambiguation_theta(system.ambiguity_score)

    # Grounding
    theta_ground = compute_grounding_theta(
        int(system.grounding_ratio * system.n_concepts),
        system.n_concepts
    )

    # Hierarchy
    theta_hier = compute_hierarchy_theta(system.hierarchy_depth)

    # Schema
    theta_schema = system.schema_completeness

    # Weighted combination
    theta = (
        0.25 * theta_coh +
        0.20 * theta_conn +
        0.20 * theta_disamb +
        0.15 * theta_ground +
        0.10 * theta_hier +
        0.10 * theta_schema
    )

    return np.clip(theta, 0.0, 1.0)


def classify_semantic_coherence(theta: float) -> SemanticCoherence:
    """Classify semantic coherence from theta."""
    if theta < 0.2:
        return SemanticCoherence.INCOHERENT
    elif theta < 0.4:
        return SemanticCoherence.FRAGMENTED
    elif theta < 0.6:
        return SemanticCoherence.PARTIAL
    elif theta < 0.8:
        return SemanticCoherence.COHERENT
    else:
        return SemanticCoherence.UNIFIED


# =============================================================================
# EXAMPLE SYSTEMS
# =============================================================================

SEMANTIC_SYSTEMS: Dict[str, SemanticSystem] = {
    "random_text": SemanticSystem(
        name="Random Word Sequence",
        structure_format=StructureFormat.FLAT_TEXT,
        n_documents=1,
        n_entities=0,
        n_relations=0,
        n_concepts=50,
        coherence_score=0.1,
        hierarchy_depth=0,
        connectivity=0.0,
        ambiguity_score=0.9,
        grounding_ratio=0.0,
        schema_completeness=0.0,
    ),
    "raw_text": SemanticSystem(
        name="Unstructured Prose",
        structure_format=StructureFormat.FLAT_TEXT,
        n_documents=10,
        n_entities=50,
        n_relations=20,
        n_concepts=100,
        coherence_score=0.4,
        hierarchy_depth=1,
        connectivity=0.2,
        ambiguity_score=0.6,
        grounding_ratio=0.3,
        schema_completeness=0.1,
    ),
    "html_tagged": SemanticSystem(
        name="HTML-Tagged Document",
        structure_format=StructureFormat.TAGGED,
        n_documents=5,
        n_entities=100,
        n_relations=50,
        n_concepts=150,
        coherence_score=0.5,
        hierarchy_depth=4,
        connectivity=0.4,
        ambiguity_score=0.5,
        grounding_ratio=0.4,
        schema_completeness=0.3,
    ),
    "markdown_structured": SemanticSystem(
        name="Well-Structured Markdown",
        structure_format=StructureFormat.HIERARCHICAL,
        n_documents=3,
        n_entities=75,
        n_relations=60,
        n_concepts=120,
        coherence_score=0.65,
        hierarchy_depth=5,
        connectivity=0.5,
        ambiguity_score=0.35,
        grounding_ratio=0.5,
        schema_completeness=0.5,
    ),
    "schema_org_annotated": SemanticSystem(
        name="Schema.org Annotated Page",
        structure_format=StructureFormat.RELATIONAL,
        n_documents=1,
        n_entities=50,
        n_relations=100,
        n_concepts=80,
        coherence_score=0.7,
        hierarchy_depth=3,
        connectivity=0.7,
        ambiguity_score=0.2,
        grounding_ratio=0.8,
        schema_completeness=0.75,
    ),
    "knowledge_graph": SemanticSystem(
        name="Knowledge Graph (Wikidata-style)",
        structure_format=StructureFormat.KNOWLEDGE_GRAPH,
        n_documents=1000,
        n_entities=10000,
        n_relations=50000,
        n_concepts=5000,
        coherence_score=0.8,
        hierarchy_depth=7,
        connectivity=0.85,
        ambiguity_score=0.1,
        grounding_ratio=0.9,
        schema_completeness=0.8,
    ),
    "formal_ontology": SemanticSystem(
        name="Formal OWL Ontology",
        structure_format=StructureFormat.ONTOLOGY,
        n_documents=1,
        n_entities=500,
        n_relations=2000,
        n_concepts=300,
        coherence_score=0.95,
        hierarchy_depth=8,
        connectivity=0.95,
        ambiguity_score=0.02,
        grounding_ratio=1.0,
        schema_completeness=0.95,
    ),
    "bert_embeddings": SemanticSystem(
        name="BERT Contextual Embeddings",
        structure_format=StructureFormat.FLAT_TEXT,
        n_documents=100,
        n_entities=500,
        n_relations=0,  # Implicit in embeddings
        n_concepts=1000,
        coherence_score=0.75,
        hierarchy_depth=2,
        connectivity=0.6,  # Semantic similarity clustering
        ambiguity_score=0.15,  # Context disambiguates
        grounding_ratio=0.5,
        schema_completeness=0.0,
    ),
}


def semantic_theta_summary():
    """Print theta analysis for example semantic systems."""
    print("=" * 85)
    print("SEMANTIC STRUCTURE THETA ANALYSIS (Information Organization)")
    print("=" * 85)
    print()
    print(f"{'System':<30} {'Format':<12} {'Coh':>6} {'Conn':>6} "
          f"{'Amb':>6} {'Grnd':>6} {'θ':>8} {'Level':<12}")
    print("-" * 85)

    for name, system in SEMANTIC_SYSTEMS.items():
        theta = compute_semantic_theta(system)
        level = classify_semantic_coherence(theta)
        fmt = system.structure_format.value[:10]

        print(f"{system.name:<30} "
              f"{fmt:<12} "
              f"{system.coherence_score:>6.2f} "
              f"{system.connectivity:>6.2f} "
              f"{system.ambiguity_score:>6.2f} "
              f"{system.grounding_ratio:>6.2f} "
              f"{theta:>8.3f} "
              f"{level.value:<12}")

    print()
    print("Key: θ combines coherence, connectivity, disambiguation, grounding")
    print("     Formal ontology represents highest semantic organization (θ ~ 0.95)")


if __name__ == "__main__":
    semantic_theta_summary()
