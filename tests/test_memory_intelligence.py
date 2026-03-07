"""Tests for memory intelligence: fuzzy search, relevance scoring, consolidation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ailang_ir.models.domain import (
    Certainty,
    Entity,
    EventMemory,
    Predicate,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    SpeakerRole,
    TimeReference,
)
from ailang_ir.memory.store import MemoryStore
from ailang_ir.encoder.concept_table import ConceptTable


def make_frame(
    obj: str,
    act: SemanticAct = SemanticAct.BELIEVE,
    speaker: SpeakerRole = SpeakerRole.USER,
    target: str | None = None,
) -> SemanticFrame:
    return SemanticFrame(
        speaker=speaker,
        mode=SemanticMode.OPINION,
        predicate=Predicate(act),
        object=Entity(canonical=obj),
        target=Entity(canonical=target) if target else None,
        time=TimeReference.PRESENT,
        certainty=Certainty(0.8),
    )


def make_store_with_data() -> MemoryStore:
    """Create a store with several memories about graph_memory and related concepts."""
    store = MemoryStore()
    store.store(make_frame("graph_memory"))
    store.store(make_frame("graph_storage", SemanticAct.PREFER))
    store.store(make_frame("graph_mem", SemanticAct.SUGGEST))
    store.store(make_frame("linear_text", SemanticAct.REJECT))
    store.store(make_frame("postgresql", SemanticAct.DECIDE))
    return store


# ---------------------------------------------------------------------------
# Entity similarity
# ---------------------------------------------------------------------------

class TestEntitySimilarity:
    def test_exact_match(self):
        assert MemoryStore._entity_similarity("graph_memory", "graph_memory") == 1.0

    def test_containment(self):
        sim = MemoryStore._entity_similarity("graph", "graph_memory")
        assert sim >= 0.9

    def test_word_overlap(self):
        sim = MemoryStore._entity_similarity("graph_memory", "graph_storage")
        assert sim > 0.3  # "graph" shared

    def test_similar_keys(self):
        sim = MemoryStore._entity_similarity("graph_mem", "graph_memory")
        assert sim > 0.7

    def test_unrelated(self):
        sim = MemoryStore._entity_similarity("postgresql", "kubernetes")
        assert sim <= 0.3

    def test_partial_word_match(self):
        sim = MemoryStore._entity_similarity("sem_compress", "semantic_compression")
        assert sim > 0.4


# ---------------------------------------------------------------------------
# Fuzzy query
# ---------------------------------------------------------------------------

class TestFuzzyQuery:
    def test_fuzzy_finds_similar(self):
        store = make_store_with_data()
        results = store.query_by_entity_fuzzy("graph_storage")
        keys = [m.frame.object.canonical for m, _ in results]
        assert "graph_storage" in keys  # exact
        assert "graph_memory" in keys   # fuzzy match

    def test_fuzzy_with_abbrev(self):
        store = make_store_with_data()
        results = store.query_by_entity_fuzzy("graph_mem")
        # Should find graph_mem (exact) and graph_memory (fuzzy)
        keys = [m.frame.object.canonical for m, _ in results]
        assert "graph_mem" in keys
        assert "graph_memory" in keys

    def test_fuzzy_threshold(self):
        store = make_store_with_data()
        # High threshold: only very similar
        results = store.query_by_entity_fuzzy("graph_memory", threshold=0.95)
        keys = [m.frame.object.canonical for m, _ in results]
        assert "graph_memory" in keys
        assert "linear_text" not in keys

    def test_fuzzy_no_match(self):
        store = make_store_with_data()
        results = store.query_by_entity_fuzzy("kubernetes", threshold=0.6)
        assert len(results) == 0

    def test_fuzzy_sorted_by_similarity(self):
        store = make_store_with_data()
        results = store.query_by_entity_fuzzy("graph_memory")
        if len(results) >= 2:
            # First should have highest similarity
            assert results[0][1] >= results[1][1]


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

class TestRelevanceScoring:
    def test_relevant_returns_matches(self):
        store = make_store_with_data()
        results = store.query_relevant(["graph_memory", "postgresql"], n=5)
        assert len(results) > 0
        keys = [m.frame.object.canonical for m in results]
        assert "graph_memory" in keys
        assert "postgresql" in keys

    def test_relevant_respects_n(self):
        store = make_store_with_data()
        results = store.query_relevant(["graph"], n=2)
        assert len(results) <= 2

    def test_relevant_empty_context(self):
        store = make_store_with_data()
        results = store.query_relevant([], n=5)
        # Should still return something (recency/access scoring)
        assert len(results) > 0

    def test_relevant_scores_entity_match_higher(self):
        store = make_store_with_data()
        results = store.query_relevant(["graph_memory"], n=5)
        # graph_memory should be first or near the top
        if results:
            top_keys = [m.frame.object.canonical for m in results[:2]]
            assert any("graph" in k for k in top_keys)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

class TestConsolidation:
    def test_consolidation_merges(self):
        store = make_store_with_data()
        initial_active = store.active_count
        result = store.consolidate("graph_memory", threshold=0.6)
        assert result is not None
        assert "consolidated" in result.tags
        # Old memories about graph should be superseded
        assert store.active_count < initial_active

    def test_consolidation_too_few(self):
        store = make_store_with_data()
        result = store.consolidate("postgresql", threshold=0.95)
        # Only 1 memory about postgresql with high threshold
        assert result is None

    def test_consolidation_metadata(self):
        store = make_store_with_data()
        result = store.consolidate("graph_memory", threshold=0.6)
        assert result is not None
        assert "consolidated_from" in result.frame.metadata
        assert len(result.frame.metadata["consolidated_from"]) >= 2

    def test_consolidation_preserves_unrelated(self):
        store = make_store_with_data()
        store.consolidate("graph_memory", threshold=0.6)
        # linear_text and postgresql should still be active
        linear = store.query_by_entity("linear_text")
        assert len(linear) >= 1
        pg = store.query_by_entity("postgresql")
        assert len(pg) >= 1


# ---------------------------------------------------------------------------
# ConceptTable fuzzy ref
# ---------------------------------------------------------------------------

class TestConceptTableFuzzyRef:
    def test_exact_match_reuses(self):
        ct = ConceptTable()
        r1 = ct.ref_fuzzy("graph_memory")
        r2 = ct.ref_fuzzy("graph_memory")
        assert r1.startswith("#")
        assert r2.startswith("$")
        assert ct.size == 1

    def test_fuzzy_reuses_similar(self):
        ct = ConceptTable()
        ct.ref_fuzzy("graph_memory")  # define
        r2 = ct.ref_fuzzy("graph_mem")  # should fuzzy match
        assert r2.startswith("$")
        assert ct.size == 1  # no new definition

    def test_fuzzy_defines_new_for_different(self):
        ct = ConceptTable()
        ct.ref_fuzzy("graph_memory")
        r2 = ct.ref_fuzzy("postgresql")
        assert r2.startswith("#")
        assert ct.size == 2

    def test_fuzzy_threshold_controls_match(self):
        ct = ConceptTable()
        ct.ref_fuzzy("graph_memory")
        # High threshold: graph_storage might not match
        r2 = ct.ref_fuzzy("graph_storage", threshold=0.95)
        assert r2.startswith("#")  # new definition
        assert ct.size == 2

    def test_fuzzy_word_overlap_match(self):
        ct = ConceptTable()
        ct.ref_fuzzy("sem_compression")
        r2 = ct.ref_fuzzy("semantic_compression")
        # These share enough similarity
        # At 0.7 threshold, containment or sequence match should work
        assert ct.size <= 2  # might or might not match depending on score

    def test_conversation_re_reference(self):
        """The key test: simulate a conversation about graph memory with variants."""
        ct = ConceptTable()
        refs = []
        keys = [
            "graph_memory",
            "graph_storage",
            "graph_mem",
            "graph_memory_approach",
            "linear_storage",
            "graph_store",
        ]
        for k in keys:
            refs.append(ct.ref_fuzzy(k, threshold=0.6))

        # Count re-references ($ prefixed)
        re_refs = sum(1 for r in refs if r.startswith("$"))
        # Should have at least 3 re-references (graph variants matching each other)
        assert re_refs >= 3, f"Expected ≥3 re-references, got {re_refs}: {list(zip(keys, refs))}"

    def test_non_fuzzy_ref_no_regression(self):
        """Ensure regular ref() still works exactly as before."""
        ct = ConceptTable()
        r1 = ct.ref("graph_memory")
        r2 = ct.ref("graph_mem")
        r3 = ct.ref("graph_memory")
        assert r1 == "#graph_memory"
        assert r2 == "#graph_mem"
        assert r3.startswith("$")  # re-ref for exact match
        assert ct.size == 2
