"""Tests for core domain models."""

import pytest
from ailang_ir.models.domain import (
    Certainty,
    Entity,
    Predicate,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    SpeakerRole,
    TimeReference,
    EventMemory,
    RelationEdge,
)


class TestCertainty:
    def test_valid_range(self):
        c = Certainty(0.84)
        assert c.value == 0.84

    def test_boundary_zero(self):
        c = Certainty(0.0)
        assert c.label == "speculative"

    def test_boundary_one(self):
        c = Certainty(1.0)
        assert c.label == "near_certain"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Certainty(1.5)
        with pytest.raises(ValueError):
            Certainty(-0.1)

    def test_compact_format(self):
        assert Certainty(0.84).compact() == "C84"
        assert Certainty(0.0).compact() == "C0"
        assert Certainty(1.0).compact() == "C100"

    def test_labels(self):
        assert Certainty(0.15).label == "speculative"
        assert Certainty(0.45).label == "hedged"
        assert Certainty(0.75).label == "confident"
        assert Certainty(0.95).label == "near_certain"


class TestEntity:
    def test_equality_by_canonical(self):
        e1 = Entity("graph_memory", surface_forms=("Graph memory",))
        e2 = Entity("graph_memory", surface_forms=("graph-based memory",))
        assert e1 == e2

    def test_different_canonical(self):
        e1 = Entity("graph_memory")
        e2 = Entity("linear_memory")
        assert e1 != e2

    def test_hashable(self):
        e1 = Entity("graph_memory")
        e2 = Entity("graph_memory")
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1


class TestPredicate:
    def test_canonical_without_modifier(self):
        p = Predicate(SemanticAct.BELIEVE)
        assert p.canonical == "believe"

    def test_canonical_with_modifier(self):
        p = Predicate(SemanticAct.BELIEVE, modifier="strongly")
        assert p.canonical == "strongly_believe"


class TestSemanticFrame:
    def test_default_frame(self):
        f = SemanticFrame()
        assert f.speaker == SpeakerRole.UNKNOWN
        assert f.mode == SemanticMode.ASSERTION
        assert f.act == SemanticAct.UNKNOWN

    def test_summary(self):
        f = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("semantic_frame_model"),
            certainty=Certainty(0.84),
            time=TimeReference.PRESENT,
        )
        summary = f.summary()
        assert "user" in summary
        assert "opinion" in summary
        assert "believe" in summary
        assert "semantic_frame_model" in summary
        assert "C84" in summary

    def test_frame_id_unique(self):
        f1 = SemanticFrame()
        f2 = SemanticFrame()
        assert f1.frame_id != f2.frame_id


class TestEventMemory:
    def test_active_by_default(self):
        m = EventMemory()
        assert m.is_active

    def test_superseded(self):
        m = EventMemory()
        m.superseded_by = "other_id"
        assert not m.is_active

    def test_touch(self):
        m = EventMemory()
        assert m.access_count == 0
        m.touch()
        assert m.access_count == 1
        assert m.last_accessed is not None


class TestRelationEdge:
    def test_edge_creation(self):
        e = RelationEdge(
            source=Entity("graph_memory"),
            target=Entity("linear_text"),
            relation="preferred_over",
        )
        assert e.source.canonical == "graph_memory"
        assert e.target.canonical == "linear_text"
        assert e.relation == "preferred_over"
