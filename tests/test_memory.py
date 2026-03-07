"""Tests for memory store."""

from ailang_ir.memory import MemoryStore
from ailang_ir.models.domain import (
    Certainty,
    Entity,
    Predicate,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    SpeakerRole,
    TimeReference,
    RelationEdge,
)


def _make_frame(act: SemanticAct, obj_key: str, speaker: SpeakerRole = SpeakerRole.USER) -> SemanticFrame:
    return SemanticFrame(
        speaker=speaker,
        predicate=Predicate(act),
        object=Entity(obj_key),
        certainty=Certainty(0.7),
    )


class TestMemoryStore:
    def setup_method(self):
        self.store = MemoryStore()

    def test_store_and_retrieve(self):
        frame = _make_frame(SemanticAct.BELIEVE, "test_concept")
        mem = self.store.store(frame)
        assert self.store.size == 1
        retrieved = self.store.get(mem.memory_id)
        assert retrieved is not None
        assert retrieved.frame.act == SemanticAct.BELIEVE

    def test_deduplication(self):
        frame1 = _make_frame(SemanticAct.BELIEVE, "test_concept")
        frame2 = _make_frame(SemanticAct.BELIEVE, "test_concept")
        self.store.store(frame1)
        self.store.store(frame2)
        assert self.store.size == 1

    def test_different_act_not_deduplicated(self):
        frame1 = _make_frame(SemanticAct.BELIEVE, "test_concept")
        frame2 = _make_frame(SemanticAct.PREFER, "test_concept")
        self.store.store(frame1)
        self.store.store(frame2)
        assert self.store.size == 2

    def test_query_by_entity(self):
        self.store.store(_make_frame(SemanticAct.BELIEVE, "concept_a"))
        self.store.store(_make_frame(SemanticAct.PREFER, "concept_b"))
        results = self.store.query_by_entity("concept_a")
        assert len(results) == 1
        assert results[0].frame.object.canonical == "concept_a"

    def test_query_by_act(self):
        self.store.store(_make_frame(SemanticAct.BELIEVE, "a"))
        self.store.store(_make_frame(SemanticAct.BELIEVE, "b"))
        self.store.store(_make_frame(SemanticAct.PREFER, "c"))
        results = self.store.query_by_act(SemanticAct.BELIEVE)
        assert len(results) == 2

    def test_query_by_speaker(self):
        self.store.store(_make_frame(SemanticAct.BELIEVE, "a", SpeakerRole.USER))
        self.store.store(_make_frame(SemanticAct.BELIEVE, "b", SpeakerRole.SYSTEM))
        results = self.store.query_by_speaker(SpeakerRole.USER)
        assert len(results) == 1

    def test_query_by_tag(self):
        self.store.store(_make_frame(SemanticAct.BELIEVE, "a"), tags=["important"])
        self.store.store(_make_frame(SemanticAct.PREFER, "b"), tags=["draft"])
        results = self.store.query_by_tag("important")
        assert len(results) == 1

    def test_query_recent(self):
        for i in range(5):
            self.store.store(_make_frame(SemanticAct.BELIEVE, f"concept_{i}"))
        recent = self.store.query_recent(3)
        assert len(recent) == 3

    def test_supersede(self):
        frame1 = _make_frame(SemanticAct.BELIEVE, "old_idea")
        mem1 = self.store.store(frame1)
        frame2 = _make_frame(SemanticAct.BELIEVE, "new_idea")
        mem2 = self.store.supersede(mem1.memory_id, frame2)
        assert not mem1.is_active
        assert mem1.superseded_by == mem2.memory_id
        assert self.store.active_count == 1

    def test_contradiction_detection(self):
        self.store.store(_make_frame(SemanticAct.PREFER, "approach_a"))
        contra = _make_frame(SemanticAct.REJECT, "approach_a")
        contradictions = self.store.find_contradictions(contra)
        assert len(contradictions) == 1

    def test_no_contradiction_different_speaker(self):
        self.store.store(_make_frame(SemanticAct.PREFER, "approach_a", SpeakerRole.USER))
        contra = _make_frame(SemanticAct.REJECT, "approach_a", SpeakerRole.SYSTEM)
        contradictions = self.store.find_contradictions(contra)
        assert len(contradictions) == 0

    def test_edges(self):
        edge = RelationEdge(
            source=Entity("graph_memory"),
            target=Entity("linear_text"),
            relation="preferred_over",
        )
        self.store.store_edge(edge)
        results = self.store.get_edges_for("graph_memory")
        assert len(results) == 1
        assert results[0].relation == "preferred_over"

    def test_dump(self):
        self.store.store(_make_frame(SemanticAct.BELIEVE, "test"))
        dump = self.store.dump()
        assert len(dump) == 1
        assert "memory_id" in dump[0]
        assert "summary" in dump[0]


class TestFullPipelineRoundTrip:
    """End-to-end test: text → parse → store → retrieve → encode → decode."""

    def test_round_trip(self):
        from ailang_ir.parser import RuleBasedParser
        from ailang_ir.encoder import SymbolicEncoder
        from ailang_ir.decoder import Reconstructor

        parser = RuleBasedParser()
        encoder = SymbolicEncoder()
        decoder = Reconstructor()
        store = MemoryStore()

        # Parse and store
        text = "I think semantic frames are important."
        frame = parser.parse(text)
        mem = store.store(frame)

        # Retrieve and encode
        retrieved = store.get(mem.memory_id)
        code = encoder.encode(retrieved.frame)

        # Decode
        reconstructed = decoder.reconstruct_from_code(code)

        # Verify meaning preservation
        assert "semantic" in reconstructed.lower() or "SEMANTIC" in code
        assert retrieved.frame.act == SemanticAct.BELIEVE
        assert retrieved.access_count >= 1
