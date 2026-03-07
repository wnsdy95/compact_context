"""Tests for natural language reconstructor."""

from ailang_ir.decoder import Reconstructor
from ailang_ir.encoder import SymbolicEncoder, ConceptTable
from ailang_ir.models.domain import (
    Certainty,
    Entity,
    Predicate,
    ReconstructionPlan,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    SpeakerRole,
    TimeReference,
)


class TestReconstructor:
    def setup_method(self):
        self.recon = Reconstructor()

    def test_declarative_opinion(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("semantic_frames"),
            certainty=Certainty(0.75),
            time=TimeReference.PRESENT,
        )
        text = self.recon.reconstruct(frame)
        assert "think" in text.lower() or "semantic frames" in text.lower()

    def test_question_reconstruction(self):
        frame = SemanticFrame(
            mode=SemanticMode.QUESTION,
            predicate=Predicate(SemanticAct.QUERY),
            object=Entity("best_approach"),
        )
        text = self.recon.reconstruct(frame)
        assert "?" in text

    def test_summary_style(self):
        frame = SemanticFrame(
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity("graph_memory"),
            target=Entity("linear_text"),
            certainty=Certainty(0.83),
        )
        text = self.recon.reconstruct(frame, style="summary")
        assert "PREFER" in text
        assert "GRAPH_MEMORY" in text
        assert "LINEAR_TEXT" in text
        assert "C83" in text

    def test_from_code(self):
        text = self.recon.reconstruct_from_code("U|OP|BELIEVE|TEST_CONCEPT|C84|NOW")
        assert len(text) > 0
        assert "test concept" in text.lower()

    def test_from_plan(self):
        frame = SemanticFrame(
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.EXPLAIN),
            object=Entity("normalization_rules"),
        )
        plan = ReconstructionPlan(frame=frame, style="summary")
        text = self.recon.reconstruct_from_plan(plan)
        assert "EXPLAIN" in text

    def test_conversational_style(self):
        frame = SemanticFrame(
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("simple_design"),
            certainty=Certainty(0.7),
        )
        text = self.recon.reconstruct(frame, style="conversational")
        assert "think" in text.lower()

    def test_low_certainty_qualifier(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.OBSERVE),
            object=Entity("test_pattern"),
            certainty=Certainty(0.35),
        )
        text = self.recon.reconstruct(frame)
        # Low certainty should produce a qualifier
        lower = text.lower()
        assert "maybe" in lower or "possibly" in lower or "perhaps" in lower


class TestReconstructorV2:
    def setup_method(self):
        self.recon = Reconstructor()
        self.encoder = SymbolicEncoder()

    def test_reconstruct_from_v2_code(self):
        ct = ConceptTable()
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("test_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, ct)
        text = self.recon.reconstruct_from_code(code, "declarative", ct)
        assert len(text) > 0

    def test_v2_round_trip_meaning(self):
        ct = ConceptTable()
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("semantic_frames"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, ct)
        text = self.recon.reconstruct_from_code(code, "declarative", ct)
        lower = text.lower()
        assert "think" in lower or "semantic" in lower or "sema" in lower

    def test_v2_with_target_round_trip(self):
        ct = ConceptTable()
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity("graph_memory"),
            target=Entity("linear_text"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, ct)
        text = self.recon.reconstruct_from_code(code, "declarative", ct)
        assert len(text) > 0

    def test_v2_reref_decode(self):
        """Re-referenced concept ($id) should decode correctly."""
        ct = ConceptTable()
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("important_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        # First encode defines it
        self.encoder.encode_v2(frame, ct)
        # Second encode re-references
        code2 = self.encoder.encode_v2(frame, ct)
        assert "$" in code2
        text = self.recon.reconstruct_from_code(code2, "declarative", ct)
        assert len(text) > 0

    def test_v1_code_still_works(self):
        """v1 pipe-delimited code should still work without concept_table."""
        text = self.recon.reconstruct_from_code("U|OP|BELIEVE|TEST_CONCEPT|C84|NOW")
        assert "test concept" in text.lower()
