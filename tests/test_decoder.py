"""Tests for natural language reconstructor."""

from ailang_ir.decoder import Reconstructor
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
