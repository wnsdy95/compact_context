"""Tests for symbolic encoder."""

from ailang_ir.encoder import SymbolicEncoder
from ailang_ir.models.domain import (
    Certainty,
    Entity,
    Predicate,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    SpeakerRole,
    TimeReference,
)


class TestSymbolicEncoder:
    def setup_method(self):
        self.encoder = SymbolicEncoder()

    def test_basic_encode(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("test_concept"),
            certainty=Certainty(0.84),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode(frame)
        assert code == "U|OP|BELIEVE|TEST_CONCEPT|C84|NOW"

    def test_encode_with_target(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity("graph_memory"),
            target=Entity("linear_text"),
            certainty=Certainty(0.83),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode(frame)
        assert "OVER:LINEAR_TEXT" in code
        assert code == "U|AS|PREFER|GRAPH_MEMORY|OVER:LINEAR_TEXT|C83|NOW"

    def test_deterministic(self):
        """Same frame should always produce same code."""
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.HYPOTHESIS,
            predicate=Predicate(SemanticAct.SUGGEST),
            object=Entity("new_approach"),
            certainty=Certainty(0.6),
            time=TimeReference.FUTURE,
        )
        code1 = self.encoder.encode(frame)
        code2 = self.encoder.encode(frame)
        assert code1 == code2

    def test_missing_object(self):
        frame = SemanticFrame(speaker=SpeakerRole.USER)
        code = self.encoder.encode(frame)
        assert "?OBJ" in code


class TestDecodeFields:
    def setup_method(self):
        self.encoder = SymbolicEncoder()

    def test_basic_decode(self):
        fields = self.encoder.decode_fields("U|OP|BELIEVE|TEST_CONCEPT|C84|NOW")
        assert fields["speaker"] == "U"
        assert fields["mode"] == "OP"
        assert fields["act"] == "BELIEVE"
        assert fields["object"] == "TEST_CONCEPT"
        assert fields["certainty"] == "C84"
        assert fields["time"] == "NOW"

    def test_decode_with_target(self):
        fields = self.encoder.decode_fields("U|AS|PREFER|GRAPH_MEMORY|OVER:LINEAR_TEXT|C83|NOW")
        assert fields["object"] == "GRAPH_MEMORY"
        assert fields["target"] == "LINEAR_TEXT"

    def test_decode_certainty_value(self):
        assert self.encoder.decode_certainty("C84") == 0.84
        assert self.encoder.decode_certainty("C0") == 0.0
        assert self.encoder.decode_certainty("C100") == 1.0

    def test_round_trip(self):
        """Encode then decode should preserve all fields."""
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("round_trip_test"),
            certainty=Certainty(0.75),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode(frame)
        fields = self.encoder.decode_fields(code)
        assert fields["speaker"] == "U"
        assert fields["mode"] == "OP"
        assert fields["act"] == "BELIEVE"
        assert fields["object"] == "ROUND_TRIP_TEST"
        assert self.encoder.decode_certainty(fields["certainty"]) == 0.75
