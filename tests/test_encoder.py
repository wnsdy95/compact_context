"""Tests for symbolic encoder."""

from ailang_ir.encoder import SymbolicEncoder, ConceptTable
from ailang_ir.encoder.codebook import _certainty_to_hex, _hex_to_certainty
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


class TestCertaintyHex:
    def test_certainty_to_hex_boundaries(self):
        assert _certainty_to_hex(0.0) == "0"
        assert _certainty_to_hex(1.0) == "f"

    def test_certainty_round_trip(self):
        for h in "0123456789abcdef":
            val = _hex_to_certainty(h)
            assert 0.0 <= val <= 1.0
            assert _certainty_to_hex(val) == h

    def test_certainty_70_pct(self):
        h = _certainty_to_hex(0.70)
        recovered = _hex_to_certainty(h)
        assert abs(recovered - 0.70) < 0.1


class TestV2Encoder:
    def setup_method(self):
        self.encoder = SymbolicEncoder()
        self.ct = ConceptTable()

    def test_v2_basic_encode(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("test_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, self.ct)
        # Header: U + o + bl + hex(0.70) + n
        assert code.startswith("Uobl")
        assert "#test_concept" in code or "#test_conc" in code
        assert "|" not in code  # no pipes in v2

    def test_v2_encode_with_target(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity("graph_memory"),
            target=Entity("linear_text"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, self.ct)
        assert ">" in code

    def test_v2_header_length(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("x"),
            certainty=Certainty(0.5),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, self.ct)
        header = code.split()[0]
        assert len(header) == 6

    def test_v2_shorter_than_v1(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("1to1_sentence_mapping_difficult"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        v1 = self.encoder.encode(frame)
        v2 = self.encoder.encode_v2(frame, self.ct)
        assert len(v2) < len(v1)

    def test_v2_reref_shorter(self):
        """Re-referencing the same concept should use $id form."""
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("some_long_concept_name"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code1 = self.encoder.encode_v2(frame, self.ct)
        code2 = self.encoder.encode_v2(frame, self.ct)
        assert "#" in code1  # first mention defines
        assert "$" in code2  # second mention references
        assert len(code2) < len(code1)

    def test_v2_decode_fields(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("test_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, self.ct)
        fields = self.encoder.decode_fields_v2(code, self.ct)
        assert fields["speaker"] == "U"
        assert fields["mode"] == "o"
        assert fields["act"] == "bl"
        assert fields["time"] == "n"
        assert "object" in fields

    def test_v2_decode_with_target(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity("typed_models"),
            target=Entity("loose_dictionaries"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, self.ct)
        fields = self.encoder.decode_fields_v2(code, self.ct)
        assert "target" in fields

    def test_v2_disassemble(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("test_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v2(frame, self.ct)
        debug = self.encoder.disassemble(code, self.ct)
        assert "OPINION" in debug
        assert "BELIEVE" in debug
        assert "PRESENT" in debug

    def test_v2_all_acts(self):
        """All acts should encode to 2-char codes."""
        for act in SemanticAct:
            frame = SemanticFrame(
                predicate=Predicate(act),
                object=Entity(f"obj_{act.value}"),
            )
            ct = ConceptTable()
            code = self.encoder.encode_v2(frame, ct)
            header = code.split()[0]
            assert len(header) == 6

    def test_v2_all_modes(self):
        """All modes should encode to 1-char codes."""
        for mode in SemanticMode:
            frame = SemanticFrame(
                mode=mode,
                predicate=Predicate(SemanticAct.BELIEVE),
                object=Entity(f"obj_{mode.value}"),
            )
            ct = ConceptTable()
            code = self.encoder.encode_v2(frame, ct)
            header = code.split()[0]
            assert len(header) == 6

    def test_v2_deterministic(self):
        ct1 = ConceptTable()
        ct2 = ConceptTable()
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.SUGGEST),
            object=Entity("approach"),
            certainty=Certainty(0.6),
            time=TimeReference.FUTURE,
        )
        code1 = self.encoder.encode_v2(frame, ct1)
        code2 = self.encoder.encode_v2(frame, ct2)
        assert code1 == code2


class TestV3Encoder:
    def setup_method(self):
        self.encoder = SymbolicEncoder()
        self.ct = ConceptTable()

    def test_v3_shorter_than_v2(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("1to1_sentence_mapping_difficult"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        ct_v2 = ConceptTable()
        v2 = self.encoder.encode_v2(frame, ct_v2)
        v3 = self.encoder.encode_v3(frame, self.ct)
        assert len(v3) < len(v2)

    def test_v3_header_same_as_v2(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("test_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v3(frame, self.ct)
        header = code.split()[0]
        assert len(header) == 6
        assert code.startswith("Uobl")

    def test_v3_no_underscores_in_key(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("semantic_frames_approach"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v3(frame, self.ct)
        obj_part = code.split()[1]
        key = obj_part[1:]  # strip # prefix
        assert "_" not in key

    def test_v3_with_target(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity("graph_memory"),
            target=Entity("linear_text"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v3(frame, self.ct)
        assert ">" in code

    def test_v3_decode_round_trip(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("semantic_frames"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        code = self.encoder.encode_v3(frame, self.ct)
        # v3 uses same decode as v2
        fields = self.encoder.decode_fields_v2(code, self.ct)
        assert fields["speaker"] == "U"
        assert fields["mode"] == "o"
        assert fields["act"] == "bl"
        assert "object" in fields

    def test_v3_reref(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity("some_long_concept"),
            certainty=Certainty(0.70),
            time=TimeReference.PRESENT,
        )
        c1 = self.encoder.encode_v3(frame, self.ct)
        c2 = self.encoder.encode_v3(frame, self.ct)
        assert "#" in c1
        assert "$" in c2
        assert len(c2) < len(c1)

    def test_v3_deterministic(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.SUGGEST),
            object=Entity("normalization_rules"),
            certainty=Certainty(0.6),
            time=TimeReference.FUTURE,
        )
        ct1 = ConceptTable()
        ct2 = ConceptTable()
        assert self.encoder.encode_v3(frame, ct1) == self.encoder.encode_v3(frame, ct2)

    def test_v3_all_acts_valid_header(self):
        for act in SemanticAct:
            ct = ConceptTable()
            frame = SemanticFrame(
                predicate=Predicate(act),
                object=Entity(f"obj_{act.value}"),
            )
            code = self.encoder.encode_v3(frame, ct)
            assert len(code.split()[0]) == 6
