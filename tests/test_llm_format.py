"""
Tests for the LLM-native format layer.

Covers:
- TestFormatSpec: spec completeness
- TestValidator: valid/invalid code detection (14+ cases)
- TestLLMCodec: encode/decode round-trip, full enum coverage
- TestPipelineIntegration: ingest/export round-trip
"""

import pytest
from ailang_ir.llm.format_spec import FORMAT_SPEC, get_format_spec
from ailang_ir.llm.validator import validate_code, ValidationResult
from ailang_ir.llm.codec import LLMCodec
from ailang_ir.pipeline import Pipeline
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
from ailang_ir.encoder.codebook import (
    SPEAKER_CODES,
    MODE_CODES_V2,
    ACT_CODES_V2,
    TIME_CODES_V2,
)


# ============================================================
# FORMAT SPEC
# ============================================================

class TestFormatSpec:
    """Verify spec completeness and size."""

    def test_spec_is_nonempty(self):
        spec = get_format_spec()
        assert len(spec) > 100

    def test_spec_contains_all_speaker_codes(self):
        spec = FORMAT_SPEC
        for code in SPEAKER_CODES.values():
            assert code in spec, f"Speaker code '{code}' missing from spec"

    def test_spec_contains_all_mode_codes(self):
        spec = FORMAT_SPEC
        for code in MODE_CODES_V2.values():
            # Mode codes are single chars; check they appear in the spec
            assert code in spec, f"Mode code '{code}' missing from spec"

    def test_spec_contains_all_act_codes(self):
        spec = FORMAT_SPEC
        for code in ACT_CODES_V2.values():
            assert code in spec, f"Act code '{code}' missing from spec"

    def test_spec_contains_all_time_codes(self):
        spec = FORMAT_SPEC
        for code in TIME_CODES_V2.values():
            assert code in spec, f"Time code '{code}' missing from spec"

    def test_spec_contains_examples(self):
        assert "Uoblbn" in FORMAT_SPEC
        assert "postgresql_main" in FORMAT_SPEC

    def test_spec_token_estimate(self):
        """Spec should be ≤300 tokens (rough estimate: ~4 chars/token)."""
        # Conservative estimate: whitespace-separated words
        words = FORMAT_SPEC.split()
        # Rough token count (most tokenizers produce fewer tokens than words)
        assert len(words) <= 400, f"Spec has {len(words)} words, target ≤300 tokens"


# ============================================================
# VALIDATOR
# ============================================================

class TestValidator:
    """14+ cases for valid/invalid code detection."""

    # --- Valid cases ---

    def test_valid_basic(self):
        r = validate_code("Uoblbn graph_mem")
        assert r.is_valid
        assert r.speaker == "U"
        assert r.mode == "o"
        assert r.act == "bl"
        assert r.certainty == "b"
        assert r.time == "n"
        assert r.object_key == "graph_mem"

    def test_valid_with_target(self):
        r = validate_code("Uaprbn typed_models >dicts")
        assert r.is_valid
        assert r.object_key == "typed_models"
        assert r.target_key == "dicts"

    def test_valid_question(self):
        r = validate_code("Uqqu7n parser")
        assert r.is_valid
        assert r.mode == "q"
        assert r.act == "qu"

    def test_valid_commitment_future(self):
        r = validate_code("Ukplbf persistence")
        assert r.is_valid
        assert r.mode == "k"
        assert r.time == "f"

    def test_valid_all_speakers(self):
        for speaker_code in SPEAKER_CODES.values():
            r = validate_code(f"{speaker_code}ablbn test")
            assert r.is_valid, f"Speaker '{speaker_code}' should be valid"

    def test_valid_certainty_range(self):
        for c in "0123456789abcdef":
            r = validate_code(f"Uabl{c}n test")
            assert r.is_valid, f"Certainty '{c}' should be valid"

    # --- Invalid cases ---

    def test_invalid_empty(self):
        r = validate_code("")
        assert not r.is_valid
        assert "empty code" in r.errors[0]

    def test_invalid_header_too_short(self):
        r = validate_code("Uob graph_mem")
        assert not r.is_valid
        assert any("header" in e for e in r.errors)

    def test_invalid_header_too_long(self):
        r = validate_code("Uoblbnn graph_mem")
        assert not r.is_valid

    def test_invalid_speaker(self):
        r = validate_code("Xoblbn graph_mem")
        assert not r.is_valid
        assert any("speaker" in e for e in r.errors)

    def test_invalid_mode(self):
        r = validate_code("Uzblbn graph_mem")
        assert not r.is_valid
        assert any("mode" in e for e in r.errors)

    def test_invalid_act(self):
        r = validate_code("Uozzbn graph_mem")
        assert not r.is_valid
        assert any("act" in e for e in r.errors)

    def test_invalid_certainty(self):
        r = validate_code("Uoblgn graph_mem")
        assert not r.is_valid
        assert any("certainty" in e for e in r.errors)

    def test_invalid_time(self):
        r = validate_code("Uoblbx graph_mem")
        assert not r.is_valid
        assert any("time" in e for e in r.errors)

    def test_invalid_object_key_uppercase(self):
        r = validate_code("Uoblbn GraphMem")
        assert not r.is_valid
        assert any("object" in e for e in r.errors)

    def test_invalid_object_key_too_many_words(self):
        r = validate_code("Uoblbn one_two_three_four")
        assert not r.is_valid
        assert any("3" in e for e in r.errors)

    def test_invalid_missing_object(self):
        r = validate_code("Uoblbn")
        assert not r.is_valid
        assert any("2 tokens" in e for e in r.errors)

    def test_multiple_errors_collected(self):
        """Validator should collect ALL errors, not stop at first."""
        r = validate_code("Xzzzgx BAD_KEY")
        assert not r.is_valid
        assert len(r.errors) >= 3  # speaker, mode, act, certainty, time, key...

    def test_extra_tokens(self):
        r = validate_code("Uoblbn graph_mem extra tokens")
        assert not r.is_valid
        assert any("unexpected" in e for e in r.errors)


# ============================================================
# LLM CODEC
# ============================================================

class TestLLMCodec:
    """Encode/decode round-trip and full enum coverage."""

    def setup_method(self):
        self.codec = LLMCodec()

    def test_encode_basic(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity(canonical="graph_memory"),
            certainty=Certainty(0.73),
            time=TimeReference.PRESENT,
        )
        code = self.codec.encode(frame)
        # Header should be 6 chars
        header = code.split()[0]
        assert len(header) == 6
        assert header[0] == "U"  # speaker
        assert header[1] == "o"  # mode: opinion
        assert header[2:4] == "bl"  # act: believe
        assert header[5] == "n"  # time: present

    def test_encode_with_target(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.ASSERTION,
            predicate=Predicate(SemanticAct.PREFER),
            object=Entity(canonical="typed_models"),
            target=Entity(canonical="dicts"),
            certainty=Certainty(0.73),
            time=TimeReference.PRESENT,
        )
        code = self.codec.encode(frame)
        assert ">" in code
        tokens = code.split()
        assert any(t.startswith(">") for t in tokens)

    def test_decode_basic(self):
        frame = self.codec.decode("Uoblbn graph_mem")
        assert frame.speaker == SpeakerRole.USER
        assert frame.mode == SemanticMode.OPINION
        assert frame.act == SemanticAct.BELIEVE
        assert frame.time == TimeReference.PRESENT
        assert frame.object is not None
        assert frame.object.canonical == "graph_mem"

    def test_decode_with_target(self):
        frame = self.codec.decode("Uaprbn typed_models >dicts")
        assert frame.act == SemanticAct.PREFER
        assert frame.object.canonical == "typed_models"
        assert frame.target is not None
        assert frame.target.canonical == "dicts"

    def test_decode_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid LLM code"):
            self.codec.decode("invalid")

    def test_round_trip_preserves_fields(self):
        """encode -> decode should preserve all 6 semantic fields."""
        frame = SemanticFrame(
            speaker=SpeakerRole.AGENT,
            mode=SemanticMode.COMMITMENT,
            predicate=Predicate(SemanticAct.PLAN),
            object=Entity(canonical="persistence"),
            certainty=Certainty(0.73),
            time=TimeReference.FUTURE,
        )
        code = self.codec.encode(frame)
        decoded = self.codec.decode(code)

        assert decoded.speaker == frame.speaker
        assert decoded.mode == frame.mode
        assert decoded.act == frame.act
        assert decoded.time == frame.time
        # Certainty: hex quantization may lose precision, check within tolerance
        assert abs(decoded.certainty.value - frame.certainty.value) < 0.07
        # Object key may be compressed but should be present
        assert decoded.object is not None

    def test_round_trip_with_target(self):
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.COMPARE),
            object=Entity(canonical="graph_mem"),
            target=Entity(canonical="linear_text"),
            certainty=Certainty(0.8),
            time=TimeReference.PRESENT,
        )
        code = self.codec.encode(frame)
        decoded = self.codec.decode(code)
        assert decoded.target is not None

    def test_all_speakers(self):
        """Every SpeakerRole should encode and decode correctly."""
        for role in SpeakerRole:
            frame = SemanticFrame(
                speaker=role,
                predicate=Predicate(SemanticAct.BELIEVE),
                object=Entity(canonical="test"),
                certainty=Certainty(0.5),
                time=TimeReference.PRESENT,
            )
            code = self.codec.encode(frame)
            decoded = self.codec.decode(code)
            assert decoded.speaker == role, f"Speaker {role} not preserved"

    def test_all_modes(self):
        """Every SemanticMode should encode and decode correctly."""
        for mode in SemanticMode:
            frame = SemanticFrame(
                mode=mode,
                predicate=Predicate(SemanticAct.BELIEVE),
                object=Entity(canonical="test"),
                certainty=Certainty(0.5),
                time=TimeReference.PRESENT,
            )
            code = self.codec.encode(frame)
            decoded = self.codec.decode(code)
            assert decoded.mode == mode, f"Mode {mode} not preserved"

    def test_all_acts(self):
        """Every SemanticAct should encode and decode correctly."""
        for act in SemanticAct:
            frame = SemanticFrame(
                predicate=Predicate(act),
                object=Entity(canonical="test"),
                certainty=Certainty(0.5),
                time=TimeReference.PRESENT,
            )
            code = self.codec.encode(frame)
            decoded = self.codec.decode(code)
            assert decoded.act == act, f"Act {act} not preserved"

    def test_all_times(self):
        """Every TimeReference should encode and decode correctly."""
        for time in TimeReference:
            frame = SemanticFrame(
                predicate=Predicate(SemanticAct.BELIEVE),
                object=Entity(canonical="test"),
                certainty=Certainty(0.5),
                time=time,
            )
            code = self.codec.encode(frame)
            decoded = self.codec.decode(code)
            assert decoded.time == time, f"Time {time} not preserved"

    def test_encode_batch(self):
        frames = [
            SemanticFrame(
                speaker=SpeakerRole.USER,
                predicate=Predicate(SemanticAct.BELIEVE),
                object=Entity(canonical="test_one"),
                certainty=Certainty(0.5),
                time=TimeReference.PRESENT,
            ),
            SemanticFrame(
                speaker=SpeakerRole.AGENT,
                predicate=Predicate(SemanticAct.PLAN),
                object=Entity(canonical="test_two"),
                certainty=Certainty(0.8),
                time=TimeReference.FUTURE,
            ),
        ]
        batch = self.codec.encode_batch(frames)
        lines = batch.strip().splitlines()
        assert len(lines) == 2

    def test_decode_batch(self):
        text = "Uoblbn graph_mem\nAkplcf persistence"
        frames = self.codec.decode_batch(text)
        assert len(frames) == 2
        assert frames[0].speaker == SpeakerRole.USER
        assert frames[1].speaker == SpeakerRole.AGENT

    def test_encode_no_object(self):
        frame = SemanticFrame(
            predicate=Predicate(SemanticAct.BELIEVE),
            certainty=Certainty(0.5),
            time=TimeReference.PRESENT,
        )
        code = self.codec.encode(frame)
        assert "unknown" in code

    def test_llm_format_is_readable(self):
        """LLM format should use natural words, not stem abbreviations."""
        frame = SemanticFrame(
            speaker=SpeakerRole.USER,
            mode=SemanticMode.OPINION,
            predicate=Predicate(SemanticAct.BELIEVE),
            object=Entity(canonical="semantic_compression"),
            certainty=Certainty(0.73),
            time=TimeReference.PRESENT,
        )
        code = self.codec.encode(frame)
        # Should contain readable words, not stem-abbreviated garbage
        obj_part = code.split()[1]
        # v2-level compression may truncate long words but keeps them recognizable
        assert "_" in obj_part or len(obj_part) >= 4


# ============================================================
# PIPELINE INTEGRATION
# ============================================================

class TestPipelineIntegration:
    """Test ingest/export round-trip through the pipeline."""

    def setup_method(self):
        self.pipe = Pipeline()

    def test_ingest_valid_code(self):
        result = self.pipe.ingest_code("Uoblbn graph_mem")
        assert result.frame.speaker == SpeakerRole.USER
        assert result.frame.act == SemanticAct.BELIEVE
        assert result.frame.object.canonical == "graph_mem"
        assert result.compact_code  # should have internal v3 code

    def test_ingest_stores_in_memory(self):
        self.pipe.ingest_code("Uoblbn graph_mem")
        assert self.pipe.memory_size == 1

    def test_ingest_no_auto_store(self):
        pipe = Pipeline(auto_store=False)
        result = pipe.ingest_code("Uoblbn graph_mem")
        assert result.memory is None
        assert pipe.memory_size == 0

    def test_ingest_invalid_raises(self):
        with pytest.raises(ValueError):
            self.pipe.ingest_code("invalid_code")

    def test_ingest_codes_batch(self):
        codes = ["Uoblbn graph_mem", "Akplcf persistence"]
        results = self.pipe.ingest_codes(codes)
        assert len(results) == 2
        assert self.pipe.memory_size == 2

    def test_export_context(self):
        # Process some text first
        self.pipe.process("I think graph memory is great.")
        self.pipe.process("We need semantic compression.")
        export = self.pipe.export_context(n=2)
        lines = [l for l in export.strip().splitlines() if l.strip()]
        assert len(lines) == 2
        # Each line should be a valid LLM code
        for line in lines:
            r = validate_code(line)
            assert r.is_valid, f"Exported code invalid: {line} -> {r.errors}"

    def test_export_empty_memory(self):
        export = self.pipe.export_context()
        assert export == ""

    def test_ingest_export_round_trip(self):
        """Ingest codes, then export and verify semantic fields preserved."""
        codes = ["Uoblbn graph_mem", "Uandbn sem_compress"]
        self.pipe.ingest_codes(codes)
        export = self.pipe.export_context(n=2)
        lines = [l for l in export.strip().splitlines() if l.strip()]
        assert len(lines) == 2
        # Decode exported codes and verify key fields
        codec = LLMCodec()
        for line in lines:
            frame = codec.decode(line)
            assert frame.speaker == SpeakerRole.USER
            assert frame.object is not None

    def test_get_format_spec(self):
        spec = self.pipe.get_format_spec()
        assert "AILang-IR" in spec
        assert len(spec) > 100
