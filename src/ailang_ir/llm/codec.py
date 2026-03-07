"""
LLM Codec: bidirectional conversion between SemanticFrame and LLM format.

The LLM format uses the same 6-char header as v2/v3 encoding but with
natural-word object keys (v2-level compression, no stem abbreviation).

This is an interface layer — not a replacement for v3 internal storage.

  LLM <-> LLM Codec (natural keys) <-> SemanticFrame <-> v3 Encoder (storage)
"""

from __future__ import annotations

from dataclasses import dataclass

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
    SPEAKER_DECODE,
    MODE_CODES_V2,
    MODE_DECODE_V2,
    ACT_CODES_V2,
    ACT_DECODE_V2,
    TIME_CODES_V2,
    TIME_DECODE_V2,
    _certainty_to_hex,
    _hex_to_certainty,
)
from ailang_ir.normalize.vocabulary import NormalizationVocabulary
from ailang_ir.llm.validator import validate_code


_LLM_KEY_FILLERS = {
    "for", "of", "with", "about", "to", "in", "on", "at",
    "from", "by", "than", "over", "into", "like", "between",
    "through", "after", "before", "under", "against",
    "using", "instead", "also", "well",
}


def _llm_key(vocab: NormalizationVocabulary, canonical: str) -> str:
    """
    Compress a canonical key for LLM format.

    If the key is already compact (≤2 lowercase words), pass through
    for round-trip stability. Otherwise remove fillers and cap at 2
    content words — but keep words INTACT (no truncation) so LLMs
    can understand the key.
    """
    words = canonical.split("_")
    already_compact = (
        1 <= len(words) <= 2
        and all(w.isalnum() for w in words if w)
        and canonical == canonical.lower()
    )
    if already_compact:
        return canonical
    # Remove filler words but keep content words intact (no truncation)
    content = [w for w in words if w and w not in _LLM_KEY_FILLERS]
    if not content:
        content = words[:2]
    return "_".join(content[:2])


@dataclass
class LLMCodec:
    """
    Bidirectional codec between SemanticFrame and LLM-native format.

    Encode: SemanticFrame -> LLM format string (natural-word keys)
    Decode: LLM format string -> SemanticFrame
    """

    def encode(self, frame: SemanticFrame) -> str:
        """
        Encode a SemanticFrame into LLM-readable format.

        Format: HEADER object_key [>target_key]
        Header: S(1) M(1) AA(2) C(1) T(1) = 6 chars
        Object key: v2-level compression (natural words, no stem abbreviation)
        """
        vocab = NormalizationVocabulary()

        # Build 6-char header
        s = SPEAKER_CODES.get(frame.speaker, "?")
        m = MODE_CODES_V2.get(frame.mode, "?")
        aa = ACT_CODES_V2.get(frame.act, "uk")
        c = _certainty_to_hex(frame.certainty.value)
        t = TIME_CODES_V2.get(frame.time, "?")
        header = f"{s}{m}{aa}{c}{t}"

        # Object key: v2 compression + max 2 words for token efficiency
        if frame.object:
            obj_key = _llm_key(vocab, frame.object.canonical)
        else:
            obj_key = "unknown"

        parts = [header, obj_key]

        # Target key (for comparisons)
        if frame.target:
            tgt_key = _llm_key(vocab, frame.target.canonical)
            parts.append(f">{tgt_key}")

        return " ".join(parts)

    def decode(self, code: str) -> SemanticFrame:
        """
        Decode an LLM-produced code into a SemanticFrame.

        Validates the code first; raises ValueError if structurally invalid.
        """
        result = validate_code(code)
        if not result.is_valid:
            raise ValueError(
                f"Invalid LLM code: {'; '.join(result.errors)}"
            )

        # Map header fields to enums
        speaker = SPEAKER_DECODE.get(result.speaker, SpeakerRole.UNKNOWN)  # type: ignore[arg-type]
        mode = MODE_DECODE_V2.get(result.mode, SemanticMode.ASSERTION)  # type: ignore[arg-type]
        act = ACT_DECODE_V2.get(result.act, SemanticAct.UNKNOWN)  # type: ignore[arg-type]
        certainty_val = _hex_to_certainty(result.certainty)  # type: ignore[arg-type]
        time = TIME_DECODE_V2.get(result.time, TimeReference.UNKNOWN)  # type: ignore[arg-type]

        # Build entity from object key
        obj_entity = Entity(canonical=result.object_key) if result.object_key else None

        # Build target entity if present
        tgt_entity = Entity(canonical=result.target_key) if result.target_key else None

        return SemanticFrame(
            speaker=speaker,
            mode=mode,
            predicate=Predicate(act),
            object=obj_entity,
            target=tgt_entity,
            time=time,
            certainty=Certainty(certainty_val),
        )

    def encode_batch(self, frames: list[SemanticFrame]) -> str:
        """Encode multiple frames into newline-separated LLM format."""
        return "\n".join(self.encode(f) for f in frames)

    def decode_batch(self, text: str) -> list[SemanticFrame]:
        """Decode newline-separated LLM format codes into frames."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return [self.decode(line) for line in lines]
