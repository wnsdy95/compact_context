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

# Acts where stance is critical for understanding — append #label in export
_STANCE_ACTS = {
    SemanticAct.AGREE: "agree",
    SemanticAct.DISAGREE: "disagree",
    SemanticAct.PREFER: "prefer",
    SemanticAct.REJECT: "reject",
    SemanticAct.SUGGEST: "suggest",
    SemanticAct.DECIDE: "decide",
}
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

import re

_LLM_KEY_FILLERS = {
    "for", "of", "with", "about", "to", "in", "on", "at",
    "from", "by", "than", "over", "into", "like", "between",
    "through", "after", "before", "under", "against",
    "using", "instead", "also", "well",
}

# Words to strip from source snippets (more aggressive than key fillers)
_SNIPPET_STOPWORDS = {
    "i", "we", "you", "he", "she", "they", "it", "me", "us",
    "my", "your", "our", "his", "her", "their", "its",
    "the", "a", "an", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "and", "or", "but", "so", "if", "then", "also", "too",
    "very", "really", "just", "actually", "basically", "definitely",
    "think", "believe", "feel", "know",
}

# Matches numbers with optional units (e.g., "6h", "30min", "$500", "99%")
_NUMBER_PATTERN = re.compile(r'\$?\d+[\d,.]*\s*(?:%|hours?|hrs?|h|minutes?|mins?|min|seconds?|secs?|s|days?|d|weeks?|months?|years?|gb|mb|kb|tb|ms|k|m)?', re.IGNORECASE)


def _condense_source(text: str, max_words: int = 12) -> str:
    """Condense source text into a short snippet preserving key details.

    Keeps numbers, proper-noun-like words, and content words.
    Strips filler/stopwords aggressively. Caps at max_words.
    """
    if not text:
        return ""

    # Remove sentence-ending punctuation, keep internal punctuation
    cleaned = text.strip().rstrip(".")

    words = cleaned.split()
    content: list[str] = []
    for w in words:
        low = w.lower().strip(".,;:!?\"'()")
        # Always keep numbers and words with digits
        if any(c.isdigit() for c in w):
            content.append(w.strip(".,;:!?\"'()"))
        # Keep quoted terms
        elif w.startswith('"') or w.startswith("'"):
            content.append(w.strip(".,;:!?\"'()"))
        # Skip stopwords
        elif low in _SNIPPET_STOPWORDS:
            continue
        # Keep content words
        else:
            content.append(w.strip(".,;:!?\"'()"))

    if not content:
        # Fallback: use first few words
        content = [w.strip(".,;:!?") for w in words[:max_words]]

    return " ".join(content[:max_words]).lower()


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

    def encode(self, frame: SemanticFrame, act_labels: bool = False,
               source_snippet: bool = False) -> str:
        """
        Encode a SemanticFrame into LLM-readable format.

        Format: HEADER object_key [>target_key] [#act_label] [| snippet]
        Header: S(1) M(1) AA(2) C(1) T(1) = 6 chars
        Object key: natural words, no truncation
        act_labels: if True, append #act_label for stance-critical acts
        source_snippet: if True, append condensed source text after |
        """
        vocab = NormalizationVocabulary()

        # Build 6-char header
        s = SPEAKER_CODES.get(frame.speaker, "?")
        m = MODE_CODES_V2.get(frame.mode, "?")
        aa = ACT_CODES_V2.get(frame.act, "uk")
        c = _certainty_to_hex(frame.certainty.value)
        t = TIME_CODES_V2.get(frame.time, "?")
        header = f"{s}{m}{aa}{c}{t}"

        # Object key: natural words, no truncation
        if frame.object:
            obj_key = _llm_key(vocab, frame.object.canonical)
        else:
            obj_key = "unknown"

        parts = [header, obj_key]

        # Target key (for comparisons)
        if frame.target:
            tgt_key = _llm_key(vocab, frame.target.canonical)
            parts.append(f">{tgt_key}")

        # Act label for stance-critical acts
        if act_labels and frame.act in _STANCE_ACTS:
            parts.append(f"#{_STANCE_ACTS[frame.act]}")

        # Source snippet for detail preservation
        if source_snippet and frame.source_text:
            snippet = _condense_source(frame.source_text)
            if snippet:
                parts.append(f"| {snippet}")

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

    def encode_batch(self, frames: list[SemanticFrame], act_labels: bool = False,
                     source_snippet: bool = False) -> str:
        """Encode multiple frames into newline-separated LLM format."""
        return "\n".join(
            self.encode(f, act_labels=act_labels, source_snippet=source_snippet)
            for f in frames
        )

    def decode_batch(self, text: str) -> list[SemanticFrame]:
        """Decode newline-separated LLM format codes into frames."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return [self.decode(line) for line in lines]
