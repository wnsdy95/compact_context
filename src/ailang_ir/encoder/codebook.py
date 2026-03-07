"""
Symbolic encoder: converts SemanticFrames into compact deterministic codes.

The compact form uses pipe-delimited fields:
  SPEAKER|MODE|ACT|OBJECT[|OVER:TARGET]|CERTAINTY|TIME

Examples:
  U|OP|BELIEVE|SENTENCE_1TO1_MAPPING_HARD|C84|NOW
  U|REQ|NEED|NATURAL_LANGUAGE_RECONSTRUCTION|C91|FUT
  U|OP|PREFER|GRAPH_MEMORY|OVER:LINEAR_TEXT|C83|NOW
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ailang_ir.models.domain import (
    SemanticFrame,
    SpeakerRole,
    SemanticMode,
    SemanticAct,
    TimeReference,
)


# ---------------------------------------------------------------------------
# Codebook mappings — short deterministic codes
# ---------------------------------------------------------------------------

SPEAKER_CODES: dict[SpeakerRole, str] = {
    SpeakerRole.USER: "U",
    SpeakerRole.SYSTEM: "S",
    SpeakerRole.AGENT: "A",
    SpeakerRole.THIRD_PARTY: "T",
    SpeakerRole.UNKNOWN: "?",
}

MODE_CODES: dict[SemanticMode, str] = {
    SemanticMode.ASSERTION: "AS",
    SemanticMode.OPINION: "OP",
    SemanticMode.HYPOTHESIS: "HY",
    SemanticMode.REQUEST: "REQ",
    SemanticMode.COMMAND: "CMD",
    SemanticMode.QUESTION: "Q",
    SemanticMode.COMMITMENT: "CMT",
    SemanticMode.OBSERVATION: "OBS",
    SemanticMode.REFLECTION: "REF",
}

ACT_CODES: dict[SemanticAct, str] = {
    SemanticAct.BELIEVE: "BELIEVE",
    SemanticAct.PREFER: "PREFER",
    SemanticAct.SUGGEST: "SUGGEST",
    SemanticAct.NEED: "NEED",
    SemanticAct.DECIDE: "DECIDE",
    SemanticAct.REJECT: "REJECT",
    SemanticAct.AGREE: "AGREE",
    SemanticAct.DISAGREE: "DISAGREE",
    SemanticAct.CREATE: "CREATE",
    SemanticAct.MODIFY: "MODIFY",
    SemanticAct.DELETE: "DELETE",
    SemanticAct.QUERY: "QUERY",
    SemanticAct.OBSERVE: "OBSERVE",
    SemanticAct.COMPARE: "COMPARE",
    SemanticAct.PLAN: "PLAN",
    SemanticAct.WARN: "WARN",
    SemanticAct.EXPLAIN: "EXPLAIN",
    SemanticAct.UNKNOWN: "UNK",
}

TIME_CODES: dict[TimeReference, str] = {
    TimeReference.PAST: "PAST",
    TimeReference.PRESENT: "NOW",
    TimeReference.FUTURE: "FUT",
    TimeReference.ATEMPORAL: "ATEMP",
    TimeReference.UNKNOWN: "T?",
}

# Reverse mappings for decoding
SPEAKER_DECODE = {v: k for k, v in SPEAKER_CODES.items()}
MODE_DECODE = {v: k for k, v in MODE_CODES.items()}
ACT_DECODE = {v: k for k, v in ACT_CODES.items()}
TIME_DECODE = {v: k for k, v in TIME_CODES.items()}


@dataclass
class SymbolicEncoder:
    """
    Encodes a SemanticFrame into a compact pipe-delimited symbolic string.
    Decodes a symbolic string back into field values.

    The encoding is deterministic: the same frame always produces
    the same code, enabling deduplication and comparison.
    """
    speaker_codes: dict[SpeakerRole, str] = field(default_factory=lambda: dict(SPEAKER_CODES))
    mode_codes: dict[SemanticMode, str] = field(default_factory=lambda: dict(MODE_CODES))
    act_codes: dict[SemanticAct, str] = field(default_factory=lambda: dict(ACT_CODES))
    time_codes: dict[TimeReference, str] = field(default_factory=lambda: dict(TIME_CODES))

    def encode(self, frame: SemanticFrame) -> str:
        """
        Encode a SemanticFrame into compact symbolic form.

        Format: SPEAKER|MODE|ACT|OBJECT[|OVER:TARGET]|CERTAINTY|TIME
        """
        parts: list[str] = []

        # Speaker
        parts.append(self.speaker_codes.get(frame.speaker, "?"))

        # Mode
        parts.append(self.mode_codes.get(frame.mode, "?"))

        # Act
        parts.append(self.act_codes.get(frame.act, "UNK"))

        # Object
        if frame.object:
            parts.append(frame.object.canonical.upper())
        else:
            parts.append("?OBJ")

        # Target (for comparisons)
        if frame.target:
            parts.append(f"OVER:{frame.target.canonical.upper()}")

        # Certainty
        parts.append(frame.certainty.compact())

        # Time
        parts.append(self.time_codes.get(frame.time, "T?"))

        return "|".join(parts)

    def decode_fields(self, code: str) -> dict[str, str]:
        """
        Decode a compact symbolic code back into named fields.

        Returns a dict with keys: speaker, mode, act, object, target (optional),
        certainty, time.
        """
        parts = code.split("|")
        if len(parts) < 5:
            raise ValueError(f"Invalid compact code (too few fields): {code}")

        result: dict[str, str] = {}
        idx = 0

        result["speaker"] = parts[idx]; idx += 1
        result["mode"] = parts[idx]; idx += 1
        result["act"] = parts[idx]; idx += 1
        result["object"] = parts[idx]; idx += 1

        # Check for optional OVER:target
        if idx < len(parts) and parts[idx].startswith("OVER:"):
            result["target"] = parts[idx][5:]  # strip "OVER:" prefix
            idx += 1

        if idx < len(parts):
            result["certainty"] = parts[idx]; idx += 1
        if idx < len(parts):
            result["time"] = parts[idx]; idx += 1

        return result

    def decode_speaker(self, code: str) -> SpeakerRole:
        return SPEAKER_DECODE.get(code, SpeakerRole.UNKNOWN)

    def decode_mode(self, code: str) -> SemanticMode:
        return MODE_DECODE.get(code, SemanticMode.ASSERTION)

    def decode_act(self, code: str) -> SemanticAct:
        return ACT_DECODE.get(code, SemanticAct.UNKNOWN)

    def decode_time(self, code: str) -> TimeReference:
        return TIME_DECODE.get(code, TimeReference.UNKNOWN)

    def decode_certainty(self, code: str) -> float:
        """Parse 'C84' → 0.84"""
        if code.startswith("C") and code[1:].isdigit():
            return int(code[1:]) / 100.0
        return 0.5
