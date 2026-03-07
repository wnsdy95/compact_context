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
from ailang_ir.encoder.concept_table import ConceptTable
from ailang_ir.normalize.vocabulary import NormalizationVocabulary


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

    # -------------------------------------------------------------------
    # v2 encoding: fixed-width header + concept references
    # -------------------------------------------------------------------

    def encode_v2(self, frame: SemanticFrame, concept_table: ConceptTable) -> str:
        """
        Encode a SemanticFrame into v2 compact form.

        Format: HEADER OBJECT [>TARGET]
        Header = S M AA C T (6 chars, fixed position, no delimiters)
        """
        vocab = NormalizationVocabulary()

        # Header: S(1) M(1) AA(2) C(1) T(1) = 6 chars
        s = SPEAKER_CODES.get(frame.speaker, "?")
        m = MODE_CODES_V2.get(frame.mode, "?")
        aa = ACT_CODES_V2.get(frame.act, "uk")
        c = _certainty_to_hex(frame.certainty.value)
        t = TIME_CODES_V2.get(frame.time, "?")
        header = f"{s}{m}{aa}{c}{t}"

        # Object
        if frame.object:
            compressed = vocab.compress_object_key(frame.object.canonical)
            obj_ref = concept_table.ref(compressed)
        else:
            obj_ref = "#?obj"

        parts = [header, obj_ref]

        # Target
        if frame.target:
            compressed_t = vocab.compress_object_key(frame.target.canonical)
            tgt_ref = concept_table.ref(compressed_t)
            parts.append(f">{tgt_ref}")

        return " ".join(parts)

    def decode_fields_v2(self, code: str, concept_table: ConceptTable) -> dict[str, str]:
        """
        Decode a v2 compact code back into named fields.

        Returns dict with keys: speaker, mode, act, object, target (optional),
        certainty, time.
        """
        tokens = code.split()
        if len(tokens) < 2:
            raise ValueError(f"Invalid v2 code (too few tokens): {code}")

        header = tokens[0]
        if len(header) != 6:
            raise ValueError(f"Invalid v2 header length: {header}")

        result: dict[str, str] = {}
        result["speaker"] = header[0]
        result["mode"] = header[1]
        result["act"] = header[2:4]
        result["certainty"] = header[4]
        result["time"] = header[5]

        # Resolve object reference
        obj_token = tokens[1]
        result["object"] = _resolve_ref(obj_token, concept_table)

        # Check for target
        if len(tokens) >= 3 and tokens[2].startswith(">"):
            tgt_token = tokens[2][1:]  # strip ">"
            result["target"] = _resolve_ref(tgt_token, concept_table)

        return result

    def disassemble(self, code: str, concept_table: ConceptTable) -> str:
        """
        Human-readable debug disassembly of a v2 code.

        Example output:
          Speaker: USER | Mode: OPINION | Act: BELIEVE | Cert: 70% | Time: PRESENT
          Object: 1to1_sent_map_diff (#0)
        """
        fields = self.decode_fields_v2(code, concept_table)
        speaker = SPEAKER_DECODE.get(fields["speaker"], SpeakerRole.UNKNOWN)
        mode = MODE_DECODE_V2.get(fields["mode"], SemanticMode.ASSERTION)
        act = ACT_DECODE_V2.get(fields["act"], SemanticAct.UNKNOWN)
        cert = _hex_to_certainty(fields["certainty"])
        time = TIME_DECODE_V2.get(fields["time"], TimeReference.UNKNOWN)

        lines = [
            f"Speaker: {speaker.value.upper()} | Mode: {mode.value.upper()} | "
            f"Act: {act.value.upper()} | Cert: {int(cert*100)}% | Time: {time.value.upper()}",
            f"Object: {fields['object']}",
        ]
        if "target" in fields:
            lines.append(f"Target: {fields['target']}")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # v3 encoding: v2 header + ultra-compact stem-abbreviated keys
    # -------------------------------------------------------------------

    def encode_v3(self, frame: SemanticFrame, concept_table: ConceptTable) -> str:
        """
        Encode a SemanticFrame into v3 ultra-compact form.

        Same header as v2 (6 chars). Object keys use stem abbreviation
        dictionary + max 3 tokens + no underscore separators.
        Target: ~30% of raw text size.
        """
        vocab = NormalizationVocabulary()

        s = SPEAKER_CODES.get(frame.speaker, "?")
        m = MODE_CODES_V2.get(frame.mode, "?")
        aa = ACT_CODES_V2.get(frame.act, "uk")
        c = _certainty_to_hex(frame.certainty.value)
        t = TIME_CODES_V2.get(frame.time, "?")
        header = f"{s}{m}{aa}{c}{t}"

        if frame.object:
            compressed = vocab.compress_object_key_v3(frame.object.canonical)
            obj_ref = concept_table.ref(compressed)
        else:
            obj_ref = "#?obj"

        parts = [header, obj_ref]

        if frame.target:
            compressed_t = vocab.compress_object_key_v3(frame.target.canonical)
            tgt_ref = concept_table.ref(compressed_t)
            parts.append(f">{tgt_ref}")

        return " ".join(parts)

    # v2/v3 enum decoders (shared — same header format)
    def decode_mode_v2(self, code: str) -> SemanticMode:
        return MODE_DECODE_V2.get(code, SemanticMode.ASSERTION)

    def decode_act_v2(self, code: str) -> SemanticAct:
        return ACT_DECODE_V2.get(code, SemanticAct.UNKNOWN)

    def decode_time_v2(self, code: str) -> TimeReference:
        return TIME_DECODE_V2.get(code, TimeReference.UNKNOWN)


# ---------------------------------------------------------------------------
# v2 codebook mappings
# ---------------------------------------------------------------------------

MODE_CODES_V2: dict[SemanticMode, str] = {
    SemanticMode.ASSERTION: "a",
    SemanticMode.OPINION: "o",
    SemanticMode.HYPOTHESIS: "h",
    SemanticMode.REQUEST: "r",
    SemanticMode.COMMAND: "c",
    SemanticMode.QUESTION: "q",
    SemanticMode.COMMITMENT: "k",
    SemanticMode.OBSERVATION: "b",
    SemanticMode.REFLECTION: "f",
}

ACT_CODES_V2: dict[SemanticAct, str] = {
    SemanticAct.BELIEVE: "bl",
    SemanticAct.PREFER: "pr",
    SemanticAct.SUGGEST: "sg",
    SemanticAct.NEED: "nd",
    SemanticAct.DECIDE: "dc",
    SemanticAct.REJECT: "rj",
    SemanticAct.AGREE: "ag",
    SemanticAct.DISAGREE: "dg",
    SemanticAct.CREATE: "cr",
    SemanticAct.MODIFY: "md",
    SemanticAct.DELETE: "dl",
    SemanticAct.QUERY: "qu",
    SemanticAct.OBSERVE: "ob",
    SemanticAct.COMPARE: "cm",
    SemanticAct.PLAN: "pl",
    SemanticAct.WARN: "wn",
    SemanticAct.EXPLAIN: "ex",
    SemanticAct.UNKNOWN: "uk",
}

TIME_CODES_V2: dict[TimeReference, str] = {
    TimeReference.PAST: "p",
    TimeReference.PRESENT: "n",
    TimeReference.FUTURE: "f",
    TimeReference.ATEMPORAL: "a",
    TimeReference.UNKNOWN: "?",
}

# Reverse mappings for v2
MODE_DECODE_V2 = {v: k for k, v in MODE_CODES_V2.items()}
ACT_DECODE_V2 = {v: k for k, v in ACT_CODES_V2.items()}
TIME_DECODE_V2 = {v: k for k, v in TIME_CODES_V2.items()}


def _certainty_to_hex(value: float) -> str:
    """Convert certainty 0.0–1.0 to a single hex char (0–f, 16 levels)."""
    level = min(15, max(0, int(value * 15.999)))
    return format(level, "x")


def _hex_to_certainty(h: str) -> float:
    """Convert a single hex char back to certainty 0.0–1.0."""
    level = int(h, 16)
    return level / 15.0


def _resolve_ref(token: str, concept_table: ConceptTable) -> str:
    """Resolve a #key or $id reference to the concept key."""
    if token.startswith("#"):
        return token[1:]
    elif token.startswith("$"):
        from ailang_ir.encoder.concept_table import decode_id
        cid = decode_id(token[1:])
        resolved = concept_table.resolve(cid)
        if resolved is None:
            return f"?unresolved({token})"
        return resolved
    return token
