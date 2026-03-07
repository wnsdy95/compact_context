"""
Natural language reconstructor.

Rebuilds readable text from SemanticFrames or compact symbolic codes.
Prioritizes meaning preservation over stylistic richness.

The reconstruction is lossy by design — the goal is to preserve
the essential meaning, not to reproduce the original surface form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ailang_ir.models.domain import (
    Certainty,
    Entity,
    Predicate,
    ReconstructionPlan,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    Sentiment,
    SpeakerRole,
    TimeReference,
)
from ailang_ir.encoder.codebook import SymbolicEncoder


# ---------------------------------------------------------------------------
# Template fragments for reconstruction
# ---------------------------------------------------------------------------

MODE_PREFIXES: dict[SemanticMode, str] = {
    SemanticMode.ASSERTION: "",
    SemanticMode.OPINION: "I think ",
    SemanticMode.HYPOTHESIS: "Perhaps ",
    SemanticMode.REQUEST: "Could you ",
    SemanticMode.COMMAND: "",
    SemanticMode.QUESTION: "",
    SemanticMode.COMMITMENT: "I will ",
    SemanticMode.OBSERVATION: "I notice that ",
    SemanticMode.REFLECTION: "Thinking about it, ",
}

ACT_VERBS: dict[SemanticAct, dict[str, str]] = {
    # act -> {tense: verb phrase}
    SemanticAct.BELIEVE: {"present": "believe that", "past": "believed that", "future": "will believe that"},
    SemanticAct.PREFER: {"present": "prefer", "past": "preferred", "future": "will prefer"},
    SemanticAct.SUGGEST: {"present": "suggest", "past": "suggested", "future": "would suggest"},
    SemanticAct.NEED: {"present": "need", "past": "needed", "future": "will need"},
    SemanticAct.DECIDE: {"present": "decide on", "past": "decided on", "future": "will decide on"},
    SemanticAct.REJECT: {"present": "reject", "past": "rejected", "future": "will reject"},
    SemanticAct.AGREE: {"present": "agree with", "past": "agreed with", "future": "will agree with"},
    SemanticAct.DISAGREE: {"present": "disagree with", "past": "disagreed with", "future": "will disagree with"},
    SemanticAct.CREATE: {"present": "create", "past": "created", "future": "will create"},
    SemanticAct.MODIFY: {"present": "modify", "past": "modified", "future": "will modify"},
    SemanticAct.DELETE: {"present": "remove", "past": "removed", "future": "will remove"},
    SemanticAct.QUERY: {"present": "ask about", "past": "asked about", "future": "will ask about"},
    SemanticAct.OBSERVE: {"present": "observe that", "past": "observed that", "future": "will observe that"},
    SemanticAct.COMPARE: {"present": "compare", "past": "compared", "future": "will compare"},
    SemanticAct.PLAN: {"present": "plan to", "past": "planned to", "future": "will plan to"},
    SemanticAct.WARN: {"present": "warn about", "past": "warned about", "future": "will warn about"},
    SemanticAct.EXPLAIN: {"present": "explain that", "past": "explained that", "future": "will explain that"},
    SemanticAct.UNKNOWN: {"present": "state that", "past": "stated that", "future": "will state that"},
}

CERTAINTY_QUALIFIERS: list[tuple[float, str]] = [
    (0.30, "possibly"),
    (0.50, "maybe"),
    (0.70, ""),           # moderate certainty — no qualifier needed
    (0.85, ""),           # confident — clean statement
    (1.01, "definitely"), # near-certain
]

TIME_SUFFIXES: dict[TimeReference, str] = {
    TimeReference.PAST: " (in the past)",
    TimeReference.PRESENT: "",
    TimeReference.FUTURE: " (going forward)",
    TimeReference.ATEMPORAL: "",
    TimeReference.UNKNOWN: "",
}


def _key_to_phrase(key: str) -> str:
    """Convert a normalized key back into a readable phrase.

    E.g. '1to1_sentence_mapping' → 'one-to-one sentence mapping'
    """
    text = key.lower()
    text = text.replace("1to1", "one-to-one")
    text = text.replace("1ton", "one-to-many")
    text = text.replace("nton", "many-to-many")
    text = text.replace("_", " ")
    return text


def _certainty_qualifier(value: float) -> str:
    """Return an appropriate certainty qualifier word."""
    for threshold, word in CERTAINTY_QUALIFIERS:
        if value < threshold:
            return word
    return ""


@dataclass
class Reconstructor:
    """
    Reconstructs natural language from a SemanticFrame or compact code.

    Supports multiple output styles:
    - declarative: formal statement
    - conversational: natural spoken form
    - summary: ultra-compact summary line
    """
    encoder: SymbolicEncoder = field(default_factory=SymbolicEncoder)

    def reconstruct(self, frame: SemanticFrame, style: str = "declarative") -> str:
        """Reconstruct natural language from a SemanticFrame."""
        if style == "summary":
            return self._summary_style(frame)
        elif style == "conversational":
            return self._conversational_style(frame)
        else:
            return self._declarative_style(frame)

    def reconstruct_from_code(self, code: str, style: str = "declarative") -> str:
        """Reconstruct natural language from a compact symbolic code."""
        frame = self._code_to_frame(code)
        return self.reconstruct(frame, style)

    def reconstruct_from_plan(self, plan: ReconstructionPlan) -> str:
        """Reconstruct using an explicit ReconstructionPlan."""
        return self.reconstruct(plan.frame, plan.style)

    # -------------------------------------------------------------------
    # Style implementations
    # -------------------------------------------------------------------

    def _declarative_style(self, frame: SemanticFrame) -> str:
        """Formal declarative reconstruction."""
        parts: list[str] = []

        # Speaker context
        speaker = self._speaker_label(frame.speaker)

        # Mode prefix
        mode_prefix = MODE_PREFIXES.get(frame.mode, "")

        # Question mode is special
        if frame.mode == SemanticMode.QUESTION:
            return self._question_style(frame)

        # Certainty qualifier
        cert_q = _certainty_qualifier(frame.certainty.value)
        if cert_q:
            mode_prefix = f"{cert_q.capitalize()}, {mode_prefix.lower()}" if mode_prefix else f"{cert_q.capitalize()}, "

        # Act verb phrase — use third-person for "the user/system" subjects
        tense = self._time_to_tense(frame.time)
        act_verbs = ACT_VERBS.get(frame.act, ACT_VERBS[SemanticAct.UNKNOWN])
        verb = act_verbs.get(tense, act_verbs["present"])
        # Third-person singular conjugation for present tense
        third_person = frame.mode not in (
            SemanticMode.OPINION, SemanticMode.HYPOTHESIS,
            SemanticMode.COMMITMENT, SemanticMode.REFLECTION,
        )
        if third_person and tense == "present":
            verb = self._conjugate_3rd(verb)

        # Object phrase
        obj_phrase = _key_to_phrase(frame.object.canonical) if frame.object else "something"

        # Target phrase (for comparisons)
        target_phrase = ""
        if frame.target:
            target_phrase = f" over {_key_to_phrase(frame.target.canonical)}"

        # Build sentence
        if frame.mode in (SemanticMode.OPINION, SemanticMode.HYPOTHESIS,
                          SemanticMode.OBSERVATION, SemanticMode.REFLECTION):
            # First person: "I think X is Y"
            sentence = f"{mode_prefix}{obj_phrase}{target_phrase}"
        elif frame.mode == SemanticMode.COMMITMENT:
            sentence = f"{mode_prefix}{verb} {obj_phrase}{target_phrase}"
        elif frame.mode == SemanticMode.COMMAND:
            sentence = f"{verb.capitalize()} {obj_phrase}{target_phrase}"
        else:
            # General: "Speaker verb object"
            sentence = f"{mode_prefix}{speaker} {verb} {obj_phrase}{target_phrase}"
            sentence = sentence.strip()

        # Time suffix
        time_suffix = TIME_SUFFIXES.get(frame.time, "")

        result = f"{sentence}{time_suffix}."
        # Capitalize first letter
        return result[0].upper() + result[1:] if result else result

    def _conversational_style(self, frame: SemanticFrame) -> str:
        """More natural, spoken-like reconstruction."""
        obj = _key_to_phrase(frame.object.canonical) if frame.object else "that"
        target = f" over {_key_to_phrase(frame.target.canonical)}" if frame.target else ""

        cert_q = _certainty_qualifier(frame.certainty.value)

        if frame.mode == SemanticMode.QUESTION:
            return self._question_style(frame)

        if frame.mode == SemanticMode.OPINION:
            if cert_q:
                return f"I {cert_q} think {obj}{target}."
            return f"I think {obj}{target}."

        if frame.mode == SemanticMode.HYPOTHESIS:
            return f"Maybe {obj}{target}."

        if frame.act == SemanticAct.PREFER and frame.target:
            return f"I'd go with {obj}{target}."

        if frame.act == SemanticAct.NEED:
            return f"We need {obj}."

        if frame.act == SemanticAct.SUGGEST:
            return f"How about {obj}?"

        # Fallback
        tense = self._time_to_tense(frame.time)
        verb = ACT_VERBS.get(frame.act, ACT_VERBS[SemanticAct.UNKNOWN]).get(tense, "involves")
        return f"It {verb} {obj}{target}."

    def _summary_style(self, frame: SemanticFrame) -> str:
        """Ultra-compact single-line summary."""
        act = frame.act.value.upper()
        obj = frame.object.canonical.upper() if frame.object else "?"
        target = f" > {frame.target.canonical.upper()}" if frame.target else ""
        return f"{act}: {obj}{target} [{frame.certainty.compact()}]"

    def _question_style(self, frame: SemanticFrame) -> str:
        """Reconstruct a question."""
        obj = _key_to_phrase(frame.object.canonical) if frame.object else "this"
        return f"What about {obj}?"

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _speaker_label(self, speaker: SpeakerRole) -> str:
        labels = {
            SpeakerRole.USER: "the user",
            SpeakerRole.SYSTEM: "the system",
            SpeakerRole.AGENT: "the agent",
            SpeakerRole.THIRD_PARTY: "someone",
            SpeakerRole.UNKNOWN: "someone",
        }
        return labels.get(speaker, "someone")

    @staticmethod
    def _conjugate_3rd(verb_phrase: str) -> str:
        """Simple third-person singular conjugation for present tense verb phrases.

        'believe that' → 'believes that'
        'observe that' → 'observes that'
        'create' → 'creates'
        """
        parts = verb_phrase.split(" ", 1)
        v = parts[0]
        rest = f" {parts[1]}" if len(parts) > 1 else ""
        if v.endswith(("s", "sh", "ch", "x", "z")):
            return f"{v}es{rest}"
        if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
            return f"{v[:-1]}ies{rest}"
        return f"{v}s{rest}"

    def _time_to_tense(self, time: TimeReference) -> str:
        mapping = {
            TimeReference.PAST: "past",
            TimeReference.PRESENT: "present",
            TimeReference.FUTURE: "future",
            TimeReference.ATEMPORAL: "present",
            TimeReference.UNKNOWN: "present",
        }
        return mapping.get(time, "present")

    def _code_to_frame(self, code: str) -> SemanticFrame:
        """Convert a compact code back into a SemanticFrame for reconstruction."""
        fields = self.encoder.decode_fields(code)

        speaker = self.encoder.decode_speaker(fields.get("speaker", "?"))
        mode = self.encoder.decode_mode(fields.get("mode", "AS"))
        act = self.encoder.decode_act(fields.get("act", "UNK"))
        time = self.encoder.decode_time(fields.get("time", "T?"))
        certainty_val = self.encoder.decode_certainty(fields.get("certainty", "C50"))

        obj_key = fields.get("object", "")
        obj_entity = Entity(canonical=obj_key.lower()) if obj_key and obj_key != "?OBJ" else None

        target_key = fields.get("target")
        target_entity = Entity(canonical=target_key.lower()) if target_key else None

        return SemanticFrame(
            speaker=speaker,
            mode=mode,
            predicate=Predicate(act),
            object=obj_entity,
            target=target_entity,
            time=time,
            certainty=Certainty(certainty_val),
        )
