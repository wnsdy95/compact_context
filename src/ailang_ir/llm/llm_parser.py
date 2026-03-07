"""
LLM-assisted semantic parser using Anthropic API.

Sends natural language + FORMAT_SPEC to an LLM, receives AILang-IR codes,
validates them, and converts to SemanticFrames. Falls back to rule-based
parser on any failure.

The anthropic SDK is an optional dependency — core AILang-IR works without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ailang_ir.models.domain import SemanticFrame, SpeakerRole
from ailang_ir.llm.format_spec import get_format_spec
from ailang_ir.llm.validator import validate_code
from ailang_ir.llm.codec import LLMCodec
from ailang_ir.parser.rule_parser import RuleBasedParser


_SPEAKER_LABELS = {
    SpeakerRole.USER: "U",
    SpeakerRole.AGENT: "A",
    SpeakerRole.SYSTEM: "S",
    SpeakerRole.THIRD_PARTY: "T",
    SpeakerRole.UNKNOWN: "?",
}

_SYSTEM_PROMPT = """\
You are a semantic encoder. Given natural language, produce AILang-IR codes.

{spec}

Rules:
- Output ONLY the code, no explanation, no markdown fencing
- One code per input line
- Speaker is indicated before each line as [U], [A], [S], or [T]
- Use the most specific act that applies
- Certainty: use context clues (hedges → lower, definite → higher)
- Object key: extract the CORE concept (1-2 words, underscore separated)
- For comparisons/preferences with "over"/"than", use >target_key
"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT.format(spec=get_format_spec())


def _import_anthropic():
    """Import anthropic SDK with clear error message."""
    try:
        import anthropic
        return anthropic
    except ImportError:
        raise ImportError(
            "LLMParser requires the 'anthropic' package. "
            "Install it with: pip install anthropic"
        )


@dataclass
class LLMParser:
    """
    LLM-assisted semantic parser.

    Uses Anthropic API to convert natural language into AILang-IR codes.
    Falls back to RuleBasedParser on any failure (API error, invalid output, etc.)

    Usage:
        parser = LLMParser()
        frame = parser.parse("I think graph memory is the best approach.")
    """
    model: str = "claude-haiku-4-5-20251001"
    api_key: str | None = None
    max_tokens: int = 256
    _client: object = field(default=None, repr=False)
    _codec: LLMCodec = field(default_factory=LLMCodec)
    _fallback: RuleBasedParser = field(default_factory=RuleBasedParser)

    def __post_init__(self):
        if self._client is None:
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                anthropic = _import_anthropic()
                self._client = anthropic.Anthropic(api_key=key)

    def parse(
        self,
        text: str,
        speaker: SpeakerRole = SpeakerRole.USER,
    ) -> SemanticFrame:
        """
        Parse text using LLM. Falls back to rule-based on any failure.

        Returns a SemanticFrame regardless of LLM availability.
        """
        if self._client is None:
            return self._fallback.parse(text, speaker)

        try:
            code = self._call_llm_single(text, speaker)
            return self._decode_or_fallback(code, text, speaker)
        except Exception:
            return self._fallback.parse(text, speaker)

    def parse_batch(
        self,
        texts: list[str],
        speakers: list[SpeakerRole] | SpeakerRole = SpeakerRole.USER,
    ) -> list[SemanticFrame]:
        """
        Parse multiple texts in a single LLM call for efficiency.

        speakers: single role applied to all, or list matching texts length.
        """
        if isinstance(speakers, SpeakerRole):
            speaker_list = [speakers] * len(texts)
        else:
            speaker_list = speakers

        if self._client is None:
            return [self._fallback.parse(t, s) for t, s in zip(texts, speaker_list)]

        try:
            codes = self._call_llm_batch(texts, speaker_list)
            frames = []
            for code, text, speaker in zip(codes, texts, speaker_list):
                frames.append(self._decode_or_fallback(code, text, speaker))
            return frames
        except Exception:
            return [self._fallback.parse(t, s) for t, s in zip(texts, speaker_list)]

    def _decode_or_fallback(
        self,
        code: str,
        text: str,
        speaker: SpeakerRole,
    ) -> SemanticFrame:
        """Validate and decode a code, falling back on failure."""
        code = code.strip()
        if not code:
            return self._fallback.parse(text, speaker)

        vr = validate_code(code)
        if not vr.is_valid:
            return self._fallback.parse(text, speaker)

        try:
            return self._codec.decode(code)
        except (ValueError, KeyError):
            return self._fallback.parse(text, speaker)

    def _call_llm_single(self, text: str, speaker: SpeakerRole) -> str:
        """Call LLM for a single text."""
        label = _SPEAKER_LABELS.get(speaker, "?")
        user_msg = f"[{label}] {text}"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": user_msg}],
        )
        return self._extract_text(response)

    def _call_llm_batch(
        self,
        texts: list[str],
        speakers: list[SpeakerRole],
    ) -> list[str]:
        """Call LLM for multiple texts in one request."""
        lines = []
        for text, speaker in zip(texts, speakers):
            label = _SPEAKER_LABELS.get(speaker, "?")
            lines.append(f"[{label}] {text}")

        user_msg = "\n".join(lines)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = self._extract_text(response)
        codes = [line.strip() for line in raw.strip().splitlines() if line.strip()]

        # Pad or truncate to match input length
        while len(codes) < len(texts):
            codes.append("")
        return codes[:len(texts)]

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from Anthropic API response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
