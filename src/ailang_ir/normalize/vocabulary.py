"""
Normalization vocabulary for mapping surface expressions to canonical semantic categories.

This module provides deterministic normalization rules so that
different phrasings of the same meaning converge to the same
internal representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from ailang_ir.models.domain import SemanticAct, SemanticMode, Sentiment, TimeReference


# ---------------------------------------------------------------------------
# Surface → canonical mapping tables
# ---------------------------------------------------------------------------

# Each key is a canonical value; the list contains surface patterns (lowercase)
# that should map to it.

ACT_SURFACE_MAP: dict[SemanticAct, list[str]] = {
    SemanticAct.BELIEVE: [
        "think", "believe", "feel that", "suppose", "reckon",
        "consider", "am convinced", "suspect",
    ],
    SemanticAct.PREFER: [
        "prefer", "like better", "favor", "would rather",
        "lean toward", "lean towards",
    ],
    SemanticAct.SUGGEST: [
        "suggest", "propose", "recommend", "advise",
        "would say", "how about", "should use", "let's use",
        "try using", "try", "use",
    ],
    SemanticAct.NEED: [
        "need", "require", "must have", "should have",
        "have to", "got to",
    ],
    SemanticAct.DECIDE: [
        "decide", "chose", "settled on", "going with",
        "picked", "determined",
    ],
    SemanticAct.REJECT: [
        "reject", "refuse", "decline", "don't want",
        "do not want", "won't", "will not accept",
    ],
    SemanticAct.AGREE: [
        "agree", "concur", "right", "exactly",
        "that's correct", "yes",
    ],
    SemanticAct.DISAGREE: [
        "disagree", "don't think so", "no", "not really",
        "I don't agree", "that's wrong",
    ],
    SemanticAct.CREATE: [
        "create", "build", "make", "construct", "implement",
        "develop", "write", "generate",
    ],
    SemanticAct.MODIFY: [
        "modify", "change", "update", "edit", "revise",
        "adjust", "alter", "refactor",
    ],
    SemanticAct.DELETE: [
        "delete", "remove", "drop", "discard", "eliminate",
        "get rid of",
    ],
    SemanticAct.QUERY: [
        "what is", "what are", "how does", "how do",
        "why", "when", "where", "is it", "can you",
    ],
    SemanticAct.OBSERVE: [
        "notice", "see", "observe", "found", "discovered",
        "it seems", "appears", "looks like",
    ],
    SemanticAct.COMPARE: [
        "compare", "versus", "vs", "better than", "worse than",
        "difference between", "similarities",
    ],
    SemanticAct.PLAN: [
        "plan", "going to", "will", "intend to",
        "schedule", "aim to",
    ],
    SemanticAct.WARN: [
        "warn", "be careful", "watch out", "caution",
        "danger", "risk",
    ],
    SemanticAct.EXPLAIN: [
        "explain", "because", "reason is", "the point is",
        "in other words", "meaning",
    ],
}

MODE_SURFACE_MAP: dict[SemanticMode, list[str]] = {
    SemanticMode.ASSERTION: [
        "is", "are", "was", "were", "it is", "that is",
        "clearly", "certainly", "definitely",
    ],
    SemanticMode.OPINION: [
        "I think", "I believe", "in my opinion", "I feel",
        "seems to me", "my view",
    ],
    SemanticMode.HYPOTHESIS: [
        "maybe", "perhaps", "might", "could be", "possibly",
        "what if", "hypothetically", "I guess",
    ],
    SemanticMode.REQUEST: [
        "please", "could you", "would you", "can you",
        "I'd like", "I would like",
    ],
    SemanticMode.COMMAND: [
        "do this", "make it", "you must", "you should",
        "just do", "go ahead",
    ],
    SemanticMode.QUESTION: [
        "?", "what", "how", "why", "when", "where",
        "is it", "do you", "can we",
    ],
    SemanticMode.COMMITMENT: [
        "I will", "I'll", "I promise", "I'm going to",
        "count on me", "I commit",
    ],
    SemanticMode.OBSERVATION: [
        "I notice", "I see that", "it appears", "looking at",
        "I found", "I observe",
    ],
    SemanticMode.REFLECTION: [
        "thinking about", "reflecting on", "looking back",
        "in hindsight", "reconsidering",
    ],
}

SENTIMENT_SURFACE_MAP: dict[Sentiment, list[str]] = {
    Sentiment.POSITIVE: [
        "good", "great", "excellent", "nice", "wonderful",
        "love", "like", "happy", "pleased", "fantastic",
    ],
    Sentiment.NEGATIVE: [
        "bad", "terrible", "awful", "hate", "dislike",
        "poor", "wrong", "unfortunately", "sadly",
    ],
    Sentiment.NEUTRAL: [],  # default when nothing matches
    Sentiment.MIXED: [
        "but", "however", "although", "on one hand",
        "trade-off", "mixed feelings",
    ],
}

TIME_SURFACE_MAP: dict[TimeReference, list[str]] = {
    TimeReference.PAST: [
        "was", "were", "had", "did", "used to",
        "previously", "before", "yesterday", "last",
    ],
    TimeReference.PRESENT: [
        "is", "are", "now", "currently", "right now",
        "at the moment", "today",
    ],
    TimeReference.FUTURE: [
        "will", "going to", "shall", "tomorrow", "next",
        "soon", "eventually", "later",
    ],
    TimeReference.ATEMPORAL: [
        "always", "never", "in general", "by definition",
        "universally",
    ],
}

# Certainty hints — maps keywords to approximate certainty values
CERTAINTY_HINTS: list[tuple[str, float]] = [
    ("definitely", 0.95),
    ("certainly", 0.93),
    ("clearly", 0.90),
    ("probably", 0.75),
    ("likely", 0.72),
    ("I think", 0.70),
    ("I believe", 0.72),
    ("seems", 0.60),
    ("maybe", 0.45),
    ("perhaps", 0.42),
    ("might", 0.40),
    ("possibly", 0.35),
    ("I guess", 0.35),
    ("not sure", 0.30),
    ("unlikely", 0.20),
    ("doubtful", 0.18),
]


# ---------------------------------------------------------------------------
# NormalizationVocabulary
# ---------------------------------------------------------------------------

@dataclass
class NormalizationVocabulary:
    """
    Holds surface-to-canonical mapping tables and provides
    lookup methods for the parser.

    The vocabulary is deterministic and rule-based.
    It can be extended at runtime by adding entries to the maps.
    """
    act_map: dict[SemanticAct, list[str]] = field(default_factory=lambda: dict(ACT_SURFACE_MAP))
    mode_map: dict[SemanticMode, list[str]] = field(default_factory=lambda: dict(MODE_SURFACE_MAP))
    sentiment_map: dict[Sentiment, list[str]] = field(default_factory=lambda: dict(SENTIMENT_SURFACE_MAP))
    time_map: dict[TimeReference, list[str]] = field(default_factory=lambda: dict(TIME_SURFACE_MAP))
    certainty_hints: list[tuple[str, float]] = field(default_factory=lambda: list(CERTAINTY_HINTS))

    def match_act(self, text: str) -> SemanticAct:
        """Find the best-matching SemanticAct for the given text."""
        lower = text.lower()
        best_act = SemanticAct.UNKNOWN
        best_pos = len(lower)  # earliest match wins
        for act, patterns in self.act_map.items():
            for pattern in patterns:
                pos = lower.find(pattern.lower())
                if pos != -1 and pos < best_pos:
                    best_pos = pos
                    best_act = act
        return best_act

    def match_mode(self, text: str) -> SemanticMode:
        """Find the best-matching SemanticMode."""
        lower = text.lower()
        # Question mark is a strong signal
        if "?" in text:
            return SemanticMode.QUESTION
        best_mode = SemanticMode.ASSERTION
        best_pos = len(lower)
        for mode, patterns in self.mode_map.items():
            for pattern in patterns:
                pos = lower.find(pattern.lower())
                if pos != -1 and pos < best_pos:
                    best_pos = pos
                    best_mode = mode
        return best_mode

    def match_sentiment(self, text: str) -> Sentiment:
        """Find the best-matching Sentiment."""
        lower = text.lower()
        scores: dict[Sentiment, int] = {s: 0 for s in Sentiment}
        for sentiment, patterns in self.sentiment_map.items():
            for pattern in patterns:
                if pattern.lower() in lower:
                    scores[sentiment] += 1
        # If both positive and negative are present, it's mixed
        if scores[Sentiment.POSITIVE] > 0 and scores[Sentiment.NEGATIVE] > 0:
            return Sentiment.MIXED
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return Sentiment.NEUTRAL
        return best

    def match_time(self, text: str) -> TimeReference:
        """Find the best-matching TimeReference."""
        lower = text.lower()
        scores: dict[TimeReference, int] = {t: 0 for t in TimeReference}
        for time_ref, patterns in self.time_map.items():
            for pattern in patterns:
                if pattern.lower() in lower:
                    scores[time_ref] += 1
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return TimeReference.PRESENT  # default assumption
        return best

    def estimate_certainty(self, text: str) -> float:
        """Estimate certainty level from surface cues. Returns 0.0–1.0."""
        lower = text.lower()
        for pattern, value in self.certainty_hints:
            if pattern.lower() in lower:
                return value
        return 0.7  # default moderate certainty

    def normalize_object_key(self, raw: str) -> str:
        """
        Convert a raw object description into a normalized key.

        E.g. "one-to-one sentence mapping" → "sentence_1to1_mapping"
        """
        text = raw.lower().strip()
        # Strip trailing punctuation
        text = re.sub(r'[.!?,;:]+$', '', text).strip()
        # Common substitutions
        text = re.sub(r'\bone[- ]to[- ]one\b', '1to1', text)
        text = re.sub(r'\bone[- ]to[- ]many\b', '1toN', text)
        text = re.sub(r'\bmany[- ]to[- ]many\b', 'NtoN', text)
        # Remove filler words
        fillers = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                   "will", "would", "could", "should", "it", "that", "this",
                   "very", "really", "quite", "just"}
        tokens = re.split(r'[\s\-/]+', text)
        tokens = [t for t in tokens if t and t not in fillers]
        return "_".join(tokens)
