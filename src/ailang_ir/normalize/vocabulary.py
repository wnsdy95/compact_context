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
        "develop", "write", "generate", "added", "add",
        "implemented", "built", "created", "wrote",
    ],
    SemanticAct.MODIFY: [
        "modify", "change", "update", "edit", "revise",
        "adjust", "alter", "refactor", "updated", "changed",
        "fixed", "fix", "improved", "improve",
    ],
    SemanticAct.DELETE: [
        "delete", "remove", "drop", "discard", "eliminate",
        "get rid of", "removed", "deleted",
    ],
    SemanticAct.QUERY: [
        "what is", "what are", "how does", "how do",
        "why", "when", "where", "is it", "can you",
    ],
    SemanticAct.OBSERVE: [
        "notice", "see", "observe", "found", "discovered",
        "it seems", "appears", "looks like",
        "shows", "reveals", "indicates", "confirms",
        "results show", "verification shows",
        "passing", "passed", "completed", "done",
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
                pat_lower = pattern.lower()
                # Short patterns (≤3 chars like "no", "yes", "use") need word boundary
                if len(pat_lower) <= 3:
                    m = re.search(r'\b' + re.escape(pat_lower) + r'\b', lower)
                    if m and m.start() < best_pos:
                        best_pos = m.start()
                        best_act = act
                else:
                    pos = lower.find(pat_lower)
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
        # Remove filler words (pronouns, articles, auxiliaries, common non-content words)
        fillers = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                   "will", "would", "could", "should", "it", "that", "this",
                   "very", "really", "quite", "just",
                   # Pronouns — never meaningful as object key content
                   "i", "we", "you", "he", "she", "they", "me", "us",
                   "my", "your", "our", "his", "her", "their", "its",
                   # Common non-content verbs and adverbs
                   "have", "has", "had", "do", "does", "did",
                   "also", "too", "then", "still", "even",
                   # Prepositions and conjunctions
                   "for", "of", "on", "in", "at", "by", "to", "as", "so", "if",
                   "and", "or", "but", "not", "no", "yes",
                   }
        tokens = re.split(r'[\s\-/]+', text)
        tokens = [re.sub(r'[^a-z0-9_]', '', t) for t in tokens]
        tokens = [t for t in tokens if t and t not in fillers]
        return "_".join(tokens)

    def compress_object_key(self, raw: str) -> str:
        """
        Compress a normalized key into a shorter form for v2 encoding.

        Rules:
        1. Remove extended filler words (prepositions, etc.)
        2. Truncate words longer than 7 chars to first 4 chars
        3. Cap total length at 24 chars
        """
        # Start from normalized form
        key = self.normalize_object_key(raw)
        extended_fillers = {
            "for", "of", "with", "about", "to", "in", "on", "at",
            "from", "by", "than", "over", "into", "like", "between",
            "through", "after", "before", "under", "against",
        }
        tokens = key.split("_")
        tokens = [t for t in tokens if t not in extended_fillers]
        # Truncate long words: >7 chars → first 4 chars
        tokens = [t[:4] if len(t) > 7 else t for t in tokens]
        result = "_".join(tokens)
        # Cap at 24 chars
        if len(result) > 24:
            result = result[:24].rstrip("_")
        return result

    def compress_object_key_v3(self, raw: str) -> str:
        """
        Ultra-compact key compression for v3 encoding.

        Strategy:
        1. Start from v2 compressed key (fillers removed, long words truncated)
        2. Keep max 3 content tokens
        3. Apply stem abbreviation dictionary (common stems → 2 chars)
        4. Remaining tokens → first 2 chars
        5. Concatenate without separators
        """
        base = self.compress_object_key(raw)
        tokens = base.split("_")[:3]
        parts = [STEM_ABBREVS.get(t, t[:2]) for t in tokens if t]
        return "".join(parts)


# ---------------------------------------------------------------------------
# Stem abbreviation dictionary for v3 encoding
# ---------------------------------------------------------------------------

STEM_ABBREVS: dict[str, str] = {
    # Structural / general
    "1to1": "11", "1ton": "1n", "nton": "nn",
    "new": "nw", "old": "ol", "best": "bs", "good": "gd",
    "right": "rt", "great": "gt", "poor": "pr", "diff": "df",
    "next": "nx", "prev": "pv", "used": "ud", "go": "go",
    "we": "we", "use": "us", "set": "st", "get": "gt",
    "add": "ad", "run": "rn", "try": "tr", "way": "wy",
    "well": "wl", "bad": "bd", "high": "hi", "low": "lo",
    # Domain-specific
    "sent": "sn", "mapping": "mp", "map": "mp", "reco": "rc",
    "natural": "nl", "lang": "lg", "graph": "gr", "memory": "mm",
    "memo": "mm", "linear": "ln", "text": "tx", "storage": "st",
    "store": "st", "sema": "sm", "semantic": "sm",
    "frames": "fr", "frame": "fr", "comp": "cp",
    "norm": "nm", "parser": "ps", "parse": "ps",
    "model": "md", "models": "md", "typed": "tp", "loose": "ls",
    "dict": "dc", "impl": "im", "pers": "pr",
    "cycle": "cy", "encoder": "en", "encode": "en",
    "decoder": "de", "decode": "de",
    "quality": "ql", "output": "ou", "input": "in",
    "layer": "ly", "appr": "ap", "approach": "ap",
    "arch": "ac", "impr": "ip", "improve": "ip",
    "handle": "hn", "ambi": "ab", "passive": "pv",
    "rules": "rl", "rule": "rl", "string": "sg",
    "code": "cd", "codes": "cd", "compact": "cm",
    "pres": "ps", "preserve": "ps", "raw": "rw",
    "ratio": "ra", "stra": "sa", "strategy": "sa",
    "segm": "sg", "segment": "sg", "mean": "mn",
    "meaning": "mn", "cons": "cn", "consistent": "cn",
    "unfo": "uf", "misc": "mc", "defi": "df",
    "avoid": "av", "system": "sy", "project": "pj",
    "context": "cx", "session": "ss", "user": "ur",
    "agent": "ag", "build": "bl", "create": "cr",
    "update": "up", "delete": "dl", "search": "sr",
    "query": "qr", "data": "da", "type": "ty",
    "value": "vl", "key": "ky", "index": "ix",
    "config": "cf", "error": "er", "debug": "db",
    "test": "ts", "spec": "sp", "later": "lt",
}
