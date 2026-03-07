"""
Core domain models for AILang-IR semantic representation.

These types define the canonical structured meaning objects that the system
operates on. Raw text is evidence; these models are the operational units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime
import uuid


# ---------------------------------------------------------------------------
# Value enums — constrained semantic vocabulary
# ---------------------------------------------------------------------------

class SpeakerRole(Enum):
    """Who produced the utterance."""
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class SemanticMode(Enum):
    """Epistemic / speech-act mode of the utterance."""
    ASSERTION = "assertion"        # stating a fact
    OPINION = "opinion"            # subjective belief
    HYPOTHESIS = "hypothesis"      # tentative/uncertain claim
    REQUEST = "request"            # asking for something
    COMMAND = "command"            # directive
    QUESTION = "question"          # seeking information
    COMMITMENT = "commitment"      # promising future action
    OBSERVATION = "observation"    # reporting perceived state
    REFLECTION = "reflection"      # meta-cognitive statement


class SemanticAct(Enum):
    """Core predicate category — what is being done semantically."""
    BELIEVE = "believe"
    PREFER = "prefer"
    SUGGEST = "suggest"
    NEED = "need"
    DECIDE = "decide"
    REJECT = "reject"
    AGREE = "agree"
    DISAGREE = "disagree"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    QUERY = "query"
    OBSERVE = "observe"
    COMPARE = "compare"
    PLAN = "plan"
    WARN = "warn"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"


class TimeReference(Enum):
    """Temporal anchor of the semantic content."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    ATEMPORAL = "atemporal"  # timeless truths / definitions
    UNKNOWN = "unknown"


class Sentiment(Enum):
    """Affective valence."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Priority(Enum):
    """Operational importance."""
    P0 = "P0"  # critical
    P1 = "P1"  # high
    P2 = "P2"  # normal
    P3 = "P3"  # low
    UNSET = "unset"


# ---------------------------------------------------------------------------
# Certainty — continuous value with semantic boundaries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Certainty:
    """
    Confidence level of the semantic content, 0.0 to 1.0.

    Semantic boundaries:
      0.0–0.3  low confidence / speculative
      0.3–0.6  moderate / hedged
      0.6–0.85 confident
      0.85–1.0 near-certain / factual claim
    """
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Certainty must be in [0.0, 1.0], got {self.value}")

    @property
    def label(self) -> str:
        if self.value < 0.3:
            return "speculative"
        if self.value < 0.6:
            return "hedged"
        if self.value < 0.85:
            return "confident"
        return "near_certain"

    def compact(self) -> str:
        """Compact integer form for symbolic encoding, e.g. 'C84'."""
        return f"C{int(self.value * 100)}"

    def __repr__(self) -> str:
        return f"Certainty({self.value:.2f}, {self.label})"


# ---------------------------------------------------------------------------
# Entity — a named thing in the semantic world
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    """
    A semantic entity — any identifiable concept, object, or referent.

    `canonical` is the normalized name used for storage and matching.
    `surface_forms` preserves the original text variants seen.
    """
    canonical: str
    surface_forms: tuple[str, ...] = ()
    entity_type: str = "concept"  # concept | person | system | artifact | ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.canonical)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.canonical == other.canonical
        return NotImplemented


# ---------------------------------------------------------------------------
# Predicate — the relational core of a semantic statement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predicate:
    """
    The action or relation expressed in a semantic frame.

    Combines a SemanticAct with optional refinement.
    """
    act: SemanticAct
    modifier: str | None = None  # e.g. "strongly", "tentatively"

    @property
    def canonical(self) -> str:
        if self.modifier:
            return f"{self.modifier}_{self.act.value}"
        return self.act.value


# ---------------------------------------------------------------------------
# SemanticFrame — the primary meaning unit
# ---------------------------------------------------------------------------

@dataclass
class SemanticFrame:
    """
    The core structured meaning representation.

    One SemanticFrame captures the essential meaning of a single
    semantic event or statement. It is the canonical unit of operation
    for storage, retrieval, compression, and reconstruction.
    """
    frame_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    speaker: SpeakerRole = SpeakerRole.UNKNOWN
    mode: SemanticMode = SemanticMode.ASSERTION
    predicate: Predicate = field(default_factory=lambda: Predicate(SemanticAct.UNKNOWN))
    object: Entity | None = None
    target: Entity | None = None
    time: TimeReference = TimeReference.UNKNOWN
    certainty: Certainty = field(default_factory=lambda: Certainty(0.5))
    sentiment: Sentiment = Sentiment.UNKNOWN
    priority: Priority = Priority.UNSET
    source_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def act(self) -> SemanticAct:
        return self.predicate.act

    def summary(self) -> str:
        """One-line human-readable summary."""
        obj = self.object.canonical if self.object else "?"
        tgt = f" → {self.target.canonical}" if self.target else ""
        return (
            f"[{self.speaker.value}] {self.mode.value}:"
            f" {self.predicate.canonical}({obj}{tgt})"
            f" {self.certainty.compact()} {self.time.value}"
        )


# ---------------------------------------------------------------------------
# EventMemory — a stored semantic event with temporal metadata
# ---------------------------------------------------------------------------

@dataclass
class EventMemory:
    """
    A semantic frame anchored in time and stored for future retrieval.

    Adds storage-oriented metadata on top of the raw SemanticFrame.
    """
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    frame: SemanticFrame = field(default_factory=SemanticFrame)
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime | None = None
    superseded_by: str | None = None  # memory_id of replacement, if any
    tags: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None

    def touch(self) -> None:
        """Record an access."""
        self.access_count += 1
        self.last_accessed = datetime.now()


# ---------------------------------------------------------------------------
# RelationEdge — a typed link between two entities or frames
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationEdge:
    """
    A directed semantic relation between two entities.

    The graph of RelationEdges forms the semantic memory graph.
    """
    source: Entity
    target: Entity
    relation: str  # e.g. "prefers_over", "contradicts", "supports", "part_of"
    weight: float = 1.0
    evidence_frame_id: str | None = None  # the frame that established this edge
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CompressionRule — a deterministic mapping for symbolic encoding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompressionRule:
    """
    A rule for mapping a semantic pattern to a compact symbolic code.

    Used by the encoder to produce deterministic compact representations.
    """
    rule_id: str
    pattern: dict[str, str]  # field-name -> expected value pattern
    compact_template: str    # e.g. "{speaker}|{mode}|{act}|{object}|{certainty}|{time}"
    priority: int = 0        # higher = matched first


# ---------------------------------------------------------------------------
# ReconstructionPlan — a plan for rebuilding natural language
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionPlan:
    """
    A structured plan for reconstructing natural language from a semantic frame.

    The decoder uses this to control output style and completeness.
    """
    frame: SemanticFrame
    style: str = "declarative"  # declarative | conversational | summary
    include_certainty: bool = True
    include_sentiment: bool = False
    include_time: bool = True
    max_tokens: int | None = None
    template: str | None = None  # optional override template
