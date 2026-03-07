"""
Semantic memory store.

Stores EventMemory objects and supports retrieval by entity, predicate,
topic similarity, and temporal range. Supports repeated-meaning merging
and contradiction detection.

This is an in-memory MVP. Storage backend can be swapped later
without changing the interface.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ailang_ir.models.domain import (
    Certainty,
    EventMemory,
    Entity,
    Predicate,
    Priority,
    RelationEdge,
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    Sentiment,
    SpeakerRole,
    TimeReference,
)
from ailang_ir.encoder.concept_table import ConceptTable


@dataclass
class MemoryStore:
    """
    In-memory semantic event store.

    Stores EventMemory entries and maintains an entity relation graph.
    Supports retrieval, deduplication, and basic conflict detection.
    """
    _memories: dict[str, EventMemory] = field(default_factory=dict)
    _edges: list[RelationEdge] = field(default_factory=list)
    # Index: entity canonical → list of memory_ids
    _entity_index: dict[str, list[str]] = field(default_factory=dict)
    # Index: act → list of memory_ids
    _act_index: dict[SemanticAct, list[str]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self._memories)

    @property
    def active_count(self) -> int:
        return sum(1 for m in self._memories.values() if m.is_active)

    # -------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------

    def store(self, frame: SemanticFrame, tags: list[str] | None = None) -> EventMemory:
        """
        Store a SemanticFrame as an EventMemory.

        Before storing, checks for duplicate or contradictory memories.
        Returns the stored (or merged) EventMemory.
        """
        # Check for duplicate
        existing = self._find_duplicate(frame)
        if existing:
            existing.touch()
            return existing

        memory = EventMemory(
            frame=frame,
            tags=tags or [],
        )
        self._memories[memory.memory_id] = memory
        self._index_memory(memory)
        return memory

    def store_edge(self, edge: RelationEdge) -> None:
        """Store a relation edge in the semantic graph."""
        self._edges.append(edge)

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    def get(self, memory_id: str) -> EventMemory | None:
        """Get a specific memory by ID."""
        mem = self._memories.get(memory_id)
        if mem:
            mem.touch()
        return mem

    def query_by_entity(self, entity_key: str) -> list[EventMemory]:
        """Find all active memories mentioning an entity."""
        ids = self._entity_index.get(entity_key, [])
        return [
            m for mid in ids
            if (m := self._memories.get(mid)) and m.is_active
        ]

    def query_by_act(self, act: SemanticAct) -> list[EventMemory]:
        """Find all active memories with a given act."""
        ids = self._act_index.get(act, [])
        return [
            m for mid in ids
            if (m := self._memories.get(mid)) and m.is_active
        ]

    def query_by_speaker(self, speaker: SpeakerRole) -> list[EventMemory]:
        """Find all active memories from a given speaker."""
        return [
            m for m in self._memories.values()
            if m.is_active and m.frame.speaker == speaker
        ]

    def query_by_tag(self, tag: str) -> list[EventMemory]:
        """Find all active memories with a given tag."""
        return [
            m for m in self._memories.values()
            if m.is_active and tag in m.tags
        ]

    def query(self, predicate: Callable[[EventMemory], bool]) -> list[EventMemory]:
        """General-purpose query with a custom predicate function."""
        return [
            m for m in self._memories.values()
            if m.is_active and predicate(m)
        ]

    def query_recent(self, n: int = 10) -> list[EventMemory]:
        """Get the N most recently stored active memories."""
        active = [m for m in self._memories.values() if m.is_active]
        active.sort(key=lambda m: m.timestamp, reverse=True)
        return active[:n]

    def get_edges_for(self, entity_key: str) -> list[RelationEdge]:
        """Get all relation edges involving an entity."""
        return [
            e for e in self._edges
            if e.source.canonical == entity_key or e.target.canonical == entity_key
        ]

    # -------------------------------------------------------------------
    # Conflict detection and merging
    # -------------------------------------------------------------------

    def find_contradictions(self, frame: SemanticFrame) -> list[EventMemory]:
        """
        Find stored memories that may contradict the given frame.

        A contradiction is detected when:
        - same object entity
        - opposing acts (e.g. PREFER vs REJECT, AGREE vs DISAGREE)
        - different speaker or different time may not be contradictions
        """
        if not frame.object:
            return []

        candidates = self.query_by_entity(frame.object.canonical)
        contradictions = []

        opposing_acts = {
            SemanticAct.PREFER: {SemanticAct.REJECT},
            SemanticAct.REJECT: {SemanticAct.PREFER, SemanticAct.AGREE},
            SemanticAct.AGREE: {SemanticAct.DISAGREE, SemanticAct.REJECT},
            SemanticAct.DISAGREE: {SemanticAct.AGREE},
            SemanticAct.CREATE: {SemanticAct.DELETE},
            SemanticAct.DELETE: {SemanticAct.CREATE},
        }

        opposites = opposing_acts.get(frame.act, set())
        for mem in candidates:
            if mem.frame.act in opposites and mem.frame.speaker == frame.speaker:
                contradictions.append(mem)

        return contradictions

    def supersede(self, old_id: str, new_frame: SemanticFrame) -> EventMemory:
        """
        Supersede an existing memory with a new one.

        Marks the old memory as superseded and stores the new frame.
        """
        old = self._memories.get(old_id)
        if old is None:
            raise KeyError(f"Memory {old_id} not found")

        new_mem = self.store(new_frame)
        old.superseded_by = new_mem.memory_id

        # Record the supersession as a relation edge
        if old.frame.object and new_mem.frame.object:
            self.store_edge(RelationEdge(
                source=new_mem.frame.object,
                target=old.frame.object,
                relation="supersedes",
                evidence_frame_id=new_mem.frame.frame_id,
            ))

        return new_mem

    # -------------------------------------------------------------------
    # Fuzzy search and relevance
    # -------------------------------------------------------------------

    @staticmethod
    def _entity_similarity(key1: str, key2: str) -> float:
        """Compute similarity between two entity keys.

        Uses multiple heuristics and returns the max score:
        1. Exact match → 1.0
        2. Containment → 0.9
        3. Word-level Jaccard → 0.0-1.0
        4. SequenceMatcher fallback → 0.0-1.0
        """
        if key1 == key2:
            return 1.0
        # Containment
        if key1 in key2 or key2 in key1:
            return 0.9
        # Word overlap (Jaccard)
        w1 = set(key1.split("_"))
        w2 = set(key2.split("_"))
        if w1 and w2:
            jaccard = len(w1 & w2) / len(w1 | w2)
        else:
            jaccard = 0.0
        # SequenceMatcher
        seq = SequenceMatcher(None, key1, key2).ratio()
        return max(jaccard, seq)

    def query_by_entity_fuzzy(
        self, key: str, threshold: float = 0.6,
    ) -> list[tuple[EventMemory, float]]:
        """Find active memories with entities similar to key.

        Returns list of (memory, similarity) tuples sorted by similarity desc.
        """
        results: dict[str, float] = {}  # memory_id → best similarity
        for entity_key, mids in self._entity_index.items():
            sim = self._entity_similarity(key, entity_key)
            if sim >= threshold:
                for mid in mids:
                    mem = self._memories.get(mid)
                    if mem and mem.is_active:
                        results[mid] = max(results.get(mid, 0.0), sim)

        scored = [
            (self._memories[mid], sim)
            for mid, sim in results.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def query_relevant(
        self, context_keys: list[str], n: int = 10,
    ) -> list[EventMemory]:
        """Find most relevant memories for the given context keys.

        Scoring:
          entity_score = max similarity to any context key (0.6 weight)
          recency_score = 1 / (1 + hours since creation) (0.2 weight)
          access_score = log(1 + access_count) / 10 (0.2 weight)
        """
        now = datetime.now()
        scored: dict[str, float] = {}

        for mid, mem in self._memories.items():
            if not mem.is_active:
                continue

            # Entity score: max similarity to any context key
            entity_score = 0.0
            mem_keys = []
            if mem.frame.object:
                mem_keys.append(mem.frame.object.canonical)
            if mem.frame.target:
                mem_keys.append(mem.frame.target.canonical)
            for mk in mem_keys:
                for ck in context_keys:
                    entity_score = max(entity_score, self._entity_similarity(mk, ck))

            # Recency score
            hours = max((now - mem.timestamp).total_seconds() / 3600, 0)
            recency_score = 1.0 / (1.0 + hours)

            # Access score
            access_score = math.log1p(mem.access_count) / 10.0

            total = entity_score * 0.6 + recency_score * 0.2 + access_score * 0.2
            scored[mid] = total

        top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:n]
        return [self._memories[mid] for mid, _ in top]

    def consolidate(
        self, entity_key: str, threshold: float = 0.6,
    ) -> EventMemory | None:
        """Merge multiple memories about the same entity into one summary.

        Returns the consolidated memory, or None if fewer than 2 matches.
        """
        matches = self.query_by_entity_fuzzy(entity_key, threshold)
        if len(matches) < 2:
            return None

        # Pick most recent as representative
        memories = [m for m, _ in matches]
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        representative = memories[0]

        # Build consolidated frame from most recent
        consolidated_frame = SemanticFrame(
            speaker=representative.frame.speaker,
            mode=representative.frame.mode,
            predicate=representative.frame.predicate,
            object=representative.frame.object,
            target=representative.frame.target,
            time=representative.frame.time,
            certainty=representative.frame.certainty,
            sentiment=representative.frame.sentiment,
            source_text=f"[consolidated from {len(memories)} memories]",
            metadata={"consolidated_from": [m.memory_id for m in memories]},
        )

        # Supersede all old memories
        for mem in memories:
            mem.superseded_by = "consolidated"

        # Store new
        new_mem = EventMemory(
            frame=consolidated_frame,
            tags=["consolidated"],
        )
        self._memories[new_mem.memory_id] = new_mem
        self._index_memory(new_mem)
        return new_mem

    # -------------------------------------------------------------------
    # Deduplication
    # -------------------------------------------------------------------

    def _find_duplicate(self, frame: SemanticFrame) -> EventMemory | None:
        """
        Check if an essentially identical meaning already exists.

        Two frames are considered duplicates if they share:
        - same act
        - same object canonical key
        - same speaker
        """
        if not frame.object:
            return None

        candidates = self.query_by_entity(frame.object.canonical)
        for mem in candidates:
            if (
                mem.frame.act == frame.act
                and mem.frame.speaker == frame.speaker
                and mem.frame.object
                and mem.frame.object.canonical == frame.object.canonical
            ):
                return mem
        return None

    # -------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------

    def _index_memory(self, memory: EventMemory) -> None:
        """Add a memory to the internal indexes."""
        mid = memory.memory_id
        frame = memory.frame

        # Entity index
        if frame.object:
            self._entity_index.setdefault(frame.object.canonical, []).append(mid)
        if frame.target:
            self._entity_index.setdefault(frame.target.canonical, []).append(mid)

        # Act index
        self._act_index.setdefault(frame.act, []).append(mid)

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    def dump(self) -> list[dict]:
        """Dump all active memories as a list of dicts for debugging."""
        result = []
        for mem in self._memories.values():
            if not mem.is_active:
                continue
            entry = {
                "memory_id": mem.memory_id,
                "summary": mem.frame.summary(),
                "access_count": mem.access_count,
                "tags": mem.tags,
                "timestamp": mem.timestamp.isoformat(),
            }
            if mem.frame.source_text:
                entry["source_text"] = mem.frame.source_text
            result.append(entry)
        return result

    # -------------------------------------------------------------------
    # Persistence — JSON file save/load
    # -------------------------------------------------------------------

    def save(
        self, path: str | Path,
        concept_table: ConceptTable | None = None,
    ) -> None:
        """Save the entire memory store to a JSON file."""
        path = Path(path)
        data: dict[str, Any] = {
            "version": 2,
            "memories": {
                mid: _serialize_memory(mem)
                for mid, mem in self._memories.items()
            },
            "edges": [_serialize_edge(e) for e in self._edges],
        }
        if concept_table is not None:
            data["concept_table"] = concept_table.dump()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(
        cls, path: str | Path,
    ) -> tuple["MemoryStore", ConceptTable | None]:
        """Load a memory store from a JSON file.

        Returns (store, concept_table). concept_table is None for v1 files.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        for mid, mem_data in data.get("memories", {}).items():
            mem = _deserialize_memory(mem_data)
            store._memories[mid] = mem
            store._index_memory(mem)
        for edge_data in data.get("edges", []):
            store._edges.append(_deserialize_edge(edge_data))
        ct = None
        if "concept_table" in data:
            ct = ConceptTable.from_dump(data["concept_table"])
        return store, ct


# ---------------------------------------------------------------------------
# Serialization helpers (module-level, private)
# ---------------------------------------------------------------------------

def _serialize_frame(f: SemanticFrame) -> dict[str, Any]:
    return {
        "frame_id": f.frame_id,
        "speaker": f.speaker.value,
        "mode": f.mode.value,
        "act": f.predicate.act.value,
        "predicate_modifier": f.predicate.modifier,
        "object": _serialize_entity(f.object) if f.object else None,
        "target": _serialize_entity(f.target) if f.target else None,
        "time": f.time.value,
        "certainty": f.certainty.value,
        "sentiment": f.sentiment.value,
        "priority": f.priority.value,
        "source_text": f.source_text,
        "metadata": f.metadata,
        "created_at": f.created_at.isoformat(),
    }


def _deserialize_frame(d: dict[str, Any]) -> SemanticFrame:
    return SemanticFrame(
        frame_id=d["frame_id"],
        speaker=SpeakerRole(d["speaker"]),
        mode=SemanticMode(d["mode"]),
        predicate=Predicate(
            act=SemanticAct(d["act"]),
            modifier=d.get("predicate_modifier"),
        ),
        object=_deserialize_entity(d["object"]) if d.get("object") else None,
        target=_deserialize_entity(d["target"]) if d.get("target") else None,
        time=TimeReference(d["time"]),
        certainty=Certainty(d["certainty"]),
        sentiment=Sentiment(d["sentiment"]),
        priority=Priority(d["priority"]),
        source_text=d.get("source_text"),
        metadata=d.get("metadata", {}),
        created_at=datetime.fromisoformat(d["created_at"]),
    )


def _serialize_entity(e: Entity) -> dict[str, Any]:
    return {
        "canonical": e.canonical,
        "surface_forms": list(e.surface_forms),
        "entity_type": e.entity_type,
        "metadata": e.metadata,
    }


def _deserialize_entity(d: dict[str, Any]) -> Entity:
    return Entity(
        canonical=d["canonical"],
        surface_forms=tuple(d.get("surface_forms", ())),
        entity_type=d.get("entity_type", "concept"),
        metadata=d.get("metadata", {}),
    )


def _serialize_memory(m: EventMemory) -> dict[str, Any]:
    return {
        "memory_id": m.memory_id,
        "frame": _serialize_frame(m.frame),
        "timestamp": m.timestamp.isoformat(),
        "access_count": m.access_count,
        "last_accessed": m.last_accessed.isoformat() if m.last_accessed else None,
        "superseded_by": m.superseded_by,
        "tags": m.tags,
    }


def _deserialize_memory(d: dict[str, Any]) -> EventMemory:
    return EventMemory(
        memory_id=d["memory_id"],
        frame=_deserialize_frame(d["frame"]),
        timestamp=datetime.fromisoformat(d["timestamp"]),
        access_count=d.get("access_count", 0),
        last_accessed=datetime.fromisoformat(d["last_accessed"]) if d.get("last_accessed") else None,
        superseded_by=d.get("superseded_by"),
        tags=d.get("tags", []),
    )


def _serialize_edge(e: RelationEdge) -> dict[str, Any]:
    return {
        "source": _serialize_entity(e.source),
        "target": _serialize_entity(e.target),
        "relation": e.relation,
        "weight": e.weight,
        "evidence_frame_id": e.evidence_frame_id,
        "metadata": e.metadata,
    }


def _deserialize_edge(d: dict[str, Any]) -> RelationEdge:
    return RelationEdge(
        source=_deserialize_entity(d["source"]),
        target=_deserialize_entity(d["target"]),
        relation=d["relation"],
        weight=d.get("weight", 1.0),
        evidence_frame_id=d.get("evidence_frame_id"),
        metadata=d.get("metadata", {}),
    )
