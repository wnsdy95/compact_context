"""
Semantic memory store.

Stores EventMemory objects and supports retrieval by entity, predicate,
topic similarity, and temporal range. Supports repeated-meaning merging
and contradiction detection.

This is an in-memory MVP. Storage backend can be swapped later
without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from ailang_ir.models.domain import (
    EventMemory,
    Entity,
    RelationEdge,
    SemanticAct,
    SemanticFrame,
    SpeakerRole,
)


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
