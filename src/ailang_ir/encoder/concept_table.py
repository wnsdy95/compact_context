"""
ConceptTable: symbol table for v2 compact encoding.

Maps concept keys to short base-36 IDs, enabling re-reference
of previously seen concepts with minimal bytes (e.g. $0, $1a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


def encode_id(n: int) -> str:
    """Encode a non-negative integer as a base-36 string (0-9, a-z)."""
    if n < 0:
        raise ValueError(f"ID must be non-negative, got {n}")
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[n % 36])
        n //= 36
    return "".join(reversed(digits))


def decode_id(s: str) -> int:
    """Decode a base-36 string back to an integer."""
    return int(s, 36)


@dataclass
class ConceptTable:
    """
    Symbol table that maps concept keys to short numeric IDs.

    First mention of a concept registers it (define), subsequent
    mentions use the short $id form (lookup).
    """
    _key_to_id: dict[str, int] = field(default_factory=dict)
    _id_to_key: dict[int, str] = field(default_factory=dict)
    _next_id: int = 0

    def define(self, key: str) -> int:
        """Register a new concept key. Returns its ID. Idempotent."""
        if key in self._key_to_id:
            return self._key_to_id[key]
        cid = self._next_id
        self._key_to_id[key] = cid
        self._id_to_key[cid] = key
        self._next_id += 1
        return cid

    def lookup(self, key: str) -> int | None:
        """Look up a concept key. Returns ID or None if not defined."""
        return self._key_to_id.get(key)

    def resolve(self, cid: int) -> str | None:
        """Resolve an ID back to its concept key."""
        return self._id_to_key.get(cid)

    def has(self, key: str) -> bool:
        """Check if a concept key is already defined."""
        return key in self._key_to_id

    @property
    def size(self) -> int:
        return len(self._key_to_id)

    def dump(self) -> dict[str, Any]:
        """Serialize to a plain dict for persistence."""
        return {
            "entries": {k: v for k, v in self._key_to_id.items()},
            "next_id": self._next_id,
        }

    @classmethod
    def from_dump(cls, data: dict[str, Any]) -> "ConceptTable":
        """Restore from a serialized dict."""
        table = cls()
        for key, cid in data.get("entries", {}).items():
            table._key_to_id[key] = cid
            table._id_to_key[cid] = key
        table._next_id = data.get("next_id", 0)
        return table

    def ref(self, key: str) -> str:
        """Return the reference string for a key.

        First call for a key: defines it and returns ``#key``.
        Subsequent calls: returns ``$base36_id``.
        """
        existing = self.lookup(key)
        if existing is not None:
            return f"${encode_id(existing)}"
        self.define(key)
        return f"#{key}"

    def ref_fuzzy(self, key: str, threshold: float = 0.7) -> str:
        """Like ref(), but checks for similar existing keys before defining.

        If a sufficiently similar key already exists, returns its $id reference
        instead of defining a new entry. This enables re-reference for
        semantically equivalent but lexically different concept keys.

        Returns:
            ``$base36_id`` if exact or fuzzy match found.
            ``#key`` if new key is defined.
        """
        # Exact match first
        existing = self.lookup(key)
        if existing is not None:
            return f"${encode_id(existing)}"

        # Fuzzy match against existing keys
        best_key: str | None = None
        best_score = 0.0
        key_words = set(key.split("_"))
        for existing_key in self._key_to_id:
            # Word overlap
            ew = set(existing_key.split("_"))
            if key_words and ew:
                jaccard = len(key_words & ew) / len(key_words | ew)
            else:
                jaccard = 0.0
            # Containment
            if key in existing_key or existing_key in key:
                contain = 0.9
            else:
                contain = 0.0
            # Sequence match
            seq = SequenceMatcher(None, key, existing_key).ratio()
            score = max(jaccard, contain, seq)
            if score > best_score:
                best_score = score
                best_key = existing_key

        if best_key is not None and best_score >= threshold:
            cid = self._key_to_id[best_key]
            return f"${encode_id(cid)}"

        # No match — define new
        self.define(key)
        return f"#{key}"
