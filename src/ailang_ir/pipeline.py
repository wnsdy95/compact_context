"""
Unified semantic pipeline.

Single entry point that chains parser → encoder → decoder → memory.
This is the primary public API for AILang-IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ailang_ir.models.domain import (
    EventMemory,
    SemanticFrame,
    SpeakerRole,
)
from ailang_ir.parser.rule_parser import RuleBasedParser
from ailang_ir.encoder.codebook import SymbolicEncoder
from ailang_ir.encoder.concept_table import ConceptTable
from ailang_ir.decoder.reconstructor import Reconstructor
from ailang_ir.memory.store import MemoryStore
from ailang_ir.llm.codec import LLMCodec
from ailang_ir.llm.format_spec import get_format_spec
from ailang_ir.llm.validator import validate_code


@dataclass
class ProcessResult:
    """Result of processing a single text through the pipeline."""
    frame: SemanticFrame
    compact_code: str
    memory: EventMemory | None = None
    concept_table: ConceptTable | None = None

    def reconstruct(self, style: str = "declarative") -> str:
        """Reconstruct natural language from the stored frame."""
        r = Reconstructor()
        if self.concept_table is not None and "|" not in self.compact_code:
            return r.reconstruct_from_code(self.compact_code, style, self.concept_table)
        return r.reconstruct(self.frame, style)

    @property
    def summary(self) -> str:
        return self.frame.summary()

    def __repr__(self) -> str:
        return f"ProcessResult(code={self.compact_code!r})"


@dataclass
class Pipeline:
    """
    Unified semantic processing pipeline.

    Usage:
        pipe = Pipeline()
        result = pipe.process("I think graph memory is better than linear text.")
        print(result.compact_code)
        print(result.reconstruct())

    The pipeline maintains a MemoryStore that accumulates across calls.
    """
    parser: RuleBasedParser = field(default_factory=RuleBasedParser)
    encoder: SymbolicEncoder = field(default_factory=SymbolicEncoder)
    decoder: Reconstructor = field(default_factory=Reconstructor)
    memory: MemoryStore = field(default_factory=MemoryStore)
    concept_table: ConceptTable = field(default_factory=ConceptTable)
    encoding_version: int = 3
    auto_store: bool = True  # automatically store frames in memory

    def process(
        self,
        text: str,
        speaker: SpeakerRole = SpeakerRole.USER,
        tags: list[str] | None = None,
    ) -> ProcessResult:
        """
        Process a single text through the full pipeline.

        Steps:
        1. Parse → SemanticFrame
        2. Encode → compact code
        3. (optional) Store in memory
        4. Return ProcessResult
        """
        frame = self.parser.parse(text, speaker)
        code = self._encode_frame(frame)

        mem = None
        if self.auto_store:
            mem = self.memory.store(frame, tags=tags)

        return ProcessResult(
            frame=frame, compact_code=code, memory=mem,
            concept_table=self.concept_table if self.encoding_version >= 2 else None,
        )

    def process_batch(
        self,
        texts: list[str],
        speaker: SpeakerRole = SpeakerRole.USER,
        tags: list[str] | None = None,
    ) -> list[ProcessResult]:
        """Process multiple texts."""
        return [self.process(t, speaker, tags) for t in texts]

    def process_multi(
        self,
        text: str,
        speaker: SpeakerRole = SpeakerRole.USER,
        tags: list[str] | None = None,
    ) -> list[ProcessResult]:
        """
        Process text that may contain multiple sentences.
        Splits on sentence boundaries and processes each.
        """
        frames = self.parser.parse_multi(text, speaker)
        results = []
        for frame in frames:
            code = self._encode_frame(frame)
            mem = None
            if self.auto_store:
                mem = self.memory.store(frame, tags=tags)
            results.append(ProcessResult(
                frame=frame, compact_code=code, memory=mem,
                concept_table=self.concept_table if self.encoding_version >= 2 else None,
            ))
        return results

    def _encode_frame(self, frame: SemanticFrame) -> str:
        """Encode a frame using the configured encoding version."""
        if self.encoding_version >= 3:
            return self.encoder.encode_v3(frame, self.concept_table)
        elif self.encoding_version == 2:
            return self.encoder.encode_v2(frame, self.concept_table)
        else:
            return self.encoder.encode(frame)

    def decode(self, code: str, style: str = "declarative") -> str:
        """Decode a compact code back to natural language."""
        return self.decoder.reconstruct_from_code(code, style, self.concept_table)

    def search(self, entity_key: str) -> list[EventMemory]:
        """Search memory by entity key."""
        return self.memory.query_by_entity(entity_key)

    def recent(self, n: int = 10) -> list[EventMemory]:
        """Get recent memories."""
        return self.memory.query_recent(n)

    def contradictions_for(self, text: str) -> list[EventMemory]:
        """Check if a new statement contradicts stored memories."""
        frame = self.parser.parse(text)
        return self.memory.find_contradictions(frame)

    def dump_memory(self) -> list[dict]:
        """Dump all active memories for inspection."""
        return self.memory.dump()

    @property
    def memory_size(self) -> int:
        return self.memory.size

    def stats(self) -> dict:
        """Pipeline statistics."""
        return {
            "total_memories": self.memory.size,
            "active_memories": self.memory.active_count,
            "vocabulary_acts": len(self.parser.vocab.act_map),
            "vocabulary_modes": len(self.parser.vocab.mode_map),
        }

    # -------------------------------------------------------------------
    # LLM format interface
    # -------------------------------------------------------------------

    def ingest_code(self, code: str) -> ProcessResult:
        """
        Ingest an LLM-produced code: validate, convert to frame, store.

        Raises ValueError if the code is structurally invalid.
        """
        codec = LLMCodec()
        frame = codec.decode(code)

        # Encode to internal v3 for storage
        internal_code = self._encode_frame(frame)

        mem = None
        if self.auto_store:
            mem = self.memory.store(frame)

        return ProcessResult(
            frame=frame, compact_code=internal_code, memory=mem,
            concept_table=self.concept_table if self.encoding_version >= 2 else None,
        )

    def ingest_codes(self, codes: list[str]) -> list[ProcessResult]:
        """Ingest multiple LLM-produced codes."""
        return [self.ingest_code(c) for c in codes]

    def export_context(self, n: int = 10) -> str:
        """
        Export the most recent n memories as LLM-readable format.

        Returns newline-separated LLM format codes.
        """
        codec = LLMCodec()
        memories = self.memory.query_recent(n)
        frames = [m.frame for m in memories]
        return codec.encode_batch(frames)

    def get_format_spec(self) -> str:
        """Return the LLM format specification for system prompt injection."""
        return get_format_spec()
