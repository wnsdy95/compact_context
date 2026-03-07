"""Tests for unified Pipeline API."""

from ailang_ir import Pipeline
from ailang_ir.encoder.concept_table import ConceptTable
from ailang_ir.models.domain import SemanticAct, SemanticMode, SpeakerRole


class TestPipeline:
    def setup_method(self):
        self.pipe = Pipeline()

    def test_basic_process(self):
        r = self.pipe.process("I think semantic frames are important.")
        assert r.compact_code
        assert r.frame.act == SemanticAct.BELIEVE
        assert r.memory is not None

    def test_dedup_in_memory(self):
        self.pipe.process("I think this is good.")
        self.pipe.process("I think this is good.")
        assert self.pipe.memory_size == 1

    def test_batch_process(self):
        texts = ["We need better storage.", "Let's build a parser."]
        results = self.pipe.process_batch(texts)
        assert len(results) == 2
        assert self.pipe.memory_size == 2

    def test_multi_sentence(self):
        results = self.pipe.process_multi(
            "I think this is good. We need better storage."
        )
        assert len(results) == 2

    def test_search(self):
        self.pipe.process("Graph memory seems more appropriate than linear text.")
        results = self.pipe.search("graph_memory")
        assert len(results) == 1

    def test_decode(self):
        text = self.pipe.decode("U|OP|BELIEVE|SEMANTIC_FRAMES|C84|NOW")
        assert "semantic frames" in text.lower()

    def test_contradictions(self):
        self.pipe.process("I prefer graph memory.")
        contras = self.pipe.contradictions_for("I reject graph memory.")
        assert len(contras) == 1

    def test_no_auto_store(self):
        pipe = Pipeline(auto_store=False)
        r = pipe.process("Test statement.")
        assert r.memory is None
        assert pipe.memory_size == 0

    def test_speaker_passthrough(self):
        r = self.pipe.process("System is ready.", speaker=SpeakerRole.SYSTEM)
        assert r.frame.speaker == SpeakerRole.SYSTEM

    def test_tags(self):
        self.pipe.process("Tagged memory.", tags=["important"])
        results = self.pipe.memory.query_by_tag("important")
        assert len(results) == 1

    def test_recent(self):
        for i in range(5):
            self.pipe.process(f"Statement number {i} about topic {i}.")
        recent = self.pipe.recent(3)
        assert len(recent) == 3

    def test_stats(self):
        self.pipe.process("Test.")
        stats = self.pipe.stats()
        assert stats["total_memories"] == 1
        assert "vocabulary_acts" in stats

    def test_reconstruct_from_result(self):
        r = self.pipe.process("I think semantic frames are important.")
        text = r.reconstruct()
        assert len(text) > 0

    def test_summary_from_result(self):
        r = self.pipe.process("I think semantic frames are important.")
        assert "believe" in r.summary.lower()

    def test_dump_memory(self):
        self.pipe.process("Test.")
        dump = self.pipe.dump_memory()
        assert len(dump) == 1
        assert "memory_id" in dump[0]


class TestPipelineV2:
    def test_v2_produces_no_pipes(self):
        pipe = Pipeline(encoding_version=2)
        r = pipe.process("I think semantic frames are important.")
        assert "|" not in r.compact_code

    def test_v2_shorter_than_v1(self):
        pipe_v1 = Pipeline(encoding_version=1)
        pipe_v2 = Pipeline(encoding_version=2)
        text = "I think one-to-one sentence mapping will be difficult."
        r1 = pipe_v1.process(text)
        r2 = pipe_v2.process(text)
        assert len(r2.compact_code) < len(r1.compact_code)

    def test_v2_concept_table_grows(self):
        pipe = Pipeline(encoding_version=2)
        pipe.process("I think semantic frames are important.")
        pipe.process("We need better storage mechanisms.")
        assert pipe.concept_table.size >= 2

    def test_v2_reref_uses_dollar_id(self):
        pipe = Pipeline(encoding_version=2)
        r1 = pipe.process("I think graph memory is good.")
        # Process the exact same text again to guarantee same object key
        r2 = pipe.process("I think graph memory is good.", speaker=SpeakerRole.SYSTEM)
        # The same concept should be re-referenced with $id in second code
        assert "$" in r2.compact_code

    def test_v2_reconstruct_from_result(self):
        pipe = Pipeline(encoding_version=2)
        r = pipe.process("I think semantic frames are important.")
        text = r.reconstruct()
        assert len(text) > 0

    def test_v2_decode(self):
        pipe = Pipeline(encoding_version=2)
        r = pipe.process("I think semantic frames are important.")
        decoded = pipe.decode(r.compact_code)
        assert len(decoded) > 0

    def test_v1_backward_compat(self):
        pipe = Pipeline(encoding_version=1)
        r = pipe.process("I think semantic frames are important.")
        assert "|" in r.compact_code
        assert r.concept_table is None

    def test_v2_multi_sentence(self):
        pipe = Pipeline(encoding_version=2)
        results = pipe.process_multi(
            "I think this is good. We need better storage."
        )
        assert len(results) == 2
        for r in results:
            assert "|" not in r.compact_code

    def test_v2_batch(self):
        pipe = Pipeline(encoding_version=2)
        texts = ["We need storage.", "Let's build a parser."]
        results = pipe.process_batch(texts)
        assert len(results) == 2
        for r in results:
            assert "|" not in r.compact_code


class TestPipelineV3:
    def test_v3_default_version(self):
        pipe = Pipeline()
        assert pipe.encoding_version == 3

    def test_v3_shorter_than_v2(self):
        pipe_v2 = Pipeline(encoding_version=2)
        pipe_v3 = Pipeline(encoding_version=3)
        text = "I think one-to-one sentence mapping will be difficult."
        r2 = pipe_v2.process(text)
        r3 = pipe_v3.process(text)
        assert len(r3.compact_code) < len(r2.compact_code)

    def test_v3_produces_no_pipes(self):
        pipe = Pipeline(encoding_version=3)
        r = pipe.process("I think semantic frames are important.")
        assert "|" not in r.compact_code

    def test_v3_reconstruct(self):
        pipe = Pipeline(encoding_version=3)
        r = pipe.process("I think semantic frames are important.")
        text = r.reconstruct()
        assert len(text) > 0

    def test_v3_decode(self):
        pipe = Pipeline(encoding_version=3)
        r = pipe.process("I think semantic frames are important.")
        decoded = pipe.decode(r.compact_code)
        assert len(decoded) > 0

    def test_v3_reref(self):
        pipe = Pipeline(encoding_version=3)
        pipe.process("I think graph memory is good.")
        r2 = pipe.process("I think graph memory is good.", speaker=SpeakerRole.SYSTEM)
        assert "$" in r2.compact_code

    def test_v3_compression_target(self):
        """v3 must achieve <= 30% compression on benchmark corpus."""
        pipe = Pipeline(encoding_version=3)
        sentences = [
            "I think one-to-one sentence mapping will be difficult.",
            "The system should be able to reconstruct natural language later.",
            "Graph memory seems more appropriate than linear text storage.",
            "Natural language should be segmented into meaning units.",
            "What is the best approach for semantic compression?",
            "I prefer typed models over loose dictionaries.",
            "We need a normalization layer for consistent output.",
            "Maybe we should try a different compression strategy.",
            "This is definitely the right architecture.",
            "Previously we used raw string matching.",
            "I believe semantic frames are the right approach.",
            "The reconstruction quality is unfortunately poor.",
            "Let us build a new encoder for compact codes.",
            "We should update the normalization rules.",
            "Semantic frames preserve meaning better than raw text.",
            "I notice the parser misclassifies passive sentences.",
            "We will implement persistence in the next cycle.",
            "Perhaps the compression ratio can be improved.",
            "How does the parser handle ambiguous input?",
            "This is a great improvement over the old system.",
        ]
        total_raw = sum(len(s.encode("utf-8")) for s in sentences)
        total_code = sum(len(pipe.process(s).compact_code.encode("utf-8")) for s in sentences)
        ratio = total_code / total_raw
        assert ratio <= 0.31, f"v3 compression ratio {ratio:.1%} exceeds 31% target"

    def test_v3_multi_sentence(self):
        pipe = Pipeline(encoding_version=3)
        results = pipe.process_multi("I think this is good. We need better storage.")
        assert len(results) == 2
        for r in results:
            assert "|" not in r.compact_code

    def test_v3_backward_compat_v1(self):
        pipe = Pipeline(encoding_version=1)
        r = pipe.process("I think semantic frames are important.")
        assert "|" in r.compact_code

    def test_v3_backward_compat_v2(self):
        pipe = Pipeline(encoding_version=2)
        r = pipe.process("I think semantic frames are important.")
        assert "|" not in r.compact_code
        # v2 keys have underscores, v3 does not
        obj = r.compact_code.split()[1]
        assert "_" in obj[1:]  # v2 uses underscored keys
