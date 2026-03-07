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
