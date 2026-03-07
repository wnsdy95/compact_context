"""Tests for rule-based semantic parser."""

from ailang_ir.parser import RuleBasedParser
from ailang_ir.models.domain import SemanticAct, SemanticMode, SpeakerRole, TimeReference


class TestRuleBasedParser:
    def setup_method(self):
        self.parser = RuleBasedParser()

    def test_empty_input(self):
        frame = self.parser.parse("")
        assert frame.speaker == SpeakerRole.USER
        assert frame.source_text == ""

    def test_opinion_with_believe(self):
        frame = self.parser.parse("I think this approach is better.")
        assert frame.mode == SemanticMode.OPINION
        assert frame.act == SemanticAct.BELIEVE

    def test_question_detection(self):
        frame = self.parser.parse("What is the best approach?")
        assert frame.mode == SemanticMode.QUESTION
        assert frame.act == SemanticAct.QUERY

    def test_future_time(self):
        frame = self.parser.parse("We will implement this later.")
        assert frame.time == TimeReference.FUTURE

    def test_object_extraction(self):
        frame = self.parser.parse("I think semantic frames are important.")
        assert frame.object is not None
        assert "semantic" in frame.object.canonical

    def test_comparison_extraction(self):
        frame = self.parser.parse("Graph memory seems more appropriate than linear text storage.")
        assert frame.object is not None
        assert frame.target is not None
        assert "graph" in frame.object.canonical
        assert "linear" in frame.target.canonical

    def test_speaker_passthrough(self):
        frame = self.parser.parse("Test", speaker=SpeakerRole.SYSTEM)
        assert frame.speaker == SpeakerRole.SYSTEM

    def test_source_text_preserved(self):
        text = "The system should normalize input."
        frame = self.parser.parse(text)
        assert frame.source_text == text

    def test_certainty_from_hedge(self):
        frame = self.parser.parse("Maybe we should try another approach.")
        assert frame.certainty.value < 0.5

    def test_certainty_from_confidence(self):
        frame = self.parser.parse("This is definitely the right way.")
        assert frame.certainty.value > 0.9

    def test_parse_multi(self):
        text = "I think this is good. We should proceed. What about testing?"
        frames = self.parser.parse_multi(text)
        assert len(frames) == 3
        assert frames[2].mode == SemanticMode.QUESTION
