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


class TestAgentTextKeyQuality:
    """Test object key quality on realistic AI agent conversation text."""

    def setup_method(self):
        self.parser = RuleBasedParser()

    def _key(self, text, speaker=SpeakerRole.AGENT):
        f = self.parser.parse(text, speaker)
        return f.object.canonical if f.object else ""

    def test_transition_word_stripped(self):
        key = self._key("However, it does not handle stored procedures.")
        assert "however" not in key
        assert "stored" in key

    def test_transition_additionally(self):
        key = self._key("Additionally, we need to update the schema.")
        assert "additionally" not in key
        assert "schema" in key

    def test_transition_therefore(self):
        key = self._key("Therefore, the migration should be planned carefully.")
        assert "therefore" not in key
        assert "migration" in key

    def test_it_says_stripped(self):
        key = self._key("It says there are 12 unused imports.", SpeakerRole.USER)
        assert "says" not in key
        assert "12" in key or "unused" in key

    def test_it_does_not_stripped(self):
        key = self._key("It does not support stored procedures.")
        assert "stored" in key or "procedure" in key

    def test_can_you_request(self):
        key = self._key("Can you fix it and add a test?", SpeakerRole.USER)
        assert "can" not in key
        assert "you" not in key

    def test_i_have_prefix(self):
        key = self._key("I have deployed the search API fix to staging.")
        assert "search" in key or "api" in key or "staging" in key

    def test_i_cleaned_up(self):
        key = self._key("I cleaned up all 12 unused imports.")
        assert "cleaned" not in key
        assert "unused" in key or "imports" in key

    def test_colon_label(self):
        key = self._key("Key concerns: JSON columns need conversion.")
        assert "json" in key

    def test_negation_stripped(self):
        key = self._key("It does not handle stored procedure conversion.")
        assert "handle" not in key
        assert "stored" in key

    def test_fillers_removed(self):
        """Pronouns, quantifiers, demonstratives should be stripped from keys."""
        key = self._key("We have about 30 of them.", SpeakerRole.USER)
        assert "them" not in key
        assert "about" not in key
        assert "there" not in key

    def test_agent_action_verbs_stripped(self):
        key = self._key("I profiled the search API and found two bottlenecks.")
        assert "search" in key or "api" in key or "bottleneck" in key
