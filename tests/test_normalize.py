"""Tests for normalization vocabulary."""

from ailang_ir.normalize.vocabulary import NormalizationVocabulary
from ailang_ir.models.domain import SemanticAct, SemanticMode, Sentiment, TimeReference


class TestActMatching:
    def setup_method(self):
        self.vocab = NormalizationVocabulary()

    def test_believe(self):
        assert self.vocab.match_act("I think this is right") == SemanticAct.BELIEVE

    def test_prefer(self):
        assert self.vocab.match_act("I prefer graph memory") == SemanticAct.PREFER

    def test_suggest(self):
        assert self.vocab.match_act("I suggest we try this") == SemanticAct.SUGGEST

    def test_need(self):
        assert self.vocab.match_act("We need better storage") == SemanticAct.NEED

    def test_create(self):
        assert self.vocab.match_act("Let's build a new parser") == SemanticAct.CREATE

    def test_query(self):
        assert self.vocab.match_act("What is the best approach?") == SemanticAct.QUERY

    def test_unknown_fallback(self):
        assert self.vocab.match_act("xyzzy foobar") == SemanticAct.UNKNOWN


class TestModeMatching:
    def setup_method(self):
        self.vocab = NormalizationVocabulary()

    def test_question_mark(self):
        assert self.vocab.match_mode("Is this correct?") == SemanticMode.QUESTION

    def test_opinion(self):
        assert self.vocab.match_mode("I think this is good") == SemanticMode.OPINION

    def test_hypothesis(self):
        assert self.vocab.match_mode("Maybe we should try another way") == SemanticMode.HYPOTHESIS

    def test_assertion_default(self):
        assert self.vocab.match_mode("The sky is blue") == SemanticMode.ASSERTION


class TestSentimentMatching:
    def setup_method(self):
        self.vocab = NormalizationVocabulary()

    def test_positive(self):
        assert self.vocab.match_sentiment("This is great work") == Sentiment.POSITIVE

    def test_negative(self):
        assert self.vocab.match_sentiment("This is terrible") == Sentiment.NEGATIVE

    def test_neutral(self):
        assert self.vocab.match_sentiment("The function returns a value") == Sentiment.NEUTRAL

    def test_mixed(self):
        assert self.vocab.match_sentiment("It's good but unfortunately slow") == Sentiment.MIXED


class TestTimeMatching:
    def setup_method(self):
        self.vocab = NormalizationVocabulary()

    def test_future(self):
        assert self.vocab.match_time("We will implement this later") == TimeReference.FUTURE

    def test_past(self):
        assert self.vocab.match_time("We previously used linear storage") == TimeReference.PAST

    def test_present_default(self):
        assert self.vocab.match_time("xyzzy foobar") == TimeReference.PRESENT


class TestCertaintyEstimation:
    def setup_method(self):
        self.vocab = NormalizationVocabulary()

    def test_high_certainty(self):
        assert self.vocab.estimate_certainty("This is definitely correct") > 0.9

    def test_low_certainty(self):
        assert self.vocab.estimate_certainty("I'm not sure about this") < 0.4

    def test_moderate_default(self):
        assert 0.6 <= self.vocab.estimate_certainty("The system processes input") <= 0.8


class TestObjectKeyNormalization:
    def setup_method(self):
        self.vocab = NormalizationVocabulary()

    def test_one_to_one(self):
        assert self.vocab.normalize_object_key("one-to-one sentence mapping") == "1to1_sentence_mapping"

    def test_filler_removal(self):
        key = self.vocab.normalize_object_key("the best approach for this")
        assert "the" not in key.split("_")
        assert "for" in key.split("_") or "best" in key.split("_")

    def test_punctuation_stripped(self):
        key = self.vocab.normalize_object_key("semantic compression?")
        assert "?" not in key
