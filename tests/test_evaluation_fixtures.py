"""
Curated evaluation fixtures for regression testing.

Each fixture defines:
- input text
- expected semantic fields (act, mode, object key patterns, time, etc.)
- expected compact code patterns

These serve as the canonical correctness baseline.
Any parser/encoder change must pass these fixtures.
"""

import pytest
from ailang_ir.parser import RuleBasedParser
from ailang_ir.encoder import SymbolicEncoder
from ailang_ir.decoder import Reconstructor
from ailang_ir.models.domain import SemanticAct, SemanticMode, TimeReference, Sentiment


@pytest.fixture
def pipeline():
    return RuleBasedParser(), SymbolicEncoder(), Reconstructor()


# ---------------------------------------------------------------------------
# Fixture data: (text, expected_fields)
# ---------------------------------------------------------------------------

FIXTURES = [
    # --- Opinions / Beliefs ---
    {
        "text": "I think one-to-one sentence mapping will be difficult.",
        "act": SemanticAct.BELIEVE,
        "mode": SemanticMode.OPINION,
        "time": TimeReference.FUTURE,
        "obj_contains": "1to1",
        "code_starts": "U|OP|BELIEVE|",
    },
    {
        "text": "I believe semantic frames are the right approach.",
        "act": SemanticAct.BELIEVE,
        "mode": SemanticMode.OPINION,
        "obj_contains": "semantic",
    },
    # --- Hypotheses ---
    {
        "text": "Maybe we should use graph storage instead.",
        "act": SemanticAct.SUGGEST,
        "mode": SemanticMode.HYPOTHESIS,
        "obj_contains": "graph",
        "certainty_below": 0.5,
    },
    {
        "text": "Perhaps the compression ratio can be improved.",
        "mode": SemanticMode.HYPOTHESIS,
        "obj_contains": "compression",
        "certainty_below": 0.5,
    },
    # --- Requests / Needs ---
    {
        "text": "We need a normalization layer for consistent output.",
        "act": SemanticAct.NEED,
        "obj_contains": "normalization",
    },
    {
        "text": "The system should have better error handling.",
        "time": TimeReference.PRESENT,
    },
    # --- Questions ---
    {
        "text": "What is the best approach for semantic compression?",
        "act": SemanticAct.QUERY,
        "mode": SemanticMode.QUESTION,
        "obj_contains": "semantic_compression",
        "code_contains": "|Q|QUERY|",
    },
    {
        "text": "How does the parser handle ambiguous input?",
        "mode": SemanticMode.QUESTION,
        "obj_contains": "parser",
    },
    # --- Comparisons ---
    {
        "text": "Graph memory seems more appropriate than linear text storage.",
        "obj_contains": "graph",
        "has_target": True,
        "target_contains": "linear",
        "code_contains": "OVER:",
    },
    {
        "text": "I prefer typed models over loose dictionaries.",
        "act": SemanticAct.PREFER,
        "obj_contains": "typed",
        "has_target": True,
        "target_contains": "loose",
    },
    # --- Commands / Assertions ---
    {
        "text": "Natural language should be segmented into meaning units.",
        "obj_contains": "meaning_units",
    },
    {
        "text": "This is definitely the right architecture.",
        "certainty_above": 0.9,
    },
    # --- Observations ---
    {
        "text": "I notice the parser misclassifies passive sentences.",
        "mode": SemanticMode.OBSERVATION,
        "obj_contains": "parser",
    },
    # --- Creation / Modification ---
    {
        "text": "Let's build a new encoder for compact codes.",
        "act": SemanticAct.CREATE,
        "obj_contains": "encoder",
    },
    {
        "text": "We should update the normalization rules.",
        "act": SemanticAct.MODIFY,
    },
    # --- Sentiment ---
    {
        "text": "This is a great improvement over the old system.",
        "sentiment": Sentiment.POSITIVE,
    },
    {
        "text": "The reconstruction quality is unfortunately poor.",
        "sentiment": Sentiment.NEGATIVE,
    },
    # --- Temporal ---
    {
        "text": "Previously we used raw string matching.",
        "time": TimeReference.PAST,
    },
    {
        "text": "We will implement persistence in the next cycle.",
        "time": TimeReference.FUTURE,
        "obj_contains": "persistence",
    },
    # --- Round-trip integrity ---
    {
        "text": "Semantic frames preserve meaning better than raw text.",
        "has_target": True,
        "round_trip_preserves": "semantic",
    },
]


class TestEvaluationFixtures:
    """Run all curated fixtures through the full pipeline."""

    def setup_method(self):
        self.parser = RuleBasedParser()
        self.encoder = SymbolicEncoder()
        self.decoder = Reconstructor()

    @pytest.mark.parametrize(
        "fixture",
        FIXTURES,
        ids=[f["text"][:50] for f in FIXTURES],
    )
    def test_fixture(self, fixture):
        text = fixture["text"]
        frame = self.parser.parse(text)
        code = self.encoder.encode(frame)

        # Act check
        if "act" in fixture:
            assert frame.act == fixture["act"], (
                f"Expected act {fixture['act']}, got {frame.act} for: {text}"
            )

        # Mode check
        if "mode" in fixture:
            assert frame.mode == fixture["mode"], (
                f"Expected mode {fixture['mode']}, got {frame.mode} for: {text}"
            )

        # Time check
        if "time" in fixture:
            assert frame.time == fixture["time"], (
                f"Expected time {fixture['time']}, got {frame.time} for: {text}"
            )

        # Sentiment check
        if "sentiment" in fixture:
            assert frame.sentiment == fixture["sentiment"], (
                f"Expected sentiment {fixture['sentiment']}, got {frame.sentiment} for: {text}"
            )

        # Object key contains
        if "obj_contains" in fixture:
            assert frame.object is not None, f"Expected object for: {text}"
            assert fixture["obj_contains"] in frame.object.canonical, (
                f"Expected '{fixture['obj_contains']}' in object key "
                f"'{frame.object.canonical}' for: {text}"
            )

        # Target existence
        if fixture.get("has_target"):
            assert frame.target is not None, f"Expected target for: {text}"

        # Target key contains
        if "target_contains" in fixture:
            assert frame.target is not None, f"Expected target for: {text}"
            assert fixture["target_contains"] in frame.target.canonical, (
                f"Expected '{fixture['target_contains']}' in target key "
                f"'{frame.target.canonical}' for: {text}"
            )

        # Certainty bounds
        if "certainty_above" in fixture:
            assert frame.certainty.value > fixture["certainty_above"], (
                f"Expected certainty > {fixture['certainty_above']}, "
                f"got {frame.certainty.value} for: {text}"
            )
        if "certainty_below" in fixture:
            assert frame.certainty.value < fixture["certainty_below"], (
                f"Expected certainty < {fixture['certainty_below']}, "
                f"got {frame.certainty.value} for: {text}"
            )

        # Code pattern checks
        if "code_starts" in fixture:
            assert code.startswith(fixture["code_starts"]), (
                f"Expected code starting with '{fixture['code_starts']}', got '{code}'"
            )
        if "code_contains" in fixture:
            assert fixture["code_contains"] in code, (
                f"Expected '{fixture['code_contains']}' in code '{code}'"
            )

        # Round-trip meaning preservation
        if "round_trip_preserves" in fixture:
            reconstructed = self.decoder.reconstruct_from_code(code)
            keyword = fixture["round_trip_preserves"]
            assert keyword in reconstructed.lower() or keyword.upper() in code, (
                f"Round-trip lost '{keyword}'. Code: {code}, Reconstructed: {reconstructed}"
            )
