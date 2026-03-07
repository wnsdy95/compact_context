"""Tests for the LLM-assisted parser (mock-based, no API key required)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from ailang_ir.models.domain import (
    SemanticAct,
    SemanticFrame,
    SemanticMode,
    SpeakerRole,
)
from ailang_ir.llm.llm_parser import LLMParser, _build_system_prompt


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockResponse:
    content: list


def make_mock_response(text: str) -> MockResponse:
    return MockResponse(content=[MockTextBlock(text=text)])


def make_parser_with_mock(response_text: str) -> LLMParser:
    """Create an LLMParser with a mocked client."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_mock_response(response_text)
    parser = LLMParser()
    parser._client = mock_client
    return parser


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_format_spec(self):
        prompt = _build_system_prompt()
        assert "AILang-IR Compressed Format" in prompt
        assert "Act" in prompt

    def test_contains_rules(self):
        prompt = _build_system_prompt()
        assert "ONLY the code" in prompt
        assert "CORE concept" in prompt


# ---------------------------------------------------------------------------
# Single parse
# ---------------------------------------------------------------------------

class TestParseSingle:
    def test_valid_llm_response(self):
        parser = make_parser_with_mock("Uoblbn graph_memory")
        frame = parser.parse("I think graph memory is the best approach.")
        assert frame.speaker == SpeakerRole.USER
        assert frame.mode == SemanticMode.OPINION
        assert frame.act == SemanticAct.BELIEVE
        assert frame.object is not None
        assert frame.object.canonical == "graph_memory"

    def test_valid_with_target(self):
        parser = make_parser_with_mock("Uaprbn rest_api >graphql")
        frame = parser.parse("I prefer REST over GraphQL.")
        assert frame.act == SemanticAct.PREFER
        assert frame.target is not None
        assert frame.target.canonical == "graphql"

    def test_agent_speaker(self):
        parser = make_parser_with_mock("Aasgbn redis_cache")
        frame = parser.parse("I suggest using Redis for caching.", SpeakerRole.AGENT)
        assert frame.speaker == SpeakerRole.AGENT
        assert frame.act == SemanticAct.SUGGEST

    def test_various_acts(self):
        test_cases = [
            ("Uandbn persistence", SemanticAct.NEED),
            ("Uadcbn use_postgres", SemanticAct.DECIDE),
            ("Uadgbn microservices", SemanticAct.DISAGREE),
            ("Uaagbn incremental", SemanticAct.AGREE),
            ("Uqqubn deployment", SemanticAct.QUERY),
            ("Uawnbn data_loss", SemanticAct.WARN),
        ]
        for code, expected_act in test_cases:
            parser = make_parser_with_mock(code)
            frame = parser.parse("test text")
            assert frame.act == expected_act, f"Expected {expected_act} for code {code}"

    def test_certainty_levels(self):
        # Low certainty (maybe, perhaps)
        parser = make_parser_with_mock("Uhbl3n approach")
        frame = parser.parse("Maybe we should try a different approach.")
        assert frame.certainty.value < 0.4

        # High certainty (definitely)
        parser = make_parser_with_mock("Uabldn architecture")
        frame = parser.parse("This is definitely the right architecture.")
        assert frame.certainty.value > 0.8


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_on_invalid_code(self):
        """Invalid LLM output → fallback to rule-based parser."""
        parser = make_parser_with_mock("INVALID_GARBAGE")
        frame = parser.parse("I think graph memory is good.")
        # Should still produce a frame (from fallback)
        assert isinstance(frame, SemanticFrame)
        assert frame.speaker == SpeakerRole.USER

    def test_fallback_on_empty_response(self):
        parser = make_parser_with_mock("")
        frame = parser.parse("I prefer typed models.")
        assert isinstance(frame, SemanticFrame)

    def test_fallback_on_7char_header(self):
        """Known LLM error: 7-char header from act/certainty boundary confusion."""
        parser = make_parser_with_mock("Uoblbbn graph_mem")  # 7 chars
        frame = parser.parse("I believe in graph memory.")
        assert isinstance(frame, SemanticFrame)

    def test_fallback_on_api_exception(self):
        """API error → fallback gracefully."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API timeout")
        parser = LLMParser()
        parser._client = mock_client
        frame = parser.parse("Test input.")
        assert isinstance(frame, SemanticFrame)

    def test_fallback_on_no_client(self):
        """No API key → always uses fallback."""
        parser = LLMParser()  # No client, no API key
        frame = parser.parse("I think this works.")
        assert isinstance(frame, SemanticFrame)
        assert frame.speaker == SpeakerRole.USER

    def test_fallback_on_uppercase_hex(self):
        """Known LLM error: uppercase hex certainty."""
        parser = make_parser_with_mock("UoblBn graph_mem")  # 'B' not 'b'
        frame = parser.parse("I believe in graph memory.")
        assert isinstance(frame, SemanticFrame)


# ---------------------------------------------------------------------------
# Batch parse
# ---------------------------------------------------------------------------

class TestParseBatch:
    def test_batch_valid(self):
        response = "Uoblbn graph_memory\nUaprbn typed_models >dicts"
        parser = make_parser_with_mock(response)
        frames = parser.parse_batch([
            "I think graph memory is good.",
            "I prefer typed models over dictionaries.",
        ])
        assert len(frames) == 2
        assert frames[0].act == SemanticAct.BELIEVE
        assert frames[1].act == SemanticAct.PREFER
        assert frames[1].target is not None

    def test_batch_with_speaker_list(self):
        response = "Uoblbn approach\nAaagbn approach"
        parser = make_parser_with_mock(response)
        frames = parser.parse_batch(
            ["I think this is good.", "I agree with that."],
            speakers=[SpeakerRole.USER, SpeakerRole.AGENT],
        )
        assert len(frames) == 2
        assert frames[0].speaker == SpeakerRole.USER
        assert frames[1].speaker == SpeakerRole.AGENT

    def test_batch_partial_valid(self):
        """Some codes valid, some not → mixed LLM + fallback."""
        response = "Uoblbn graph_memory\nINVALID\nUaprbn rest >graphql"
        parser = make_parser_with_mock(response)
        frames = parser.parse_batch([
            "Graph memory is good.",
            "Testing is important.",
            "I prefer REST over GraphQL.",
        ])
        assert len(frames) == 3
        # First and third from LLM, second from fallback
        assert frames[0].object.canonical == "graph_memory"
        assert isinstance(frames[1], SemanticFrame)  # fallback
        assert frames[2].target is not None

    def test_batch_fewer_responses_than_inputs(self):
        """LLM returns fewer codes than inputs → fallback for missing."""
        response = "Uoblbn graph_memory"  # Only 1 line for 3 inputs
        parser = make_parser_with_mock(response)
        frames = parser.parse_batch([
            "Graph memory is good.",
            "Testing works.",
            "REST is simple.",
        ])
        assert len(frames) == 3

    def test_batch_api_error(self):
        """API failure during batch → all fallback."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")
        parser = LLMParser()
        parser._client = mock_client
        frames = parser.parse_batch(["Test 1.", "Test 2."])
        assert len(frames) == 2
        assert all(isinstance(f, SemanticFrame) for f in frames)

    def test_batch_no_client(self):
        """No client → all fallback."""
        parser = LLMParser()
        frames = parser.parse_batch(["Test 1.", "Test 2."])
        assert len(frames) == 2


# ---------------------------------------------------------------------------
# API call verification
# ---------------------------------------------------------------------------

class TestAPICall:
    def test_single_call_format(self):
        parser = make_parser_with_mock("Uoblbn test")
        parser.parse("Hello world.", SpeakerRole.USER)

        call_args = parser._client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-haiku-4-5-20251001"
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert "[U] Hello world." in messages[0]["content"]

    def test_batch_call_format(self):
        parser = make_parser_with_mock("Uoblbn a\nAaagbn b")
        parser.parse_batch(
            ["First.", "Second."],
            speakers=[SpeakerRole.USER, SpeakerRole.AGENT],
        )

        call_args = parser._client.messages.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        assert "[U] First." in content
        assert "[A] Second." in content

    def test_system_prompt_sent(self):
        parser = make_parser_with_mock("Uoblbn test")
        parser.parse("Test.")

        call_args = parser._client.messages.create.call_args
        system = call_args.kwargs["system"]
        assert "AILang-IR" in system
        assert "semantic encoder" in system

    def test_custom_model(self):
        parser = make_parser_with_mock("Uoblbn test")
        parser.model = "claude-sonnet-4-6"
        parser.parse("Test.")

        call_args = parser._client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_import_error_message(self):
        """If anthropic not installed, error message tells user how to install."""
        with patch.dict("sys.modules", {"anthropic": None}):
            from ailang_ir.llm.llm_parser import _import_anthropic
            with pytest.raises(ImportError, match="pip install anthropic"):
                _import_anthropic()
