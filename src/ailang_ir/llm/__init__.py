"""LLM-native interface layer for AILang-IR."""

from ailang_ir.llm.format_spec import FORMAT_SPEC, get_format_spec
from ailang_ir.llm.validator import ValidationResult, validate_code
from ailang_ir.llm.codec import LLMCodec
from ailang_ir.llm.llm_parser import LLMParser

__all__ = [
    "FORMAT_SPEC",
    "get_format_spec",
    "ValidationResult",
    "validate_code",
    "LLMCodec",
    "LLMParser",
]
