"""
Validator for LLM-produced AILang-IR codes.

Validates structural correctness of LLM format codes:
  HEADER object_key [>target_key]

Reuses codebook reverse mappings to avoid duplication.
Collects all errors (does not stop at first).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ailang_ir.encoder.codebook import (
    SPEAKER_DECODE,
    MODE_DECODE_V2,
    ACT_DECODE_V2,
    TIME_DECODE_V2,
)


# Valid characters for hex certainty
_HEX_CHARS = set("0123456789abcdef")

# Object key pattern: 1-3 lowercase words, underscore separated
_OBJECT_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating an LLM format code."""
    is_valid: bool
    errors: tuple[str, ...]
    # Parsed fields (available even if partially invalid)
    speaker: str | None = None
    mode: str | None = None
    act: str | None = None
    certainty: str | None = None
    time: str | None = None
    object_key: str | None = None
    target_key: str | None = None


def validate_code(code: str) -> ValidationResult:
    """
    Validate an LLM-produced AILang-IR code.

    Checks:
    - Header is exactly 6 characters
    - Speaker char is valid (U/S/A/T/?)
    - Mode char is valid (a/o/h/r/c/q/k/b/f)
    - Act code (2 chars) is valid
    - Certainty is valid hex (0-f)
    - Time char is valid (n/p/f/a/?)
    - Object key: 1-3 lowercase words, underscore separated
    - Target key (if present): same format as object key

    Returns ValidationResult with all errors collected.
    """
    errors: list[str] = []
    speaker = mode = act = certainty = time_code = None
    object_key = target_key = None

    stripped = code.strip()
    if not stripped:
        return ValidationResult(
            is_valid=False,
            errors=("empty code",),
        )

    # Split into tokens
    tokens = stripped.split()
    if len(tokens) < 2:
        errors.append(f"expected at least 2 tokens (header + object), got {len(tokens)}")

    # --- Header validation ---
    header = tokens[0] if tokens else ""
    if len(header) != 6:
        errors.append(f"header must be 6 chars, got {len(header)}: '{header}'")
    else:
        # Speaker [0]
        speaker = header[0]
        if speaker not in SPEAKER_DECODE:
            errors.append(f"invalid speaker '{speaker}', expected one of: {', '.join(sorted(SPEAKER_DECODE))}")

        # Mode [1]
        mode = header[1]
        if mode not in MODE_DECODE_V2:
            errors.append(f"invalid mode '{mode}', expected one of: {', '.join(sorted(MODE_DECODE_V2))}")

        # Act [2:4]
        act = header[2:4]
        if act not in ACT_DECODE_V2:
            errors.append(f"invalid act '{act}', expected one of: {', '.join(sorted(ACT_DECODE_V2))}")

        # Certainty [4]
        certainty = header[4]
        if certainty not in _HEX_CHARS:
            errors.append(f"invalid certainty '{certainty}', expected hex digit 0-f")

        # Time [5]
        time_code = header[5]
        if time_code not in TIME_DECODE_V2:
            errors.append(f"invalid time '{time_code}', expected one of: {', '.join(sorted(TIME_DECODE_V2))}")

    # --- Object key validation ---
    if len(tokens) >= 2:
        obj_token = tokens[1]
        # Check for accidental > prefix
        if obj_token.startswith(">"):
            errors.append("object key cannot start with '>'")
        else:
            object_key = obj_token
            _validate_key(object_key, "object", errors)

    # --- Target key and act label validation (optional) ---
    target_start = 2
    if len(tokens) > target_start:
        tgt_token = tokens[target_start]
        if tgt_token.startswith(">"):
            raw_target = tgt_token[1:]
            if not raw_target:
                errors.append("target key is empty after '>'")
            else:
                target_key = raw_target
                _validate_key(target_key, "target", errors)
            target_start += 1
        # Act label tokens (#agree, #disagree, etc.) are allowed
        remaining = [t for t in tokens[target_start:] if not t.startswith("#")]
        if remaining:
            errors.append(f"unexpected extra tokens: {' '.join(remaining)}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        speaker=speaker,
        mode=mode,
        act=act,
        certainty=certainty,
        time=time_code,
        object_key=object_key,
        target_key=target_key,
    )


def _validate_key(key: str, label: str, errors: list[str]) -> None:
    """Validate an object or target key."""
    if not _OBJECT_KEY_RE.match(key):
        errors.append(f"{label} key '{key}' contains invalid characters (use lowercase, digits, underscores)")
        return
    words = key.split("_")
    if len(words) > 3:
        errors.append(f"{label} key '{key}' has {len(words)} words, max 3 allowed")
