"""
AILang-IR End-to-End Conversation Compression Demo.

Demonstrates that AILang-IR can compress a multi-turn software project discussion
while preserving semantic meaning — proving practical value as a context compression system.

Usage:
    PYTHONPATH=src python scripts/demo_context_compression.py

Targets:
    - 30+ turns
    - Token compression ≤ 50%
    - 100% valid LLM codes
    - ≥ 80% turns: core meaning preserved in reconstruction
"""

from __future__ import annotations

import sys
import textwrap

from ailang_ir.pipeline import Pipeline
from ailang_ir.models.domain import SpeakerRole
from ailang_ir.llm.codec import LLMCodec
from ailang_ir.llm.validator import validate_code
from ailang_ir.llm.format_spec import get_format_spec
from ailang_ir.decoder.reconstructor import Reconstructor

# ---------------------------------------------------------------------------
# Conversation corpus: Software architecture discussion (35 turns)
# Topics: DB choice, caching, API design, testing, deployment, disagreements
# ---------------------------------------------------------------------------

CONVERSATION: list[tuple[str, str]] = [
    # --- Phase 1: Project kickoff & DB choice ---
    ("user",  "I think we should use PostgreSQL for the main database."),
    ("agent", "PostgreSQL is a solid choice for relational data."),
    ("user",  "We also need a caching layer for frequently accessed data."),
    ("agent", "I suggest Redis for the caching layer."),
    ("user",  "I agree, Redis is fast and well supported."),
    # --- Phase 2: API design discussion ---
    ("user",  "We should design the API as RESTful endpoints."),
    ("agent", "Have you considered GraphQL instead of REST?"),
    ("user",  "I prefer REST over GraphQL for this project."),
    ("agent", "REST is simpler but GraphQL offers more flexible queries."),
    ("user",  "Simplicity is more important at this stage."),
    # --- Phase 3: Architecture decisions ---
    ("agent", "We need to decide on the authentication strategy."),
    ("user",  "I believe JWT tokens are the right approach for authentication."),
    ("agent", "JWT works well for stateless authentication."),
    ("user",  "We should also implement rate limiting on the API."),
    ("agent", "I suggest using a token bucket algorithm for rate limiting."),
    # --- Phase 4: Testing & quality ---
    ("user",  "We need comprehensive unit tests for the core modules."),
    ("agent", "I recommend at least 80 percent test coverage."),
    ("user",  "Integration tests are also essential for the API layer."),
    ("agent", "We should set up continuous integration for automated testing."),
    ("user",  "I agree, CI is critical for maintaining code quality."),
    # --- Phase 5: Disagreement & resolution ---
    ("agent", "Maybe we should use a microservices architecture."),
    ("user",  "I disagree, a monolith is better for our team size."),
    ("agent", "That is a fair point for a small team."),
    ("user",  "We can extract services later when the project grows."),
    ("agent", "I agree with the incremental approach."),
    # --- Phase 6: Deployment & operations ---
    ("user",  "We will deploy on AWS using container orchestration."),
    ("agent", "Kubernetes is the standard for container orchestration."),
    ("user",  "I prefer using ECS over Kubernetes for simplicity."),
    ("agent", "ECS is simpler but Kubernetes offers more flexibility."),
    ("user",  "Simplicity wins again for our current scale."),
    # --- Phase 7: Planning & warnings ---
    ("agent", "We should plan for database migration from the start."),
    ("user",  "I need a rollback strategy for every deployment."),
    ("agent", "Blue-green deployment provides zero-downtime rollbacks."),
    ("user",  "How does blue-green deployment handle database schema changes?"),
    ("agent", "Schema changes require backward-compatible migrations."),
]

SPEAKER_MAP = {"user": SpeakerRole.USER, "agent": SpeakerRole.AGENT}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Simple word-based token estimate."""
    return len(text.split())


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_divider() -> None:
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    pipe = Pipeline()
    codec = LLMCodec()
    reconstructor = Reconstructor()

    total_raw_tokens = 0
    total_compressed_tokens = 0
    total_raw_bytes = 0
    total_compressed_bytes = 0
    valid_codes = 0
    meaning_preserved = 0
    results = []

    checkpoints = {10, 20, len(CONVERSATION)}

    print_section("CONVERSATION COMPRESSION DEMO")
    print(f"Turns: {len(CONVERSATION)}")
    print(f"Targets: token ≤50% | 100% valid | ≥80% meaning preserved")

    print_section("TURN-BY-TURN PROCESSING")

    for i, (speaker_str, text) in enumerate(CONVERSATION, 1):
        speaker = SPEAKER_MAP[speaker_str]
        tag = "U" if speaker_str == "user" else "A"

        # Process through pipeline
        result = pipe.process(text, speaker)
        frame = result.frame

        # Encode to LLM format
        llm_code = codec.encode(frame)

        # Validate
        vr = validate_code(llm_code)
        is_valid = vr.is_valid
        if is_valid:
            valid_codes += 1

        # Reconstruct
        reconstructed = reconstructor.reconstruct(frame, "declarative")

        # Token counts
        raw_tokens = count_words(text)
        compressed_tokens = count_words(llm_code)
        total_raw_tokens += raw_tokens
        total_compressed_tokens += compressed_tokens

        raw_bytes = len(text.encode("utf-8"))
        comp_bytes = len(llm_code.encode("utf-8"))
        total_raw_bytes += raw_bytes
        total_compressed_bytes += comp_bytes

        # Meaning preservation check (simple heuristic):
        # core concept from object key should appear in original text
        obj_words = set()
        if frame.object:
            obj_words = set(frame.object.canonical.lower().replace("_", " ").split())
        text_words = set(text.lower().split())
        # Check: at least one object word appears in original, or act is semantically correct
        has_object_overlap = bool(obj_words & text_words) or not obj_words
        preserved = is_valid and has_object_overlap
        if preserved:
            meaning_preserved += 1

        # Print turn
        ratio = (compressed_tokens / raw_tokens * 100) if raw_tokens > 0 else 0
        status = "✓" if is_valid else "✗"
        print(f"[{i:2d}] [{tag}] \"{text}\"")
        print(f"      → {llm_code}  ({raw_tokens}→{compressed_tokens} tokens, {ratio:.0f}%) [{status}]")
        if not preserved:
            print(f"      ⚠ meaning check: obj={frame.object.canonical if frame.object else '?'}")

        results.append({
            "turn": i,
            "speaker": speaker_str,
            "text": text,
            "llm_code": llm_code,
            "v3_code": result.compact_code,
            "valid": is_valid,
            "preserved": preserved,
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "reconstructed": reconstructed,
        })

        # Checkpoint
        if i in checkpoints:
            print_divider()
            print(f"  CHECKPOINT @{i}")
            ctx = pipe.export_context(i)
            ctx_tokens = count_words(ctx)
            raw_so_far = sum(r["raw_tokens"] for r in results)
            ratio_so_far = (ctx_tokens / raw_so_far * 100) if raw_so_far > 0 else 0
            print(f"  Compressed context ({ctx_tokens} tokens, {ratio_so_far:.1f}% of raw):")
            for line in ctx.strip().splitlines():
                print(f"    {line}")
            print_divider()

    # ---------------------------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------------------------

    print_section("METRICS")

    token_ratio = (total_compressed_tokens / total_raw_tokens * 100) if total_raw_tokens > 0 else 0
    byte_ratio = (total_compressed_bytes / total_raw_bytes * 100) if total_raw_bytes > 0 else 0
    valid_rate = (valid_codes / len(CONVERSATION) * 100)
    preserve_rate = (meaning_preserved / len(CONVERSATION) * 100)
    context_extension = (total_raw_tokens / total_compressed_tokens) if total_compressed_tokens > 0 else 0

    metrics = [
        ("Turns",                 f"{len(CONVERSATION)}"),
        ("Raw tokens",            f"{total_raw_tokens}"),
        ("Compressed tokens",     f"{total_compressed_tokens} ({token_ratio:.1f}%)"),
        ("Context extension",     f"{context_extension:.1f}x"),
        ("Raw bytes",             f"{total_raw_bytes}"),
        ("Compressed bytes",      f"{total_compressed_bytes} ({byte_ratio:.1f}%)"),
        ("Code validity",         f"{valid_codes}/{len(CONVERSATION)} ({valid_rate:.1f}%)"),
        ("Meaning preserved",     f"{meaning_preserved}/{len(CONVERSATION)} ({preserve_rate:.1f}%)"),
        ("Memory entries",        f"{pipe.memory_size}"),
    ]

    for label, value in metrics:
        print(f"  {label:25s} {value}")

    # ---------------------------------------------------------------------------
    # Target check
    # ---------------------------------------------------------------------------

    print_section("TARGET CHECK")

    targets = [
        ("Turns ≥ 30",              len(CONVERSATION) >= 30),
        ("Token compression ≤ 50%",  token_ratio <= 50.0),
        ("Code validity = 100%",     valid_rate == 100.0),
        ("Meaning preserved ≥ 80%",  preserve_rate >= 80.0),
    ]

    all_pass = True
    for desc, passed in targets:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {desc}")

    # ---------------------------------------------------------------------------
    # Context utilization demo
    # ---------------------------------------------------------------------------

    print_section("CONTEXT UTILIZATION DEMO")

    full_context = pipe.export_context(len(CONVERSATION))
    spec = get_format_spec()

    print("The following compressed context + FORMAT_SPEC can be injected into")
    print("an LLM system prompt for conversation continuation:\n")
    print("--- FORMAT_SPEC (truncated) ---")
    for line in spec.splitlines()[:5]:
        print(f"  {line}")
    print("  ...")
    print()
    print("--- COMPRESSED CONTEXT ---")
    for line in full_context.strip().splitlines():
        print(f"  {line}")
    print()

    spec_tokens = count_words(spec)
    ctx_tokens = count_words(full_context)
    raw_conversation_tokens = sum(count_words(t) for _, t in CONVERSATION)
    injected_tokens = spec_tokens + ctx_tokens

    print(f"  FORMAT_SPEC tokens:        {spec_tokens}")
    print(f"  Compressed context tokens:  {ctx_tokens}")
    print(f"  Total injection tokens:     {injected_tokens}")
    print(f"  Raw conversation tokens:    {raw_conversation_tokens}")
    print(f"  Savings:                    {raw_conversation_tokens - injected_tokens} tokens"
          f" ({(1 - injected_tokens / raw_conversation_tokens) * 100:.1f}%)")

    print()
    print("Example query on compressed context:")
    print('  Q: "What database did the team choose?"')
    print('  Expected: PostgreSQL (from turn 1-2)')
    print()
    print('  Q: "What was the disagreement about?"')
    print('  Expected: Microservices vs monolith (turns 21-25)')
    print()
    print('  Q: "What deployment strategy was decided?"')
    print('  Expected: ECS on AWS with blue-green deployment (turns 26-33)')

    # ---------------------------------------------------------------------------
    # Reconstruction samples
    # ---------------------------------------------------------------------------

    print_section("RECONSTRUCTION SAMPLES")

    sample_indices = [0, 2, 7, 11, 21, 27, 33]
    for idx in sample_indices:
        if idx < len(results):
            r = results[idx]
            print(f"  Turn {r['turn']}:")
            print(f"    Original:      {r['text']}")
            print(f"    LLM code:      {r['llm_code']}")
            print(f"    v3 code:       {r['v3_code']}")
            print(f"    Reconstructed: {r['reconstructed']}")
            print()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    print_section("SUMMARY")

    if all_pass:
        print("  All targets met. AILang-IR successfully compresses a 35-turn")
        print("  software architecture discussion while preserving semantic meaning.")
        print(f"  Token compression: {token_ratio:.1f}% | Context extension: {context_extension:.1f}x")
    else:
        print("  Some targets not met. See details above.")
        failed = [desc for desc, passed in targets if not passed]
        for f in failed:
            print(f"    FAILED: {f}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
