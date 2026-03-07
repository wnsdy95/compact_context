"""
AILang-IR Real Value Verification.

Validates 4 claims with live LLM tests:
  1. LLM parser vs rule-based parser quality
  2. Fuzzy search re-reference in real conversation
  3. Compressed context → LLM understanding
  4. Concrete use case: raw text vs AILang-IR

Requires:
  - anthropic SDK installed
  - ANTHROPIC_API_KEY in .env or environment

Usage:
    PYTHONPATH=src python scripts/verify_real_value.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Edit .env or export it.", file=sys.stderr)
    sys.exit(1)

import anthropic
from ailang_ir.pipeline import Pipeline
from ailang_ir.models.domain import SpeakerRole, SemanticFrame
from ailang_ir.llm.llm_parser import LLMParser
from ailang_ir.llm.codec import LLMCodec
from ailang_ir.llm.format_spec import get_format_spec
from ailang_ir.llm.validator import validate_code
from ailang_ir.memory.store import MemoryStore
from ailang_ir.encoder.concept_table import ConceptTable


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Test corpus: 15 realistic conversation turns
# ---------------------------------------------------------------------------

CONVERSATION = [
    ("user",  "I think we should use PostgreSQL for the main database."),
    ("agent", "PostgreSQL is a solid choice for relational data."),
    ("user",  "We also need a caching layer for frequently accessed data."),
    ("agent", "I suggest Redis for the caching layer."),
    ("user",  "I agree, Redis is fast and well supported."),
    ("user",  "We should design the API as RESTful endpoints."),
    ("agent", "Have you considered GraphQL instead of REST?"),
    ("user",  "I prefer REST over GraphQL for simplicity."),
    ("agent", "Maybe we should use a microservices architecture."),
    ("user",  "I disagree, a monolith is better for our team size."),
    ("agent", "That is a fair point for a small team."),
    ("user",  "We will deploy on AWS using container orchestration."),
    ("agent", "I suggest ECS over Kubernetes for simplicity."),
    ("user",  "How does blue-green deployment handle database schema changes?"),
    ("agent", "Schema changes require backward-compatible migrations."),
]

SPEAKER_MAP = {"user": SpeakerRole.USER, "agent": SpeakerRole.AGENT}


def call_claude(system: str, user: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Direct Anthropic API call."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


# ===================================================================
# VERIFICATION 1: LLM Parser vs Rule-Based Parser
# ===================================================================

def verify_1_llm_vs_rule():
    section("VERIFICATION 1: LLM Parser vs Rule-Based Parser")

    llm_parser = LLMParser(model="claude-haiku-4-5-20251001")
    pipe = Pipeline()
    codec = LLMCodec()

    print(f"{'Turn':>4}  {'Speaker':>7}  {'Rule-based':25s}  {'LLM Parser':25s}  {'Winner'}")
    print("-" * 95)

    rule_quality = 0
    llm_quality = 0
    llm_valid = 0

    for i, (speaker_str, text) in enumerate(CONVERSATION, 1):
        speaker = SPEAKER_MAP[speaker_str]

        # Rule-based
        rule_result = pipe.process(text, speaker, tags=None)
        rule_code = codec.encode(rule_result.frame)
        rule_obj = rule_result.frame.object.canonical if rule_result.frame.object else "?"

        # LLM parser
        llm_frame = llm_parser.parse(text, speaker)
        llm_code = codec.encode(llm_frame)
        llm_obj = llm_frame.object.canonical if llm_frame.object else "?"

        # Validate LLM code
        vr = validate_code(llm_code)
        if vr.is_valid:
            llm_valid += 1

        # Quality heuristic: does the object key contain a meaningful word from the text?
        text_words = set(text.lower().replace(",", "").replace(".", "").replace("?", "").split())
        rule_words = set(rule_obj.split("_"))
        llm_words = set(llm_obj.split("_"))

        rule_meaningful = len(rule_words & text_words)
        llm_meaningful = len(llm_words & text_words)

        if llm_meaningful > rule_meaningful:
            winner = "LLM"
            llm_quality += 1
        elif rule_meaningful > llm_meaningful:
            winner = "Rule"
            rule_quality += 1
        else:
            winner = "Tie"

        tag = "U" if speaker_str == "user" else "A"
        print(f"[{i:2d}]  [{tag}]     {rule_code:25s}  {llm_code:25s}  {winner}")

        time.sleep(0.3)  # Rate limit

    total = len(CONVERSATION)
    print()
    print(f"  LLM valid codes:     {llm_valid}/{total} ({llm_valid/total*100:.0f}%)")
    print(f"  LLM better keys:     {llm_quality}/{total}")
    print(f"  Rule better keys:    {rule_quality}/{total}")
    print(f"  Tie:                 {total - llm_quality - rule_quality}/{total}")

    v1_pass = llm_valid / total >= 0.9
    print(f"\n  VERDICT: {'PASS' if v1_pass else 'FAIL'} — LLM valid rate {'≥' if v1_pass else '<'} 90%")
    return v1_pass


# ===================================================================
# VERIFICATION 2: Fuzzy Search Re-Reference in Real Conversation
# ===================================================================

def verify_2_fuzzy_reref():
    section("VERIFICATION 2: Fuzzy Search Re-Reference in Real Conversation")

    pipe = Pipeline()
    for speaker_str, text in CONVERSATION:
        pipe.process(text, SPEAKER_MAP[speaker_str])

    store = pipe.memory
    print(f"  Stored memories: {store.size}")
    print()

    # Test fuzzy search
    test_queries = ["graph", "database", "rest", "deploy", "schema"]
    print("  Fuzzy search results:")
    total_fuzzy_hits = 0
    for query in test_queries:
        results = store.query_by_entity_fuzzy(query, threshold=0.5)
        if results:
            total_fuzzy_hits += len(results)
            keys = [(m.frame.object.canonical, f"{sim:.2f}") for m, sim in results[:3]]
            print(f"    '{query}' → {keys}")
        else:
            print(f"    '{query}' → (no matches)")

    # Test ConceptTable ref_fuzzy
    print()
    print("  ConceptTable ref_fuzzy:")
    ct = ConceptTable()
    all_keys = []
    for mem in store.query_recent(50):
        if mem.frame.object:
            all_keys.append(mem.frame.object.canonical)

    re_refs = 0
    for key in all_keys:
        r = ct.ref_fuzzy(key, threshold=0.6)
        marker = "$" if r.startswith("$") else "#"
        if r.startswith("$"):
            re_refs += 1
        print(f"    {key:30s} → {r}")

    print()
    print(f"  Total keys: {len(all_keys)}, Unique concepts: {ct.size}, Re-references: {re_refs}")

    v2_pass = re_refs >= 1
    print(f"\n  VERDICT: {'PASS' if v2_pass else 'FAIL'} — re-references {'≥' if v2_pass else '<'} 1")
    return v2_pass


# ===================================================================
# VERIFICATION 3: Compressed Context → LLM Understanding
# ===================================================================

def verify_3_context_understanding():
    section("VERIFICATION 3: Compressed Context → LLM Understanding")

    # Process conversation
    pipe = Pipeline()
    codec = LLMCodec()
    for speaker_str, text in CONVERSATION:
        pipe.process(text, SPEAKER_MAP[speaker_str])

    compressed_context = pipe.export_context(len(CONVERSATION))
    format_spec = get_format_spec()
    raw_conversation = "\n".join(f"[{s}] {t}" for s, t in CONVERSATION)

    # Questions to ask about the conversation
    questions = [
        ("What database was chosen?", "PostgreSQL", "postgres"),
        ("What was the disagreement about?", "microservices vs monolith", "monolith"),
        ("What deployment platform was selected?", "AWS", "aws"),
        ("What caching solution was suggested?", "Redis", "redis"),
    ]

    # Test A: Give compressed context + FORMAT_SPEC to Claude
    system_compressed = f"""You have access to a compressed conversation log in AILang-IR format.

{format_spec}

Here is the compressed conversation:
{compressed_context}

Answer questions about this conversation concisely (1-2 sentences)."""

    # Test B: Give raw conversation text to Claude
    system_raw = f"""Here is a conversation transcript:

{raw_conversation}

Answer questions about this conversation concisely (1-2 sentences)."""

    print("  Testing whether Claude can understand compressed context...\n")

    comp_correct = 0
    raw_correct = 0

    comp_tokens = len(system_compressed.split())
    raw_tokens = len(system_raw.split())

    print(f"  Compressed system prompt: ~{comp_tokens} words")
    print(f"  Raw system prompt:        ~{raw_tokens} words")
    print(f"  Ratio:                    {comp_tokens/raw_tokens*100:.0f}%")
    print()

    for question, expected, keyword in questions:
        # Compressed
        answer_comp = call_claude(system_compressed, question)
        comp_has_keyword = keyword.lower() in answer_comp.lower()
        if comp_has_keyword:
            comp_correct += 1

        # Raw
        answer_raw = call_claude(system_raw, question)
        raw_has_keyword = keyword.lower() in answer_raw.lower()
        if raw_has_keyword:
            raw_correct += 1

        comp_mark = "✓" if comp_has_keyword else "✗"
        raw_mark = "✓" if raw_has_keyword else "✗"

        print(f"  Q: {question}")
        print(f"    Compressed [{comp_mark}]: {answer_comp[:100]}...")
        print(f"    Raw        [{raw_mark}]: {answer_raw[:100]}...")
        print()

        time.sleep(0.5)

    total = len(questions)
    print(f"  Compressed correct: {comp_correct}/{total}")
    print(f"  Raw correct:        {raw_correct}/{total}")

    v3_pass = comp_correct >= total * 0.5  # At least 50% correct
    print(f"\n  VERDICT: {'PASS' if v3_pass else 'FAIL'} — compressed understanding {'≥' if v3_pass else '<'} 50%")
    return v3_pass


# ===================================================================
# VERIFICATION 4: Real Use Case — Context Window Savings
# ===================================================================

def verify_4_use_case():
    section("VERIFICATION 4: Real Use Case — Context Window Savings")

    pipe = Pipeline()
    codec = LLMCodec()

    # Simulate a long conversation (use our 15 turns × 3 = 45 turns)
    extended_conv = CONVERSATION * 3
    for speaker_str, text in extended_conv:
        pipe.process(text, SPEAKER_MAP[speaker_str])

    # Raw text size
    raw_text = "\n".join(f"[{s}] {t}" for s, t in extended_conv)
    raw_bytes = len(raw_text.encode("utf-8"))
    raw_words = len(raw_text.split())

    # Compressed context
    compressed = pipe.export_context(len(extended_conv))
    comp_bytes = len(compressed.encode("utf-8"))
    comp_words = len(compressed.split())

    # With FORMAT_SPEC overhead
    spec = get_format_spec()
    spec_words = len(spec.split())
    total_injection = comp_words + spec_words

    print(f"  Conversation: {len(extended_conv)} turns")
    print(f"  Unique memories: {pipe.memory_size} (deduplication active)")
    print()
    print(f"  Raw text:          {raw_words:5d} words  |  {raw_bytes:5d} bytes")
    print(f"  Compressed:        {comp_words:5d} words  |  {comp_bytes:5d} bytes")
    print(f"  FORMAT_SPEC:       {spec_words:5d} words")
    print(f"  Total injection:   {total_injection:5d} words  (compressed + spec)")
    print()

    savings_pct = (1 - total_injection / raw_words) * 100 if raw_words > 0 else 0
    extension_factor = raw_words / total_injection if total_injection > 0 else 0

    print(f"  Word savings:      {savings_pct:.1f}%")
    print(f"  Context extension: {extension_factor:.1f}x")
    print()

    # The real value proposition
    print("  USE CASE: Context Window Optimization")
    print("  " + "-" * 50)
    print(f"  If your context window is 4,000 tokens:")
    print(f"    Raw:        fits ~{4000 // max(raw_words // len(extended_conv), 1)} turns of conversation")
    print(f"    Compressed: fits ~{(4000 - spec_words) // max(comp_words // pipe.memory_size, 1)} unique memories")
    print(f"                ({extension_factor:.1f}x more conversation history)")
    print()
    print("  ADDITIONAL VALUE:")
    print(f"    - Deduplication: {len(extended_conv)} turns → {pipe.memory_size} unique memories")
    print(f"    - Structured search: query by entity, act, speaker, fuzzy match")
    print(f"    - Contradiction detection: built-in")
    print(f"    - Memory consolidation: merge related memories")

    # Test: ask Claude to use compressed context for a task
    print()
    print("  LIVE TEST: Can Claude use compressed context for a real task?")
    print("  " + "-" * 50)

    system = f"""{get_format_spec()}

Compressed conversation context:
{pipe.export_context(20)}

You are continuing this software architecture discussion. Based on the compressed context, write a brief project summary (3-5 bullet points) covering the key decisions made."""

    summary = call_claude(system, "Summarize the key architectural decisions from our discussion.")

    print(f"  Claude's summary from compressed context:")
    for line in summary.strip().splitlines():
        print(f"    {line}")

    # Check if key topics appear
    key_topics = ["postgres", "redis", "rest", "monolith", "aws", "ecs", "blue-green", "deploy"]
    found = sum(1 for t in key_topics if t.lower() in summary.lower())

    print()
    print(f"  Key topics found in summary: {found}/{len(key_topics)}")

    v4_pass = found >= 3
    print(f"\n  VERDICT: {'PASS' if v4_pass else 'FAIL'} — key topics {'≥' if v4_pass else '<'} 3 found")
    return v4_pass


# ===================================================================
# MAIN
# ===================================================================

def main():
    section("AILang-IR REAL VALUE VERIFICATION")
    print("  This script validates that AILang-IR provides real, measurable value")
    print("  through live LLM tests — not synthetic benchmarks.\n")

    results = {}

    try:
        results["V1: LLM Parser vs Rule-Based"] = verify_1_llm_vs_rule()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["V1: LLM Parser vs Rule-Based"] = False

    try:
        results["V2: Fuzzy Re-Reference"] = verify_2_fuzzy_reref()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["V2: Fuzzy Re-Reference"] = False

    try:
        results["V3: Context Understanding"] = verify_3_context_understanding()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["V3: Context Understanding"] = False

    try:
        results["V4: Real Use Case"] = verify_4_use_case()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["V4: Real Use Case"] = False

    # Final summary
    section("FINAL RESULTS")
    all_pass = True
    for name, passed in results.items():
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}] {name}")

    print()
    if all_pass:
        print("  ALL VERIFICATIONS PASSED.")
        print("  AILang-IR provides measurable value in real LLM usage scenarios.")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"  {len(failed)} verification(s) failed:")
        for f in failed:
            print(f"    - {f}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
