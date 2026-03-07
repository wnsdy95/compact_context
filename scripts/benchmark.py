"""
AILang-IR effectiveness and compression benchmark.
Measures compression ratio, semantic preservation, and practical utility.
"""

from ailang_ir import Pipeline
from ailang_ir.encoder import SymbolicEncoder
from ailang_ir.decoder import Reconstructor

pipe = Pipeline()
encoder = SymbolicEncoder()
decoder = Reconstructor()

# Test corpus
SENTENCES = [
    "I think one-to-one sentence mapping will be difficult.",
    "The system should be able to reconstruct natural language later.",
    "Graph memory seems more appropriate than linear text storage.",
    "Natural language should be segmented into meaning units.",
    "What is the best approach for semantic compression?",
    "I prefer typed models over loose dictionaries.",
    "We need a normalization layer for consistent output.",
    "Maybe we should try a different compression strategy.",
    "This is definitely the right architecture.",
    "Previously we used raw string matching.",
    "I believe semantic frames are the right approach.",
    "The reconstruction quality is unfortunately poor.",
    "Let us build a new encoder for compact codes.",
    "We should update the normalization rules.",
    "Semantic frames preserve meaning better than raw text.",
    "I notice the parser misclassifies passive sentences.",
    "We will implement persistence in the next cycle.",
    "Perhaps the compression ratio can be improved.",
    "How does the parser handle ambiguous input?",
    "This is a great improvement over the old system.",
]

# Simulated multi-turn conversation (repeated semantics)
CONVERSATION = [
    "I think we should use graph memory.",
    "Graph-based storage seems like the right approach.",
    "I believe graph memory is the way to go.",
    "We need to implement graph memory for this project.",
    "The graph memory approach is definitely better.",
    "I prefer graph memory over linear storage.",
    "Let's go with graph storage.",
    "Graph memory will work well for our use case.",
    "I think linear storage is insufficient.",
    "We should avoid linear text storage.",
]


def section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# 1. COMPRESSION RATIO
# ============================================================
section("1. COMPRESSION RATIO")

total_raw = 0
total_code = 0
total_summary = 0

for text in SENTENCES:
    r = pipe.process(text)
    raw_b = len(text.encode("utf-8"))
    code_b = len(r.compact_code.encode("utf-8"))
    summ = r.reconstruct("summary")
    summ_b = len(summ.encode("utf-8"))
    ratio = code_b / raw_b
    total_raw += raw_b
    total_code += code_b
    total_summary += summ_b
    print(f"  {ratio:.2f}x | {raw_b:3d}B → {code_b:3d}B | {r.compact_code}")

print()
avg_ratio = total_code / total_raw
print(f"  Raw total:     {total_raw} bytes")
print(f"  Code total:    {total_code} bytes")
print(f"  Summary total: {total_summary} bytes")
print(f"  Code/Raw ratio: {avg_ratio:.2%}")
print(f"  Summary/Raw:    {total_summary/total_raw:.2%}")
print(f"  Savings (code): {1 - avg_ratio:.1%}")


# ============================================================
# 2. SEMANTIC PRESERVATION
# ============================================================
section("2. SEMANTIC FIELD PRESERVATION (round-trip)")

preserved_full = 0
field_totals = {"speaker": 0, "mode": 0, "act": 0, "object": 0, "certainty": 0, "time": 0}
field_preserved = {"speaker": 0, "mode": 0, "act": 0, "object": 0, "certainty": 0, "time": 0}

for text in SENTENCES:
    r = pipe.process(text)
    fields = encoder.decode_fields(r.compact_code)

    checks = {
        "speaker": fields.get("speaker", "?") != "?",
        "mode": fields.get("mode", "?") != "?",
        "act": fields.get("act", "UNK") != "UNK",
        "object": fields.get("object", "?OBJ") != "?OBJ",
        "certainty": "certainty" in fields,
        "time": fields.get("time", "T?") != "T?",
    }

    for k, v in checks.items():
        field_totals[k] += 1
        if v:
            field_preserved[k] += 1

    if all(checks.values()):
        preserved_full += 1

print(f"  Full field preservation: {preserved_full}/{len(SENTENCES)} ({preserved_full/len(SENTENCES):.0%})")
print()
print("  Per-field preservation:")
for k in field_totals:
    pct = field_preserved[k] / field_totals[k]
    bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
    print(f"    {k:12s} {bar} {pct:.0%} ({field_preserved[k]}/{field_totals[k]})")


# ============================================================
# 3. DEDUPLICATION EFFECTIVENESS
# ============================================================
section("3. DEDUPLICATION (repeated semantics)")

pipe2 = Pipeline()
for text in CONVERSATION:
    pipe2.process(text)

raw_total = sum(len(t.encode("utf-8")) for t in CONVERSATION)
stored = pipe2.memory_size
print(f"  Input sentences:  {len(CONVERSATION)}")
print(f"  Stored memories:  {stored}")
print(f"  Dedup ratio:      {stored}/{len(CONVERSATION)} = {stored/len(CONVERSATION):.0%} stored")
print(f"  Raw input size:   {raw_total} bytes")
codes = [encoder.encode(m.frame) for m in pipe2.recent(stored)]
code_total = sum(len(c.encode("utf-8")) for c in codes)
print(f"  Stored code size: {code_total} bytes")
print(f"  Effective compression: {code_total/raw_total:.1%} of raw conversation")


# ============================================================
# 4. RECONSTRUCTION FIDELITY
# ============================================================
section("4. RECONSTRUCTION FIDELITY (meaning preservation)")

test_pairs = [
    ("I think graph memory is better than linear text.", ["graph", "memory", "linear", "text"]),
    ("We need a normalization layer.", ["normalization", "layer"]),
    ("What is the best approach for semantic compression?", ["semantic", "compression"]),
    ("I prefer typed models over loose dictionaries.", ["typed", "model", "loose", "dictionar"]),
    ("This is definitely the right architecture.", ["architecture"]),
]

total_keywords = 0
preserved_keywords = 0

for text, keywords in test_pairs:
    r = pipe.process(text)
    recon = r.reconstruct("declarative").lower()
    code = r.compact_code.lower()
    combined = recon + " " + code

    hits = 0
    for kw in keywords:
        total_keywords += 1
        if kw in combined:
            preserved_keywords += 1
            hits += 1

    status = "OK" if hits == len(keywords) else f"LOST {len(keywords)-hits}/{len(keywords)}"
    print(f"  [{status:8s}] {text[:55]}")

print()
print(f"  Keyword preservation: {preserved_keywords}/{total_keywords} ({preserved_keywords/total_keywords:.0%})")


# ============================================================
# 5. HONEST ASSESSMENT
# ============================================================
section("5. CRITICAL ASSESSMENT")

print("""
  STRENGTHS:
  - Deterministic: same input always produces same code
  - Zero external dependencies (no ML, no network)
  - Full round-trip: text → frame → code → frame → text
  - Deduplication catches exact semantic matches
  - Compact codes are human-readable and debuggable
  - Contradiction detection works for opposing act pairs
  - JSON persistence enables cross-session memory

  WEAKNESSES:
  - Compression ratio is POOR (~90-120% of raw text)
    The compact codes are often LONGER than the original text.
    This is because object keys preserve full noun phrases.
  - Keyword-based parsing is FRAGILE
    Complex sentences, passive voice, nested clauses degrade quality.
  - Deduplication is TOO STRICT
    "I think graph memory is good" and "graph storage seems best"
    are semantically similar but stored as separate memories.
  - Reconstruction is LOSSY and AWKWARD
    Reconstructed text is grammatically imperfect and loses nuance.
  - No actual semantic understanding
    This is pattern matching, not comprehension.
    An LLM would extract meaning far more accurately.
  - English-only, no multilingual support

  VERDICT:
  The system demonstrates the architectural concept but does NOT yet
  deliver practical compression or semantic quality advantages over
  simply storing raw text. The value proposition requires either:
    (a) LLM-assisted parsing for real semantic extraction, or
    (b) semantic similarity for cross-phrasing deduplication, or
    (c) both.

  The current rule-based approach is a skeleton, not a solution.
""")
