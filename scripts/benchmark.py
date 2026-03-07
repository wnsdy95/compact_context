"""
AILang-IR effectiveness and compression benchmark.
Measures compression ratio, semantic preservation, and practical utility.
Compares v1 (pipe-delimited) vs v2 (assembly IR) encoding.
"""

from ailang_ir import Pipeline
from ailang_ir.encoder import SymbolicEncoder, ConceptTable
from ailang_ir.decoder import Reconstructor

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
# 1. COMPRESSION RATIO — v1 vs v2
# ============================================================
section("1. COMPRESSION RATIO — v1 vs v2")

pipe_v1 = Pipeline(encoding_version=1)
pipe_v2 = Pipeline(encoding_version=2)

total_raw = 0
total_v1 = 0
total_v2 = 0

print()
print(f"  {'Sentence':<50s} {'Raw':>4s} {'v1':>4s} {'v2':>4s}  {'v1%':>5s} {'v2%':>5s}")
print(f"  {'-'*50} {'----':>4s} {'----':>4s} {'----':>4s}  {'-----':>5s} {'-----':>5s}")

for text in SENTENCES:
    r1 = pipe_v1.process(text)
    r2 = pipe_v2.process(text)
    raw_b = len(text.encode("utf-8"))
    v1_b = len(r1.compact_code.encode("utf-8"))
    v2_b = len(r2.compact_code.encode("utf-8"))
    total_raw += raw_b
    total_v1 += v1_b
    total_v2 += v2_b
    v1_pct = v1_b / raw_b
    v2_pct = v2_b / raw_b
    short = text[:48] + ".." if len(text) > 50 else text
    print(f"  {short:<50s} {raw_b:4d} {v1_b:4d} {v2_b:4d}  {v1_pct:5.0%} {v2_pct:5.0%}")

print()
v1_ratio = total_v1 / total_raw
v2_ratio = total_v2 / total_raw
print(f"  Raw total:      {total_raw} bytes")
print(f"  v1 total:       {total_v1} bytes  ({v1_ratio:.1%})")
print(f"  v2 total:       {total_v2} bytes  ({v2_ratio:.1%})")
print(f"  Improvement:    v1={v1_ratio:.1%} → v2={v2_ratio:.1%}  ({(v1_ratio-v2_ratio)/v1_ratio:.0%} reduction)")


# ============================================================
# 2. SEMANTIC FIELD PRESERVATION (v2 round-trip)
# ============================================================
section("2. SEMANTIC FIELD PRESERVATION (v2 round-trip)")

pipe_rt = Pipeline(encoding_version=2)

preserved_full = 0
field_totals = {"speaker": 0, "mode": 0, "act": 0, "object": 0, "certainty": 0, "time": 0}
field_preserved = {"speaker": 0, "mode": 0, "act": 0, "object": 0, "certainty": 0, "time": 0}

for text in SENTENCES:
    r = pipe_rt.process(text)
    fields = encoder.decode_fields_v2(r.compact_code, pipe_rt.concept_table)

    checks = {
        "speaker": fields.get("speaker", "?") != "?",
        "mode": fields.get("mode", "?") != "?",
        "act": fields.get("act", "uk") != "uk",
        "object": fields.get("object", "?obj") != "?obj",
        "certainty": "certainty" in fields,
        "time": fields.get("time", "?") != "?",
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
    bar = "#" * int(pct * 20) + "." * (20 - int(pct * 20))
    print(f"    {k:12s} [{bar}] {pct:.0%} ({field_preserved[k]}/{field_totals[k]})")


# ============================================================
# 3. DEDUPLICATION + CONVERSATION COMPRESSION (v2)
# ============================================================
section("3. CONVERSATION COMPRESSION (v2 with re-referencing)")

pipe_conv = Pipeline(encoding_version=2)
codes = []
for text in CONVERSATION:
    r = pipe_conv.process(text)
    codes.append(r.compact_code)

raw_total = sum(len(t.encode("utf-8")) for t in CONVERSATION)
code_total = sum(len(c.encode("utf-8")) for c in codes)
stored = pipe_conv.memory_size
ct_size = pipe_conv.concept_table.size

print(f"  Input sentences:  {len(CONVERSATION)}")
print(f"  Stored memories:  {stored}")
print(f"  ConceptTable entries: {ct_size}")
print(f"  Raw input size:   {raw_total} bytes")
print(f"  v2 codes size:    {code_total} bytes")
print(f"  Effective compression: {code_total/raw_total:.1%} of raw conversation")
print()
print("  Individual codes:")
for text, code in zip(CONVERSATION, codes):
    marker = "$" if "$" in code else "#"
    print(f"    [{marker}] {code:<35s}  <- {text[:40]}")


# ============================================================
# 4. RECONSTRUCTION FIDELITY
# ============================================================
section("4. RECONSTRUCTION FIDELITY (v2 meaning preservation)")

pipe_fid = Pipeline(encoding_version=2)
test_pairs = [
    ("I think graph memory is better than linear text.", ["graph", "memory", "linear", "text"]),
    ("We need a normalization layer.", ["norm", "layer"]),
    ("What is the best approach for semantic compression?", ["sema", "comp"]),
    ("I prefer typed models over loose dictionaries.", ["typed", "model", "loose", "dict"]),
    ("This is definitely the right architecture.", ["arch"]),
]

total_keywords = 0
preserved_keywords = 0

for text, keywords in test_pairs:
    r = pipe_fid.process(text)
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
# 5. ASSESSMENT
# ============================================================
section("5. v2 ASSESSMENT")

print(f"""
  v1 compression ratio: {v1_ratio:.1%} (often LARGER than original)
  v2 compression ratio: {v2_ratio:.1%} (first mentions)
  Improvement: {(v1_ratio-v2_ratio)/v1_ratio:.0%} reduction

  KEY CHANGES in v2:
  - Fixed-width 6-char header (was 8-18 chars with delimiters)
  - 1-char mode codes (was 2-3 chars)
  - 2-char act codes (was 4-8 chars)
  - 1-char time codes (was 3-5 chars)
  - 1 hex char certainty (was 3-4 chars)
  - Compressed object keys (truncated words, removed fillers)
  - ConceptTable re-referencing ($id for repeated concepts)
  - No pipe delimiters (space-separated, position-based header)

  REMAINING LIMITATIONS:
  - Parser is still rule-based (keyword matching)
  - Deduplication is still exact-match
  - Reconstruction is still template-based
""")
