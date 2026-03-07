"""
AILang-IR effectiveness and compression benchmark.
Compares v1 (pipe-delimited) vs v2 (assembly IR) vs v3 (stem-abbreviated) encoding.
"""

from ailang_ir import Pipeline
from ailang_ir.encoder import SymbolicEncoder, ConceptTable
from ailang_ir.decoder import Reconstructor

encoder = SymbolicEncoder()

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
# 1. COMPRESSION RATIO — v1 vs v2 vs v3
# ============================================================
section("1. COMPRESSION RATIO — v1 vs v2 vs v3")

pipe_v1 = Pipeline(encoding_version=1)
pipe_v2 = Pipeline(encoding_version=2)
pipe_v3 = Pipeline(encoding_version=3)

total_raw = 0
total_v1 = 0
total_v2 = 0
total_v3 = 0

print()
print(f"  {'Sentence':<45s} {'Raw':>4s} {'v1':>4s} {'v2':>4s} {'v3':>4s}  {'v1%':>4s} {'v2%':>4s} {'v3%':>4s}")
print(f"  {'-'*45} {'----':>4s} {'----':>4s} {'----':>4s} {'----':>4s}  {'----':>4s} {'----':>4s} {'----':>4s}")

for text in SENTENCES:
    r1 = pipe_v1.process(text)
    r2 = pipe_v2.process(text)
    r3 = pipe_v3.process(text)
    raw_b = len(text.encode("utf-8"))
    v1_b = len(r1.compact_code.encode("utf-8"))
    v2_b = len(r2.compact_code.encode("utf-8"))
    v3_b = len(r3.compact_code.encode("utf-8"))
    total_raw += raw_b
    total_v1 += v1_b
    total_v2 += v2_b
    total_v3 += v3_b
    short = text[:43] + ".." if len(text) > 45 else text
    print(f"  {short:<45s} {raw_b:4d} {v1_b:4d} {v2_b:4d} {v3_b:4d}  {v1_b/raw_b:4.0%} {v2_b/raw_b:4.0%} {v3_b/raw_b:4.0%}")

print()
v1_r = total_v1 / total_raw
v2_r = total_v2 / total_raw
v3_r = total_v3 / total_raw
print(f"  Raw total:      {total_raw} bytes")
print(f"  v1 total:       {total_v1} bytes  ({v1_r:.1%})")
print(f"  v2 total:       {total_v2} bytes  ({v2_r:.1%})")
print(f"  v3 total:       {total_v3} bytes  ({v3_r:.1%})")
print(f"  v1 -> v3:       {v1_r:.1%} -> {v3_r:.1%}  ({(1-v3_r/v1_r):.0%} reduction)")


# ============================================================
# 2. SEMANTIC FIELD PRESERVATION (v3)
# ============================================================
section("2. SEMANTIC FIELD PRESERVATION (v3 round-trip)")

pipe_rt = Pipeline(encoding_version=3)
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
for k in field_totals:
    pct = field_preserved[k] / field_totals[k]
    bar = "#" * int(pct * 20) + "." * (20 - int(pct * 20))
    print(f"    {k:12s} [{bar}] {pct:.0%} ({field_preserved[k]}/{field_totals[k]})")


# ============================================================
# 3. CONVERSATION COMPRESSION (v3)
# ============================================================
section("3. CONVERSATION COMPRESSION (v3)")

pipe_conv = Pipeline(encoding_version=3)
codes = []
for text in CONVERSATION:
    r = pipe_conv.process(text)
    codes.append(r.compact_code)

raw_total = sum(len(t.encode("utf-8")) for t in CONVERSATION)
code_total = sum(len(c.encode("utf-8")) for c in codes)

print(f"  Input: {len(CONVERSATION)} sentences, {raw_total} bytes")
print(f"  v3:    {code_total} bytes ({code_total/raw_total:.1%})")
print(f"  ConceptTable: {pipe_conv.concept_table.size} entries")
print()
for text, code in zip(CONVERSATION, codes):
    marker = "$" if "$" in code else "#"
    print(f"    [{marker}] {code:<25s} <- {text[:40]}")


# ============================================================
# 4. ASSESSMENT
# ============================================================
section("4. SUMMARY")

print(f"""
  Compression evolution:
    v1: {v1_r:.1%}  (pipe-delimited, full enum names)
    v2: {v2_r:.1%}  (fixed header, compressed keys)
    v3: {v3_r:.1%}  (stem abbreviation, no separators)

  v3 key changes over v2:
  - Stem abbreviation dictionary (~100 common domain terms -> 2-char codes)
  - Max 3 content tokens per object key
  - No underscore separators (concatenated abbreviated tokens)
  - Same 6-char header as v2 (no header changes needed)
""")
