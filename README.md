# AILang-IR

Semantic intermediate representation for AI conversation context. Converts natural language into structured semantic frames for compact storage, deduplication, structured search, and stance tracking.

**Not a raw text replacement.** AILang-IR is a semantic memory index — it works alongside raw text to provide structure, compression, and queryability.

## Key Features

- **Semantic parsing**: NL → structured frames (speaker, mode, act, object, certainty, time)
- **Compact encoding**: 3 encoding versions (v1/v2/v3), LLM-native format with natural-word keys
- **Hybrid export**: compact codes + source snippets for detail preservation (100% understanding on novel corpora)
- **Deduplication**: repeated conversations collapse to unique memories
- **Structured search**: query by entity, act, speaker, fuzzy match
- **Stance tracking**: agree/disagree/prefer/reject with act labels
- **Context extension**: 3.3x more history in same context window (with dedup)

## Installation

```bash
pip install -e .
```

Requires Python 3.11+. No external dependencies for core functionality.

## Quick Start

### Python API

```python
from ailang_ir import Pipeline

pipe = Pipeline()

# Process conversation
pipe.process("I think PostgreSQL is the best database.")
pipe.process("I disagree, MongoDB is better for our use case.")
pipe.process("Our API handles about 10000 requests per second.")

# Export as LLM-readable format (codes only)
print(pipe.export_context(n=10))
# Uoblbn postgresql_database
# Uadgbn mongodb_better #disagree
# Uaukbn api_handles

# Export with source snippets (preserves details like numbers)
print(pipe.export_context(n=10, source_snippets=True))
# Uoblbn postgresql_database | postgresql best database
# Uadgbn mongodb_better #disagree | disagree mongodb better for use case
# Uaukbn api_handles | api handles about 10000 requests per second

# Search by entity
results = pipe.search("postgresql")

# Check for contradictions
contradictions = pipe.contradictions_for("MongoDB is terrible.")

# Get dedup stats
print(pipe.stats())
```

### CLI

```bash
# Compress text to semantic codes
echo "I think PostgreSQL is great." | python -m ailang_ir compress

# With source snippets for detail preservation
echo "Processing takes 6 hours." | python -m ailang_ir compress --snippets

# Without act labels
echo "I disagree with that." | python -m ailang_ir compress --no-act-labels

# Export stored memories
python -m ailang_ir export --snippets memory.json

# Print format spec (for LLM system prompts)
python -m ailang_ir spec

# Interactive REPL
python -m ailang_ir interactive --snippets --store memory.json
```

Interactive commands: `/speaker <u|s|a>`, `/export [n]`, `/snippets`, `/stats`, `/quit`

### LLM Integration

Inject the format spec into your system prompt, then use compressed context:

```python
from ailang_ir import Pipeline, get_format_spec

pipe = Pipeline()
# ... process conversation ...

system_prompt = f"""
{get_format_spec()}

Compressed conversation:
{pipe.export_context(n=50, source_snippets=True)}
"""
```

## Format Reference

Each line: `HEADER object_key [>target] [#act_label] [| source snippet]`

Header = 6 positional chars: `S M AA C T`

| Pos | Field | Values |
|-----|-------|--------|
| 0 | Speaker | U=user, A=agent, S=system, T=third, ?=unknown |
| 1 | Mode | a=assert, o=opinion, h=hypothesis, r=request, q=question |
| 2-3 | Act | bl=believe, sg=suggest, pr=prefer, ag=agree, dg=disagree, nd=need, dc=decide, rj=reject, qu=query, ob=observe, cm=compare, pl=plan, wn=warn, ex=explain |
| 4 | Certainty | 0-f hex (0=0%, f=100%) |
| 5 | Time | n=now, p=past, f=future, a=atemporal |

Examples:
```
Uoblbn postgresql_main              -- User believes postgresql is main choice
Uadgbn monolith #disagree           -- User disagrees about monolith
Uaprbn rest >graphql #prefer        -- User prefers rest over graphql
Uoblbn data_pipeline | takes 6 hours -- With source snippet
```

## Architecture

```
NL Text → Parser → SemanticFrame → Encoder → Compact Code
                                  ↓
                              MemoryStore (dedup, search, contradictions)
                                  ↓
                            LLMCodec → LLM-readable export
```

Layers are kept separate: ingestion, normalization, semantic extraction, frame construction, compression, storage, retrieval, reconstruction.

## Export Modes

| Mode | Flag | Understanding | Size | Best for |
|------|------|--------------|------|----------|
| Codes only | (default) | 70% | ~95% of raw | Maximum compression |
| Codes + labels | `act_labels=True` | 70% | ~95% of raw | Stance-critical conversations |
| Hybrid | `source_snippets=True` | 100% | ~171% of raw* | Detail preservation |

*Hybrid break-even: 86% at 30 turns, 57% at 45 turns, 22% at 120 turns (with dedup).

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 310 tests
```

## Project Structure

```
src/ailang_ir/
  __init__.py          # Public API: Pipeline, LLMCodec, validate_code
  __main__.py          # CLI
  pipeline.py          # Unified Pipeline API
  models/domain.py     # SemanticFrame, Entity, enums
  parser/rule_parser.py # Rule-based NL parser
  encoder/codebook.py  # Symbolic encoder (v1/v2/v3)
  encoder/concept_table.py # ConceptTable for re-referencing
  decoder/reconstructor.py # Code → NL reconstruction
  memory/store.py      # MemoryStore (dedup, search, persistence)
  normalize/vocabulary.py # Normalization + compression vocabulary
  llm/
    codec.py           # LLM-native encode/decode + source snippets
    format_spec.py     # FORMAT_SPEC for system prompt injection
    validator.py       # Structural validation of LLM codes
    llm_parser.py      # LLM-based parser (requires API key)
```

## License

MIT
