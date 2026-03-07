# FRONTIER.md
## Current Compressed State and Re-Entry Frontier

This file stores the compressed working state of the current or most recent cycle.

It is not durable memory.
It is not a transcript.
It is not a long narrative log.

Its purpose is simple:

- preserve the current working state in compressed form
- identify the exact unresolved frontier
- allow the next cycle to restart cleanly without dragging the entire prior context

This file may change frequently.

---

## Core Principle

Do not preserve everything.
Preserve only what is needed to restart correctly.

A good frontier should make the next cycle possible with minimal ambiguity and minimal context drag.

---

## What Belongs Here

Store only:

- what was completed in the latest meaningful cycle
- what remains incomplete
- what assumptions currently hold
- the exact bottleneck or unresolved point
- immediate risks / uncertainties
- the smallest trustworthy next action
- optional relevant file/module pointers for restart

---

## What Must Not Be Stored Here

Do not store:

- broad project philosophy
- durable architecture rules that belong in `MEMORY.md`
- verbose transcripts
- emotional or decorative recap
- raw brainstorming dumps
- unrelated historical clutter

---

## Update Rule

Update this file at the end of any meaningful cycle when one of the following is true:

- an implementation slice is complete
- a stable analysis result has been produced
- a scope boundary has been reached
- progress is blocked by a new decision
- context has become noisy enough that clean restart is better than continuation

When updating, replace stale frontier state instead of endlessly appending noise.

If historical tracking is needed, keep it concise and archived explicitly.

---

## Re-Entry Rule

A fresh cycle should restart from this file after:
1. reading `CLAUDE.md`
2. reading `SYSTEM_PROMPT.md`
3. running `BOOTSTRAP.md`
4. reading `MEMORY.md`
5. inspecting repository reality if code work is involved

Then:
- read `FRONTIER.md`
- locate the active unresolved frontier
- resume from that frontier, not from the full old conversation

---

## Frontier Entry Format

Use the following structure.

### Cycle ID
`CYCLE-YYYYMMDD-<short-key>`

### Status
`active | blocked | ready-for-implementation | ready-for-review | archived`

### Objective
What this cycle was trying to accomplish.

### Completed
- ...
- ...

### Current State
- ...
- ...

### Active Frontier
The exact unresolved point where the next cycle should begin.

### Risks / Unknowns
- ...
- ...

### Next Recommended Action
The smallest trustworthy next step.

### Relevant Files / Modules
- ...
- ...

### Notes to Future Self
Only include compact restart-critical notes.

---

## Active Frontier

### Cycle ID
`CYCLE-20260307-SEMANTIC-CORE-MVP`

### Status
`ready-for-review`

### Objective
Implement the first working semantic core: domain models, normalization vocabulary, rule-based parser, symbolic encoder, natural language decoder, and in-memory event store. Validate with a full test suite.

### Completed
- Governance file consistency pass (BOOTSTRAP.md updated with SYSTEM_PROMPT.md reference)
- Python project structure: `src/ailang_ir/` with 6 modules (models, normalize, parser, encoder, decoder, memory)
- Typed domain models: Entity, Predicate, SemanticFrame, EventMemory, RelationEdge, CompressionRule, ReconstructionPlan + value enums
- Normalization vocabulary: surface→canonical mappings for acts, modes, sentiment, time, certainty. Object key normalization.
- Rule-based parser: natural language → SemanticFrame. Handles opinions, questions, comparisons, hypotheses. Extracts object/target entities.
- Symbolic encoder: SemanticFrame → deterministic compact code (e.g. `U|OP|BELIEVE|CONCEPT|C84|NOW`). Bidirectional decode.
- Natural language reconstructor: SemanticFrame → readable text in 3 styles (declarative, conversational, summary). Round-trip from compact code.
- In-memory event store: storage, dedup, entity/act/speaker/tag queries, recent query, contradiction detection, supersession, relation graph.
- 82 tests passing across all modules.

### Current State
- Full pipeline works: text → parse → encode → decode → reconstruct
- Memory store supports storage, retrieval, dedup, and contradiction detection
- All 82 tests pass
- No external dependencies beyond Python stdlib + pytest

### Active Frontier
The semantic core MVP is functionally complete. Next areas of improvement:
1. **Parser quality**: Object extraction is still coarse for complex sentences. Consider noun-phrase chunking or lightweight NLP.
2. **Reconstruction fluency**: Declarative style produces grammatically imperfect output for some sentence structures.
3. **Codebook governance**: The symbolic codebook should be versioned if it evolves.
4. **Persistence**: Memory store is in-memory only. Add file/DB persistence for cross-session durability.
5. **Evaluation fixtures**: Add a curated set of input→expected-output pairs for regression testing.

### Risks / Unknowns
- Parser depends on keyword matching — ambiguous sentences may be misclassified.
- Object normalization can produce overly long keys for complex clauses.
- No multilingual support yet (English-only).
- Codebook is static — no mechanism for learning or adapting.

### Next Recommended Action
Choose ONE of:
- (a) Add a curated evaluation fixture set (10-20 sentence→frame pairs) for regression testing
- (b) Implement file-based persistence for MemoryStore
- (c) Improve parser object extraction with better noun-phrase boundaries

### Relevant Files / Modules
- `src/ailang_ir/models/domain.py` — core types
- `src/ailang_ir/normalize/vocabulary.py` — normalization rules
- `src/ailang_ir/parser/rule_parser.py` — rule-based parser
- `src/ailang_ir/encoder/codebook.py` — symbolic encoder
- `src/ailang_ir/decoder/reconstructor.py` — NL reconstructor
- `src/ailang_ir/memory/store.py` — event memory store
- `tests/` — 82 tests across 6 test files

### Notes to Future Self
The MVP is intentionally narrow. Do not over-engineer compression or storage before the semantic representation is validated on more diverse inputs.

---

## Archival Rule

When the active frontier is resolved:
- either replace it with the next active frontier
- or move the old frontier into a concise archived section below

Do not allow unresolved and resolved frontiers to mix unclearly.

---

## Archived Frontiers

### CYCLE-20260307-WORKFLOW-FOUNDATION [archived]
Status: `completed`
Outcome: Governance document architecture established. All root files synchronized. BOOTSTRAP.md updated with SYSTEM_PROMPT.md reference.

Use only if needed. Keep concise.