# MEMORY.md
## Durable Memory Store

This file stores only durable, high-value memory that should persist across sessions.

It is not a scratchpad.
It is not a work log.
It is not a running transcript.
It is not the place for temporary frontier state.

Use this file only for memory that should survive across multiple future cycles and materially improve continuity, correctness, or architectural coherence.

---

## Purpose

`MEMORY.md` exists to preserve stable truth.

It should contain only information that remains useful even after transient context has been discarded.

If a note will stop mattering after the current implementation slice, it does not belong here.
Put it in `FRONTIER.md` instead.

---

## What Belongs Here

Store only:

- stable project goals
- durable architectural decisions
- domain model commitments
- naming conventions that materially matter
- important constraints
- user collaboration preferences that affect engineering output
- key rejected alternatives worth remembering
- critical unresolved strategic questions that persist beyond one cycle
- warnings that should remain visible across sessions

---

## What Must Not Be Stored Here

Do not store:

- temporary todos
- per-cycle status
- short-lived implementation notes
- repetitive summaries of the latest work
- verbose chat-like context
- already-obvious code details with no future decision value
- unresolved next step details that belong only to the current cycle
- raw frontier / restart instructions

Those belong in `FRONTIER.md`.

---

## Memory Entry Format

Use this format for every durable memory entry.

### [MEM-YYYYMMDD-KEY]
Type: <goal|decision|constraint|preference|question|warning|rejection>
Status: <active|superseded|resolved|experimental>
Scope: <repo|architecture|module|workflow|user|product>
Summary: <one-sentence durable memory>
Details:
- ...
- ...
Why it matters:
- ...
Source:
- <where this came from>
Revision policy:
- <when this should be updated, superseded, or left unchanged>

---

## Durable Memory Quality Rule

Before adding a memory entry, ask:

1. Will this still matter after several future cycles?
2. Would losing this create real confusion, misalignment, or repeated mistakes?
3. Is this a durable truth rather than a temporary working note?

If the answer is not clearly yes, do not store it here.

---

## Supersession Rule

Do not silently overwrite durable memory.

When a durable memory changes:
- mark the old one as `superseded`
- add a new entry with the updated truth
- explain what changed
- preserve continuity of reasoning

---

## Relationship to FRONTIER.md

`MEMORY.md` stores durable truth.
`FRONTIER.md` stores the compressed current state and exact re-entry point.

Do not mix them.

Use:
- `MEMORY.md` for what should persist
- `FRONTIER.md` for what should resume next

---

## Initial Durable Memories

### [MEM-20260307-PROJECT-DIRECTION]
Type: goal
Status: active
Scope: architecture
Summary: This repository is intended to build an AI-oriented semantic intermediate representation system, not a general programming language runtime.
Details:
- The core direction centers on semantic frames, compact symbolic forms, graph-aware memory, and natural language reconstruction.
- The system should prioritize meaning preservation over raw text preservation.
Why it matters:
- It prevents architectural drift toward irrelevant compiler or VM design.
Source:
- initial project framing
Revision policy:
- update only if the product direction fundamentally changes

### [MEM-20260307-LAYER-SEPARATION]
Type: decision
Status: active
Scope: architecture
Summary: Parsing, normalization, semantic framing, compression, storage, retrieval, and reconstruction must remain conceptually separate layers.
Details:
- Early prototypes may temporarily combine layers, but durable architecture should separate them.
Why it matters:
- This keeps the system testable, inspectable, and extensible.
Source:
- system design discussion
Revision policy:
- revise only with explicit architectural justification

### [MEM-20260307-SEMANTIC-OVER-STRING]
Type: decision
Status: active
Scope: architecture
Summary: Canonical operations should act on structured meaning objects, not raw strings.
Details:
- Raw text may be retained as evidence or debugging context.
- Core reasoning and storage should target structured representations.
Why it matters:
- This is the core principle behind the repository's value proposition.
Source:
- semantic IR design
Revision policy:
- do not revise casually

### [MEM-20260307-USER-COLLAB-STYLE]
Type: preference
Status: active
Scope: user
Summary: The user prefers direct, structured, technically serious collaboration with explicit trade-offs and minimal fluff.
Details:
- Strong conceptual framing is useful.
- Shallow boilerplate should be avoided.
Why it matters:
- It materially affects how plans, code explanations, and design proposals should be delivered.
Source:
- repeated interaction pattern
Revision policy:
- update only if user preference clearly changes

### [MEM-20260307-CONTEXT-COMPRESSION-RULE]
Type: decision
Status: active
Scope: workflow
Summary: At the end of meaningful work cycles, transient context should be compressed into durable memory and frontier state rather than carried forward indefinitely.
Details:
- Broad session context should not be treated as a stable source of truth.
- Durable memory and explicit frontier state should support re-entry.
- Context accumulation must not replace structure.
Why it matters:
- This prevents drift, repetition, stale assumptions, and reasoning bloat across long-running work.
Source:
- heartbeat / workflow design discussion
Revision policy:
- revise only if the operating workflow changes substantially

### [MEM-20260307-MEMORY-FRONTIER-SEPARATION]
Type: decision
Status: active
Scope: workflow
Summary: Durable memory and current frontier state must be stored separately, with `MEMORY.md` holding stable truth and `FRONTIER.md` holding current compressed state and re-entry instructions.
Details:
- `MEMORY.md` should remain small, durable, and high-signal.
- `FRONTIER.md` may change often and should capture active continuation state.
Why it matters:
- This keeps long-term memory clean while enabling precise restart behavior.
Source:
- workflow refinement
Revision policy:
- revise only if repository operating model changes

### [MEM-20260307-PYTHON-STACK]
Type: decision
Status: active
Scope: architecture
Summary: The implementation uses Python 3.11+ with standard library only (no ML/NLP dependencies in core). pytest for testing.
Details:
- Project structure: `src/ailang_ir/` with `models`, `normalize`, `parser`, `encoder`, `decoder`, `memory` modules.
- All domain models use dataclasses and enums — no Pydantic dependency.
- Parser is rule-based (deterministic, no ML). This is intentional for the MVP to avoid opaque model coupling.
Why it matters:
- Future contributors should not add heavy dependencies without justification.
- The rule-based parser is a deliberate choice, not a gap. ML-backed parsing is a future option, not a current requirement.
Source:
- initial implementation cycle
Revision policy:
- revise if a justified need for external dependencies arises (e.g. lightweight NLP for noun-phrase chunking)

### [MEM-20260307-COMPACT-CODE-FORMAT]
Type: decision
Status: active
Scope: architecture
Summary: Compact symbolic codes use pipe-delimited format: `SPEAKER|MODE|ACT|OBJECT[|OVER:TARGET]|CERTAINTY|TIME`.
Details:
- Speaker codes: U/S/A/T/?
- Mode codes: AS/OP/HY/REQ/CMD/Q/CMT/OBS/REF
- Certainty: C0–C100 (integer percent)
- Time: PAST/NOW/FUT/ATEMP/T?
- OVER: prefix for comparison targets
Why it matters:
- This format must remain stable for deduplication and cross-session comparison to work.
- Any change to the codebook format is a breaking change that requires migration.
Source:
- encoder implementation, aligned with project plan examples
Revision policy:
- treat as semi-stable; version the codebook if format changes

---

## Memory Maintenance Rules

When adding a new memory:
- prefer one durable insight over multiple weak notes
- write for future continuity
- keep summaries tight
- keep details concrete
- avoid decorative language
- prefer stable truth over narrative recap

When revising memory:
- preserve history through explicit supersession
- avoid ambiguity about what is currently active
- do not let contradictory active memories remain unresolved

When uncertain whether something belongs here:
- ask whether it would still matter after several sessions
- if not, leave it out