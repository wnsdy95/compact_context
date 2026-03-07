# AGENTS.md
## Repository Operating Guide

This file defines how the agent should work inside this repository.

---

## Primary Goal

Build an AI-oriented semantic intermediate representation system that can transform natural language into structured meaning representations for storage, retrieval, reasoning, and reconstruction.

This repository should evolve toward a system that is:
- semantically grounded
- testable
- modular
- incrementally extensible

---

## Current Project Direction

The system being built is not a general programming language.

It is an AI memory and reasoning intermediate representation, tentatively centered around:
- semantic frames
- compact symbolic encoding
- graph-oriented memory
- reconstruction into natural language

Any implementation should preserve that direction.

---

## Architectural Priorities

### 1. Separate layers clearly
Keep these concerns distinct:

- natural language ingestion
- normalization
- semantic parsing
- frame construction
- symbolic compression
- memory storage
- retrieval/query
- natural language reconstruction

Do not collapse multiple layers into one module unless the code is intentionally experimental and clearly labeled.

### 2. Design around meaning, not strings
Raw text may be retained as source evidence, but canonical operations should act on structured meaning objects.

### 3. Prefer typed domain models
Core semantic structures should be represented with explicit schemas or typed models.

Examples:
- Entity
- Predicate
- SemanticFrame
- EventMemory
- RelationEdge
- CompressionRule
- ReconstructionPlan

### 4. Build the MVP around a stable core
The first durable version should support:
- parsing a sentence into a semantic frame
- normalizing common predicates and entities
- emitting a compact symbolic form
- reconstructing approximate natural language
- storing event memories with metadata

### 5. Preserve inspectability
Every major transformation should be debuggable.

For each pipeline stage, it should be easy to inspect:
- raw input
- normalized form
- extracted semantic fields
- compact code
- reconstructed output

---

## Repository Expectations

### Code style
- prefer explicit names
- keep functions focused
- avoid giant files
- avoid silent mutation where possible
- keep core logic testable without infrastructure

### Module boundaries
Each directory should represent a real responsibility.
Do not create folders that only contain one unstable utility unless that utility is expected to grow.

### Comments
Comments should explain:
- why something exists
- why a trade-off was made
- why a case is special

Do not comment obvious syntax.

### Tests
Tests should focus on:
- semantic normalization
- frame consistency
- code generation determinism
- reconstruction sanity
- conflict resolution
- edge cases and ambiguity handling

### Logging / Debugging
Any transformation pipeline should support debug output in development mode.

---

## Preferred Development Order

1. define domain model
2. define normalization vocabulary
3. define semantic frame schema
4. implement parser adapter or rule-based extractor
5. implement compact code generator
6. implement decoder / reconstructor
7. implement memory store abstraction
8. implement retrieval and conflict resolution
9. add evaluation fixtures
10. improve compression strategy

---

## Anti-Patterns

Do not:
- entangle UI concerns with semantic core
- bury meaning in unstructured dictionaries everywhere
- overfit early design to one prompt style
- assume 1:1 sentence-to-meaning mapping
- optimize binary compression before semantic correctness
- create fake abstractions before the domain is clear

---

## Decision Records

When a major architectural decision is made, record:
- what was chosen
- what alternatives were considered
- why this was chosen
- what future cost it creates

These records should be durable.

---

## If Requirements Are Vague

When the user asks for implementation under ambiguity:
1. infer the most likely domain objective
2. state critical assumptions
3. implement the safest coherent slice
4. leave extension points explicit

Do not stall on avoidable ambiguity.

---

## Expected Deliverable Quality

All deliverables should be suitable for a serious early-stage engineering project, not a toy prototype unless explicitly requested.

That means:
- coherent naming
- realistic structure
- practical extensibility
- testability
- clear documentation

If a file rule conflicts with repository reality, inspect reality first, then update the rule intentionally.