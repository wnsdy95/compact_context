# AILang-IR Project Plan

## Overview

AILang-IR is an AI-oriented intermediate representation system for converting natural language into structured semantic forms that are easier to store, retrieve, reason over, compress, and reconstruct.

This project is **not** a general-purpose programming language runtime. It is a semantic memory and reasoning layer designed for AI systems that need durable context handling and high-fidelity meaning preservation.

---

## Core Goal

Build a system that transforms natural language into an internal semantic representation optimized for:

- compact storage
- semantic retrieval
- relation-aware reasoning
- controllable reconstruction back into natural language
- long-context compression without uncontrolled drift

---

## Problem Statement

Raw natural language is expensive to preserve as long-term AI context.

Natural language has several structural weaknesses for memory systems:

- the same meaning can appear in many surface forms
- the same sentence can mean different things in different contexts
- raw transcripts are noisy and expensive to keep indefinitely
- long conversations accumulate stale assumptions and drift
- direct string-level storage makes semantic retrieval and conflict detection difficult

The project therefore aims to treat **meaning**, not raw text, as the canonical unit of operation.

---

## Product Direction

The system should:

1. ingest natural language
2. normalize common semantic patterns
3. extract structured semantic frames
4. optionally map those frames into compact symbolic codes
5. store relationships in memory-friendly structures
6. retrieve and reason over the stored meaning
7. reconstruct useful natural language when needed

---

## High-Level Architecture

The system should evolve in clearly separated layers:

1. **Natural Language Ingestion**  
   Accept raw user text or message input.

2. **Normalization**  
   Normalize repeated surface expressions into stable semantic categories.

3. **Semantic Parsing**  
   Extract core fields such as speaker, act, object, time, modality, certainty, and sentiment.

4. **Semantic Frame Construction**  
   Convert extracted meaning into a typed internal schema.

5. **Compact Symbolic Encoding**  
   Represent frequent semantic structures in shorter deterministic codes.

6. **Memory Storage**  
   Store semantic objects and relations for future retrieval and reasoning.

7. **Retrieval / Conflict Resolution**  
   Find relevant prior meaning, merge repeated intent, and detect contradictions.

8. **Natural Language Reconstruction**  
   Rebuild readable language from semantic state when needed.

---

## Core Design Principles

### 1. Meaning over strings
Canonical operations should target structured meaning objects, not raw strings.

### 2. Event over sentence
The primary storage unit should be an event or semantic frame, not a raw sentence.

### 3. Layer separation
Parsing, normalization, compression, storage, retrieval, and reconstruction should remain conceptually distinct.

### 4. Inspectability
Every transformation stage should be debuggable and visible.

### 5. Compression with recoverability
Compression is valuable only if important meaning can still be reconstructed usefully.

### 6. Durable memory vs current frontier
Long-lived truths and current working state must remain separate.

---

## Semantic Representation Model

A natural-language sentence should not be stored as-is. It should first be transformed into a semantic structure such as:

```json
{
  "speaker": "user",
  "mode": "opinion",
  "act": "believe",
  "object": "semantic_frame_model",
  "time": "present",
  "certainty": 0.84,
  "sentiment": "neutral",
  "priority": "P1"
}
```

This frame may later be mapped into a more compact representation.

---

## Example Semantic Compression

Natural language:

> I think one-to-one sentence mapping will be difficult.

Semantic frame:

```json
{
  "speaker": "user",
  "mode": "hypothesis",
  "act": "believe",
  "object": "sentence_1to1_mapping_hard",
  "time": "present",
  "certainty": 0.84
}
```

Compact symbolic form:

```text
U|HY|BELIEVE|SENTENCE_1TO1_MAPPING_HARD|C84|NOW
```

---

## Domain Model Direction

The first stable domain model should likely include the following core types:

- `Entity`
- `Predicate`
- `SemanticFrame`
- `EventMemory`
- `RelationEdge`
- `CompressionRule`
- `ReconstructionPlan`

These should be explicit and typed rather than loose nested dictionaries.

---

## MVP Scope

The first meaningful MVP should support the following:

### MVP-1: Semantic Core
- define typed domain models
- define base semantic fields
- define normalization vocabulary

### MVP-2: Parser MVP
- convert a single sentence into a partial semantic frame
- extract at least: speaker, act, object, time
- allow unknown or null fields when confidence is low

### MVP-3: Encoder MVP
- map normalized semantic frames into deterministic symbolic codes
- define an initial codebook for common acts and concepts

### MVP-4: Decoder MVP
- reconstruct approximate natural language from semantic frames or compact codes
- prioritize meaning preservation over stylistic richness

### MVP-5: Memory MVP
- store event memories with metadata
- support basic retrieval by entity, predicate, or topic
- support repeated-meaning merging where useful

---

## Suggested Development Order

1. define the domain model
2. define normalization vocabulary
3. implement semantic frame schema
4. implement parser MVP
5. implement encoder MVP
6. implement decoder MVP
7. implement memory store abstraction
8. implement retrieval and conflict checks
9. add fixtures and tests
10. improve compression and conflict resolution

---

## Example Working Fields

The initial semantic frame can use fields like:

- `speaker`
- `mode`
- `act`
- `object`
- `target`
- `time`
- `certainty`
- `sentiment`
- `priority`
- `source_text`
- `metadata`

This is intentionally compact. The schema can expand later if real usage justifies it.

---

## Storage Philosophy

The system should avoid treating full conversation history as the stable source of truth.

Instead:

- preserve durable architectural truths in `MEMORY.md`
- preserve current compressed working state in `FRONTIER.md`
- discard broad transient context once meaningful compression is complete

This keeps the system resistant to drift and context bloat.

---

## Durable Memory Rule

The project should preserve only what materially improves future correctness and continuity.

Good durable memory examples:

- project direction
- architecture rules
- domain-model commitments
- strong user collaboration preferences
- critical strategic questions

Bad durable memory examples:

- raw transcripts
- repeated status logs
- temporary TODO clutter
- decorative summaries

---

## Frontier Rule

The project should always maintain an explicit active frontier.

A good frontier states:

- what was completed
- what remains incomplete
- the exact unresolved bottleneck
- risks / unknowns
- the smallest trustworthy next action

This allows fresh re-entry without dragging the whole session context forward.

---

## Example User Input Transformations

### Example 1
Natural language:

> Natural language should be segmented into meaning units.

Compact form:

```text
U|OP|SUGGEST|SEMANTIC_SEGMENTATION|C88|NOW
```

### Example 2
Natural language:

> The system should be able to reconstruct natural language later.

Compact form:

```text
U|REQ|NEED|NATURAL_LANGUAGE_RECONSTRUCTION|C91|FUT
```

### Example 3
Natural language:

> Graph memory seems more appropriate than linear text storage.

Compact form:

```text
U|OP|PREFER|GRAPH_MEMORY|OVER:LINEAR_TEXT|C83|NOW
```

---

## Major Risks

### 1. Over-compression
Too much compression may destroy reconstructable meaning.

### 2. Blurry boundaries
If parsing, normalization, and encoding are mixed together, the system becomes fragile.

### 3. False certainty
The parser must not pretend to know more than it actually knows.

### 4. Codebook drift
A symbolic codebook can become messy if it is not versioned or governed.

### 5. Memory pollution
If durable memory absorbs temporary work-state notes, future continuity degrades.

### 6. Premature architecture
The system should not over-design for hypothetical scale before the real semantic core stabilizes.

---

## Evaluation Questions

The project should eventually answer:

- Does semantic storage outperform raw text retrieval for the intended use case?
- Does compact encoding preserve enough meaning for reconstruction?
- Can repeated meanings across different phrasings be merged reliably?
- Can the system identify contradictions across time?
- Does the frontier-driven workflow reduce context drift in long sessions?

---

## Initial Success Criteria

The project can be considered directionally successful when:

- a sentence can be turned into a semantic frame reliably enough for simple cases
- the same meaning expressed differently maps to compatible internal structures
- symbolic encoding is deterministic
- reconstructed language preserves the intended meaning reasonably well
- active work can be restarted from compressed memory and frontier state

---

## Out of Scope for Early Phases

The following should not be treated as first-phase requirements:

- full natural language understanding
- perfect one-to-one reversibility
- heavy binary optimization
- broad multilingual support
- deep autonomous agent orchestration
- speculative infrastructure for massive scale

These may matter later, but they should not distract from semantic correctness in the MVP.

---

## Immediate Next Steps

1. finalize root governance files and operating workflow
2. define the first version of the semantic domain model
3. choose repository structure for semantic core modules
4. implement parser MVP on a narrow sentence set
5. create fixtures and evaluation examples
6. add deterministic encoding rules
7. validate reconstruction quality on simple examples

---

## One-Sentence Project Definition

**AILang-IR is a semantic intermediate representation system that converts natural language into structured, compressible, retrievable meaning for AI memory, reasoning, and reconstruction.**

