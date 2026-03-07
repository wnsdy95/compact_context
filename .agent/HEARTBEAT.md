# HEARTBEAT.md
## Continuous Improvement and Re-Entry Protocol

This file is not only for periodic self-check.
It governs how work cycles end, compress, and restart.

The agent must not accumulate uncontrolled session context indefinitely.
At the end of meaningful work cycles, it must compress state, preserve durable memory, discard excess context, and re-enter from the true unresolved frontier.

---

## Core Principle

Do not drag the full conversational or implementation context forever.

When a work cycle reaches a meaningful stopping point:
1. extract durable knowledge
2. summarize the current state
3. record unresolved bottlenecks
4. define the next frontier
5. treat the next cycle as a fresh re-entry from compressed memory and frontier state

The goal is not to remember everything.
The goal is to preserve what must survive and remove what creates drift.

---

## When a Cycle Ends

A cycle should be considered complete when one of the following is true:

- a meaningful implementation slice is finished
- a planning / analysis phase has produced a stable conclusion
- the current scope boundary has been reached
- further progress would require a new decision or a fresh pass
- context is becoming noisy, repetitive, or less trustworthy

When this happens, do not continue mindlessly.

Perform cycle compression.

---

## End-of-Cycle Compression Protocol

At the end of a cycle, the agent must produce a compressed state consisting of:

### 1. Durable memory candidates
What new information should persist across future sessions?

Only include:
- stable goals
- architectural decisions
- constraints
- naming conventions that matter
- user collaboration preferences that affect output
- important rejected alternatives
- critical unresolved questions

Store these in `MEMORY.md` only if they are truly durable across future cycles.

### 2. Current state summary
Summarize the current technical state in compact form:
- what was completed
- what remains incomplete
- what assumptions now hold
- what changed from prior understanding

This summary must be concise and operational.

Store this compressed working state in `FRONTIER.md`, not in `MEMORY.md`, unless part of it has durable long-term value.

### 3. Problem frontier
Identify the exact next unresolved problem.

Do not say "continue improving."
Say exactly where re-entry should begin.

Good examples:
- "Parser currently extracts speaker/act/object, but time normalization is inconsistent."
- "Compression layer exists, but codebook collisions are unresolved."
- "SemanticFrame schema is stable enough, next step is decoder round-trip testing."

Record this frontier explicitly in `FRONTIER.md`.

### 4. Risk / uncertainty summary
Record what remains uncertain:
- unvalidated assumption
- ambiguous design branch
- likely failure mode
- missing test coverage
- unresolved repository fit concern

Store active restart-relevant risks in `FRONTIER.md`.
Store only long-lived strategic warnings in `MEMORY.md`.

### 5. Recommended next action
State the smallest trustworthy next move.

This next action must be concrete enough that a fresh cycle can resume from it without depending on the full prior session context.

Store the immediate next action in `FRONTIER.md`.

---

## Context Reset Principle

After end-of-cycle compression, the agent should behave as if broad transient context has been discarded.

It should not depend on carrying the entire prior session forward.
It should depend on:
- repository reality
- durable memory in `MEMORY.md`
- compressed current state in `FRONTIER.md`
- explicit next frontier

This prevents drift, repetition, and context bloat.

---

## Re-Entry Protocol

When beginning a fresh cycle after compression:

1. read `CLAUDE.md`
2. read `SYSTEM_PROMPT.md`
3. run `BOOTSTRAP.md`
4. read `MEMORY.md`
5. read `FRONTIER.md`
6. inspect repository reality again if implementation is involved
7. locate the saved frontier
8. restart from the frontier, not from the entire previous conversation

The next cycle should begin from the highest-value unresolved point, not from historical inertia.

---

## Ongoing Heartbeat Checks

At meaningful intervals, ask:

### Goal
- Am I still solving the real problem?

### Layer
- Which architectural layer am I touching?

### Scope
- Am I expanding beyond the requested slice?

### Assumptions
- What is confirmed vs inferred?

### Drift
- Is context accumulation making reasoning worse?

### Compression readiness
- If I had to end this cycle now, what must be preserved durably in `MEMORY.md`?
- What belongs only to `FRONTIER.md` as compressed current state?
- What can be safely discarded?
- What is the exact frontier for restart?

---

## Drift and Bloat Warning Signs

If any of the following are true, end the cycle and compress:

- the same context is being re-read without new value
- reasoning is becoming repetitive
- the implementation path is no longer crisp
- too many temporary assumptions are active
- prior conversation is carrying more weight than repository reality
- the next step is becoming less clear instead of more clear

---

## Memory Discipline

Do not dump raw session history into `MEMORY.md`.

Instead:
- write durable cross-session truth to `MEMORY.md`
- write compressed current state and exact re-entry point to `FRONTIER.md`

Memory must remain useful under re-entry.
Frontier must remain sharp enough to restart from.

Do not preserve entire session history.
Preserve only what improves future correctness and continuity.

---

## End-of-Cycle Output Pattern

Before stopping a cycle, produce internally or explicitly:

1. Completed:
   - ...

2. Durable updates for MEMORY.md:
   - ...

3. Compressed current state for FRONTIER.md:
   - ...

4. Current frontier:
   - ...

5. Risks / unknowns:
   - ...

6. Recommended next action:
   - ...

If this cannot be stated clearly, the cycle is not yet well-compressed.

---

## Governing Rule

Never let accumulated context become a substitute for structure.

Compress.
Preserve what matters.
Discard what drifts.
Restart from the frontier.

At the end of a meaningful cycle:
- update `MEMORY.md` with durable truths only
- update `FRONTIER.md` with compressed working state and exact re-entry point
- do not rely on broad transient context for future continuation