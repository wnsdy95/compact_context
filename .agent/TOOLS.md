# TOOLS.md
## External Tool Usage Guide

This file governs how tools, libraries, APIs, and external services should be selected and used.

---

## Tool Philosophy

A tool is justified only when it meaningfully reduces complexity, risk, or development time without introducing disproportionate lock-in, opacity, or fragility.

Do not add tools casually.

---

## Selection Rules

Before introducing a tool, evaluate:

### 1. Necessity
What concrete problem does it solve?

### 2. Surface area
How much complexity does it add to the repository?

### 3. Lock-in
How difficult would it be to replace later?

### 4. Observability
Can failures be understood and debugged?

### 5. Cost
What are the runtime, maintenance, and cognitive costs?

### 6. Security
Does it introduce secrets, external data exposure, or unsafe execution paths?

---

## Preferred Order of Choice

Prefer, in general:

1. standard library
2. small well-maintained dependency
3. proven ecosystem tool
4. external hosted service
5. custom infrastructure

Do not build custom infrastructure when a stable and modest dependency is enough.
Do not add heavy dependencies when the standard library is sufficient.

---

## AI / LLM Tooling Rules

When using AI models, prompts, or external inference providers:

- separate prompt templates from business logic
- log model inputs/outputs safely in development when useful
- avoid coupling system truth to a single opaque model response
- support validation or post-processing where correctness matters
- treat model output as untrusted until validated

For semantic systems:
- prefer explicit normalization after model output
- preserve source evidence
- keep reconstruction separate from parsing where possible

---

## Secrets and Credentials

Never hardcode:
- API keys
- tokens
- passwords
- private URLs
- environment-specific secrets

Use environment variables or secret management.

All new secrets must be documented by name and purpose.

---

## External Calls

Any external call should have:
- timeout handling
- failure handling
- clear retry policy when appropriate
- structured error reporting where possible

Do not assume network success.

---

## Data Storage Tools

Before adopting a database or storage backend, clarify:
- what is the canonical domain object
- what is queried most often
- whether the data is append-heavy, graph-like, or document-like
- what consistency is needed

For semantic memory systems, storage choice should follow the access pattern, not hype.

---

## Experimental Tools

Experimental tools may be used only if:
- clearly isolated
- labeled as experimental
- easy to remove
- not deeply entangled with the stable core

---

## Documentation Requirement

Whenever a significant new tool is introduced, document:
- why it was chosen
- what alternatives were rejected
- how it should be used
- what would cause it to be replaced