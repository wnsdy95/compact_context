# BOOTSTRAP.md
## First-Run Onboarding Ritual

When entering this repository for the first time in a session, perform the following sequence.

Do not begin implementation before this sequence is complete.

---

## Step 1. Re-anchor system and identity
Read:
- SYSTEM_PROMPT.md
- SOUL.md
- IDENTITY.md
- USER.md

Internalize:
- who you are
- who you are serving
- how you are expected to behave

---

## Step 2. Re-anchor repository rules
Read:
- AGENTS.md
- TOOLS.md
- MEMORY.md
- FRONTIER.md

Understand:
- project direction
- repository constraints
- prior durable decisions
- allowed tool behavior

---

## Step 3. Inspect repository reality
Before making plans, inspect the actual repository.

Identify:
- main directories
- package/dependency files
- test setup
- current architecture
- entry points
- existing documentation
- pending technical debt signals

Never assume the repository matches documentation perfectly.

---

## Step 4. Build a working model
Form an internal model of:
- what exists
- what is missing
- what is stable
- what is experimental
- where the requested task belongs

---

## Step 5. Frame the task
For the current user request, identify:

- desired outcome
- scope boundary
- relevant modules
- likely risks
- unknowns that matter
- smallest coherent implementation slice

---

## Step 6. Check for collision with memory
Before changing architecture or conventions, compare the proposed work against MEMORY.md, FRONTIER.md and existing repository patterns.

If conflict exists:
- preserve current direction unless there is strong reason to revise
- explain the reason for revision clearly

---

## Step 7. Execute deliberately
Only then:
- propose a plan
- make changes
- explain trade-offs
- record durable decisions if needed

---

## First-Run Output Discipline

At the beginning of a session, the agent should be able to answer internally:

- What is this project trying to become?
- What architectural layer am I touching?
- What prior decisions constrain me?
- What is the smallest trustworthy next move?

If these answers are weak, inspect more before building.

---

## Bootstrap Oath

"I will not confuse prior intention with current reality.
I will inspect, model, align, and then act."