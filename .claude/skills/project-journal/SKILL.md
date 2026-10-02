---
name: project-journal
description: Keep the Auto Thinker Engine project journal up to date - log milestones, blockers, bugs, design decisions and run results as they happen in reports/journey/journal.md, grow the interview question bank, and build the end-of-project summary. Use after any stage gate, failed or surprising run, bug fix, workaround, or design decision, and whenever the user asks to log progress, record a hiccup, summarise the journey, or prepare interview questions.
---

# Project journal

The journal is the source for two things the user will make at the end: a
summary of how the project was built, and the questions an interviewer could
ask about it (with answers backed by what actually happened). Write entries as
events happen, while the evidence is fresh. Never rewrite history: correct an
old entry by adding a new one that references it.

## Files

- `reports/journey/journal.md` - append-only log, newest at the bottom.
- `reports/journey/interview_questions.md` - question bank, grouped by theme.
- `reports/journey/summary.md` - written only when the user asks for the summary.

## When to add an entry

Add one entry per event, in the same turn the event is resolved (or, for an
open blocker, when it is found):

- **milestone** - a plan stage or gate passed (smoke test, integration run, learning gate).
- **result** - a training/eval run finished or was stopped; include the numbers.
- **blocker** - something stopped progress (environment, hardware, download, data).
- **bug** - wrong behaviour found in code or data, with how it was detected.
- **decision** - a design or experimental choice between alternatives.
- **mistake** - an error made during the work (by Claude or the user) and its recovery.

Small edits, routine test runs and status checks do not need entries.

## Entry format

```markdown
## YYYY-MM-DD HH:MM - <short title>
**Type:** milestone | result | blocker | bug | decision | mistake
**Stage:** <plan stage, e.g. Stage 3 - first working car>

**What happened:** symptom or event, with the evidence: exact error text,
metric values, file:line.
**Cause:** the root cause once known (write "open" if not yet known).
**How we handled it:** what was changed or decided, and why this option over
the alternatives. Name the files.
**Verified by:** the test, run or measurement that shows it worked.
**Lesson:** one sentence someone could reuse.
**Interview angle:** the question this entry answers (also add it to the bank).
```

Rules:

- Use real timestamps (run `date '+%Y-%m-%d %H:%M'`).
- Numbers over adjectives: "eval return 44 -> 165, success 0/80 episodes",
  not "improved a lot".
- Record failures and dead ends as carefully as successes; they make the best
  interview material.
- Keep each entry under ~15 lines; link to logs/files instead of pasting them.
- Do not record secrets, tokens, or personal data.

## Interview question bank

After each entry with an interview angle, add or extend a question in
`reports/journey/interview_questions.md` under the right theme, as:

```markdown
- **Q:** <question an interviewer might ask>
  **A (short):** <2-3 sentence answer grounded in the project>
  **Evidence:** journal entries <dates/titles>, files
```

Themes: Problem & architecture, Reinforcement learning / world models,
Self-supervised learning (JEPA), Simulation & infrastructure, Debugging stories,
Experiment design & evaluation, Trade-offs & what I would do differently.

## Building the summary (only when asked)

1. Read the whole journal and the question bank.
2. Write `reports/journey/summary.md` with: goal and architecture (one diagram
   or table), timeline by stage, the 5 hardest problems (symptom -> cause ->
   fix -> lesson), final results with numbers, open issues, and what to do next.
3. Keep claims traceable to journal entries; mark anything unverified.
