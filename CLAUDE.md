# Auto Thinker Engine

Staged plan: `~/.claude/plans/robust-drifting-galaxy.md` (historical Dreamer work).

**Current work (branch `experiment/camera-world-model-navigation`): always follow
`reports/navigation_review_plan.md`** (full user-approved plan, M0–M6) with its
condensed form `reports/navigation_plan.md`, the runbook
`reports/navigation_runbook.md`, and the latest status in
`reports/navigation_m0_assessment_20261008.md` and the journal tail. Ask before
multi-hour collection/training; Town05 stays held out; never present the
privileged PDM-Lite expert as camera driving.

## Project journal

Keep `reports/journey/journal.md` current with the `project-journal` skill
(`.claude/skills/project-journal/SKILL.md`): add an entry whenever a stage gate
passes or fails, a run finishes or is stopped, a blocker or bug is found or
fixed, a design decision is made, or a mistake is recovered from. Add matching
questions to `reports/journey/interview_questions.md`. Do this in the same
turn as the event, without waiting to be asked.

## Training progress

When the user asks how training is going (or after launching a long run), use
the `training-progress` skill (`.claude/skills/training-progress/SKILL.md`):
show a `--once` snapshot in the chat and give the live-bar command for their
terminal.
