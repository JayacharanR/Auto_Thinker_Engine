# Auto Thinker Engine

Staged plan: `~/.claude/plans/robust-drifting-galaxy.md`.

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
