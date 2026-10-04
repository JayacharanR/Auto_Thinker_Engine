# Agent handoff: Auto Thinker Engine

You are joining an ongoing research/engineering project. Read this whole brief,
then the files listed in "Reading order", before changing anything. Status
below is as of 2026-10-04 19:05; check the live state (section "First commands")
because training may have moved on.

## 1. What the project is trying to achieve

A self-driving agent in the CARLA simulator, and a controlled experiment on
**whether self-supervised video pretraining helps reinforcement learning drive**.

- **Agent:** DreamerV3 (PyTorch port `NM512/dreamerv3-torch`): an RSSM world model
  plus an actor-critic trained in imagination. Tasks and rewards come from
  CarDreamer (CARLA 0.9.15 task suite).
- **Research question (Phase 3 / "Stage 5"):** three image encoders, everything
  else held identical:
  - `cnn`: dreamerv3-torch's CNN, trained from scratch by RL;
  - `custom_jepa`: our own ViT-S JEPA, pretrained self-supervised on real dash-cam
    video (comma2k19), then frozen;
  - `vjepa2`: Meta's V-JEPA2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), frozen.
  Frozen encoders run once per environment step at collection time; replay stores
  their embedding (`feat`), so Dreamer updates never run the ViT.
- **Observation for the comparison:** front RGB camera (64x64 for the agent) plus
  a 23-d route vector (10 ego-frame waypoints every ~3.2 m, speed, heading error).
  The bird's-eye view (BEV) was only used in Stage 3 to prove the loop learns.
- **Actions:** CarDreamer's discrete table (3 accelerations x 5 steerings, one-hot
  actor). **Reward:** CarDreamer's waypoint reward plus a time penalty of 0.1 per
  step (CarDreamer default 0; without it the agent learned to stop mid-turn).

## 2. Hardware and environment (laptop, no Docker)

- RTX 5070 Ti Laptop GPU, 12 GB, Blackwell sm_120; Core Ultra 9 275HX (24 cores);
  30 GB RAM; CachyOS (Arch); user shell is **fish**; tool shell is zsh.
- Python 3.10 via **uv** (`.venv`), torch 2.11 + cu128 (needed for sm_120).
- CARLA 0.9.15 at `~/software/CARLA_0.9.15` (`CARLA_ROOT`), headless
  (`-RenderOffScreen -quality-level=Low`), started and cleaned up by
  `jobs/slurm_run.sh` (works without Slurm).
- Submodules `third_party/CarDreamer` and `third_party/dreamerv3_torch` are patched
  by tracked patches in `patches/` (applied by `scripts/setup_cardreamer.sh`).
  Never commit submodule internals; regenerate the patch files instead.
- `scripts/configure_carla.sh` sets Town03_Opt as startup map and **renames
  CARLA's Traffic Manager map cache (`Maps/TM` -> `TM.disabled`)** - that cache
  segfaults the 0.9.15 Python client; without it traffic tasks work.

## 3. Project history in one paragraph

Moved from a remote A4000 Docker server (never completed a Dreamer update) to this
laptop. A staged plan (`~/.claude/plans/robust-drifting-galaxy.md`) was executed:
env fixes (torch for Blackwell; a patch bug that froze all training after the
first update), CARLA bring-up, Stage 3 "first working car" (BEV + CNN learned the
right turn only after adding the time penalty; 20/20 evaluation episodes), Phase 2
JEPA pretraining on comma2k19 (EMA beat SIGReg on a speed probe; steering is not
linearly decodable on this data), Stage 5 encoder comparison, robustness
evaluation, and now traffic. Many infrastructure fixes along the way: crash-safe
job queues, atomic checkpoints, CARLA segfault recovery, fixed eval schedule,
early stopping. The full story with numbers is in `reports/journey/journal.md`.

## 4. Results so far (all in outputs/, which is git-ignored)

**Right-turn encoder comparison** (`carla_right_turn_simple`, camera + route,
seeds 42 and 123; `outputs/comparison/carla_right_turn_simple_camera_route/`
`comparison.md`, `comparison_curves.png`):

| | cnn | custom_jepa | vjepa2 |
|---|---|---|---|
| Seeds solved within budget | 1/2 | 2/2 | 2/2 |
| Steps to 80% eval success | 32.5k / never | 30.0k +- 2.5k | 21.7k +- 9.2k |

**Zero-shot robustness** of those 6 agents (`outputs/robustness/robustness.md`,
videos in `outputs/robustness/<arm>_seed<n>/<condition>/videos/`): success at
night cnn 0% vs custom_jepa 100% vs vjepa2 100%; heavy rain 45/95/100%; sunset
50/100/100%; dense cross traffic 20/35/20% (55-80% collisions).

**Traffic curriculum (in progress):** each arm's seed-42 right-turn agent is
fine-tuned on `carla_right_turn_hard` (dense cross traffic), 50k steps max, evals
every 5k, early stop after 3 evals >= 90%. cnn finished 50k without learning
(eval success oscillated 0-70%, ended 30%); custom_jepa was running at the time
of writing, vjepa2 queued. Outputs:
`outputs/comparison/carla_right_turn_hard_finetune/`, queue `outputs/traffic_queue/`.

## 5. Roadmap agreed with the user

1. Robustness evaluation - DONE.
2. Traffic (`carla_right_turn_hard`) - IN PROGRESS. If all arms stay flat,
   consider a gentler curriculum (`right_turn_medium` first), a longer budget or a
   stronger collision penalty - discuss with the user first.
3. Third seed (456) for the right-turn comparison - DEFERRED by the user; needed
   before presenting that result as firm. Command is in the journal entry
   "Roadmap agreed with the user".
4. Traffic rules: `carla_traffic_lights` / `carla_stop_sign` (CarDreamer built-ins).
5. Pedestrians: needs a custom CarDreamer task (not built in).

## 6. Rules and conventions (follow strictly)

- `CLAUDE.md` (project instructions): keep `reports/journey/journal.md` current
  via the `project-journal` skill - add an entry in the same turn whenever a gate
  passes/fails, a run finishes or is stopped, a bug/blocker is found or fixed, a
  decision is made, or a mistake is recovered from; add matching questions to
  `reports/journey/interview_questions.md`. Use the `training-progress` skill when
  the user asks about progress.
- **Never edit a job script (`jobs/*.sh`) while it runs** - bash reads scripts
  incrementally. Check with the anchored pattern below first.
- **Never start a queue that is already running.** Queues now take an flock
  (`<queue dir>/.queue.lock`) and refuse a second copy, but check anyway; a
  duplicate once jammed a run (journal "Mistake: a second queue copy").
- **Find running jobs with anchored patterns only** - plain `pgrep -f <text>`
  matches your own shell:
  `pgrep -af '^bash jobs/|^[^ ]*python[0-9.]* [^ ]*scripts/(train_|evaluate_|probe_)|CarlaUE4-Linux' | grep -v defunct`
- **Stop a queue cleanly with `touch <queue dir>/STOP`** (trainers save a
  checkpoint on SIGTERM; never `kill -9` trainers). Restart by rerunning the same
  command: runs resume from `latest.pt`.
- Launch long jobs detached with sleep blocked:
  `setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why=training bash jobs/<job>.sh > <log> 2>&1 &`
- Git: commit on branch `laptop-long-runs`, then fast-forward `main` without a
  checkout that changes files (`git branch -f main laptop-long-runs && git switch main`).
  This machine has **no GitHub credentials**; the user pushes (`git push origin main`).
  End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 uv run pytest tests/ -q`
  (132 pass + 1 GPU-only). Limit CPU threads when testing next to a training job.
- Lint: `.venv/bin/ruff check` (pre-existing E402 in scripts are accepted).
- Ask before multi-hour runs or reward/benchmark changes; the user decides scope.

## 7. Reading order

Start here (context and history):
1. `CLAUDE.md` - project rules.
2. `reports/agent_handoff.md` - this file.
3. `reports/journey/journal.md` - chronological log of every milestone, bug,
   decision and mistake with numbers. Read fully; newest entries are at the bottom.
4. `reports/journey/interview_questions.md` - condensed Q&A view of the results.
5. `~/.claude/plans/robust-drifting-galaxy.md` - the original staged plan.
6. `configs/experiment_contract.yaml` - the experiment contract as actually run.
7. `README.md` - setup, laptop runbook, how to run each phase.

Core code:
8. `scripts/train_cardreamer.py` - Dreamer runner: `train_arm`, `build_agent`,
   `make_carla_env`, `task_argv` (validated `--env-set`), checkpointing, fixed eval
   schedule, early stopping, `--init-from`, comparison aggregation
   (`run_comparison`, `summarize_run`, `plot_learning_curves`).
9. `src/dreamer/carla_wrappers.py` - observation contracts (bev / camera /
   camera_route), route features, terminal vs time-out handling, per-episode
   metrics incl. route completion.
10. `src/dreamer/cardreamer_encoder_hook.py` - frozen feature extractor (built from
    the Phase 2 checkpoint config) and `PrecomputedFeatureEncoder`.
11. `src/dreamer/encoder_adapter.py` - encoders/adapter, `freeze_parameters`.
12. `configs/laptop.yaml` (Dreamer profile), `configs/phase3_transfer_arms.yaml`
    (arms; `custom_jepa` checkpoint = `outputs/checkpoints/phase2/best.pt` = EMA).
13. `scripts/evaluate_agent.py` - evaluation with front-camera + BEV videos,
    `--env-set` conditions (e.g. `world.weather=ClearNight`).

Jobs and tools:
14. `jobs/lib_queue.sh` (status log, health monitor, STOP file, lock, retries),
    `jobs/comparison.sh`, `jobs/robustness.sh`, `jobs/roadmap_1_2.sh`,
    `jobs/long_runs.sh`, `jobs/slurm_run.sh`.
15. `scripts/watch_queue.py`, `scripts/watch_run.py` (progress; `--once` for a
    snapshot), `scripts/inspect_episode.py` (frames + decoded actions),
    `scripts/robustness_table.py`.

Phase 2 (JEPA), if needed:
16. `scripts/train_phase2_jepa.py`, `src/jepa/{encoder,predictor,masking,sigreg}.py`,
    `src/data/comma2k19_dataset.py`, `scripts/preprocess_comma2k19.py`,
    `scripts/probe_diagnostics.py`, `src/eval/linear_probe.py`,
    `configs/phase2_jepa_laptop.yaml`.

Patches and tests:
17. `patches/cardreamer_compat.patch`, `patches/dreamerv3_torch_compat.patch`.
18. `tests/` - especially `test_train_loop.py` (end-to-end runner on a fake
    CarDreamer task: checkpoints, crash recovery, eval schedule, early stop,
    init-from) and `test_carla_wrappers.py`.

## 8. First commands

```bash
cd ~/Project/Auto_Thinker_Engine
git status -sb && git log --oneline -5
pgrep -af '^bash jobs/|^[^ ]*python[0-9.]* [^ ]*scripts/(train_|evaluate_|probe_)|CarlaUE4-Linux' | grep -v defunct
ls -t outputs/*/status.txt | head -3
python3 scripts/watch_queue.py --out outputs/traffic_queue --once
cat outputs/comparison/carla_right_turn_simple_camera_route/comparison.md
cat outputs/robustness/robustness.md
```

## 9. Known open issues

- Traffic task: cnn did not learn in 50k fine-tuning steps; await custom_jepa and
  vjepa2 before changing the setup.
- CARLA 0.9.15 segfaults roughly every 1-1.5 h of simulation (on resets); queues
  retry automatically and resume from checkpoints - expect RETRY lines.
- Seed 42 vjepa2 right-turn run stopped at 66.7k and is summarised as partial;
  its early evals were taken before the eval schedule was fixed.
- Heavy rain at `-quality-level=Low` renders as an extreme grey wash; treat rain
  results as a strong distribution shift, not realistic rain.
