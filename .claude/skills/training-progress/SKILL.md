---
name: training-progress
description: Show the user live progress of Auto Thinker Engine training - find what is running (job queues, Dreamer/CARLA runs, Phase 2 JEPA runs), print a progress-bar snapshot with rate, ETA, eval results and GPU health, and give the exact command for a live updating bar in their own terminal. Use whenever the user asks for progress, a progress bar, an ETA, "is anything running", or how training is going, and after launching any long run.
---

# Training progress

Training runs detached (via `jobs/*.sh` queues), so progress is read from the
files they write. The watchers below only read files; they never touch the
running jobs.

## 1. Find what is running

Use anchored patterns. `pgrep -f` with a plain substring also matches your own
shell (its command line contains the pattern) and gives false positives:

```bash
pgrep -af '^bash jobs/|^[^ ]*python[0-9.]* [^ ]*scripts/(train_|probe_)|CarlaUE4-Linux' \
  | grep -v defunct || echo "nothing running"
ls -t outputs/*/status.txt | head -3      # most recent queues
tail -3 <queue dir>/status.txt            # START / OK / FAIL / SKIP / STOP per step
```

Queues and what they write:

| Queue script | Queue dir (`--out`) | Run logs |
|---|---|---|
| `jobs/comparison.sh` | `outputs/comparison_queue` | `outputs/comparison/<task>_camera_route/<arm>_seed<n>/` |
| `jobs/long_runs.sh` | `outputs/long_runs` | Phase 2 `outputs/checkpoints/phase2_*`, Dreamer `outputs/logs/...` |
| `jobs/stage3_time_penalty.sh` | `outputs/stage3_time_penalty` | `outputs/logs/cnn_bev_tp01_seed42` |

## 2. Show a snapshot in the chat

Run with `--once` (plain text, no terminal control codes) and paste the lines
into your reply inside a code block:

```bash
python3 scripts/watch_queue.py --out <queue dir> --once        # whole queue
python3 scripts/watch_run.py <run logdir> --target <steps> --once   # one Dreamer run
```

Example:

```
cnn_seed42      [██░░░░░░░░]  21.4%   21,400/100,000 env steps   4.4 st/s  ETA 4h58m
whole queue     [█░░░░░░░░░]   7.0%  steps done 0/4  ETA ~19h40m
GPU 61°C 97% 7.9 GB 70 W | CPU +74.0°C | RAM free 20.1G | disk free 97G
```

`watch_queue.py` needs a queue dir with `status.txt` (and `steps.json` for
queues other than `long_runs`); for a run started outside a queue use
`watch_run.py` on its log directory.

## 3. Give the user the live bar

A command run from the chat (including `! <cmd>`) only shows its output when it
ends, so a live bar must run in the user's own terminal. Give the exact command
with the project path (the user's shell is fish):

```fish
cd ~/Project/Auto_Thinker_Engine; and python3 scripts/watch_queue.py --out outputs/comparison_queue
```

It redraws every 2 s and exits when the queue finishes; Ctrl-C stops watching
only, never training. Stopping training cleanly is `touch <queue dir>/STOP`.

## 4. Explain what the user sees

- First bar: the step running now (`<arm>_seed<n>`, `phase2_ema`, ...). Dreamer
  steps count env steps; the bar pauses during evaluation (eval steps do not
  count). Phase 2 steps count optimizer steps, updated each epoch and smoothed.
- `st/s` and ETA come from the run's own episode timestamps over the last
  10 min. In the first ~10 min of a Dreamer run they are too optimistic (the
  window includes the fast random-action prefill).
- `whole queue`: progress weighted by each step's measured duration
  (`steps.json`); its ETA adds the current step's ETA to the remaining steps.
- `watch_run.py` also shows `train ret(10)` (mean return of the last 10 training
  episodes) and, after each 10k-step eval, eval return, route completion,
  success and collision rates - the numbers to report.
- Health line: GPU temperature/load/memory and free RAM/disk. The queue's
  health monitor stops training cleanly at GPU >= 92 C for 90 s, RAM < 2 GB or
  disk < 5 GB; mention it if values approach those limits.

## 5. After launching a long run

Wait until the trainer reports `Training from env step ...` in
`<queue dir>/<step>.log`, then show a `--once` snapshot and the live-bar command.
