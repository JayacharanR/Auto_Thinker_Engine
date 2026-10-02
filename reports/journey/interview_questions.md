# Interview question bank

Questions an interviewer could ask about this project, with short answers
grounded in what actually happened. Grows with `journal.md`; format in
`.claude/skills/project-journal/SKILL.md`. Answers marked (pending) depend on
runs not finished yet.

## Problem & architecture

- **Q:** What does the system do, in one minute?
  **A (short):** A DreamerV3 agent (RSSM world model + actor-critic trained in imagination) drives CARLA tasks from CarDreamer. The research question is whether a self-supervised video encoder - our own JEPA pretrained on real dash-cam video (comma2k19), or Meta's V-JEPA2 - transfers better than a CNN trained from scratch, with everything else held identical.
  **Evidence:** journal 2026-10-01 22:55; `configs/experiment_contract.yaml`

- **Q:** How did you structure the project to de-risk it?
  **A (short):** Stages with gates: environment -> simulator smoke test -> a first car that learns with an easy observation (bird's-eye view) -> JEPA pretraining with a cheap probe to select the method -> the controlled encoder comparison. Each stage has a measurable pass condition.
  **Evidence:** journal 2026-10-01 22:55

## Reinforcement learning / world models

- **Q:** Why did you start with a bird's-eye view and discrete actions?
  **A (short):** It is the configuration CarDreamer's own DreamerV3 results use, and the BEV contains the route. Proving the collect -> replay -> world model -> actor loop learns there separates pipeline bugs from representation questions.
  **Evidence:** journal 2026-10-01 23:30

- **Q:** What did you do when the agent stopped improving?
  **A (short):** Looked at what it saw and did: BEV frames at episode end and the action histogram showed it always drove straight through the intersection where the route turns right. Compared settings with the reference implementation (4x fewer updates per env step, half the BEV resolution) and resumed training with the reference update ratio rather than restarting. (pending: outcome)
  **Evidence:** journal 2026-10-02 01:05

- **Q:** How do you treat episodes that end by time-out?
  **A (short):** As truncations, not terminal states: discount stays 1 so the critic bootstraps. The original code marked them terminal, which teaches the value function that time-outs are dead ends.
  **Evidence:** journal 2026-10-01 23:30; `src/dreamer/carla_wrappers.py`

## Self-supervised learning (JEPA)

- **Q:** Why JEPA rather than pixel reconstruction (MAE)?
  **A (short):** JEPA predicts representations of masked regions, so it does not spend capacity on unpredictable pixel detail; that suits driving, where geometry and motion matter more than texture.
  **Evidence:** `reports/` plan; journal 2026-10-02 00:20

- **Q:** How do you prevent representation collapse, and how did you choose the method?
  **A (short):** Two options behind one switch: an EMA teacher with stop-gradient (V-JEPA) and SIGReg (LeJEPA), which pushes embeddings toward an isotropic Gaussian via random 1-D projections and the Epps-Pulley test. Both get the same step budget and the winner is the one with the larger linear steering-probe gain over a random encoder, not the lower loss. (pending: which won)
  **Evidence:** journal 2026-10-02 00:20; `src/jepa/sigreg.py`

- **Q:** Why tube masking?
  **A (short):** If a patch is hidden in one frame but visible in the next, the model can copy it instead of reasoning. Masking the same spatial blocks in every frame forces it to infer content from context.
  **Evidence:** journal 2026-10-02 00:20; `src/jepa/masking.py`

- **Q:** Why turn off action conditioning in the predictor?
  **A (short):** The predictor was given the clip's own steering and speed. With spatial masking, that hands it exactly the signal the steering probe measures, so the encoder would not need to learn it.
  **Evidence:** journal 2026-10-02 00:20

## Simulation & infrastructure

- **Q:** How did you make the simulator setup reproducible?
  **A (short):** Every change to third-party code lives in a tracked patch applied by a setup script; the CARLA map default is set by a script; a `doctor` command checks GPU architecture, patches, Vulkan and CARLA; a launcher starts CARLA headless, waits for readiness and always cleans it up.
  **Evidence:** journal 2026-10-01 23:18; `scripts/setup_cardreamer.sh`, `run.py`

- **Q:** How did a 300M-parameter frozen video encoder fit RL training on a 12 GB laptop GPU?
  **A (short):** Encode each camera frame once at collection time (rolling 4-frame clip, bf16, no grad) and store the 1024-d embedding in replay. Dreamer updates then never run the ViT; the old design re-encoded ~1000 clips per update. Measured: 9.0 GB total with CARLA, 7.8 env steps/s.
  **Evidence:** journal 2026-10-02 01:15

- **Q:** How did you budget compute on a single 12 GB laptop GPU?
  **A (short):** Measured each job before queueing it (Dreamer ~12 env steps/s CNN, ~7.8 with a frozen ViT-L; JEPA 6 steps/s EMA, 3.7 SIGReg; probe 2.5 min/epoch), kept peak memory under 12 GB with CARLA running (9.0 GB worst case), and ran jobs one at a time in a resumable queue.
  **Evidence:** journal 2026-10-02 01:15, 01:35; `jobs/long_runs.sh`

- **Q:** How did you keep long experiments safe on a laptop?
  **A (short):** Runs are resumable and checkpoint on SIGTERM; a queue runs them one at a time under `systemd-inhibit` (no suspend) with a health monitor that stops trainers cleanly on GPU overheating, low RAM or low disk, and a STOP file for manual stops. A test of the emergency stop showed `pkill -f` can kill unrelated shells, so the pattern is anchored to the Python process.
  **Evidence:** journal "Crash protection for multi-hour runs"; `jobs/long_runs.sh`

## Debugging stories

- **Q:** Tell me about a hard bug.
  **A (short):** A patch meant to keep frozen encoders frozen made every parameter non-trainable after the first update, so the agent silently stopped learning; fp32 hid it, and the unit test only ran one step. fp16's GradScaler exposed it ("no inf checks recorded"). Fixed with a persistent per-parameter freeze marker and a test that checks the second update.
  **Evidence:** journal 2026-10-01 23:10

- **Q:** What went wrong with the data pipeline?
  **A (short):** The download links were dead, the speed field name was wrong (it would have trained on zeros), telemetry timing was ~60 ms off, and per-sample HEVC decoding from frame 0 made training CPU-bound. Fixed with a one-time preprocessor aligned on recorded frame times and memory-mapped clips.
  **Evidence:** journal 2026-10-02 00:12

## Experiment design & evaluation

- **Q:** How do you evaluate a driving agent beyond success rate?
  **A (short):** Per episode: return, route completion (fraction of planned waypoints passed), termination reason (collision / out of lane / destination / time-out), speed and wall clock. Route completion shows progress while success is still 0%; the comparison reports success, route completion, collision rate, steps to 80% success, and compute cost.
  **Evidence:** journal "Route completion metric"; `src/dreamer/carla_wrappers.py`

- **Q:** How do you make the encoder comparison fair?
  **A (short):** Same task, seeds, step budget, evaluation protocol, route branch (identical MLP), decoder target and adapter capacity; only the image encoder differs. Report mean +/- std over seeds, steps to 80% success, wall clock and peak VRAM.
  **Evidence:** `configs/experiment_contract.yaml`; `scripts/train_cardreamer.py` (`run_comparison`)

- **Q:** How did you make sure your encoder evaluation was not fooling you?
  **A (short):** Every probe runs against a random-init encoder of the same architecture, and selection uses only the gain over it. Splits are by driving segment, including the inner holdout that picks the ridge strength: a random-clip holdout once produced a fake +0.24 R2 gain for an untrained encoder because neighbouring clips are near-duplicates.
  **Evidence:** journal "Linear probe: 30x faster, and a leaky holdout found"; `src/eval/linear_probe.py`

## Trade-offs & what I would do differently

- **Q:** What would you do differently?
  **A (short):** Save every submodule change as a patch immediately; test the second training step and resume paths from day one; validate loaders against real files before writing training code; make long jobs checkpoint on SIGTERM.
  **Evidence:** journal lessons 23:02, 23:10, 23:18, 00:12, 01:00
