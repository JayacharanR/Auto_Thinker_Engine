# Auto Thinker Engine - project journal

Append-only log of milestones, blockers, bugs, decisions and results.
Format and rules: `.claude/skills/project-journal/SKILL.md`.
Entries before 2026-10-01 22:55 (remote A4000 server period) are summarised in
`reports/progress_and_implementation_plan.md`.

---

## 2026-10-01 22:55 - Moved from the remote A4000 server to the laptop; staged plan
**Type:** decision
**Stage:** Planning

**What happened:** Work on the remote RTX A4000 Docker server had stalled before a single Dreamer update ran. The project moved to an RTX 5070 Ti Laptop (12 GB, Blackwell sm_120), running natively without Docker.
**Cause:** n/a
**How we handled it:** A critical review of the code produced 12 blockers (env pins, lost server fixes, observation without route, ignored configs, silent checkpoint loads, slow data pipeline, leaky masking, etc.) and a 6-stage plan: host prep -> Python env -> CARLA -> first working car (BEV, CNN, discrete) -> JEPA on a laptop budget (EMA vs SIGReg switch) -> camera+route encoder comparison. Plan: `~/.claude/plans/robust-drifting-galaxy.md`.
**Verified by:** n/a
**Lesson:** Prove the simplest closed loop learns before adding the research variable (pretrained encoders).
**Interview angle:** How did you structure the project to de-risk it?

## 2026-10-01 23:00 - Blackwell GPU could not run the pinned PyTorch
**Type:** blocker
**Stage:** Stage 0/1 - environment

**What happened:** `nvidia-smi` failed with "Driver/library version mismatch"; the pinned `torch<2.5` (CUDA 12.1) has no kernels for sm_120.
**Cause:** Kernel module and userspace driver out of sync after an update; Blackwell needs CUDA >= 12.8 builds.
**How we handled it:** Reboot (user). Pinned `torch==2.11.*` / `torchvision==0.26.*` from the PyTorch cu128 index via `[tool.uv.sources]`, regenerated `uv.lock`. Added a doctor check that the GPU's arch is in `torch.cuda.get_arch_list()`.
**Verified by:** arch list includes sm_120; bf16 matmul on GPU; `run.py doctor --require-cuda` passes.
**Lesson:** A torch build without kernels for your GPU imports fine and fails later - check the arch list explicitly.
**Interview angle:** What environment problems did you hit moving to new hardware?

## 2026-10-01 23:02 - Interrupted submodule clone after reboot
**Type:** blocker
**Stage:** Stage 1 - environment

**What happened:** `third_party/CarDreamer` had a `.git` but no files (all staged as deleted); `dreamerv3_torch` was empty.
**Cause:** The clone was cut off by the reboot.
**How we handled it:** `git reset --hard` in the empty CarDreamer checkout (no local work existed), `git submodule update --init --recursive`, then applied the dreamer compat patch.
**Verified by:** both submodules at the pinned commits; patch applies.
**Lesson:** Check a half-initialised submodule for local work before resetting it.
**Interview angle:** -

## 2026-10-01 23:10 - Dreamer silently stopped learning after its first update
**Type:** bug
**Stage:** Stage 1 - environment

**What happened:** A GPU check with fp16 AMP crashed in the actor optimizer: `AssertionError: No inf checks were recorded for this optimizer`.
**Cause:** The project's patch to dreamerv3-torch's `tools.RequiresGrad` snapshotted `requires_grad` on entry to keep frozen encoders frozen, but `__exit__` set every parameter to False - so from the second update the snapshot was all-False and no world model, actor or critic parameter got gradients. In fp32 this was silent (zero grads, no-op step); the CPU test hid it by forcing `requires_grad=True` and running a single step.
**How we handled it:** Restored upstream semantics (enable on entry, disable on exit) and made freezing a persistent per-parameter marker (`_dreamer_frozen`) set by `freeze_parameters()` in `src/dreamer/encoder_adapter.py`. Regenerated `patches/dreamerv3_torch_compat.patch`.
**Verified by:** New regression test `TestRepeatedUpdates` (fails on the old patch, passes now); 40 GPU updates with model loss 3014 -> 1184.
**Lesson:** Test the second training step, not just the first; mixed precision can expose bugs fp32 hides.
**Interview angle:** Tell me about a hard bug you found.

## 2026-10-01 23:12 - CARLA download returned 403
**Type:** blocker
**Stage:** Stage 2 - CARLA

**What happened:** `https://tiny.carla.org/carla-0-9-15-linux` redirected to a CDN that answered 403.
**Cause:** CDN access restriction.
**How we handled it:** Traced the redirect; the Backblaze origin (`carla-releases.s3.us-east-005.backblazeb2.com`) served the same 8.4 GB archive (size matched). Recorded in the README.
**Verified by:** extraction completed; smoke test passed.
**Lesson:** Follow redirects by hand when a mirror fails; the origin is often still up.
**Interview angle:** -

## 2026-10-01 23:18 - Recreated the lost CarDreamer server fixes as a tracked patch
**Type:** decision
**Stage:** Stage 2 - CARLA

**What happened:** The server-side fixes to CarDreamer (map reuse, world timeout, Traffic Manager bypass, monitor shutdown) had lived only in the server's working tree and were lost.
**Cause:** Edits to a submodule were never saved as a patch.
**How we handled it:** `patches/cardreamer_compat.patch`, applied by `scripts/setup_cardreamer.sh` (generic apply/already-applied/reject logic). The old Traffic Manager fix disabled the TM for `carla_right_turn_simple`, but that task spawns autopilot cross traffic - so it had likely removed the task's traffic. Replaced with lazy TM creation (only tasks that use traffic start one) plus an explicit opt-out env var. Monitor thread made daemon with a bounded join; `Town03_Opt` default map via `scripts/configure_carla.sh`.
**Verified by:** smoke test 63.7 steps/s; CARLA ~2 GB VRAM; no orphan processes.
**Lesson:** Anything edited inside a submodule must become a tracked patch the same day.
**Interview angle:** How did you make the simulator setup reproducible?

## 2026-10-01 23:27 - Mistake: `git stash` reverted the environment pin mid-session
**Type:** mistake
**Stage:** Stage 3

**What happened:** To compare lint results before/after, Claude ran `git stash`; this also reverted `pyproject.toml`, and the next `uv run` began re-syncing the old torch.
**Cause:** Using the working tree for a read-only comparison.
**How we handled it:** Killed the resync, `git stash pop`, re-synced; torch 2.11 confirmed. Later comparisons used `git show HEAD:file | ruff --stdin-filename`.
**Verified by:** `git status` showed all changes restored; tests passed.
**Lesson:** Never mutate the working tree to look at an old version.
**Interview angle:** -

## 2026-10-01 23:30 - Stage 3 runner: observation contract, metrics, strict loading
**Type:** decision
**Stage:** Stage 3 - first working car

**What happened:** The runner fed only the front camera (no route), ignored the YAML Dreamer config, wrote an empty `metrics.json`, treated time-outs as terminal states, and loaded checkpoints with `strict=False`.
**Cause:** Accumulated shortcuts from the server period.
**How we handled it:** `src/dreamer/carla_wrappers.py` (BEV observation, discount 1 on time-outs, per-episode JSONL metrics), `configs/laptop.yaml` profile with `--set` overrides, strict resume/eval loading, CNN arm on dreamerv3-torch's own encoder, clean shutdown.
**Verified by:** 2k-step integration run: 288 updates, all outputs written, clean exit. Resume 1500 -> 2000 exact.
**Lesson:** Make the agent's observation contain the information the task needs (the route) before tuning anything.
**Interview angle:** Why did you start with a bird's-eye view?

## 2026-10-01 23:34 - Resume re-ran the random prefill
**Type:** bug
**Stage:** Stage 3

**What happened:** The first resume test prefilled 1498 random steps and jumped the step counter to 3498.
**Cause:** Prefill was computed as profile prefill minus steps already in replay; a resumed run had already prefilled.
**How we handled it:** No prefill on resume (fresh runs require an empty log dir).
**Verified by:** resume continued 1500 -> 2000 with 63 updates (= 500/8).
**Lesson:** Test resume with real numbers, not just "it loads".
**Interview angle:** -

## 2026-10-01 23:39 - Long runs outlive the agent's shell
**Type:** blocker
**Stage:** Stage 3

**What happened:** The 100k run (~3 h) would exceed the 2 h limit of a background tool command and be killed.
**Cause:** Tool-managed background jobs have a time limit.
**How we handled it:** Launched detached with `setsid nohup ... &`. Added `scripts/watch_run.py`, a live progress bar that only reads the run's log files.
**Verified by:** run survived for 1.5 h; watcher showed steps/s and ETA.
**Lesson:** Decouple long jobs from the session that starts them.
**Interview angle:** -

## 2026-10-02 00:12 - comma2k19: dead links, wrong field name, timing offset
**Type:** bug
**Stage:** Stage 4 - JEPA data

**What happened:** The download script's torrent link returned 404; its Hugging Face mode would fetch the whole ~100 GB repo. After downloading Chunk 1, the verifier found 0/188 complete segments.
**Cause:** comma2k19 stores speed as `processed_log/CAN/speed`, not `car_speed` - the old loader would have trained on all-zero speed. Telemetry was also aligned with `t - t[0]` instead of the recorded frame times (~60 ms offset).
**How we handled it:** Downloader fetches `raw_data/Chunk_N.zip` from `commaai/comma2k19`; field renamed; `scripts/preprocess_comma2k19.py` aligns with `global_pose/frame_times` (same clock as CAN) and decodes each segment once to 128 px / 10 Hz memmaps.
**Verified by:** 188/188 segments complete; 112,420 frames in 4.5 min; sample frames and telemetry plausible (speed 0 at a red light).
**Lesson:** Validate a dataset loader against the real files before training on it.
**Interview angle:** How did you build the data pipeline, and what went wrong?

## 2026-10-02 00:20 - JEPA design changes for a 12 GB laptop
**Type:** decision
**Stage:** Stage 4 - JEPA

**What happened:** Original setup: 224 px, 16 frames (1568 tokens), `nn.MultiheadAttention`, masks that let the model copy from neighbouring frames, predictor conditioned on the clip's own steering.
**Cause:** Settings sized for a bigger GPU; masking/conditioning choices weakened the learning signal.
**How we handled it:** 128 px, 8 frames (256 tokens); SDPA attention; V-JEPA tube masks (same spatial blocks in every frame); EMA vs SIGReg (LeJEPA) switch with equal step budgets, chosen later by a linear steering probe vs a random encoder; action conditioning off (it hands the predictor the signal the probe measures); hue jitter off (75% of CPU time per clip).
**Verified by:** unit tests (no leak, tube property, SIGReg 1.1 Gaussian vs 4507 collapsed, SDPA == reference); CPU training + resume for both regularisers; loader 9.7 batches/s.
**Lesson:** Choose collapse prevention by a downstream probe, not by the pretraining loss.
**Interview angle:** Why JEPA instead of pixel reconstruction? How do you prevent collapse?

## 2026-10-02 01:00 - Stopping the run: SIGINT ignored
**Type:** blocker
**Stage:** Stage 3

**What happened:** Ctrl-C (SIGINT) to the detached trainer did nothing for 2 minutes.
**Cause:** Background jobs started by a non-interactive shell ignore SIGINT, and Python then does not install its KeyboardInterrupt handler - so the save-on-interrupt path never ran.
**How we handled it:** SIGTERM; the launcher's trap stopped CARLA. Lost ~3k steps since the 82.5k checkpoint. Open: handle SIGTERM in the trainer to checkpoint before exit.
**Verified by:** no CARLA or trainer processes left; checkpoint at step 82,500.
**Lesson:** Make detached jobs checkpoint on SIGTERM, not only Ctrl-C.
**Interview angle:** -

## 2026-10-02 01:05 - First working car: drives straight, never turns right
**Type:** result
**Stage:** Stage 3 - learning gate (not passed)

**What happened:** CNN + BEV + discrete actions, 82.5k env steps (`outputs/logs/cnn_bev_seed42`). Eval return 44 (12.5k) -> 165 (82.5k), plateau from ~50k. Every episode (train and eval) ends `out_of_lane` after ~100-110 steps at ~10 km/h; destination never reached (gate: >= 80% success by 100k).
**Cause:** BEV frames show the car going straight through the intersection where the route bends right; actions are ~85% "straight". Compared with CarDreamer's own DreamerV3 setup: train_ratio 512 vs our 128 (4x fewer updates per step), BEV 128 px vs our 64 px.
**How we handled it:** Plan: resume from 82.5k with `train_ratio=512` overnight (keeps what was learned).
**Verified by:** open.
**Lesson:** When a policy plateaus at the same failure every episode, look at what it sees and what it does before tuning.
**Interview angle:** What did you do when the agent stopped improving?

## 2026-10-02 01:08 - Camera+route contract works; route lookahead was only 8 m
**Type:** milestone
**Stage:** Stage 5 - check (CNN arm)

**What happened:** 2k-step CNN run with front camera + 23-d route vector + continuous actions passed (288 updates, 4.1 GB). The stored route showed waypoints 0.8 m apart, so 10 waypoints covered only 8 m.
**Cause:** CarDreamer's planner samples every 0.8 m (up to 60 waypoints).
**How we handled it:** Take every 4th waypoint (~30 m lookahead) in `route_features`; test added.
**Verified by:** unit test on a 60-waypoint line (3.2 m spacing).
**Lesson:** Inspect the actual values a feature takes in the real environment.
**Interview angle:** How did you design the route input?

## 2026-10-02 01:15 - V-JEPA2 arm runs in CARLA within the 12 GB budget
**Type:** milestone
**Stage:** Stage 5 - check (V-JEPA2 arm)

**What happened:** First attempt crashed: `tensors with negative strides are not currently supported`. After the fix, the 2k-step run passed: ViT-L features computed once per env step and stored in replay; 288 updates; training VRAM 4.6 GB, total GPU 9.0 GB with CARLA; 7.8 env steps/s (CNN: ~12).
**Cause:** CarDreamer's camera image is a channel-reversed numpy view.
**How we handled it:** Contiguous copy before `torch.as_tensor`; regression test uses a reversed view. Also fixed: the adapter pooled `(B, T, D)` replay features over time (treated them as ViT tokens) - now flattened.
**Verified by:** CARLA run exit 0; 120 tests pass.
**Lesson:** Precomputing frozen-encoder features at collection time removes ~1000 ViT forwards per Dreamer update.
**Interview angle:** How did you make a 300M-parameter frozen encoder fit RL training on a laptop?

## 2026-10-02 01:25 - Trainer now checkpoints on SIGTERM
**Type:** bug
**Stage:** Stage 3 (follow-up to 01:00)

**What happened:** The open item from stopping the 100k run: a detached trainer could only be stopped with SIGTERM, which skipped the save-on-interrupt path.
**Cause:** Save logic only handled KeyboardInterrupt (SIGINT), which detached jobs ignore.
**How we handled it:** `scripts/train_cardreamer.py` installs a SIGTERM handler for the run that raises KeyboardInterrupt, so `kill`/`timeout` save `latest.pt` before exiting; the handler is reset in `finally`.
**Verified by:** test suite (120 passed); exercised on the next long run.
**Lesson:** Treat SIGTERM as the normal way a long job ends.
**Interview angle:** -

## 2026-10-02 01:35 - All three arms run in CARLA; Phase 2 cost measured
**Type:** milestone
**Stage:** Stage 4/5 checks

**What happened:** custom_jepa arm (2k steps, camera+route, continuous) passed using a 300-step test checkpoint: encoder rebuilt from the checkpoint's own config (128 px x 8 frames), 384-d `feat`, 7.7 env steps/s, 3.5 GB. Phase 2 measured on the GPU: EMA 385 clips/s (6 steps/s, 3.9 GB total), SIGReg 227 clips/s (~3.7 steps/s, 7.1 GB) - 30k steps = ~1.4 h / ~2.2 h. Linear probe: 148 s per epoch for trained + random encoders.
**Cause:** n/a
**How we handled it:** Probe epochs 50 -> 10 in the laptop config; Phase 2 progress bar disabled in non-terminal logs. Long runs queued in `jobs/long_runs.sh` (Phase 2 EMA -> SIGReg -> probes -> automatic selection by R2 gain over random -> Stage 3 resume with train_ratio 512); selection step tested with fake probe results.
**Verified by:** CARLA runs exit 0; selection test picked the larger gain and copied its checkpoint.
**Lesson:** Measure every long job's throughput before queueing it, so the queue fits the time available.
**Interview angle:** How did you budget compute on a single laptop GPU?

## 2026-10-02 15:54 - Long trainings deferred; improve code first
**Type:** decision
**Stage:** Planning

**What happened:** The user chose to keep the multi-hour runs (Phase 2 x2, Stage 3 resume, comparison) for later and spend the current session on code improvements.
**Cause:** n/a
**How we handled it:** `jobs/long_runs.sh` stays ready (resumable, failure-tolerant, status file). Next: cheaper probe, route-progress metric, and a short measurement to decide how to retry Stage 3.
**Verified by:** n/a
**Lesson:** -
**Interview angle:** -
