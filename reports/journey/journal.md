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

## 2026-10-02 15:58 - Linear probe: 30x faster, and a leaky holdout found
**Type:** bug
**Stage:** Stage 4 - encoder selection

**What happened:** The probe re-encoded all 25k clips (with augmentation) every epoch: 148 s/epoch, ~50 min for both checkpoints. Rewritten as encode-once + closed-form ridge, it took 50 s - but both encoders then scored negative validation R2 with a weak ridge (alpha 0.1), and a spurious "gain over random" of 0.24 on an untrained 300-step checkpoint.
**Cause:** The ridge strength was chosen on randomly held-out clips; clips from the same drive are near-duplicates, so selection rewarded memorising segments, which does not transfer to unseen validation drives.
**How we handled it:** Hold out whole segments (15%) for the ridge choice, wider alpha grid (1e-2..1e5); deterministic loaders for feature extraction (no augmentation, fixed starts). `src/eval/linear_probe.py`, `create_comma2k19_dataloaders(deterministic=True)`.
**Verified by:** synthetic test (linear signal R2 > 0.99, noise R2 < 0.05); on the untrained checkpoint both encoders now give R2 ~ 0 (-0.026 vs -0.018) with strong ridge chosen - the honest result.
**Lesson:** Any split used for model selection must respect the same grouping as the final train/val split.
**Interview angle:** How did you make sure your encoder evaluation was not fooling you?

## 2026-10-02 16:06 - How to retry Stage 3: measured, 128 px BEV does not fit
**Type:** decision
**Stage:** Stage 3 - learning gate retry

**What happened:** Two candidate fixes for the "never turns right" plateau, each measured with a 2k-step CARLA run at CarDreamer's train_ratio 512: (A) 64 px BEV, resume the 82.5k run; (B) 128 px BEV (CarDreamer's resolution), fresh run.
**Cause:** (B) ran out of GPU memory: 11.4 of 11.6 GB with CARLA, batch 16x64 at 128 px. (A): 5.5 env steps/s, 3.9 GB, 850 updates per 2k steps (4x the learning per env step of the first run).
**How we handled it:** Chose (A): resume from 82.5k to 150k (~3.5 h), queued as the last step of `jobs/long_runs.sh`. (B) would need a smaller batch, changing a second variable at once.
**Verified by:** `outputs/logs/speed_bev64/metrics.json`; OOM traceback in `outputs/speed_bev128.log`.
**Lesson:** Change one variable at a time, and measure memory before committing hours to a configuration.
**Interview angle:** What did you do when the agent stopped improving?

## 2026-10-02 16:06 - Route completion metric
**Type:** decision
**Stage:** Stage 3/5 - metrics

**What happened:** `route_completion` was always 0, so a run with 0% success showed no sign of partial progress.
**Cause:** Nothing computed it.
**How we handled it:** Episode recorder counts waypoints passed (CarDreamer's per-step `num_completed + num_obsolete`) over the route length at reset; 1.0 on arrival. Added to `episodes.jsonl`, eval summaries, comparison table and `watch_run.py`.
**Verified by:** unit test (crash after 10 of 40 waypoints = 0.25; arrival = 1.0); 122 tests pass.
**Lesson:** When the headline metric is stuck at zero, add a graded one that can show progress.
**Interview angle:** How do you evaluate a driving agent beyond success rate?

## 2026-10-02 16:07 - Cost of the full encoder comparison
**Type:** result
**Stage:** Stage 5 - planning

**What happened:** Measured camera+route throughput at train_ratio 128: CNN 10.7 env steps/s (4.1 GB), custom_jepa 7.7 (3.5 GB), vjepa2 7.8 (4.6 GB); CNN at train_ratio 512: 5.5. Full design (3 arms x 3 seeds x 200k steps): ~65 h at ratio 128, ~105 h at ratio 512.
**Cause:** One laptop GPU, and CARLA steps in real time with the agent.
**How we handled it:** Open decision, after the Stage 3 retry shows whether ratio 512 is needed: likely one seed per arm first (~17-35 h), then more seeds for the arms that matter. Added `scripts/inspect_episode.py` (frames + decoded actions per episode) to diagnose runs without rerunning them.
**Verified by:** `metrics.json` of the check runs.
**Lesson:** Budget the experiment design against measured throughput before promising results.
**Interview angle:** How did you budget compute on a single laptop GPU?

## 2026-10-02 16:32 - Public-repo audit before more commits
**Type:** decision
**Stage:** Repository hygiene

**What happened:** The 15:58 commit (`8af2e1a`) was already pushed to the public GitHub repo, so the audit covered pushed and pending files.
**Cause:** n/a
**How we handled it:** Scanned all tracked and untracked files for API tokens/keys, passwords, home-directory paths and email addresses: none found. Data, outputs, `.venv` and caches were already ignored. Added ignores for `.env*`, `.claude/settings.local.json` (per-user Claude Code permissions), tool caches, and weights/checkpoints/replay (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.npz`, `*.npy`) written anywhere outside `outputs/`.
**Verified by:** `git check-ignore` on sample paths; the journal skill stays tracked.
**Lesson:** Audit before the first push, and ignore by file type as well as by folder.
**Interview angle:** -

## 2026-10-02 16:32 - Crash protection for multi-hour runs (and a dangerous pkill)
**Type:** bug
**Stage:** Long runs

**What happened:** Before running at full load: a laptop run can die from suspend, GPU overheating, RAM or disk exhaustion. While testing the queue's emergency stop, `pkill -f scripts/train_phase2_jepa.py` killed the shell that ran it (exit 144).
**Cause:** `pkill -f` matches any process whose command line contains the text - including shells that merely mention it (the queue, a terminal, a monitoring command).
**How we handled it:** `jobs/long_runs.sh`: health monitor every 30 s (GPU temp/memory/util/power, CPU temp, RAM, disk -> `health.log`); stops trainers if GPU >= 92 C for 90 s, RAM < 2 GB or disk < 5 GB, or on `touch outputs/long_runs/STOP`, then skips remaining steps; launched under `systemd-inhibit` so sleep/lid-close cannot suspend it. The stop pattern is anchored to a Python interpreter running the script. Phase 2 trainer now checkpoints on SIGTERM mid-epoch and exits 143.
**Verified by:** fake trainer stopped while a decoy shell survived; forced "low disk" trip skipped later steps; test sends SIGTERM at step 3 -> checkpoint at 3 -> resume to 50; 124 tests pass.
**Lesson:** Kill by an anchored pattern or PID, never by a substring that your own tooling may contain.
**Interview angle:** How did you keep long experiments safe on a laptop?

## 2026-10-02 16:50 - First long-run attempt: wrong checkpoint choice, killed workers, corrupt save
**Type:** bug
**Stage:** Stage 4 - Phase 2 long run

**What happened:** 12 min into Phase 2 EMA (~5.5k steps), `best.pt` had not changed since epoch 2 and the log was empty. Stopping the queue via the STOP file then crashed the trainer ("DataLoader worker killed by signal: Terminated") and left `latest.pt` as a 671-byte fragment.
**Cause:** (1) Validation prediction loss is not comparable across epochs for JEPA - the EMA targets improve - so it fell to 0.049 at step 782 and rose to 0.090 while embedding std stayed ~0.85 (no collapse); "best by val loss" would have sent a nearly untrained encoder to the probe. (2) DataLoader workers share the trainer's command line, so the stop pattern hit them too. (3) `torch.save` writes in place, so dying mid-save corrupts the file. (4) Python buffers stdout to files.
**How we handled it:** `final.pt` at the end of the budget, probe/selection use final weights; val loss kept as a diagnostic. `atomic_save` (write .tmp, rename) for Phase 2 and Dreamer checkpoints. Stop only processes whose parent is not a trainer (newline-safe cmdline check). `PYTHONUNBUFFERED=1`; readable STOP reason; one TensorBoard run per checkpoint dir. Lost: ~12 min of EMA training.
**Verified by:** fake trainer with forked workers: only the main got SIGTERM and exited cleanly; interrupted-save test keeps the previous checkpoint; 125 tests pass.
**Lesson:** Rehearse the stop path of a long job before relying on it, and never select self-supervised checkpoints by their own moving-target loss.
**Interview angle:** How do you choose a checkpoint for a self-supervised model?

## 2026-10-02 17:25 - Checkpoint cadence, end-to-end runner test, queue progress view
**Type:** decision
**Stage:** Long runs

**What happened:** Reviewing crash exposure: Phase 2 saves every epoch (~1 min), but the Dreamer runner saved only every eval interval (10k env steps, ~30 min at train_ratio 512). A first end-to-end test of the runner took >10 min and slowed the live Phase 2 run (532 -> ~416 clips/s).
**Cause:** Checkpoints tied to evaluation; the test's CPU torch used ~12 threads at full size and competed with the Phase 2 DataLoader workers.
**How we handled it:** Runner checkpoints every 2,500 env steps inside each eval chunk (`--checkpoint-every`). New `tests/test_train_loop.py` runs `train_arm` end to end on a fake CarDreamer task (prefill, cadence 110/160/210/260, evals, metrics, refusal to overwrite, resume) with 2 threads and a tiny model: 4.4 s. `scripts/watch_queue.py`: live bars for the current step and the whole queue, plus GPU/RAM health. Merged to main locally; push needs the user's GitHub credentials (none on this machine).
**Verified by:** 125 tests pass; watcher matches status/logs (Phase 2 at 42%, queue 6.8%).
**Lesson:** Cap CPU threads in tests that run next to a training job.
**Interview angle:** -

## 2026-10-02 23:46 - Correction: right_turn_simple has no cross traffic
**Type:** mistake
**Stage:** Stage 2 (correction of 2026-10-01 23:18)

**What happened:** The 23:18 entry claimed the server's Traffic Manager bypass "likely removed the task's traffic". Checked against the task config: CarDreamer spawns flow traffic only when `min_flow_dist` is set, which only the medium/hard variants do.
**Cause:** Inferred from code that spawns traffic without checking the condition guarding it.
**How we handled it:** Corrected here; the lazy Traffic Manager is still the right design (tasks with traffic start one, others never do).
**Verified by:** `load_task_configs('carla_right_turn_simple').env` has no `min_flow_dist`.
**Lesson:** Check the guard condition before claiming what code does at runtime.
**Interview angle:** -

## 2026-10-02 23:46 - Phase 2 results: EMA learns motion, SIGReg does not; steering probe is blind
**Type:** result
**Stage:** Stage 4 - JEPA pretraining and selection

**What happened:** Both 30k-step runs finished (EMA 66 min, SIGReg 103 min; no collapse warnings; GPU peak 63 C). The queue's steering probe ranked them by noise (EMA -0.038, SIGReg -0.013, random -0.018 R2) and picked SIGReg. Diagnostics (`scripts/probe_diagnostics.py`, validation R2): speed - EMA 0.82/0.85/0.88 (mean/2x2/4x4 pooling), random 0.64/0.78/0.78, SIGReg 0.41/0.61/0.69; steering ~0 for every encoder and pooling (best 0.03).
**Cause:** Clip-mean steering on comma2k19 highway driving is not linearly decodable even with spatial layout kept, so it cannot rank encoders. SIGReg with lambda 0.05 and no stop-gradient produced features less informative than random init for speed.
**How we handled it:** Selection criterion changed to speed R2 gain over random with 4x4 pooling (EMA +0.10, SIGReg -0.09); EMA final weights installed as `outputs/checkpoints/phase2/best.pt` for the custom_jepa arm. Probe extended with spatial-grid pooling and multiple targets.
**Verified by:** `outputs/probe_results/diagnostics.json`, `selection.json`; 126 tests pass.
**Lesson:** Validate that a probe can separate trained from random encoders before using it to choose between methods.
**Interview angle:** How did you make sure your encoder evaluation was not fooling you?

## 2026-10-02 23:46 - Stage 3 at 150k steps: from "drive off" to "stop and wait"
**Type:** result
**Stage:** Stage 3 - learning gate (not passed)

**What happened:** Resumed 82.5k -> 150k at train_ratio 512 (3 h, 33,750 updates, 6.2 env steps/s). Behaviour changed at ~100k: instead of leaving the lane at the turn, the car enters the intersection, partly turns, then brakes and holds until the 500-step time limit (81% of eval actions: full brake). Route completion 55-59%, success 0%.
**Cause:** CarDreamer's reward for this task has `time: 0.0` and the speed term is 0 at standstill, so waiting is free while a failed turn costs the out-of-lane penalty: a stable local optimum.
**How we handled it:** Added `--env-set KEY=VALUE` (validated CarDreamer overrides, recorded in the run spec) so a small time penalty can be tested; decision on the next run taken with the user.
**Verified by:** `scripts/inspect_episode.py` frames/actions; eval.jsonl; reward config.
**Lesson:** A graded metric (route completion) and a look at the frames distinguish a new failure mode from no progress.
**Interview angle:** What did you do when the agent stopped improving?

## 2026-10-03 00:15 - Stage 3 retry with a time penalty (user decision)
**Type:** decision
**Stage:** Stage 3 - learning gate retry

**What happened:** Chosen with the user over "continue unchanged to 300k" and "start the comparison now": a fresh CNN/BEV run (`outputs/logs/cnn_bev_tp01_seed42`, 150k steps, train_ratio 512) with CarDreamer's `reward.scales.time` = 0.1 (default 0), launched 00:13 via `jobs/stage3_time_penalty.sh` (~6.6 h).
**Cause:** Waiting at the intersection was free; a 0.1/step penalty makes a full-length stall cost ~-50, small next to the +2/step for driving at the desired speed.
**How we handled it:** Fresh run rather than resuming, so replay does not mix two reward functions. Queue machinery moved to `jobs/lib_queue.sh` (shared by both job scripts). Documented deviation from CarDreamer's default reward; if it works, every comparison arm uses the same reward.
**Verified by:** Same seed, identical random prefill trajectories: every 501-step episode scores exactly 50.1 lower (15.9 -> -34.2, 50.9 -> 0.8, 20.2 -> -29.9); override recorded in the run spec.
**Lesson:** Verify a config override by its effect, not by the absence of an error.
**Interview angle:** Did you change the benchmark's reward? Why, and how did you keep the comparison fair?

## 2026-10-03 07:38 - Stage 3 gate passed: the car completes the right turn
**Type:** milestone
**Stage:** Stage 3 - learning gate (passed)

**What happened:** Fresh CNN/BEV run with time penalty 0.1 (150k steps, train_ratio 512, 6.6 h, 73,850 updates, GPU peak 61 C). Eval success 0% at 12.5k, **100% from 22.5k** onward in 13 of 14 evals (one dip to 10% at 72.5k, recovered next eval); eval return ~354 vs ~165 without the penalty; every training episode from 80k to 150k reached the destination (415/415). Final 20-episode evaluation: **20/20 success, 0 collisions, 0 out of lane**, ~139 steps per episode, 12 km/h; videos in `outputs/eval_results/cnn_bev_tp01_150k/videos`.
**Cause:** The earlier plateau was entirely the free stall in the reward; with it priced, the agent learned the turn within ~20k steps.
**How we handled it:** Gate recorded as passed with the time-penalty reward; that reward becomes part of the comparison contract for every arm.
**Verified by:** `eval.jsonl`, `episodes.jsonl`, `outputs/eval_results/cnn_bev_tp01_150k/evaluation_metrics.json`.
**Lesson:** When an agent settles into a degenerate behaviour, check what the reward makes free before adding capacity or compute.
**Interview angle:** What was the turning point in getting the car to drive?

## 2026-10-03 07:38 - Evaluator mangled discrete actions
**Type:** bug
**Stage:** Stage 3 - evaluation

**What happened:** The 20-episode evaluation crashed on its first step: "Invalid one-hot action".
**Cause:** With the safety shield disabled, `evaluate_agent.py` still passed actions through `SafetySupervisor.filter_action`, which sanitises to `[acc, steer]` and truncated the 15-way one-hot to 2 values.
**How we handled it:** Bypass the supervisor entirely when the shield is off.
**Verified by:** 20-episode evaluation completed (20/20 success).
**Lesson:** A "disabled" component that still transforms its input is not disabled.
**Interview angle:** -

## 2026-10-03 07:49 - Encoder comparison designed, smoke-tested and launched
**Type:** decision
**Stage:** Stage 5 - encoder comparison

**What happened:** Comparison set up as `jobs/comparison.sh`: cnn vs custom_jepa (Phase 2 EMA) vs vjepa2 on carla_right_turn_simple, front camera + route vector, discrete actions, time penalty 0.1, train_ratio 512, 100k env steps per arm, seed 42 first (~20 h). A 1k-step smoke test of all three arms plus aggregation passed (run specs identical apart from the arm; CARLA cleaned up after each arm).
**Cause:** Discrete actions kept so that only the observation changes from the passed Stage 3 setup (one variable at a time); the contract's earlier "continuous" was an unresolved choice.
**How we handled it:** One queue step per arm (independent resume), aggregate-only final step (no CARLA), `--seeds`, queue watcher generalised (`--out`, `steps.json`). Measured throughput at ratio 512: cnn 4.4, custom_jepa 4.7, vjepa2 3.6 env steps/s (the frozen JEPA arm is faster than CNN: only its adapter trains).
**Verified by:** smoke comparison table and run specs; 127 tests pass.
**Lesson:** Smoke-test the whole multi-step job, including aggregation, before committing a day of compute.
**Interview angle:** How do you make the encoder comparison fair?

## 2026-10-03 16:50 - Comparison run 1: JEPA arm solves the task from the camera; CARLA crashed twice
**Type:** result
**Stage:** Stage 5 - encoder comparison

**What happened:** custom_jepa (frozen Phase 2 EMA ViT-S + adapter, front camera + route): first training success at 9.3k env steps, eval success 0% / 70% / **100%** at 12.5k / 22.5k / 32.5k and 100% to 100k; after 50k, 325/325 training episodes reached the destination with 0 collisions; 4.2 h, 6.5 env steps/s, 3.5 GB. (Stage 3 CNN with the privileged bird's-eye view: 100% at 22.5k.) The cnn arm died after ~1 h (checkpoint 22.5k) and vjepa2 after 25 min (checkpoint 7.5k): `apply_settings` timed out during an episode reset.
**Cause:** The CARLA server segfaulted (`Signal 11`, UE4 crash handler) - no OOM kill or GPU fault in the kernel log, GPU peak 59 C. It happens on reset, when CarDreamer toggles synchronous mode and respawns actors; a known instability of the 0.9.15 binary. The first crash's log was lost because every launch overwrote `carla_2000.log`.
**How we handled it:** Trainer saves a checkpoint on `RuntimeError` (lost simulator) as well as on SIGTERM; `jobs/comparison.sh` retries a failed arm up to 6 times (CARLA restarted, training resumed) with one CARLA log per attempt. Comparison relaunched: cnn resumes from 22.5k, vjepa2 from 7.5k, custom_jepa kept.
**Verified by:** Recovery rehearsal: CARLA killed with SIGSEGV mid-run -> checkpoint at exactly the crash step (1606) -> RETRY -> resumed at 1606 -> finished 3000 -> aggregated. Unit test with a simulated simulator crash. 129 tests pass.
**Lesson:** In long simulation runs, plan for the simulator to crash: checkpoint at the failure and restart automatically.
**Interview angle:** How did you handle an unreliable simulator in multi-day experiments?

## 2026-10-03 23:23 - Comparison run 2 stopped by the user; cnn done, vjepa2 at 67k
**Type:** result
**Stage:** Stage 5 - encoder comparison (paused)

**What happened:** cnn resumed from 22.5k and finished 100k (eval success 0% at 12.5k, 100% from 32.5k). vjepa2 needed three automatic retries after CARLA crashes (21:05, 22:19, 22:24) and reached 100% eval success by its ~30k eval; stopped at 23:23 with a checkpoint at 67,153 steps. custom_jepa (run 1): 0% / 70% / 100% at 12.5k / 22.5k / 32.5k.
**Cause:** Stopped on request via the STOP file; the retries confirm CARLA segfaults recur roughly every 1-1.5 h of simulation, and the recovery handled each one.
**How we handled it:** Resume later by rerunning `jobs/comparison.sh` (cnn and custom_jepa are skipped, vjepa2 resumes from 67k). Open issue: after a resume the eval schedule restarts from the resume step (cnn has no 22.5k eval; vjepa2 evaluated at 17k/30k/40k/56k/66k), so steps-to-80%-success is not measured at the same steps for every arm - evals should be aligned to multiples of eval_every before drawing conclusions on sample efficiency.
**Verified by:** `outputs/comparison_queue/status.txt`, each arm's eval.jsonl, vjepa2 latest.pt step 67153.
**Lesson:** A recovery mechanism can quietly change the measurement protocol; check that metrics are still comparable after restarts.
**Interview angle:** How did you handle an unreliable simulator in multi-day experiments?

## 2026-10-04 01:45 - Fixed eval schedule; resumed with a second seed overnight
**Type:** bug
**Stage:** Stage 5 - encoder comparison

**What happened:** After crash recovery, evals restarted counting from the resume step, so arms were measured at different steps (vjepa2 at 17k/30k/40k/56k/66k; cnn lost its 22.5k eval).
**Cause:** Each eval chunk ended at "current step + eval_every".
**How we handled it:** Evals at fixed steps prefill + k * eval_every (12.5k, 22.5k, ...), including after a resume. Queue relaunched with SEEDS="42 123": vjepa2 seed 42 resumes from 67k, then all three arms with seed 123 on the fixed schedule (night covers ~vjepa2 42 + cnn 123 + part of custom_jepa 123; the rest resumes on the next start). Seed 42's early vjepa2 evals stay misaligned, one reason the second seed matters.
**Verified by:** test: crash after the 160 eval, resume -> evals at 160, 260, 360, 380; 129 tests pass.
**Lesson:** Tie measurement schedules to absolute progress, not to when a process started.
**Interview angle:** How do you make the encoder comparison fair?

## 2026-10-04 02:17 - Cheaper second seed: early stopping, finer evals, matched-step summary
**Type:** decision
**Stage:** Stage 5 - encoder comparison

**What happened:** Seed 42 result (camera + route): cnn, custom_jepa and vjepa2 all reach 100% eval success between ~30.8k and 32.5k steps and stay there (custom_jepa 70% already at 22.5k; vjepa2 stopped at 66.7k with four consecutive 100% evals). Running seed 123 for 100k steps per arm (~16-20 h) would mostly re-confirm a solved task.
**Cause:** n/a (user decision: cut cost, keep the measurement that matters).
**How we handled it:** Seed 123 at 50k steps per arm, evals every 5k (finer steps-to-threshold), early stop once 3 consecutive evals reach 90% (`--early-stop-evals`, history read across resumes) - ~6-7 h for all three arms. vjepa2 seed 42 not trained further. Comparison summary now: success at matched steps (`success_by_12.5k/22.5k/32.5k/42.5k`, NaN when that eval was lost to a crash, so missing data never reads as 0%), unfinished runs summarised from their evals (`--include-partial`, never trains), learning-curve plot `comparison_curves.png`. Bug caught on the way: aggregate-only with partial runs fell through to training (CARLA was down, nothing written); fixed with a test.
**Verified by:** unit tests (early stop at 260 with K=2; partial aggregation never trains; matched-step values); real-CARLA smoke of the job (flags, 500-step eval grid, aggregate); 131 tests pass.
**Lesson:** Spend compute on the measurement that can still change the conclusion; stop runs once the question they answer is settled.
**Interview angle:** How did you budget compute on a single laptop GPU?

## 2026-10-04 12:03 - Seed 123: both pretrained encoders learn, the CNN does not (within 50k)
**Type:** result
**Stage:** Stage 5 - encoder comparison (2 seeds)

**What happened:** Seed 123 (50k steps, evals every 5k, early stop K=3): cnn 0% at every eval to 50k; custom_jepa 90% at 27.5k, 100% from 37.5k (early-stopped at 47.5k); vjepa2 80% already at 12.5k, 100% at 27.5k, with dips to 70-80% before 100% at 47.5-50k. Two seeds combined: success within budget cnn 1/2, custom_jepa 2/2, vjepa2 2/2; steps to 80% success custom_jepa 30.0k +- 2.5k, vjepa2 21.7k +- 9.2k, cnn 32.5k (seed 42) / not reached (seed 123); 0 collisions everywhere. One CARLA crash per arm, all recovered automatically. Seed 123 took ~7 h instead of ~16-20 h.
**Cause:** Interpretation (2 seeds, suggestive not conclusive): frozen pretrained video encoders make learning from the front camera more reliable and, for V-JEPA2, faster; the CNN has to learn its features from reward alone and failed on one seed within 50k steps.
**How we handled it:** Results in `outputs/comparison/carla_right_turn_simple_camera_route/` (`comparison.md`, `comparison_curves.png`). Next steps under discussion with the user.
**Verified by:** eval.jsonl / metrics.json per run; aggregate table.
**Lesson:** A second seed can flip the conclusion of a one-seed comparison.
**Interview angle:** Did the pretrained encoders help?
