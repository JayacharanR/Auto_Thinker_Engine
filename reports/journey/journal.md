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

## 2026-10-04 12:13 - Roadmap agreed with the user
**Type:** decision
**Stage:** Planning (after the two-seed comparison)

**What happened:** Next steps chosen, in order: (1) zero-shot robustness evaluation of the six trained comparison agents (night, rain, fog, cross traffic) with front-camera + bird's-eye videos; (2) traffic: train on `carla_right_turn_hard` (dense cross traffic); (4) traffic rules: `carla_traffic_lights` / `carla_stop_sign`; (5) pedestrians (custom task). Deferred: (3) a third seed (456) for the right-turn comparison - needed before presenting that result as firm, not blocking anything. Command: `SEEDS=456 STEPS=50000 EXTRA_SET="--set eval_every=5000" EARLY_STOP=3 AGG_SEEDS="42 123 456" bash jobs/comparison.sh` (~7 h).
**Cause:** n/a
**How we handled it:** Recorded here and in the session memory.
**Verified by:** n/a
**Lesson:** -
**Interview angle:** -

## 2026-10-04 12:32 - Traffic Manager segfault root-caused: a corrupt map cache
**Type:** bug
**Stage:** Roadmap (2) - traffic tasks

**What happened:** The first evaluation on `carla_right_turn_hard` (dense cross traffic) killed the Python process with SIGSEGV (exit 245) when the first traffic car spawned - the same "get_trafficmanager() segfault" the server notes had worked around by disabling traffic. A bare `client.get_trafficmanager(8000)` crashed too (exit 139), in the Python 3.10 and 3.8 builds of carla 0.9.15 alike.
**Cause:** gdb backtrace: `TrafficManagerLocal::SetupLocalMap` -> `InMemoryMap::Load(content)` -> `SimpleWaypoint::SetLeftWaypoint` -> `GetForwardVector` on a near-null pointer. The client was parsing the pre-built Traffic Manager map cache the server sends (`CarlaUE4/Content/Carla/Maps/TM/Town03_Opt.bin`), and that cache crashes the 0.9.15 loader.
**How we handled it:** Renamed the cache folder (`TM` -> `TM.disabled`), so the Traffic Manager builds its map from the road topology (0.1 s). Made it reproducible in `scripts/configure_carla.sh`. Dead ends on the way: a Python 3.8 Traffic Manager host process (same crash), Python 3.7 (unavailable via uv).
**Verified by:** Traffic Manager starts ("No InMemoryMap cache found"), an autopilot car drives 122 m in 150 ticks; CarDreamer `right_turn_hard` evaluation runs with cross traffic (the right-turn V-JEPA2 agent collides 3/3 zero-shot, as expected).
**Lesson:** Get a native backtrace before working around a segfault - here it pointed straight at a data file, not the code.
**Interview angle:** Tell me about a hard bug you found.

## 2026-10-04 13:35 - Zero-shot robustness: pretrained encoders survive night and rain, the CNN does not
**Type:** result
**Stage:** Roadmap (1) - robustness evaluation

**What happened:** 6 agents (3 encoders x seeds 42/123, trained on the right turn in clear weather) x 5 unseen conditions x 10 episodes, ~35 min. Success (mean of seeds): night - cnn **0%** (both seeds leave the lane every episode), custom_jepa **100%**, vjepa2 **100%**; heavy rain - cnn 45%, custom_jepa 95%, vjepa2 100%; sunset - cnn 50%, both pretrained 100%; dense cross traffic (never seen) - cnn 20%, custom_jepa 35%, vjepa2 20%, with 55-80% collisions. The cnn seed-42 agent that is 100% in training conditions drops to 0% at night. Videos (front camera + bird's-eye inset + HUD) in `outputs/robustness/<arm>_seed<n>/<condition>/videos/`; table and chart `outputs/robustness/robustness.md|png`.
**Cause:** Interpretation: frozen encoders pretrained on large, varied video (comma2k19 includes night driving; V-JEPA2 web-scale video) keep lane geometry recognisable under lighting changes, while a CNN trained only on sunny CARLA frames does not. Traffic needs new behaviour (yielding), which no agent has learned yet - hence step 2.
**How we handled it:** Kept as the headline robustness result (2 seeds); traffic fine-tuning (step 2) started next.
**Verified by:** 30 `evaluation_metrics.json` files; table generated by `scripts/robustness_table.py`.
**Lesson:** Out-of-distribution evaluation can separate methods that look identical in-distribution.
**Interview angle:** Did the pretrained encoders help?

## 2026-10-04 13:35 - Traffic fine-tuning stopped by the user at 12.5k steps
**Type:** result
**Stage:** Roadmap (2) - traffic curriculum (paused)

**What happened:** cnn seed 42, fine-tuning from its right-turn agent on carla_right_turn_hard: eval at 7.5k steps 40% success / 60% collisions (zero-shot it was 40% / 60%); stopped at 13:35, checkpoint 12.5k. custom_jepa and vjepa2 not started.
**Cause:** Paused on request.
**How we handled it:** Resume by rerunning the step-2 part of `jobs/roadmap_1_2.sh` (the robustness step is skipped as done; cnn resumes from 12.5k).
**Verified by:** `outputs/traffic_queue/status.txt`, cnn latest.pt step 12500.
**Lesson:** -
**Interview angle:** -

## 2026-10-04 17:05 - Mistake: a second queue copy jammed the traffic run
**Type:** mistake
**Stage:** Roadmap (2) - traffic curriculum

**What happened:** The user had already resumed `jobs/roadmap_1_2.sh` (16:30); Claude launched a second copy (16:32) without noticing - its process check printed the running jobs but was not read as a blocker. The duplicate's first attempt failed, its automatic retry started a second trainer on the same CARLA server and run directory, and the two deadlocked (~25 min, only ~500 env steps, 2-step collision episodes). Killing the duplicate left its car on the ego spawn point, so the real trainer looped on "Failed to spawn actor" indefinitely.
**Cause:** No single-instance guard on queues; the crash-retry logic amplified the duplicate; CARLA keeps a killed client's actors.
**How we handled it:** Killed only the duplicate's process group; stopped the real run cleanly (checkpoint at 12,988), which also restarted CARLA without the stray car; relaunched one copy - progressing again. `jobs/lib_queue.sh` now takes an flock per queue directory and refuses a second instance. Cost: ~30 min and a few hundred odd transitions in a 13k-step replay (negligible).
**Verified by:** lock test (second copy exits with an error); run advancing 12,988 -> 13,910 in 3 min.
**Lesson:** Treat "something is already running" output as a stop sign, and make job launchers refuse to run twice.
**Interview angle:** -

## 2026-10-04 19:22 - Onboarding snapshot: CNN traffic fine-tuning finished; JEPA running
**Type:** result
**Stage:** Roadmap (2) - traffic curriculum

**What happened:** CNN seed 42 finished 50k traffic fine-tuning steps at 18:48:32. Eval success ranged 0-70%; final 10 episodes: 30% success, 70% collisions, 79.9% route completion. No early stop. Custom JEPA was running at ~13.4k/50k at 19:21; eval success/collision at 7.5k = 10%/80%, at 12.5k = 0%/90%. V-JEPA2 remains queued.
**Cause:** Traffic yielding remains unsolved; JEPA's early results do not establish its final outcome.
**How we handled it:** Read-only status inspection; no jobs started, stopped or changed. The sandbox hides host processes, so a separately approved host-level anchored pgrep confirmed one custom-JEPA trainer and one CARLA server.
**Verified by:** `outputs/traffic_queue/status.txt`, per-arm `eval.jsonl`, CNN `metrics.json`, queue watcher and host process check.
**Lesson:** Confirm progress with fresh logs and host process visibility; do not interpret an isolated sandbox's empty process list as idle hardware.
**Interview angle:** Has video pretraining solved traffic negotiation yet?

## 2026-10-04 19:22 - Audit: encoder comparison also changes RSSM input capacity
**Type:** bug
**Stage:** Stage 5 - experiment interpretation

**What happened:** The claim of identical adapter capacity / unchanged RSSM is too strong. CNN uses upstream MultiEncoder without the 1024-d image adapter: image 4096 + route 1024 = 5120 dimensions. Frozen arms use image adapter 1024 + route 1024 = 2048. Saved seed-42 RSSM observation-projection weights are (512, 5632) for CNN and (512, 2560) for both transfer arms; adapter weights are (1024, 384) vs (1024, 1024).
**Cause:** `build_agent` preserves the upstream CNN baseline while PrecomputedFeatureEncoder projects frozen features; WorldModel sizes the RSSM observation layer from each encoder's outdim.
**How we handled it:** Recorded the caveat; no code or benchmark changes. Results compare the current encoder pipelines, but do not isolate pretraining from architecture, capacity, resolution or temporal input. A matched random-init ViT RL control has not been run.
**Verified by:** `scripts/train_cardreamer.py:137`, upstream MultiEncoder/ConvEncoder and RSSM constructors, and CPU FakeTensorMode inspection of the three seed-42 checkpoints (no training or GPU allocation).
**Lesson:** Verify saved parameter shapes before claiming everything downstream of a representation is identical.
**Interview angle:** What confounds remain in the encoder comparison?

## 2026-10-04 19:22 - Audit: rejected duplicate queue can clear an active STOP request
**Type:** bug
**Stage:** Queue lifecycle

**What happened:** `jobs/lib_queue.sh:17` removes OUT/STOP when sourced, before start_queue acquires flock; comparison.sh also writes steps.json before the lock. A duplicate that later fails the lock can therefore erase a pending stop request or overwrite watcher metadata. stop_trainers also targets matching trainers globally and excludes evaluate_agent.
**Cause:** Queue initialization has side effects before ownership is established; the stop pattern is not scoped to queue descendants and covers training/probe scripts only.
**How we handled it:** Recorded by static inspection, without launching a duplicate, exercising STOP or editing any running job script. Keep the existing operational rules; repair requires a later code-change task when jobs are idle.
**Verified by:** Read `jobs/lib_queue.sh`, `jobs/comparison.sh` and `jobs/robustness.sh`; no destructive reproduction performed.
**Lesson:** Acquire ownership before changing shared queue state, and scope shutdown to the processes owned by that queue.
**Interview angle:** What lifecycle risks remain after adding the queue lock?

## 2026-10-04 19:22 - Audit: metric accounting and documentation need reconciliation
**Type:** bug
**Stage:** Evaluation / provenance

**What happened:** Standalone evaluation estimates completion from the capped visible waypoint list (max 60), whereas training uses the planner's whole-route count: successful robustness episodes report 0.95 instead of training's 1.0. Resume resets reported wall-clock/peak VRAM/train tracker statistics, so comparison compute numbers cover the last invocation. V-JEPA2 seed-42 latest.pt is at 72,500 steps, while partial aggregation reports 66,695 (last eval). Manifest saves a config hash, not the effective config claimed in the handoff; latest.pt does store dreamer_config.
**Cause:** Separate metric definitions, invocation-local timers/trackers, eval-derived partial progress, and stale documentation. README/Phase 3 YAML also retain continuous actions, old resolutions/budgets, steering-based selection and other superseded descriptions; actual custom JEPA uses 128 px / 8 frames from its checkpoint.
**How we handled it:** Recorded discrepancies without changing results or code. Interpret success/collision flags separately from completion and compute accounting; use checkpoints, job arguments and the chronological journal to establish actual run settings. Fog was planned but the completed robustness suite is baseline/night/rain/sunset/traffic.
**Verified by:** evaluator and recorder code, BasePlanner, robustness episode JSON, checkpoint metadata, `summarize_run`, manifest utility, README and configs.
**Lesson:** A common metric name does not guarantee a common definition or accounting period.
**Interview angle:** Which reported metrics still need an audit before presenting the results?

## 2026-10-05 01:42 - Direct dense-traffic fine-tuning finished without solving traffic
**Type:** result
**Stage:** Roadmap (2) - traffic curriculum

**What happened:** All seed-42 arms finished 50k steps; queue ended 00:57:32. Final success/collision: CNN 30%/70%, custom JEPA 0%/100%, V-JEPA2 20%/80%; no early stops or 80% success crossings. Last-three-eval mean success: 30%, 6.7%, 13.3%. Last-10k training success: 36.6%, 4.3%, 8.0%, respectively.
**Cause:** Open; the direct jump from an empty intersection to dense aggressive cross traffic did not teach reliable negotiation. This is not evidence that video pretraining is universally worse.
**How we handled it:** Reviewed all eval histories and sampled saved frames/actions. Fresh evaluation stream seed 2025: CNN 10/30 hard successes and 10/10 simple-turn retention; custom JEPA 2/30 hard successes after retry. Other retention/V-JEPA2 checks were still running at this entry. Reward unchanged; shields off.
**Verified by:** `outputs/comparison/carla_right_turn_hard_finetune/`, `outputs/traffic_queue/status.txt`, `outputs/eval_results/traffic_final_20261005_seed2025/`.
**Lesson:** Check the full learning history and retained skills before spending another identical budget on a flat task.
**Interview angle:** Did direct traffic fine-tuning work?

## 2026-10-05 01:42 - CARLA crashed during fresh evaluation; incomplete results were retried
**Type:** blocker
**Stage:** Traffic evaluation

**What happened:** Custom-JEPA fresh hard evaluation lost CARLA after episode 24: server Signal 11, then native client TimeoutException and exit 250. No evaluation_metrics.json was written, so this attempt is incomplete.
**Cause:** The known CARLA 0.9.15 reset crash; this is not a policy failure metric.
**How we handled it:** Preserved original logs; reran incomplete evaluations with up to three attempts and separate logs. Completed CNN results were reused. Retry of custom JEPA completed 30 episodes (2 successes, 28 collisions).
**Verified by:** `custom_jepa_hard.log`, `carla_custom_jepa_hard.log`, retry logs and the completed evaluation JSON under the fresh evaluation directory.
**Lesson:** Retry incomplete simulator evaluations and never count a simulator crash as an agent outcome.
**Interview angle:** How do you separate simulator failures from policy failures?

## 2026-10-05 01:42 - User authorized an unattended overnight curriculum with unchanged reward
**Type:** decision
**Stage:** Roadmap (2) - traffic curriculum retry

**What happened:** User requested an overnight run without further questions. Prepared and launched `scripts/run_traffic_curriculum.py` detached under systemd-inhibit at 01:40:30; waiting for existing short evaluations to finish before using CARLA. Original successful simple-turn checkpoints initialize custom JEPA, V-JEPA2 and CNN, in that order.
**Cause:** Direct hard-task fine-tuning was flat; use the built-in medium task (traffic gaps 8-16 m) before hard (6-8 m), rather than extending the failed checkpoints.
**How we handled it:** Per arm: original medium baseline (20 episodes), up to 30k medium training steps, separate medium validation (30) and simple retention (10). Promote only at >=80% success / <=20% collisions and >=90% simple success / <=10% collisions; then up to 25k hard steps plus validation/retention. Reward time penalty remains 0.1, train_ratio 512, camera+route/discrete unchanged. Fixed eval grid every 2.5k; early stop at three >=90% evals. Gate evaluation stream seed 2026; TM seed unchanged.
**Verified by:** Four controller tests plus six existing trainer tests passed; added watcher-startup regression test also passed; lint clean. Best checkpoints preserved independently of latest; queue lock precedes STOP/metadata changes; STOP targets only owned main clients, retries CARLA failures, eight-hour cap and hardware health guards. Progress: `python3 scripts/watch_queue.py --out outputs/overnight_traffic_queue`.
**Lesson:** An overnight experiment should have promotion gates, preserved candidates and a bounded compute cost; improvement remains a hypothesis to test.
**Interview angle:** How did you redesign the unsuccessful traffic run?

## 2026-10-05 01:42 - Watcher mislabeled a started queue waiting for hardware as unstarted
**Type:** bug
**Stage:** Training progress

**What happened:** The new controller wrote its queue-start banner while waiting for existing evaluations, but watch_queue displayed "Queue not started" until the first START step.
**Cause:** queue_state discarded start banners and returned an empty state before the first step.
**How we handled it:** Retain a __started__ marker and display the waiting/between-steps state with the overall progress bar. No running jobs/*.sh files were changed.
**Verified by:** New watcher regression test, five controller/watcher tests passing, lint clean, live --once snapshot showing 0/21 stages and machine health.
**Lesson:** Waiting for resources is a real queue state and should be visible in monitoring.
**Interview angle:** -

## 2026-10-05 01:51 - Fresh traffic evaluation completed; overnight medium training started
**Type:** result
**Stage:** Traffic evaluation / curriculum launch

**What happened:** Completed all six fresh evaluations of the previous 50k fine-tuned checkpoints. Hard successes/collisions over 30 episodes: CNN 10/20, custom JEPA 2/28, V-JEPA2 5/25. Simple-turn successes over ten episodes: 10, 9, 10; no simple-turn collisions. Mean hard return: 31.2, 5.0, 26.6. These remain one training seed and one evaluation stream, not additional training seeds.
**Cause:** Traffic negotiation remains unsolved; existing turning skill is largely retained. Dense-task failure does not imply wholesale loss of lane/turn control.
**How we handled it:** Saved `outputs/eval_results/traffic_final_20261005_seed2025/evaluation_summary.md|json` and learning curves. The detached overnight controller waited until evaluations ended, then measured the original custom-JEPA medium baseline (5/20 successes, 15/20 collisions) and launched medium training at 01:49:41. Trainer confirmed initialization from the original 100k simple-turn checkpoint and training from step 2500 to 30000; first Dreamer update logged and episodes advancing beyond 2600.
**Verified by:** Complete evaluation JSON files, `outputs/overnight_traffic_queue/status.txt`, trainer log and live watcher. Five controller/watcher tests and six existing trainer tests passed, lint/diff checks clean. At 01:51: GPU 55 C, 7.3 GB, available RAM 20.2 GB, disk 91 GB. Early watcher rates include random prefill and are optimistic.
**Lesson:** Validate old results before a new overnight run, and verify the actual trainer starts rather than only the detached launcher.
**Interview angle:** What did the fresh evaluation establish before the curriculum retry?

## 2026-10-05 11:19 - Overnight medium curriculum finished; all promotion gates failed
**Type:** result
**Stage:** Roadmap (2) - traffic curriculum

**What happened:** Queue ended 06:10:27 after ~4 h 30 min. All arms completed 30k medium steps; no hard stage ran. Original medium successes / selected validation successes: CNN 3/20 -> 11/30; custom JEPA 5/20 -> 4/30; V-JEPA2 13/20 -> 4/30. Selected collisions: 19/30, 26/30, 26/30. Simple retention: 10/10, 10/10, 8/10 (V-JEPA2 two timeouts); no lane violations. One JEPA CARLA failure recovered.
**Cause:** Open; traffic negotiation remains unsolved despite mostly retained turning skill. One seed and unequal baseline/validation episode counts do not establish a reliable CNN gain.
**How we handled it:** Audited all histories and nine complete evaluations. Best snapshot steps verified: CNN 30k, JEPA 27.5k, V-JEPA2 7.5k; all latest checkpoints 30k. Final weights finite; saved Adam counters show <=15 skipped model updates / 13,750 attempts despite recurring NaN/Inf gradient aggregates. Proposed one CNN BEV medium diagnostic before more camera sweeps; not launched or adopted as a benchmark change.
**Verified by:** `reports/traffic_curriculum_assessment_20261005.md`, queue results/status, derived assessment_audit.json and assessment_curves.png; host anchored process check found no active jobs. Watcher 100%, including skipped hard stages.
**Lesson:** A failed curriculum needs a diagnostic that separates traffic visibility from the ability to learn yielding; extending every arm unchanged has weak support.
**Interview angle:** What did the overnight curriculum establish, and what should come next?

## 2026-10-05 11:19 - Arrival success can overlap collision in traffic evaluation
**Type:** bug
**Stage:** Evaluation / metric interpretation

**What happened:** V-JEPA2 original medium baseline reported 13/20 success and 8/20 collision; one episode has both flags. Collision-free arrivals are 12/20. Its selected 7.5k training evaluation also reports 50% success plus 60% collision. All selected medium validation successes are collision-free.
**Cause:** CarDreamer terminal conditions are independent; evaluator defines success as destination_reached and labels it before collision. Thus success is an arrival metric, not automatically a safe-arrival metric.
**How we handled it:** Preserved historical definitions and raw files; recorded collision-free counts as supplementary audit data. Safer success/selection is a proposed protocol change, requiring user agreement before implementation.
**Verified by:** Original V-JEPA2 baseline episode JSON, medium eval history, scripts/evaluate_agent.py terminal handling and CarDreamer carla_wpt_env.py conditions.
**Lesson:** Report joint outcomes when terminal flags can co-occur; never infer safety from destination arrival alone.
**Interview angle:** Why can traffic success and collision percentages sum to more than 100%?

## 2026-10-05 11:35 - User authorized and started the bounded CNN BEV traffic diagnostic
**Type:** decision
**Stage:** Roadmap (2) - diagnosing traffic learning

**What happened:** User approved the proposed run and requested a live bar. Launched one detached controller at 11:31:46 under systemd-inhibit; baseline finished 11:33:24, then CNN training started. Original BEV medium baseline: 11/30 destination successes, 19/30 collisions, no lane violations; all 11 successes collision-free. Trainer confirmed initialization from the solved 150k-step BEV checkpoint and advanced to 2,642/30,000 steps after random prefill.
**Cause:** Both direct hard and medium camera curricula remained poor; test whether full BEV traffic visibility helps learn negotiation before another three-arm sweep.
**How we handled it:** New scripts/run_bev_traffic_diagnostic.py reuses the existing controller's lock, health guards, owned-client STOP and checkpoint capture. Medium only, CNN seed 42, BEV/discrete; train_ratio 512, reward time penalty 0.1 unchanged. Fresh replay/optimizers, 30k cap including 2.5k prefill, eval every 2.5k, early stop after three >=90% evals. Baseline/selected validation 30 episodes each, simple retention 10, eval stream 2026, three-hour wall-clock cap. No hard training or jobs/*.sh edits.
**Verified by:** Five queue safety/selection/watcher tests passed, lint/diff checks clean, source contract checked on CPU, host check confirmed idle before launch; trainer initialization/update log and progressing episode records. Queue: outputs/bev_traffic_diagnostic_queue; run: outputs/diagnostics/bev_traffic_medium_seed42.
**Lesson:** A diagnostic should change one source of information, preserve the reference task/reward, and measure its own starting policy before training.
**Interview angle:** How did you test whether traffic visibility was limiting learning?

## 2026-10-05 16:24 - Resumed the paused BEV diagnostic from its saved checkpoint
**Type:** decision
**Stage:** Roadmap (2) - diagnosing traffic learning

**What happened:** User requested resume after a pause. Queue records show manual STOP at 12:36:34 and trainer checkpoint saved at 24,969 steps before exit; validation and retention were skipped. Host anchored process check found no active CARLA job. Restarted the same controller at 16:24:02; trainer confirmed --resume at 24,969 and advanced to 25,000/30,000.
**Cause:** User-directed pause and resume; the interrupted training stage's FAIL status reflects the manual stop, not a completed policy evaluation.
**How we handled it:** Preserved earlier logs and metadata under outputs/bev_traffic_diagnostic_queue/history/before_resume_20261005_162349. Controller reacquired its queue lock before clearing STOP; reused completed baseline, replay and saved optimizer state. Target remains 30k, reward/actions/BEV observation unchanged, with automatic selected-checkpoint validation and simple-turn retention afterward. Restarted the three-hour controller cap; no jobs/*.sh edits or duplicate trainers.
**Verified by:** CPU checkpoint step/contract inspection, status.txt, resumed trainer log and live watcher at 83.3%. Best checkpoint remains the earlier selected 5k snapshot pending later evaluations.
**Lesson:** Resume a paused experiment from its latest recovery checkpoint while retaining its independently selected best snapshot and completed baseline.
**Interview angle:** How did you resume training without discarding progress or duplicating jobs?

## 2026-10-05 17:01 - BEV traffic diagnostic completed with improvement on two evaluation streams
**Type:** result
**Stage:** Roadmap (2) - diagnosing traffic learning

**What happened:** BEV adaptation completed 30k steps at 16:39:28; validation/retention ended 16:42:09. Stream 2026 medium successes: original 11/30 -> selected 15/30, collisions 19/30 -> 15/30. New post-run stream 2027 checks ended 16:56:10: original 9/30 -> selected 20/30, collisions 21/30 -> 10/30. Pooled successes 20/60 -> 35/60, collisions 40/60 -> 25/60; all successes collision-free. Simple retention 10/10; no lane violations. Promotion still fails.
**Cause:** Open; measurable gains in this one-seed BEV diagnostic support further bounded learning, but do not establish generalization or isolate camera perception as the only bottleneck.
**How we handled it:** Verified best checkpoint 27.5k, latest 30k, finite weights; reviewed episode/action histories. Last two training evals were 70%; all three collisions in the final ten eval episodes had no braking in the last second. Proposed a single BEV continuation to total 60k with unchanged reward and fixed cap, not launched. Published reference budgets for dense traffic are ~150k–200k under different settings.
**Verified by:** reports/bev_traffic_assessment_20261005.md; complete diagnostic/holdout JSON; derived audit/plot/action summaries. New evaluation used a fixed preselected candidate; no training or benchmark changes. Host check confirmed the original queue was finished before evaluations.
**Lesson:** Validate late learning on another stream before interpreting a noisy 70% training evaluation; report retained skills and remaining collisions together.
**Interview angle:** Did the BEV diagnostic improve traffic performance?

## 2026-10-05 17:01 - TTC SAFE telemetry excludes crossing vehicles
**Type:** bug
**Stage:** Evaluation / traffic risk diagnostics

**What happened:** Collision-video HUD displays TTC SAFE. The vendor TTC candidate filter rejects different roads or lanes, so crossing vehicles can be omitted; get_ttc returns zero when no eligible target is found. A module-isolated reproduction included a same-road lead vehicle and excluded a nearby other-road target.
**Cause:** TTC utility models vehicles ahead on the same lane, not general intersection conflicts; distance divided by relative-speed magnitude is not a crossing-path predictor. Evaluator suppresses TTC errors and does not distinguish unavailable data clearly.
**How we handled it:** Recorded the limitation and proposed crossing-path risk telemetry plus brake/actor/reward traces. No running job scripts or benchmark definitions changed. This TTC is not an input to the current learned policy/reward and the shield is disabled, so the finding affects diagnosis rather than explaining all policy failures.
**Verified by:** toolkit/carla_manager/utils.py is_vehicle_ahead/get_ttc, scripts/evaluate_agent.py, validation_failure_strip.jpg and ttc_filter_audit.json under outputs/bev_traffic_diagnostic_queue.
**Lesson:** Match risk telemetry to the maneuver being evaluated; a lead-vehicle TTC metric can miss intersection conflicts.
**Interview angle:** Why did the video say SAFE immediately before a collision?

## 2026-10-05 17:23 - Fixed completed-run detection before extending the BEV budget
**Type:** bug
**Stage:** Diagnostic queue lifecycle

**What happened:** Before launching the user-approved 60k continuation, inspection found train_bev treated the existence of metrics.json as completion, so the finished 30k run would skip training under a higher requested budget. Reusing its old queue would also reuse cached validation files for the earlier candidate.
**Cause:** Training completion did not account for the requested step target; evaluation cache paths are tied to the queue rather than checkpoint identity.
**How we handled it:** Compare saved env_steps with args.steps, while honoring prior early stops. Added continuation and same-target regression tests. Use a new outputs/bev_traffic_continuation_60k_queue for fresh final evaluations; copy only the unchanged original baseline. No jobs/*.sh changes.
**Verified by:** Seven queue/continuation tests passed, lint and diff checks clean; actual trainer launched with --resume --steps 60000 rather than marking training DONE.
**Lesson:** A completion marker is only valid for its recorded budget, and candidate evaluation caches need distinct provenance.
**Interview angle:** How did you avoid accidentally reusing completed results when extending training?

## 2026-10-05 17:23 - User authorized BEV continuation from 30k to 60k; trainer verified running
**Type:** decision
**Stage:** Roadmap (2) - improving traffic negotiation

**What happened:** User approved the proposed continuation and requested a live bar. Host check confirmed idle hardware. Controller started at 17:21:56 after one automatic approval-review timeout and a successful retry; the rejected attempt did not execute. Trainer confirmed resume at 30,000 and advanced to 30,271/60,000.
**Cause:** The prior BEV agent improved from 20/60 to 35/60 successes across two evaluation streams, with intact turning skill and late 70% training evals; one bounded continuation tests whether learning continues.
**How we handled it:** Archived latest/best weights, selection metadata and histories at outputs/diagnostics/bev_traffic_medium_seed42/history/at_30000. Reuse replay/optimizer and original baseline; reward time=0.1, BEV/discrete, medium traffic, train_ratio=512, seed42 and eval stream2026 unchanged. New queue outputs/bev_traffic_continuation_60k_queue; detached systemd-inhibit, three-hour cap, health/STOP guards; automatic 30-episode selected validation and ten simple retention episodes afterward. No hard stage.
**Verified by:** CPU checkpoint inspection (latest30k, best27.5k), trainer resume/update records, live watcher at50.5%; GPU52C,7.7GB, available RAM19.7GB. No random prefill repeated. Training source/config and 30k snapshot provenance retained.
**Lesson:** Preserve the previous evaluated candidate before its mutable best checkpoint advances, and verify an extended run actually resumes before leaving it unattended.
**Interview angle:** How did you preserve the previous experiment while continuing its learning?

## 2026-10-05 20:04 - BEV 60k continuation improved traffic handling; stream variability remains
**Type:** result
**Stage:** Roadmap (2) - traffic negotiation

**What happened:** Continuation finished training 18:45:41 and scheduled checks 18:48:22, approximately 86 minutes after launch. Selected/latest checkpoints both 60k, finite weights. Scheduled medium success 25/30, collisions 5/30; simple retention 10/10 with no collisions. The configured gate passed. Additional evaluations completed 19:55:15: medium stream2027 25/30, stream2028 17/30; hard zero-shot stream2028 21/30 with nine collisions. No hard training occurred.
**Cause:** More learning helped this one-seed BEV agent, but remaining failures and their variation across streams are open. Three matched stream IDs do not imply paired identical scenarios or independent training replicas.
**How we handled it:** Evaluated the archived 27.5k candidate on stream2028 (12/30 arrivals, 17 collisions, one timeout). Across streams2026/2027/2028, 30k-budget vs60k arrivals were 47/90 -> 67/90, collisions 42/90 -> 23/90; all standalone arrivals collision-free and no lane failures. Preserved formal gate result while recording that the newest medium stream fails it. Published reports/bev_60k_assessment_20261005.md and audit/plot; no new training launched.
**Verified by:** Complete episode JSONs in continuation, bev_60k_assessment_seed2028 and bev_medium_budget_comparison_20261005 queues; derived completed_assessment_audit.json. Final three training evals80/80/90%; optimizer counters show <=21 skipped updates across28,750 attempts. Host anchored check found no active jobs after assessment.
**Lesson:** Extra training can produce useful gains while a successful scheduled gate still overstates performance on another traffic stream.
**Interview angle:** Did the additional 30k steps improve traffic driving, and is the task solved?

## 2026-10-05 20:04 - Low-speed collision penalty can leave a crash with near-success return
**Type:** bug
**Stage:** Reward diagnosis / next experiment

**What happened:** In90 continuation training collisions, nine had positive episode return and three had terminal total reward above-1. A non-arrival crash at94% completion earned+299.685 with terminal total reward-0.855. The final60k evaluation's collision earned+296.515, terminal-0.372, despite six brake commands in its last second.
**Cause:** carla_wpt_env.py:95 scales the collision component by post-step ego speed (-30*abs(speed)); it approaches zero for a stationary ego. Most crashes were already strongly penalized (median terminal total reward-161.36). Positive undiscounted return does not prove actor exploitation or explain all failures.
**How we handled it:** Saved collision_reward_audit.json and behavior_audit_30k_vs60k.json, documenting total rewards versus reward components. Proposed, not applied: two30k pilots from the same60k weights, fresh replay/reset optimizers in both; unchanged control versus additional fixed collision cost300, all other settings held constant. Compare final checkpoints under canonical benchmark reward on predeclared new streams and check retention. Estimated~3h; reward change/training require user approval.
**Verified by:** Vendor reward source, post-step reward ordering, timestamp-matched recorded train/eval NPZ rewards/actions and episode flags; reports/bev_60k_assessment_20261005.md. No reward/benchmark code changed and no further training launched.
**Lesson:** Penalizing ego speed at impact can underprice being hit while stopped; isolate the hypothesis with a control before assuming reward changes will improve safety.
**Interview angle:** Why propose a fixed collision-cost ablation instead of another unchanged traffic sweep?

## 2026-10-05 20:59 - User approved the controlled collision-cost pilot; queue launched
**Type:** decision
**Stage:** Roadmap (2) - reward ablation

**What happened:** User authorized the recommended run and requested a live bar. Launched detached scripts/run_collision_cost_ablation.py at20:58:44 under systemd-inhibit. Queue outputs/collision_cost_ablation_queue is measuring the60k reference on predeclared streams2029/2030 before the two sequential pilots; evaluator verified running on CUDA with canonical reward and no shield.
**Cause:** The60k BEV policy improved to67/90 medium arrivals but retained23/90 collisions; test the measured low-speed collision penalty limitation against an equal-budget unchanged-reward control.
**How we handled it:** Archived finite60k weights as source_60k.pt with SHA256, immutable protocol and code/config/patch snapshots. Both pilots start from those same weights with fresh replay/optimizers,30k steps including2.5k prefill, seed42, medium/BEV/discrete, train_ratio512. Control cost0 versus once-per-episode fixed cost300; other reward terms unchanged. Disable early stopping and compare final checkpoints. Automatic reference/control/variant checks:30 medium episodes on each of2029/2030,10 simple episodes on2029; evaluator explicitly restores fixed cost0. No hard training, no jobs/*.sh edits. Expected~3h; cap4h with STOP and health guards.
**Verified by:** 39 relevant CPU tests passed, one GPU-only skipped; lint/diff/patch reverse checks clean. Both resolved configs and source weights verified before launch, host anchored process check idle, GPU40C and87GB disk free. Reference evaluator log confirms canonical overrides; live watcher shows queue0/11 stages and~3h08m ETA. Protocol reports/collision_cost_ablation_v1.md.
**Lesson:** An approved reward hypothesis needs an unchanged control, equal budgets, frozen provenance and canonical scoring before launch.
**Interview angle:** How did you turn the collision-reward finding into a controlled training experiment?

## 2026-10-05 20:59 - Guarded fixed cost against latched collision telemetry and inherited evaluation reward
**Type:** bug
**Stage:** Reward implementation / evaluation contract

**What happened:** Implementation inspection found CollisionHandler._data remains latched until reset, while CarlaBaseEnv.step checks terminal flags before refreshing observations and calculating reward. A fixed cost applied on every collision-flagged reward step could charge twice. Separately, evaluate_agent.py inherits checkpoint training overrides, so a variant evaluated without an explicit reset would receive modified benchmark returns.
**Cause:** Sensor/terminal ordering and evaluator override inheritance; existing terminal semantics were not changed by this experiment.
**How we handled it:** CarlaWptEnv applies collision_fixed once at first detection and clears the guard on reset; default0 preserves historical reward behavior. Saved vendor edits in patches/cardreamer_compat.patch. Collision-only reward/speed traces go to collision_events.jsonl without adding policy observations. Controller appends canonical time0.1/fixed0 overrides, saves evaluation provenance, and rejects mismatched reward/source resume checkpoints or evaluation caches.
**Verified by:** Tests execute the actual vendor reward at speeds0/0.01/4m/s, repeated detection/reset, non-collision and legacy defaults; validate last-override precedence for simple/medium/hard tasks, diagnostic traces and wrong-reward resume refusal. Running reference evaluator prints canonical overrides; complete39-test targeted suite passed.
**Lesson:** Test event persistence and evaluation inheritance when adding a terminal cost, rather than assuming one sensor event means one reward application.
**Interview angle:** What prevented the collision-cost experiment from silently changing its treatment or benchmark?

## 2026-10-05 21:06 - Reference measured; control pilot verified learning beyond prefill
**Type:** milestone
**Stage:** Collision-cost pilot launch

**What happened:** Canonical reference checks completed by 21:03:15: medium streams 2029/2030 gave 26/30 and 25/30 collision-free arrivals with four/five collisions; simple retention 10/10, no collisions. Control trainer launched next, confirmed initialization from the archived 60k weights, fresh 2.5k prefill and training to 30k. At 21:06 it reached 3,124 steps; fixed300 remains queued.
**Cause:** Approved experiment progressing normally; baseline stream variation reinforces the need to compare both pilots against the same measured starting policy.
**How we handled it:** Preserved baseline metrics in queue results.json. Verified one active trainer with medium/BEV/discrete, seed42, ratio512, fixed cost0 and no early stopping; no duplicate launch or jobs/*.sh edit. Queue will run the fixed-cost pilot and canonical validation automatically. Both share the same immutable source weights, with fresh pilot state.
**Verified by:** Trainer initialization/training/update logs and advancing episode records, anchored host process check, reference episode JSONs, matching code snapshot hashes. Watcher 3/11 stages, 8.1% overall, approximately 2h50m remaining; GPU57C/7.7GB, available RAM20.4GB, disk86GB.
**Lesson:** Confirm both source initialization and actual step advancement before treating a detached training launch as successful.
**Interview angle:** How did you verify the approved pilot was actually training?

## 2026-10-05 21:06 - Corrected queue ETA during startup and random prefill
**Type:** bug
**Stage:** Live progress reporting

**What happened:** Watcher initially showed ~1h38m total at control startup, omitting the current pilot when its rate was unknown; after prefill it estimated40.7 steps/s and11 minutes for training. Actual subsequent learning rate was approximately6.3 steps/s.
**Cause:** Unknown current ETA was treated as zero, and the random-prefill records report step0 until the runner advances the logger in one jump after prefill. Rate estimation included that jump as learned environment steps.
**How we handled it:** scripts/watch_queue.py now uses the planned remaining duration when the active stage has no measured rate and excludes step-zero prefill records from the rate window. Read-only watcher changes leave the active training protocol and source hashes intact. No jobs/*.sh edits.
**Verified by:** Seven queue/watcher tests passed, including startup ETA and prefill-rate regressions; lint/diff clean. Live snapshot then showed3,124/30,000 at6.3 steps/s, current ETA1h11m and queue ETA~2h50m.
**Lesson:** A progress estimator must account for unknown current work and distinguish random prefill from learning throughput.
**Interview angle:** Why did the live bar briefly predict an unrealistically short training time?

## 2026-10-05 21:18 - Resumed the user-stopped collision-cost pilot from step 3,998
**Type:** decision
**Stage:** Collision-cost pilot / resume

**What happened:** User reported stopping the run and requested resume. Queue records show manual STOP at 21:08:13 and exit at 21:08:15; the control checkpoint saved at 3,998 steps. Host anchored process check found no active trainer, evaluator, controller or CARLA instance. Restarted the same queue at 21:17:38; trainer confirmed --resume at 3,998 and episode records advanced to 4,136 by 21:18.
**Cause:** User-directed pause, not a completed training failure; FAIL control_seed42 records the interrupted process exit.
**How we handled it:** Verified finite checkpoint weights, optimizer state, 14 saved replay episodes and matching code/source/reward contract. Archived prior logs/status/STOP/metadata under outputs/collision_cost_ablation_queue/history/before_resume_20261005_211717. Controller reacquired its lock before clearing STOP and reused all three completed reference evaluations. Control continues to 30k with replay/optimizers intact; fixed300 remains queued for its own fresh 30k pilot. Reward, actions, task, evaluation streams and final-checkpoint selection unchanged; four-hour controller cap restarts on resume. No jobs/*.sh edits or duplicate queues.
**Verified by:** CPU checkpoint and SHA256 inspections, resumed trainer initialization/update logs and advancing episode JSON; live watcher at 13.8% control / 9.6% overall, 3/11 stages and estimated 2h52m remaining. GPU54C/7.8GB, available RAM20.7GB, disk86GB. No random prefill repeated.
**Lesson:** A manually stopped experiment should resume its exact checkpoint and reuse completed checks while preserving the stop's history.
**Interview angle:** How did you resume the collision-cost pilot without restarting training or mixing reward versions?

## 2026-10-06 00:21 - Collision-cost pilot stopped early; completed control showed no pooled success gain
**Type:** result
**Stage:** Collision-cost pilot assessment

**What happened:** After resume, control completed 30k at 22:37:01, with one recovered native CARLA timeout. Canonical medium successes were 23/30 and 28/30, pooled 51/60, matching the 60k reference's 26/30 and 25/30. Control had eight collisions and one lane failure versus reference nine collisions/no lane failures; both retained 10/10 simple turns. Fixed300 stopped at 21,278/30,000 after a controller exception at 23:34:44, so the planned ablation is unfinished.
**Cause:** Fixed-cost interruption was infrastructure failure, not a learning gate. Control's 100% final ten-episode training evaluation did not translate into a pooled standalone gain; remaining driving differences are open.
**How we handled it:** Evaluated an immutable partial fixed-cost snapshot under canonical reward (fixed cost explicitly zero), same streams and no shield; checks completed 00:17:29. Partial safe arrivals 24/30 and 24/30 =48/60, with 11 collisions, one lane failure, no timeouts and 10/10 retention. Raw arrivals are49/60 because one also collided. Kept original results.json completed:false and labeled unequal-budget comparisons provisional. Recommended finishing the remaining8,722 steps (~30min including checks), not another new reward sweep; no training launched.
**Verified by:** Complete episode JSONs, finite checkpoint weights and Adam counters, assessment_audit.json/curves under outputs/collision_cost_ablation_assessment_20261006; reports/collision_cost_assessment_20261006.md. Variant traces show71 exactly-once fixed-cost applications; zero positive returns among47 failed post-prefill training collisions. Host anchored check found no active jobs after short evaluations.
**Lesson:** A corrected reward component does not establish better driving, and an ended queue is not an equal-budget completed experiment.
**Interview angle:** Did the fixed collision penalty improve traffic driving?

## 2026-10-06 00:21 - Fixed fatal CPU telemetry timeout and incomplete-stage reporting
**Type:** bug
**Stage:** Unattended queue reliability / provenance

**What happened:** My Queue.check_health let a ten-second sensors timeout abort the fixed300 trainer despite GPU53C, available RAM20.2GB and disk85GB. Exception cleanup saved its checkpoint at21,278. The final banner appeared without an active-stage FAIL; stale SKIP records from the prior pause suppressed new unattempted-stage records. Original experiment results remained completed:false and lacked comparison_summary.md.
**Cause:** Display-only CPU telemetry was treated as mandatory; abnormal run exits bypassed failure status and final accounting scanned the whole historical log rather than the current invocation.
**How we handled it:** CPU read timeout/OSError now records unavailable telemetry while GPU/RAM/disk guards continue; exceptions record FAIL and error, and remaining-stage accounting uses the current invocation. Added explicit runtime recovery matching exact old/new hashes for queue/controller repairs while refusing training/reward/budget changes. Original manifest/snapshots preserved; recorded patch/recovery manifest and separate effective runtime snapshot prepared. STOP remains intact; no training restarted.
**Verified by:** 29 targeted tests passed including timeout/missing sensors, retained GPU overheat STOP, exception status, resume and recovery provenance; lint/diff clean. Recovery preflight accepted the repair and verified the original experiment.json and STOP were unchanged.
**Lesson:** Optional monitoring must not discard an unattended learning budget, and recovery must preserve both the original protocol and the reason code changed.
**Interview angle:** Why did the overnight reward experiment end before its budget, and how was recovery made safe?

## 2026-10-06 00:45 - User approved bounded overnight continuation with replay and validation safeguards
**Type:** decision
**Stage:** Overnight BEV improvement

**What happened:** User requested a long unattended training run using current findings and a live progress bar. Prepared scripts/run_overnight_bev_improvement.py: finish the remaining8,722-step fixed300 pilot, compare completed candidates on known weak medium2028, then continue the stronger retained-turning policy for up to120k additional steps in30k blocks, eight-hour cap.
**Cause:** The control matched the60k reference at51/60 canonical safe arrivals; partial fixed30048/60 did not establish improvement. Earlier30k-to60k continuation improved47/90 to67/90, while fresh-state control did not add a pooled gain.
**How we handled it:** Preserve replay/Adam state in a separate branch; freeze original assessed checkpoints. Validate each block across2028/2029/2030 plus simple retention, retain a best checkpoint independently of latest, stop after three blocks without accepted gain or severe regression. Canonical holdout checks on3031/4031, simple3031 and zero-shot hard5031 finish the run. No stronger penalty or hard-task training. Fixed optional CPU monitoring repair remains active; scoped STOP, health guards and crash retries retained. reports/overnight_bev_protocol_20261006.md records the protocol.
**Verified by:** 34 targeted tests passed, including a full mocked controller regression proving it keeps baseline weights and evaluates separate final streams, resume-not-skip of stale metrics, replay independence, canonical scoring and incremental progress. Lint/diff clean. Runtime recovery preflight accepted exact repair hashes, finite optimizer-bearing control30k and fixed30021,278 checkpoints, original manifest and STOP preserved before launch.
**Lesson:** A long budget needs preserved learning state, a measurable safety criterion and protected baseline; increased training alone cannot guarantee improvement.
**Interview angle:** How was the overnight run designed to avoid repeating an uninformative continuation?

## 2026-10-06 00:48 - Overnight queue launched and resumed trainer advanced beyond saved checkpoint
**Type:** milestone
**Stage:** Overnight BEV improvement / launch

**What happened:** Detached inhibited queue started00:45:33. Its owned recovery reused all reference/control checks and resumed fixed300 from21,278/30k. New episode records reached21,827 by00:47; the first stage is6.3% through its remaining8,722 steps, queue0.5% and ETA~7h05m.
**Cause:** User-approved training underway; original pilot completion precedes evidence-based source selection and long continuation.
**How we handled it:** Verified one active trainer (PID18822), outer controller18381 and its intentional recovery child18383, one CARLA server and active sleep/idle/lid inhibitor18377. New STOP path outputs/overnight_bev_20261006/STOP propagates to the recovery queue while active. Source pilot state preserved on resume; no prefill repeated, jobs scripts unchanged.
**Verified by:** Trainer logs show Training from env step21278 to30000 and advancing train episodes21422/21548/21678/21827; host anchored process check and systemd-inhibit --list. Watcher snapshot GPU55C/7.2GB, available RAM20.8GB, disk85GB; controller manifest/source snapshots saved. Current episodes are startup progress, not validation evidence of improvement.
**Lesson:** Verify advancement and process ownership before handing off a detached overnight job.
**Interview angle:** How did you verify the overnight controller had really resumed learning?

## 2026-10-06 01:20 - Selected overnight continuation source
**Type:** decision
**Stage:** Approved overnight BEV continuation

**What happened:** Completed the 30k pilot comparison and selected control at step30000, validation rank (23, 78, -10, -1). Cloned replay and optimizer state; original checkpoints preserved.
**Cause:** Stage outcome during approved continuation; any newly reported exception needs a separate root-cause assessment.
**How we handled it:** See /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json and status.txt; canonical reward evaluation, original source preserved.
**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.
**Lesson:** Preserve the best validated state when continuing an imperfect policy.
**Interview angle:** How did the overnight continuation handle this stage?

## 2026-10-06 02:48 - Overnight BEV block1 completed
**Type:** result
**Stage:** Approved overnight BEV continuation

**What happened:** Reached step60000; medium safety rank (27, 82, -6, -1), simple safe arrivals 10/10. Accepted=True; best checkpoint remains step60000. These streams are checkpoint-selection diagnostics, not holdouts.
**Cause:** Stage outcome during approved continuation; any newly reported exception needs a separate root-cause assessment.
**How we handled it:** See /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json and status.txt; canonical reward evaluation, original source preserved.
**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.
**Lesson:** Preserve the best validated state when continuing an imperfect policy.
**Interview angle:** How did the overnight continuation handle this stage?

## 2026-10-06 04:20 - Overnight BEV block2 completed
**Type:** result
**Stage:** Approved overnight BEV continuation

**What happened:** Reached step90000; medium safety rank (29, 89, -1, 0), simple safe arrivals 10/10. Accepted=True; best checkpoint remains step90000. These streams are checkpoint-selection diagnostics, not holdouts.
**Cause:** Stage outcome during approved continuation; any newly reported exception needs a separate root-cause assessment.
**How we handled it:** See /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json and status.txt; canonical reward evaluation, original source preserved.
**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.
**Lesson:** Preserve the best validated state when continuing an imperfect policy.
**Interview angle:** How did the overnight continuation handle this stage?

## 2026-10-06 05:57 - Overnight BEV block3 completed
**Type:** result
**Stage:** Approved overnight BEV continuation

**What happened:** Reached step120000; medium safety rank (25, 80, -9, -1), simple safe arrivals 10/10. Accepted=False; best checkpoint remains step90000. These streams are checkpoint-selection diagnostics, not holdouts.
**Cause:** Stage outcome during approved continuation; any newly reported exception needs a separate root-cause assessment.
**How we handled it:** See /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json and status.txt; canonical reward evaluation, original source preserved.
**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.
**Lesson:** Preserve the best validated state when continuing an imperfect policy.
**Interview angle:** How did the overnight continuation handle this stage?

## 2026-10-06 07:33 - Overnight BEV block4 completed
**Type:** result
**Stage:** Approved overnight BEV continuation

**What happened:** Reached step150000; medium safety rank (0, 1, -13, -4), simple safe arrivals 0/10. Accepted=False; best checkpoint remains step90000. These streams are checkpoint-selection diagnostics, not holdouts.
**Cause:** Stage outcome during approved continuation; any newly reported exception needs a separate root-cause assessment.
**How we handled it:** See /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json and status.txt; canonical reward evaluation, original source preserved.
**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.
**Lesson:** Preserve the best validated state when continuing an imperfect policy.
**Interview angle:** How did the overnight continuation handle this stage?

## 2026-10-06 07:46 - Overnight BEV run completed
**Type:** result
**Stage:** Approved overnight BEV continuation

**What happened:** Finished 4 blocks; selected step90000. Baseline and selected canonical holdout results: /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/summary.md. Medium streams3031/4031, simple3031 and zero-shot hard5031 were not used for training checkpoint selection; no hard-task training occurred.
**Cause:** Stage outcome during approved continuation; any newly reported exception needs a separate root-cause assessment.
**How we handled it:** See /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json and status.txt; canonical reward evaluation, original source preserved.
**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.
**Lesson:** Preserve the best validated state when continuing an imperfect policy.
**Interview angle:** How did the overnight continuation handle this stage?

## 2026-10-08 13:22 - Overnight assessment: modest separate-stream gain; preserved90k model survives late collapse
**Type:** result
**Stage:** BEV traffic diagnostic / completed overnight assessment

**What happened:** Oct6 queue completed all32 stages in7h00m59s and added120k steps. It selected control30k; medium selection safe arrivals78/90→82/90→89/90→80/90→1/90 at30/60/90/120/150k. Best90k preserved. Final separate-stream medium improved56/60→58/60, collisions3→2, lane departures1→0; simple10/10 retained; zero-shot hard26/30→27/30, collisions4→3.
**Cause:** Longer continuation found a better checkpoint but did not improve monotonically. Separate-stream gains are two medium episodes and one hard episode; one training seed and stochastic, unpaired streams do not establish a robust large gain.
**How we handled it:** Independently audited raw flags, summaries, requests/manifests, hashes and finite model/Adam tensors; wrote reports/overnight_bev_assessment_20261008.md and tracked audit JSON, generated selection/final figure. Updated handoff. Keep outputs/overnight_bev_20261006/best.pt (step90k), never blindly resume collapsed latest150k. No training started.
**Verified by:** scripts/assess_overnight_bev.py validated36 evaluation sets/900 episodes and ten checkpoint files; original code hashes match. Full CPU suite166 passed/one GPU-only skipped; changed code lint and vendor reverse-patch checks passed. Anchored host check found no active jobs.
**Lesson:** Evaluate preserved checkpoints on separate streams; the final training checkpoint can be much worse than an earlier one.
**Interview angle:** Did the overnight training deliver an improvement, and which model should be retained?

## 2026-10-08 13:22 - Fixed-cost pilot completed without safety gain; stopping collapse needs diagnosis
**Type:** bug
**Stage:** Reward hypothesis and late-training stability assessment

**What happened:** Completed fixed300/control/reference checks all51/60 safe medium arrivals; fixed300 had9 collisions versus control8/one lane failure, allsimple10/10. Overnight therefore trained with fixed cost0. At150k, medium had72/90 timeouts and simple0/10 with near-zero speed; training eval dropped from100% arrivals/10% collisions at147.5k to0% arrivals at150k.
**Cause:** Exact stopping-collapse cause open. All checkpoint weights/Adam tensors finite, and counters advanced by59,961 world-model/59,992 actor/59,988 value updates over120k steps. Intermittent AMP gradient overflows, including actor inf at149k, do not prove causation. Checkpoints omit scaler/RNG state and milestone replay snapshots; evaluator seed labels increment without per-episode reseeding, while Traffic Manager seed stays2024.
**How we handled it:** Kept failure checkpoint/logs as evidence and documented limitations without changing benchmark, seeds or rewards. Recommend reproducing best90k across additional streams, then diagnosing the final2.5k steps and adding frequent retention/rollback and complete recovery provenance before any short controlled continuation. User requested local commit of the reviewed experiment work/results; no push requested or attempted.
**Verified by:** Completed ablation results, raw final episode JSONs, finite optimizer audit, training eval curve, tools.py optimizer/checkpoint inspection and evaluate_agent.py/VehicleManager seed inspection. See reports/overnight_bev_assessment_20261008.md.
**Lesson:** Correct collision accounting is not proof of policy benefit, and low collision rate can hide a policy that no longer drives.
**Interview angle:** Why avoid another long continuation or larger collision penalty immediately?

## 2026-10-08 14:47 - Research presentation and portable driving showcase completed
**Type:** milestone
**Stage:** Research communication / current experiment evidence

**What happened:** Created a 30-slide editable PPTX and matching static PDF, covering abstract, seven-paper literature tables, methodology/architecture, datasets, metrics, results, gaps and proposed improvements. Packaged six PNG/PDF/CSV graphs, three SVG/PNG diagrams and 14 original MP4s, with five clips embedded in the PPT.
**Cause:** User requested a simple, technically accurate presentation and an organized driving-video showcase of the project's current state.
**How we handled it:** Built outputs/showcase_20261008 with an offline index.html gallery, speaker notes, primary-paper links, raw result snapshots, assessment reports, clip provenance and a rebuild script; packaged outputs/Auto_Thinker_Engine_Showcase_20261008.zip. Kept camera comparisons distinct from privileged-BEV diagnostics and presented the fixed-cost null result and 150k collapse alongside the preserved 90k result.
**Verified by:** Reopened PPT: 30 slides/30 notes, all shapes inside canvas, five internally embedded MP4s; copied and embedded video hashes match originals. Checked 65 local gallery links, 104 bundled file hashes, ZIP CRC integrity, native diagram connectors, SVG parsing, contact sheets and a rendered vector-PDF architecture page. Actual PowerPoint playback remains viewer-dependent; standalone MP4s/gallery provide a fallback.
**Lesson:** A research presentation should preserve experimental boundaries and provenance, with failed episodes visible alongside successful examples.
**Interview angle:** How did you turn experiment artifacts into a reproducible research presentation?

## 2026-10-08 15:13 - Fixed browser playback of showcase recordings; corrected compatibility verification
**Type:** bug
**Stage:** Research showcase / recorded driving media

**What happened:** User reported Firefox's "no video with supported video format and MIME type" and Chrome play doing nothing. The 14 showcase MP4s contained MPEG-4 Part 2 (`mp4v`) from evaluate_agent.py's OpenCV recorder. The previous presentation milestone's file/hash checks did not establish browser playback; that missing check was my mistake.
**Cause:** An MP4 container does not guarantee a browser-supported video codec; copied original recordings were incompatible with the gallery's browser playback path.
**How we handled it:** Changed the showcase generator to transcode presentation copies into H.264 baseline/yuv420p/faststart MP4 and VP9/yuv420p WebM. The gallery offers WebM first and MP4 second; regenerated the five embedded PPT videos and updated clip provenance with separate source/conversion hashes. Original experiment recordings and training code remain intact. Rebuilt the original ZIP plus outputs/Auto_Thinker_Engine_Showcase_20261008_BrowserFixed.zip.
**Verified by:** Native WebDriver BiDi in isolated Firefox 157.0.1 played all 14 actual file:// gallery clips and all 14 direct MP4s with advancing video time. Full frame decoding confirmed source frame counts, dimensions and durations; all 14 MP4s place moov before mdat. Verified 93 gallery links, internal PPT media hashes, ZIP CRC and 120 bundled file hashes. Browser test/report: outputs/showcase_20261008/07_sources/validate_browser_playback.mjs and browser_playback_validation.json. Chrome was unavailable for direct testing.
**Lesson:** Test actual browser decoding and playback, rather than treating a file extension, MIME label or successful thumbnail extraction as proof of compatibility.
**Interview angle:** Why did valid MP4 recordings fail in browsers, and how did you verify the fix?

## 2026-10-08 21:04 - Approved camera navigation redesign; implementation branch created
**Type:** decision
**Stage:** Navigation M0

**What happened:** User approved implementation of the reviewed long-route plan. Created experiment/camera-world-model-navigation from 67b3040, preserving existing dirty journal/presentation work and vendor patches.
**Cause:** Right-turn continuation does not establish reliable multi-junction camera driving; the review identified weak teacher/evaluation controls and missing predictive-planning baselines.
**How we handled it:** Added reports/navigation_plan.md and configs/navigation.yaml as a separate contract. Prioritize PDM-Lite, existing evaluation criteria, encounter data and an early camera imitation baseline; JEPA/causal planning follow controlled comparisons. No legacy reward or benchmark change and no long training launch.
**Verified by:** Branch/status check, live host job check (idle), NVIDIA 5070 Ti 12GB, 79GB free disk. Source audit confirmed temporal action pooling, rectangular-position limitation, upstream force-kill cleanup and expert actor deletion requiring integration controls.
**Lesson:** Establish a measured teacher, benchmark and simple planner before investing in a larger learned stack.
**Interview angle:** Why redesign the project instead of continuing the right-turn agent?

## 2026-10-08 21:28 - M0 native map reload failure and expert anti-stall audit
**Type:** bug
**Stage:** Navigation M0

**What happened:** First 200-step profile aborted before camera frames: CARLA SIGSEGV during load_world(Town03), followed by Python-client SIGABRT after 30s timeout. Pinned PDM-Lite also forces full throttle after 170 stopped ticks (8.5s).
**Cause:** Native map-reload failure on this installation; exact Unreal cause open. The expert's forced movement is explicitly intended to avoid blocked-agent scoring.
**How we handled it:** Preserve outputs/navigation_m0_profile_20261008. Added tracked vendor patch removing forced throttle and replacing expert actor deletion with an explicit invalid-scene error. Move evaluator into an owned child process and start a fresh server directly on each registered map.
**Verified by:** Console/carla.log, upstream autopilot.py/config.py, 18 passing frame/metric/adversarial harness tests. Host check confirmed the failed server exited; no force kill used.
**Lesson:** Native crashes bypass Python finally blocks; simulator ownership belongs outside the native-client worker.
**Interview angle:** What made the stronger expert and reused harness require adaptation?

## 2026-10-08 23:18 - M0 profile passes after adapter and route-generation corrections
**Type:** milestone
**Stage:** Navigation M0 integration

**What happened:** The corrected expert profile produced 200 aligned three-camera frames; measured 13.81 frames/s and 32.46ms p95 expert/control time. Harness route length 1233.36m, completion 8.11%, zero recorded infractions, intentionally truncated at 200 steps.
**Cause:** Prior failures included my adapter accepting a timestamp where upstream calls a no-argument tick, an upstream builder thread left running on failure, and sparse route re-interpolation expanding 1238m to 2251m before driving.
**How we handled it:** Corrected the signature with regression coverage; stop_scenario clears the thread flag. Generate endpoints with the harness's1m planner and reject unexpected length drift before driving. Town03_Opt and Town03 OpenDRIVE hashes match; use the working startup variant. Preserve all prior attempt directories.
**Verified by:** 19 navigation/harness tests; profile/trace/route_validation under outputs/navigation_m0_profile_v2_20261008. All 100 video frames decode as H.264/yuv420p, 1536×288; inspected side-camera orientations. Fresh host check idle before the full bounded route launched.
**Lesson:** Validate actual upstream call signatures and effective routes, not only imports and configuration intent.
**Interview angle:** What did the first multi-camera navigation profile establish?

## 2026-10-08 23:32 - First long expert route completes; summary status bug corrected
**Type:** result
**Stage:** Navigation M0

**What happened:** Modified privileged PDM-Lite completed 1233.361 m in 255.4 simulated seconds/262.895 harness wall seconds: 100% completion, Driving Score 100, zero recorded infractions. 5108 synchronized three-camera frames; 19.37fps overall; 33.84ms p95 expert/control latency excluding rendering/video encoding.
**Cause:** This establishes a working long-route reference and camera/evaluator integration, not learned camera driving or calibrated reliability. My parser incorrectly treated upstream `Perfect` as non-completion.
**How we handled it:** Recognize both `Perfect` and `Completed`; add regression coverage; preserve original summary before rebuilding derived counts. Raw evaluator records remain unchanged. Recorded 2554 H.264 video frames.
**Verified by:** outputs/navigation_m0_long_20261008/audit.json: source/archive hashes, all video frames decoded, complete frame sequence; first 200 ego poses/yaws/controls exactly match the earlier profile. Surrounding-actor determinism remains unmeasured.
**Lesson:** Cross-check wrapper summaries against the evaluator's raw status vocabulary and preserve original evidence when fixing reporting.
**Interview angle:** What does the first successful long drive prove, and what does it leave unproven?

## 2026-10-08 23:32 - Town10HD route export drift caught before driving
**Type:** bug
**Stage:** Navigation M0 multi-town compatibility

**What happened:** Town10HD loaded successfully before Traffic Manager initialization, but the registered 389.22m mission re-interpolated to 160.95m. The agent correctly refused setup with zero frames. My runner incorrectly labeled that attempt evaluation_complete because the upstream harness returned without crashing.
**Cause:** Exporting the planner's snapped path endpoints changed the next shortest-path query; offline replay independently reproduced160.85m. A normal evaluator return does not mean successful agent setup.
**How we handled it:** Plan from the exact serialized original start/end coordinates; preserve failed manifests/attempts. Explicitly fail setup errors and incomplete profiles. Generate a fresh first-qualifying seeded development pair, without driving-outcome filtering. New335.45m Town10HD route passes60 synchronized frames.
**Verified by:** 22 navigation tests including exported-route round-trip and setup-failure regressions; outputs/navigation_m0_town10_v3_20261008. No existing jobs or benchmark routes changed.
**Lesson:** Validate the artifact consumed by the evaluator, not an intermediate planner output; distinguish execution success from policy success.
**Interview angle:** How did route provenance checks prevent a silently easier navigation task?

## 2026-10-08 23:38 - All training-town probes pass; runtime stop-sign coverage corrects XML assumption
**Type:** milestone
**Stage:** Navigation M0

**What happened:** Town01/02/03/04/10HD_Opt each passed 60-frame synchronized camera/expert probes, with no recorded infractions before intentional truncation. Loading maps before Traffic Manager initialization succeeded. Runtime stop-actor counts are 0/0/20/8/3 respectively, contradicting the review's implication that Town03/04 cannot provide stop-sign encounters.
**Cause:** OpenDRIVE signal inventories omit packaged runtime stop actors. The earlier runbook's inference of no Town03 stop opportunities was incorrect.
**How we handled it:** Persist runtime actor inventories; correct runbook/handoff and require lane/trigger-specific encounter denominators before coverage claims. Preserve all failed attempts and deprecated route manifests. No Town05 drive or camera training performed.
**Verified by:** Per-town runtime_sign_inventory.json and audited profiles/videos under outputs/navigation_m0_*_20261008; reports/navigation_m0_assessment_20261008.md/json. Final CPU suite 189 passed/one GPU-only skipped; new-code lint/format and git diff whitespace checks pass.
**Lesson:** Runtime semantics, not only source-map XML, determine which traffic rules can actually be exercised.
**Interview angle:** Why did the stop-sign coverage conclusion change after simulator inspection?

## 2026-10-08 23:38 - Observable-world repeatability passes the short prefix test
**Type:** result
**Stage:** Navigation M0 counterfactual prerequisites

**What happened:** Two independent 200-frame Town03 seed42 probes matched exactly: ego controls/pose, poses and velocities of 19 dynamic actors including ego, and traffic-light states/elapsed phases. Maximum measured differences were zero.
**Cause:** Fixed synchronous20Hz stepping and fresh seeded scenario/traffic initialization reproduce this10s prefix on this installation.
**How we handled it:** Added audit-only world_trace.jsonl and comparator with declared tolerances; actor correspondence uses seeded creation order and requires matching types/roles. Added tests that fail on changed surrounding traffic or light states. Hidden simulator state and alternative-action branch determinism remain unproven.
**Verified by:** outputs/navigation_m0_repeat_b_20261008/audit.json comparing repeat_a; both attempts retained, full video decoding/source hashes checked. All owned processes exited; no active training or simulation found by anchored process check.
**Lesson:** Identical ego behavior alone is insufficient evidence for counterfactual replay; inspect surrounding actors and signal timing too.
**Interview angle:** How did you validate repeatability before proposing rejected-trajectory replay?

## 2026-10-08 23:58 - Plan recorded in full; navigation docs spacing and a stop-sign claim corrected
**Type:** mistake
**Stage:** Navigation M0 documentation

**What happened:** The user asked that the reviewed navigation plan always be kept in mind. The navigation reports, journal entries, routes README and handoff had lost the space before or after most numbers ("Driving Score100", "from389m to161m", "port2100"). One interview answer still said the Town03 profile had no stop-sign opportunities, which the runtime inventory (20 stop actors) had already disproved.
**Cause:** The text was mangled when it was written in the previous session. The question-bank answer predated the runtime sign inventory and was never revisited.
**How we handled it:** Stored the user's full plan in reports/navigation_review_plan.md, and pointed CLAUDE.md and auto-memory at it. Re-inserted the spaces with a rule-based fixer limited to navigation-era text, leaving identifiers alone (best90k, Town03, seed42, hashes, paths, code spans). Checked the handoff with a word diff to confirm the changes were whitespace-only. Corrected the question-bank answer in place and marked the correction. No numbers, results or raw outputs changed.
**Verified by:** Word diff of the handoff; grep for remaining word–digit joins; full CPU+GPU suite 190 passed; ruff check/format clean on navigation code. Re-ran assess_navigation.py on all five training-town probes, and each audit passed with a source snapshot and route hash.
**Lesson:** Re-read generated reports as a reader would, not only for numerical accuracy; a later finding should be propagated to every earlier answer that relied on the old assumption.
**Interview angle:** How did you keep documentation consistent when a later simulator finding overturned an earlier assumption?

## 2026-10-09 00:20 - Encounter inventory: Town10HD load timeout aborted the client and orphaned the server
**Type:** bug
**Stage:** Navigation M0 encounter catalog

**What happened:** The first five-town rule-applicability inventory finished Town01–04. On Town10HD_Opt, `load_world` exceeded the 20s client timeout and CARLA's C++ client aborted the Python process (SIGABRT), so the owned server (PID 45942) kept running with no owner.
**Cause:** I ran the inventory in the same process that owned the server and reused the 20s startup timeout for a map load. The native abort skipped Python cleanup. This is the same failure class as the M0 map-reload crash, and I had not reused that fix.
**How we handled it:** Confirmed PID 45942 matched the run's server.json, then stopped it with SIGTERM to its own process group (no SIGKILL, no broad matching). Moved each town's inventory into a child process while the parent owns the server lifecycle (same pattern as run_navigation.py), with STOP handling and a 120s load timeout. Preserved the failed attempt at outputs/navigation_encounter_inventory_20261009.
**Verified by:** The rerun (outputs/navigation_encounter_inventory_v2_20261009) completed all five towns. Town01–04 summaries are identical to the failed attempt, and an anchored process check found no CARLA server afterwards.
**Lesson:** Every process that talks to CARLA's native client must sit beneath an owner that survives its abort; apply an established isolation pattern to every new simulator tool, not just the main harness.
**Interview angle:** Why does each simulator client run in a child process under a separate owner?

## 2026-10-09 00:25 - Runtime rule-applicability inventory for all five training towns
**Type:** milestone
**Stage:** Navigation M0 encounter catalog

**What happened:** Inventoried every junction movement in Town01/02/03/04/10HD_Opt using the pinned evaluator's own red-light (`get_traffic_light_waypoints`) and stop-sign (`is_actor_affected_by_stop`) geometry. 678 movements, 541 valid: 326 signal-controlled, 49 stop-controlled, 166 uncontrolled. A valid movement has an approach of at least 30m, and the harness's 1m planner must take that junction movement within 10% of the geometric length. Valid stop movements cover 20 of 31 runtime stop actors: Town03 11/20, Town04 8/8, Town10HD 1/3. Town01/02 have only signals.
**Cause:** Rule coverage needs lane-level opportunity denominators that use the same applicability logic as scoring; actor counts are not enough.
**How we handled it:** Added src/navigation/encounters.py and scripts/build_encounter_inventory.py. They load maps without a Traffic Manager, never drive, refuse Town05 and freeze nothing. Of the 11 excluded stop actors, 5 have approaches under 30m and 6 (all Town03) trigger only on junction-internal connector lanes. The evaluator would expect a full stop inside the junction on those connectors, so routes through them would record artifact stop infractions. Recorded this as a design constraint for the catalog, not as an outcome filter.
**Verified by:** Per-town JSON, with source copies, in outputs/navigation_encounter_inventory_v2_20261009. The Town03 probe (outputs/navigation_encounter_probe_town03_20261009) and the rerun are identical. Planner turn labels agree with signed heading change. Added a helper test; 24 navigation tests pass and ruff is clean.
**Lesson:** The stop-sign family is location-limited: 49 valid movements on 20 actors cannot give 50 independent encounters. Confidence intervals must be clustered by location, and the catalog needs a predeclared rule for reusing locations.
**Interview angle:** How did you count traffic-rule opportunities so they match how the evaluator scores violations?

## 2026-10-09 00:05 - Encounter-catalog design decisions approved; M0 work committed
**Type:** decision
**Stage:** Navigation M0 encounter catalog

**What happened:** The user replied "start" to three proposals, taken as approval of the recommended options: (1) the stop-sign family reuses its 20 locations across traffic seeds, with intervals clustered by location; (2) red-light encounters use one predeclared light-hold rule for every driver; (3) commit the M0 work locally. Committed dfc532a on experiment/camera-world-model-navigation (not pushed; submodule changes left out).
**Cause:** Only 49 valid stop movements exist at 20 actors, and a natural light cycle cannot guarantee a red on arrival.
**How we handled it:** Reused the pinned harness wherever possible. HardBreakRoute covers following a stopped vehicle; SignalizedJunctionLeftTurn and NonSignalizedJunctionLeft/RightTurn cover crossing/yield. The only new scenario is NavigationSignalHold, built on the pinned TrafficLightFreezer and PriorityAtJunction pattern and registered without editing the vendor checkout. The harness now records every scenario that starts or is skipped (upstream silently skips failed setups). Seeds and route-length checks are keyed on the XML route id.
**Verified by:** Commit dfc532a; unit tests for selection determinism, stratification, exclusion and outcome-blindness.
**Lesson:** Reuse the evaluator's scenarios and criteria so that expert and student encounters have identical semantics, and add only the setup the evaluator lacks.
**Interview angle:** How did you force a red-light opportunity without changing how violations are scored?

## 2026-10-09 00:15 - Town03 pilot: signal hold works, release zone too small for queues
**Type:** bug
**Stage:** Navigation M0 encounter catalog

**What happened:** A six-encounter Town03 pilot (one per family; outputs/navigation_encounter_pilot_20261009) started every declared scenario. Hold lights matched the inventory (59, 72), route lengths matched within 0.15m, and seeds came from the route id. The release/memory hold behaved as designed (stop at the line, 10s, clearance, green, go). In the signal_stop encounter, a background queue held the ego 25.6m from the line, outside the 20m release zone, so the light stayed red for the full 45s maximum. Four encounters recorded the evaluator's min-speed infraction (speed relative to traffic) during their mandated waits.
**Cause:** The hold rule assumed an empty approach. MinSpeedTest compares ego speed to other traffic, so imposed waits trip it.
**How we handled it:** Raised the signal_stop release zone to 40m, to cover a short queue (release/memory keeps 20m). Added rule_safe_completed: completed, with no infraction other than min-speed. Kept the strict any-infraction count and a per-episode min-speed count, so nothing is hidden. Both changes respond to setup behavior, not to how well the expert drove. Froze configs/encounters/dev_v1 (catalog sha256 716c3600…), whose encounters are byte-identical to the piloted candidate apart from that parameter.
**Verified by:** Per-frame light state from world_trace versus ego speed and distance to the stop line; actor roles show the crossing scenario's oncoming flow and HardBreakRoute's lead vehicle; catalog diff against the candidate.
**Lesson:** Pilot every scenario's mechanics against its intent before freezing; a rule that works on an empty road can silently become a fixed-duration wait in traffic.
**Interview angle:** Why report min-speed infractions separately instead of counting them as unsafe?

## 2026-10-09 00:30 - Encounter runner smoke test: simulator segfault, retry and stop-scoping bugs
**Type:** bug
**Stage:** Navigation M0 encounter catalog

**What happened:** A Town10HD runner smoke test was meant to stop after one chunk, but my wait loop watched for a nonexistent chunk name (the frozen ids start at 248, not 246), so it kept running until I sent STOP after 22 of 52 encounters. STOP worked: the runner, evaluator and server exited within 6s. During the run, the CARLA server segfaulted (signal 11) at the start of route 263, before any frame. The chunk's retry ran 263–265 cleanly. Review found two of my bugs. The runner would have rerun routes that the evaluator had recorded in raw leaderboard.json before an abort. And the worker marked every route in a stopped multi-route attempt as an execution failure, so route 269, which finished at 100%, was not counted as completed.
**Cause:** I hard-coded chunk ids from an earlier candidate. Both bugs carried single-route M0 assumptions (summary.json always written; one route per attempt) into multi-route runs.
**How we handled it:** Raw records now settle routes when summary.json is missing; only the route running at a stop or crash is marked failed. Retry rule: only a missing record or a "Simulation crashed" record is retried, at most once, with every attempt kept. Agent failures, collisions and timeouts are never retried. Added tests for both bugs. The smoke run is preserved as outputs/navigation_encounter_runner_smoke_20261009 and is not part of the expert ceiling, which will be a fresh full run. Its 21 settled encounters: all rule-safe, zero collisions, zero red or stop violations.
**Verified by:** 196 tests pass (CPU+GPU); carla.log shows the native segfault; the a1 attempt's records; an anchored process check shows nothing left running.
**Lesson:** Every single-run assumption must be re-audited when moving to multi-route attempts, and a smoke test needs a stop condition computed from the actual frozen catalog.
**Interview angle:** How did you make long simulator evaluations resumable without letting retries become outcome selection?
