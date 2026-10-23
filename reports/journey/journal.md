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

**What happened:** A Town10HD runner smoke test was meant to stop after one chunk, but my wait loop watched for a nonexistent chunk name (the frozen ids start at 248, not 246), so it kept running until I sent STOP after 22 of 52 encounters. STOP worked: the runner, evaluator and server exited within 6s. During the run, the evaluator lost the simulator (30s timeout) at the start of route 263, before any frame (corrected 2026-10-09 09:10: originally misreported as a segfault). The chunk's retry ran 263–265 cleanly. Review found two of my bugs. The runner would have rerun routes that the evaluator had recorded in raw leaderboard.json before an abort. And the worker marked every route in a stopped multi-route attempt as an execution failure, so route 269, which finished at 100%, was not counted as completed.
**Cause:** I hard-coded chunk ids from an earlier candidate. Both bugs carried single-route M0 assumptions (summary.json always written; one route per attempt) into multi-route runs.
**How we handled it:** Raw records now settle routes when summary.json is missing; only the route running at a stop or crash is marked failed. Retry rule: only a missing record or a "Simulation crashed" record is retried, at most once, with every attempt kept. Agent failures, collisions and timeouts are never retried. Added tests for both bugs. The smoke run is preserved as outputs/navigation_encounter_runner_smoke_20261009 and is not part of the expert ceiling, which will be a fresh full run. Its 21 settled encounters: all rule-safe, zero collisions, zero red or stop violations.
**Verified by:** 196 tests pass (CPU+GPU); carla.log shows the native segfault; the a1 attempt's records; an anchored process check shows nothing left running.
**Lesson:** Every single-run assumption must be re-audited when moving to multi-route attempts, and a smoke test needs a stop condition computed from the actual frozen catalog.
**Interview angle:** How did you make long simulator evaluations resumable without letting retries become outcome selection?

## 2026-10-09 01:00 - Parallel CARLA slots after measuring a latency-bound single server
**Type:** decision
**Stage:** Navigation M0 expert ceiling

**What happened:** The user noticed the GPU running at about 20% during the approved single-server expert run and asked for better use. Measured: CARLA about 1.9 cores and the evaluator/expert about 0.7, out of 24 (88% idle); GPU 11–17% at 40W; about 2GB VRAM per server; 19.7GB RAM available. The synchronous step–render–plan loop was latency-bound, not resource-bound.
**Cause:** The plan deferred parallel servers until measurements justified them; these measurements now do.
**How we handled it:** I implemented slots in a separate git worktree, so the live run never read half-edited code. Ports are 2100/8100 + 100×slot; slot 0 keeps the original lock; only sibling navigation servers are accepted; server starts are serialized and gated on at least 3GB free VRAM and 6GB available RAM; attempts refused before starting are not driving failures. Chunk composition, seeds and retry rules do not depend on the worker count. I then stopped the single-server run with STOP at 35/300 (preserved at outputs/navigation_expert_ceiling_20261009, results not inspected before stopping) and restarted fresh at outputs/navigation_expert_ceiling_parallel_20261009 with 3 workers. The 35 overlapping encounters will test whether parallel slots change outcomes.
**Verified by:** 31 navigation tests, including slot ports, sibling-versus-foreign servers and refused attempts. With three servers: GPU 33–37% at 7.5/12.2GB, CPU 67% idle, no resource waits.
**Lesson:** Low utilization in synchronous simulation usually means latency-bound, not overloaded; measure, then parallelize at the episode level, without editing code a live run is still reading.
**Interview angle:** How did you speed up a synchronous simulator evaluation without changing its results?

## 2026-10-09 09:05 - Overnight expert run killed by host memory pressure; gate measured too early
**Type:** bug
**Stage:** Navigation M0 expert ceiling

**What happened:** The 3-slot expert run reached 258/300 at 01:46. Claude Code then killed the background runner tree because the system was critically low on memory. Three Town10HD_Opt servers had started within 9 s of each other (01:40:48, 01:46:01, 01:46:10). Nothing was left running and raw outputs were intact. The user asked to rerun without memory getting full.
**Cause:** My launch gate measured RAM/VRAM as soon as a server answered RPC. CARLA boots on its default map, and the evaluator loads the requested map afterwards, so a Town10HD server's real footprint (about 4–5GB RSS and about 5GB VRAM) wasn't counted yet. The 6GB RAM margin was too small, and nothing reacted once chunks were running.
**How we handled it:** A start now completes only once its chunk is driving (a non-empty trace), and new servers need 12GB available RAM. A watchdog STOPs the newest chunk below 6GB, and its route is reattempted. I annotated the three killed attempts with interrupted.json, listing the evidence (reaper notice, RequestExit in carla.log, no error.json). Their recorded routes are kept, and the kill spends no retry. The resumed run held 13–21GB available with zero watchdog pauses.
**Verified by:** Memory sampling every 15–20s during the resume; unit tests for interrupted attempts and the watchdog.
**Lesson:** Gate parallel simulator starts on the footprint after the map loads, not on the boot state, and keep a live watchdog: a host-level kill is far more expensive than a paused chunk.
**Interview angle:** How did you make parallel simulation safe on a 32GB laptop?

## 2026-10-09 09:10 - Crash accounting charged never-started routes; VRAM gate raised; "segfault" misread corrected
**Type:** mistake
**Stage:** Navigation M0 expert ceiling

**What happened:** On resume, route 262 failed to start again: the evaluator died on a 30s simulator timeout while a second Town10HD map loaded next to a running one (VRAM peaked at 11.7/12.2GB). Two of my errors surfaced. (1) When a chunk died on its first route, the runner charged a crash to every route in the chunk, including nine that never started. (2) I had described these failures (and the smoke-test one on route 263) as CARLA segfaults because carla.log contains "signal 11". The same lines end every normal SIGTERM shutdown, including the clean M0 long drive, so they are not crash evidence.
**Cause:** Single-route assumptions in the retry bookkeeping, and reading the simulator log without a clean baseline.
**How we handled it:** A crash now charges only the route in the evaluator's current_route.json, or the first unrecorded route if none started; the others are "not reached". New servers need 6GB free VRAM, so Town10HD runs one server at a time. The assessment detects evaluator failures from error.json. Correction to the 2026-10-09 00:30 entry: route 263 was a 30s simulator timeout at route start, not a demonstrated segfault. The VRAM link is a likely cause, not a proven one.
**Verified by:** current_route.json and empty traces of the failed attempts; the M0 long drive's carla.log shows the same shutdown lines; a test checks that only the running route is charged; final accounting charges routes 258 and 262 only.
**Lesson:** Compare failure logs against a clean run before naming a cause, and charge a failure to the trial that was actually running.
**Interview angle:** How did you tell a simulator crash from the simulator's normal shutdown noise?

## 2026-10-09 09:25 - v1 expert ceiling complete but 35 signal holds never started; v2 rerun launched
**Type:** bug
**Stage:** Navigation M0 expert ceiling

**What happened:** The first full catalog run (outputs/navigation_expert_ceiling_parallel_20261009) settled all 300 encounters. The 35 shared encounters match the stopped single-server run exactly (status, completion, every infraction count), so parallel slots do not change results. The setup audit then showed that NavigationSignalHold never started on 35 of 100 signal encounters (17 signal_stop, 18 release/memory, mostly Town03/04): "No traffic-light landmark on the ego approach". Upstream skipped them with a printed warning, so they ran on natural light cycles and were not the declared test. I had copied the pinned PriorityAtJunction landmark lookup without checking it on these maps, and the six-encounter pilot happened to use approaches that have landmarks.
**Cause:** OpenDRIVE light landmarks are missing on many Town03/04 approaches.
**How we handled it:** The hold now resolves the ego light with the pinned red-light criterion's own lane geometry (the same as the inventory), and refuses on zero or several matches. A check of four previously skipped encounters bound lights 57, 68, 198 and 168, matching the inventory. The assessment now audits every declared setup and reports invalid ones separately, never as passes. v1 is preserved with its report (reports/navigation_expert_ceiling_v1_20261009.json). Over valid encounters only, v1 had: signal_stop 33/33, release/memory 30/32, crossing/yield 45/50, stopped lead vehicle 50/50, route/junction 50/50, stop sign 50/50; collisions all count. A fresh v2 run of all 300 with one code version started at 09:26.
**Verified by:** Per-encounter scenario_setup logs versus the catalog; the four-encounter check; 201 tests pass.
**Lesson:** A setup that can fail silently must be audited on every trial, not only in a pilot; a pilot sample can miss a map-specific failure.
**Interview angle:** How did you discover that a third of your red-light tests never actually forced a red light?

## 2026-10-09 23:50 - M0 v2 expert calibration resumed from preserved records
**Type:** decision
**Stage:** Navigation M0 expert ceiling

**What happened:** User requested continuing M0. The v2 progress file still said running (43/300), but no process remained. Raw evaluator records contained 52 completed encounters; three active traces ended together around 09:35:33 with no error.json or STOP. Exact termination cause is unknown.
**Cause:** The queue progress updates at chunk boundaries and was stale after an abrupt termination. This is not evidence of a native simulator crash or a particular external cause.
**How we handled it:** Resume the already-authorized v2 evaluation with the unchanged frozen catalog/seed42 and two workers,12GB RAM/6GB VRAM launch margins and 6GB critical-RAM STOP threshold. Preserve completed records and use the existing one-retry rule for missing records on routes37/54/64; do not invent an interruption exemption. Run under persistent user service navigation-m0-v2-resume-20261009. No training, reward change or benchmark replacement.
**Verified by:** Both resumed workers produced fresh driving traces; service active,5.1GB GPU memory used and 18GiB RAM available. Catalog hash 716c 36005d 6c 00af 8d 5905491a 9acc 8c 4363e 541a 5fd 4acf 309a 5463e 2aa 535c unchanged. A separate read-only assessment service waits on the runner lock to generate the final v2 JSON.
**Lesson:** Recover progress from durable evaluator records rather than a stale queue display, and preserve uncertainty about abrupt termination causes.
**Interview angle:** How did you safely resume an interrupted evaluation without rerunning completed outcomes?

## 2026-10-09 23:50 - Partial-run and all-attempt collision reporting corrected
**Type:** bug
**Stage:** Navigation M0 assessment integrity

**What happened:** The assessor called unrun v2 scenarios setup failures and mixed pending no-scenario routes into failure rates. It summed collisions only over valid final encounters: v1 showed 5 despite 7 recorded events across all attempts. Glob order could also bind a final result to the wrong attempt's setup evidence. Unanimous bootstrap samples returned[1,1], suggesting unjustified certainty.
**Cause:** The assessor assumed a completed run and lost provenance when filtering invalid setups and retries.
**How we handled it:** Bind setup evidence to the final record's attempt; distinguish unsettled, failed, missing and wrong-light setups. Keep planned counts and pending/invalid IDs explicit, with rates over valid settled trials. Count every collision in every raw attempt, with a separate valid-final count. Return null for empty/degenerate cluster-bootstrap bounds; retain explicitly unclustered Wilson intervals. Preserve the original v1 report and add reports/navigation_expert_ceiling_v1_audited_20261009.json.
**Verified by:**37 targeted tests passed/one optional-import test skipped; tests cover retries, missing setups, pending trials, all-attempt collisions and degenerate intervals. Changed-file lint/format and git diff whitespace checks pass. No evaluator, expert, frozen catalog or live runner source changed.
**Lesson:** Failed setup and missing execution are different conditions; neither should erase observed collisions or create artificial statistical confidence.
**Interview angle:** How did an apparently conservative setup filter accidentally undercount collisions?

## 2026-10-11 10:15 - Host shutdowns confirmed; M0 v2 resumed with completed records retained
**Type:** decision
**Stage:** Navigation M0 expert calibration

**What happened:** User requested continuation. No runner remained; 64 completed raw records survived, including one blocked-agent failure and zero recorded collisions. System journals show shutdowns on Oct9 at09:35 and23:51, stopping the evaluator trees and the waiting assessment service.
**Cause:** Host shutdown interrupted the work. This resolves the previously unknown termination cause; no memory or simulator-crash cause is inferred for these interruptions.
**How we handled it:** Saved system-journal evidence and added interrupted.json to the five unfinished attempts spanning both shutdowns. All completed records, including the blocked result, remain; the existing interruption rule spends no simulator retry. Resumed the same frozen catalog/seed42 with two workers,12GB RAM/6GB VRAM launch headroom and6GB RAM STOP threshold under navigation-m0-v2-resume-20261011. Restored the final audit as navigation-m0-v2-assessment-20261011.
**Verified by:** Host services active, fresh camera traces, roughly4.7GB VRAM used and20GiB RAM available. Evidence and before/after audits are under outputs/navigation_expert_ceiling_v2_20261009. Catalog hash remains `716c36005d6c00af8d5905491a9acc8c4363e541a5fd4acf309a5463e2aa535c`; the spaced hash in the Oct9 23:50 journal entry was my formatting error, not a changed catalog.
**Lesson:** A persistent service survives the chat session, not host power-off; preserve interruption evidence and recover from durable records.
**Interview angle:** How did you distinguish host shutdown from a simulator failure when resuming evaluation?

## 2026-10-11 10:15 - Live assessment no longer calls an active trial a crash
**Type:** bug
**Stage:** Navigation M0 reporting

**What happened:** The read-only live assessor reused settlement logic intended for an exited evaluator. A currently running trial with no final record was therefore reported as a charged crash.
**Cause:** Missing record means failure only after the worker exits. The tool sandbox also cannot see host service PIDs, so process-liveness checks must run in the host namespace.
**How we handled it:** Verify the evaluator PID and exact output argument before labeling missing records; active trials become in progress in live reports. Run live process-aware audits on the host. The final audit already waits for the runner lock to release. Live evaluator, runner, catalog and scoring code remain unchanged.
**Verified by:** Five assessment regression tests passed; lint/format checks passed; host PID checks recognized both active workers. Added tests for active-versus-exited trials and unrelated/reused PIDs.
**Lesson:** Do not apply end-of-attempt failure accounting to an in-progress snapshot, and do not infer host liveness from a restricted process view.
**Interview angle:** Why did a live progress report temporarily claim an encounter had crashed while it was still driving?

## 2026-10-11 14:11 - Physical traffic-light identity corrects ten false setup rejections
**Type:** bug
**Stage:** Navigation M0 closeout audit

**What happened:** The automatic final report marked ten Town10HD signal-stop encounters wrong_light. Their chosen runtime actors had different numerical IDs from the inventory, but occupied exactly the intended physical light locations.
**Cause:** My cross-session setup audit treated CARLA actor IDs as persistent identities. One Town10HD launch permuted the light IDs. Direct ID comparison can both reject a correct lamp and accept the wrong lamp when IDs are reused.
**How we handled it:** Resolve expected lamps through the frozen catalog's hash-verified inventory, validate the declared entry lane, and compare against uniquely matching runtime physical coordinates within 1 cm. Preserve the original report; write reports/navigation_expert_ceiling_v2_audited_20261011.json. No evaluator, route, seed or outcome changed.
**Verified by:** All 100 holds match physical identity with maximum error 0.624 mm; full world traces confirm selected lamp positions and red/green states. Regression tests accept permuted IDs and reject matching IDs at wrong positions or on wrong lanes. Targeted suite: 40 passed, one optional-import test skipped; lint/format clean.
**Lesson:** Simulator actor IDs identify a session object, not a persistent map entity; validate cross-session identity with frozen map geometry.
**Interview angle:** Why did ten apparently wrong traffic-light setups turn out to be valid without rerunning them?

## 2026-10-11 14:11 - M0 complete: expert ceiling calibrated on all 300 development encounters
**Type:** milestone
**Stage:** Navigation M0

**What happened:** The v2 queue completed at 2026-10-11 11:20:16. Corrected results: 291/300 rule-safe completions, 298/300 arrivals, 6 collision events, zero recorded red/stop violations; strict no-infraction completion 63/300. Per family: held-red 50/50, release/memory 47/50, crossing/yield 45/50, stopped-lead 49/50, route/junction 50/50, stop-sign 50/50. All 200 declared scenario setups validated.
**Cause:** The remaining nine rule-safe failures comprise six collision episodes, two blocked episodes and one lane departure. Four crossing collisions share Town10HD junction 895. Root causes and fault are not inferred from aggregate scores.
**How we handled it:** Preserve failures and source provenance; close M0 with reports/navigation_m0_closeout_20261011.md. Next is M1's atomic data engine with explicit failure/intervention labels and separation from development locations, followed by a small audited collection. No camera training or new long run launched.
**Verified by:** All 300 final episodes audited: 201,215 continuous aligned timestamps, route/seed checks, video headers; 38 source archives verify with one version per driving file. All 100 signal episodes show red and green states. Fully decoded all six collision videos and saved approximate contact-position bookmarks. One native evaluator abort on route 298 was retried; five host-shutdown interruptions retained; no memory-watchdog pauses in the final continuation.
**Lesson:** A 97% encounter reference is useful for comparison, but not proof of long-route reliability or a reason to imitate every teacher action indiscriminately.
**Interview angle:** What did M0 establish, and which failures should guide the camera student's data collection?

## 2026-10-15 14:24 - Approved spatial split resolves training/development location exhaustion
**Type:** decision
**Stage:** Navigation M1 data design

**What happened:** Inspection found dev_v1 covers all usable stop-sign junctions and nearly every junction overall; only four unused junctions remained, with no usable stop encounter. This made the intended rule-complete, location-separated training impossible under that catalog.
**Cause:** M0's coverage-first calibration suite was frozen before reserving training locations.
**How we handled it:** Asked under the user's benchmark-change rule; the user explicitly approved a new location split and separate held-out suite. Preserved dev_v1 byte-for-byte as legacy expert calibration. Froze location_split_v1 using geometry-only seed 20261015, first partition meeting coverage constraints (attempt 3): 62 train / 17 held-out junctions across five towns; separate 300-encounter catalogs, 50 per family. Planned routes and actual collection trajectories are checked against their partition.
**Verified by:** Hashed catalogs/locations in configs/encounters/location_split_v1; seeded/outcome-blind/disjointness regression. Held-out stop trials cover only two junctions, so 50 trials are not 50 independent locations. New held-out suite has not been driven; legacy 291/300 is not its expert ceiling.
**Lesson:** Reserve training geography before maximizing development coverage; disclose reused locations and keep a changed evaluation plan separate from earlier results.
**Interview angle:** Why did you need a new spatial split after finishing the first expert benchmark?

## 2026-10-15 14:24 - M1 writer and six-family pilot pass with exact RGB and action feedback
**Type:** milestone
**Stage:** Navigation M1 implementation/pilot

**What happened:** Implemented atomic 32-sample shards, lossless RGB, typed policy-only reading, separate privileged labels and conservative episode eligibility. The six-encounter pilot produced 1,460 samples across all five towns/six families; all arrived with no safety/rule infraction and zero actuator mismatches. An earlier 468-sample smoke also passed. A bounded-noise retry produced exactly 96 samples, 18 noise-control ticks, zero feedback mismatches and no recorded collision before intentional truncation.
**Cause:** M0's annotated demo videos lack the raw input/label schema and crash-recovery guarantees needed for learning. PNG measured about 77 GB / 100k timestamps on the first scene.
**How we handled it:** Use lossless libx264rgb video inside ZIP shards: write/fsync/rename, hash archive members and decoded pixels, ignore incomplete .part files. Kept PNG originals; repack of the same 468 samples fell from 770 KB to 331 KB/timestamp with exact pixel hashes. The broader pilot projects 8.9 GB for 20k and 44.6 GB for 100k, before caches. A sample commits only after next-tick get_control confirms execution. Unsafe/unknown/safety-stopped episodes remain prediction data with explicit censoring, not clean imitation targets. Added seeded approach-only steering pulses with a privileged hazard brake and an exact sample cap.
**Verified by:** Full CPU suite 219 passed/two skipped before the final queue-resume regression; 14 collection tests pass afterward, including crash-prefix recovery, frame order/corruption, no privileged label reads, feedback timing, outcome eligibility and resume without duplicates. All pilot RGB frames/labels decoded, hashed and audited. Reports: reports/navigation_m1_pilot_20261015.md/json.
**Lesson:** Measure storage on actual multi-view frames before scaling, and distinguish submitted controls from observed actuator execution.
**Interview angle:** How does your dataset survive simulator crashes while preserving observation/action alignment?

## 2026-10-15 14:24 - NVIDIA update mismatch blocks scaling; launch checks now fail closed
**Type:** blocker
**Stage:** Navigation M1 initial collection

**What happened:** The noise probe's evaluator aborted on a 30 s simulator timeout before any camera frame. The next host health check reported NVML Driver/library version mismatch. Kernel module 615.71.09 remained loaded while nvidia-utils was upgraded to 615.78.08-2 at 14:05 during this session; pacman.log also shows the NVIDIA kernel package upgrade. This is a confirmed driver-state fault; its causal role in the earlier timeout is not proven.
**Cause:** Userspace drivers were updated without reloading the kernel module. I also let a short retry launch after its GPU query failed; that should have been gated. It completed the 96-sample cap test, but a successful short render does not establish driver readiness.
**How we handled it:** Preserved the failed attempt and valid capped data; used STOP to end/confirm the owned retry. Changed resource_shortfall and the bulk collector to refuse simulation when GPU memory cannot be queried. Prepared the resumable 20k collector with a 90-minute invocation budget and automatic full data audit, but did not launch it. Reboot and nvidia-smi verification are required before further GPU work; no driver packages, reboot or system configuration changed by this agent.
**Verified by:** /proc/driver/nvidia/version, host nvidia-smi failure and /var/log/pacman.log timestamps; regression for failed GPU readiness. Anchored host check confirms no simulator remains.
**Lesson:** An unknown GPU-memory result is a failed readiness check, not permission to continue after a driver update.
**Interview angle:** Why did you stop scaling a successful data pilot, and how did you prevent another wasted run?

## 2026-10-15 15:00 - Reboot clears GPU blocker; spatial guard catches a boundary spawn
**Type:** bug
**Stage:** Navigation M1 initial collection

**What happened:** After reboot, NVIDIA kernel/userspace both report 615.78.08 and readiness passes. Started the bounded 20k collector at14:52. Town03 route94 then crashed at0.25 simulated seconds on the held-out-junction guard, before any noise pulse. Stopped via root STOP; all owned processes exited. Retained297 timestamps/13 shards: three rule-safe completions and one aborted encounter.
**Cause:** The planner's first nonjunction point is0.0000153m from held-out junction103. CARLA's nearest-road lookup switches between road24/lane1 and road123/lane1 for almost stationary ego coordinates. Junction-ID checks alone miss coincident road endpoints. The harness also exits0 on an agent crash, so exit codes alone did not count it as a collector error.
**How we handled it:** Preserved the entire run and marked it superseded_do_not_train in collection_review.json. Kept the strict guard. Added explicit violation evidence/fatal queue stop and record-based agent-crash detection; ordinary collision outcomes remain retained. Never changed a frozen catalog or used outcome-based route replacement.
**Verified by:** Offline CARLA map reconstruction of all five recorded ego frames; original console traceback; full297-sample pixel/action audit passes, which does not establish spatial eligibility. Anchored host check confirmed clean shutdown.
**Lesson:** Distinct road IDs do not guarantee spatial separation at a shared endpoint, and a zero harness exit code does not guarantee the agent ran successfully.
**Interview angle:** How did you catch training-location leakage before scaling collection?

## 2026-10-15 15:00 - Geometry clearance audited; fresh bounded 20k collection running
**Type:** decision
**Stage:** Navigation M1 initial collection

**What happened:** Applied one geometry-only training eligibility rule to all300 routes: compare1m-spaced planned points with0.5m-spaced held-out junction points in3D, requiring at least1m clearance. Excluded13 and retained287, preserving every family/town stratum (crossing45, following50, route45, release50, signal-stop47, stop50). The62/17 location assignments, frozen train/dev catalogs and all evaluation metrics are unchanged.
**Cause:** Need conservative numerical clearance while retaining the user-approved split and strict runtime rejection. The clearance is not a claim of complete visual/geographical independence.
**How we handled it:** Froze the eligibility manifest/map hashes in request.json; changing it or driving source refuses resume. Started fresh output outputs/navigation_m1_20k_v2_20261015 at14:57:40 under navigation-m1-20k-v2-20261015.service. Single simulator;20k target;90-minute budget plus cleanup; automatic full audit on exit. Historical pilot data need the clearance audit before training use.
**Verified by:**16 collection tests pass, including endpoint coincidence, overpass separation, zero-exit agent crash and immutable resume; lint/format/whitespace clean. First four encounters committed571 samples, all rule-safe, zero collisions/control mismatches; the first noisy-mode episode made0 actual perturbation ticks, so perturbation coverage remains to audit. Host47C,21GiB RAM available. No camera-student training or Town05 use.
**Lesson:** Freeze a geometry eligibility manifest before collection and preserve excluded or aborted attempts rather than weakening the split guard.
**Interview angle:** How do you fix a collection boundary error without tuning the held-out benchmark?

## 2026-10-15 16:55 - M1 initial 20k gate passes; failures and spatial audit retained
**Type:** milestone
**Stage:** Navigation M1 assessment

**What happened:** Corrected collection completed at15:39:02 in2,480.25s:20,000 timestamps/665 shards/8.888GB across73 episodes.63 episodes/17,567 timestamps are eligible for imitation.71 arrivals,68 rule-safe completions,three recorded vehicle collisions (all crossing/yield),zero recorded red/stop violations.252 actual perturbation ticks and seven privileged safety-stop ticks were observed.
**Cause:** Ten episodes are excluded:three collision outcomes,five safety-stopped episodes,one30s native CARLA timeout with missing final outcome/profile, and the intentional35-sample final cap. Do not count collection arrival rates as benchmark reliability or assume zero unobserved incidents in the failed evaluator.
**How we handled it:** Preserved every raw attempt/outcome; assessed only the corrected run. Kept unsafe/unknown/censored data out of imitation. Added scripts/assess_navigation_collection.py and reports/navigation_m1_assessment_20261015.md/json. No100k collection launched.
**Verified by:** Every committed RGB frame passed decoded-pixel hashes and action/frame audits;zero feedback mismatches. Independent offline check of all20k stored ego positions foundzero partition violations over43 observed train junctions. Driving source hashes matched before M2 edits. All five towns and six families represented;onlyfour clean stop-sign episodes and187 yellow timestamps. Heldout_dev/Town05 remain unused.
**Lesson:** Passing a data-integrity gate does not make unsafe expert behavior a clean target or establish sufficient rule coverage for a learned driver.
**Interview angle:** What did the20k collection actually establish, and which data were excluded?

## 2026-10-15 16:55 - First camera CNN/GRU trained; internal diagnostics expose stopping and perception gaps
**Type:** result
**Stage:** Navigation M2 first baseline

**What happened:** Implemented raw-image caching, connected-junction train/validation splitting, CNN spatial features with full-resolution front branch, causal GRU and separate path/speed/brake/perception heads. Trained1,000 steps in441.06s including cache preparation. Internal split:50/13 episodes,14,745/2,822 timestamps. Best selected step300; speed MAE2.596m/s versus matched train-mean3.736, but stopped-target MAE3.300m/s. Light accuracy49.78% is below majority56.80%; zero path residual0.00365m beats learned0.03794m.
**Cause:** Open; this is a weak first imitation baseline. Exact sampler replay gave40.29% stopped targets, so absence of stop examples alone is not the explanation. Short1.6s training windows, limited rule diversity and independent speed/brake decisions require targeted diagnosis. Later training did not improve selection score.
**How we handled it:** Retained best/latest and full optimizer/RNG/sampler state, used BF16 without a scaler, archived the three training sources after hash verification. Added matched-window/simple-baseline and image-blanking assessment; kept heldout_dev/Town05 untouched. No repeat long training launched.
**Verified by:** Fresh inference-process CPU test rejects extra privileged fields and simulator/training imports; causal/streaming equivalence and optimizer/RNG recovery tests pass. Blanked-image speed MAE3.714m/s is sensitivity evidence, not a trained navigation-only comparison. Reports/navigation_m2_student_smoke_20261015.md/json and source graphs contain all checkpoints/metrics.
**Lesson:** A lower aggregate speed error can coexist with poor stopping, below-majority perception and no benefit from learned route correction.
**Interview angle:** Why did you reject a model that improved offline speed error?

## 2026-10-15 16:55 - Camera route diagnostic fails blocked; setup bug preserved and fixed
**Type:** result
**Stage:** Navigation M2 closed-loop integration

**What happened:** First setup failed before driving: the pinned local agent and validator define different Track enums with identical printed names. Fixed the adapter to use the validator's canonical enum; preserved the failed attempt. The corrected200-frame profile moved15.96m with no recorded collision. Full1,233.361m Town03 diagnostic then failed blocked at1.91% completion after20.90m travel/241.4 simulated seconds;zero recorded collisions/red/stop violations,zero overrides/interventions.
**Cause:** At the blocked state, learned target speed remained about1m/s while brake probability exceeded0.5; conventional control held the brake. A specific signal/traffic cause is not established. This is not safe navigation despite zero observed collisions. The setup cause was an upstream enum mismatch, not invalid cameras or changed scoring.
**How we handled it:** Kept the exact policy/route/seed/evaluator; inference ran in a fresh process with array/JSON allowlists and import guards. Declared ego odometry/localization for navigation; teacher/other-actor/rule facts remain outside policy IPC. Retained full failed video/world traces and marked them evaluation-only. Next priorities: semantic stop/go, focused signal labels, longer streaming memory and DAgger on TRAIN routes.
**Verified by:**4,828 aligned frames at27.92fps;packet/inference/IPC p95 12.57ms. Fully decoded2,414 H.264/yuv420p MP4 frames. Actual pinned-validator regression passes;full CPU suite232 passed/two skipped. All services/processes exited cleanly. No M2 driving gate, learned-rule claim or JEPA contribution established.
**Lesson:** Validate both release and stopping; a stationary collision-free policy fails navigation, and setup failures must be distinguished from policy outcomes.
**Interview angle:** How did you prove the first camera policy ran without oracle inputs, and why did it still fail?

## 2026-10-15 16:55 - Final-suite registration timing deviation remains open
**Type:** mistake
**Stage:** Navigation experimental protocol

**What happened:** I proceeded to the first student before freezing the final Town05 routes, contrary to the plan's intended pre-student registration. Configuration still correctly says benchmark_frozen:false. No Town05 image, training data, model selection or driving result has been used, but an unseen town is not the same as a preregistered route suite.
**Cause:** I checked the approved training/location split and bounded M2 budget but missed the separate final-route-freeze prerequisite.
**How we handled it:** Recorded the deviation explicitly in the handoff, plan and student assessment; left Town05 unused. The correction remains open: freeze a geometry/coverage-only suite and its analysis before Town05 testing, independent of student outcomes. Do not claim it was frozen before this baseline. The new heldout_dev suite also still needs its own expert reference before student comparisons.
**Verified by:** No final Town05 route manifest exists; benchmark_frozen remains false and the cache contains only approved train-town episodes. No final-test number is reported.
**Lesson:** Reserve locations and preregister the final routes as distinct checks; neither should be inferred from the other.
**Interview angle:** What protocol deviation did you disclose before reporting any final test result?

## 2026-10-15 17:01 - Initial signal labels lack visibility validity; future profiles remain auxiliary
**Type:** decision
**Stage:** Navigation M2 scope and label audit

**What happened:** Inspection of the clean cache found523 known next-light states outside the80m distance validity range and2,738 beyond40m. Labels identify a route-applicable lamp but do not establish image visibility/occlusion. The v0 four-second future profile is supervised/logged; actuation currently follows corrected spatial route geometry and the immediate speed/brake heads.
**Cause:** Initial collection records privileged next-light state/distance without a camera-visibility target. The first controller is a conventional spatial/current-speed baseline, not completed time-profile tracking or predictive trajectory ranking.
**How we handled it:** Qualified the49.78% state-label accuracy as a diagnostic rather than an audited visible-lamp benchmark. Added visibility/localization/masking and complete profile tracking to next M2 work. No dataset, loss, controller or score was retroactively changed; no new training launched.
**Verified by:** Exact cache-array counts and policy_runtime/controller code; reports/navigation_m2_student_smoke_20261015.md/json. All current jobs remain exited.
**Lesson:** Privileged route applicability does not prove visual observability, and a forecast head is not a deployed planner merely because its output is logged.
**Interview angle:** Why qualify your first traffic-light accuracy and future-planning claims?

## 2026-10-15 17:04 - Speed-only stopping diagnostic misses actuator conflicts
**Type:** bug
**Stage:** Navigation M2 assessment

**What happened:** The original metric reported only0.13% of go targets with a predicted speed below0.5m/s, despite the real policy becoming blocked. It did not include the independent brake head used by the actual controller.
**Cause:** Regression output is not the full actuator decision: control holds when predicted speed<0.15m/s OR brake probability>=0.5. The first validation metric omitted the latter branch.
**How we handled it:** Added an action-level diagnostic on the exact same1,352 internal positions without changing the frozen candidate/selection/results. It holds242/556 teacher-stopped positions (43.53%) and70/645 teacher-go/nonbraking positions (10.85%);349 positions contain positive-speed/brake contradictions. All17 zero-ego-speed teacher-go positions release in this sample, so the observed route deadlock also requires student-state/generalization diagnosis.
**Verified by:** scripts/assess_navigation_student.py and regenerated report JSON; fixed training source hashes still match. These are offline teacher-label agreement counts, not driving success probabilities. Best step300 and the failed route remain unchanged.
**Lesson:** Evaluate the controller's full decision rule; a speed-only collapse check can miss braking-induced immobility and poor stop recall simultaneously.
**Interview angle:** Why did the speed metric suggest little over-stopping while the car got stuck?

## 2026-10-16 09:58 - Separate semantic stopping from cruising in a bounded M2 candidate
**Type:** decision
**Stage:** Navigation M2 countermeasure

**What happened:** Continue from the frozen v0 failed blocked route and349/1,352 positive-speed/brake conflicts. Added camera-stop-go-v2: semantic zero-speed intent gates current/future speed; cruising regression excludes stopped labels; original comfort-brake head is auxiliary. No oracle rules or forced crawl.
**Cause:** Independent target-speed and actuator-brake imitation permits contradictory intent; offline labels also contain known lamps beyond the80m distance range. Visibility remains unaudited.
**How we handled it:** Fresh2,000-step budget, maximum1,800s, LR1e-4, batch2,32-frame clips; same raw cache and location split. Predeclared selection=0.5*(missed-stop+false-hold)+0.02*go-speed MAE. Known lamp labels outside valid80m range are masked (not a visibility claim). Archive sources at launch. No reward/benchmark changes.
**Verified by:**38 targeted CPU tests pass; refactored legacy model produces bitwise-identical outputs/hidden states against its archived original. Freeze six first eligible Town03 TRAIN encounters, one/family, seed4242+route id, for expert/frozen-v0/new-policy sanity diagnostics; all outcomes retained. See reports/navigation_m2_stop_go_pilot_20261016.json.
**Lesson:** Separate stop intent from comfort braking, and select using the deployed controller decision.
**Interview angle:** How did you address speed/brake conflicts without forcing the car to move?

## 2026-10-16 10:04 - Stop-go candidate measured; partial offline gains do not pass M2
**Type:** result
**Stage:** Navigation M2 countermeasure

**What happened:** Fresh2,000-step run completed in113.81s using the existing cache; internal selection chose step1,500. On identical32-frame validation windows/1,312 supervised positions, frozen v0 vs v2: false holds44/582(7.56%) to28/582(4.81%), stop recall250/560(44.64%) to238/560(42.50%), speed MAE2.621 to2.583m/s, light accuracy50.38% to35.21%. Contradictions347 to0 by architecture.
**Cause:** Stop intent is now coherent, but the student still cannot identify most required stops; improved release does not establish safer driving. The bundle also changed loss/history/LR/training budget, so it is not an isolated ablation.
**How we handled it:** Preserve both frozen checkpoints and source archives; comparison uses same windows/burn-in and cache, with no heldout_dev/Town05 input. Proceed only to the frozen small TRAIN diagnostic, not long training or deployment.
**Verified by:** reports/navigation_m2_stop_go_20261016.json; both source archives verify. CPU suite235 passed/two skipped before recorder fix.
**Lesson:** Eliminating an output contradiction is a structural property, not evidence that a model learned when to stop.
**Interview angle:** Why did you not call coherent stop/go outputs an improvement in driving?

## 2026-10-16 10:04 - Recorder prefix collision found; preserve stopped pilot and fix episode paths
**Type:** bug
**Stage:** Navigation CARLA recording infrastructure

**What happened:** The pinned leaderboard_autopilot local evaluator writes args.record+".log", overwriting it on every route. The new pilot used the same path for its console. Binary CARLA_RECORDER bytes replaced text logs; old multi-route runs also retain only the last native recording per invocation. Per-episode MP4/world traces/raw scores are separate.
**Cause:** I treated record as a directory based on the other upstream evaluator and failed to check the active pinned local implementation before naming the console.
**How we handled it:** Stopped pilot with its STOP file; retained outputs/navigation_m2_stop_go_pilot_20261016 and every outcome. Added distinct recordings/route_ID_repN.log prefixes in the lifecycle adapter, refusing overwrite and restoring args.record. Restart the same six cases/seeds/checkpoints in a fresh retry directory; no route/criterion/reward changes.
**Verified by:** Regression invokes the adapter for two routes with the actual local filename convention and preserves console text/each binary; pending fresh CARLA verification. The first pilot has six expert records, last interrupted by STOP; no camera trials yet.
**Lesson:** Check the active vendor file semantics; distinct file extensions alone do not prevent log/recorder collisions.
**Interview angle:** How did you detect and recover from binary recorder corruption without discarding outcomes?

## 2026-10-16 10:10 - Qualify earlier speed/brake conflict interpretation
**Type:** decision
**Stage:** Navigation M2 target semantics

**What happened:** Audited clean teacher labels:987 training and268 validation timestamps combine positive target>0.15m/s with actuator brake>0.5. Every one has actual ego speed above its target; none has ego speed<0.2m/s. These are ordinary deceleration examples, not inherently contradictory teacher behavior.
**Cause:** Earlier journal/report wording called every positive-speed/brake co-occurrence a conflict. That count alone cannot establish an actuator error; actual speed and intended deceleration matter.
**How we handled it:** Preserve counts/checkpoints and qualify them as co-occurrence diagnostics. The concrete blocked-policy state with zero ego speed, positive target and brake remains a demonstrated release failure. Semantic-stop v2 separates zero-speed intent from such comfort braking; no scoring/data changes.
**Verified by:** outputs/navigation_m2_stop_go_20261016/teacher_braking_audit.json, same frozen cache arrays and internal location roles.
**Lesson:** A positive speed target can require braking when the vehicle is traveling faster than it.
**Interview angle:** Why is positive target speed plus a brake command not automatically a contradiction?

## 2026-10-16 10:25 - Frozen TRAIN comparison finishes: progress gain, rule failures remain
**Type:** result
**Stage:** Navigation M2 closed-loop countermeasure

**What happened:** All18 fixed TRAIN trials finished. Expert/v0/v2 arrivals6/6,3/6,4/6; rule-compliant arrivals6/6,2/6,3/6; strict any-infraction-free3/6,1/6,2/6. V0 blocked three times and ran one red; v2 blocked twice, ran one red and violated one stop sign. Zero recorded ego collisions in all24 raw records, including the first stopped infrastructure attempt.
**Cause:** Open perception/student-state/release gaps. V2 passes junction111 that blocked v0, but still stops with coherent zero targets (stop probabilities0.93/0.72). Both cameras cross release/memory123 while the bound light stays Red. Stop143 ends blocked at98.11% (2.76m from goal) after a stop violation; not an arrival.
**How we handled it:** Do not promote the model or extend training blindly. Preserve all attempts/checkpoints/videos. Next: fresh TRAIN-only DAgger at blocked/near-goal states, stop/release event coverage and localized rule perception. Only85 stationary teacher-go training frames occur after the first5s, out of14,745 total; this motivates targeted data, not a claim of proven causation.
**Verified by:** reports/navigation_m2_stop_go_20261016.md/json and navigation_m2_stop_go_pilot_audited_20261016.json. Setup/signal-identity/source audits pass;18 videos decoded13,317 frames;18 distinct recorder headers and26,623 ego positions (zero partition violations). Future manifests/traces now identify actual checkpoint variant instead of inherited v1 tag; raw pilot records unchanged.237 CPU tests pass/two skipped; lint/format/whitespace clean. All services idle.
**Lesson:** Better route progress and fewer deadlocks do not compensate for a new traffic-rule violation.
**Interview angle:** What did the semantic-stop change improve, and why did you still reject it as a safe driver?

## 2026-10-16 10:25 - Extra memory diagnostic initially omitted weight loading; invalid output excluded
**Type:** mistake
**Stage:** Navigation M2 streaming diagnosis

**What happened:** An extra analysis-only script constructed both architectures but omitted load_state_dict. It produced zero stop recall for both random models; I caught the unexpected values while checking raw predictions and canonical cache inputs, before using them for any conclusion.
**Cause:** Missing checkpoint-loading line in the one-off diagnostic, not in the trainer, main assessor, or actual camera runtime (all of which load correctly).
**How we handled it:** Retained initial output/source as streaming_internal_diagnostic_invalid_unloaded_model.json and streaming_assessment_invalid_unloaded_model.py. Added strict checkpoint loading, archived corrected script, reran both frozen policies. No training/selection/real driving outcome changed.
**Verified by:** Corrected whole-episode internal validation on2,614 shared positions: v0/v2 stop recall500/1,098(45.54%)/476/1,098(43.35%), false holds116/1,202(9.65%)/50/1,202(4.16%). Hidden carries10Hz history through each episode; first16 frames burned only once. These are teacher trajectories, not student-state coverage or proof of recurrent drift. Run folder has scripts, weight hashes and both raw outputs.
**Lesson:** Verify loaded parameters and canonical inputs before interpreting a surprising model diagnostic.
**Interview angle:** How did you prevent an analysis script mistake from changing the experimental conclusion?

## 2026-10-16 11:23 - DAgger collection separates teacher corrections and preserves internal validation
**Type:** decision
**Stage:** Navigation M2 correction data

**What happened:** Implemented an isolated camera rollout with privileged expert labels/braking and explicitly marked3s recovery after4s stationary expert-go. Student requested, teacher, selected and next-tick observed controls remain distinct. Assisted episodes are censored, never clean imitation; a separate reader exposes only instantaneous expert corrections from audited rule-safe attempts.
**Cause:** The stop/go model still blocks with coherent zero targets and misses rule stops; only85 stationary teacher-go training frames occur after5s. Repeating the old training is unsupported.
**How we handled it:** Freeze old internal validation geography as well as heldout_dev/Town05. Geometry clearance retains235/300 TRAIN routes. Fresh six-family Town03 pilot86/101/111/123/133/143, seed202610160+route id, one server/900s. All prior videos remain diagnostic-only. No future trajectory targets from assisted mixtures and no control/reward/benchmark change.
**Verified by:**244 CPU tests/two skips; recovery cancels on hazards, privileged imports stay blocked, assisted episodes cannot enter clean imitation, and collision outcomes cannot provide correction labels. Frozen reports/navigation_m2_dagger_guard_20261016.json and navigation_m2_dagger_protocol_20261016.md declare integrity/exposure gates before collection.
**Lesson:** DAgger must correct student states without relabeling student actions as expert behavior or moving the validation boundary.
**Interview angle:** How can you collect useful expert corrections without leaking oracle inputs or calling assisted driving a learned-policy success?

## 2026-10-16 11:35 - DAgger data pilot passes integrity/exposure gate with1,729 new timestamps
**Type:** milestone
**Stage:** Navigation M2 expert corrections

**What happened:** Six fresh Town03 TRAIN rollouts committed1,729 timestamps/59 shards/701.20MB. Instant expert labels:500 stopped,1,139 go;143 stationary go,118 after5s. Two recovery events;1,168/3,454 physics ticks selected privileged controls. All six assisted arrivals were rule-compliant, with zero recorded collision/red/stop events; this is collection behavior, not camera improvement.
**Cause:** Targeted student states expose false holds and release errors missing from most clean expert trajectories.
**How we handled it:** Store623 teacher-selected and1,106 student-selected10Hz action samples with separate expert labels; assisted episodes remain censored and never clean imitation. Full pixel/frame/action, source/checkpoint, scenario/signal and independent3,454-position audits pass, with zero internal/external partition violations. Applicable lamp ID/state matches its world snapshot; visibility remains unknown.
**Verified by:** reports/navigation_m2_dagger_assessment_20261016.json; all six videos fully decoded. Exposure gate passed before training. The watcher initially marked simulation complete before pixel auditing ended; corrected future progress to stay running through audit and to report audit failure. Original request/source archive retained.
**Lesson:** Measure the correction states actually collected; assisted success is data-collection evidence only.
**Interview angle:** What did DAgger add beyond repeating clean demonstrations?

## 2026-10-16 11:35 - Bounded correction fine-tune protects parent and preserves validation
**Type:** decision
**Stage:** Navigation M2 correction learning

**What happened:** Prepare a separate1.34GB raw correction cache, keeping the13.605GB original cache unchanged. Warm-start the unpromoted v2 best1,500 with a fresh AdamW;1,000 added updates, LR5e-5, batch2,32-frame history,1,200s cap.
**Cause:** Need to test whether new expert-go/release labels improve the same model, rather than change architecture/controller again.
**How we handled it:** Half original clips/half correction clips; correction endpoints balance stationary teacher-go errors and required stops. Current speed/brake/path/perception targets come from the expert, never student actions. Every correction future-speed mask is zero. Same frozen internal validation/selection score; evaluate/include the parent at step0, so a worse fine-tune cannot replace it. Freeze all training sources/provenance and save optimizer/RNG/sampler. No heldout_dev/Town05 or reward/benchmark changes.
**Verified by:** Cache/mixer tests reject assisted future targets, failed exposure gates and cross-episode clips; source candidate/assessment hashes retained. Results pending.
**Lesson:** A correction experiment should change the training evidence while protecting its parent and validation boundary.
**Interview angle:** How do you learn from assisted rollouts without treating their future as an expert or counterfactual trajectory?

## 2026-10-16 11:45 - Correction fine-tune rejected; protected parent remains selected
**Type:** result
**Stage:** Navigation M2 correction learning

**What happened:** Added1,000 updates in89.46s including the separate1.34GB cache. None of ten trained evaluations beat parent step0 (selection0.35115). Latest1000 worsened stop recall238/560(42.50%) to133/560(23.75%), speed MAE2.583 to3.248m/s; false holds28/582(4.81%) to10/582(1.72%). Lower false holds do not compensate for missed stops.
**Cause:** Open. Exact sampler replay contains14,030/32,000 stopped targets(43.84%), so absence of stop examples alone is not the explanation. Six-location diversity, ambiguous rule visibility and3.2s reset history remain hypotheses.
**How we handled it:** Keep the parent, reject latest1000, preserve all source/data/checkpoint/optimizer/RNG evidence. Selected model tensors are bit-identical to old v2; latest changed56 parameter tensors. Do not waste a CARLA repeat on identical selected parameters or resume latest blindly. Next investigate localized applicable-signal/stop-line labels and full stop/release sequences.
**Verified by:** reports/navigation_m2_dagger_finetune_20261016.json; outputs/navigation_m2_dagger_finetune_20261016/checkpoint_comparison.json and sampler_audit.json; reports/navigation_m2_dagger_20261016.md/figure.247 CPU tests/two skipped; unchanged internal split/M1 cache, frozen training sources verify. No new learned driving result, reward/benchmark change or Town05/heldout_dev use. All bounded jobs have completed.
**Lesson:** Protect and explicitly retain the parent when correction learning changes the safety tradeoff in the wrong direction.
**Interview angle:** What prevented a completed DAgger training run from replacing a less unsafe parent?

## 2026-10-16 12:30 - Audit applicable-light geometry before another student experiment
**Type:** decision
**Stage:** Navigation M2 localized rule perception

**What happened:** The correction fine-tune failed its unchanged selection gate. Existing next-light state labels identify a route-applicable actor, not an image-visible light; the recorded world snapshot does retain full ego pitch/roll for projection.
**Cause:** Open whether failures stem from perception, memory, limited diversity or optimization. Current labels do not distinguish these explanations.
**How we handled it:** Implement a bounded static Town03 light-housing/evaluator-stop-line inventory and offline projection/stop-duration audit of the existing six TRAIN correction episodes. No new driving/training or runtime map input. Treat geometric frustum overlap as an upper bound, never proof of visibility or a ready-to-use state label. Block the training geometry module from camera inference.
**Verified by:** Nine new CPU tests pass, including random full-pose matrices checked against installed CARLA, side-camera rotations, near-plane ambiguity, cross-server light identity and privilege separation. CARLA0.9.15 bounding-box source confirms component-to-world coordinates; avoid applying actor transforms twice.
**Lesson:** Validate that a supervision target can be observed before blaming the learner for missing it.
**Interview angle:** Why can a correct simulator traffic-light label still be unsuitable as visible-image supervision?

## 2026-10-16 12:38 - Resource guard counted a standalone probe as its own competing job
**Type:** bug
**Stage:** Navigation M2 pixel observability probe

**What happened:** The first stationary pixel probe exited before server startup: check_resources reported the calling scripts/probe_navigation_signal_pixels.py process itself as an existing simulation/training job. No images or driving result were produced.
**Cause:** The anchored process pattern correctly includes probe_ scripts but did not exclude the current PID.
**How we handled it:** Skip only os.getpid(); retain checks for every other matching job. Preserve the failed attempt marker and use a fresh retry directory. No jobs scripts or running processes changed.
**Verified by:** Regression test permits self and still rejects a second probe;13 targeted CPU tests pass. Retry remains a bounded TRAIN-only diagnostic.
**Lesson:** Process-exclusion guards must distinguish their caller from another worker.
**Interview angle:** How did you fix false duplicate-job detection without weakening concurrency protection?

## 2026-10-16 12:46 - Applicable-light projection audit narrows perception and memory hypotheses
**Type:** result
**Stage:** Navigation M2 localized rule perception

**What happened:** Static Town03 capture completed4.92s,38 lamps/107 housings. Projected1,729 audited TRAIN correction timestamps;766 have known lamp states. At least one housing projects into some camera in all766;227 have none in front. Largest effective-input housing is under6px tall in717/766(93.6%). Thirteen teacher zero-speed-intent events total500 frames; four last7.3–14.1s and contain428 frames(85.6%).
**Cause:** Perception versus memory cause remains open. This disproves the narrow all-camera-offscreen explanation for these frames, but does not establish bulb visibility, orientation or legibility. Existing full-history teacher validation already showed weak stop recognition.
**How we handled it:** Preserve per-frame candidate boxes/line projections, deterministic images, timelines and reproducible stats; visibility stays null and no state supervision is approved. API/evaluator stop positions differ3.5–6m on inspected lamps; keep reference semantics distinct. Source also confirms50m planner end padding, so next-light labels need not be an encounter within the scored mission. No policy/loss/controller/benchmark or M1 cache changes.
**Verified by:** reports/navigation_m2_rule_observability_20261016.md/json; output source/manifest/map/frame checks; projection matrices verified against CARLA.261 CPU tests pass,2 skipped including14 new checks. Earlier DAgger censoring wording means the four episodes with actual assistance; the other two are uncensored but all six remain excluded from clean imitation.
**Lesson:** Small target pixels and long events are measured constraints, not proof that either architectural change will improve driving.
**Interview angle:** What did the observability audit rule out, and what did it leave uncertain?

## 2026-10-16 12:46 - Stationary pixel probe hits settled-pose spawn failure and synchronous shutdown hang
**Type:** bug
**Stage:** Navigation M2 pixel observability probe

**What happened:** Fresh retry failed before imaging with "Spawn failed because of collision at spawn position" at a recorded settled ego pose. The owner sent its normal graceful shutdown, which timed out after20s; CARLA PID240326 remained waiting. STOP is present. A5s owned-process/port-verified RPC attempt to restore asynchronous ticking also timed out. No paired images or driving result exist.
**Cause:** Settled physical pose can fail spawn overlap tests; the diagnostic set synchronous mode before spawning and did not restore it when setup raised.
**How we handled it:** Spawn above road, disable physics and set exact recorded pose; verify attached camera transform/drift. Restore original world settings in finally; regression test covers setup failure. Preserve failed attempt/source/cleanup records. Do not start another server or force-kill. Additional graceful SIGTERM/SIGINT permission requested because the user's STOP-only rule prevents assuming a manual cleanup exception.
**Verified by:** Worker traceback, cleanup_failed.json, host anchored process/state check, RPC timeout;14 targeted CPU tests. These fixes are not yet verified by a completed live probe. Current blocker is owned-server cleanup, not training; selected camera policy remains unchanged.
**Lesson:** Synchronous simulators need an exception-safe ticking restore before process cleanup.
**Interview angle:** Why did a harmless static-camera setup failure leave CARLA alive, and how was the cleanup path corrected?

## 2026-10-16 12:49 - Visual review removes a chart ceiling that clipped speed requests
**Type:** bug
**Stage:** Navigation M2 diagnostic presentation

**What happened:** Visual inspection found the new timeline graph used an11m/s upper axis limit, hiding larger expert/student requested targets.
**Cause:** A fixed plotting range chosen before inspecting all recorded values.
**How we handled it:** Autoscale the upper limit and retain the original output/source. Publish the corrected graph with separate unclipped_plot_provenance.json and archived plotting source; no metrics or policy outputs change.
**Verified by:** Regenerated all six timelines from the same1,729 frozen rows; source, input and figure hashes saved. Lint/format/whitespace checks pass.
**Lesson:** Inspect chart bounds as well as the underlying statistics before publishing results.
**Interview angle:** How can a correct metric still be presented misleadingly in a plot?

## 2026-10-16 13:20 - Approved graceful retry clears stopped server; exact-frame static probe completes
**Type:** milestone
**Stage:** Navigation M2 signal-pixel evidence

**What happened:** User continued after the explicit graceful-cleanup request. Verified PID240326's executable,port and creation time; one extra SIGTERM exited it immediately. No SIGINT/SIGKILL. Retry2 passed spawn setup but timed out on the first camera frame; it restored world ticking and released the server. Retry3 completed four frozen TRAIN poses in8.71s,36 PNGs and13 projected housing crops.
**Cause:** The standalone static probe waited for a current GPU camera frame before allowing subsequent render ticks; first-frame delivery was not ready. Recorded settled poses also needed the already-implemented spawn-above/physics-off/exact-placement path.
**How we handled it:** Advance two extra ticks in the frozen scene, then drain the exact requested frame across all views; never substitute latest or zeros. Preserve each failed attempt. All four Red/Green/Red interventions are verified, ego drift<=1.6e-7m,camera-matrix error<=1.6e-5,all lights frozen. Seven crops visibly respond at routes111/123/133; route101 and back-facing housings have no clear response. Geometry alone cannot label visible state.
**Verified by:** outputs/navigation_m2_signal_pixels_retry3_20261016/assessment.json and signal_crop_pairs.png; exact-frame delayed-GPU regression test. The stopped server's graceful_cleanup_retry.json records authorization/identity/signal. All simulator jobs exited.
**Lesson:** Verify image evidence and frame identity separately from simulator light state.
**Interview angle:** How did paired rendering distinguish readable signal pixels from merely projected light housings?

## 2026-10-16 13:20 - Frozen student barely responds to verified red/green color changes
**Type:** result
**Stage:** Navigation M2 rule perception diagnosis

**What happened:** Production checkpoint loading verified all selected v2 tensors before/after inference. Each branch shares32 repeated-red history samples and then32 repeated Red,Green orRed-repeat images; stationary ego/navigation held identical. At all four poses the auxiliary state class stays Red in every branch. For the three poses with clear bulb changes, final P(green) changes only+0.0165,-0.0480,+0.0256 percentage points. Red branches still request5.01,6.49,6.31m/s with stop probability12.26%,4.42%,4.65%.
**Cause:** Exact mechanism open, but observable color changes do not meaningfully reach the state/stop heads in this controlled diagnostic. More stop examples alone or longer inference memory is not an established fix.
**How we handled it:** Keep the parent unchanged; no training/optimizer or driving claim. Archive every source/image/checkpoint hash and strict loaded-weight equality. Next implement localized phase supervision with broader TRAIN geography before another student fine-tune. Phase response, applicability and driving safety stay distinct.
**Verified by:** outputs/navigation_m2_signal_response_20261016/assessment.json; preprocessing equality test against production CameraPolicy.17 targeted CPU tests pass. Static repeated images omit original traffic and are not the M3 learned-policy driving counterfactual suite.
**Lesson:** An apparently correct Red prediction can be a prior that never changes when the actual lamp turns green.
**Interview angle:** How did you test whether the camera network reads signal color rather than predicting a location prior?

## 2026-10-16 13:30 - Static supervision preparation fixes approach/junction road-ID boundary
**Type:** bug
**Stage:** Navigation M2 phase-supervision data

**What happened:** Initial geometry-only preparation produced0 poses, preserving an empty first plan. The nearest waypoint to a crossing line was already inside the junction with a different road ID; an endpoint-ID equality check rejected every valid approach. No simulator/data run used that plan.
**Cause:** Topology changes at the line, while sparse waypoint samples straddle it. Town03 route133 has a closest point0.087m away on junction road668, whereas the line belongs to approach road1; the last approach sample is1.823m away.
**How we handled it:** Project onto polyline segments whose starting lane is the applicable approach. Retain the1m geometric gap check; no benchmark/scoring change. Add a regression for a crossing-line center between road IDs, refuse empty/incomplete plans, freeze freshv2.
**Verified by:** Fresh v2 gives51 poses/17 junctions/five TRAIN towns with no skipped geometry;25 targeted CPU tests pass. Initial plan retained for provenance.
**Lesson:** A road-ID discontinuity at a junction must not be mistaken for a missing geometric association.
**Interview angle:** Why did a valid stop line fail nearest-waypoint lane matching?

## 2026-10-16 13:30 - Freeze broader static phase-pair pilot before another camera experiment
**Type:** decision
**Stage:** Navigation M2 localized signal supervision

**What happened:** Freeze51 TRAIN poses across17 junctions/all5 towns, first4 eligible junctions per town (Town10HD has1 here). Red/green/yellow/red-repeat and seeded fixed background phases;612 native RGB images expected. All previous internal-validation and external-heldout locations remain excluded.
**Cause:** Frozen v2 responds weakly to verified bulb-color changes. Three familiar readable poses cannot support a general perception claim; broader localized supervision is needed.
**How we handled it:** Protocol and plan SHA d890973176b5d0fd24754cc04eee0bfd39327adcc327944edd1af500a1099b42 frozen before imaging. One server,900s cap, STOP, raw source/frame/map/pose/physical-identity audits. Static-response filter calibrated only on prior TRAIN probes; data gate requires all cases and responsive crops in5 towns/>=10 junctions. No action/future labels and no driving/model improvement claim. Supervised phase labels must count in later label-efficiency budgets.
**Verified by:** reports/navigation_m2_signal_pairs_protocol_20261016.md and plan_v2 JSON;25 targeted tests. No model/reward/benchmark change or multi-hour job.
**Lesson:** Use controlled training interventions to improve supervision, while preserving geography and honest label accounting.
**Interview angle:** How do static counterfactual images help perception without becoming benchmark gaming or oracle inputs?

## 2026-10-16 13:43 - Strict map guard catches packaged default map before static imaging
**Type:** bug
**Stage:** Navigation M2 multi-town static corpus

**What happened:** First corpus attempt requested Town01_Opt but startup_map.json showed Town03_Opt. Worker rejected "Wrong live TRAIN map" before producing images.
**Cause:** Packaged CARLA opened its default map despite the launch URL; the new collector omitted the explicit load_world already used by the inventory workflow.
**How we handled it:** Reuse explicit client.load_world when needed, then verify canonical town and exact frozen OpenDRIVE hash before capture. Preserve failed attempt; no mislabeled images entered data.
**Verified by:** Loader/hash regression test and subsequent five-town capture. No Traffic Manager or benchmark change.
**Lesson:** Verify and explicitly load the requested map; launch arguments alone are not proof.
**Interview angle:** How did you prevent collecting images from the wrong simulator town?

## 2026-10-16 13:43 - Missing shared-renderer geometry argument loses a final town summary
**Type:** mistake
**Stage:** Navigation M2 static data durability

**What happened:** I passed only the output path to the shared renderer and omitted its geometry path. It saved Town01 images but failed during final provenance serialization with "Namespace object has no attribute geometry". That attempt has no completed audit and is excluded.
**Cause:** Incomplete call-site adaptation of the shared renderer's arguments; per-town metadata was initially written only at the end.
**How we handled it:** Supply the required geometry path; write atomic metadata and image hashes after each completed pose. Preserve all attempts and use a fresh root; do not infer missing state/frame certification from filenames.
**Verified by:** The next attempt completed all51 frozen cases and612 image decodes/hashes; per-pose metadata is present. No checkpoint or learned-policy result changed.
**Lesson:** Commit capture metadata with each small unit, not just at the end of a town.
**Interview angle:** Why are saved RGB files alone insufficient to declare a data-collection attempt valid?

## 2026-10-16 13:43 - Five-town static capture completes but v1 label-data gate fails
**Type:** result
**Stage:** Navigation M2 localized supervision

**What happened:**51 poses/17 junctions/all5 TRAIN towns,612 PNGs,73.00s, all setup/integrity checks complete. The frozen v1 filter approves56 head/view candidates and55 yellow candidates across15 junctions but only4 towns; Town10HD none. Its5-town gate FAILS. No training consumes it.
**Cause:** In inspected Town10HD crops, actual R/G/R responses are stable, but green bulbs render near white: green-over-red18–22 fails the v1 cutoff30. RG MAE8.95–26.46 exceeds repeat variation0.54–2.03. This is a label-filter hue assumption, not evidence that the pixels contain no phase information.
**How we handled it:** Preserve the failed gate/source/PNGs. Develop a separately versioned local on/off-response test on TRAIN captures, with tests rejecting global drift/unstable repeats. Original v1 function and audit remain unchanged. No reward/benchmark or camera-setting change.
**Verified by:** outputs/navigation_m2_signal_corpus_retry2_20261016/data_audit.json and Town10HD crop sheet;28 targeted CPU tests. No heldout_dev/Town05 data or model improvement claim.
**Lesson:** Annotation heuristics can have a domain bias even when simulator phase labels are correct.
**Interview angle:** Why did you retain a failed data gate despite seeing real signal changes in the images?

## 2026-10-16 13:43 - Freeze response-based validator and fresh phase-pair collection
**Type:** decision
**Stage:** Navigation M2 label validity

**What happened:** Freeze validator2: same mean-change/repeat thresholds, plus separate red-on and green-on intensity gains>=20 and max channel change>=25. Remove the ungrounded absolute-green-hue assumption; global brightness drift cannot satisfy both gain signs. Preserve v1 failure.
**Cause:** Need state supervision that can use bulb-position/phase evidence in the observed near-white shader style, without simply lowering a color threshold after a failed gate.
**How we handled it:** New protocol_v2 and plan_v3 SHA92335457bb8521593450eff76007a119e663a49b01c9820689b049b7c32c911e before fresh images. Same51 poses/17 junctions/5 towns and unchanged integrity/coverage gates; new seeded backgrounds. New annotation version, not a revised driving benchmark. No model fitting yet.
**Verified by:** Tests preserve original hue-rule failure and accept localized near-white phase changes while rejecting static/global/unstable controls. Fresh results pending; geometry/internal validation guards unchanged.
**Lesson:** A flawed training-label validator needs a versioned replacement and new evidence, not a retroactive pass.
**Interview angle:** How did the annotation revision differ from gaming an evaluation threshold?

## 2026-10-16 14:00 - Native Town10HD crash recovered without repeating completed towns
**Type:** bug
**Stage:** Navigation M2 static corpus recovery

**What happened:** Fresh validator2 collection completed48 poses in Town01–04, then CARLA caught native Signal11 during Town10HD camera warmup; no Town10 images were committed. RPC teardown timed out but the server itself exited. Cause of the native crash remains open.
**Cause:** Native simulator failure, not an observed GPU resource shortage or learned-policy error.
**How we handled it:** Add a bounded resume orchestrator that requires original archived/current worker bytes and frozen plan hashes to match. Verify completed-pose PNG hashes, skip all four complete towns, archive the failed Town10 directory and retry only its3 poses. Preserve original request/source and a separate resume-source/ledger. No force kill or duplicate server.
**Verified by:** outputs/navigation_m2_signal_corpus_v2_20261016/resume_attempts ledger records only Town10HD retried; final51 cases/612 decoded/hash-checked PNGs. Worker-source mismatch regression refuses mutation before any recovery. All simulator jobs exited.
**Lesson:** Per-pose durability and exact worker provenance make simulator crashes recoverable without new data selection.
**Interview angle:** How did you recover a simulator crash without rerunning or mixing the already collected data?

## 2026-10-16 14:00 - Fresh response-based corpus passes unchanged five-town data gate
**Type:** milestone
**Stage:** Navigation M2 localized phase supervision

**What happened:** The fresh v2-validator corpus completes51 poses/612 images.63 head/view pairs have local opposite-signed phase responses,62 support yellow, across16 distinct junctions/all5 TRAIN towns. Every frozen pose, source/frame/map/physical identity, non-target light stability and image decode/hash check passes. Earlier v1 gate remains false.
**Cause:** The response test can use repeatable bulb-position changes without assuming strongly green-colored pixels; it still rejects global drift and unstable repeats.
**How we handled it:** Release this explicitly static supervised corpus only. No controls/future labels, original-traffic visibility or learned driving result. Preserve old failure and all retries. For the next prototype, freeze a whole-junction component split and count all phase supervision in future label budgets.
**Verified by:** outputs/navigation_m2_signal_corpus_v2_20261016/data_audit.json; plan_v3 SHA92335457bb8521593450eff76007a119e663a49b01c9820689b049b7c32c911e;34 targeted tests pass.
**Lesson:** A passed supervision-data gate is permission to test a perception component, not to claim safer driving.
**Interview angle:** What evidence separates successful data collection from a successful learned driver?

## 2026-10-16 14:00 - Freeze a spatial RGB localizer experiment before another driving fine-tune
**Type:** decision
**Stage:** Navigation M2 learned signal perception

**What happened:** Implement a small fully convolutional stride2 state/localization grid on native RGB, without global pooling or positional/map inputs. Supervise static phase-supported housing centers; ignore all other lamp boxes. Keep13/4 whole-junction split,468/144 images and179/72 unique foreground centers.
**Cause:** Frozen camera model barely reacts to verified colors; localized spatial supervision is a concrete alternative to another data-only speed-head continuation.
**How we handled it:** Protocol:1,000 updates, batch16, LR1e-3, BF16/FP32,1,200s cap, fixed foreground CE+0.2 background CE selection. Evaluate random step0 and every200. Full optimizer/RNG/source/data recovery. Export best_component.pt with perception-only kind, never a driving checkpoint. Original policy/cache/benchmark unchanged. Validation is development-only: annotation calibration already inspected some of these TRAIN locations.
**Verified by:** reports/navigation_m2_signal_grid_protocol_20261016.md; translation-equivariance, ignored-gradient, crop/validation isolation, failed-gate and worker-provenance tests pass.34 targeted tests; results pending.
**Lesson:** Preserve spatial evidence and test it directly before expecting a recurrent controller to recover it after pooling.
**Interview angle:** Why does the new signal component use spatial outputs instead of another globally pooled auxiliary classifier?

## 2026-10-16 14:09 - Sparse-center localizer is not ready; later updates overfit
**Type:** result
**Stage:** Navigation M2 signal component

**What happened:**1,000 updates/13.17s,37,772 parameters. Fixed selection chooses200:32/72 center labels(44.44%), green4/18,yellow6/18,red22/36; background active cells@.5=9.38%. The duplicated-red class prior alone gives50% center accuracy. Latest1000 has28/72(38.89%), within4px80.56% but background active8.86%; foreground CE grows1.25 at200 to4.66 at1000. No component/policy promotion.
**Cause:** Open; small data, approximate housing centers and spatial generalization remain hypotheses. Nearby predictions are more frequent than correct center states, which motivates testing target density rather than a longer repeat.
**How we handled it:** Preserve sources, all validation points/checkpoints/optimizer/RNG. Original driving model unchanged. Next use regional training supervision inside supported housings, while keeping every evaluation center/background mask and all hyperparameters fixed. No new driving or benchmark claim.
**Verified by:** outputs/navigation_m2_signal_grid_20261016/review.json and validation.jsonl; archived code and corpus provenance.35 targeted tests now include unchanged-evaluation-mask checks. The run's scalar-log warning is fixed via explicit detach for future logging; model/loss values unchanged.
**Lesson:** Better nearby activation does not establish a correctly localized, calibrated perception head.
**Interview angle:** Why was the completed spatial signal model still rejected for integration?

## 2026-10-16 14:09 - Regional training-target test preserves the original component evaluation exactly
**Type:** decision
**Stage:** Navigation M2 signal target alignment

**What happened:** Add a separate regional TRAIN target plane; the evaluation plane remains the original sparse centers/ignored boxes/background. Same model, seed, sampler,1,000 updates, batch16, LR1e-3 and selection. No architecture, image, split or score changes.
**Cause:** A one-cell center target can encourage a narrow response that drifts within an approximate projected housing; this is a hypothesis, not a demonstrated cause.
**How we handled it:** Reconstruct the original dataset with its archived builder and compare against the regional builder. Require all612 RGB arrays, all612 evaluation masks, sample metadata, roles and validation indices to match byte-for-byte; only training targets differ. Freeze this bounded test before fitting and retain the original negative run.
**Verified by:** outputs/navigation_m2_signal_grid_20261016/region_evaluation_invariance.json; same144 validation images/72 centers. No canonical driving benchmark or source data changes.
**Lesson:** Change training supervision without quietly moving the evaluation target.
**Interview angle:** How did you improve a localization loss while proving the measured task stayed unchanged?

## 2026-10-16 14:36 - Regional signal supervision also fails the component diagnostic
**Type:** result
**Stage:** Navigation M2 localized signal perception

**What happened:** The fixed regional-target experiment completes1,000 updates/11.14s. Selected1000 scores28/72 centers(38.89%):red26/36,yellow0/18,green2/18; within4px probability>=.5 only2.78%, background activation2.99%. Its selection score1.57853 is worse than center supervision1.55816; neither beats the50% Always-Red center prior.
**Cause:** Open; changing target density alone does not solve spatial/state generalization on this small correlated corpus.
**How we handled it:** Reject integration, retain both runs and original byte-identical evaluation proof. No longer repeat, driving-policy replacement or metric change.
**Verified by:** outputs/navigation_m2_signal_grid_region_20261016/review.json and best_component.pt SHA95e31671a8c9ae72717e1f4eceda2331f9bab5b4a44353e3f4ec9d0727066771; reports/navigation_m2_signal_perception_20261016.md/json.
**Lesson:** A hypothesis-driven training-label change can fail honestly even when it reduces background activation.
**Interview angle:** Why did broader positive regions not justify integrating the second traffic-light model?

## 2026-10-16 14:36 - Exact-crop appearance diagnostic separates state recognition from detection
**Type:** result
**Stage:** Navigation M2 error isolation

**What happened:** A fixed class-balanced L2 logistic regression on native annotated housing crops resized16x32 gets61/72 validation states(84.72%),179/179 training, in2.57s including loading. Red30/36,yellow17/18,green14/18; four whole-junction development groups remain unchanged.
**Cause:** Localized appearance has usable information; the two full-image grids fail to exploit it reliably. This does not isolate architecture or establish a causal explanation for all driving failures.
**How we handled it:** Explicitly mark PRIVILEGED ORACLE CROP diagnostic. Freeze features/model settings in the request before fitting, export coefficients and verify reloaded predictions exactly. No inference boxes supplied to the camera driver; do not call84.7% a mathematical ceiling, untouched test or driving improvement.
**Verified by:** outputs/navigation_m2_signal_oracle_crops_20261016/assessment.json and oracle_rgb_linear.npz SHA928e47ef9c29a2ddb64b4d1aeeadaa144351cc63b7f8bed394a7e6ba4873f832.
**Lesson:** An oracle localization diagnostic can identify useful visual evidence without being presented as a deployable system.
**Interview angle:** What does high accuracy on exact traffic-light crops prove, and what does it omit?

## 2026-10-16 14:36 - Declared COCO detector baseline misses the smallest signals
**Type:** result
**Stage:** Navigation M2 reference perception baseline

**What happened:** Before inference, freeze official FasterRCNN-ResNet50-FPN-v2 COCO weights, confidence.5, IoU.3/.5 and the unchanged144 validation RGB images. The bounded94.70s diagnostic detects14/72 supported housing/phase labels at IoU.3,8/72 at.5; the learned crop classifier is correct on all14 matches. Native-height breakdown:0/40 under8px,12/28 at8–15px,2/4 at>=16px. These are correlated sparse labels, not independent encounters or full detection AP.
**Cause:** Open beyond the observed strong size dependence; generic COCO priors alone do not provide sufficient CARLA signal coverage.
**How we handled it:** Preserve the negative baseline and external supervised-pretraining declaration. Weights downloaded from official PyTorch to the project cache with hash verification; strict loading. Internal resize interpolates existing512x288 RGB, not a new sensor. No threshold search, driver integration or matched JEPA claim.
**Verified by:** outputs/navigation_m2_reference_detector_20261016/assessment.json, predictions and overlays; full weight SHAdd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf. Three-view GPU inference mean182.7ms/p95195.3ms, peak allocated1.67GB; inference alone exceeds10Hz budget.
**Lesson:** A stronger pretrained detector needs both domain/size validation and latency checks before deployment in a control loop.
**Interview angle:** Why was the externally pretrained detector rejected despite every matched state being correct?

## 2026-10-16 14:36 - Continue M2 through auditable perception and temporal state before driving fine-tuning
**Type:** decision
**Stage:** Navigation M2 next experiment

**What happened:** Consolidate fixed-pixel response, fresh data gate, both negative spatial fits, privileged-crop classifier and frozen detector into a reproducible report/graph. The original camera policy and original cache manifest retain their recorded SHA256 hashes; no heldout_dev/Town05 data or new driving result.
**Cause:** More stop/go updates cannot be assumed to repair a representation that barely changes with verified signal pixels. Detection, applicable-lane selection and complete stop/release memory each need measurable evidence.
**How we handled it:** Next build RGB small-object proposal/state supervision with audited negatives/unknown regions and broader TRAIN coverage; measure latency, then visual tracking and applicability, then full7–14s stop/release sequences. Integrate only after evidence improves and run matched overrides-off TRAIN pilots. No multi-hour run, benchmark/reward/input change, jobs-script edit or push. Replace stale live cleanup blocker in current docs; retain failed-attempt history.
**Verified by:** scripts/assess_navigation_signal_perception.py verifies evidence/component/weight hashes and rebuilds reports/navigation_m2_signal_perception_20261016.json/figure; full suite285 passed,2 skipped,64 historical AMP warnings in17.65s. Live anchored process check finds no jobs or CARLA, GPU idle43C; legacy traffic queue is complete (its health snapshot is historical).
**Lesson:** Treat component diagnosis and data integrity as progress while withholding the driving claim until the unchanged closed-loop test succeeds.
**Interview angle:** How do you prevent a successful data pilot or perception diagnostic from becoming an overstated driving result?

## 2026-10-16 15:12 - Freeze a two-stage small-anchor signal detector experiment
**Type:** decision
**Stage:** Navigation M2 housing localization

**What happened:** Implement a binary RGB housing detector initialized from the declared COCO FasterRCNN model, with smaller vertical anchors and frozen backbone/ROI features. Separate object detection from the existing saved RGB crop-state classifier. Keep all468/144 images,13/4 junctions and179/72 supported boxes from the valid static corpus.
**Cause:** Previous scratch grids fail, while exact-crop appearance61/72 and generic detector recall14/72 suggest testing spatial scale and domain adaptation directly.
**How we handled it:** Freeze reports/navigation_m2_signal_detector_protocol_20261016.md before fitting:200 updates, batch2, FP32 AdamW5e-4,900s cumulative cap; eval0/50/100/200, select fixed-seed mean validation detector loss. Ignore uncertain projected lights in BOTH RPN and ROI negative sampling after correct resize, while preserving positives. Inference receives RGB only. Full source/optimizer/RNG/checkpoint provenance and STOP/resume; no driving-policy or benchmark change.
**Verified by:**15 targeted tests pass, covering independent IoMin examples, transformed ignore boxes, both samplers, frozen features, split sampling and forbidden training-data imports. Dataset preflight checks the original split/denominator. Live anchored process check shows no running trainer/CARLA; legacy traffic queue is complete.
**Lesson:** A two-stage perception test needs explicit treatment of uncertain background and a matched adapted-step0 control.
**Interview angle:** How do you adapt a detector when only some visible traffic lights have trustworthy annotations?

## 2026-10-16 15:19 - Small-anchor adaptation improves sparse localization but is not ready to drive
**Type:** result
**Stage:** Navigation M2 signal detector

**What happened:** Fixed200-update run completes111.54s;1,194,265 trainable/43,256,153 total parameters, peak allocated1.69GB. The predeclared loss selects50:recall30/72 atIoU.3,22/72 at.5,detected-and-correct state21/72;13/40 under8px found. Adapted step0:2/72 and5,049 approximate-background candidates; selected50 has681 across144 images. Frozen backbone/ROI parameters and buffers remain byte-identical.
**Cause:** Domain adaptation with small anchors improves this sparse development task; small-light misses, box/state alignment, background candidates and latency remain open.
**How we handled it:** Keep selected50 despite latest200 reaching45/72 and35/72 joint states; its loss.12498 is worse than50's.11866. Preserve all4 prediction sets, sources/optimizer/RNG, selected and latest. No policy promotion or held-out use. Mean three-view inference186.2ms,p95190.6ms still exceeds10Hz budget. Old91-class COCO14/72 remains a separate architecture reference.
**Verified by:** outputs/navigation_m2_signal_detector_20261016/review.json; selected SHA8df27ee218056e392cc5ddb0f50fef9263a1e8b81517b1d5ecbeba8038a6e49b. Independent maximum-cardinality one-to-one box matching reproduces all counts; no prediction is counted twice. Full suite290 passed,2 skipped.
**Lesson:** Report the selected checkpoint and all operational limits even when the latest model has a more attractive recall.
**Interview angle:** Why report30/72 when a later detector checkpoint finds45/72 lights?

## 2026-10-16 15:19 - Resume review catches interrupted-evaluation bookkeeping corner case
**Type:** bug
**Stage:** Navigation M2 checkpoint recovery

**What happened:** Code review after the completed run found that STOP during a scheduled validation could save the new training step with the previous validation. Resuming at final200 would otherwise skip the interrupted validation and mark the run complete.
**Cause:** Training step and last completed validation step were not tracked separately. The completed run did not resume and is unaffected.
**How we handled it:** Future worker checkpoints now store validation_step and rerun pending scheduled validation before further training/completion. Keep the original run's archived worker unchanged; changed-worker resume remains refused rather than silently mixing versions.
**Verified by:** Inspection of save/resume/selection paths; regression verifies validation interruption restores Torch/NumPy sampler RNG. No interrupted-run end-to-end recovery is claimed from this review;8 focused detector tests pass.
**Lesson:** A completed optimizer step does not imply its scheduled evaluation committed successfully.
**Interview angle:** How can a resumable run accidentally skip its final evaluation?

## 2026-10-16 15:19 - Freeze crop-alignment augmentation on fixed detector outputs
**Type:** decision
**Stage:** Navigation M2 normalized crop-state recognition

**What happened:** Keep the selected50 detector, its30 IoU.3 one-to-one matches, every predicted box and72-label denominator fixed. Test outward-only TRAIN crop augmentation for the saved linear RGB state classifier; no new simulator run.
**Cause:** Only21/30 matched predicted boxes receive the correct state, while exact annotated crops scored61/72; box alignment is a testable hypothesis, not yet an established cause.
**How we handled it:** Freeze reports/navigation_m2_signal_crop_alignment_protocol_20261016.md:16 variants/source crop,179 sources,2,864 crops,seed20261016; same C1 balanced L2 classifier/features. Weight variants1/16 to preserve total training-example weight179. Expansions retain all certified source pixels. Score both oracle GT and fixed predicted ROIs separately; do not select detector200 or tune on validation.
**Verified by:** Deterministic/bounds/full-housing-retention tests and frozen prediction/component hashes. One-to-one audit and selected detector remain unchanged; results pending.
**Lesson:** Robustness augmentation should preserve both the target evidence and the original source-example weighting.
**Interview angle:** How can you test crop alignment without improving the measured detector by changing its boxes?

## 2026-10-16 15:23 - Outward crop augmentation improves fixed-ROI state recognition
**Type:** result
**Stage:** Navigation M2 signal crop alignment

**What happened:** The frozen CPU experiment completes in3.11s, including0.184s fitting/44 optimizer iterations. On exactly the same selected-detector30 predicted boxes, correct states improve21 to24. Overall detected-and-correct denominator remains72 (24/72), and detector recall stays30/72. Privileged exact-crop validation improves61/72 to68/72 separately.
**Cause:** Consistent with crop-alignment sensitivity; augmentation also broadens the appearance training distribution, so this does not isolate a single physical cause.
**How we handled it:** Preserve old classifier, all detector outputs/selection, weighted2,864-crop manifest, source hashes and exported coefficient parity. No geometry at prediction, no validation tuning or driving claim. Selected pipeline still misses42 labels and has background/lane/latency issues.
**Verified by:** outputs/navigation_m2_signal_crop_alignment_20261016/assessment.json; artifact SHAca08ac5a7c301c76489a9fc8cde692a9da213c0fa679fd7307e7dc0d2a6b5de2; same frozen one-to-one assignments and72-label denominator. Eight focused detector tests pass.
**Lesson:** Small, supervised robustness changes can improve the intended intermediate task while leaving all measured detector outputs fixed.
**Interview angle:** What improved after crop augmentation, and why is it still not a safer-driving result?

## 2026-10-16 15:23 - Freeze controlled FP32 versus FP16 inference audit
**Type:** decision
**Stage:** Navigation M2 inference optimization

**What happened:** Selected detector latency186ms exceeds10Hz. Test FP16 autocast against FP32 on identical144 native images, selected50 checkpoint and the new fixed crop classifier; no fitting.
**Cause:** Precision is a bounded operational optimization to measure before choosing another architecture; accuracy/threshold changes must remain visible.
**How we handled it:** Pre-inference request freezes.5 confidence, IoU.3/.5, three warmups per mode, alternating paired order on48 identical three-view batches and300s cap. RGB-only inference, one-to-one sparse-label scoring and unchanged-parameter/buffer check. Compare FP32 counts to the archived selected detector and24-state classifier result before interpreting FP16. Preserve all numerical changes, not just speedup.
**Verified by:** scripts/assess_navigation_signal_precision.py; request/source archive committed before warmup. No runtime input, sensor, benchmark, reward or controller change.
**Lesson:** Faster arithmetic is useful only alongside an explicit accuracy and state-preservation audit.
**Interview angle:** How do you evaluate mixed-precision perception without hiding numerical regressions?

## 2026-10-16 15:24 - FP16 improves detector latency but does not meet10Hz interval
**Type:** result
**Stage:** Navigation M2 controlled inference

**What happened:** Paired inference on144 identical images/mode completes28.96s. FP32 three-view mean184.1ms,p95195.5ms,1.67GB peak; FP16 mean106.5ms,p95112.5ms,1.35GB. FP32 reproduces selected30/72 detections,24/72 joint states; FP16 gives31/72 and25/72, so it is not numerically identical (1,469 versus1,452 predictions).
**Cause:** Lower precision speeds arithmetic and changes rounding/NMS/threshold outcomes. The one-label increase is not evidence of improved safety.
**How we handled it:** Preserve both outputs, unchanged checkpoint/classifier and parameter/buffer checks. No runtime promotion; FP16 remains above100ms before other processing.
**Verified by:** outputs/navigation_m2_signal_precision_20261016/assessment.json and archived request/sources; all model parameters/buffers equal saved FP32 checkpoint after inference.
**Lesson:** Report numerical differences even when an optimization happens to increase an aggregate count.
**Interview angle:** Why did the faster FP16 detector still fail the intended timing budget?

## 2026-10-16 15:24 - Freeze one proposal-count latency tradeoff
**Type:** decision
**Stage:** Navigation M2 inference cost

**What happened:** Test an additional FP16 inference configuration with200 post-NMS RPN proposals/image instead of1,000. Keep pre-NMS limits,100 final detections,.5 confidence, selected50 weights and crop classifier fixed. No training or input change.
**Cause:** Fewer proposals reduce expensive frozen ROI feature work; any lost light must be reported rather than hidden by a threshold change.
**How we handled it:** New fresh output/request before warmup; controlled FP32/FP16/FP16-cap200 modes rotate order on the same144 images, three warmups each,300s cap. Preserve the first precision audit. No parameter tuning grid or driving integration.
**Verified by:** scripts/assess_navigation_signal_precision.py --include-cap200; request records each proposal cap. The same one-to-one IoU.3/.5 and fixed classifier scoring apply to every mode; results pending.
**Lesson:** An operational proposal budget is a measurable speed/coverage tradeoff, not a free optimization.
**Interview angle:** How can you reduce a two-stage detector's inference cost while keeping the tradeoff honest?

## 2026-10-16 15:33 - Reject the faster200-proposal inference default after measured coverage loss
**Type:** result
**Stage:** Navigation M2 latency/coverage tradeoff

**What happened:** Controlled FP32/FP16/FP16-cap200 audit completes all432 image evaluations. FP32 mean183.1ms,p95187.9,30/72 detections and24 joint states; FP161000-proposal mean107.5ms,p95112.2,31/72 and25 joint; FP16200-proposal mean85.6ms,p9589.3,24/72 and19 joint. Final prediction counts1,452/1,469/596. All parameters/buffers unchanged.
**Cause:** Reducing ROI work lowers latency but drops7 of31 sparse light detections and6 of25 joint states. Fewer predictions alone do not establish better precision under sparse annotations.
**How we handled it:** Reject200 proposals as the default optimization, preserve all variants and fixed thresholds. No end-to-end10Hz, rule-understanding, driving or numerical-parity claim. Full FP16 remains above100ms before preprocessing/classifier/controller.
**Verified by:** outputs/navigation_m2_signal_proposal_budget_20261016/assessment.json, requests and per-mode predictions; original precision audit preserved. Selected50 checkpoint and crop classifier fixed throughout.
**Lesson:** An inference optimization that meets a timing target can still be unacceptable because it loses perception coverage.
**Interview angle:** Why reject a detector configuration that finally fits within100ms?

## 2026-10-16 15:33 - M2 component gains documented with immutable inputs and protected driving parent
**Type:** milestone
**Stage:** Navigation M2 perception engineering, not the driving gate

**What happened:** Finish two-stage detector adaptation, weighted crop robustness and controlled inference audits. Selected detector30/72 versus adapted-step02/72; fixed-box state21/30 to24/30. Generic91-class reference14/72 remains a separate architecture baseline. No closed-loop result or M2 pass.
**Cause:** This is measurable progress on localized RGB perception, with misses, approximate backgrounds, ego-lane applicability and memory still unresolved.
**How we handled it:** Add reports/navigation_m2_signal_detector_20261016.md/json, architecture/graphs/native prediction examples and a read-only summary builder. Verify original audits instead of rewriting inputs hashed by later experiments. Update CLAUDE/handoff/plan/runbook/README and interview evidence. Progress explicitly displays latest200 recall45/72 separately from selected50 recall30/72. Next mine/audit TRAIN-only hard negatives and broaden approach/lighting views, then tracking/applicability and complete stop/release memory. Preserve validation groups; no more blind long fit.
**Verified by:** All612 image hashes, archived worker sources, coefficients/components, fixed assignment/prediction hashes and original camera-policy/cache hashes match.293 tests passed,2 skipped,64 historical AMP warnings in19.16s; targeted lint/diff checks pass. Anchored live check including optional-u finds no trainer/CARLA; GPU0%,41C,35.81GiB disk free. No jobs-script/reward/benchmark/sensor changes, held-out use, multi-hour run or push.
**Lesson:** Preserve the selected model and complete evidence trail while distinguishing a better component from a better driver.
**Interview angle:** What is actually improved at this milestone, and what must still be demonstrated?

## 2026-10-16 17:23 - Freeze TRAIN-only hard-negative mining and protected head refinement
**Type:** decision
**Stage:** Navigation M2 background discrimination

**What happened:** User continued after the selected detector/crop/latency results. Prepare a frozen mining protocol on117 red-phase TRAIN views only, excluding all four component-validation junctions and prior geographical guards. Same parent50, FP32,1,000 proposals and.5 confidence.
**Cause:** Selected model has681 approximate-background candidates/144 validation images; illustrated validation mistakes must not become training examples.
**How we handled it:** Conservative negative candidates must avoid every padded lamp region and remain stable across all four static phases; at most2/view. Freeze data gate>=20 boxes,>=6 TRAIN junctions,>=3 towns before mining. If it passes, fit only2050 ROI classifier parameters with explicit positive/hard-negative ROI replay, parent recall/state/background protection and the same200-update/900s cap. Geometry is labels only; no runtime or driving-benchmark change.
**Verified by:** reports/navigation_m2_signal_hard_negative_protocol_20261016.md; regression rejects a large negative enclosing a tiny lamp and rejects phase-responsive crops. Anchored live check shows no trainer/CARLA; GPU readiness passes. Results pending.
**Lesson:** Mine the learner's mistakes in training locations while keeping inspected validation failures out of the training set.
**Interview angle:** How can hard-negative mining improve perception without leaking validation examples?

**Timestamp correction logged 2026-10-16 17:22:** The immediately preceding mining-decision header says17:23; the actual tool timestamp was17:21. This was a transcription mistake, corrected append-only. The pre-inference protocol/request and experiment timestamps are unaffected.

## 2026-10-16 17:27 - Enlarged mining review vetoes real lights mislabeled as background
**Type:** bug
**Stage:** Navigation M2 negative-label validity

**What happened:** Automatic mining gate passes118 boxes/71 views across13 TRAIN junctions/four towns in12.63s. Enlarged context review reveals real non-target light housings among these candidates, including indices9,25,78,94,98,110. They stay unchanged because only the target lamp's phase changes. The initial native-size thumbnail sheet was too small for reliable inspection; enlarged crop+context pages expose the problem.
**Cause:** Projected actor get_light_boxes inventory is not established as exhaustive visible-housing coverage; root cause of the omissions remains open. Phase invariance alone cannot prove background when other lamps are frozen.
**How we handled it:** Do not train. Preserve automatic gate/results and add training_release.json approved:false, bound to manifest hash; the reader requires an explicit semantic release before attachment. Existing models unchanged. Inspect level/environment traffic-light geometry from CARLA as a possible conservative ignore source. Historical681 approximate-background count is not verified false positives; known-positive72-label counts remain intact.
**Verified by:** outputs/navigation_m2_signal_hard_negatives_20261016/negative_context_page0..4.png, manifest and training_release.json; original raw contact sheet retained. No head-refinement training has started.
**Lesson:** Unchanged pixels outside an incomplete inventory can still depict a real object; negative supervision needs a stronger audit than positive labels.
**Interview angle:** Why can a frozen non-target traffic light pass a hard-negative phase-invariance filter?

## 2026-10-16 17:43 - Correct overconfident visual label diagnosis with registered semantics
**Type:** mistake
**Stage:** Navigation M2 label verification

**What happened:** Enlarged RGB crops were described too confidently as real traffic-light housings. Corrected same-frame/same-optics semantics show Town01 candidates9/10 are building features (tags3/6/9,zero TrafficLight7), while Town04 candidates94/95/97 include TrafficSigns8, not TrafficLight7. Both actor and level/env inventories agree36 heads in Town01 and49 in Town04; no inventory defect has been established.
**Cause:** Tiny windows, signs and poles resemble traffic lights—the very confusion this detector needs to learn. Visual suspicion was mistaken for confirmed object identity.
**How we handled it:** Explicitly correct the earlier conclusion; keep original mining manifest and veto while checking the complete TRAIN set semantically. Freeze a semantic-release supplement before complete-set capture. Semantic labels remain training-only; other known categories can be negatives specifically for the binary traffic-light detector, not for all driving rules.
**Verified by:** outputs/navigation_m2_semantic_probe_town01_retry_20261016 and navigation_m2_semantic_probe_town04_20261016, exact image frames/poses/lens attributes/raw tags; no head fine-tune has run. First semantic attempt has optics_validity.json:false and is excluded from matched-registration proof.
**Lesson:** Use simulator pixel labels to test an ambiguous visual diagnosis before declaring geometric ground truth broken.
**Interview angle:** How did you distinguish a detector false positive from a mislabeled real traffic light?

## 2026-10-16 17:43 - Diagnostic import and optics-copy mistakes corrected before training
**Type:** mistake
**Stage:** Navigation M2 infrastructure diagnosis

**What happened:** Initial level probe imported load_training_world from the wrong module and failed before creating a server; corrected to scripts.collect_navigation_signal_pairs. A metadata read used system Python without Torch and was rerun in.venv. The first semantic sensor copied str(ActorAttribute), a descriptive wrapper string; CARLA parsed those optical values as zeros.
**Cause:** Wrong helper/interpreter assumptions and treating an API attribute representation as its numeric value.
**How we handled it:** Use project interpreter and actual helper definition, copy as_float values, record numeric optics and verify spawned sensor attributes plus RGB/semantic equality. Preserve the invalid first probe; fresh retries pass. Add missing helper-source provenance to the first completed level inventory in a supplemental file, preserving its original request.
**Verified by:** Fresh matched probes complete in7.74/7.63s, all six image poses match expected matrices within1e-3 and lens values match. Original failures retained; no force kill, orphan or training on these diagnostics.
**Lesson:** Validate actual sensor attributes after spawn rather than trusting a setter call or string conversion.
**Interview angle:** Why did a successful camera-attribute setter still produce an invalid calibration control?

## 2026-10-16 17:50 - Complete registered semantics releases110 TRAIN hard negatives
**Type:** milestone
**Stage:** Navigation M2 supervised negative data

**What happened:** Capture synchronized RGB/semantic labels at all mined TRAIN poses across Town01–04; case counts{'town01': 5, 'town02': 10, 'town03': 9, 'town04': 9}. Every view records matching actual optics, exact frame and image pose. All118 candidates have no TrafficLight7/Unknown0 in a2px padded mask;8 fail the frozen fresh/source RGB MAE<=10 check. Release110 canonical boxes across13 TRAIN junctions/four towns,440 phase copies/264 images; zero validation replay targets.
**Cause:** Windows, signs and poles were detector false positives rather than established missing traffic-light geometry. The original visual veto was appropriately kept pending stronger evidence, then corrected.
**How we handled it:** Preserve original mining/initial veto and invalid first optical-control probe. A separate source/evidence-hashed release approves only110 indices; reader verifies hashes, TRAIN roles and unchanged>=20-box/6-junction/3-town gate before mutation. Freeze parent-protected2050-parameter head refinement with original recipe; no RPN/box/feature update or runtime semantic input. Parent floor30 detections/24 joint states and<=681 approximate-background candidates, minimize existing validation loss among eligible candidates.
**Verified by:** outputs/navigation_m2_signal_negative_release_20261016/release.json SHA9a842d1179fadcea11cfdf2b4c6c3ae51d0ed2b662a3c49a0b97127f66df7077, candidate_checks, all four source archives and registered pixel probes.14 focused tests pass, including numeric optics, release veto, ROI replay and semantic margin/photo filters. No head fitting yet.
**Lesson:** Separate an automated coverage gate, a provisional visual concern and a reproducible semantic training release.
**Interview angle:** How did you certify mined negatives without giving the driving policy semantic sensors?

## 2026-10-16 17:52 - Stop a premature refinement launch and fix ROI regression tuple handling
**Type:** mistake
**Stage:** Navigation M2 replay implementation

**What happened:** The ROI replay regression failed: torchvision returns regression targets as a tuple, so appending via item assignment raised TypeError. I launched the first refinement before inspecting that completed test output and prematurely wrote14 focused tests passed; actual result was13 passed/1 failed.
**Cause:** New replay code assumed a mutable regression-target list; verification and launch were incorrectly treated as independent operations.
**How we handled it:** Immediately touch outputs/navigation_m2_signal_hard_negative_fit_20261016/STOP. It exits during initial validation at optimizer step0, before any update; preserve this stopped attempt. Convert regression targets to a list, rerun lint and all14 focused tests successfully before a fresh retry. Correct the earlier test-count claim append-only. Training-label release evidence remains valid.
**Verified by:** Stopped progress/latest checkpointstep0; replay test now checks positive/negative multiplicities and zero GT regression targets. No process kill or parent/model replacement.
**Lesson:** Inspect a dependent check's actual result before launching a run; source snapshots make the stopped mistake recoverable without rewriting history.
**Interview angle:** How did you recover from launching before a newly added replay test had passed?

## 2026-10-16 18:03 - Protected negative refinement rejects every trained checkpoint
**Type:** result
**Stage:** Navigation M2 traffic-light detection

**What happened:** Fresh retry completes200 updates in97.51s with only2050 ROI classifier parameters trainable. Parent30/72 detections,24 joint states,681 approximate backgrounds; step50 gives22/18/208,step100 gives25/22/196,step200 gives27/25/171. Loss improves.118660 to.097397 but every trained candidate fails the frozen recall floor30.
**Cause:** Fewer background detections accompany lost known-light coverage, especially Town03 (6/20 to1/20 at200). Calibration versus representation/generalization remains open.
**How we handled it:** Retain selected step0, the original parent; preserve rejected latest200. No policy promotion or duplicate driving test. Frozen tensors verified by worker; current camera policy unchanged.
**Verified by:** outputs/navigation_m2_signal_hard_negative_fit_retry_20261016/review.json and per-step predictions;299 CPU tests passed/two skipped before the new calibration test. The stopped first attempt remains separate at0 updates.
**Lesson:** Lower aggregate loss and fewer background proposals do not justify losing safety-critical positives.
**Interview angle:** Why did you reject a model whose loss and background count both improved?

## 2026-10-16 18:03 - Freeze one TRAIN-derived foreground calibration audit
**Type:** decision
**Stage:** Navigation M2 calibration diagnosis

**What happened:** Compare fixed parent50 and rejected latest200 on179 TRAIN positive GT ROIs and440 released negative copies using shared frozen features. Fit only the smallest nonnegative foreground bias restoring the parent's TRAIN positive acceptance count at.5.
**Cause:** Test whether suppression of foreground confidence explains the coverage regression without searching validation thresholds.
**How we handled it:** Protocol fixed before execution; no validation ROI in scalar fitting. Zero offset reuses exact candidate200 evaluation; otherwise score one candidate once with unchanged144 images/threshold/guards.300s cap, STOP and fresh output. No further automatic calibration or driving integration.
**Verified by:** reports/navigation_m2_signal_train_calibration_protocol_20261016.md;15 focused tests now pass, lint passes; live anchored check finds no trainer/CARLA and GPU idle. Results pending.
**Lesson:** Derive calibration from training evidence and preserve a rejection when validation generalization still fails.
**Interview angle:** How can you diagnose confidence calibration without tuning against validation outcomes?

## 2026-10-16 18:04 - TRAIN calibration chooses zero; validation regression remains
**Type:** result
**Stage:** Navigation M2 calibration diagnosis

**What happened:** Shared frozen ROI audit completes in16.45s. Parent accepts117/179 TRAIN positive and360/440 released negative ROIs; candidate200 accepts120/179 positives and91/440 negatives. The predeclared minimal nonnegative bias is exactly0.
**Cause:** This TRAIN-derived offset cannot restore validation coverage: the candidate already exceeds parent TRAIN acceptance while its full validation detection is27/72 versus30/72. This does not isolate feature/box/sampling/domain causes or prove all calibration methods fail.
**How we handled it:** Reuse exact candidate200 validation without another GPU evaluation or parameter change; candidate still fails recall guard. No further automatic calibration. Retain original parent; next broaden TRAIN positive approach/view coverage before a fresh bounded joint proposal/head experiment.
**Verified by:** outputs/navigation_m2_signal_train_calibration_20261016/assessment.json; source and parent/candidate tensor equality checks, unchanged.5 threshold. No calibrated checkpoint created because weights would be identical.
**Lesson:** Training ROI acceptance is not deployment recall; proposals, localization and held-out appearance still matter.
**Interview angle:** What did a zero TRAIN-derived calibration offset actually establish?

## 2026-10-16 18:09 - Close the negative-refinement experiment with protected parent and audited evidence
**Type:** milestone
**Stage:** Navigation M2 perception diagnostic; M2 remains open

**What happened:** Summarize approved110 canonical negatives/440 phase copies, all four checkpoints and zero-bias diagnosis. Independent one-to-one audit shows25 parent positives retained,5 lost,2 gained. The original detector remains selected; no driving gain or policy promotion.
**Cause:** Background suppression alone does not solve sparse small-light coverage or rule behavior.
**How we handled it:** Add reports/navigation_m2_signal_hard_negatives_20261016.md/json and four-panel graph; read-only scripts/summarize_navigation_signal_negatives.py rebuilds them. Update CLAUDE, handoff, plan, runbook, README and prior detector report. Next broaden TRAIN positive approaches/views with a fresh protocol before bounded joint proposal/head adaptation; preserve all validation groups and explicit supervision budgets.
**Verified by:** All612 original image hashes, archived sources/semantic evidence, release membership, parent/selected full tensor equality, latest non-classifier equality, camera policy/cache/crop hashes.300 CPU tests passed,2 skipped,64 historical AMP warnings in15.78s; targeted lint/diff pass. Anchored final host check no trainer/CARLA, GPU0%,42C,35.33GiB free. No job-script/reward/benchmark changes, held-out use, multi-hour run or push.
**Lesson:** A rejected fit can improve the data engine and diagnosis without being presented as a better model.
**Interview angle:** What reproducible progress came from a training experiment that failed its protection gate?

## 2026-10-16 19:08 - Freeze TRAIN-only approach expansion with original validation preserved
**Type:** decision
**Stage:** Navigation M2 positive perception coverage

**What happened:** User continued after rejected negative-only fit. Geometry preparation freezes99 novel poses/36 incoming approaches/13 original TRAIN junctions across Town01–04 (26/30/24/19 poses);1,188 planned RGB images. Original corpus had only one approach/junction at40/20/6m. New first3 light/incoming-lane approaches in route-ID order use32/12/2m;9 duplicate/clamped poses excluded.
**Cause:** Need broader positive evidence before another fit. Reapplying the seeded split to a13-group subset would reassign validation locations; explicitly prohibit that.
**How we handled it:** New protocol/plan preserve original13/4 roles,144 images/72 validation labels, all spatial guards. Every pose stays>=1m from non-TRAIN junction geometry and is deduplicated against originals. One server,900s cap, unchanged exact-frame renderer/response-v2 labels. Gate>=45 new responsive pairs,>=10 junctions/all4 towns,>=6 multi-incoming-lane junctions. Only if passed: protected200-update joint RPN/predictor fit from original parent,50/50 original/supplement sampler, verified old negatives, fixed validation/loss/thresholds and separate RPN proposal recall. No automatic long repeat or driving integration.
**Verified by:** reports/navigation_m2_signal_expansion_protocol_20261016.md and plan SHAcdde875dd38c8e803f82d46c96c8fb7e4783f828ca4f1f01613b3efd9ab25ef5. Focused split/append/sampler tests plus lint checked before launch; results pending. TRAIN supplement requires explicit append and cannot become a standalone resplit dataset. Live anchored check no trainer/CARLA, GPU idle.
**Lesson:** More training views must not silently redefine validation or treat outgoing turns as independent visual approaches.
**Interview angle:** How do you expand a small driving-perception corpus without moving the validation split?

## 2026-10-16 19:12 - Approach expansion passes; validation payload remains identical
**Type:** milestone
**Stage:** Navigation M2 positive data gate

**What happened:** All99 novel poses/1,188 native RGB images complete in100.03s. Response-v2 audit supports116 head/view pairs (115 yellow), all13 TRAIN junctions/four towns,36 responsive incoming lanes and12 multi-lane junctions. Frozen data gate passes.
**Cause:** Distinct incoming lanes and32/12/2m add positive geometry/appearance beyond the old one-approach40/20/6m set. This is a data-coverage result, not model or driving improvement.
**How we handled it:** Append only TRAIN data; original/supplement sampling stays50/50. Total1,656 TRAIN images/642 phase-correlated positive labels,144 validation images/72 labels. Original440 verified negative copies retained, zero validation replay. Standalone supplement loading refuses a new split. Proceed with the already frozen bounded200-update joint RPN/predictor experiment, original parent and fixed guards.
**Verified by:** outputs/navigation_m2_signal_expansion_20261016/data_audit.json and attachment_audit.json; all image/archive hashes verified, before/after validation RGB/sample/target/annotation SHA9ab66584c129e8a84a676512bad8b693dd9f82d2df5de2f1a8c74a3edcb0bd2f. No new held-out location or runtime input. Source edits finished before capture, no job scripts changed.
**Lesson:** Demonstrate byte-level validation preservation when extending a supervised corpus.
**Interview angle:** What did the multi-approach data gate prove, and what did it not prove?

## 2026-10-16 19:15 - Expanded joint fit improves operational component metrics but loss retains parent
**Type:** result
**Stage:** Navigation M2 proposals and housing classification

**What happened:** Protected joint RPN/predictor fit completes200 updates in109.48s;1,194,265 trainable parameters. Initial parent30 detected/24 joint/681 approximate backgrounds; step50 gives27/23/291,100 gives30/28/256,200 gives36/26/209. RPN proposal recall58/72→69/72 atIoU.3,38→51 atIoU.5. Latest under8px15/40 versus13/40; per-town4/8,6/20,12/12,14/32 versus4/8,6/20,12/12,8/32.
**Cause:** Expanded positive views plus joint proposal/head adaptation improve these sparse metrics; this is a combined intervention, not an isolated data ablation. Fixed validation loss.126094 remains above parent.118660.
**How we handled it:** Follow frozen selection: keep parent0. Latest200 passes recall/state/background protection but is unselected by loss, unlike the earlier head-only candidate which failed recall. Preserve both; no runtime integration, promotion, threshold change or longer repeat. Next do a bounded fixed parent/latest diagnostic decomposing detector loss and GT-ROI acceptance, with no fitting or selection change.
**Verified by:** outputs/navigation_m2_signal_joint_expansion_20261016/review.json and validation/prediction records. Frozen state unchanged;305 CPU tests pass,2 skipped,64 historical AMP warnings in16.59s. Original validation payload remains identical.
**Lesson:** Report operational gains even when the predeclared surrogate selection keeps the old model; never silently change the rule afterwards.
**Interview angle:** What do you do when a new checkpoint improves detection metrics but loses on the frozen selection loss?

## 2026-10-16 19:17 - Freeze descriptive parent/latest loss decomposition
**Type:** decision
**Stage:** Navigation M2 selection diagnosis

**What happened:** Operational sparse metrics improve at joint200 while frozen loss retains parent0. Compare exactly these two checkpoints once: four FP32 detector-loss components with original seed/batch/validation, plus binary foreground acceptance on72 shared frozen GT ROI features at.5.
**Cause:** Need to explain disagreement before another experiment; do not substitute an outcome-favorable selection criterion post hoc.
**How we handled it:** New source/protocol frozen before inference;300s cap, STOP, GPU lock. Reproduce saved totals within1e-6; no optimizer, calibration, threshold search, RGB detection repeat, alternate candidate or selection change. GT ROI quantities explicitly privileged diagnostic only.
**Verified by:** reports/navigation_m2_signal_loss_audit_protocol_20261016.md; scripts/assess_navigation_signal_losses.py lint passes. Core changes already305 CPU tests pass/two skipped; diagnostic results pending.
**Lesson:** Use an immutable descriptive audit to distinguish a surrogate-loss disagreement from a reporting error.
**Interview angle:** How do you investigate model-selection disagreements without moving the goalposts?

## 2026-10-16 19:20 - Loss decomposition reproduces totals; freeze sampling-normalization check
**Type:** result
**Stage:** Navigation M2 surrogate-loss diagnosis

**What happened:** Fixed parent/latest audit finishes21.49s and reproduces both saved losses exactly. Classifier.066034→.063204,ROI box.043388→.054930,RPN objectness.008110→.006782,RPN regression.001128→.001179. Privileged exact-GT ROI acceptance23→29/72 at.5; Town03 stays3/20. These are not detection/precision or driving scores.
**Cause:** ROI box term numerically drives higher total loss. Whether higher per-positive residual or changed positive ROI sampling explains it is still open. GT ROI scores suggest head/feature/context generalization remains weak, without isolating one causal mechanism.
**How we handled it:** Keep parent selected and preserve candidate metrics. Freeze one supplementary descriptive loss pass with ROI sampler counters: record all/positive ROI counts and normalize reconstructed SmoothL1 mass by positives. Same fixed two checkpoints/seed/batches and1e-6 saved-loss parity; no return-tensor/RNG/selection change. Reuse prior GT-ROI results rather than repeating those features.300s cap, fresh output/STOP.
**Verified by:** outputs/navigation_m2_signal_joint_loss_audit_20261016/assessment.json and GT rows; sampling supplement protocol and lint pass before launch. No optimizer or driving integration.
**Lesson:** A regression loss divided by all proposals can change when the proposal sampler changes; count positives before interpreting it as box-quality decline.
**Interview angle:** Why can better proposals produce higher box-regression loss?

## 2026-10-16 19:22 - ROI sampling explains the box-loss increase without changing selection
**Type:** result
**Stage:** Navigation M2 loss normalization

**What happened:** Counter-only audit completes19.76s and reproduces both saved losses exactly. Total sampled ROIs9,216 for each model; positive ROIs171→222. Mean SmoothL1 per positive decreases2.33836→2.28034 while box loss normalized over all ROIs increases.043388→.054930. Prior GT-ROI results reused unchanged.
**Cause:** More positive sampled proposals increase the summed box term under the fixed all-ROI normalization. Different sampled populations still prevent treating per-positive residual as a controlled physical localization comparison.
**How we handled it:** Keep original loss-based parent selection, preserve joint200 as an unselected development candidate with improved36/72 detection and209 approximate backgrounds. No post-hoc normalization-based promotion. Next work should target ROI feature/context generalization and prospectively specify operational selection before a fresh fit, rather than repeat longer to force the old loss down.
**Verified by:** outputs/navigation_m2_signal_joint_sampling_audit_20261016/assessment.json and source/protocol hashes. No returned sampler tensor or RNG change, no GT feature/detection repeat, no optimizer or policy modification.
**Lesson:** Selection losses can depend on how many positives the model proposes; audit their denominator before interpreting model quality.
**Interview angle:** What concrete evidence explained the loss/recall disagreement?

## 2026-10-16 19:26 - Close approach expansion with improved unselected candidate and denominator audit
**Type:** milestone
**Stage:** Navigation M2 component engineering; driving gate remains open

**What happened:** Complete99-pose/1,188-image TRAIN expansion,200-update/109.48s joint fit and two bounded descriptive audits. Latest final36/72 versus30/72, joint26 versus24, backgrounds209 versus681, proposals69 versus58. One-to-one pairing retains26/loss4/gain10. Candidate passes protection but frozen loss selects unchanged parent0.
**Cause:** More useful proposals plus combined positive/head adaptation help sparse development metrics; ROI classification/context remains weak. All-ROI-normalized regression loss also changes with positive sampling171→222, despite per-positive residual2.338→2.280. Neither finding is a driving or matched JEPA result.
**How we handled it:** Add reports/navigation_m2_signal_expansion_20261016.md/json and four-panel graph; read-only reproducible summary verifies original/supplement data and validation payload, all archived sources, selected/frozen tensor equality and unchanged camera-policy/cache/crop hashes. Update handoff/CLAUDE/README/plan/runbook and preserve all prior experiments. Next ROI context/scale and limited feature adaptation with prospective selection, then tracking/applicability/complete stop-release memory; no blind longer fit or retroactive selection.
**Verified by:** scripts/summarize_navigation_signal_expansion.py verifies1,800 images and SHA9ab66584c129e8a84a676512bad8b693dd9f82d2df5de2f1a8c74a3edcb0bd2f validation identity.305 CPU tests passed/two skipped,64 historical AMP warnings in16.59s; targeted lint/diff pass. Final anchored host check no trainer/CARLA, GPU0%/39C,34.75GiB free. Candidate SHAa44f65acd4e3d2941c7f36e6dcc7e0f5d6a3853e3be6a7d1fbd691ed0370f1f9. No job-script/reward/benchmark changes, held-out use, policy promotion, multi-hour run or push.
**Lesson:** Keep component gains, the frozen selection decision and model-selection diagnostics as separate traceable facts.
**Interview angle:** What reproducible outcome did the approach-expansion continuation achieve?

## 2026-10-16 19:38 - Freeze cached ROI context adapter and prospective component selection
**Type:** decision
**Stage:** Navigation M2 classifier/context generalization

**What happened:** Continue after joint fit's operational gain and sampler-loss audit. Use completed UNSELECTED joint200 (SHAa44f65acd4e3d2941c7f36e6dcc7e0f5d6a3853e3be6a7d1fbd691ed0370f1f9) as an explicit development initialization, not a retroactive selection or driver promotion.
**Cause:** Fixed GT-ROI acceptance29/72 remains weak despite69/72 RPN coverage; more proposal training alone does not address classifier/context limitations.
**How we handled it:** Freeze RPN/visual/ROI features/box regression/old classifier. Add zero-initialized65,665-parameter bounded residual foreground MLP. Cache four TRAIN variants per642 positive and440 verified negative copies; positives expand<=25% each side, negatives only shrink inside certified boxes. First16 proposals/TRAIN image get soft old-logit consistency, never asserted background labels.1000 cached updates/900s total cap, FP32 AdamW1e-3, CE+KL1.0; fixed.5 threshold/classifier. New prospective component rule requires strict detection/joint improvement over36/26, background<=209, every town and under8px floors; rank joint,recall,-background. Prior runs keep their original selections; no driver integration or threshold tuning.
**Verified by:** reports/navigation_m2_signal_roi_adapter_protocol_20261016.md,43 focused tests and lint/diff pass. Zero adapter returns identical logits/boxes; only adapter trainable; unsafe negative expansion and validation caching rejected. Original144-image/72-label payload hash protected. Host idle before work. Results pending.
**Lesson:** Fit a small nonlinear correction while preserving proposal/localization behavior, and declare operational selection before the new fit.
**Interview angle:** How can you adapt an ROI classifier cheaply without changing the detector's proposal or box-regression behavior?

## 2026-10-16 19:42 - ROI residual selects500 under the prospective guard; latest1000 regresses
**Type:** milestone
**Stage:** Navigation M2 perception component; not driving gate

**What happened:** Completed1,000 cached updates in147.61s,65,665 parameters,1.94GB peak. Cache30,824 TRAIN features:2,568 positive/1,760 verified-negative variants plus26,496 unlabelled soft-target proposals. Zero correction exactly reproduces base logits/box outputs. Selected500 gives44/72 detection,36/.5,29 joint,160 approximate backgrounds versus initialization36/25/26/209. Under8px15→20/40. Town counts4/8,6/20,12/12,22/32; all floors preserved.
**Cause:** Bounded nonlinear classification plus safe context and consistency improves these sparse development metrics with base proposal/feature/localization weights fixed; not an isolated architecture/data ablation.
**How we handled it:** Prospective selector keeps500 (250 has same44/29 but176 backgrounds). Latest1000 gives40/26/156 and Town03 falls6→5/20, so reject. Do not resume latest or auto-integrate selected component. Only original selected detector remains retained for historical results; driver unchanged. Next one separate, fixed-recipe expanded-TRAIN state-classifier experiment on unchanged detector boxes, not further adapter tuning.
**Verified by:** outputs/navigation_m2_signal_roi_adapter_20261016/review.json; selected SHA2dbd3e56c7de29c5f1b79c284f7fe0f704ec659e43b0a4885583ca2f108ee507. All frozen tensors/normalization unchanged.310 CPU tests pass,2 skipped,64 historical AMP warnings in17.19s;43 focused checks passed before fitting. Original validation payload hash unchanged.
**Lesson:** A tiny nonlinear residual can improve a frozen detector; protected selection still matters because longer fitting regresses.
**Interview angle:** What improved in the ROI adapter experiment and why was latest rejected?

## 2026-10-16 19:46 - Freeze one expanded-TRAIN state fit on selected fixed boxes
**Type:** decision
**Stage:** Navigation M2 state alignment

**What happened:** ROI adapter500 passes with44 detections but only29 correct states. Prepare one separate state-classifier fit using existing642 TRAIN GT phase crops, original+15 outward augmentations, same16x32 RGB/C1 balanced L2 logistic model as the prior robust classifier.
**Cause:** Fifteen matched predicted ROIs remain state-wrong; broader TRAIN appearance/context may improve the fixed state head without another detector/adapter fit.
**How we handled it:** Original/supplement each total321 source weight,16 variants per label preserve total642. ONE CPU fit,300s cap/STOP, no threshold/hyperparameter search. Fixed44 boxes/scores and original30-box compatibility guard: require joint>29 and original-state correctness>=24. Oracle states reported separately, never selection. Full578 predicted ROI states inferred from RGB without labels, then independent one-to-one scoring. Preserve old artifact; local bundle may reference accepted components, no driver change.
**Verified by:** reports/navigation_m2_signal_state_expansion_protocol_20261016.md and script lint. Detector SHA2dbd3e... fixed; exported sklearn parity and box/score identity required. Previous310 CPU tests pass/two skipped; source code preserves frozen validation signature. Results pending.
**Lesson:** Improve state classification on fixed predictions so localization changes cannot explain the gain.
**Interview angle:** How do you isolate a traffic-light state-head improvement from better detection?

## 2026-10-16 19:48 - Expanded state fit passes fixed-ROI guard; oracle diagnostic regresses
**Type:** result
**Stage:** Navigation M2 state alignment

**What happened:** One C1 balanced RGB logistic fit uses642 TRAIN labels/10,272 correlated crops, total weight642 split321/321 original/supplement. Fit0.880s,10.15s total,50 iterations. On unchanged578 predictions/44 detected labels, joint correct states29→31/72. Old detector fixed30-box compatibility24→27. Oracle exact-crop diagnostic68→66/72.
**Cause:** Expanded TRAIN appearance/context improves these actual predicted-ROI state decisions, while exact annotated-crop performance declines; not a universal perception or driving gain.
**How we handled it:** Accept by predeclared predicted-ROI/compatibility guard, not oracle score. Preserve old classifier and all source detections. New perception_bundle.json references selected ROI adapter500 plus expanded state coefficients; original camera policy unchanged, no runtime integration. No additional fit or tuning.
**Verified by:** outputs/navigation_m2_signal_state_expansion_20261016/assessment.json; coefficient SHAf4171de1e652e1f53b64089145e38dacd4a353199a4a097c3712d6c2259c17c4. Export/sklearn parity and all box/score identities exact; independent one-to-one44 detections/31 correct state. Source archives and original validation signature checked.
**Lesson:** Improvements measured on actual predicted ROIs can differ from privileged exact crops; report both and preserve the prospective selection rule.
**Interview angle:** Why accept a state head whose exact-crop diagnostic got worse?

## 2026-10-16 19:58 - Close ROI/state milestone with selected bundle and remaining safety gaps
**Type:** milestone
**Stage:** Navigation M2 perception components; driving gate open

**What happened:** Select ROI residual500 plus expanded RGB state classifier:44/72 detected,36/.5,31 joint correct,160 approximate backgrounds. Initialization36/25/26/209; original selected detector30/22/24/681 remains its historical artifact. All36 initialization detections retained,8 gained, all at Town10HD component junction; other towns unchanged. Latest1000 rejected.
**Cause:** Small nonlinear classification/context adaptation and broader state supervision improve these sparse development components. Red does not improve:24/36 matched,15 red/9 yellow,12 misses; overall28 known labels missed and13 matched states wrong. No learned driving, rule-understanding or matched JEPA result.
**How we handled it:** Add reports/navigation_m2_signal_roi_adapter_20261016.md/json, graphs/native examples and reproducible CPU summary. Local perception_bundle.json binds selected detector2dbd3e... and statef4171d.... Update CLAUDE/handoff/README/plan/runbook; progress explicitly separates selected/latest and combined-state scores. Next existing-TRAIN temporal tracking, ego-route applicability and full stop/release observability before another policy fit/pilot, keeping validation errors out of training.
**Verified by:** Reproducer verifies all1,800 native images/30,824 TRAIN cache rows, source/evidence hashes, validation payload9ab66584..., frozen base/normalization and unchanged original driver/cache/detector/state hashes.310 CPU tests pass,2 skipped,64 historical AMP warnings in17.19s; targeted lint/diff and real progress snapshots pass. Final anchored host check no trainer/CARLA, GPU0%/42C,34.37GiB free. No job-script/reward/benchmark changes, held-out use, multi-hour run, driver promotion or push.
**Lesson:** Preserve selected and rejected checkpoints, inspect class-specific misses and separate a stronger perception bundle from a working driver.
**Interview angle:** What is improved now and what prevents claiming a safe long-drive model?

## 2026-10-16 20:15 - Freeze split-safe causal signal replay before another camera-policy fit
**Type:** decision
**Stage:** Navigation M2 temporal perception

**What happened:** Implement frozen RGB-only selected-bundle inference and causal per-camera bearing/state tracking, plus a fresh bounded offline replay. Independent current component split excludes old correction routes111/133 at Town03/655 (491 timestamps); retain routes86/101/123/143 (1,238). Only complete signal-bearing routes101/123 are inferred:518 timestamps/1,554 native images, chosen before model output.
**Cause:** Static44/72 detection/31 joint-state accuracy does not establish temporal evidence, head identity or applicability. Existing14.1s light wait begins at the first recorded sample: left-censored, so it cannot establish full approach/onset observability.
**How we handled it:** Freeze reports/navigation_m2_signal_temporal_protocol_20261016.md before inference. Fixed geometric/color-blind association, .3s probability smoothing, two consecutive observations/.6 confidence, stale>.2s Unknown, expiry>.5s. Applicability always unresolved; no control, depth, stop line or cross-camera identity claim. Independent scoring uses GT projected boxes only after RGB inference, preserves unverified visibility/phase labels and reports raw/tracked identical-match denominators. STOP/fresh output/600s cap, no fitting or simulator. First preflight caught a mistyped expected corpus hash before output/inference; corrected to independently hashed original. System Python lacked Torch during an initial metadata read; switched to project .venv.
**Verified by:** Full318 tests pass/two skipped/64 historical AMP warnings in18.43s;146 navigation tests pass/one skipped; targeted Ruff and diff checks pass. Preflight verifies original source archives, committed shard/media/label manifests, split guards and contiguous episodes. Host idle40C/GPU0%,~35GB free, no trainer/CARLA before launch. Replay result pending.
**Lesson:** Exclude newly reserved component locations from old TRAIN data and distinguish a recorded wait/release from a complete observed approach.
**Interview angle:** Why does a static signal detector need a causal replay and a second split guard?

## 2026-10-16 20:18 - Fixed causal replay completes; sparse coverage remains the main limitation
**Type:** result
**Stage:** Navigation M2 temporal perception

**What happened:** One fixed518-timestamp TRAIN replay completes118.99s. Of440 known-state timestamps,213 have an IoU>=.3 projected applicable-housing match.769/1,534 projected housing/view/time targets matched. Same-match raw correct715/769, smoothed argmax721/769; confident tracked660 correct/690 emitted,79 Unknown retained in denominator. Raw6 red-called-green on610 matched red targets; confident tracked0, but smoothed argmax still1. These are scoring-associated, visibility-uncertified component labels, not driving/safety evidence.
**Cause:** Familiar release route123 has179/179 known timestamps with a match; following route101 only34/261, longest gap10.9s. Smoothing cannot supply unseen evidence or select the applicable lamp. There are133 assigned-track changes/734 consecutive matched pairs; duplicate proposal ambiguity vs physical identity still needs diagnosis.
**How we handled it:** Preserve all outputs/parameters and do not promote tracker. Red→green on123: raw0s, smoothed argmax.1s, confident.2s; red transition on101: raw1.2s/confident2s, not a driving response.14.1s red wait is left-censored. Mean197.18ms/p95206.61ms per RGB triplet+tracking, excluding decode/scoring/IO;10Hz not achieved. No fit, simulator or policy change. Continue CPU evidence reproduction/duplicate-identity analysis, then record a concrete next experiment.
**Verified by:** outputs/navigation_m2_signal_temporal_20261016/assessment.json, request/source archive,518 prediction records and visually inspected timeline. Full318 tests passed before launch. Final source/artifact verification and report pending.
**Lesson:** A high state score conditional on detected housing can coexist with poor timestamp coverage; smoothing has latency and cannot solve applicability.
**Interview angle:** Why does93% correct state on detected targets not mean the car can obey signals?

## 2026-10-16 20:25 - Close temporal replay with the missing-approach diagnosis
**Type:** milestone
**Stage:** Navigation M2 temporal perception; driver gate open

**What happened:** Added reports/navigation_m2_signal_temporal_20261016.md/json and timeline, with reproducible CPU summary. Post-hoc decomposition of133 scoring-associated track changes:87 older IDs still overlap the same GT housing atIoU>=.3,21 active elsewhere,25 absent. Thus duplicate hypotheses affect association; this is not a calibrated physical-ID switch rate. No tracker tuning/reselection was performed.
**Cause:** Existing two signal-bearing sequences lack a complete approach/stop/release. Route101's planner/API upcoming-light distance goes113.1→37.2m; largest native housing2.0–4.1px (median2.7), stops are vehicle-caused. Route123 begins4.6m before API reference already in14.1s red-stop intent, native housing8.1–13.8px; wait/release covered179/179 known timestamps but onset is left-censored. Poor far coverage cannot automatically be blamed on recurrence or treated as a driving failure; facing/visibility remain unverified.
**How we handled it:** Keep selected perception components and driver/cache unchanged; no promotion. Next prepare split-safe TRAIN full moving approach/braking/wait/release collection, with >=3s post-release and censoring/actual coverage audits; verify facing/visibility/ego-route applicability before policy fitting. Keep stop-sign/crossing/vehicle families distinct; preserve collisions and interventions. Separate duplicate-head arbitration and causal-memory hypotheses;197ms triplet inference still misses10Hz. No long run scheduled.
**Verified by:** scripts/summarize_navigation_signal_temporal.py verifies all replay archived sources/input hashes, selected artifacts,518 unique predictions, exact per-route counts/confusions/transitions, ID decomposition sum and unchanged original driver/cache hashes. Timeline visually inspected.318 tests passed/two skipped/64 historical AMP warnings before replay; all changed-code Ruff/format checks and git diff --check pass; completed progress1554/1554 images. Updated CLAUDE/handoff/README/plan/runbook and matching Q&A. Final anchored host check no trainer/CARLA, GPU0%/44C,34.36GiB free. No benchmark/reward/jobs edits, held-out use, multi-hour run or push.
**Lesson:** A wait/release sequence plus a separate distant-light episode does not demonstrate continuous approach observability; fill the actual data gap instead of lengthening training blindly.
**Interview angle:** What concrete evidence justifies the next collection instead of another model fit?

## 2026-10-18 14:12 - Authorize execution through M6 and freeze continuous-approach pilot
**Type:** decision
**Stage:** Navigation M2; M0–M6 execution goal active

**What happened:** User explicitly requested continued execution until all M6 goals. Treat this as authorization for necessary runs within the approved plan, retaining milestone/evaluation gates, original reward/benchmarks, STOP-only lifecycle and all historical data. Active goal tracks M2 driving, M3 rule tests, M4 fair JEPA study, M5 planner comparison and M6 frozen Town05 evaluation; none is claimed complete prematurely.
**Cause:** Existing replay lacks a continuous approach/braking/wait/release. More blind policy training is not justified by static state scores.
**How we handled it:** Add strict expert-collection guard reserving both existing driving-internal and static component-validation junctions; recompute geometric clearance, enforce actual20Hz packet location and record guard SHA/light identity. Fresh pilot freezes original routes33/Town01(seed202610180),134/Town03(seed202610280), original signal-hold settings/cameras;900s total/300s each. New coverage gate requires >=40m first API distance, >=1s go motion, non-censored onset, actual stationary wait, same-actor green release and >=3s afterward. This is TRAIN eligibility, not changed driving success. Source hashes/plan archived before outcomes.
**Verified by:**30 focused tests pass; full suite pending. Offline plan/clearance succeeds, outputs/navigation_m2_approach_pilot_20261018. Additional spatial reservations leave eligible frozen signal_stop approaches only in Town01 and Town03; Town02/Town04/Town10HD absence is explicitly recorded, not repaired by taking validation data. No simulator launched yet.
**Lesson:** An end-to-end authorization permits sustained execution, but cannot waive evidence, partition boundaries or honest failure reporting.
**Interview angle:** How are continuous training encounters collected without changing evaluation or leaking validation geography?

## 2026-10-18 14:16 - Pilot rejects queued stops; preserve failed approach gate and isolate no-lead training
**Type:** result
**Stage:** Navigation M2 continuous-approach collection

**What happened:** Both pilot expert routes complete rule-compliantly,0 collision/red/stop violations, each1 minimum-speed infraction.455 exact10Hz samples/15 shards/181.38MB pass integrity, but0/2 pass continuous direct-signal coverage. Both stop25–27m before planner/API reference and have no sustained stationary teacher light-hazard flag.
**Cause:** Recorded world snapshots show two stopped lead vehicles in each ego lane: nearest6.9m and8.0m at6.45s. The initial teacher stops are queued behind traffic, not verified direct lamp-controlled stops. Town01's first API distance37.6m also misses the frozen >=40m data gate. The upstream hazard flags are not exhaustive stop-cause labels; do not relabel these as direct-signal training.
**How we handled it:** Preserve pilot/audit and reject its approach supplement. Next prospectively isolate no-lead TRAIN collection with an explicitly logged empty-background scene; prevent background spawning rather than remove actors. Keep original routes, scene/scoring for every evaluation and all failed data unchanged. This curriculum factor is training-only, requires strict collection guard and cannot be enabled for student/benchmark evaluation. Town01 may remain ineligible; do not lower its gate or re-admit old outcomes.
**Verified by:** outputs/navigation_m2_approach_pilot_20261018/data_audit.json, per-episode summary/setup and exact-frame world actors;322 tests passed before pilot. Expansion is not launched on failed coverage.
**Lesson:** A red light, a stopped ego and a successful route do not prove that the policy saw the cause of stopping; isolate and label queued versus direct-signal cases.
**Interview angle:** Why reject apparently successful expert demonstrations?

## 2026-10-18 14:20 - Freeze empty-background TRAIN isolation and prepare late final-suite registration
**Type:** decision
**Stage:** Navigation M2 direct-signal curriculum; M6 registration preparation

**What happened:** New no-lead pilot freezes original Town03 route134/seed202610180 with empty-background TRAIN scene. Background spawning is prevented; no actor is deleted. Runner/harness reject this scene for student or unguarded collection, record scope/metadata and audit every world snapshot for unexpected vehicles. Original evaluation traffic/scoring remain canonical. Add teacher speed-reduction actor metadata for attribution; do not infer cause solely from hazard flags.
**Cause:** The canonical-background pilot contained stopped lead cars; its0/2 direct-approach eligibility stays rejected. Town01's37.6m API start also stays below40m rather than lowering that gate.
**How we handled it:** Fresh protocol reports/navigation_m2_empty_approach_protocol_20261018.md, fresh prepared sources/guard in outputs/navigation_m2_empty_approach_pilot_20261018, same STOP/caps/geography and continuous gate.324 tests pass/two skipped/64 historical AMP warnings, including background canonical delegation/TRAIN-only controls, before launch. Separately write an outcome-blind Town05 geometry registration script/protocol; it is not launched yet. Disclose missed original pre-student registration and freeze before any further fit/final evaluation; no Town05 training or model outcome use.
**Verified by:** Source/geometry preflight succeeds and324 tests pass in21.16s. No-lead collection result and final-registration script checks pending.
**Lesson:** Use a declared training curriculum to isolate a confound while keeping evaluation intact, and fix missed registration timing transparently before more model development.
**Interview angle:** How can simplified training scenes be fair without simplifying the benchmark?

## 2026-10-18 14:21 - No-lead pilot passes complete approach coverage; begin geometry-only final registration
**Type:** milestone
**Stage:** Navigation M2 data coverage, not driving completion

**What happened:** One guarded no-lead Town03 route134 yields174 exact10Hz samples/6 shards/62.00MB. Coverage passes: first API reference49.3m,48 moving pre-stop samples, observed onset,7.0s stationary wait, same-light green release,4.8s post-release and resumed motion. Every world snapshot has only ego. Expert finishes with0 collisions/red/stop/min-speed infractions; not a camera result or comparison against canonical traffic.
**Cause:** Removing the declared TRAIN lead-traffic confound exposes the direct-signal approach/stop/release absent from the original queued pilot. No data gate was relaxed.
**How we handled it:** Preserve both pilots and their different curricula, no promotion. Next broaden TRAIN collection; first finish the overdue outcome-blind Town05 registration before further student fitting. Dedicated inventory command loads only static rule applicability, never a policy, never training RGB/actions. Protocol declares12 routes/three seeds,5 stop locations,>=30 planned stop and red opportunities, unchanged scoring/canonical traffic and missed registration timing. Candidate tests remain deferred until M2–M5 evidence.
**Verified by:** outputs/navigation_m2_empty_approach_pilot_20261018/data_audit.json plus exact-frame/actuation/RGB checks and full324 pre-run tests. New registration opportunity test passes; source/lint checks pass. Geometry inventory and route selection pending.
**Lesson:** Controlled TRAIN curricula can fill an observability gap, but only camera rollouts under declared evaluation traffic establish driving performance.
**Interview angle:** What evidence shows the missing training sequence is now available?

## 2026-10-18 14:25 - Correct explicit-signal versus runtime-stop count before final route selection
**Type:** bug
**Stage:** Navigation M6 geometry registration, no policy test

**What happened:** Town05 inventory finds54 traffic lights and33 runtime stop actors, not the earlier assumed five total stops. Pinned applicability yields242 valid movements,31 distinct stop actors in those movements. The five-count came from XML signals only; CARLA includes additional object-derived stop landmarks. No candidate was loaded and no final route/result existed.
**Cause:** Counting only <signals>/<signal type=206> omitted runtime signs encoded otherwise. Our initial registration script's len(runtime_stops)==5 condition incorrectly equated those populations.
**How we handled it:** Keep raw inventory/source unchanged. Before final route generation, uniquely join each of the five explicit mapped landmarks to a physical runtime stop within1m; require those same approved five locations twice across twelve routes. Preserve all33 actors and count every extra stop infraction. Clarify protocol and correct only the invalid runtime-total assumption; no actor/criterion exclusion or changed five-location coverage requirement. Disclose late registration timing and freeze before further fitting.
**Verified by:** outputs/navigation_final_geometry_20261018/assessment.json and offline landmark positions, unique nearest physical matches. Source/lint and directional opportunity test pass. Final route selection pending.
**Lesson:** Map XML signal counts are not runtime sign populations; verify physical applicability and preserve all scoring actors.
**Interview angle:** Why can a simulator have33 stop actors when an XML scan reports five?

## 2026-10-18 14:28 - Correct pilot-versus-full-plan town coverage and inspect launch rejection
**Type:** mistake
**Stage:** Navigation M2 expansion execution

**What happened:** The full prepared expansion correctly selects24 episodes: two signal routes perTown01/Town02/Town03/Town04 and three seeds, leaving onlyTown10HD without eligible signal routes. Earlier14:12/CLAUDE wording wrongly inferred thatTown02/Town04 had no eligible geometry from their absence in the deliberately two-town pilot.
**Cause:** The pilot's no_eligible_signal_towns field is computed after the pilot-town filter; it means absent from that pilot, not unavailable in the full eligible pool. I interpreted it too broadly despite the independent candidate list.
**How we handled it:** Correct the scope in current documentation and preserve all old plans/results. Full expansion plan has the correct four-town source schedule; no validation location was included. Automatic approval review rejected its launch because the offline Town05 geometry-freeze process remained active and might compete for CARLA. No expansion process started. Check host processes/RPC port and prove the freeze is offline before any safe retry; do not bypass rejection or duplicate jobs.
**Verified by:** outputs/navigation_m2_empty_approaches_20261018/plan.json lists routes33,34,77,78,134,135,223,225 and24 frozen seeds; no progress file after rejected launch. No training/model results affected. Host checks pending.
**Lesson:** Absence from a selected pilot is not absence from the eligible pool; inspect the complete schedule and verify resource ownership after an approval rejection.
**Interview angle:** How did a pilot metadata interpretation error get caught before influencing the dataset?

## 2026-10-18 14:31 - Preserve unfrozen endpoint-only attempt and add geometry coverage anchors
**Type:** decision
**Stage:** Navigation M6 registration, before policy outcomes

**What happened:** Endpoint-only selection produced two partial routes but thousands of sampled1–3km paths missed mandatory stop88. Stopped that OFFLINE selection with configs/routes/town05_final_v1/STOP; no suite.json exists and no driver/model was loaded. Short native route-planner checks confirm all five explicit sign locations can be traversed, including stop89 with27m local approach.
**Cause:** Random shortest-path endpoints rarely constrain the required lane/movement. The standalone short-dev approach>=30m validity filter is not a reason to drop a final-route sign.
**How we handled it:** New prospective v2 route generator picks a fixed planner-verified movement through each required stop, then interpolates seeded start→movement start/end→destination, with original1–3km/full-coverage/red-setup gates. Same five physical locations,12 routes/three seeds,all33 runtime stops/criteria and every failure retained. Archive partialv1; no test-result-dependent route replacement. Source/protocol/tests frozen before v2 selection; M2 curriculum expansion continues separately on the sole simulator.
**Verified by:** All five explicit signals match unique runtime poses; three focused tests verify directional/end-censored opportunities, all serialized segments and extra-runtime-stop preservation. Final v2 source/lint checks pass; selection pending. Automatic review accepted the collection retry after proving the original freeze was CPU-only, no RPC listener or other simulator/collector existed.
**Lesson:** Enforce planned route coverage through geometric mission constraints before testing models, rather than hoping random endpoints cover rare controls or silently deleting requirements.
**Interview angle:** How can route-generation changes preserve benchmark integrity before it is frozen?

## 2026-10-18 14:35 - Freeze and independently verify12-route Town05 final suite
**Type:** milestone
**Stage:** Navigation M6 registration only; M2–M6 driving/research still open

**What happened:** configs/routes/town05_final_v2 freezes12 routes of1005.5–2453.9m, seeds3031/4031/5031,84 planned stop opportunities and36 planned held-red opportunities. All five explicit mapped stop locations are mandatory twice; all33 runtime stops/criteria remain. No driver/model was loaded or evaluated. V1 remains stopped/unfrozen with partial geometry.
**Cause:** Finish the missed preregistration before further model fitting, with full disclosure that existing camera/component development already occurred.
**How we handled it:** Freeze suite/XML/map/source/protocol hashes and immutable geometry constraints, preserve late-registration deviation and every native actor. Independent script calls the actual pinned interpolate_trajectory and exact scenario-entry lane walk, verifies every required stop and signal binding. Candidate checkpoint registration/final driving await M2–M5 and new heldout_dev reference. Do not tune on Town05 or treat planned as actual opportunity counts.
**Verified by:** scripts/audit_navigation_final_registration.py passes all12 native serialized lengths/stop passages/held-signal bindings; reports/navigation_m6_registration_20261018.md/json. Three focused registration tests pass; no benchmark infractions excluded.24-episode four-town TRAIN collection continues on the sole simulator, sources unchanged during collection.
**Lesson:** Registration geometry is a prerequisite for a final evaluation, not the final evaluation itself; preserve that distinction and timing deviations.
**Interview angle:** What is frozen and verified before a held-out driving comparison?

## 2026-10-18 14:39 - Four-town curriculum expansion completes; keep12/24 eligible and guard timing shortcuts
**Type:** result
**Stage:** Navigation M2 continuous TRAIN data

**What happened:**24 fixed-source expert curriculum episodes complete with no infrastructure errors,4,185 synchronized timestamps/144 shards/1.731GB.12 episodes at four locations inTown01/Town03/Town04 pass direct-approach coverage, totaling2,070 timestamps; all others remain excluded from this supplement. Town02 route77 fails stationary-during-wait,78 is additionally39.5m from API reference;Town01/33 andTown03/135 miss >=40m. No gate is lowered. These are expert TRAIN results, not camera driving.
**Cause:** Controlled no-lead scenes expose valid direct approaches at four selected locations. However their wait is always about7s:3s stationary hold+4s phase clearance. Three seeds without background traffic do not create independent scene diversity and permit a recurrent timing shortcut.
**How we handled it:** Preserve all data/gates and do not claim12 independent encounters. Before fitting the causal countermeasure, prospectively vary TRAIN stationary holds3/7/11s while keeping all evaluation scenarios unchanged. Retain fixed-wait data as a diagnostic/control. Also audit20Hz trace positions: CollectionExpert.packet currently enforces location at10Hz capture, so earlier wording claiming every20Hz enforcement was too broad. Fix future collection and verify all existing20Hz traces before accepting the supplement.
**Verified by:** outputs/navigation_m2_empty_approaches_20261018/data_audit.json/outcomes.json;24/24 terminal progress. Four accepted source locations are34/134/223/225. No new camera fit or promotion occurred. Source snapshots/raw failed episodes preserved.
**Lesson:** Synchronized complete episodes are necessary but not sufficient; prevent a fixed-timer shortcut and distinguish repeated seeds from independent locations.
**Interview angle:** Why randomize the training wait duration before testing traffic-light understanding?

## 2026-10-18 14:43 - Verify all8,364 historical20Hz positions and freeze varied-wait pilot
**Type:** milestone
**Stage:** Navigation M2 data integrity and timing curriculum

**What happened:** Independent offline audit checks every recorded20Hz position across the completed24-episode curriculum:8,364 control ticks, all within strict TRAIN geography. It also records first actual moving/braking timestamps separately from semantic zero-speed-intent onset. Old metadata/eligibility is not rewritten.
**Cause:** The old collector called its guard at10Hz capture. New source checks every20Hz call after superclass hero initialization and before control mixing, while preserving the current-frame boundary. Earlier20Hz enforcement wording is corrected without assuming intervening positions were safe.
**How we handled it:** Future TRAIN holds may be explicitly3/7/11s, guarded expert/no-lead only; all evaluation options remain unchanged and invalid duration/student/unguarded use fail. A fresh Town03/134 pilot sets7s stationary hold (expected~11s actual wait); preserve original XML and all fixed-wait controls. New complete-episode memory protocol is prospective, no fit launched.
**Verified by:** reports/navigation_m2_empty_approach_positions_20261018.json (24 episodes/8364 ticks);329 tests pass/two skipped/64 historical AMP warnings in22.60s,35 focused checks before pilot. Source-frozen pilot prepared in outputs/navigation_m2_varied_wait_pilot_20261018, result pending.
**Lesson:** Audit the actual full control trace and variable timing before treating short sampling or recurrent memory as sufficient evidence.
**Interview angle:** How were unsaved control ticks and a timer shortcut checked?

## 2026-10-18 14:48 - Varied-wait pilot passes; whole-history training helpers tested before expansion
**Type:** milestone
**Stage:** Navigation M2 timing curriculum and causal-learning implementation

**What happened:** Fresh Town03/134 pilot with7s TRAIN stationary hold yields214 frames/7 shards/70.09MB,11.0s actual stationary wait and4.8s after green. It passes the original continuous-data gate, with declared no-background/7s override and no actor deletion. Only expert behavior is established.
**Cause:** The override changes the observation/release timing while preserving the scenario's4s clearance and all evaluation defaults, preventing one fixed7s wait from being the only new training pattern.
**How we handled it:** Prepare24-episode four-town varied-wait schedule3/7/11s (expected~7/11/15s total). Add TRAIN-only causal-learning cache/whole-episode forwarding with visual checkpoint chunks, fixed burn-in option and optional exclusion of unapproved perception supervision. Default student loss remains bit-identical; camera architecture/controller unchanged. Original validation and component geography remain reserved. No new camera fit yet.
**Verified by:** Pilot data_audit/background_scope;16 focused student/history tests pass, including full-forward/gradient parity, early fixed burn-in, no false state-label gradients and original-split filtering. Full suite completion checked before launch. New training module forbidden in camera process. Fresh prepared expansion outputs/navigation_m2_varied_waits_20261018; results pending.
**Lesson:** Verify that timing variation is actually recorded, and validate causal-gradient/label boundaries before training a new recurrent policy.
**Interview angle:** How can long recurrent training fit the GPU without resetting its history or supervising invisible signals?

## 2026-10-18 15:00 - Varied-wait data/audits pass; launch one protected full-history camera fit
**Type:** decision
**Stage:** Navigation M2 causal camera countermeasure

**What happened:** Varied-wait expansion completes24/24 episodes,5,151 timestamps/170 shards/1.941GB, no infrastructure errors.12 accepted episodes at four locations in three towns total2,550 timestamps; actual accepted waits cover7.0/11.0/15.0s (route225 adds~.1s). All original failed coverage cases remain excluded. Independent10,296-tick20Hz pose audit passes strict TRAIN geography. No camera result yet.
**Cause:** Complete observation/history and varied release timing address the missing-sequence/timer confounds before trying another bounded fit.
**How we handled it:** Freeze causal-memory protocol and fresh outputs/navigation_m2_causal_history_20261018. Same selected camera-stop-go-v2 parent/controller,1000 updates/3600s cumulative cap,AdamW5e-5,BF16,whole-episode GRU history/visual checkpoint chunks8,burn-in8. New supplement has action/path/speed supervision only; unknown perception labels have no loss. Preserve original13 validation episodes/32-frame label positions, remove3 newly reserved component locations from ORIGINAL TRAIN sampling (47 episodes remain), protect causal and clipped stop/go metrics. Keep parent0 if no checkpoint passes; no identical-weight driving rerun. Full optimizer/RNG/validation-step recovery, STOP and25GB reserve.
**Verified by:**334 CPU tests pass/two skipped/64 historical warnings in26.16s; five causal tests verify gradient/history parity, loss masking, unchanged default loss, location filtering, exact validation-label positions and selector regressions. Source/lint/diff pass. Completed varied data and matching independent20Hz report verified; host no simulator/trainer, GPU0%/39C,30.37GiB free. Cache fits reserve; cache/parent baseline checked before optimizer updates. M6 suite already frozen/native-audited. Fit result pending.
**Lesson:** Test the combined data/history hypothesis with fixed label positions and explicit parent protection; only later driving evidence can justify advancement.
**Interview angle:** How is a whole-history model compared fairly with the earlier clipped-history parent?

## 2026-10-18 15:09 - Full-history fit completes; reject every update and keep selected parent
**Type:** result
**Stage:** Navigation M2 camera countermeasure; driving gate still open

**What happened:** One combined continuous-data/whole-episode fit completes1000 updates. Original47 TRAIN episodes plus12 varied-wait sequences; original13 validation episodes/1312 scored labels unchanged. Parent causal stop recall42.50%,false holds4.81%,go MAE1.980m/s. Latest1000:23.39%,0.17%,2.576m/s; overall speed MAE2.583→3.490m/s. ALL trained checkpoints fail protection; selected0/best_policy.pt remains byte-identical to parent80bee4... No new driving run on identical selected weights.
**Cause:** Fewer false holds came with substantially more missed required stops and worse go-speed error. Complete history/varied expert sequences plus this optimizer/architecture did not fix perception/control generalization. Parent causal and clipped metrics were almost identical, so extra inference history alone was not evidence of a fix. Do not claim the failure causally identifies a single component.
**How we handled it:** Preserve full optimizer/RNG/source/cache/results and rejected latest; no blind resume or longer repeat. Next implement the approved localized-perception→learned driving connection: frozen RGB detector/state facts, a zero-initialized learned stop residual, and frozen parent steering/context to prevent destructive forgetting. Declare new inputs/cadence/labels/cost and revalidate before camera deployment; no ground-truth signal/control rule added.
**Verified by:** outputs/navigation_m2_causal_history_20261018/validation.jsonl/review.json and completed progress.334 tests passed before launch. Watcher KeyError from absent go_predicted_stop_fraction fixed without changing trainer: optional field and explicit last-validation/selected-step display, manual live snapshot passes. Protocol deviation: code also evaluated750 although declared list was0/250/500/1000;750 was rejected and had no selection effect. Keep/report all five evaluations, no retrospective omission. Supplemental labels remain unapproved for perception.
**Lesson:** Reject an apparent anti-stall improvement when required-stop recall deteriorates; preserve the parent and change the hypothesis instead of lengthening the same run.
**Interview angle:** Why did the longer-context experiment fail its gate despite fewer false holds?

## 2026-10-18 15:18 - Sampler diagnosis and frozen-parent RGB residual comparison
**Type:** decision
**Stage:** Navigation M2; M3–M6 remain open

**What happened:** Exact replay of1000 logged sampler draws finds42.25% stop labels on original updates,52.45% on approach updates,47.35% combined. Rejected latest loses signal-memory stop hits33→0/278, signal-stop146→88/201 and stop-sign41→25/63. There is no demonstrated shortage of stop labels; causal mechanism remains open.
**Cause:** Destructive generalization in the combined whole-network fine-tune; temporal context alone is insufficient evidence.
**How we handled it:** Freeze reports/navigation_m2_signal_residual_protocol_20261018.md: all parent parameters immutable, zero-initialized learned nine-logit stop residual, RGB object attention/64-unit GRU. Compare actual localized RGB facts against identical null-token architecture with matched1000-update budgets. Native three-view detector inference2.5Hz, past-only age-marked inputs10Hz; no oracle association or handcrafted light rule. Original validation/parent/cache preserved, new reserved component TRAIN locations excluded, all failed fits retained. Cache must finish/source-verify before either fit.
**Verified by:** reports/navigation_m2_causal_diagnosis_20261018.json;339 tests pass/two skipped/64 historical warnings in18.27s. New tests prove zero-residual exact parent parity, frozen parameter/path/cruise invariance after an update, masked-token and causal-memory behavior, native rotations, past-only2.5Hz cadence/age and original split protection. M4 inherited full-label leakage/common perception cost explicitly documented.
**Lesson:** Isolate a learned decision change and compare it with a matched no-facts ablation before attributing improvement to localized perception.
**Interview angle:** How do you test a perception countermeasure without forgetting the working controller?

## 2026-10-18 15:30 - Guard candidate evaluation and prepare matched stop-residual execution
**Type:** milestone
**Stage:** Navigation M2 implementation; M6 execution safeguards only

**What happened:** RGB cache runs19,461 strict TRAIN/internal-validation timestamps, currently~54% at14.6frames/s. New matched RGB/null trainer and actual privilege-guarded child inference pass. All348 tests pass/two skipped/64 historical warnings in27.22s. Camera behavior is not yet improved or evaluated.
**Cause:** Preserve the selected parent while testing localized perception, and prepare later evaluations without prematurely consuming the frozen test.
**How we handled it:** Implement causal fact caching, full optimizer/RNG resume, per-family stop protection, immutable exported architecture and serial two-arm orchestration. Add a stage-following read-only watcher. Extend encounter runner for SHA-bound camera candidates, preserving expert default/seeds and native scoring. Add separate M6 candidate registry, qualification/report hashes, fixed result root, exact final traffic seeds, source identity, no collection/overrides/profiling, all36 denominator and resumable infrastructure-only retries. No Town05 driver run or candidate registration exists.
**Verified by:** New tests cover actual child process inference, cache/runtime cadence, parent parameter invariance, family regressions, valid future-label masking, subprocess arguments, changed-checkpoint rejection, incomplete M2–M5 gate rejection and final seed/route/collection/override boundaries. Full-suite legacy preprocessing fixture initially failed because it bypassed __init__ without the new fact_arm=None; initialize that fixture explicitly, rerun full suite passes. No active cache source changed.
**Lesson:** Prepare evaluation execution early while enforcing the distinction between implemented safeguards and completed driving evidence.
**Interview angle:** How do you prevent repeated final tests or accidental label leakage while iterating on a driver?

## 2026-10-18 15:38 - Prepare rectangular JEPA and causal research boundaries without claiming results
**Type:** bug
**Stage:** Navigation M4/M5 prerequisites; M2 experiment continues

**What happened:** Existing ViT positional interpolation derives time from total tokens and stored square spatial count. Navigation8×144×256 with16px patches has grid4×9×16=576; the old method treats576 as9×8×8. Equal token counts can silently encode different spatial/time geometry. Historical JEPA/Dreamer paths are unchanged.
**Cause:** Square-resolution assumptions make total token count insufficient to recover T/H/W axes.
**How we handled it:** New navigation-only rectangular_jepa.py interpolates an explicit3D positional grid; retains spatial tokens and past-only episode-bounded clips. Add whole-group nested label-budget helper (actual fractions reported, no validation/Town05 labels) and untrained ordered-control dynamics primitive with independent3-model ensemble. Later controls cannot influence earlier predictions; no future target argument enters forward. No M4/M5 training, calibration or performance claim. Full-label inherited heads/common perception costs still prohibit a shortcut low-label study.
**Verified by:** Six focused tests pass for source-grid exact output/mask parity, rectangular equal-count axis disambiguation/interpolation gradients, causal episode windows, nested annotation caps, future-control leakage/order sensitivity, parameter independence and caller RNG preservation. The first interpolation test assertion needed detach().item() before pytest scalar comparison; implementation gradients remain tested. Cache/archive/source/parent/guard identities verified unchanged; current cache~84%, no fit result yet.
**Lesson:** Verify data/coordinate/causality boundaries before spending the downstream research budget, and never report untrained primitives as a completed milestone.
**Interview angle:** Why can a rectangular camera input silently break a video transformer's positions?

## 2026-10-18 15:39 - Host process visibility correction during serial execution
**Type:** bug
**Stage:** Navigation execution ownership

**What happened:** A sandbox anchored pgrep returned no active processes although the cache heartbeat continued. Running the same anchored host check with escalation identifies exactly cachePID659257 and waiting orchestratorPID1170437; no CARLA, trainer or job queue. Host GPU95%,52C,3666/12227MiB.
**Cause:** Host process visibility differs from the sandbox process view; an empty sandbox listing is not evidence that host work has stopped.
**How we handled it:** Use escalated anchored host process/GPU checks for ownership decisions. The existing host resource/lock checks had already run before cache launch; no duplicate or competing simulator/trainer was started. No job was stopped or source changed.
**Verified by:** Host pgrep plus continuing cache progress~94%; exactly one upstream cache and one waiting serial orchestrator.
**Lesson:** Verify ownership in the namespace where the job actually runs before treating absence as completion.
**Interview angle:** Why can a process listing disagree with a live training heartbeat?

## 2026-10-18 15:46 - Reject matched RGB/null stop residuals; reserve storage for missing reference
**Type:** result
**Stage:** Navigation M2; parent retained, new heldout_dev reference next

**What happened:** Cache completes19,461 timestamps/4,894 native detector triplets in1319.28s. Matched1000-update fits finish RGB12.56s/null6.57s (cached training, not detector time). All trained checkpoints rejected in BOTH arms. Parent recall42.50%,false holds4.81%,go MAE1.980m/s; latestRGB34.29%,2.41%,1.938; latestnull31.07%,3.26%,1.984. Memory stops latestRGB0/278; same unseen label positions/counts. Selected0 in both is byte-identical80bee4 parent; no new driver rollout on identical weights.
**Cause:** Localized RGB facts plus a learned stop residual lower approach TRAIN loss but fail original-location stop generalization; the matched comparison does not establish a driving benefit. Frozen steering/cruise prevent weight forgetting but cannot guarantee better stop decisions. Stop-label support/observation/reset/prior differences need diagnosis; do not assume one cause.
**How we handled it:** Preserve every checkpoint/source/validation/sampler and both negative arms. No longer repeat. Execute required new300-encounter expert reference while CPU audits inspect the failure. Previous reference consumes~5.8GiB; with30.71GB free this could breach25GB reserve. Verify all accepted raw source shard hashes and completed downstream caches/fits, then evict ONLY1,974,067,456 regenerable rgb/front cache bytes (actual recorded byte sum governs); retain manifest, per-array SHA/rebuild receipt, all labels, lossless raw shards, derived fact features and original M1 cache/models. Explicit missing-image errors require fresh-path regeneration.
**Verified by:** Both completed validation.jsonl/review.json and full recovery state; original exact zero-residual baseline. Five causal tests pass before storage release; host confirms no jobs/CARLA/trainers, GPU idle42C. Cache eviction receipt outputs/navigation_m2_causal_approach_cache_20261018/evicted_image_arrays.json binds all preserved raw hashes. Final/free-byte reading checked before reference start.
**Lesson:** Freeze the controller and keep a no-facts ablation, but reject both when stopped-scene generalization regresses; allocate disk for required evidence rather than repeat a losing fit.
**Interview angle:** Did localized perception improve the driver, and how was a negative finding handled?

## 2026-10-19 12:21 - New held-out expert reference audited; invalid setups stay in300 denominator
**Type:** result
**Stage:** Navigation M0/M2 reference; camera driving still open

**What happened:** New heldout_dev settles300/300 across17 locations: native299 arrivals/296 rule-safe, but two HardBreak setups116/189 are missing. Audited headline297 valid arrivals,294/300 rule-safe(98.0%),57/300 strict-safe, two collisions102/177, one red187, one blocked140. All50 stops reuse two locations. Cluster95% rule-safe95.75–99.44%; raw strict speed penalties retained.
**Cause:** Native arrival/rule records do not prove the declared scenario was instantiated. Existing valid-only family diagnostics were insufficient as a300-case headline.
**How we handled it:** Add planned_headline to assess_encounters.py, retaining invalid/missing cases as failures and every raw-attempt collision; assessment_version4. Original raw/logs/legacy reports unchanged, no route/criterion/seed replacement. New report reports/navigation_heldout_expert_20261019.md plus audited JSON. One evaluator abort on266 gets its predeclared infrastructure retry; eight later routes were not reached, not failed driver repeats.
**Verified by:** All33 attempt source archives match hashes;100 held-red physical identities pass.198 scripted starts,100 no-setup routes, two missing. Seven assessment tests pass, including missing/wrong/pending-case denominator and raw retry collisions; full suite checked before next simulator trial.
**Lesson:** Keep every planned scenario in the headline even when simulator setup fails; separate that from valid-only driving diagnostics.
**Interview angle:** Why did an initially296/300 result become294/300 after auditing?

## 2026-10-19 12:21 - Freeze one TRAIN-only stopping-gap trial after three rejected fits
**Type:** decision
**Stage:** Navigation M2 observability/data countermeasure

**What happened:** Exact original sampler replay matches saved RNG:40.475% stopped labels through selected1500 versus47.35% new-mixture updates. Parent memory stop hits842/855 TRAIN versus33/278 validation; any-lamp red score AUC on validation signal-stop.271 demonstrates unresolved applicability, not a deployed classifier. Existing TRAIN static phase evidence at12m27/27 poses versus2m19/36 motivates measuring stop pose; counts are unpaired and do not prove invisibility or a cause.
**Cause:** Location/observation generalization remains open; there is no established stop-label shortage. More residual fitting was rejected.
**How we handled it:** New guarded TrainingGapExpert changes only TRAIN teacher IDM red-light minimum gap6→12m, with consistent teacher labels/control and explicit metadata. Canonical expert/student evaluation unchanged. Freeze one route134 pilot with7s hold, unchanged strict geography/continuous-data criteria,900s/300s caps and STOP. No new camera fit or state-label release until pose/optical evidence is checked.
**Verified by:**16 focused TRAIN guard/IDM/assessment tests pass; actual inherited IDM consumes12, separate reference instance retains6, green remains unchanged; camera child forbids the teacher import. Prospective reports/navigation_m2_light_gap_protocol_20261019.md, prepared source-frozen outputs/navigation_m2_light_gap_pilot_20261019; result pending.
**Lesson:** Measure whether the teacher's privileged stopping pose preserves the student's observation before changing model complexity again.
**Interview angle:** Why test a training teacher's stopping distance before another network fit?

## 2026-10-19 18:13 - Stop-gap pilot gives no measured coverage gain; preserve diagnostic data only
**Type:** result
**Stage:** Navigation M2 teacher-observation hypothesis

**What happened:** One route134 pilot completes214 aligned timestamps in25.23s,11.0s wait/5.1s after release, unchanged coverage gate passes. Actual stationary planner-reference distance5.7→11.1m. Same projected front/left/right housing coverage; front height7.62→6.24px. No optical benefit or camera-model gain established. Separately matched historical TRAIN32-pose data:27 movements with both2/12m renders give responsive housings14/27 at2m versus27/27 at12m; this is scene-dependent, not proof this pilot benefits.
**Cause:** Earlier stopping does not uniformly improve observations; visibility and applicable-lane association remain open.
**How we handled it:** No expansion or imitation fit from this teacher trial. Keep raw data/audits and explicit collection_review.json imitation_release=false. Standard teacher remains6m. Found pilot root generic policy_kind/training-mixture labels lagged its explicit gap metadata; fixed future header/collection classification without editing old artifacts. Stop-gap report records that discrepancy.
**Verified by:** Completed data_audit/metadata; native frames/recorded same-frame world geometry projection for both stopping poses. Both remain visibility-uncertified. User-visible result remains unchanged original parent. No benchmark/reward/route/camera modification.
**Lesson:** A camera-aware stopping hypothesis needs measured observation improvement; moving the teacher earlier is not sufficient.
**Interview angle:** Why was a clean earlier-stop demonstration kept out of imitation training?

## 2026-10-19 18:13 - Freeze spatial encoder comparison and verify official backbone exports
**Type:** decision
**Stage:** Navigation M2 new baseline/M4 research implementation; gates still open

**What happened:** Prepare one new common spatial camera head across five encoders, fresh heads and nested59-episode/16639-timestamp label budgets. Shared continuous native RGB/opponent fields preserve tiny colored pixels without annotations, thresholds or control rules. No642-label detector or old full-label driving head is inherited. Keep original13 validation episodes/1312 scored positions, all held-out boundaries and every failed prior fit.
**Cause:** Prior models overfit location and fail stop generalization; the approved stronger-backbone/label-efficiency study directly tests representation value. Combined architecture changes versus historical v2 cannot isolate an encoder effect; matched arms within the new study can.
**How we handled it:** Freeze navigation_m4_spatial_protocol_20261019.md: five encoders × four actual whole-episode label budgets × three seeds,2000 equal-draw whole-episode updates, plus six predeclared common-component ablations. Implement spatial attention/causal GRU/loss, past-only RGB access, quantized caches and actual guarded-child deployment. Official ImageNet/DINO checkpoints SHA-registered; own comma checkpoint unchanged. In-domain JEPA/M3/M5/final driving still pending.
**Verified by:** Nine focused field/causality/loss/split tests plus actual guarded spatial child and legacy regression tests pass. GPU strict-load smoke outputs three-view4x4 grids, all weights frozen: ResNet18 D512/16.55ms, DINOv2-S D384/13.56ms, comma-JEPA D384/6.95ms per8-frame clip, no IO/driving claim. Downloads use official PyTorch/Meta URLs. Planned native field cache has zero annotations and fits25GB reserve.
**Lesson:** Keep a shared observable frontend, fresh heads and explicit annotation accounting so encoder comparisons cannot inherit hidden full-label supervision.
**Interview angle:** How can a small signal survive downsampling without adding privileged rule inputs?

## 2026-10-19 18:23 - Shared native RGB fields and fixed annotation budgets complete
**Type:** milestone
**Stage:** Navigation M4 data; M2 camera gate still open

**What happened:** Source-frozen native RGB field cache completes19461 timestamps in408.02s, exact source navigation/ego and raw-frame hashes checked. Labels are copied from original47 strict TRAIN episodes plus12 standard-gap varied-wait episodes;13 validation episodes/1312 scored positions unchanged. Nested whole-episode allocations:733,1257,4069,16639 labels. Actual fractions must be reported, not nominal5/10/25/100%. No diagnostic gap-pilot labels or642-label component weights enter this study.
**Cause:** Preserve tiny RGB evidence cheaply and make every encoder receive the same visible inputs and label exposure.
**How we handled it:** Shared photometric tokens are deterministic/unlabeled, three native views,11 channels/144 cells; frozen context encoders retain4x4 spatial cells. Implement fresh common-head initializer with identical shared weights across backbone dimensions, causal whole-episode training, exact registered evaluation steps, full optimizer/RNG recovery and guarded spatial deployment. ResNet cache/profiling precede first2000-update fit.
**Verified by:**364 full tests pass/two skipped/64 historical warnings; eight focused spatial tests pass including common initialization, unchanged split, small-pixel retention, past-only gradients and actual privilege-guarded child inference. Three official/own pretrained encoders already strict-loaded and CUDA-profiled. Cache manifest binds arrays/raw shards/request/source hashes.
**Lesson:** Count actual complete-episode annotations and retain only observable common features; do not inherit a full-label perception or driving head into a low-label claim.
**Interview angle:** Why do actual label fractions differ from the nominal study budgets?

## 2026-10-19 18:23 - Clarify earlier matched optical-pose wording
**Type:** mistake
**Stage:** Navigation M2 documentation

**What happened:** The previous stopping-gap result entry said "TRAIN32-pose data". The expanded corpus has99 poses at nominal32/12/2m;32 referred to a distance condition, not the number of poses. The matched subset has27 movements with both2m and12m renders:14/27 versus27/27 responsive, as already reported.
**Cause:** Ambiguous abbreviated wording in the journal entry.
**How we handled it:** Append this correction without rewriting the prior entry. All underlying data, matched counts and the decision not to expand the teacher gap trial remain unchanged.
**Verified by:** outputs/navigation_m2_signal_expansion_20261016/*/cases.json and data_audit.json.
**Lesson:** Keep spatial distances, numbers of poses and matched-movement denominators distinct.
**Interview angle:** How were provisional observation counts kept traceable?

## 2026-10-19 18:32 - First fresh spatial camera fit completes; driving pilot begins
**Type:** milestone
**Stage:** Navigation M2 candidate/M4 first arm; no driving promotion

**What happened:** ResNet18 feature extraction completes19461 timestamps in66.99s. Longest1190-frame TRAIN gradient profile uses0.885GB VRAM; cold11.11s then44.25/45.87ms, finite gradients and zero updates. Registered2000-update seed42/full-label fit completes; prospective selector chooses500, validation stop recall96.79%, false holds4.47%, go MAE1.523m/s. Later updates regress and are retained.
**Cause:** Fresh spatial/context/photometric attention generalizes offline substantially better than rejected parent residuals; architecture and representation changes cannot yet be separated.
**How we handled it:** Freeze the selected checkpoint and run the unchanged six-case engineering pilot with native scoring, approved sensor-only IPC and overrides disabled. Original80bee4 remains historical selected driver until real driving evidence. Remaining matched encoders/budgets/seeds and M3/M5/M6 remain open.
**Verified by:** Registered0/500/1000/1500/2000 validation rows, fixed1312 scored positions and original13 validation episodes; source/hash request and complete optimizer/RNG saved. outputs/navigation_spatial_resnet_f100_seed42_20261019; profile_steady report.
**Lesson:** A fast fit and strong offline stopping score warrant an actual driving test, not a success claim.
**Interview angle:** Why did the selected checkpoint precede the final training update?

## 2026-10-19 18:32 - Correct pilot output-path handling before simulator launch
**Type:** mistake
**Stage:** Navigation execution tooling

**What happened:** First spatial pilot launcher failed computing an absolute-root-relative ledger path from a relative output argument. No CARLA or driving trial had started.
**Cause:** New launcher did not normalize its output Path.
**How we handled it:** Resolve output before creating artifacts; retain the original attempted directory and launch fresh navigation_spatial_resnet_pilot_v2_20261019. No route/model/settings change or driving retry.
**Verified by:** Ruff passes; live native route log/heartbeat confirm the corrected run.
**Lesson:** Normalize filesystem identities at CLI boundaries before writing provenance.
**Interview angle:** How do you distinguish an infrastructure retry from repeating an unfavorable driving result?

## 2026-10-19 18:42 - Spatial driver's offline gain fails to establish launch; measure ego-speed shortcut
**Type:** bug / decision
**Stage:** Navigation M2 feasibility; full M4 fitting deferred before starting

**What happened:** Initial native spatial pilot cases remain stationary despite96.79% offline stop recall. Post-hoc CPU-only original1312-position probe: measured-speed false holds4.47%; setting only neural ego-speed input zero raises false holds83.51%; fixed5m/s collapses stop recall to0%. Blanked RGB-feature false holds25.77% shows visual dependence also exists. Artificial input interventions are not physical counterfactual evidence. TRAIN has392 stationary-go labels versus8910 stationary-stop;273/392 launch labels lie in the first8 timestamps omitted by burn-in.
**Cause:** Strong evidence of a speed-conditioned shortcut and a launch supervision mismatch. Causal role of each remains unconfirmed; do not call high stop recall learned red-light behavior.
**How we handled it:** Retain the entire ongoing six-case pilot and selected a2c479 model, never promote it. Do not blindly launch66 matching fits on a failed feasibility prototype. Freeze a single fresh M2 neural-input ablation in navigation_visual_decision_protocol_20261019.md: mask ego speed only inside the model; actual measured speed still reaches the unchanged physical controller. All labels/data/validation/burn-in/sampler/2000 updates unchanged so the first test isolates the input. No forced-go, teacher label/reward/scoring change, heldout_dev/Town05 use or rule override. Registered full encoder study remains pending.
**Verified by:** reports/navigation_spatial_speed_diagnosis_20261019.json and source script; original observed CPU metrics agree within numerical precision. Three unlabeled JEPA mask/geographic/scene-variance tests plus14 combined focused tests pass; continuation and66-fit runner are implemented but not launched.
**Lesson:** Teacher-forced validation can reward an ego-state shortcut that deadlocks when the policy controls its own state; audit launch transitions and actual closed-loop behavior before a broad study.
**Interview angle:** Why can excellent offline stop recall produce a car that never starts?

## 2026-10-19 18:46 - Spatial pilot0/6; input ablation completes and enters identical pilot
**Type:** result / milestone
**Stage:** Navigation M2; no promotion or M4 completion

**What happened:** Spatial a2c479 candidate completes the planned six native tests with0/6 arrivals, all0% ordered progress: one scenario timeout/five blocked. Zero recorded collisions/red/stop violations reflect no movement, not safety. Setup audit verifies every declared setup and physical light identity, all six videos decode, camera synchronization/traces/source archives/unique native recorders pass.
**Cause:** Speed-conditioned shortcut/launch mismatch as separately diagnosed; high offline96.79% stop recall was misleading about autonomous decisions.
**How we handled it:** Retain failed candidate and all artifacts; no duplicate long timeout or66-fit expansion of it. Fresh no-neural-speed ablation completes2000 updates/28.51s, selects2000 with40.0% stop recall/7.56% false holds/3.200m/s go MAE. This lower offline score is disclosed, not compared as a new driving gain. Launch its same six-case pilot with overrides off; physical controller retains actual speed.
**Verified by:** outputs/navigation_spatial_resnet_pilot_v2_20261019/assessment.json audit_passed=true, all_attempt_collisions0; new fit request/best/review/validation and guarded child inference. Full370 tests pass/two skipped/64 historical AMP warnings in27.78s.
**Lesson:** Count immobility as failure even when collision/violation counters remain zero, and retain failed offline-to-online generalization evidence.
**Interview angle:** How was a high-scoring model explicitly rejected?

## 2026-10-19 19:02 - First full camera destination drive completes, unsafe; corrected encoder study starts
**Type:** milestone / result
**Stage:** Navigation M2 feasibility and M4 execution; safety gates remain open

**What happened:** Fresh visual-decision c07ea178 reaches6/6 fixed pilot destinations:4 compliant/3 strict,1 red/1 stop violation and zero recorded collisions. Original speed-conditioned a2c479 reached0/6. Identical2000 label draws and initial weights verify; actual speed remains in physical control. On original1233.361m Town03 route/seed42 it reaches100% in372.9 simulated/259.3 wall seconds, but5 vehicle collision events (including repeated contact events with one actor),2 red/1 stop violation and1 minimum-speed infraction: failed safety result. Count all five, no fault filtering.
**Cause:** Removing the neural ego-speed shortcut permits launch and route following; perception, applicability and interaction safety remain inadequate. This is not a reliable self-driving model.
**How we handled it:** Keep all old/new failures; freeze the corrected common architecture before other encoder fits, reuse the exact ResNet feasibility fit, run all66 matched label/encoder/seed/component cells serially. No642-label detector or full-label head inherited. Scratch longest1190-frame gradient profile finite/.329s steady/1.369GB VRAM/zero optimizer updates. Study outputs/navigation_visual_study_20261019 is running; source/optimizer/RNG/sampling/validation hashes and regenerable-cache receipts preserved.
**Verified by:** Both six-case independent audits pass, exact physical light/setup/video/source/recorder checks; long7458 contiguous aligned frames,14.02ms p95 packet/inference/IPC, no intervention,3729-frame H.264 video full decode and native recorder passes. scripts/summarize_navigation_visual_decision.py reproduces JSON/figure/report.370 full tests/two skipped plus three matrix tests pass.
**Lesson:** Reaching a distant goal is progress in feasibility; any collision/rule failure still prevents a safety promotion.
**Interview angle:** What changed between the stationary model and the first1.23km camera drive, and why is it still rejected?

## 2026-10-19 19:02 - Integration route exposes two held-out development locations
**Type:** mistake / decision
**Stage:** Navigation geographic independence audit

**What happened:** The historical M0 engineering route was reused as prospectively specified, but its actual trace crosses newly heldout_dev Town03 junctions1135/103 for142/105 ticks. I should have checked its geography before launching. These camera observations are development exposure even though no catalog encounter was run and no labels are added to TRAIN.
**Cause:** The old long route predates the new global location split; treating its engineering purpose as sufficient separation missed physical overlap.
**How we handled it:** Disclose exposure in reports/protocol/README/handoff; keep the frozen300 cases and every denominator, never remove these sites after seeing results or tune on their failures. The corrected shared architecture/protocol was fixed before final long-drive outcomes. Training/pretraining source arrays exclude both locations; Town05 remains unused by student evaluation. No new split or benchmark change is made. Full17-location untouched-geography claims are no longer valid.
**Verified by:** Native offline CARLA Map mapping of every stored ego center; reports/navigation_integration_exposure_20261019.json. Full all-collision safety headline retained.
**Lesson:** Audit an old integration route against new split boundaries before reusing it, not just against its historical purpose.
**Interview angle:** How was test geography exposure disclosed without filtering the benchmark?

## 2026-10-19 19:33 - TRAIN-only JEPA continuation starts after39 matched fits
**Type:** milestone
**Stage:** Navigation M4, not a driving gain

**What happened:** All13 ResNet,13 DINOv2 and13 comma-JEPA cells finish, including color-disabled controls; exact2000 sampler draws are checked across matched encoder/seed/budget cells. The registered TRAIN-only JEPA continuation starts10000 updates. Zero-update24-view gradient profile:8.0s cold, then.209/.115s, .873GB VRAM, finite gradients; rough27-minute budget estimate. At1000 fixed updates, holdout latent loss.3495→.2495, within-clip variance.4394→.4623, between-scene variance.2588→.1214. Moving-target loss is not downstream quality or a selector.
**How we handled it:** Keep fixed final-update export/EMA/variance guards, exact TRAIN-only source/episode holdout and no action labels. Original13 internal driving validation episodes unchanged. Additional random-ViT diagnostic still required after main66 cells; no full M4 completion yet.
**Verified by:** Source-frozen study requests/results, JEPA profile/validation/checkpoint metadata;384 full tests pass/two skipped/64 historical warnings, plus six storage focused tests.
**Lesson:** Distinguish unlabeled latent diagnostics from measured driving or label-efficiency gains.
**Interview angle:** Does decreasing JEPA loss establish a useful driving encoder?

## 2026-10-19 19:33 - Correct filesystem-compression assumption and restore all trace paths
**Type:** mistake
**Stage:** Resource management, data preservation

**What happened:** I losslessly gzipped1170 completed world traces (9.029GB logical) expecting useful physical headroom, without first checking the filesystem. All original-byte SHA/size checks passed and a complete six-case native/video audit reproduced identically through temporary restored paths. Host findmnt then showed Btrfs compress=zstd:1: logical reduction7.906GB was not comparable physical recovery. No open deleted world-trace descriptors existed.
**Cause:** Confusing logical file sizes with physical space on a compressing filesystem.
**How we handled it:** Restore every original trace/path/mode/mtime after full SHA/size verification, then remove redundant gzip/sidecars.1170/1170 restored, all receipts/source snapshots preserved under trace_storage_pilot/full and trace_restore outputs. Source PNG shards, videos, original caches, checkpoints and outcomes remain intact. Actual host free space26.77GB; preserve25GB reserve and measure future storage costs. M4 training continued unchanged throughout.
**Verified by:** outputs/navigation_trace_restore_20261019/summary.json; six corruption/round-trip/restore/overwrite tests pass; restored raw count1170, archives0. Old native pilot assessment reproduces byte-identically.
**Lesson:** Inspect filesystem compression and physical free-space changes before a data-storage transformation; reverse an unnecessary layout change without losing evidence.
**Interview angle:** Why did a7.9GB logical reduction fail to free that much disk?

## 2026-10-19 19:33 - Controlled rule-test contracts and explicit scripted handoff implemented
**Type:** implementation
**Stage:** Navigation M3 infrastructure, no native student trials

**What happened:** Six one-treatment pair contracts and a bounded scripted-prefix controller have seven focused passing tests. The prefix returns no control after immutable handoff; evaluated agent proposals stay unchanged. Memory/yellow/stop/cross treatments explicitly disclose setup assistance and are not autonomous route successes. Expert failure cannot exclude an otherwise valid test.
**How we handled it:** Prospective navigation_rule_counterfactual_protocol_20261019.md; physical setup, actor/light binding, frustum/memory validity, native routes, reference and all candidate results still pending. The camera child now forbids setup/storage/training helper imports. No canonical benchmark/reward change or gate claim.
**Verified by:** Prefix overshoot/timeout/immutability, one-factor pair and non-filtering certificate tests; actual guarded legacy/new camera child remains passing.
**Lesson:** Keep component setup assistance visible and separate from the period used to assess learned control.
**Interview angle:** How can a memory test use a controlled initial pose without hiding assistance?

## 2026-10-19 20:16 - Discover stationary collision filtering; official recorder reanalysis corrects safety headlines
**Type:** bug / result / correction
**Stage:** Navigation M0/M2/M6 evidence integrity

**What happened:** Actual pinned scenario_runner_autopilot CollisionTest._count_collisions returns when ego speed<.1m/s, explicitly treating stationary collisions as not ego fault. This violates the approved all-collision headline; my earlier wording "all collisions" for native counters was too broad. Offline official-format recorder decoding finds six contact starts across expert102/170/177 and13 on the long camera drive, versus native2/5. Every row in all313 complete recordings matches official CARLA show_recorder_collisions(h,a)/(a,h) queries exactly; one partial crash recording is retained incomplete. No policy/traffic mission was started for the file-query server.
**Cause:** Upstream native criterion includes fault/speed and duplicate filters that were not caught by prior adversarial tests. Unfiltered recorder data were fortunately preserved.
**How we handled it:** Preserve native scores/artifacts; reanalyze under the already agreed no-fault-filter contract. New expert reference rule-safe294→293/300 and strict57→56/300 because170 was native Perfect but contacted another vehicle twice. Keep two missing setups failed, every contact in all attempts and raw samples/ordered query rows. Short pilots remain0 contacts; long drive13 contact starts/2 red/1 stop remains unsafe. Native event counts remain supplemental, not physical-contact totals. Added decoder/partial/cross-query tests, exact official source archive and correction report. Final runner now needs unfiltered contact-free evidence and setup identity, not native success alone.
**Verified by:** reports/navigation_collision_correction_20261019.md/json; outputs/navigation_recorder_rpc_20261019/verification.json313 complete agree/one partial/zero failures. Native opportunity hooks preserve original criterion return values and record actual eligible actor/lane pairs; fake-class passthrough and legacy harness tests pass, native live probe still pending.
**Lesson:** Preserving raw simulator recordings enables correction of hidden benchmark filters without repeating favorable/unfavorable driving outcomes.
**Interview angle:** Why was an expert's native Perfect result reclassified as unsafe?

## 2026-10-19 20:16 - Safely interrupt and resume exact study for scoring verification
**Type:** decision
**Stage:** Navigation M4 lifecycle

**What happened:** With53/66 fits complete, touch study STOP propagates to scratch seed123, saving1766/2000 updates plus optimizer/RNG state. Host anchored process audit confirms stopped. Owned CARLA file-query check completes and exits. Every registered study source hash still matches; handled STOP receipts renamed and identical study resumes from1766, with234 updates remaining, no new validation schedule or budget.
**Verified by:** STOP.handled receipts, original/resumed requests, full checkpoint and study ledger. GPU source boundaries unchanged; no force kill or job-script edits. Temporary raw supplement RGB is reconstructed in RAM and SHA-matches the evicted original exactly, allowing scratch backbones to see images without repeated PNG decode; no CNN features cached.
**Lesson:** A scientific accounting defect can be audited without discarding or restarting a costly matched fit.
**Interview angle:** How do you pause a matched training arm without spending its budget twice?

## 2026-10-19 20:42 - Register physical rule prototypes and queue seven expert-only setup probes
**Type:** implementation / decision
**Stage:** Navigation M3; no student rule outcomes

**What happened:** Prepared24 separate case geometries across all six one-treatment families, two signal towns/two strict stop locations and three paired seeds. Two earlier prototype directories retain22-case geometry failures: selected Town02/77 right turn has no eligible transverse crossing; a heading-masked geometry search still fails it. First independently geometry-eligible Town02 route80 supplies the complete prototype. No driver outcome or test-route replacement informed this choice; canonical300/Town05 routes unchanged.
**How we handled it:** Explicit controlled-empty background and native reference/student adapter hooks, immutable scripted handoff, current-observation logging before next-frame phase mutation, reactive non-ego vehicles and per-view housing-frustum diagnostics. Runtime strict geography guards preserve driving/component validation and Town05. No training labels are released, and native scores for these scripted-prefix components cannot become autonomous route success. Student execution refuses an uncertified setup. Seven expert-only native probes are queued after the exact66-fit study releases its lock, with source/route/case hashes, STOP and all failures retained.
**Verified by:** All24 actual case contracts pass; two adversarial contract tests reject rehashed treatment/foreign geography and collection/final/override/seed drift. Legacy/new camera tests pass; latest full400 tests/two skipped/64 historical warnings. Physical runtime validity remains pending; unit tests are not a native-rule result. Detected before probe: upstream PDM initializes _vehicle/_world on its first tick, so controlled setup now uses declared parent-side CarlaDataProvider hero during setup. No driving attempt had started.
**Lesson:** Geometry eligibility and component assistance must be explicit, with every physical condition certified before student interpretation.
**Interview angle:** Why are these controlled prefixes not evidence of autonomous long-route success?

## 2026-10-19 20:42 - Prepare M5 physical candidate/CV/RSSM primitives without claiming a planner
**Type:** implementation
**Stage:** Navigation M5 baseline groundwork, untrained/unvalidated

**What happened:** Shared3path×5speed candidates obey bounded acceleration/deceleration and actual wait duration; common cost uses ordered projection on the declared route, so lateral wiggles cannot earn fake progress. CV uses crossing motion, uncertain circular actor envelopes and stale-track age. Small continuous RSSM consumes ordered controls and has a prior-only inference API; future targets are confined to its training posterior/loss. Invalid/censored future labels are masked before loss and cannot become safe negatives. No learned predictor has been trained or deployed.
**How we handled it:** Implement a camera-only frozen COCO SSDLite detector with explicit external-label provenance and approximate flat-ground ray ranges/odometry tracking. Strict official weight SHA a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2,3.44M parameters, no local CARLA detector labels. First four timestamps of first strict TRAIN source run80–100ms/three views on CPU, zero detections; no GT accuracy/coverage or GPU/driving claim. Do not promote empty tracks as a safe scene. Physical perception calibration, complete ensembles/bootstraps, off-policy controlled branches, all costs and actual CV/RSSM/JEPA comparisons remain missing.
**Verified by:** Ordered-action/future-causality/censored-NaN loss, no false route-progress, physical stop/wait, crossing-risk and ego-motion-only tracker tests pass. No simulator actor state enters the perceived-track modules or inference boundary.
**Lesson:** A forecast equation is not a validated planner; measure perception and reactive alternative-action outcomes before promotion.
**Interview angle:** How do you keep a world model from labeling safety-intervened actions safe?

## 2026-10-19 20:42 - Reduce future recording overhead after measured codec profile
**Type:** decision
**Stage:** Recording resources; no policy/scoring change

**What happened:**320 full camera triplets encoded/decoded in a CPU-only preserved-video profile: ultrafast566fps/p952.17ms/9.45MB; veryfast328fps/p953.86ms/6.42MB, all320 frames retained. Source is already compressed, so pixel error is only a codec diagnostic, not semantic visibility proof.
**How we handled it:** Future evidence videos use the same H.264/baseline/yuv420p/CRF23/10Hz geometry with veryfast/two encoder threads. Every original video and lossless training shard remains unchanged. Inference uses native RGB before video compression; full video decoding remains audited. Measure actual physical disk headroom, not logical compression estimates.
**Verified by:** reports/navigation_video_profile_20261019.json and archived executed source; no simulator, labels, reward or canonical evaluator modified.
**Lesson:** Profile recording cost and preserve raw policy inputs separately from presentation video.
**Interview angle:** Can evidence-video compression change the policy's camera input?

## 2026-10-19 20:59 - Complete all66 registered encoder fits and freeze primary candidate
**Type:** result / decision
**Stage:** Navigation M4, offline evidence only

**What happened:** All60 matched label/seed cells plus five color-disabled controls and one navigation/color baseline completed and independently verified, with exactly2000 updates and the scheduled five validation points. At100% labels the mean selection scores are CNN.4126, ResNet.4183, DINO.4214, comma-JEPA.3421 and CARLA-JEPA.3854. The prospective lowest-mean/full-label/fixed-seed42 rule selects comma-JEPA SHA7015fef0776c5cbe03a96ed2e2f0043abb4eecfd2aa2f5ada8534ccfd3ef2b6b. This is not a driving qualification. CARLA continuation finished10000 updates in1419.96s; its lower latent loss did not beat comma-JEPA on this downstream selector.
**How we handled it:** Preserve every cell/seed/control and matched sampler sequences, original validation and annotations, main source archives and costs. Published reproducible curves/table/JSON. Additional identical-architecture random frozen ViT diagnostic is registered and extracting features after native probes finished; unchanged main training loop, same labels/seed2000 draws. Four remaining seed42 encoder pilots are next; reuse the completed ResNet pilot, never rerun it for favorable results.
**Verified by:** reports/navigation_m4_visual_study_20261019.md/json/png, navigation_m4_selection_protocol_20261019.md, completed study request/results/source snapshots. M4 driving comparison and all subsequent safety gates remain open.
**Lesson:** A lower familiar-validation score is a reason to test a frozen candidate, not evidence of safer driving or universal representation superiority.
**Interview angle:** How was the driving candidate selected without choosing a favorable seed after driving?

## 2026-10-19 20:59 - Complete seven native setup probes; reject false memory interpretation
**Type:** bug / result / decision
**Stage:** Navigation M3 physical setup, no student rule trials

**What happened:** First prototype completed empty red/green then failed lead setup because get_actors used a pre-spawn snapshot. Preserved that failed attempt and reconciled its stale running status after owner exit. A common one-tick actor acknowledgment in every arm, before sensor registration, fixes the infrastructure defect; all seven fresh expert probes exit0. Exit0 is not physical validity or student success. The memory reference holds on red but its right camera sees the applicable housing throughout the entire8s interval (about135px high): the intended out-of-view condition is invalid. Offline strict-TRAIN geometry search finds27 eligible signal approaches and only Town02 routes79/80/82 outside all views at3m; native confirmation is still required.
**How we handled it:** Keep the invalid Town01 prototype and all raw frames/videos/recordings; no pass, no student trial, no canonical route replacement. Expert native red eligibility instrumentation sees the expected bound actor/lane and leaves original criterion returns unchanged. Record actual handoff speeds: close-yellow about4.98m/s despite6m/s requested, crossing3.18 despite4m/s requested; do not silently label these as exact requested speeds. Correct the setup runner exception status for future invocations.
**Verified by:** outputs/navigation_rule_setup_pilot_20261019 and navigation_rule_setup_pilot_ack_20261019, all requested/executed controls, native opportunity logs, setup traces and reports/navigation_memory_geometry_search_20261019.json.
**Lesson:** Physical rendering and actual kinematics must establish a counterfactual condition before a behavior can support a learning claim.
**Interview angle:** Why does a red-hold success fail the memory experiment?

## 2026-10-19 23:43 - Resume external interruption without retrying settled driving failures
**Type:** lifecycle / mistake / result
**Stage:** Navigation M2–M5

**What happened:** Host inspection at23:36 finds no simulator/train/queue processes; all saved logs stop around21:27, with stale running heartbeats. Cause is not established. Comma-JEPA pilot is complete:2/6 arrivals,0 rule-safe,1red/1stop,130 unfiltered recorder contact starts (4 native events),125 contacts on the blocked stop case. I initially said3arrivals in commentary and immediately corrected to2 from native records. CARLA-JEPA case86 is a settled timeout; case101 was interrupted without a native record.
**How we handled it:** New source-frozen continuation replays every settled outcome and runs only missing routes. Retain the interrupted partial recorder and all its observed contacts; never count it as verified zero. Expert-reference and branch queues had not started any physical case, so explicit unstarted resumes retain old requests/sources and receipts. No force kill or benchmark change. A resume implementation briefly overwrote the original reference runner archive; restored its exact original bytes after independently verifying SHA48f5c3767133c313983d94e007ce0f92126553b12c93cfa189d8d46d9c15e78a. Separate resume source/request preserved. Future archive writes must use a new generation directory.
**Result:** Random-ViT additional diagnostic complete:.4122 versus comma.3809/CARLA.3824 for fixedseed42, all2000 draws and initialized heads identical. Offline encoder winner is unsafe in driving and is not promoted. All main66 artifacts remain intact. Seven expert setup probes independently pass frame/control/actor/source/video checks with0recordercontacts; Town01memory remains physically invalid.
**Lesson:** A completed unfavorable outcome is immutable even when a later process interruption requires resuming the experiment.
**Interview angle:** Which part of an interrupted evaluation can you rerun without cherry-picking?

## 2026-10-19 23:43 - Extend factual control audit and disclose source contacts
**Type:** implementation / result
**Stage:** Navigation M4 limitations / M5 data quality

**What happened:** Official CARLA recorder packets retain actual20Hz steer/throttle/brake, beyond the10Hz collection labels. Implemented strict packed-format decoder and full-pose alignment to same-frame world traces; action at RGBframef is acknowledged by native control atf+1, never the stale control atf. First actual source aligns uniquely with maximum position difference0.0000243m. Five decoder/partial/ambiguous/frame-gap tests pass; full72episode acknowledgment audit is running.
**Source integrity:** All72 preserved study-source recorders decode completely. One TRAIN episode, original:Town10HD_Opt:250:202610182:7767, contains one contact despite historical native-clean eligibility. Preserve the completed66-study data/results unchanged and disclose this limitation; do not silently remove an example after results. Supplementary12 source recorders have0contacts. New prediction data use unfiltered recorder contacts and censoring.
**Perception:** Frozen SSDLite native diagnostic165frames finds15/58 nearby origins within10m, median matched origin error1.606m,p952.113m,CPU82ms threeviews. Not visibility-certified recall. Stronger original COCOFPN arm registered before its predictions; prior CPU run externally interrupted before final report, so retain it and retry same fixed165frames with per-frame progress/prediction persistence. No local labels or encoder-study changes.
**Storage:** Two exact duplicate checkpoint pairs now share Btrfs COW extents, preserving every path,hash,mtime and independent future writes; tiny clone write-isolation test passes. Actual immediate free-space increase0.342GB, not the733MB logical duplicate total. Receipt navigation_checkpoint_cow_20261019.
**Lesson:** Recover exact executed actions from preserved evidence; never substitute teacher proposals or fault-filtered collision labels.
**Interview angle:** Why is action alignment offset by one physics frame?

## 2026-10-19 23:45 - Verify exact factual actions and conservative native rule coverage
**Type:** result / scoring correction
**Stage:** Navigation M5 data / M6 evaluator

**What happened:** Full72-source audit initially reports36 final acknowledgments absent from the camera trace. These are the final destruction/ack physics tick, present in the complete native recording after the last RGB/world row. The decoder now distinguishes pose-matched trace frames from contiguous native tail frames. Re-audit72/72 succeeds, all19461 control acknowledgments match within1e-5,0differences; all pose alignments unique, no guessed stationary offset. Both first failure and corrected audit/source receipts preserved.
**Coverage issue:** Actual stop reference records the same sign on approach lane19 and internal junction lane783. Counting actor/lane pairs would double this encounter. The final audit now maps session-local actor IDs to unique physical inventory positions and merges lane pairs into a distinct sign/mission lower bound. Repeated visits are also merged and explicitly not claimed as exact per-visit rate denominators. Actual30red/30stop and all5stop locations must be observed, never imputed from planned routes; missing missions stay in36 denominator. No native criterion return, benchmark route, reward or driver changed.
**Verified by:** outputs/navigation_executed_action_audit_v2_20261019/assessment.json; exact official0.9.15 control/pose format sources under navigation_recorder_control_spec_20261019. Five control/recorder tests and14 final/opportunity/registration tests pass. README/CLAUDE/handoff now distinguish130comma pilot contacts from0ResNet pilot contacts.
**Lesson:** Rule-opportunity granularity needs a physical audit as carefully as collision numerators.
**Interview angle:** Why is a sign observed on two lanes not necessarily two independent rule encounters?

## 2026-10-19 23:59 - Discover actual-pose split overlap and register complete fair-study repair
**Type:** bug / scientific correction / decision
**Stage:** Navigation M2/M4/M5 data boundaries

**What happened:** Full committed-RGB pose audit finds training/validation overlap at Town02/76 and349. TRAIN route85 has five initial RGB frames on349, despite metadata declaring76; validation route76 declares349 but has five initial frames on76. Two other TRAIN episodes use76. This is an actual source-split defect, not just the previously disclosed long-route evaluation exposure. Prior metadata-only separation was insufficient. Old66 fits, random control, failed driving pilots and continuation remain preserved; do not claim clean location-held-out label efficiency from them.
**How we handled it:** Keep every original13validation episode/2822timestamps and array segments byte-identical. Drop complete TRAIN routes48/68/85 and separately contact-bearing Town10HD/250 from clean imitation; all underlying data retained for factual prediction with correct labels/censoring. New55TRAIN episodes/15712timestamps,18534total. Every validation array comparison passes. Recompute same-algorithm nested budgets:549/1516/3857/15712 actual labels (nominal5/10/25/100%). Reserve union of actual validation locations for future research fitting; global frozen benchmark assignment/route/denominator/reward unchanged.
**Registered repair:** All60 main cells, five no-color controls, nav/color and random diagnostic rerun fresh; exact unchanged original2000-update loop, initialization/loss/precision/five validation points/selector. CARLA-JEPA restarts from originalcomma checkpoint for10000updates on corrected55episodes; no contaminated head or continuation inherited. Queue after native references and factual branch pilot. Bounded regenerable encoder features now may live in temporaryRAM with full source/recipe/hash receipts and owned cleanup, preserving25GBdisk reserve. Main mathematical encoder/training paths unchanged. Original nativepilots finish as historical implementation evidence, not repaired-study qualification.
**Additional factual sources:** Repaired dynamics inventory69episodes/19690timestamps includes3671 noisy/failed/assisted timestamps absent from imitation; all physical geography/control/recorder checks audited. Whole-location dynamics-validation split registered before model updates. Four contact-bearing episodes remain prediction data; no native fault filtering. Cost windows are being prepared with ordered actual actions and explicit censoring; there are no positive red-crossing examples, so these data alone cannot qualify a learned red-risk head.
**Verified by:** reports/navigation_spatial_actual_geography_20261019.json; navigation_m4_geography_repair_protocol_20261019.md; outputs/navigation_spatial_data_repaired_20261019/request/manifest/actual_training_guard.json; navigation_repaired_study_20261019 request/source.18encoder/pretraining tests and12action/RSSM/boundary tests pass.
**Lesson:** Validate actual camera-frame geography before forming clips, fitting any encoder or claiming held-out performance; never repair a split by removing unfavorable validation cases.
**Interview angle:** Why repeat a finished fair comparison instead of dropping five frames from validation?

## 2026-10-19 23:59 - Register outcome-blind memory geometry repair
**Type:** decision
**Stage:** Navigation M3 physical setup, no student outcomes

**What happened:** Among27 strict-TRAIN signal approaches, only Town02 routes79/80/82 satisfy projected all-view absence at the fixed3m legal stop pose. Select first two distinct eligible junctions,79/242 and80/132, before new native/student outcomes. Preserve invalid Town01/33. The two memory locations are both Town02, disclosed instead of claiming two-town memory coverage. Other rule families retain Town01/Town02. Exact camera/prefix/phase contracts and72 eventual tests remain.
**Verified by:** navigation_memory_setup_repair_protocol_20261019.md, four new case contracts pass. Route80 memory case SHA exactly matches the previously queued Town02 case, allowing reuse of its native reference rather than re-running a favorable outcome. Actual physical certification and students still pending.
**Lesson:** A still-uncertified controlled setup may need geometry repair, but its old failure and reduced geographical coverage must remain visible.

## 2026-10-20 00:11 - Complete frozen vehicle comparison and factual target windows
**Type:** result / implementation
**Stage:** Navigation M5, no planning promotion

**What happened:** Stronger original COCOFPN completes all165 fixed native snapshots. Same RGB/frame hashes and0.35/10m gates:54/58 nearby origins matched versus SSDLite15/58. Median matched origin error1.414m versus1.606m; CPU p9510782.9ms versus82.0ms per threeviews. FPN also has382 unmatched detections (209person/87truck/43car/40bus/3bicycle), so do not call it an accurate planner or treat every unmatched hypothesis as a verified false positive. Static props/parked objects and poster-like detections require a distinct audit. No GPU runtime claim or local CARLA labels.
**Factual targets:**69repaired strict sources produce3800windows(2845TRAIN/955whole-location validation),157positive collision-horizons,zero positive red horizons. Past-only RGB encoder cache is queued after corrected67study using its clean CARLA continuation. Native control ordering, masks and actual progress targets are explicit; no survival label after safety intervention or unknown tail. All69source recorders have zero handbrake/reverse frames; guard now censors such unsupported future controls. Source/cost/label arrays preserved, no dynamics updates yet. RSSM/JEPA now also accept declared35navigation/ego values; future targets remain loss-only. Per-head masks can keep known collision/progress while red labels are unknown.
**M2 next countermeasure registered:** Separate full-label fact-student protocol adds light/presence/distance/hazard training heads to the camera recurrent state, without feeding GT to inference. Two prospectiveCNN/ResNet engineering candidates, same2000updates and fixedselector; not integrated, fitted or folded intoM4 low-label curves. One gradient/boundary test passes.
**Verified by:** reports/navigation_vehicle_perception_comparison_20261020.md/json; outputs/navigation_dynamics_windows_20261019; same-frame source hashes, factual censor tests and247navigation/encounter tests(1skip), plus later focused tests. Final coverage also requires the specifically declared five mapped stops and actual per-mission target-stop observation, not any five runtime stops. No final route has been driven.
**Lesson:** A detector availability gain can coexist with unusable latency and many unmatched hypotheses; safety still requires temporal and closed-loop evidence.
**Interview angle:** Why is54/58 not a claim of93% object-detection recall?

## 2026-10-20 00:31 - Finish historical encoder pilots and24 physical references
**Type:** result / execution limitation
**Stage:** Navigation M3/M4 historical evidence

**Historical encoder pilots:** Same full-label seed42 checkpoints/sixcases,0overrides. Comma2/6arrivals,0native-rule-safe,130recordercontactstarts; CARLA1/6,0rule-safe,8contactstarts including one incomplete interrupted recording(lower bound); DINO5/6,2rule-safe,1contact; CNN1/6,1rule-safe,6contacts. CNNlast route143 is resource-censored by the original900s candidate wall cap; keep it unknown/failed in6planned denominator, not a settled behavioral failure or a retry. Reused ResNet6/6arrivals,4rule-safe,0contacts remains baseline. The source split defect applies to all oldmodels; no promotion or clean label-efficiency claim. Source-frozen raw outcomes preserved. The continuation terminal print emitted an unnecessarily huge recorder payload; future CLI now prints compact totals, with full data only in artifacts.
**Physical references:** All24expert setup arms complete,0unfiltered contacts, source/frame/actor/control/video audits pass. Independent native20Hz actuation audit checks every next-tick command in all24recorders:0differences,0missingacks. Town02/80redmemory nowactuallyholds8s with every applicable housing outside allthreeviews; pairedgreen completes. Still expert-only, no student claim. Invalid Town01memory remains; separate Town02/79repair awaits nativeverification. Yellow pair actualspeeds are approximately5m/s in botharms despite6m/srequested; report measuredspeed and distance, never label exact6.
**Verified by:** reports/navigation_historical_encoder_pilots_20261020.json; outputs/navigation_rule_references_20261019/records.json and navigation_rule_actuation_all24_20261020.
**Lesson:** Distinguish process/resource censoring, physical setup validity and learned behavior; none can stand in for the others.

## 2026-10-20 00:31 - Correct manual collector's wrong-map reload and recover normal owned cleanup
**Type:** mistake / bug / recovery
**Stage:** Navigation M5 branch collection lifecycle

**What happened:** The first standalone branch collector assumed the CARLA command-line town was loaded, then called reload_world. Actual startup remainedTown03 althoughTown01 was requested. Route planning therefore usedTown03; actualgeographyguard detectsjunction1135 on the firstcamera tick and aborts before any RGBsample/shard commits. No model/student inference or training sees the failedscene. Metadata writer/rawnativeattempt preserved as failed, never predictiondata. Normal serverSIGTERM cleanup times out20s and masks the primary error in error.json; original traceback retains both errors. Dependent67study/feature/GPU queues correctlyfail before updates.
**How we handled it:** Newcollector explicitly load_world(Town01_Opt), verifies actualtown and recordsXODRidentity before actor setup. Touch failedqueueSTOP. Verify savedownedPID36923/fullcommand/user/processgroup/creationtime; boundedRPCsync/tick recoveryfails. One STOP-gated retry of NORMAL SIGTERM exits in.6s; noSIGKILL orforce escalation. Allreceipts preserved. Hostanchoredprocess/GPUcheck confirmsfreebefore freshv2attempt and complete newdependencychain; oldfailedrequests remain intact.
**Verified by:** navigation_branch_pilot_20261019 startup_map/partition_violation/cleanup_sync_release/cleanup_tick_cue/normal_cleanup_retry.json; newnavigation_branch_pilot_v2_20261020 source/request/world_identity.
**Lesson:** A requested simulator map is not its actual map. Verify identity before planning or creating a labeledscene; preserve the primary exception when cleanup also fails.
**Interview angle:** How did the geography guard prevent a wrong-map infrastructure bug from contaminating training?

## 2026-10-20 00:43 - Finish factual branch pilot and independent closed-file audit
**Type:** bug / result / limitation
**Stage:** Navigation M5 collection, no model fitting

**What happened:** Second attempt verifies Town01 but times out on camera frame279 after278;0committed samples. Every-physics-tick manual camera cadence plus10braked warmup ticks fixes bootstrap. Third attempt completes6rollouts/480frame-exact timestamps. All960native next-tick controls and480lossless RGB pixel hashes pass independent audit; all6closed recorders complete. Go/brake/wait-go contact starts are2/0/2 in both repeats,8total, with post-contact safety tails censored.
**Cause:** Manual sensor bootstrap/cadence, then a separate asynchronous recorder-flush race: collector reads immediately after stop_recorder and reports incomplete final frames/provisional hashes. Preserve original outcomes; independent closed-file audit contains final hashes/completeness. No rerun to replace outcomes. Observable vehicle poses/velocities match, but frozen-light elapsed timers differ by up to.2675s, above registered.051s; deterministic counterfactual calibration remains unqualified.
**How we handled it:** Retain both failed attempts and their source/cleanup evidence. STOP-gated ownership-verified normal SIGTERM retry releases second hung cleanup, no forcekill. Successful third server exits normally. Audit source/results at navigation_branch_audit_v3_20261020; no imitation eligibility or model updates.
**Lesson:** A successful collection process can still fail replay comparability; audit closed evidence and simulator clocks independently.
**Interview angle:** Why do repeated collision outcomes not prove deterministic counterfactual replay?

## 2026-10-20 00:43 - Recover exact checkpoint storage and start corrected study
**Type:** blocker / recovery / execution
**Stage:** Navigation M4 actual-geography repair

**What happened:** After pilot commits, study v3 refuses startup at25.93GBfree versus unchanged26GBpreflight; zero optimizer updates. Read-only SHA inventory finds12additional groups of byte-identical historical checkpoint copies. Independent-inode Btrfs COW replacement preserves all paths, bytes and mtimes; tiny write-isolation test passes. After filesystem sync, actualfree rises1.13GB to27.06GB. No original RGBcache, source video or checkpoint deleted;25GBreserve unchanged.
**How we handled it:** Preserve failed roots/dependency outcomes. Fresh source-frozen studyv4 starts from originalcomma weights on corrected55TRAIN/15712timestamps, unchanged13VAL/2822; actual optimizer steps now advance. No source modification to registered67fit loop. Ledger reconciled to actual running root. Receipt navigation_checkpoint_cow_more_20261020 and all-weight exact-hash inventory retained.
**Lesson:** Measure committed physical storage savings; logical duplicate size and immediate filesystem counters need not agree.
**Interview angle:** How can training storage be recovered without deleting scientific evidence?

## 2026-10-20 00:43 - Implement prospective cold-start/fact comparison separately from M4
**Type:** implementation / decision
**Stage:** Navigation M2, not yet fitted or deployed

**What happened:** Separate trainer implements four predeclared candidates: CNN/ResNet each plain/fact,2000updates,seed42,burn-in0. All fresh shared driving parameters match exactly between paired arms. Semantic arrays align18534corrected timestamps and enter targets only; no privileged fact enters inference. Full-label semantic supervision is explicitly counted; original M4burn-in8/matrix remains unchanged.
**Verified by:** scripts/train_navigation_fact_student.py initialization/target-boundary checks; prospective protocol and separate labels manifest. Native safety, deployment and semantic usefulness remain untested.
**Lesson:** Paired plain/fact arms isolate semantic supervision from a simultaneous cold-start training change.

## 2026-10-20 00:51 - Queue paired camera training and bind rule certificates to native evidence
**Type:** implementation / execution / evaluation integrity
**Stage:** Navigation M2–M5, no new driving promotion

**What happened:** Three fact-student checks pass, including identical fresh driving parameters in both encoder pairs and a first-observation gradient reaching cold-start targets with burn-in0. Four2000-update candidates queue after corrected67study, followed serially by factual encoder cache, fixed165-frame GPUFPNprofile and two missing Town02/79memory references. Town02/80nativecases reused by exact caseSHA. No duplicate GPU/simulator stage.
**Hardening:** Student controlled-rule entry now requires schema-v2 evidence-bound setup certificates: exact two case hashes, physical evidence hashes and all declared runtime/config source hashes. Sixunchecked booleans alone no longer allow native student execution. Adversarial test changes partnerbytes and rejects stalecertificate; six suite/contract tests pass. No certificate has yet been issued. Separate certifier derives checks from native physical references/actuation/geography and reports actual yellow handoff speeds rather than requested6m/s. Three existing native-family dry checks pass; full memory pair still awaits native verification.
**Progress:** CorrectedJEPA passed2000updates with finite diagnostics; no downstream fit or student rule outcome yet. Watcher now shows serial GPUprofile/memory queues and distinguishes factual cache work from driving progress. Corrected storage watcher to derive22COWreplacements directly from receipt instead of a hand-entered count.
**Lesson:** Bind setup validity to actual evidence and source identity so a changed case cannot reuse a stale certificate.
**Interview angle:** Why can a file with six true validity flags be insufficient to release an experiment?

## 2026-10-20 01:13 - Finish corrected JEPA continuation and prepare factual ensemble fits
**Type:** result / implementation / pre-fit correction
**Stage:** Navigation M4/M5

**What happened:** Clean55TRAIN continuation completes10000updates in1768.34s. Fixed validation prediction loss.33923→.06720, within-clipvariance.39552→.11694, between-scenevariance.25466→.12158. Finite, nonzero variation; not driving improvement. Corrected67study now extracts/uses matched frozenfeatures, first ResNet full-label fit advances.
**M5 audit:**3800windows contain108TRAIN positive collision-horizons in24windows/3episodes,49VAL in10windows/1episode. Ordinary seed42 episode bootstrap omits all positive episodes. Before any dynamics update, register TRAIN-contact-stratified whole-episode bootstrap, preserving per-stratum episodecounts; all three seeds now24bootstrap-positive windows. Two model arms share exact batches. Zero red positives: mask/freeze red-output training, no deployment qualification. Six2000-update JEPA/RSSMmembers queue after native setup/cache/profile; futuregrids/outcomes remain targets only. CPU finite loss/backward, target-boundary/paired sampler tests pass.
**Storage:** Kernel-verified COWrange sharing handles646pairs/1.137GBlogical bytes across52historicalfiles; whole-fileSHA/mtime and independent writes preserved. Actual committed free gain only116.4MB, not logical1.14GB. Preserve inventory/source/receipt.
**Deployment:** New fact architecture dispatch passes actual isolated-worker IPC test with only SensorPacket fields, aux predictions exposed but overrides/control rule layer off. Current original M4source remains unchanged. STOP and replace still-unstarted memory queue with freshv2source snapshot after adding fact-checkpoint metadata; no physical cases duplicated.25focusedtests and8furtherIPC/sampler tests pass.
**Lesson:** Test sample availability before fitting, and report loss reduction separately from policy behavior.
**Interview angle:** Can an ensemble member accidentally see no hazardous training examples despite their presence in the dataset?

## 2026-10-20 01:23 - Audit20non-memory arms and queue corrected native camera evidence
**Type:** implementation / verification / decision
**Stage:** Navigation M2/M3/M5

**What happened:** Independent certifier dry checks all20non-memory references with0errors: native acknowledgments/actual TRAIN geography/physical signal identities/common-red background and measured handoffs. Native setup functions remain AST/exact-hash equivalent; known non-actuation differences are student gate/metadata and unused manual camera constructor. No certificate issued before both repaired memory sites complete. Cross-traffic traces show live actor crossing the ego approach after green; measured handoff/actor speeds disclosed, not relabeled exact requested values.
**Next evidence:** Source-registered5candidate queue follows the unchanged six-case pilot for all4cold-start/fact variants plus correctedM4primary. Both primary selectors freeze from offline scores before driving; every contact and missing/resource-censored case remains. No pilot result yet. Real99frame TRAIN ResNet/fact forward-backward is finite(loss10.2885),0optimizerupdates; actual isolated fact-policy worker passes with overridesoff.44focused integration tests pass across checkpoint/IPC/action/split/RSSM/native final/rule boundaries.
**Pre-run correction:** RSSMtraining now samples its prior Gaussian; validation/deployment use prior mean. The first still-unstarted dynamics queue is STOP-completed and preserved; freshv2source/request queued before any update, protocol explicit. Original67study source unchanged. Documentation current banners now identify the real corrected study and remove stale running/native-only counts; historical journal/results retained.
**Resources:** Measured old1.23km long-drive directory occupies284MB; only~1.6GBabove25GBfloor remains. Asked asynchronously for additional storage for full M5/M6, while approved training/audits continue. No old artifacts deleted and goal remains active.
**Lesson:** Qualify actual physical scenarios and freeze selectors before driving, even when offline experiments have completed.

## 2026-10-20 01:27 - Measure shared ego physics limitations and register clock repair
**Type:** result / limitation / implementation
**Stage:** Navigation M5, no planner promotion

**What happened:** Shared20Hzbicycle/actuator baseline fits11641TRAIN transitions once, with no validation search. Throttle6.662/brake4.348/drag.0343/accelbias-2.0(boundhit), steerangle.6491rad,wheelbase2.8m. On859factual4sVALwindows, final position MAE2.4947m/p9510.789m, speedMAE.8266m/s. One-step accelerationMAE3.3625m/s². Includes contact/assisted forecast windows; fitting exclusions only identify physics, never remove headline contacts. This model is unqualified for accurate candidate-action prediction. Two order/causality/bounded-actuator tests pass.
**Replay repair:** Separate prospective protocol keeps completedv3pilot/8contacts/failedtimertolerance. New six-rollout pilot queues after corrected native camera pilots: reset each lightgroup once beforegreen/freeze, require elapsed<=.051s, usecorrectedactualgeographyguard, and explicitly acknowledge recorderstop before stable complete-file hashing. Same route/seed/prefix/actions and unchanged replaytolerances, no silent relabel. All raw cleanup/native events retained;480RGB/16020Hzactionframes perrollout unchanged. Futureclosed-file audit now derives timing/hash limitations from actualevidence instead of hardcoding oldfailure text. No repaired physical result yet.
**Progress:** CorrectedResNetgroup13/13fits complete withfeatures evicted onlyafter verification; DINOcache begins. All rawsources/recipes/trainedheads remain. Additional M5/M6storage question pending; training continues above25GBfloor.
**Lesson:** Desired trajectories, executed commands and actual physics are distinct; measure their mismatch before comparing planners.
**Interview angle:** Why can a simple ego actuator model be inadequate even with a reasonable steering-angle fit?

## 2026-10-20 01:29 - Calibrate rule objectives against measured expert behavior
**Type:** result / clarification
**Stage:** Navigation M3, expert-only preliminary grid

**What happened:** Separate scoring-only function implements predeclared memory/amber/stop/departure objectives without treating setup validity as behavior.18/20non-memory reference arms meet their conditions; both close-amber experts stop safely instead of clearing within amber, so that maneuver objective fails despite0contacts/violations. Do not relax the objective after seeing this result or call safe stopping illegal. Actual yellow entry speeds~5m/s remain distinct from requested6m/s. Original Town01memory rawholds still satisfy behavior but have INVALID physical memory setup, therefore never pass. Full repaired24-arm certification/student72trials still pending.
**Verified by:** reports/navigation_reference_rule_behavior_20261020.json and exact archived scoring source; tests reject missing certificate, short red hold, contact/incomplete recorder, green staying-stopped and close-amber stay-stopped objective. No expert failure filters a valid setup.
**Correction to previous 2026-10-20 entry:** The clock-repair dataset is480RGBtimestamps total,80perrollout, with160measured20Hzcontrols perrollout. The earlier sentence saying480RGBperrollout was a journal wording mistake; raw dataset/request counts remain unchanged.
**Lesson:** Keep physical validity, maneuver objectives and safety violations as distinct measurements.
**Interview angle:** Can a rule challenge fail while the expert still drives safely?

## 2026-10-20 01:48 - Resume exact checkpoint after external tool interruption
**Type:** blocker / recovery
**Stage:** Navigation M4 execution

**What happened:** External turn interruption stopped the study and all seven queued parents. Host anchored process/GPU checks and missing session handles verified actual termination before restart.19 completed fits retained; interrupted DINO10%seed42 checkpoint1800, validation1500, selected1000; raw log reached1885.
**How we handled it:** Preserve original1885-row log/source/progress/review and all unstarted queue roots. Restore committed1800-row active log and optimizer/RNG state, then identical-source resume.85 known logged uncheckpointed updates replayed; possible unlogged work unknown. All seven recreated immutable request hashes match their originals exactly; no completed fit or settled driving outcome repeated.
**Verified by:** outputs/navigation_interruption_20261020_0134/recovery.json and restarted_queue_identity.json; live study subsequently passes26/67 fits.
**Lesson:** Recover committed optimizer/RNG state and preserve extra logs; a stale running file is not evidence that a process is alive.
**Interview angle:** How do you distinguish resumed scientific work from duplicated training after interruption?

## 2026-10-20 01:48 - Select external partition, then investigate user's local24GB alternative
**Type:** decision / blocker
**Stage:** Navigation storage, M2–M6 execution

**What happened:** User authorizes choosing additional storage. Linux root/home/mnt share the same204.1GB Btrfs filesystem. Choose existing1.4TB Games NTFS partition /dev/nvme0n1p4, UUID102C30572C303A54; ordinary desktop mount requires OS authentication and ends NotAuthorized. No format/repair/force mount or existing-file deletion attempted.
**New steering:** User asks whether remaining work can fit available24GB. External mount is no longer required for the proposed path; profile compact showcase-only H.264 on identical64clean TRAIN frames from each of five towns. Keep source RGB, native controls/contacts/scoring and existing artifacts unchanged. Original25GBreserve remains active pending concrete local-capacity decision.
**Profiling mistake:** First CPU profile assumes every mixed-origin episode has samples; fails KeyError before encoding. Preserve failed source/error, use shared end-start length in freshv2profile. No training data or optimizer affected.
**Verified by:** outputs/navigation_compact_storage_profile_20261020/error.json; outputs/navigation_compact_storage_profile_v2_20261020/request.json. scripts/provision_navigation_storage.py prepared but not launched.
**Lesson:** A large unmounted partition is not usable storage until normal OS authorization succeeds; measure a compact alternative before changing a reserve.

## 2026-10-20 01:51 - Measure physical compression and prepare14GB local plan
**Type:** result / decision
**Stage:** Navigation M2–M6 storage

**What happened:**320exact clean TRAIN triplets from5towns compare preview codecs:5fps/crf28/medium6.36MB versus10fps/crf23/veryfast15.29MB,58.4%smaller. No original videos, input pixels or scoring changed. Fresh explicit writes to actualBtrfs show99.68MBworldtrace costs~12.53MBphysical,16.66MBnative5.64MB,159.15MBvideo159.16MB; three world copies agree within29kB. Every synthetic copy SHA verifies and only synthetic copies removed.
**Plan:** Cap additional retained artifacts at14GB, proposing10GBfree floor. Keep all native recorders/traces/failures; allM3videos; forM6 preselect routes0/4/8 seed3031 identically in expert/student for full10fpsshowcase video. All72missions retain native evidence; fixedsubset gives~3.46GBestimatedM6 versus larger all-MP4 plan. Keep5GBM6allocation and2.5GBcontingency; pace/traffic remain uncertain and hardlimitsstopownedjobs honestly. No evaluation count/route/seed/reward/input changes.
**Open:** Asked user explicitly about replacing prior25GBfloor with10GB for future source generations. Active study/setup sources and25GBguards remain unchanged. CurrentGames mount failed OSNotAuthorized; not an automatic tool-approval rejection.
**Verified by:** reports/navigation_local_storage_plan_20261020.md/json; outputs/navigation_compact_storage_profile_v2_20261020; outputs/navigation_physical_storage_profile_20261020.
**Lesson:** Budget actual filesystem bytes; prioritize complete scientific evidence and predeclared showcase videos over recording redundant MP4s.
**Interview angle:** How can storage be reduced without filtering inconvenient outcomes?

## 2026-10-20 01:55 - Approve compact local storage and activate fixed-baseline guardian
**Type:** decision / implementation
**Stage:** Navigation M2–M6 execution resources

**What happened:** User explicitly approves the new14GBstorage plan and10GBfree minimum. Freeze that approval/plan hash and register net-filesystem free-space baseline in outputs/navigation_compact_execution_20261020/storage_guard. Watch the eight owned stage roots; guardian only touches theirSTOPfiles on actual reserve/budget exhaustion, never signals a process or deletes evidence. Unrelated writes count conservatively.
**Source integrity:** Existing registered study/setup/collector sources retain their25GBguards. Defer config/recording implementation to a fresh runtime generation after the current source-pinned stages; changing them while queued would invalidate physical-reference identities. New budget already active, source snapshots preserved.
**Verified by:** storage_guard/request.json, watched_roots.json and live heartbeat; approved reports/navigation_local_storage_plan_20261020.json.
**Lesson:** Apply a changed resource policy without silently modifying a running experiment's registered source.

## 2026-10-20 02:00 - Prepare full72-trial camera rule execution and independent failure grid
**Type:** implementation
**Stage:** Navigation M3, no student trials yet

**What happened:** New runner accepts only24certified paired cases across sixfamilies/two locations/threefixedseeds, freezes camera checkpoint/source/selection before execution, disables overrides, registers its ownedroot with the storageguardian and preserves every attemptedcase. Identical-source resume skips completed cases; interrupted attempts retain unknown outcomes rather than quietly rerunning them. Nativecontrols receive a separate recorder-based next-tick audit after collection.
**Assessment:** Bind actual archived sources, checkpoint, case/certificate/evidence hashes, synchronizedRGB/video and independently acknowledgedcontrols. Missing/invalid cases remain failed in72; duplicate physicalidentities cannot fill the denominator, expert summaries cannot count as students, and even orphanattemptrecorders contribute all known contacts. Produce6family×4arm/location pass/fail/unknown grid. No certificate, student result or drivinggate claimed.
**Verified by:**10focused rule/suite/assessment tests pass; scripts/run_navigation_rule_student.py and scripts/assess_navigation_rule_student.py lint clean. Current registered67study remains source-identical,39fits complete.
**Lesson:** Count planned interventions and every attempt independently of successful recorder/summary availability.
**Interview angle:** Why can72unique output names still be an invalid72-trial rule evaluation?

## 2026-10-20 02:14 - Diagnose actuator delay and compare a stronger frozen ego baseline
**Type:** result / limitation
**Stage:** Navigation M5 candidate physics

**What happened:**52TRAIN-onlysources/11233eligibletransitions compare fixedcausallags0–8 using leave-one-training-town-out regression. Lag2ticks(.1s) lowers accelerationCVMAE2.734→2.327m/s²; interceptstillhits-2bound. Snapshot position/velocity agree within.0326m/smean/.0953p95, so gross velocity timestamp mismatch is not supported. No VALsource opened forlagselection.
**Stronger baseline:** Pinned expert already contains frozen polynomial speed/slip-anglemodel. Register vendorlag0 and TRAIN-chosenlag2 before evaluation, keep both. Same2554TRAIN/859VAL foursecondwindows, includingcontacts/assistance and actualpastcommands. VALmeanposition initial2.495m, vendor0lag2.248m, vendorlag2=1.859m; p9510.789→7.995→6.691m. Still unqualified for actualalternativeactions; noplannerdrive or riskcalibration. Fractionalbrake limitation reported; vendorcoefficients are externalprior, not fittedlocally.
**Verified by:** outputs/navigation_actuator_timing_20261020 and outputs/navigation_vendor_ego_20261020, exactnative/source identities and perwindowarrays. Source-styling after archivedcopy does not change the measuredmodel.
**Lesson:** Check existing strong baseline physics and causalresponse timing before attributing errors to a learned worldmodel.
**Interview angle:** Why did a model using correctly acknowledged controls still predict ego motion poorly?

## 2026-10-20 02:14 - Stage approved compact runtime without changing active experiment sources
**Type:** implementation / decision
**Stage:** Navigation storage, future M3/M6 runtime

**What happened:** Prepare exactfourfile patch for10GBfuturefloor and fixedsixM6showcasevideos, retainingallnativeevidence and allnonfinal/M3videos. Pure selection tests prove sixof72videos and identical expert/student, no outcome filtering. Seven staged final/recordingtests pass; firstharnessattempt failed4because functiondefault captured the stagingroot, corrected only testharness and preserved failure receipt. Runtime patch unchanged.
**Compatibility:** Physical certifier now allows ONLY explicitly approved runtime.minimum_free_gb25→10; all other camera/physics/config fields must match, with approvalSHA/native before-after hashes disclosed. Test rejectsFOV/physicsdrift. Also require originalregistered reference manifest/recording hashes; no source validity from copiedbooleans. Eleven focused rule/resource tests pass.
**Execution:** Activation queue waits for all eight currentregistered stages and actualownedworkers to terminate, then acquiresnavigationlock, checks exactbeforebytes, appliesreviewedpatch and runs requiredtests. It is queued, not yetapplied. Guardian covers it too; nojobs/*.shchanged.
**Verified by:** outputs/navigation_compact_generation_20261020; patches/navigation_compact_storage_20261020.patch passes gitapply--check; live outputs/navigation_compact_activation_20261020.
**Lesson:** Preserve old source generations and make resource-only compatibility exceptions exact and auditable.

## 2026-10-20 02:21 - Restore exact shared raw RGB and recheck20physical references
**Type:** implementation / result
**Stage:** Navigation M4 efficiency / M3 setup verification

**What happened:** Missing ownedtmpfsRGBfile left its registeredlogicalsymlink broken; eachscratchworker reconstructed the same2550supplementframes. Restore846028928bytes once from unchangedlosslessshards in87.51s, verify exact originalSHA9366be26…98beb2, publish atomically read-only at the same ownedtarget. Existingfirstworker/privatearray untouched; laterworkers now usemmap and skip reconstruction. No backbonefeature caching, label/source/optimizer/update/mathematics change. OriginalM1RGB/front caches remain.
**Verified by:** outputs/navigation_shared_ram_rgb_restore_20261020/receipt.json and consumer_check.json prove fulloriginalSHA, read-onlymmap and reconstructionbypass. All active67study registeredsourcehashes remain identical. No measured walltime-savings claim before laterfits finish.
**Physical audit:** New manifest/recording identity checks and exactresource-changeequivalence re-audit all20non-memory nativearms with0errors. No certificates issued before both repaired memory sites and fresh runtime generation. Evidence outputs/navigation_nonmemory_recertification_dry_20261020/assessment.json.
**Lesson:** Reuse byte-identical regenerable raw pixels across workers; never cache trainable CNN features or quietly alter the experiment to save time.

## 2026-10-20 02:26 - Catch missing M6 budget registration before activating runtime
**Type:** bug / recovery
**Stage:** Navigation compact runtime, no final missions run

**What happened:** Review of the staged fourfile change shows futureM6root was not registered with guardian, and rootprogress starts only after its first12route seedgroup. Thus a long firstgroup would not receive STOP at the newcap. Before activation, touch oldactivationqueueSTOP and verify normalterminalexit; zero sourcefiles applied, traininguninterrupted.
**Fix:** Freshv2fivefilegeneration addsM6BudgetWatch, initialprogress and ownedrootregistration. Localwatcher stops throughSTOP on stale/missing guardian aswellas actualspace exhaustion; tests preserve retaineddata and rejectforeignroots. M3runner now uses the same guard. Finalcandidate provenance additionally pins actual run_navigation.py/run_navigation_final.py/run_encounters.py entrypoints.
**Verified by:** Original unappliedgeneration/STOProot retained; nine staged final/recording/budgettests pass, exactpatch gitapply--check passes. Freshsource-frozenactivationv2 queues after the same eightstages; helperhashes checked before apply. No benchmark route/seed/score/agent input changed.
**Lesson:** A resourcefloor and daemon are insufficient unless each long-running owner registers and can detect guardian loss.

## 2026-10-20 11:39 - Complete corrected offline studies; preserve failed native cold start
**Type:** result / bug / recovery
**Stage:** Navigation M2–M5, storage execution

**What happened:** Corrected67/67 encoder fits, all4 paired cold-start/fact fits and all6 factual dynamics members completed overnight. Canonical M4 report: comma full-label mean score.3755 vsCNN.3874, DINO.4323, ResNet.4477, CARLA-JEPA.4628; no safer-driving claim. CNN-facts primary selected2000, score.38646; stop recall55%, falseholds13.92% on its distinct burn-in0 evaluation.
**Failure:** First plain-ResNet native pilot records3/6arrivals,2/6native-rule-safe,0strict,0recordercontacts,1red/1stop. Route86 times out on the first policy reply after10s, with empty traces and no finalized profile/video. Cause of the slow first forward remains open. Assessor then raised FileNotFoundError instead of reporting missing evidence.
**How handled:** New assessor marks incomplete artifacts as evidence failures with all6cases retained. Available5videos decode; corrected audit is false. Original error/raw files preserved. Fresh continuation runs ONLY4unstarted frozen candidates, byte-identical runtime/checkpoints/cases/seed; no retry of crashed86. Nine focused inference/continuation tests pass; lint passes.
**Storage:** Host process absence verified before resuming same approved guardian baseline; no new budget allowance.14GB cap/10GB reserve protected. Fresh unstarted clock-repair queue and exact storage activation wait serially; current source-pinned runtime still25GB.
**Lesson:** A crashed driver is a measured failure even when its writer never finalized the episode artifacts.
**Interview angle:** How do you recover an evaluation queue without erasing a failed first case?

## 2026-10-20 11:39 - Export bounded causal dynamics inference and executable candidate controls
**Type:** implementation / result / limitation
**Stage:** Navigation M5

**What happened:** Exported two independent3member JEPA/RSSM ensembles after verifying all12000actualupdates, selected checkpoints, five validations/member and encoder/source hashes. Inference accepts only past RGB tokens/current declared ego-route inputs/proposed80commands; RSSM uses prior only; untrained red output omitted. Seven inference tests pass. All outputs explicitly unqualified.
**Candidate implementation:** Frozen privileged-expert ego polynomial exported without fitting:256numerical comparisons maximumdifference0.0. Fifteen path×speed proposals become80boundedcommands each with2actual prior controls and frozen TRAIN-selected two-tick delay. Two tests plus5gridchecks pass. Overspeed input previously clamped instantly to8m/s; now bounded deceleration3.5m/s², unchanged desired8m/s cap. No deployed planner existed or changed.
**Limitation:** Factual validation is weak: JEPA seeds miss44/49,49/49,49/49 labeled positive horizons at.5; RSSM all49/49. Only one positive VALepisode and zero red positives. Low latent loss is not safe action ranking; no native/CV superiority or calibrated-risk claim.
**Evidence:** outputs/navigation_dynamics_ensembles_20261020; reports/navigation_candidate_controls_protocol_20261020.md; source modules/tests.
**Lesson:** Verify the exact executable action sequence and prospective risk evidence before using latent prediction loss to justify a planner.
**Interview angle:** Why can a low-loss world model still miss every collision?

## 2026-10-20 11:47 - Diagnose spatially blind JEPA risk path before further fitting
**Type:** bug / implementation / limitation
**Stage:** Navigation M5

**What happened:** The retained trained JEPA risk head reads context(tokens).mean(1), while spatial decoder attention never enters its cost output. Swapping encoded front/right view slots changes collision probability by at most5.96e-8 across all3selected JEPA members on fixed synthetic inputs; RSSM ordered flatten responds. This assignment invariance is architectural; not proof of physical scene equivalence or the cause of every miss.
**How handled:** Preserve all six fits/exports and add a separate untrained spatial JEPA v2: explicit camera/grid identity, attention-based scene summary, risk readout from predicted spatial hidden states. No training, active inference source edit, native controller or scoring change. Future fit needs registered branch support and matched comparisons; zero-red limitation remains.
**Evidence:** outputs/navigation_dynamics_spatial_diagnosis_20261020/assessment.json; reports/navigation_spatial_dynamics_v2_protocol_20261020.md.
**Lesson:** A spatial latent decoder does not make a pooled risk head spatially aware.
**Interview angle:** How did you test whether the world model preserves which camera an obstacle occupies?

## 2026-10-20 11:55 - Preflight all physical rule arms and queue unchanged student suite
**Type:** milestone / execution / result
**Stage:** Navigation M3/M5 infrastructure

**What happened:** All24physical arms/12pairs pass native-evidence preflight, including repaired Town02 out-of-view memory. No certificates issued under the old25GB generation. After exact compact activation, a source-pinned followthrough will audit closed branches, issue current-generation certificates and run all72controlled trials of the originally preselected CNN-facts primary. No native-outcome checkpoint reselection; both engineered/emergency overrides off. Behavior failures remain reported, not automatic promotion.
**Current driving:** ResNet-facts6/6arrivals,3native-rule-safe,2strict,1unfilteredcontact,1red/1stop; all6videos/frame/source audits pass. Four remaining-candidate queue now tests plainCNN; original failed firstResNet remains6case denominator.
**M5 result:** Exact paired window draws verified across all2000updates/seed in both arms. TRAIN-only constant-prior collision BCE.043152 beats all selected learned members(.1099–.2216). Always-zero4sBrier.010283 exposes imbalance; only one positive VALepisode. Report at outputs/navigation_dynamics_diagnosis_v2_20261020. No useful-risk claim.
**Mistake recovered:** First reporting-only baseline calculation multiplied maskedNaN outcomes by0, producingNaN and a refused JSON export. Original failed root/source preserved; sanitize with np.where before arithmetic; a focused test verifies finite metrics/TRAIN-only prior. Training already used torch.where, so completed fits unaffected.
**Verified by:**447full CPUtests pass/2skip; subsequent spatial4tests and baseline1test pass. All new/changed source lint passes.
**Lesson:** Certify physical setup separately from whether the teacher/student behaves correctly.
**Interview angle:** Why compare a collision predictor with an always-safe constant?

## 2026-10-20 11:59 - Retain planned pilot wall censoring and continue only unstarted candidates
**Type:** result / bug / recovery
**Stage:** Navigation M2 native evaluation and serial queues

**What happened:** PlainCNN reaches the unchanged900s candidate wall cap:1arrival,1native-rule-safe,0strict,0observedrecordercontacts,1red among5nativecases. Route133 is censored at94.38%;143 never starts. Keep both as failures in6, with unknown unobserved behavior. No retry, extra runtime or changed route/traffic.
**Cause:** Pilot assessor still rejected a partial native record set, stopping the parent and its unstarted dependent stages. A planned terminal budget condition was treated as an orchestration exception.
**How handled:** Report6planned/5observed separately, missing143unknown-failed, censoredlastcasefailed, missing evidence/auditfalse; all5videos decode and all recorders complete. Generalized continuation accepts ONLY an independently verified original900s wall termination for partial prior cases. Native/audit corruption remains fatal. New v2root executes ONLY the two entirely unstarted candidates with frozen weights/runtime/cases; downstream unstarted clock/activation/rule stages requeued with failed roots preserved.
**Verified by:** Native summary stop_reason=wall_budget and manifest max_wall900; reassessment_v2.json keeps6denominator, tests reject incomplete unverified histories and permit verified censoring. Anchored host check showed only the storage guardian before launch.
**Lesson:** Separate a prospectively bounded failed experiment from an infrastructure failure without converting unknown outcomes into successes.
**Interview angle:** Does a planned time budget allow rerunning a missing final case? No.

## 2026-10-20 12:22 - Reject primary CNN driving; preregister a bounded structured stop correction
**Type:** result / decision / implementation
**Stage:** Navigation M2, original M3study unchanged

**What happened:** Preselected CNN-facts finishes6nativecases:1arrival/1strict/1rule-safe,5unfilteredcontactstarts versus3nativecollisionevents,1red. Last143case is wall-censored at1.92%; all original outcomes retained. Frozen comma pilot follows. No checkpoint promotion.
**Diagnosis:** Route101 blocks30.06%: Red.770, within80m presence.260, distance44.05m, brake.662, all hazard probabilities<.043. Image shows open road ahead. Presence is range-qualified, so unlike my initial wording, differing Red/presence marginals alone are NOT a strict contradiction:449red/52green labels have absent within80m presence. Distances for absent targets are unsupervised.
**Countermeasure:** Before any M3student outcome, freeze parent and1000updates each for predicted-facts versus constant-TRAIN-mean causal stop residual. Zero-output initialization preserves parent; only17derived predictions+9parent logits enter it, no raw-route/speed bypass or engineered braking rule. Same55TRAIN/13VAL data, no new annotations; verify selected parent metrics and source RGB before fitting. Protected selector and explicit30min combined cap; no automatic deployment. Six meaningful construction/selector tests pass.
**Execution:** outputs/navigation_semantic_residual_study_20261020 waits until original72trial M3followthrough completes. No concurrent GPU use. Active sleep inhibitor verified. Automatic review first rejected launch as unapproved multi-hour work; read-only evidence from active user goal/CLAUDE.md standing authorization and script1800s cap justified an approved retry of the SAME command. No workaround or user approval request.
**Lesson:** Shared auxiliary perception losses do not force a controller to use its predictions coherently; test an explicit learned connection with a bias-only control.
**Interview angle:** How did you distinguish useful structured perception from merely adding auxiliary losses?

## 2026-10-20 12:57 - Activate approved compact storage after recovering a test publication race
**Type:** bug / recovery / milestone
**Stage:** Navigation resource generation

**What happened:** All8pinned stages completed and workers exited; exact5file compact patch applied,25→10GBreserve and predeclared final6MP4selection only. Its14test validation initially failed because the guard test observed STOP before the later audit receipt was published. Production intentionally stops first.
**How handled:** Preserve original patch/application/error/log and failing test source. Move receipt assertion outside BudgetWatch context, which joins its monitor; production helper remains byte-identical. Independent recovery verifies every exact approved afterSHA/helper,14tests pass(.19s), actual final entrypoint provenance bound; never reapply the patch.
**Evidence:** outputs/navigation_compact_activation_v4_20261020/applied.json; outputs/navigation_compact_validation_recovery_20261020/assessment.json. Same immutable14GBbaseline/10GBreserve remains; no artifacts deleted.
**Lesson:** Safety marker publication and asynchronous audit completion are different synchronization points.
**Interview angle:** How do you fix a flaky safety test without delaying the safety action?

## 2026-10-20 13:00 - Verify repaired replay and start all72unassisted rule trials
**Type:** result / milestone / execution
**Stage:** Navigation M3/M5

**What happened:** Repaired six-rollout branch audit passes every registered observable prefix/same-option comparison;480RGBpixel hashes/960actual next-tick controls verified,0differences,6closed recorders,8unfilteredcontacts(go/brake/wait-go2/0/2 eachrepeat). It remains static-lead TRAINinstrumentation, not reactive/calibrated planning.
**M3:**24source/evidence-bound physical certificates issued under the approved10GBresource configuration; only the resource floor differs from original references. The preselected CNN-facts72trial study now runs with engineered/emergency overrides off. Partial trial counts are progress, not success rates. No teacher outcome filters valid cases.
**Pipeline recovery:** Original unstarted followthrough/semantic roots failed only their validation dependency. Fresh v3followthrough/v2semantic roots preserve all prior receipts; semantic request/source/math is identical except dependency identity, independently compared equal. It registered before the first student trial. Sleep inhibition is active.
**Evidence:** outputs/navigation_rule_followthrough_v3_20261020: branch_audit, certified_suite and student; active execution ledger.
**Lesson:** Observable replay comparability releases a stronger experiment, not a calibrated model claim.
**Interview angle:** What exactly did the successful replay audit establish?

## 2026-10-20 13:00 - Queue prospective frozen-control branches and complete missing matched M4controls
**Type:** implementation / decision / scope correction
**Stage:** Navigation M4/M5

**M5:** New TRAINonly flag freezes all15shared candidate80-command sequences before handoff; executes3center-path options twice using only navigation/own speed/2actual prior commands. No replan from future ego/world. Post-contact privileged stops remain separately logged/censored. A new independent auditor reproduces plans from packet/native history and rejects pre-contact interventions or prefix mismatches.8candidate/construction and2prefix-audit checks pass. Collection queues after M3/semantic fitting; max1800s/~.35GB under guardian. Old closed collection remains untouched.
**M4 scope:** The original selection protocol requires all5matched full-label seed42drivers. The recent four burn-in0M2fact/plain pilots cannot substitute for the burn-in8M4heads. Queue remaining CNN/ResNet/DINO/CARLA pilots under original6cases/seed/900scap; reuse comma4/6arrivals/10contacts. Do not claim a completed matched native comparison yet. One existing no-color comma diagnostic follows separately(.34311 vsmatched color-on.35141 offline); it cannot replace the original primary or establish a significant one-seed gain.
**Freeze:** outputs/navigation_matched_native_completion_20261020 pins current inference source/config/checkpoints through all5newpilots. Do NOT edit src/navigation or registered audit/pilot scripts while these stages are queued/running; stage later inference adapters separately. No jobs/*.sh, benchmark/reward change, source outcome removal or push.
**Documentation recovery:** An f-string name error interrupted the first append after its storage entry; appended only the missing entries, preserving that earlier entry once. No experiment affected.
**Lesson:** Additional engineering variants do not satisfy the originally promised matched baseline comparisons.
**Interview angle:** How did you prevent auxiliary-label/cold-start changes from confounding encoder comparisons?

## 2026-10-20 13:11 - User approves versioned ego-only actuator feedback
**Type:** decision / approval
**Stage:** Future Navigation M5input interface

**What happened:** User explicitly selects “Approve ego-only actuator feedback (recommended)” for the staged proposal. Allowed fields only observation/action frame and ego steer/throttle/brake, equally for CV/RSSM/JEPA; no actor IDs, traffic/teacher/world facts or intervention reason.
**How handled:** Immutable approval record binds reviewed proposal/code/tests. Prepare a new source generation; do not alter running M3or queued M4inference bytes. Legacy v1payloads stay unchanged. Apply only after pinned owners terminate; native acknowledgments and old-policy parity must verify the new interface before planner use.
**Evidence:** reports/navigation_ego_feedback_approval_20261020.json; staged codec/history4tests pass.
**Lesson:** Declare an observation-contract extension before using it, even when it contains only the ego vehicle's own actuators.

## 2026-10-20 13:30 - Original camera rule study fails all paired rule contrasts
**Type:** result / diagnosis
**Stage:** Navigation M3

**Result:** All72original CNN-facts trials have complete source/certificate/video/native evidence.15pass,57fail;22unfilteredcontactstarts across9trials,15native red/5native stop violations. Independent actuation audit72recorders,0control differences,0missing acknowledgments. Family passes: emptyLight2/12, clearingCross3/12, stoppedLead1/12, memory6/12, stopRelease0/12, yellow3/12. Both counterfactual arms pass in0/36location–seed pairs.
**Meaning:** Memory0always holds, memory1always departs across red/green history; no demonstrated memory-dependent rule response. Stop-release fails by both false holding and missed stops. Two repeated locations per family are not72independent locations. No promotion or relaxed scoring.
**Evidence:** reports/navigation_m3_camera_rules_20261020.md/json/png; immutable raw assessment_1792482786065488459 and actuation_audit_1792482778938931408 under the original student root. Reporting only aggregates settled flags; raw sources/results unchanged.
**Next:** Already-preregistered semantic residual begins after termination, restoring and independently checking original RGB. Its recipe was frozen before these student outcomes; never fit to this rule suite.
**Lesson / interview:** Why is15/72 alone inadequate? Zero pairs pass both arms; a model can accidentally pass one condition by always stopping or always going.

## 2026-10-20 13:30 - Stage approved actuator transport with exact parent control parity
**Type:** implementation / validation / mistake
**Stage:** Future Navigation M5interface

**Work:** Approved scope implemented in an isolated source tree: explicit checkpoint/worker version negotiation, strict own-only envelope, two measured-command history, ego-only get_control extraction, trace receipt and independent native-feedback auditor. Existing camera parent runs every tick, preserving visual memory; this wrapper is only a parity reference, not a trajectory planner. Final registration binds the input version and configuration bytes.
**Validation:**36focused tests pass11.65s, including exact controls/all predictions in two fresh worker architectures across contiguous histories; original encode/decode function source unchanged. Separate activation guard test1pass. Lint and dry-run patch application pass. No current src/navigation/config bytes changed; queued matched source hashes still all match.
**Mistakes:** Initial unit fixtures used tiny images, float64nav/ego and omitted an isolated recording-plan fixture;29pass/5fail then33pass/1fail. Corrected fixtures match the real adapter and copied approved plan; retained failed trees/logs. No production experiment or checkpoint affected. Two apply_patch context mismatches from formatting made no edits; corrected targeted context.
**Lifecycle:** New guarded activation waits until all four pinned roots/processes terminate, preserves before bytes, applies only tested source, reruns tests and performs one bounded TRAIN160frame/240s native acknowledgment check. Same14GB/10GBstorage plan, no fitting, final driving or promotion. Approval record and integration protocol retain exact scope.
**Lesson / interview:** Why own applied controls rather than last proposed controls? Scripted setup can ignore proposals; prospective dynamics must initialize from real ego actuators without receiving an oracle intervention tag.

## 2026-10-20 13:31 - Queue approved feedback activation after all frozen comparisons
**Type:** execution / documentation
**Stage:** Navigation M5interface

**Execution:** outputs/navigation_ego_feedback_activation_20261020 is live and waiting, launched through sleep inhibition. Actual mutation waits for four complete roots plus their real process termination/global lock; all current inference/config source hashes still match. Exact12target patch,36test receipt and user approval are bound. One future TRAIN160frame/240s acknowledgment probe, every native contact retained, no automatic promotion. Stop this owner only with its STOPfile.
**Progress:** Read-only watcher now includes activation and original-RGB audit counters; no job shell scripts or frozen runtime files edited. Ledger/handoff record failed original M3and protected correction accurately. Native feedback verification remains pending, not claimed from unit tests.

## 2026-10-20 14:01 - Reject both protected stop corrections; fixed actions reveal inverted risk
**Type:** result / diagnosis / implementation
**Stage:** Navigation M2/M5

**M2:** OriginalRGB verification and frozen-parent reproduction pass. Two1000update arms share every episode draw and select0. Facts750improves recall308→333/560(55.0→59.46%) but increases falseholds81→95/582(13.92→16.32%): violates predeclared+1point cap. No post-hoc promotion or native retry. All10scheduled validations and raw weights retained; report navigation_m2_semantic_correction_20261020.md/json/png. Scored VALhas0stops for crossing/following families, so this metric cannot alone establish obstacle safety.
**Fixed actions:** Six prospective80-command rollouts finish; independent480RGB/960acks/0differences, all observable replay comparisons pass. Four contact starts: go at1.30s and wait-go3.80s per repeat; brake no contact. Post-contact tails explicitly censored, first collision remains supported. Only one static TRAINscenario, not reactive qualification.
**M5 diagnosis:** New CPU2thread/900scap analysis completes6/6original contexts, no model updates. Past8RGB only, original selected3member ensembles, all15plans reproduced and scored; event outcomes never enter forward. JEPA/RSSMmiss14/14positive horizons at.5;CVmiss0but flags34/34negative horizons. JEPA4s brake score.355 versusgo.0015 andwait.0066 reverses observed ordering. CVcommoncostchooses centerstop; learned models choose unexecutedleft/faster, whose outcome is UNKNOWN. FP32versusregisteredGPUBF16and different perception pipelines disclosed. No qualification/promotion.
**Source audit / correction:** I initially called all4positive-source episodes expert; actual mode audit shows2noisy/2expert. Twoof3TRAINcontact episodes are rear impacts with near-stationary ego. All144pre-first-event positive horizons containbraking;136also containpost-eventcommands;13positivehorizons startafterfirstevent. This supports a dataset/action association concern, not proof of a causal shortcut. No contacts filtered. A temporary status==Completed query undercounted Perfect arrivals; corrected against summary completed flags before publication (ResNet6/6).
**Countermeasure:** Standalone UNTRAINED spatial20Hzfirst-event hazard proposal, predicted-token risk readout and41own/nav/ego conditioning; losses endatfirstevent/censor and cannot use later reactioncontrols.3tests pass1.29s, including decodergradient, spatialassignment and future-action causality. No active source changes/fit. Future fit requires diverse independent action interventions, predeclared calibration geography and a genuinely shared perception comparison; all original datasets/results retained.
**Evidence:** reports/navigation_m5_fixed_plan_diagnosis_20261020.md/png; outputs/navigation_fixed_plan_diagnosis_20261020; outputs/navigation_action_confound_audit_20261020; reports/navigation_first_event_model_proposal_20261020.md.
**Lesson / interview:** Why is next-frame accuracy insufficient? A predictor can associate braking with collision because of the observational action distribution, fail the observed alternative-action ranking, and still look competitive under a rare-event squared-error score.

## 2026-10-20 14:11 - Prepare strictly TRAIN long-drive geometry and queue a bounded engineering pair
**Type:** implementation / execution / scope
**Stage:** Navigation M2integration

**Problem:** Historical1.233km Town03drive crosses later-held-out junctions. Preserve its13contacts/2red/1stop and exposure disclosure; do not reuse it for fitting/tuning or claim a comparable before/after gain on a changed route.
**Geometry:** Offline CARLAmap, unchanged native global planner, fixedseed20261020, townorder sorted,1000exactserialized start/end draws/town, length1000–1500m/endseparation≥250m. Whitelist actual-training guard minus ALL24current rule-case junctions; require≥1m geometric clearance from every disallowed junction. Firstvalid Town04attempt27:1375.620m, junction134only,43.288m nearest reserved geometry. Exact serialized route re-interpolation matches every dense waypoint/length. No server/model/outcome selected geometry; no forbidden-route relaxation.
**Queue:** outputs/navigation_train_integration_pair_20261020 waits for all matched work and approvedfeedback activation/nativeaudit/realprocess termination. Two900s-capped arms, expert then existing ResNetfull-labelseed42checkpointSHA65d020c285002e73e5c438b6496831eb7cfac812a5dea845f2fbff9d0472750a, commonseed62071. This is an explicitly POSTHOCfamiliar-TRAINengineering parent choice based on first2matchedpilot outcomes, frozen before observing remaining DINO/CARLA/no-color results. Original M4comma primary/selection remains unchanged; no later reselect.
**Honesty:** Onejunction long route tests sustained navigation, not urban-rule reliability or untouched benchmark performance. All nativecontacts/failures/videos/source snapshots retained; actual post-drive junction geography audited. No fitting or automatic safety/M6promotion. New queue pins exact FUTUREapproved-feedback source bytes; current pinned runtime remains untouched. Stop only with this owned root/STOP. Sleep inhibition and same14GB/10GBguardian active; native driving not yet started.
**Live M4:**12/30planned cases settled; CNN4observed/2unknown,0arrivals/1contact;ResNet6/6arrivals,3strict/4rule-safe/0contacts;DINOactive. Never treat unobserved cases as successes or infer route arrival from status==Completed alone (Perfect also completes).
**Lesson / interview:** How do you add an engineering long-drive check without moving the benchmark? Keep original/frozen results and route suite untouched, select new TRAINgeometry only by preregistered map rules, disclose its one-junction scope and posthoc parent, and give it no automatic qualification power.

## 2026-10-20 14:18 - Record camera-context limitations and verify future integration source pins
**Type:** diagnosis / verification
**Stage:** Navigation M5/M2

**Camera context:** Audit-only actor-centre projection, no policy inputs changed. Rear-contact TRAINsources22/42 have37/64and47/64past centre positions outside all horizontal frusta;4/8and5/8positive windows have every8centre positions outside. Validation source20eventual collision actorID is absent in53/64full world snapshots; source capture records all vehicles/walkers/lights. These are centre/identity tests, NOT full bounding-box/occlusion visibility or causal attribution. All contacts retained. Future work needs longer causal belief memory and diverse interventions with observable actors; no blind fit extension.
**Source verification:** New TRAINlong-drive queue expected future hashes independently match the already validated feedback source tree for every registered file. Current matched/runtime/config hashes remain unchanged; native long pair still unstarted. Live owner verified, same guardian baseline. Consider a bounded TRAINonly semantic/instance/depth LABELrig profile after existing native owners finish; runtime remains the approved RGB-only boundary and label budgets must charge new supervision.

## 2026-10-20 15:09 - Matched encoder matrix finishes; feedback verifies; clean TRAINlong drive arrives
**Type:** milestone / result / infrastructure failure
**Stage:** Navigation M2/M4/M5interface

**M4:** Five matchedfull-labelseed42drivers/30plannedcases now complete,27native/3unknownfailed; originalcomma6reusedonce, no reselection. CNN0/6arrivals/1contact/1red,4native2unknown;ResNet6arrivals/0contacts/4rule-safe3strict/1red1stop;DINO1arrival/0observedcontacts,5native1unknown;CARLA3arrivals/1contact/2red;comma4arrivals/10contacts/2red1stop. Separate no-color2arrivals/0contacts/1red. No driver passes everycase. Canonical report navigation_m4_matched_native_20261020.md/json/png; raw report-generation source retained. Reusedcomma predates approved compactactivation:3recording/final-registration modules differ, nonfinalpreviewremainsfull andonlyconfigdifferencefreefloor25→10. Policy/physics/scoringunchanged; actuatorv2applied AFTERallmatrixdrives.
**Feedback:** Approvedexactpatch activatedafterallsourceownersfinished; focused36tests rerun.160realobservations/159consecutivepriorcommands/0nativeackdifferences. Probehas2unfilteredcontacts(native1event),retained; interfaceverification is notsafe driving. No activeplanneradded.
**Longdrive:** Originalexpert1.376kmTown04attemptsegfaults inengine beforefirsttraceframe, emptytraces/partialrecorder retained as infrastructurefailure. Onlygenuinelyunstartedcameraarmcontinued samepath/seed62071/checkpoint/900scap. FrozenResNetarrives1375.639m,199.5simulatedseconds/255.125nativewalls,3990frames,0unfilteredcontacts/0red0stop/0interventionticks.5minimum-speedinfractionsstillcounted,notstrictsuccess. Actualall3990posesremainallowedTRAIN. IPCp9525.10ms/end-to-end15.57fps; uninterruptedH264yuv420pvideo1995framesverified. Report navigation_train_long_drive_20261020.md/json/png withvideolink. Onejunction/familiarTRAIN/posthocparent,noexpert-relativecomparison,M4primaryreplacementorM6qualification.
**Lesson:** Publish native deployment disagreement with offline selection and preserve infrastructure failures; a clean familiar-route arrival helps integration without becoming evidence of general urban rules.

## 2026-10-20 15:09 - Profile and implement guarded TRAIN-only dense labels; guard catches actual spawn leakage
**Type:** implementation / execution / bug / data integrity
**Stage:** Navigation M1/M2/M5data

**Profile:** Original3RGBbaseline21.97physicsfps/p9563.44ms;9alignedRGB+instance+depth10.19fps/p95138.73ms.60storedtimestamps/mode;compressed44.91/94.21MB,raw79.63/291.96MB,total142.61MB.0contacts,0cross-modalityposeerror,allrawpixelreadbacksverified. Predeterminedmidpointfigreport navigation_label_rig_profile_20261020.md/png. Fixedorder/slow2m/s/singleTRAINscene,no broadthroughputclaim.
**Semantic trap:** Versioneddocs showoldclassnumbers; installed0.9.15enum/pinnedObjectLabel.handactualpixelsuseCar14/TrafficLight7. Verifyenumatcollection,rawBGRAdepthchannelorderchecked. PinnedTagger.cppuseslow16bitsUnrealActor.GetUniqueID,NOTCARLARPCactorID; preservepairwithoutassumingactorassociation. Earlier conservative render-component wording is imprecise, no data/target mapping was made from it.
**New data engine:** Explicit--training-dense-labelsrequires canonicalguardedexpertTRAIN; rejectsstudent/DAgger/final/rule/empty/gap. Extra6privateTRAINlabelsensorsneverenterteachercontrollerorpolicySensorPacket. OriginalRGB10Hzpreserved;separateinstance/depth+exactsameRGBshards2Hz,frame/physicaltimestampandposeagreement. Originalinferenceguardforbidsnewlabelmodules. Full485CPUtests/2skip49.59s+focused37passes;newindependentRGBtampertest1pass. Nojobs/*.sh,reward,benchmarkroute/scoringorpublicinputchange.
**Immutable plan:**13sources/5towns,sourceactualguardminus24rulelocations;thirdTown03/04calibrationlocationsfrozenbeforelabels,oldbackbone/driverexposuredisclosed. Each200RGBtimestamps/180scap;entire1800s/3GBunderoriginal14GB/10GBguardian. Allfuturelabelbudgetschargeeveryannotation. No fitstarted.
**Runtimecatch:** First3sources6/35/50yield39labels+200RGBeach. Route78catalogdeclaresTown02/298butfirstactualposemapsTown02/242atframe534/road290/lane-1,reservedM3memorylocation. Originalguardraisesbeforeanydenselabel,raw1frametrace/failurepreserved. EngineSignal11appearsduringcleanup;primaryfailureisGEOGRAPHY,notanexcusetorunagain. Originalrootterminalfailed4/13;route78zero-labelsourceisnevertrainingdata.
**Continuation:** Only9originalunstartedIDs101/134/124/189/225/211/231/258/297runfreshowner,identicalrecipe/source/plan/roles/no failedretry. Conservativeoriginaldeadlinefromrequestpreparationmtime,shared3GBcap,no fresh30minallowance. Current7/13attemptssettled. IndependentCPUauditorqueued;itkeepsfailedsourcein13denominator,verifiesallremainingpixels/originalRGB/nativeacks/actualmapposesandcalibrationsplit. First3partialauditsalreadyverifyall600RGB/117labelsand1200poses:nearestreservedclearances61.879m,1.000017m,41.761m. Near-threshold1mresultisnotaphysicalcertainty;allgeometricassumptionsremainreported.
**Mistakes:** Twoapply_patchcontextfailuresmade no edits; corrected againstformattedsource. Original13sourceplanisnotchangedafterspawnfailure. Strongactual-positionguardsmustremaininplaceevenwhenplannedroutegeometrypasses.

## 2026-10-20 15:21 - Finish13source collection; retain failed audit and curate only11verified sources
**Type:** result / data integrity / decision
**Stage:** Navigation M1/M2/M5data

**Execution:** Original13attempts settled without retry/replacement. Originalroute78spawnquarantinebeforelabels remainsfailed. Continuationusesoriginalconservativedeadline/shared3GBcapandunchangedsourceplan;allattemptnativecontacts0,privilegedshort/cappedcollectionnotcamera-drivingevidence.
**Independent audit:** Asecondsource101failsactualminimum1mreserved-junctionclearance (observed0.5013505m);quarantineWHOLEsourceincluding22denselabels,do notcroporfitthem. Original13sourceauditprogressremainsFAILED,all rawfailure/reportsretained. Do not relax the threshold or turnthatfailedartifactgreen.
**Accepted manifest:** Separate outputs/navigation_dense_labels_verified_20261020 binds11independentlyverifiedsources:9TRAIN/2calibration,all5TRAINtowns;1800TRAINRGB/351TRAINdenselabeltimestamps and400calibrationRGB/78calibrationlabels. All2200originalRGBpixelhashes/next-ticknativecontrolsandall429labelRGBtriplesverify,actualsourcegeographyindependentlychecked. CalibrationTown03/1221andTown04/916fixedbeforelabels;oldfrozenencoders/driverexposureisdisclosed.11accepted/2rejectedstays13denominator. No fitstarted or modelpromotion.
**Coverage correction:** Native long-drive opportunitymeter has0red/0stopevents. Explicitlyaddedtoreport/JSON/index:zero violationson1.376kmrouteprovideNOrule-handlingevidence;onlysustainedcamera-navigation/contactresult. OriginalM3failurestillstands. Thisguardagainstabsence-of-opportunityclaimispartofhonestreporting.
**Evidence:** reports/navigation_dense_labels_20261020.md; reports/navigation_results_index_20261020.md; original audits and manifest hashes. Currentallnative/collection/auditownersareterminal;onlycapacityguardianremains. Fullsourcepinthroughfinishedcollectionisretained inrequest/sourcearchives;anynextsourcegenerationmustberegisteredfresh. Source/reportlintandgitdiffwhitespacecheckpass. Twoformat-onlycuratorscriptlinebreakfixesfollowitsrun;exactgeneratingcuration_source.py/hashisretainedintheverifiedroot. Laterwatcherapply_patchcontextfailuremade noeditsandwasretargeted.
**Next:** Inspect frozen FPN/geometry againstnewcalibrationvisiblemasks/depth before committing toanotherperceptionfit;thenregistersharedcamera-onlyperceptioninputsanddiversefirst-eventactioninterventionsforCV/RSSM/JEPA. Do notrepeatlosingstopresidualsorclaimM6complete. All14GB/10GBcapacitylimitsremain;~5.98GBnetused/21.8GBfreeat15:20.
**Lesson / interview:** Afailedwhole-corpusauditcanproduceaseparatehonesteligiblemanifestforverifiedsourceswithoutchangingafailedtrial,benchmarkdenominatororgeographyrule. Preservebothrejectionandpositiveevidenceandnevertrainsilentlyonrejectedframes.

## 2026-10-20 15:37 - Frozen perception diagnostic separates detection and projection losses
**Type:** diagnosis / implementation / validation
**Stage:** Navigation M2/M5 perception

**Execution:** Exact frozen FPN production forward, unchanged thresholds and weights, all78 timestamps/234views at the two predeclared calibration sources. RGB-only predictions committed before reading instance/depth targets. Two focused matching/mask tests pass; independent analytic depth/box test adds one pass. No model update or driving promotion.
**Results:** 1,584 included visible vehicle mask/view pairs; raw1,110matches, projectable926, fused666. Cross-view renderer-ID/group denominator1,134: raw786(69.3%), projected616(54.3%), fused608(53.6%). Thus most fusion reduction is expected duplicate-view removal; only8additional matched groups lost. All included masks are VEHICLES; no pedestrian/two-wheeler evidence.140tiny masks remain separately recorded.
**Posthoc breakdown:** 15–30m encoded-depth bin432/465raw and431projected;30–60m414/591raw and273projected. All-distance totals include275masks at≥60m, beyond the projection contract. Raw matches only91/430masks at8–31pixels versus324/327at≥512pixels. Do not call all-distance coverage near-field recall. Mean109.7ms/p95115.0ms,0.939GBpeakallocated; hook timing overhead disclosed. Two familiar locations/repeated snapshots, no independent encounter confidence claim.
**Geometry:** Flat-ground versus axial visible-bottom surface gap median1.05m/p956.26m; radial alternative2.51m/14.88m. These are hypotheses, not certified actor-centre errors. Next bounded native probe freezes six known vehicle placements at strict TRAINroute6, both depth interpretations,0.30m box tolerance,300s/100MBcap; excluded from all model fitting. Confirm convention before any learned geometry target, then preserve explicit unknown detections and shared inputs across CV/RSSM/JEPA. New expert labels have0contacts and do not replace missing action-intervention coverage.
**Evidence:** reports/navigation_dense_perception_20261020.md/json/png and fixed midpoint examples; outputs/navigation_dense_perception_diagnosis_20261020 retains exact request/source/raw predictions/scoring. Depth probe outputs/navigation_depth_geometry_probe_20261020 is RUNNING, not yet verified. Same original storage guardian, no jobs scripts/reward/benchmark changes.
**Mistakes:** Removed an unused erroneous draft-test expression before tests/inference. Long report strings failed lint; inference ran with one harmless long string still present, then only that string's wording shortened after termination. Original generating source remains archived; subsequent lint passes. A read-only rg included a nonexistent optional probe filename and returned2; no runtime files changed. Matplotlib used its /tmpcache when home config was not writable; plots generated and visually inspected.
**Lesson / interview:** Separate same-view detection coverage from cross-view object coverage, respect the stated range and observed class denominator, and verify simulator depth convention before using its numbers as geometric supervision.

## 2026-10-20 15:57 - Retain native instrument failure, preserve unknown objects and reject two depth replacements
**Type:** implementation / result / infrastructure failure
**Stage:** Navigation M2/M5 shared perception

**Native failure:** The fixed six-placement depth convention probe hits native engine signal11 before its first capture.0/6observed; six unknown. Empty recorder, sensor manifest, source/request and cleanup errors retained at outputs/navigation_depth_geometry_probe_20261020. No retry, positive convention claim or fitting on this failed instrument data. Repeated cleanup RPCs each take20s after server exit; separate future-only native_cleanup helper now skips all RPCs when its exact owned process has exited, bounds live cleanup calls1s and preserves receipt/errors. Two tests pass, no process signals or frozen past source changes. Root cause of the engine crash is NOT established.
**Shared representation:** navigation-camera-road-users-v1 retains every RGB-derived detector box with camera identity, class/score, bearing, presence and approximate-ground validity. No GT, future or teacher fields. All429cached packets independently reconstructed bit for bit in the subsequent fit; first TRAINoriginal production forward/new wrapper outputs also bitwise equal. TRAIN351timestamps/4,194boxes/1,022unlocated; calibration78/1,930/247. Every original calibration box reused, not rerun. Cache134.8s/3MB. Two boundary tests verify unknowns cannot disappear into empty padding and privileged keys/hidden truncation are rejected. No current CV/RSSM/JEPA driver has adopted this pack yet.
**Cheap geometry baselines:** Predeclared fixed source-weighted ridge(alpha1) and boosting(200iterations), no sweep/continuation, CPU2threads.2,520TRAINmatches across9sources (2,501vehicles/19two-wheelers);1,110calibration vehicle matches/2locations. All429timestamps/1,287images charged. Every calibration box prediction written before calibration target reads. Original flat-ground supported921raw-matched pairs gives encoded-depthMAE1.527m, ridge3.145m,boosting1.773m. Macro absolute-log error flat.05264 versusbestlearnedboosting.06426;≥5%improvement protection fails. BOTH replacements rejected, selected_offline=null; all fits retained.117.2sexecution,0unsupportedtargets. Earlier926projectable rematched boxes versus921conditioned raw matches is an assignment difference, explicitly disclosed. No extending a losing fit.
**Limits:** Encoded visible-bottom-pixel depth is not actor-centre3Dtruth. Native convention remains unverified despite source-supported axial interpretation. Calibration snapshots are correlated/familiar, no safety CI or pedestrian generalization. This posthoc engineering diagnostic does not alter M4primary, benchmark, reward, source exclusions or M6gates. Reports navigation_box_depth_20261020.md/json/png and navigation_dense_perception_20261020.md retain all scope.
**Next:** Prepare original-role, geometry-only intervention sites at TRAIN35/50/134/189/258(all5towns) and existing calibration124/211; old route6observed branch diagnostic deliberately excluded from new TRAIN. Front/crossing/rear geometry is being frozen before native outcomes, retaining unavailable cases rather than replacing locations. Need diverse action outcomes and genuinely shared perception adoption before any new first-event ensemble fit.
**Mistakes / validation:** One report table string exceeded lint; shortened only in current report generator after output, exact generating source remains archived. Native failures were not relabelled as successes. New source checks/tests pass individually; full485tests were not repeated for these isolated optional diagnostics. No jobs scripts or running-source bytes changed.
**Lesson / interview:** Reject a learned replacement that loses to the same simple baseline on common supported cases. Retaining unknown observations is useful engineering, but it is not by itself a trained driving improvement or complete shared-perception experiment.

## 2026-10-20 16:18 - Freeze action geometry, expose cleanup timeout bug and continue only untouched cases
**Type:** implementation / execution / mistake / recovery
**Stage:** Navigation M5 causal-action data

**Geometry before outcomes:** First draft freezes14/21usable contexts but inspection catches a crossing63m away, unreachable in8s, and rear starts crossing reserved geography. Keep draft. Version2uses every8verified TRAINsource except the old route6diagnostic plus original2calibration sites; near-crossing ego starts and rear starts on the original guarded route.24/30contexts geometrically usable;6unavailable crossings retained. Same1mclearance and source roles, no location/benchmark/reward change. Independent perpendicular/overpass test passes. Native pilot selects lowest-route TRAINlead35/crossing134/rear35, three prospective options each plus one go repeat:10original attempts,900s/600MBshared cap, no auto-fit.
**First attempt and mistake:** r35_lead/go records80RGB/160actual controls and1contact at7.25s(native), branch+3.25s. Independent closed-file audit reproducesall15plans/pixels/actualownhistory/controls and both actors'geography;65commands match until safety stop, tail censored. My cleanup helper left the reused client timeout at1s. Secondcasebrake mapreloadraises1000mstimeout and clientSIGABRT; retain1capture/1failed/8unstarted. This was an avoidable implementation mistake. Fix only restoration to20s and explicit20s before next mapload; tests now assert both values. Exact code-stripping proof verifies no control/physics/data-equation difference.25targeted tests had passed before first native execution; a synthetic cleanup test missed global timeout leakage and was strengthened.
**STOP recovery:** Root STOPtouched, exact own orphan PID/binary/group/user/creationtime checked by existing recover_navigation_owned_cleanup.py. NormalSIGTERMonly completes; no force kill or unrelated process touched. Preserve aborted runningprogress snapshot, externalpostmortem and receipts; publishfailedstate rather than stale running.
**Second failure:** Only8originalunstarted cases continue, same original deadline/cap; leadwait captures0contacts, crossinggo1contact at7.15s, crossingbrake0contacts. Nextcrossingwait mapreload aborts at20s. Thus timeout restoration alone doesNOTresolve deeper reload/lifecycle issue; cause is unproven. Preserve this second failedattempt; do not retry it. SecondrootSTOPand exact ownednormalcleanup complete. Four originalcases remain whollyunstarted.
**Current execution:** outputs/navigation_intervention_finish_20261020 executes only those4originalcases with a fresh native server per case, under original900sdeadline/combined600MBcap. Same geometry, clock, data and control equations.7/10settled at16:17:38including2failed; rearbrake is live. Final combined independent audit keeps both reload failures in10denominator. No native result is yet a qualified model, no fitting/pilot extension runs automatically, and scripted traffic is not reactive traffic.
**Source scope:** All runtime src/navigation and collected helper/config bytes stay pinned through finish/audit. Guard explicitly forbids new training/geometry modules at camera inference. Tests:25collector/boundary/feedback then strengthened3cleanup/pilot pass; previous7perception/geometry tests remain separate. No jobs/*.sh edits. Storage guardian original14GB/10GBbaseline active,~6.2GBnet at16:14.
**Lesson / interview:** Scope resource settings to cleanup, preserve the first failure, and audit real native frame evidence. Recover only unstarted cases under the original allowance; two infrastructure failures do not disappear because remaining cases finish.

## 2026-10-21 00:09 - Settle the first action pilot and diagnose native callback aborts
**Type:** result / bug / mistake / recovery
**Stage:** Navigation M5 data and simulation

**Results:** All ten original pilot attempts settled: eight verified factual TRAIN episodes, two unknown reload failures, no retries. Independent audit verifies 640 RGB timestamps/1,280 native controls, all 15 proposals per episode, actual two-command history and both actors' geography; six world-observable prefix/repeat comparisons are exact. Three contact starts: lead/go twice at +3.25s, crossing/go +3.15s. Rear options have no contact within the frozen four-second branch; no red examples. Original failed owners/audit remain failed; separate eight-source manifest SHA ef35b75311dd6d683c8f98713eb3cf919e0606d00865dafd79625fe93b1680fc. See navigation_intervention_pilot_20261020.md.
**Native diagnosis:** Retained coredump backtrace resolves a SIGABRT through LightManager::QueryLightsStateToServer, native Episode streaming callback and uncaught TimeoutException. This identifies one client-abort path, not the cause of every engine signal11. GDB build-ID warning for an unrelated PyAV library retained. No native binary patch.
**Mistakes and recovery:** The first cleanup helper leaked a one-second RPC timeout; restoration alone did not fix later crashes. Changing cleanup order also failed; restored sensor-stop-before-async with 20s timeout. Fresh ClientLease probe then fails 1/2 cycles. All attempts/cores retained; no repeated failed case. STOP identity-checked normal cleanup finishes all four old owners. Pilot captures obey original deadline, but final cleanup runs beyond it; do not claim the entire stage met 900s.
**Containment evidence:** New parent never imports CARLA; each child owns its native bindings and exits before owned engine cleanup. Two fresh stationary instrumentation children exit0 and parent survives; both engines still exit-11. This verifies containment, not engine stability or driving safety. Tests cover exact process identity and STOP-before-normal-termination; no SIGKILL. A seven-hour execution gap followed an automatic-approval usage-limit rejection of a read-only core lookup; normal escalation resumed after reset. Original storage guardian restarted with identical baseline, no new allowance.
**Lesson / interview:** Isolate an uncatchable native callback in a disposable child, retain engine errors, and distinguish completed recordings from reliable infrastructure.

## 2026-10-21 00:09 - Settle all 63 frozen expansion attempts and audit before fitting
**Type:** decision / execution / result
**Stage:** Navigation M5 causal-action data

**Predeclared expansion:** Exactly 63 wholly unstarted options from geometry v2, 45 TRAIN/18 calibration, excluding all original pilot options including its failed two. Fixed go/brake/wait-go, same routes/one-metre geography/green lights, scripted nonreactive actors. Source pinned; one child/owned engine per case. 5,400s native startup/capture cap, four-GB stage cap, 120s per child, original 14GB-net/10GB-minimum guardian. No automatic fitting. Protocol navigation_intervention_expansion_protocol_20261020.md. Focused 27 tests pass; no full historical suite claim.
**Outcome:** All 63 settle in 759.164s; 60 captures =42 TRAIN/18 calibration. Three TRAIN failures retained: r231_rear/wait-go and r297_rear/go simulator timeout/client-6, r297_lead/go observable signal-clock tolerance failure. No case retried or replaced. Engine exit counts62×-11 and1×143; the queue's complete state means attempts settled, not successful infrastructure or qualified data.
**Audit running:** Separate outputs/navigation_intervention_expansion_audit_20261020 waits for exact parent PID exit, then independently checks closed images/controls/proposals/first-event censoring/both actors' geography and world prefixes. Count all raw contacts before fitting eligibility. Runtime/audit sources remain pinned until terminal. Current storage about18.9GBfree/8.9GBnet, same original budget. Live watcher now renders settled counts separately from captures and native errors.
**Next / interview:** Audit eligibility and calibration event coverage before registering new fits. A settled queue with three failed attempts cannot become 63 successful episodes, and process containment cannot be reported as fixing native crashes. M2/M3/M5 qualification and M6 remain open.

## 2026-10-21 00:15 - Audit sixty expansion recordings, fix static correspondence and freeze shared features
**Type:** result / bug / decision
**Stage:** Navigation M5

**Evidence:** Expansion audit independently verifies60/63recordings,4,800RGB/9,600actual controls,42TRAIN/18calibration,13unfilteredcontactstarts. Three missing-recording failures remain unknown. Original audit also reports38/39prefix comparisons passing; one has771.84m/180deg mismatch with zero velocity/phase difference.
**Cause and correction:** Town04 rear/wait-go has the SAME43staticlightposes but different startup actor-ID ordering; both unique-role moving vehicles are bit-exact. New standalone audit matches lights by exact type/role/pose and moving actors by unique type/role, rejects ambiguity, preserves original tolerances and all original failed artifacts. All39supplemental comparisons now exact. Three tests independently reject real motion/phase differences and duplicate geometry. RGBprefix equality is not claimed.
**Verified corpus:** Separate v2manifest retains68eligible recordings from73attempts:50TRAIN/12contacts/eightlocations;18calibration/fourcontacts/twolocations. Allsource/native/plan/pose/RGBshard hashes rechecked. Five unknowns retained; no retry. Braking now has both contact-free and rear-contact outcomes. Scripted traffic is nonreactive, all lights green; no red-cost or M5qualification claim.
**New cache:** Frozen before computation: reports/navigation_first_event_cache_protocol_20261021.md. Exactlyone branch observation per episode, eight past FPN packets equally available toCV/RSSM/JEPA; causal currentJEPAgrid plus future latent targets. Missing+4.0sRGBtarget staysUNKNOWN. Pixel-derived features committed before attaching audited targets; lasttwoactualego commands only. 1,200s/250MBpersistent-cache cap, original storageguardian unchanged, no automatic fit. Sevenfocusedtests pass; unusedlocalcaughtbylintandremovedbeforelaunch.
**Untrained prototype bug:** Its per-window hazard loss divides by variable observed prefix length, changing the censored survival objective and overweighting early events. No fit used it. Preserve original; nextgeneration will SUMprefixNLLthenaveragewindows, with tests. Original M4primary/nativefailures and allM2/M3/M5/M6gates unchanged.
**Lesson / interview:** Match static simulator assets by geometry across sessions, and separate survival likelihood from per-timestep averaging; both errors can distort a causal-planning study without changing a single image.

## 2026-10-21 00:44 - Reject first-event ensembles and return to camera rule perception
**Type:** result / decision / mistake
**Stage:** Navigation M2/M3/M5

**Completed:** Shared causal cache68episodes/197.34s; six JEPA/RSSM fits400updates each/58.90s; frozen all15candidate inference5.42s. Both arms select100/0/0 across seeds, later updates overfit. Full report and graph: reports/navigation_first_event_study_20261021.md. No native planner or M6 run.
**Results:** Calibration18episodes/fourcontacts/twolocations. Brier action-only.16328,constant.17285,CVcalibrated.17597,JEPA.17706,RSSM.17728; each misses4/4 at frozen.5. Both learned sequenceNLL1.492 beatsconstant1.519 but endpoint gatesFAIL. RawCVflags18/18,14falsealarms. No fit extension, threshold change or model promotion.
**Integrity:** Same complete eight-frame object history/current spatial tokens/actual own commands; proper prefix-summed survival likelihood. JEPA453613/RSSM622701parameters: equal updates do not mean equal compute. All15unexecuted outcomes UNKNOWN. BF16batch-shape maxlogitdifference.0625/maxCDF.000515; original selected predictions remain primary. All checkpoints, RNG/optimizer and original failed audits retained.
**Next:** Diagnose paired active-bulb labels using existing static phase images; housing-center/whole-region labels previously failed. Before NEWfits remove current reserved rule/actual validation/newcalibration geography and retain original four component-validation junctions. No benchmark, input, reward or gate change. Prior full-image detector adaptations and stop-release failures remain relevant; this audit is not proof of a driving improvement.
**Mistakes:** All15assessor launched despite import-format lint error; exact generating source archived, formatting fixed only after terminal. One image-view result accidentally printed base64 instead of forwarding the image; corrected without artifact changes. A read-only lookup used Town01 rather than actual Town01_Opt directory and failed; corrected. No failed native cases retried.
**Verified:** Focused eight new dynamics/boundary tests plus previous cache/prefix tests; report/source lint passes. Full historic suite not rerun. Only original storage guardian live; same14GB-net/10GBminimum,18.87GBfree. No jobs script edits.
**Lesson / interview:** Better survival likelihood can coexist with worse event probability and complete failure to detect contacts. Retain each metric and reject the model when its predeclared usefulness gate fails.

## 2026-10-21 00:50 - Audit active-bulb labels and start a matched native-pixel comparison
**Type:** result / decision / execution
**Stage:** Navigation M2/M3 perception

**Audit:** All1800original static images/source hashes verify in3.23s; 150poses=73eligibleTRAIN/63newlyreserved/2actual-poseexcluded/12unchangedVAL. Both newly excluded nominalTown02/298poses actually lie inside reserved junction242(case15) or132(case10044), clearance.35m<1m; neverfit them. Seveneligiblelocations/fourtowns, noTown10TRAIN. Original144VALimages/72labels unchanged, noVALpixelmasks computed.
**Target:** Fixed phase-response threshold max20RGBbrightness levels or5timesred-repeatnoise finds active pixels in92/92red,92/92green,90/90supportedyellowhousings. Originaldarkhousingpixels remainunknown. Firstsortedhousingperlocation plots visually inspected; tinybulbs oftenonepixel, highlighting stride2aliasing. This is label-coverage evidence, not signal detection or driving competence. See outputs/navigation_signal_pixel_audit_20261021/fixed_examples.png.
**Registered comparison:** reports/navigation_signal_pixel_study_protocol_20261021.md. Freshnative-stride1/RF21CNN37772parameters, center-versusbulblabels,1000updates each/seed42, identical876TRAINimages/144VAL. Same optimizer/loss/distribution, fivefixedcheckpoints including0, commonprospectivepointlocalizationloss selection, nativeRGBonly. Fit1200s/stage2800s/250MBcap, existingguardianbaseline; noautodriving orfitextension. Point-in-housingdiagnostic isNOToldboxIoU/AP; old44/72detectorresults not directly comparable.
**Current:** GPUstudyRUNNING at outputs/navigation_signal_pixel_study_20261021. Sources/request/archive pinned until terminal. Onlythisstudyandoriginalcapacityguardianlive. No jobs script edits, newinput/reward/benchmark or M4primarychange. Livewatcher followsnewstudy. M2/M3/M5 remainunqualified, M6notrun.
**Validation / mistakes:** Fivefocusedtests pass: phase-response excludesstaticred/repeatinstability, geographicreservationprecedence, RGB-onlynative-shape/translation and connected-componentseparation. Full876/144datasetpreflight confirmsroles/72denominator. Initialunusedimport/longlines caught andfixedbeforelaunch; no failed fit retried.
**Lesson / interview:** A light housing center can be dark even when its bulb is visible; native paired interventions can yield localized supervision without giving privileged phase state to the deployed network. Always apply later geography reservations to reused oldTRAINdata.

## 2026-10-21 00:54 - Bulb labels improve localization but fail background rejection
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** Paired1000update fits complete28.53s. Frozencommonloss selectscenter0andbulb1000. Bulb56/72point-localized,44/72jointstate versusselectedcontrol0/0, but37,241known-background peaks/144images and1.654%backgroundactive; bothprotectionchecksFAIL. Original44/72boxdetectorusesdifferentIoUmetric, notdirectlycomparable. No modelpromotion/nativepilot orfailed-fitextension.
**Details:** BulbjointTown01 7/8,Town03 15/20,Town04 12/12,Town10HD10/32;red22/36,yellow15/18,green7/18. Three-viewforwardp953.17ms, excludesCPUpeakextraction/IPC/simulator. Fivecheckpoints/eacharm retainedwithoptimizer/RNG. Independentclosed-predictionrecount reproducesselectedcounts. Report/learningcurves/full-viewerrorplots reports/navigation_signal_pixels_20261021.md; predeterminedfirstredhousingpervalidationtown, allpredictedpeaksdisplayed.
**Diagnosis:** Fullviews showfalsepeaks onfoliage/facades/roadedges. MeanbackgroundCE.07156maydiluterareconfidentmistakes; hypothesisnotproven. Newregisteredobjectiveaddsper-imagehardest1%backgroundCEwhilepreservingforeground+originalbackgroundterms. SameTRAIN876/VAL144andnativeRGBonlyarchitecture, startsfromexactunqualifiedbulbparent; one1000update/lr1e-4recipe, newoutputandoptimizer, no thresholdchange.
**Protection:** Newprospectivecomponentselectionmustpreserve44jointandalltown/statejointcounts; passalso requires≥90%reductioninbothbackgroundpeaksandactivefraction. Keepinitializationifnonequalifies. No drivingqualificationfromcomponentgate. Protocol navigation_signal_hard_pixels_protocol_20261021.md;1200sfit/1500sstage/150MBcap, sameguardian.
**Verified:** Sevenfocusedtests passincludingignored-regionzerogradientandhardnegativelossnotdilutingoneconfidenterror. Dataset/sourceparityrequiredatstep0. Reportunusedimportcaughtbylintandremovedbeforegeneration. AlloriginalM3/M5failures/M4primary/M6pendingunchanged; no jobs edits.
**Lesson / interview:** Correct foreground labels alone do not make a detector. Rare confident false alarms can dominate deployment even when the average background loss looks small.

## 2026-10-21 00:57 - Reject hard-background correction despite fewer false peaks
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** One1000update new-objective fit completes24.01s. Latestbackgroundpeaks37241→1957(94.7%lower), activefraction1.6541%→.0849%(94.9%lower), butjointlights44→27. Alltrainedcheckpoints failcoverageprotection; select0, no promotion/extension/nativepilot. Parentalso remainsunqualified. Red22→14,yellow15→6,green7→7;Town10HD10→2,otherlossesTown01/Town03.
**Integrity:** Retainedinitialization modeltensors andallfrozenpredictions independentlymatchexactparent. Everycheckpoint/optimizer/RNG/predictionsretained. NativeRGBonly, original876TRAIN/144VAL/72labels, nofinalTown05tuning. Report/graph reports/navigation_signal_hard_pixels_20261021.md. Seventeenfocusedtests passincludingresearch-boundaries; fullhistoric suite notrerun.
**Next:** Wider RGBhousingcontext/candidateverification withTRAIN-onlybackgroundmining. Current21pixel modelshowsa precision/coverage trade-off; do not assertcontextcausewithouttesting. Registernewarchitecture/data/budget beforefit, preserveeverytown/statecoverageandalloldfailures. No blindextensionorqualificationrelaxation. M2/M3/M5remainunqualified; M6notrun.
**Mistake:** Reportgeneratorran withanoverlongMarkdownheader lintwarningbecausecommandsweresequentialwithoutfailuregating. Data/model executionunaffected; shortenedheaderandregeneratedreportonlyafterterminal, lintpasses. Futureverification-and-launchcommandsuseexplicit&&orinspectresultfirst.
**Current:** Alltrainingandnativeowners terminal; originalstorageguardianonly, same14GBnet/10GBminimum. No jobs scriptsedited, no push. Approvedego-feedback remainsnativeverified. ActivegoalstillM0–M6, notcomplete.
**Lesson / interview:** A94.7%false-alarmreductionisnotan improvementifitloses17of44correct lights. Prespecifiedcoverageprotection preventsselectingacleaner-looking but lesscapable component.

## 2026-10-21 01:01 - Register and start RGB housing-context verification
**Type:** decision / execution
**Stage:** Navigation M2/M3 perception

**Hypothesis:** The21pixel bulb model mistakesmanylocaledges forlights; hardbackgroundweighting removesfalsepeaks butalso17correctlights. TestwhetherlargerRGBhousingcontextseparates them. No cause/successclaim yet. Oldfailedstudies andallcheckpoints remainunchanged.
**Protocol:** reports/navigation_signal_context_protocol_20261021.md. Freezeexactbulb1000proposalmodel, threshold.5/8connectivity, no runtimeproposalcap. Verifierseesnative32x32plusarea-averaged96x96context fromSAMEcurrentRGBonly; noabsolutecoordinate/map/teacher/phase-pair input. FreshCNNclassifiesbackground/red/yellow/green;1000updates/seed42/lr.0003/batch64.
**Data:** Sameaudited876TRAIN/144VAL/72labels. MineTRAINpositivesinsideknownhousingsplus≤4auditedbulbpixels; TRAINnegatives≤16proposals/image(fourhighestscore+12seededrandom), allotherprojectedlightregionsunknown. NevermineVAL. ReuseeveryoriginalVALproposal withsource/checkpointreceipt andindependentbaseline recount. PersistTRAINcache/rows/proposals; temporaryVALpatcheshashedandreproduciblefromoriginalRGB.
**Gate:** Preservetotal44jointandalltown7/15/12/10andstate22/15/7counts; rankprotectedcandidatesbyfewestbackgroundpeaks. Require≥90%backgroundreductionforcomponentpass. Ifnoneprotected, selectnoverifier; originalparentstillunqualified. Noautomaticdriving, M4primary/benchmark/gatechange.
**Current:** outputs/navigation_signal_context_20261021RUNNING,sourcepinnedthroughterminal.600scache/1200sfit/2400sstage/250MBcap, originalguardianbaseline.13focusedtests passincludingcropalignment/boundarypadding, ambiguous/unknownnegativeexclusion, allproposalscoringandRGB-onlyinput. Unusedimportcaughtandfixedbeforelaunch. No jobs script edits.
**Lesson / interview:** Addvisualcontextwithoutaddingprivilegedstate, andholdbothcoverageandfalsealarmsaccountable. A proposer-plus-verifiercannotrecover objectsabsentfromtheproposalset; itsceilingmustbereported.

## 2026-10-21 10:39 - Reject RGB context verifier and diagnose coverage gap
**Type:** result / diagnosis / decision
**Stage:** Navigation M2/M3

**Result:** Contextstudycomplete30.13s(cache25.59s),1000updates,126804parameters; everycheckpointfails44jointandtown/stateprotections. Latest14/72localized,12joint,2575backgroundpeaks; selectNO verifier. Parent56/44/37241remainsunqualified. All38543VALproposals scored, no runtimecap. Cache15678TRAINpatches(14016background/831red/412yellow/419green), source/prediction/optimizer/RNGretained.
**Frozen diagnosis:** TRAINbalancedaccuracy82.47%versusVAL40.25%; TRAINincludesknownbulbanchors. Predicted-point-onlyTRAIN75.35%,stillgap. Latestbinaryforegroundgate withoriginalstate gives28/72joint/4135backgroundpeaks, alsofails; posthocdiagnosticONLY, neverreselect. Wholehousingpointstatelabelscanincludeinactive/darkpixels; no claimwidercontextintrinsicallyuseless. See reports/navigation_signal_context_20261021.md/graph.
**Coverage:** Currenteligiblecataloghas21signaljunctionsTown01–04,only7representedbyretainedstaticTRAINpairs. Town10HDallowed134/664/841/895arestop/unsignalized; allsignaljunctionsreserved. Do NOTborrowreservedTown10locations. PrepareTRAIN-onlynewapproachesandtwofixedilluminations, retainoriginal144VAL/72labels andeveryoldfailedfit. Protocol navigation_signal_diversity_protocol_20261021.md, no newnativecaptureyet.
**Validation / mistakes:**13focusedtests pass; frozenbaselineindependentlyrecounts. Watchergenericimagebranchhidupdatesteps; newstageexceptionfixrestoresfitprogress. Read-onlyCounterinitiallyusedlist-valuedcontrolandfailedTypeError; correctedstringkey. No runtime/datachangedbylookup. Allfit/auditowners terminal; onlyoriginalguardianlive. Clocknow10:37afterexecutiongap; no assumedjobprogressduringgap.
**Lesson / interview:** Diagnose TRAINversusVALbeforeextendingafailedfit. MissingTRAINvisualdiversity is a concrete nextdataaction, but a reservedlocation is never availablemerelybecauseitsappearancewouldhelp.

## 2026-10-21 10:43 - Freeze200strictTRAINsignal cases and start contained capture
**Type:** decision / execution
**Stage:** Navigation M1/M2 perception

**Geometry:** Offlineoriginalmaps/currentreservations freeze100poses/21signaljunctions/36approaches acrossTown01–04; twofixedilluminationconditions produce200plannedcases/2400RGBimages/eightnativeworkers. Town10HDhasnoeligible signalroute andisexplicitlyskipped. One.35mclearanceposeexcluded; sevenclampedduplicatesretainedasgeometryskips. Minimumacceptedclearance3.0206m. PlanSHAf92f8b70c27e9343f927c600fd6aab6ee24ed20c20a6ee5056a0f086a4ac4140.
**Recipe:** reports/navigation_signal_diversity_protocol_20261021.md. Up totwolight/laneapproachesperjunctioninrouteorder,40/20/6mbeforestopline; clearnoon/overcastlowsun,dry,seededbackgroundphases. Same3nativeRGBcameras; staticphysics-offego,targetred/green/yellow/redrepeat,exactframebarrier/cameramatrixaudit. No driving or teacher-policytrainingclaim.
**Execution:** outputs/navigation_signal_diversity_20261021RUNNING. Onefreshownedengine/nativechildpertown/illumination; originalprocesscontainment, STOPonlynormalcleanup,noretryorreplacement.2400sstartup/capturecap,300s/child,1GBartifactcap, original14GBnet/10GBminguardian. Closedpercasereceipts survive laterchildfailure; everyplannedcase retainedasobservedorunknown. NativeSIGSEGVrisk remainsunresolved. Noautofit; independentafter-closureaudit required.
**Source / validation:** ALLcurrent src/navigation pluscollector/renderhelper/configsources pinneduntilcapture/auditterminal; do NOTedit.37focusedtests passplus4afterresourcecallfix. Prelaunchinspectioncaughtwrongcheck_resourcescall signature; fixedbefore anynativeexecution, addedVRAM/RAMcheck. No jobs scriptsedited, originalM4primary/VAL/M3/M6/rewardinputsunchanged.
**Lesson / interview:** Moredataisusefulonlywhenitsgeographyanddenominatorarefixedbeforeoutcomes. KnownTown10appearancegapsdonotjustifyborrowingheld-outlocations; useeligibleTown01–04diversityanddisclose thelimitation.

## 2026-10-21 10:52 - Audit 164 new phase cases and test the added TRAIN diversity
**Type:** result / decision / execution
**Stage:** Navigation M1/M2

**Collection:** All eight frozen tasks settled in319.84s. Town01 clear-noon fails an immediate native weather-state check before images; retain36unknown cases, no retry. Remaining seven children exit0 and capture164cases. Native exit codes are one−11 and seven143; the owner's original abnormal-exit counter8 is retained, not described as eight SIGSEGVs. Original collection status remains FAILED. Cause of the weather mismatch is open; actual mismatch values were not saved, a diagnostic limitation.
**Independent audit:** After exact ownerPID2206711exit, verify164/200cases,1968RGBimages, allphase/background-light states, frame ordering, camera/pose tolerances, physical light correspondence, independently reprojected boxes and actual1mclearance. Zero additional quarantines,36unknowns. All21locations represented; overcast100/clear64. Stable masks221red/green,219yellow, allnonempty. Separate manifestSHAf400408ae0ce7f53bbf16326592742b4df0ccdc4e79c3f16cc864cb3a7a8bdac; original failure artifacts unchanged. Audit7.08s.
**New registered fit:** Append1968verifiedTRAINimages to876prior:2844TRAIN/21locations, same144VAL/72labels byte-identical including background ignore masks. Positive phase copies626red/309yellow/313green. Fresh37772parameter bulb CNN, same seed42/1000updates/lr.001/loss/crops as old study; only data changes. Protocol navigation_signal_diverse_fit_protocol_20261021.md. Source/data preflight and10focused tests pass; no unknown cases fitted.
**Current:** outputs/navigation_signal_diverse_fit_20261021 RUNNING;1200sfit/1500sstage/150MBcap, original capacity guardian. Require parent44joint and alltown/state coverage plus90%background reduction for useful component gate. Keep common loss selection and every checkpoint; no extension or automatic native pilot. Runtime collection/audit source pin released only after both exact owners exited; this new fit has its own archived source hashes.
**Mistakes:** First dataset patch rejected due a missing '+' in the patch body; no edits applied, corrected before preflight. Initial resource-helper signature error was caught before native launch. No jobs script or benchmark/reward/input changes. M2/M3/M5 unqualified;M6 not run.
**Lesson / interview:** A failed collection can yield a separate verified subset without erasing its failures. Test a data countermeasure with the same training recipe and unchanged validation bytes before attributing improvement to it.

## 2026-10-21 10:55 - More signal diversity does not pass the unchanged recipe
**Type:** result / decision
**Stage:** Navigation M2/M3

**Result:** Fresh1000update data-only comparison completes22.49s, fit/evaluation6.38s. Common selection loss chooses0; latest1000localizes61/72,43joint,88,379backgroundpeaks and4.078%activebackground. Priorselectedbulb56/44/37,241/1.654%. No promotion, extension or native rollout. LatestTown10joint9versusparent10; Town01/03/04remain7/15/12. Foregroundloss2.661versusold1.478; backgroundCE.1391versus.07156.
**Verification:** Initialmodeltensors andoriginalVALpredictions exactlymatchpriorseed42. Independentclosed-predictionmatchingreproducesallfivecheckpoints'localized/joint/background/town/statecounts; original144VAL/72labels/ignorebytes unchanged. Allmodel/optimizer/RNG/checkpoint/predictionartifacts retained. Reportandgraph reports/navigation_signal_diversity_20261021.md include164/200captureaudit,36unknownweatherfailures,21locationcoverage andallnativeexitcodes.
**Interpretation:** AdditionalTRAINdataaloneunderthisfixedbudgetdoesnotresolveconfidence/backgroundconfusion. Morelocalizedpointsdonotjustifyloweringthegate. Scratchcontextverifieralsorejected. NextinspectsemanticforegroundrepresentationusinganexistingpretrainedRGBbackbonewithseparateprospectiveprotocol; no claimitwillwork andnoTown10reserveddatause. Signalapplicability/stoprelease/memoryandM5planningstillrequireindependentevidence.
**State:** Allnative/fit/auditowners terminal, originalstorageguardianonly. No jobs scripts edited, no benchmark/reward/interface changes or M4primaryreselection. M2/M3/M5unqualified;M6notrun. Activegoalcontinues. Sourcepinforthecompletedfitreleased; all exactarchives retained.
**Lesson / interview:** A larger, cleaner corpus is an artifact improvement, not a model-performance claim. A controlled same-recipe comparison can show that data diversity alone is insufficient, and the selected checkpoint must remain fixed even when a later metric looks better.

## 2026-10-21 11:15 - Start frozen pretrained and RGB foreground verifiers
**Type:** decision / execution
**Stage:** Navigation M2/M3 perception

**Protocol:** reports/navigation_signal_pretrained_verifier_protocol_20261021.md. Originalbulbproposalcheckpoint/threshold/all38543VALproposalsunchanged. Broader2844TRAINimages/21locations, same144VAL/72knownhousings. FixedTRAIN-onlymining: allknownpositiveproposalsplusbulbanchors,16negativeproposals/image, unknownregionsignored. NoVALmining, originalphasekeptatoutput.
**Representation:** SamecurrentRGBnative32/context96averaged32crops; twofixedbinarylogisticheads(balancedclasses,C1,maxiter1000):87RGBstatistics versus1024frozenImageNetResNet18features. WeightSHAf37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec, noCARLAlabels; BF16forward/FP16cache, allencoderweights/buffersfrozen. Changesrepresentation/task/datarelativeoldcontextmodel, not an isolatedcausalpretrainingclaim. No arbitrarycheckpointchoice, thresholdsearchorVALfit.
**Gate / budget:** Bothheadsreportedseparately; preserve44jointandalltown/statecounts,≥90%backgroundreduction. Everyproposal scored, no runtimecap. Noqualifieddriver/nativepilotfromcomponentgate.1800sfeature/600shead/3000sstage/400MBcap, existingimmutableguardian. EachblockingCPUfitisownedchild; STOPthen normalSIGTERMonlyifneeded, noSIGKILL. Exportedlinearprobabilitiesmustmatchsklearn.
**Current:** outputs/navigation_signal_pretrained_20261021RUNNING; allregisteredsourcebytespinneduntilterminal.9focusedtests passincludingfrozenBN/parameters andRGB-onlyinput/originalstatepreservation. Unusedimportcaught/fixedbeforelaunch. No jobs script/benchmark/reward/interface/M4primarychange. M2/M3/M5unqualified,M6notrun.
**Lesson / interview:** Reuse an explicitly frozen pretrained visual prior as a distinct engineering hypothesis, while keeping its extra externalsupervision and changedclassificationtask visible. The frozenproposal ceiling stilllimitsrecall.

## 2026-10-21 11:24 - Reject frozen verifiers; register a paired crop-scale test
**Type:** result / diagnosis / decision
**Stage:** Navigation M2/M3 perception

**Result:** Both frozen verifier fits finished in 87.22 seconds. RGB gives 38/72 joint lights and 9,176 background peaks; ResNet gives 26/72 and 789. Parent is 44/72 and 37,241. Both fail coverage protection; no promotion or driving rollout. All parameters/BN buffers remained frozen; exports reproduce fitted probabilities within 1e-12.
**Diagnosis:** Independent full-metric recount matches stored results. Parent/RGB/ResNet joint counts by native housing height: under 8 pixels 22/19/12 of 40; 8–15 pixels 18/16/10 of 28; >=16 pixels 4/3/4 of 4. All 18 ResNet losses occur below 16 pixels, but location and scale are confounded. Report and graph: reports/navigation_signal_pretrained_20261021.md.
**Decision:** Register one 8/16-pixel crop comparison with the same frozen ResNet, pooling, 1,024 features, TRAIN points/labels, linear-head recipe and gates. Repeat RGB as an exact-parity control. No validation threshold search, geographic borrowing or native deployment. Protocol: reports/navigation_signal_tight_verifier_protocol_20261021.md. Original study/source archives retained; current runner gains an explicit optional flag.
**State:** Host check finds only the original storage guardian, 18.23 GB free / 9.57 GB net used. Same original 14 GB cap / 10 GB minimum. Ego-only feedback approval was already applied and native-verified; no additional interface change. M2/M3/M5 unqualified; M6 not run.
**Verification / mistakes:** Diagnostic size recount and report completed; plotting inspected. Initial report lambda/line-length and new import-order lint findings fixed before launch; failure-gated checks prevent execution before lint passes. A lookup incorrectly expected result.json; authoritative assessment.json was read. No scientific artifact changed by these lookups.
**Lesson:** Fewer false alarms are insufficient if true-light coverage falls; a size-stratified diagnosis can motivate a controlled representation change without proving its cause.

## 2026-10-21 11:25 - Start the registered paired tight-crop study
**Type:** execution
**Stage:** Navigation M2/M3 perception

**What happened:** Eight focused crop-alignment, RGB-boundary and frozen-encoder tests pass; lint passes. Launched the optional tight-crop recipe at outputs/navigation_signal_tight_20261021. Original runner behavior remains the default, and prior source archives are retained.
**Scope:** Same 1,800-second feature / 600-second head / 3,000-second stage / 400 MB limits. Source hashes remain pinned until terminal. Both RGB and ResNet outcomes will be retained; no automatic native deployment, threshold changes or extension.
**Progress:** `.venv/bin/python scripts/watch_navigation_execution.py`; stop with this output directory's STOP. Original storage guardian remains active with its unchanged baseline.

## 2026-10-21 11:32 - Reject crop-only fix and select the next hypothesis on TRAIN locations
**Type:** result / diagnosis / decision / bug
**Stage:** Navigation M2/M3 perception

**Result:** Tight crops finish in 89.46 seconds: 33/72 localized, 28 joint, 2,250 background peaks; large crops 29/26/789, parent 56/44/37,241. Coverage still fails. Repeated RGB head and probabilities match bit-for-bit; TRAIN rows/labels, features and validation offsets match. All source hashes remained fixed. No model promotion.
**Diagnosis:** Predicted-point TRAIN/VAL balanced accuracy is 99.72%/72.91% for large crops and 96.63%/75.06% for tight crops. TRAIN/VAL foreground recall is 100%/47.95% and 97.24%/56.16%. RGB is 69.42%/72.61%. Remove GT-only anchors from this diagnostic; backgrounds remain sampled in TRAIN. Report/plot: reports/navigation_signal_tight_20261021.md. This supports testing generalization, not blindly extending an already well-fitted head.
**Decision:** Register TRAIN-only whole-junction regularization selection: three internal folds across 21 locations, five fixed C values, macro junction/class-balanced OOF logloss on predicted points, then one final fit/VAL assessment. Reuse large-crop feature cache; preserve all component gates and original validation. Protocol: reports/navigation_signal_regularization_protocol_20261021.md. No reward, benchmark or allowed-input change.
**Infrastructure:** Future signal collections now persist expected/actual weather and deltas before failing the unchanged 0.001 tolerance check. Three focused tests pass, including the missing-receipt regression. No native retry, and the original 36 unknown cases remain unknown. Source was changed only after all collection/audit/fit owners exited.
**Validation:** Eleven crop/verifier/research tests pass. Reports independently recount saved predictions; figures inspected. Report-header and import formatting lint findings fixed before execution. Original storage baseline remains active, about 18.08 GB free / 9.71 GB net used; no artifacts deleted. M2/M3/M5 remain unqualified; M6 not run.
**Lesson:** Near-perfect TRAIN performance is evidence against extending the same fit; choose regularization using entire TRAIN locations to avoid correlated-frame leakage.

## 2026-10-21 11:32 - Start bounded TRAIN-junction regularization selection
**Type:** execution
**Stage:** Navigation M2/M3 perception

**What happened:** Lint and five grouped-selection/research-boundary tests pass. Host check confirms only the original storage guardian before launch. CPU study starts at outputs/navigation_signal_regularized_20261021 with 15 internal-fold fits and one final fit. All 21 locations remain whole within folds; selection excludes GT-only anchors and original component validation.
**Scope:** Same cached TRAIN features/labels and component coverage gate. Fixed candidate list, 180-second child / 1,800-second stage / 60 MB cap. No native driving, duplicate queue, automatic extension, reward/input/benchmark change or artifact deletion. Sources pinned through closure; live progress uses the existing navigation watcher.

## 2026-10-21 11:36 - TRAIN-only regularization helps but still fails signal coverage
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** All 16 CPU fits converge in 42.49 seconds. Three whole-junction internal folds select C=0.0001 from five registered values before final fitting/VAL assessment. Mean location-balanced OOF loss for C=.0001/.001/.01/.1/1 is .299255/.304612/.446382/.847789/1.847459. Final head: 33/72 localized, 30 joint, 1,919 background peaks; previous ResNet 29/26/789, parent 56/44/37,241. Coverage fails despite 94.85% fewer background peaks. No promotion, extension or driving.
**Independent verification:** Recount all 15 OOF vectors against their fold-head predictions, verify all 21 locations remain whole, reproduce class/location losses and selection, check all 16 convergence receipts and selection before final fit. Recount final saved predictions reproduces every metric. Report/graph: reports/navigation_signal_regularization_20261021.md. The selected C is at the tested lower boundary; no claim of exhaustive optimum or posthoc range extension.
**Interpretation / next:** Stronger regularization improves the verifier but leaves a major generalization gap, especially Town03/Town10. Inspect representative RGB errors and assess TRAIN-only appearance augmentation or feature adaptation with the internal location folds before another component comparison. Keep original component VAL and all reserved geometry unchanged; no longer unchanged fit is justified by convergence alone.
**State:** Fit/report owners terminal; original storage guardian remains active, 18.07 GB free / 9.72 GB net consumed against the same 14 GB cap / 10 GB minimum. No jobs scripts edited, no Git push, no artifact deletion. The approved ego-only feedback remains applied/native-verified. M2/M3/M5 unqualified; M6 not run; full goal remains active.
**Validation:** Five selection/boundary tests passed before launch; source hashes checked before every fit and after closure. Report lint passed before execution and graph inspected. Earlier minor line-length findings fixed before launch. Full historical suite was not rerun for isolated offline studies.
**Lesson:** Select regularization on entire training locations, then preserve the failed result when improved internal validation still does not meet the driving component's coverage gate.

## 2026-10-21 16:46 - Inspect signal pixels and register nonlinear RGB augmentation
**Type:** diagnosis / decision
**Stage:** Navigation M2/M3 perception

**Progress classification:** Previous goal turn completed the paired scale and grouped-regularization studies, rejected both and changed the next hypothesis. This is progress, not a blocked or idle turn. Live host check confirms only the original storage guardian before new execution.
**Inspection:** Fixed-order gallery selects first lost parent-correct light per VAL town/state, then first red housing at each TRAIN location: 26 crops. Some valid housings are only 2x4 pixels. Direct source PNG and dataset-loader inspection agree. Town03/356 TRAIN paired pixel changes are real: red [255,199,183] versus [194,185,175] in green/yellow; other independently responsive yellow/green pixels are present. No image corruption or basis for relabeling was found. Gallery reports/navigation_signal_error_gallery_20261021.png/json; not a new benchmark.
**Decision:** Test a nonlinear 87-feature RGB baseline with and without gamma .7/1.4 augmentation, motivated by poor linear RGB TRAIN fitting and ResNet context overfitting. Apply monotone channelwise transforms before context averaging; no spatial or phase labels changed. Same original TRAIN points, original features exact-parity control, fixed 100-iteration histogram boosting. Whole TRAIN-junction folds choose the arm; only the chosen arm reaches component VAL. Protocol reports/navigation_signal_rgb_boost_protocol_20261021.md.
**Integrity / budget:** Preserve all old failures and 144 VAL/72 labels, original phase proposer, every town/state gate, reserved locations and M4 primary. Seven owned CPU fits, 180s each /1800s stage /100MB; original storage baseline unchanged. No new native recordings, benchmark/reward/input changes or automatic deployment. Report-title lint finding corrected before gallery execution.
**Lesson:** Inspect native pixels before another model change: a real single-pixel phase response calls for retaining fine RGB detail, while internal location folds keep adaptation separate from component validation.

## 2026-10-21 16:47 - Recover missing storage guardian before unstarted RGB fits
**Type:** blocker / recovery / execution
**Stage:** Navigation infrastructure / M2/M3

**What happened:** Clock advanced from about 11:40 to 16:46 during an execution gap. The first RGB-study launch was refused by BudgetWatch at entry because the guardian heartbeat was stale; zero images and zero fits, no request/cache/model files. Original outputs/navigation_signal_rgb_boost_20261021 remains FAILED. Cause of the external guardian exit is unknown.
**Authoritative check:** Host anchored process lookup at 16:46 finds no training, native or guardian owner. Original heartbeat stopped around 11:46; do not treat its stale running status as liveness. The earlier 11:37 host check was not current at the late launch; the runtime guard prevented unsafe work. Lookup of nonexistent baseline.json was corrected to the authoritative request.json.
**Recovery:** Resume the same guardian source/plan with --resume; original baseline 27,792,580,608 bytes, 14 GB net cap /10 GB minimum and effective floor13,792,580,608 unchanged. Fresh heartbeat and 18.10 GB free verified. Fresh invocation outputs/navigation_signal_rgb_boost_v2_20261021 receives the empty failed preflight's original absolute 1800-second deadline; no wall-budget reset, scientific retry, model repeat or artifact deletion.
**Execution:** Seven focused augmentation/group/boundary tests and lint pass. RGB study now runs with all seven fits originally unstarted. Exact source/archive hashes remain pinned until terminal. Prior study and all original model failures unchanged; no jobs scripts, benchmark/reward/input changes or native deployment.
**Lesson:** A state file saying running is not liveness; enforce a fresh heartbeat at runtime and preserve both failed preflight and original resource budget during recovery.

## 2026-10-21 16:54 - Nonlinear RGB reaches 41 joint lights; three yellow losses remain
**Type:** result / diagnosis / decision
**Stage:** Navigation M2/M3 perception

**Result:** Seven fixed-budget CPU fits complete in79.65s, including feature extraction. Whole TRAIN-junction selector chooses gamma augmentation: macro OOF logloss .360422 versus plain .390922. Final model47/72localized,41joint,1696background peaks versus parent56/44/37241. Background reduction95.45%, but coverage FAILS. No driving, promotion, threshold change or unselected-arm VAL assessment.
**Coverage:** Town01/03/04 joint7/15/12 equal parent; Town10HD7 versus10. Red22 and green7 equal parent; yellow12 versus15. Three lost parent-correct yellow samples362/374/382 each have one in-box proposal with foreground probability .300916/.108676/.137065; all are filtered at .5. Restoring original proposal order produces identical metrics. This diagnosis cannot justify moving the threshold or dropping labels.
**Independent verification:** Reproduce all six OOF vectors from saved fold predictions, whole-location separation, fitted row/augmentation counts, group/class aggregation, selected arm and selection before final fit. Recount all final metrics. Original51,094RGBfeatures reproduce exactly; gamma final fit153,282rows; serialization predictions match. Seven focused tests pass. Original empty preflight failure and original deadline retained; no scientific fit repeated. Report/graph: reports/navigation_signal_rgb_boost_20261021.md.
**Limits:** TRAIN OOF foreground recall remains low: gamma red58.16%/green55.18%/yellow54.00%. Single-pixel phase responses are real, but neither static component accuracy nor95%fewer background detections proves safe driving. Previous ResNet internal loss .299255 was better despite worse VAL joint coverage; these engineering studies do not reselect M4's primary.
**Next:** Register a state-aware RGB supervision comparison against a binary classifier with matched tree capacity, using the same TRAIN-location folds and augmented features. Evaluate whether preserving phase distinctions improves foreground decisions; no unsupported claim it will recover the three lights. Preserve all old gates/failed outputs, reserved geometry and final Town05.
**State:** All fit/report processes terminal; host verifies only resumed original guardianPID2309550, tool session49466. Same baseline27,792,580,608B/14GBnet/10GBminimum, ~18.08GBfree. Source pins released after closure; archives retained. No jobs scripts edited, no Git push, no deletion. M2/M3/M5 unqualified; M6 not run; active goal continues.
**Mistake:** One read-only diagnosis imported the plotting helper without MPLCONFIGDIR, causing a temporary-cache warning; later report uses the explicit /tmp matplotlib cache. No data/model changes resulted.
**Lesson:** A failed component can still yield actionable improvement; diagnose the exact remaining errors without changing the acceptance rule after seeing them.

## 2026-10-21 16:57 - Register state-aware RGB foreground supervision
**Type:** decision / implementation
**Stage:** Navigation M2/M3 perception

**Progress classification:** Previous goal turn completed the seven-fit nonlinear RGB study and narrowed the three remaining regressions to filtered yellow proposals. This is progress; no scientific gate passed and no final driver is qualified.
**Hypothesis:** Collapsing all phases into one foreground label may discard useful color distinctions. Compare fresh joint100 four-class boosting with binary400, each400trees, same87RGBfeatures/gamma cache/153,282TRAINrows and three whole-junction folds. Equal tree count is a capacity control, not equal CPU work or exact parameter count. Record both costs.
**Contract:** Joint weights allocate1/2total to background and1/6each to red/yellow/green, computed only within fold TRAIN; preserve the binary foreground prior. Phase metadata creates supervised TRAINlabels only, never model inputs. Internal selector remains macro binary OOF logloss, tie to binary; only chosen final head reaches component VAL. Joint inference sums foreground probability at the same.5gate and predicts state from RGB probabilities. Parent44joint and every town/state protection plus90%background reduction remain.
**Implementation:** Optional --state-aware mode in the existing RGB trainer references the exact augmented feature cache without duplication; old default behavior/source archives retained. Protocol reports/navigation_signal_phase_boost_protocol_20261021.md. State-label, weight-mass and inference-boundary tests added. Existing model confirms sklearn tree-count accounting. No new recordings, reserved locations, M4primary, rewards or benchmark changes.
**Budget:** Seven CPU fits,180s perchild/1800s stage/100MB, same original storage guardian baseline. Host check16:55finds only guardianPID2309550. No job script changes or push. Source becomes pinned when launched; full M0–M6 goal remains active.

## 2026-10-21 16:58 - Start the state-aware RGB comparison
**Type:** execution
**Stage:** Navigation M2/M3 perception

**What happened:** Lint and ten phase-label/weight-mass/augmentation/group/research-boundary tests pass. Started outputs/navigation_signal_phase_boost_20261021 using the verified existing augmentation cache. Three internal folds per arm, one selected final head; all original component gates unchanged.
**Scope:** No native driving or feature-cache duplication. Registered source hashes pinned through terminal; original storage guardian remains fresh. STOP only for owned study; no automatic deployment or extension.

## 2026-10-21 17:03 - State-aware model matches aggregate but fails state protection
**Type:** result / diagnosis / decision
**Stage:** Navigation M2/M3 perception

**Result:** Seven fresh fits complete48.71s; TRAIN-location OOF loss selects joint100(.360094) over matched400treebinary(.402281). Joint model45/72localized,44joint,1171backgroundpeaks (96.86%lower). Aggregate44matchesparent but gateFAILS: red22unchanged,green7→12,yellow15→10;Town03joint15→16,Town10HD10→9. No state/town compensation allowed. No native deployment or threshold change.
**Verification:** Independent audit reproduces every OOFvector, whole-location separation, sample counts, macroselector, selection beforefinalfit, all7actual400tree receipts, normalizedfourclass probabilities/foregroundmarginals andfinalmetrics. Actualfoldtimes binary4.72/5.19/4.97s versusjoint5.05/5.24/5.30s; finaljoint7.84s. ModelSHA08086922079b829287116df691982a8e3295108091898cec28ce020e3ea0aa53. Report/graph reports/navigation_signal_phase_boost_20261021.md inspected.
**Next registered comparison:** Same joint100recipe and exactcachedgamma data; raw87RGBcontrol versus312features adding75channel differences,75chromaticities and75local contrasts. WholeTRAINjunction folds chooseonehead beforeunchangedcomponentVAL. Rawfoldpredictions mustreproducepriorjoint100exactly; ifrawselected,finalVALmustmatchtoo. Protocol reports/navigation_signal_opponent_protocol_20261021.md. Ten focusedtests pass, including RGBfeatureinvariants, stateweightsandboundaries. No new superviseddata, metadata input or hue label manipulation.
**Integrity:** Old source archives andallfailures unchanged; currenttrainer adds explicitoptional--opponent-colors mode onlyafterpreviousownerterminal. Source pins beginatlaunch. Same7CPUfits/180s each/1800s stage/100MBnewartifacts, originalguardianbaseline14GBnet/10GBminimum. No jobs scripts or benchmark/reward/interface/M4primarychanges. M2/M3/M5unqualified,M6notrun.
**Lesson:** Matching aggregate accuracy can still hide a safety-relevant state regression; preserve the per-state protection when choosing the next representation.

## 2026-10-21 17:05 - Start exact-control RGB opponent representation comparison
**Type:** execution
**Stage:** Navigation M2/M3 perception

**What happened:** Ten focused feature/state/group/boundary tests and lint pass. Host check17:03finds only original guardian; launched outputs/navigation_signal_opponent_20261021. Three raw-control folds must match prior joint100class probabilities bit-for-bit; the selected final raw control would also need exact priorVALparity. Features from cachedRGBonly, no model metadata inputs or new data.
**Scope:** Same7CPUfits/400trees each, original gamma cache and TRAIN-location selection. 180sfit/1800sstage/100MBcap; no automatic deployment, threshold changes, benchmark changes or native trials. Sources pinned until closure; exact old artifacts retained.

## 2026-10-21 17:09 - Reject relative-color representation; verify exact control replication
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** Seven fits complete94.91s. TRAIN-junction OOF selector retains raw87RGBfeatures(.360094logloss) over312features with channel differences/chromaticity/localcontrast(.368144). Unselected opponent model never reaches componentVAL. All three rawfoldclass-probability arrays and selectedfinalTRAIN/VAL arrays match the previousjoint100study bit-for-bit; metrics unchanged45/72localized,44joint,1171background, stillFAILyellow/Town10protection.
**Independent audit:** Reproduce all six OOFvectors and class/location aggregation, verify whole-location separation,400treeandfeature-count receipts, selectedarm and selectionbeforefinalfit. Rawfoldtimes5.22/5.25/5.31s versusopponent19.16/21.44/20.94s; equal trees did not mean equal CPU cost. Report reports/navigation_signal_opponent_20261021.md/json. Ten focusedtests passed beforelaunch; reportlint/recount pass.
**Next:** Prospectively test retaining the frozen native CNN proposer's32learnedfeatures and4class probabilities in the compact verifier, alongside RGB. Current second-stage classifier discards that learned visual representation. Reuse all originalproposal identities/labels/augmenteddata, select on TRAINlocations, and preserve gates. No assumption this will solve the gap; no newfit registered or started yet.
**State:** Both state-aware and opponent studies are terminal, sourcepins released; exact archives retained. Only original guardian remains. Same immutable14GBnet/10GBminimum budget, no feature-cache copies, artifact deletion, jobs script edits or push. M2/M3/M5 remain unqualified; M6 not run. Full goal remains active.
**Lesson:** Respect an internal selector even when another representation might look attractive after development errors; exact control replication and leaving the unselected arm's component results unopened make the negative result interpretable.

## 2026-10-21 17:21 - Prepare frozen-proposer feature fusion; host review unavailable
**Type:** implementation / verification / blocker
**Stage:** Navigation M2/M3 perception

**Previous turn:** Progress: state-aware and relative-color comparisons completed, independently audited and retained as failed. No final driver is qualified. This turn prepares the next representation rather than repeating failed fits.
**Implemented:** src/navigation/signal_proposer_features.py verifies the exact original checkpoint SHA, freezes all weights, extracts32learned channels plus4probabilities from full nativeRGB, and gathers every original XY point without appending coordinates. Full-image extraction preserves convolution boundary behavior. Staged cache runner scripts/cache_navigation_signal_proposer.py verifies original/augmented RGB feature parity, source/data/point identities, model immutability and same-batch forward parity; records discrepancies against older saved scores instead of assuming batch/precision equality. No cache or model fit executed.
**Protocol:** reports/navigation_signal_proposer_fusion_protocol_20261021.md registers raw87 versusfused123features under the unchanged joint100recipe, originalTRAINlocation folds, selected-onlycomponentVAL and allstate/town gates. Upstream proposer TRAIN exposure is disclosed: these folds validate the NEW head, not an unseen end-to-end representation. Cache requires independent audit before fitting. Same original storage baseline14GBnet/10GBminimum; cache1800s/150MB and study1800s/100MB, no automatic deployment.
**Verified:** Nine CPU tests pass in0.82s: exact original classifier probability parity on native-sized RGB, all weights unchanged, frozen train mode, edge/XY/order/duplicate/empty point handling, wrong checkpoint/input rejection, phase and research boundaries. Lint and CLI import/help pass. No GPU cache, native latency or driving claim. One line-length warning fixed before CLI/test execution.
**Host review blocker:** The final host process/storage check in the preceding turn was NOT executed: automatic approval review failed because of the account usage limit, with retry text9:41PM. This is not an unsafe-action determination. Do not bypass review or treat the failed check as liveness evidence. Last successful host check17:03saw guardianPID2309550only; subsequent study handle49227exited0. Journal17:09statement that onlyguardianremains lacked a fresh final host check and is corrected here. Current guardian heartbeat is fresh at17:20, but that alone is not a new host owner inventory.
**State:** No new runtime jobs started, no sourcepin active for a newstudy, no jobs scripts edited/deletions/push. Existingmodelarchives unchanged. Ledger marks preparation only and requires host inventory/GPUreadiness before extraction. M2/M3/M5unqualified,M6notrun; fullgoal remains active. Meaningful local work remained possible, so no blocked-goal declaration.
**Lesson:** Preserve learned visual evidence for a controlled test, and distinguish a tested extractor from an executed experiment. A review failure must never be silently converted into a passed readiness check.

## 2026-10-21 17:22 - Host review recovers; start frozen-proposer cache
**Type:** recovery / execution
**Stage:** Navigation M2/M3 perception

**Previous turn:** Meaningful progress prepared/tested the exact frozen extractor and cache runner; no host bypass or newfit occurred during review unavailability. Current host escalation succeeds: only guardianPID2309550, GPU10,570MiBfree, disk18.03GBfree. Prior usage-limit review failure remains documented, but is no longer blocking this check.
**Execution:** Started outputs/navigation_signal_proposer_features_20261021 under original BudgetWatch baseline. Full-image nativeCNN extraction, original51,094TRAINpoints×3versions and38,543VALpoints; same-forward and originalRGBfeature parity, frozenweights, exactsource/data hashes. No new proposals, labels, model updates or driving. 1800s/150MBcap.
**Source pin:** All src/navigation modules and registered script/protocol dependencies stay unchanged until cache terminal; independent audit follows before any classifier fit. Original source archives retained. No jobs scripts edited, reward/benchmark/interface change or finalTown05test. M2/M3/M5unqualified;M6notrun.
**Lesson:** Revalidate a failed readiness action through the normal approval mechanism; a recovered check authorizes progress without erasing the prior unexecuted check.

## 2026-10-21 17:24 - Correct tuple-versus-JSON identity preflight without changing data
**Type:** bug / recovery / execution
**Stage:** Navigation M2/M3 feature extraction

**Failure:** Original cache owner exited1 after13.21s, zero images/features, ValueError Original ordered data identity changed. Failed output/source archives preserved. Dataset inspection verifies all2988sample records semantically identical and TRAIN/VALindices exactly equal; first difference is in-memorypoints[(145,72,1)]versusJSON[[145,72,1]]. Cause is comparing Pythoncontainer types rather than the recordedJSONrepresentation, not a data/split change.
**Fix:** Compare canonicalJSONincluding every field/orderedpoint/role/index, with finiteJSONrequired. Regression test accepts tuple serialization while rejecting point swaps, reordered samples or changed split indices. Independent audit implements its own JSONcomparison. Eight focused extractor/audit/research tests pass; lint passes. No originalrecord or scientificlabel edited.
**Continuation:** Authoritative oldhandle27826exited1; host check sees onlyguardian. Start originally unperformed extraction at outputs/navigation_signal_proposer_features_v2_20261021, using originalfailedprogress timestamp-minus-elapsed+1800absolute deadline. No allowance reset, existingfeature/model retry, deletion or benchmark change. Allcurrentnavigation source repinned until terminal; independent67forwardauditstaged, notyetexecuted.
**Mistake:** Initialpreflight lacked JSONroundtrip coverage; fixed before any GPUfeatures. Audit long-stringlintfinding fixed beforeexecution. Fullgoalactive,M2/M3/M5unqualified,M6notrun.
**Lesson:** Compare persisted scientific identities at their serialization boundary; do not mistake tuples serialized as arrays for changed data, nor relax checks that protect ordering and roles.

## 2026-10-21 17:26 - Frozen-proposer features complete; independent audit starts
**Type:** result / verification
**Stage:** Navigation M2/M3 perception

**Cache:** v2 completes107.03s,2844TRAINimages/144VAL,51,094TRAINpoints×3versions and38,543originalVALpoints. All8676same-batch extractor/originalforwardprobabilities exact; everymodeltensor frozen; original andgammaRGBfeatures exactlymatchretainedcaches. No classifierfit or nativejob. TrainSHA4050864e202d86bb03567b9e585c6b648995849ce834c4c354ad48986b8d080c,VALSHAf8cf10e058074e06e281e8134797331bebc57647f708c5734d6d861931591d2f.
**Precision:** Newbatch1BF16/FP16cache versusolder savedproposalprobabilities hasmaxabsdifference.0137584 andzero classargmaxdisagreements atall38,543points. Originalpoints/scores/evaluationunchanged; this difference is disclosed, not overwritten. Do notclaim originalbatchnumerics bitidentical.
**Audit:** Ownerhandle85313exited0. Start outputs/navigation_signal_proposer_audit_20261021: allartifacthashes/shapes/probabilityranges, plusfixedfirstsampleper21TRAINlocations×3versions andfirstnonemptyVALimageper4towns (67independentoriginal-modelhookforwards). Sampledrecomputation is not anallpointindependentGPUrecompute. Navigation sourcebytesremainpinnedthroughaudit.
**Fit preparation:** RGBtraineroptional fusionmode is staged, requirescompletedmatchingcache/audit hashes beforefitting. A rowalignmenttest verifiesoriginal/gammafeaturesandfoldexclusions; ninefocusedextractor/audit/fusion/researchtestspass. No fit started. Sameoriginalstoragebaseline/caps, previouspreflightfailureandoriginaldeadline retained; M2/M3/M5unqualified,M6notrun.

## 2026-10-21 17:27 - Independent proposer audit passes; start registered fusion fits
**Type:** verification / execution
**Stage:** Navigation M2/M3 perception

**Audit:** Completed14.20s after cacheownerexit. All artifacthashes/shapes/probabilityranges validated;67fixedforwardrecomputations through originalNativeSignalGridwithfeaturehook matchFP16cacheexactly. Samplescover21TRAINlocations×3versions and4VALtowns; this is explicit sampledindependentrecompute, not allpointrecompute. Originalsourcegeneration/ordereddata verified. Auditowner46075exited0; allcachepins released onlyafter closure.
**Fit:** Launch outputs/navigation_signal_proposer_fusion_20261021 under registeredprotocol:raw87joint100 versusfused123joint100 (32frozenCNNfeatures+4probabilities), samegamma rows/400trees/weights andwholeTRAINjunction folds. Onlyselectedhead getsfinalcomponentVAL; rawcontrolfold/finalparity enforced. Ninefocusedtests andlint pass; rowalignmenttest ensures excludedfold rows stayoutof augmentedfit. No new benchmark/input/split/reward or native deployment.
**Scope:** Same7CPUfits/180s each/1800sstage/100MBnewartifacts, original14GBnet/10GBminimum guardian. Featurecaches referenced/hashverified, notcopied. Sourcespinnedforfit; no jobscriptedits or push. M2/M3/M5unqualified,M6notrun.

## 2026-10-21 17:34 - Frozen visual fusion rejected; TRAIN support audit identifies local-label gap
**Type:** result / diagnosis / decision
**Stage:** Navigation M2/M3 perception

**Fusion result:** Sevenfits complete55.32s; TRAIN-location selector retainsraw87features(.360094) overfused123(.365732). Rawfold/final probabilities exactlyreproduce priorjoint100; unselectedfusedhead neverreachescomponentVAL. Selected45/72localized,44joint,1171backgroundstillFAILSyellow10versus15andTown10joint9versus10. IndependentOOF/group/selection/finalrecountpasses; report/graph reports/navigation_signal_proposer_fusion_20261021.md inspected. Fusedfoldtimes7.13/7.70/7.49s versusraw5.08/5.49/5.27s.
**TRAIN-only support diagnosis:** Frozenout-of-foldrawjoint predictions across3371predictedpositiveTRAINpoints (excludeGT-onlyanchors):1365have conservativeactivebulbpixelwithin detailed5×5RGBwindow,2006do not. Foreground/joint within=1259/1253; outside=633/360. Allmatchedhousings haveactivepixels somewhere; point-to-bulbdistance can be large despite boxmembership. OriginalcomponentVAL untouched. Reports/navigation_signal_train_support_20261021.md/json retain everypoint/offset/location.
**Window coverage:** Detailed9×9/17×17/25×25/31×31wouldcontainactivepixel for1569/1974/2329/2509of3371points. Thus17×17coversonly58.6%, not an automaticfix. Active masksareconservative andscale/locationconfounded; thisdoesnotprove no othercueexists orlabelcorruption. Somepositive boxpoints maynot have locallyobservablephase. Nextaudit support/context onTRAIN before registeringnewspatialrepresentationorsupervisioncomparison; noVALrelabel/drop orthresholdchange.
**Verification:** Cache107.03s/8676exactsameforwardchecks, independent14.20s/67exactforwards; original/augmentedRGBparityandfrozenweightsverified. Originalzero-featurepreflightfailure retained, JSONsemanticfix preservesall2988records/indices. Ninefocusedextractor/audit/fusion/researchtests pass; sourcepins released onlyaftercache/audit/fitowners exited. Newdiagnosticread-onlyCPU owner90105exited0; no scientificfit repeated.
**State:** Freshhostinventory confirms onlyoriginalguardianPID2309550. Disk17.98GBfree/net9.81GBagainstsame14GBnet/10GBminimum baseline. Hostapprovalreview recovered normally; earlierfailure preserved. No jobs scripts edited, artifactsdeleted,push,benchmark/reward/input change ornative driving. M2/M3/M5unqualified,M6notrun; fullgoal remainsactive.
**Lesson:** Keepa failedrepresentation result, then test whether supervision is observable from the features actually supplied; projected-boxmembership doesnot byitself establish a localphasecue.

## 2026-10-22 11:27 - TRAIN annotation/context audit; supported-supervision comparison starts
**Type:** result / decision / execution
**Stage:** Navigation M2/M3 perception

**Finding:** Audit all3,371 predicted-positive TRAIN points. No overlapping matching boxes; outside5×5 subset2,006 has median32×68px housing, median13px/max77px active-pixel distance. 1,351/2,006 outside points have all alternate-phase local differences within red-repeat variation, versus5/1,365 inside. Zero exact patch equalities. Fixed first-row-per-state/radius gallery shows housing edges/structures receiving housing phase; conservative/noisy masks prevent a proof of unobservability. See reports/navigation_signal_support_geometry_20261022.md/json/png.
**Decision:** Preregister3TRAIN-only supported-supervision folds: exclude2,006 local-unsupported positives from fitting only, preserve them in scoring; keep allbackground/GT-onlyanchors and originalfeatures/model/settings. Compare saved rawOOFcontrol with housing-level matching and sampledBGacceptance; original pointlogloss still reported. No componentVAL/finalfit/nativepromotion in thisstage. Protocol navigation_signal_supported_protocol_20261022.md.
**Verification:** Six focusedtests pass; lint passes. Executed diagnosis source archived before formatting three lint line-length issues. Dataset pairing unique by source/task/pose/view/illumination; all diagnostic rowsTRAIN. Study pins all navigation sources until terminal. Source/feature/fold hashes retained, 180s/fit900s/stage100MBcap.
**Live state:** Oct22hostinventory initially no jobs; original storageguardian resumed unchanged baseline asPID7272/session60276. Freshprelaunchcheck onlyguardian,18.01GBfree. Study session70113. No artifactsdeleted or jobs scriptsedited; original approved ego-feedback already nativeverified. M2/M3/M5unqualified,M6notrun,goalactive.
**Lesson:** A housing label can be correct while providing unsuitable supervision for a small local classifier; test supported supervision without dropping difficult evaluation examples.

## 2026-10-22 11:30 - Supported supervision passes sampled TRAIN check; full proposal audit starts
**Type:** result / decision / execution
**Stage:** Navigation M2/M3 perception

**Result:** Three fixed-budget fits complete30.75s. TRAIN OOF joint housing detections1,141→1,144/1,248; sampledbackground1,990→201(89.90%reduction). Townjoints261/115/507/258→262/117/507/258; red578unchanged,green299→300,yellow264→266. All preregistered internal continuation criteria pass. Original pointlogloss worsens.360094→1.264219; excluded housingpoints stillscored, and this tradeoff is retained. No componentVALor finalfit/deployment.
**Verification:** Kept49,088/51,094rows, exactly2,006unsupportedpositiveexcluded; all2,229GT-onlyanchors independentlyverifiedasactiveTRAINpixels. Eachfold excludeditsoriginaljunctiongroup, same400trees/87features/gammaarrays/weights. Six focusedtests initiallypass; five supervision/boundarytests also pass. Every source/hash retained; workerexit0, sourcepins releasedafterclosure.
**Next:** Registered fullTRAINaudit outputs/navigation_signal_supported_full_20261022/session59551: all715,825frozenproposals/2,844images, sameexistingfoldheads. Mustreproduce48,865sampledfeature/probabilityrows exactly andindependentlyrecountsampledmetrics. No newfit/componentVAL/nativejob. 900s/100MBcaps, originalbudgetguardian, sourcespinnedagainuntilterminal. HostinventoryonlyguardianPID7272beforelaunch.
**Lesson:** One correct phase detection per housing is a different objective from inferring phase at every housing point; report both rather than hiding the worsened pointwise loss.

## 2026-10-22 11:35 - All TRAIN proposal verification passes; registered component evaluation starts
**Type:** result / verification / execution
**Stage:** Navigation M2/M3 perception

**Full audit:** Completed220.22s; all715,825proposals/2,844TRAINimages. Supported/control correcthousing-phase1,144/1,141of1,248, knownbackground724/30,758(97.65%reduction), unknown2,391/4,524. Everytown/phasejointprotected. All48,865sampledRGBfeaturesandbothOOFvectors bitexact; independent sampledmetricandtown/phaserecount exact. Allproposals/probabilities/offsets/models/hashesretained. No newfit/componentVAL inthisstage; sourcepinsreleasedafterowner59551exit0.
**Next:** Conditional finalprotocolregistered beforeauditclosure, nowauthorizedbyitsfullTRAINgate. One joint100allTRAINfit on49,088retainedrows×3versions=147,264examples; same400trees/settings. Assess original38,543proposals/144images/72labelsONCE. Original44joint/everytown/phase/90%backgroundgate unchanged. No runtimepromotion evenifcomponentpasses.
**Verification:** Added final-fitguard requiringcompletedhash-matching fullTRAINaudit and48,865exactrows; regressionrejectsmissing/running/alteredaudits. Tenfocusedtests pass; lintpasses. Existingworker editedonlyafterallsourceownersclosed. Newowner8139, outputnavigation_signal_supported_final_20261022; sourcespinneduntilterminal. 180s/child900s/stage100MBcaps, sameoriginalbudget.
**State:** FreshhostinventoryonlyguardianPID7272beforelaunch;18.36GBfree. No artifactdeletion, jobscriptedit, nativejob, reward/input/benchmarkchange. M2/M3/M5unqualified,M6notrun,goalactive.

## 2026-10-22 11:38 - Supported supervision removes false positives but fails unchanged recall gate
**Type:** result / verification / decision
**Stage:** Navigation M2/M3 perception

**Component result:** One finalfit/evaluation completes13.08s (fit7.05s);43/72localizedandjoint,40knownbackground,98unknown. Originalparent56localized/44joint/37,241background/1,246unknown. Backgroundreduction99.89%doesNOTqualify: yellow15→10, Town10HD10→8, total44→43. Red22unchanged,green7→11; other towns7/15/12→7/16/12. PriorRGBjoint10044joint/1,171background also retained. ModelREJECTED,noruntimepromotion. SHAed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191.
**Verification:** Independentlymatchedall38,543savedcomponentprobabilitiesagainstoriginal72housings; counts/towns/phases/gatesexact. SeparatefullTRAINaudit715,825proposals,48,865exactsampledfeature/probabilitychecks andindependentrecountalreadycomplete. Final147,264fitexamples/400trees/87features; pickleparityexact. Tenfocusedtests pass,lintclean. Report/graph reports/navigation_signal_supported_20261022.md/json/png inspected; allowners70113/59551/8139/30618exited0. Sourcepinsreleasedafterclosure, archivesretained.
**Decision:** Keep supported supervision as a useful false-positive finding, not a qualifieddriver. NextTRAIN-onlyappearanceaudit of phase separability versusbrightness/saturation/size and OOFmissedhousings, thenprospectivelyregistercorrectionif supported. No validationlabelchange,thresholdrelaxation,heldoutTown10mining orfinaltest. Repeatedcomponentinspectionisdevelopment,notuntouchedtestevidence.
**Live:** Freshhostinventory11:37:49onlyoriginalguardianPID7272;18.351GBfree,originalbaseline/capsunchanged. No jobscriptchanges,artifactsdeleted,push,nativejoborbenchmarkchange. Ego-onlyfeedbackapprovalalreadyapplied/nativeverified; no duplicateintegration. M2/M3/M5unqualified,M6notrun;fullgoalACTIVE.
**Lesson:** A97–99%reductioninbackgrounddetectionscanstillfailasafety-relevantrecallgate; aggregatecleanupmustnotconcealyellow-lightortown-specificregression.

## 2026-10-22 11:49 - TRAIN appearance audit points to candidate placement; RGB refinement starts
**Type:** result / decision / execution
**Stage:** Navigation M2/M3 perception

**Diagnosis:** Existing supportedOOFhead matches1,144/1,248TRAINhousings;104misses=58nooriginalproposal(33yellow),46allfiltered. At existing auditedactive-pixelqueries, sameheadrecognizes57/58and37/46,94/104total. This isprivilegedTRAINdiagnosisonly,notlegalruntimeinputornewperformancemetric. Medianboxsizes3×5px/no-proposal,4×11px/filtered; brightness/saturationaggregatesdonotestablishdimness-onlycause. Report navigation_signal_appearance_20261022.md/json.
**Decision:** PreregisterRGB-only refinement: retain everyoriginalpoint, add R/G/Bchannelmaxwithin5×5, geometricties/mergedduplicates. NoGT/map/phase/scorethreshold/top-Kinputs. Sameoriginal.5classifieracceptance; existingthreeOOFheads/no newfit. TRAINgate: strictlymorejoint,everytown/phaseprotected,background≤10%rawRGBcontrol30,758. Explicitrecall/false-positivetradeoff; no benchmarkchange.
**Execution:** outputs/navigation_signal_refinement_train_20261022/session51471;900s/120MBcaps,all2,844TRAINimages. Recompute48,865sampledfeature/probabilityrows,retainallold/newpoints/probabilities. Eightfocusedrefinement/supervision/boundarytestsandlintpass. FreshhostonlyguardianPID7272,18.35GBfree. Allnavigation sourcesPINNEDuntilauditclosure.
**Mistake:** First add-filepatchwasrejectedatomicallybecauseoneline lackedpatchprefix; correctedbeforeanyfilecreation/execution. No scientificresultorartifactlost.
**Lesson:** Separatecandidatecoveragefromclassifiabilityusingexplicitlylabel-deriveddiagnostics,thenimplementcandidateimprovementfromlegalRGBalone. M2/M3/M5unqualified,M6notrun,goalactive.

## 2026-10-22 13:06 - RGB-only refinement passes TRAIN gate; component check starts
**Type:** result / execution
**Stage:** Navigation M2/M3 perception

**Result:** TRAIN audit complete317.25s; joint1,144→1,189/1,248. Towns262/117/507/258→272/124/534/259; red578→597,green300→305,yellow266→287. Knownbackground724→2,668; still91.33%below frozenrawRGBcontrol30,758andwithinpreregistered90%criterion. Unknown2,391→4,233retained. All715,825originalcoordinatesremain,1,054,818added,1,770,643total. Every48,865sampledfeature/probabilityrowexact; originalfullcontrolrecountmatches.
**Timing:** IncrementalRGBgeneration/newpointclassification279.90s total,median.0909s/p95.1848sperimage, excludesoldproposer/scoring; notfullruntimeprofile. PotentialCPUlatencyrequiresmeasurementbeforedeployment. Sourceowner51471exit0; pinsreleasedafterclosure, archivesretained.
**Next:** Previouslyregisteredconditionalcomponentprotocolactivated; existingfinalsupportedhead unchanged, original144images/72labels,38,543oldpointparityrequired, nofit/thresholdsweep. Owner99930/outputnavigation_signal_refinement_component_20261022. 300s/40MBcaps, sourcebytespinnedagain. Eightfocusedtestsandlintpass. FreshhostonlyguardianPID7272beforelaunch;18.286GBfree,originalbudgetunchanged. M2/M3/M5unqualified,M6notrun.
**Lesson:** More RGB-derivedcandidatescanrecoverrecallwithoutnewtraining, butfullfalse-positiveandlatencycostsmustbereportedandtested.

## 2026-10-22 13:12 - Refinement improves component recall to55/72 but yellow gate still fails
**Type:** result / verification / decision
**Stage:** Navigation M2/M3 perception

**Result:** No-fitcomponentcheckcomplete31.57s. Correct55/72versussupported43andparent44; background127versus40/37,241(99.66%belowparent),unknown193. Red30/36,green13/18,yellow12/18; parentyellow15stillprotectsFAIL. Townjoints7/17/12/19, allparenttownsprotected. All38,543originalfeature/probabilityrowsbitexact;54,099newpoints,92,642total. Independentoriginalscorerrecountmatchesallcounts/towns/phases. Report navigation_signal_refinement_20261022.md/json/png. No promotion.
**TRAIN residual diagnosis:** Remaining59misses=39nocandidate+20filtered;49recognizableatexistingOOFauditedbulbcoordinatesabsentfromnewcandidates. Oraclediagnosticnotruntimeinput. This supportsindependentRGBproposalsratherthananotherphasefit. Sourceowners99930/38565exit0; pinsreleasedafterclosure. Eightpriorfocusedtests pass.
**Next registered inventory:** Independent5×5channelmaxima withfixed16/255localcontrast, allplateaupixels, nointensity/hue/top-K/positiongate. Unionwithall1,770,643previouspoints; all2,844TRAINimages. No classifierinference/fit/componentVAL. Inventorygatesrequirecandidatecoverageandrecognizedoracle-membershipgain,≤15Mpoints/120MB. 600scap, originalstoragebaseline. Sevennew/focusedtestsandlintpass. Owner36309/outputnavigation_signal_global_inventory_20261022, sourcesPINNED. FreshhostonlyguardianPID7272beforelaunch.
**Lesson:** Retainingoriginalcandidatesdoesnotguaranteequalifiedsubgrouprecall; aggregate55/72doesnoteraseyellow12versus15. M2/M3/M5unqualified,M6notrun,goalactive.

## 2026-10-22 13:15 - Dense RGB candidates hit resource cap; register plateau consolidation
**Type:** result / decision / execution
**Stage:** Navigation M2/M3 perception

**Failure:** outputs/navigation_signal_global_inventory_20261022owner36309exits1after45.17s,2,164completeTRAINimages; nextimageexceeds15millionunionpointcap. No classifierprediction/fit/componentVAL. Allfailedrequest/source/progressretained, no truncation/retry. Originalinventoryserializedcoordinatesonlyatend, so no finalcoordinate/coverageartifactexists; do notclaimmeasurementofcompletedcoverage. Partial-growthreceiptlimitationidentified.
**Countermeasure:** Separatepredeclaredarchitecture: retain one actualcentroid-nearestpixel per8connectedchannelmaximumplateau, same5×5/16contrast. Everyregionkept, channelsindependent,tiesy/x, unionallpreviouspoints. Thischangesmodelcandidates, notbenchmarksoracceptance. Newprotocolnavigation_signal_plateau_inventory_protocol_20261022.md; same15M/120MB/600scaps. Per-imagepartialcountsnowincrementallyretainedforfailurediagnosis. Originalfailureunchanged.
**Verification:** Eightfocusedglobal/localproposal/boundarytests pass, coveringexactthreshold, allchannels,borders,flatimages,plateauties/concaveshapes/allregionretention. Oneimportformatlintfindingfixedbeforelaunch; lintclean. Allprevioussourceownersclosedbeforeedits. FreshhostonlyguardianPID7272. Newowner38672/outputnavigation_signal_plateau_inventory_20261022; allnavigation sourcebytesPINNED. No inference/fit/VALinthisstage.
**Lesson:** Boundproposalcountbeforeexpensiveclassification,andretainincrementalinventoryreceipts. Spatialconsolidationmustpreserveeverycandidate region andbeaseparatelyregisteredalgorithm, notsilenttruncationofthefailedone. M2/M3/M5unqualified,M6notrun,goalactive.

## 2026-10-22 13:19 - Plateau inventory passes; bounded full TRAIN classifier audit starts
**Type:** result / execution / decision
**Stage:** Navigation M2/M3 perception

**Inventory:** Complete63.69s;12,362,313unionpoints(all1,770,643old+10,591,670new),within15Mcap. Candidatehousingcoverage1,209→1,246/1,248;37previouslyuncoveredmissesgaincandidates;43/59misseshaveexactrecognizedoraclecoordinatesinRGBset. No newclassifierpredictionsoraccuracyclaim. Generation47.64stotal,median.0166s/image. Originaldensecapfailureunchanged. Owner38672exit0, sourcepinsreleasedafterclosure.
**Inference stage:** Registered navigation_signal_plateau_inference_protocol_20261022.md beforelaunch. ExistingthreewholejunctionOOFheads, allnewpointsclassified, oldprobabilitiescopiedwith48,865originalfeature/predictionrecomputations. Same.5acceptance/labels/order; strictjoint>1,189/everytownphaseprotected/background≤3,075.8TRAINgate. No fit/VAL/deployment. Outputnavigation_signal_plateau_train_20261022/session14275;3600s/600MBcap, expected445MBarrays, originalbudgetguardian.
**Verification:** Elevenfocusedproposal/supervision/boundarytests pass;lintclean. FreshhostonlyguardianPID7272beforelaunch,18.262GBfree/net9.531GBagainstimmutable14GBcap. Allregisterednavigation sourcesPINNEDuntilterminal. Watcheronly(notpinneddependency)extendedtoreportactualnewauditstageandframeprogress; no jobs scriptedit. Incrementalcompleted-image metricreceipts nowretained.
**Lesson:** Coverageinventoryisan inexpensive filter for candidate architectures, but full learnedclassificationandfalse-positivecostmuststillpassbeforecomponenttesting. M2/M3/M5unqualified,M6notrun,goalactive.

## 2026-10-22 13:31 - Stage honest point interface and verify faster numeric-tree GPU export
**Type:** verification / decision / mistake
**Stage:** Navigation M2/M3 runtime preparation

**Interface:** Stagedonly16-valuetokens carry calibratedunitray/cameraorigin/fullprobabilities/view/XY/null; nofakehousingbox/depth/stopline/applicability. Dynamicpaddingretains501detections, currentframeonlyrejectsstale/future/teacherfields. Threegeometry/causality/capacitytestspass, reprojectionverifiedacrossyaw/pitch/roll. Existingpolicy300boxlimitcannotbesilentlyreused. No deployedimports.
**Export:** Stagednumeric-treeGPUimplementationpreservesall100×4trees/FP64thresholdsandvalues. Syntheticreal-treeboundarytestpasses; actualfixedheadfirst4096TRAINrows maxCPU/GPUprobabilitydifference5.55e-16/6.66e-16,zeroacceptance/phasechanges. Batch256mediansscikitCPU3.519ms,tensorCPU8.483ms,tensorGPU.815ms;GPU4.32×kernelgain,CPUslower. Peak4.77MB. Notfullpipeline/transfer/featurelatency; concurrentCPUauditdisclosed. Report navigation_signal_runtime_staging_20261022.md.
**Mistake/recovery:** Initialsandboxprofile62780failed0.64s atCUDAavailabilityafterCPUreferencework;noGPUexecution. HostreadinessconfirmsCUDAtrueandonlyguardian/mainauditowners. AddedcontinuationguardforonlythiszeroGPUfailure; separatehostprofile39353complete2.14s, original300sdeadlinepreserved. Failedrootretained,CPUnumericsrecomputed,no fit/nativecase retry. ScopedhostexecutionneededforCUDA; no activeauditcode changed.
**State:** MainplateauTRAINaudit14275/PID40722stillrunningregisteredCPUimplementation, sourcepinsremain. Stagedmodules/ownprofileonlywerewritten; no src/navigation/jobscript/runtimechanges. Fourstagedtestspass,lintclean. OriginalbudgetguardianPID7272continues. Currentcomponent55/72failsyellow;M2/M3/M5unqualified,M6notrun,goalactive.
**Lesson:** Separate legalpointgeometryfromunobserveddepth/applicability, andmeasurefull-inputparitypluskernelcostbeforeclaimingacceleration; a fasttreeheadisnotafastdrivingpipeline.

## 2026-10-22 13:36 - Global candidate audit already fails monotonic background gate; finish evidence
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Partial evidence:** Mainowner14275/PID40722remainslive. Retainedcompleted-image receipts currently1114images and3352knownbackgrounddetections, exceedingpredeclaredwholeTRAINlimit3,075.8. Nonnegativeper-imagecountsmeanthiscandidatecannotpassregardlessofunprocessedimages. ThisisNOTa completed-denominatorresult; finalrecall/subgroups/unknownsremainpending.
**Decision:** Completealreadyregisteredfullauditto preservealltowncoverageandfailureevidence; no componentfollow-up, thresholdsweep orgatechange. Sourcepinsremainactive. Nextdiagnose proposaldistribution shift/negativecoverage fromTRAINonly afterclosure, withoutusingcomponentlabelstoselectthresholds. Exactpartialreceiptsretainhowmanyimageswereobservedatthisdecision.
**Parallel preparation:** Stagedpointinterface/tensorheadverifiedanddocumented; neitherchangesactiveauditordeployeddriver. Reportnavigation_signal_runtime_staging_20261022.md. Sameoriginalbudget/guardian; no nativejob, nojobscriptedits, M2/M3/M5unqualified,M6notrun,goalactive.

## 2026-10-22 20:28 - Complete plateau audit rejected; negative-distribution gap audited
**Type:** result / verification / decision
**Stage:** Navigation M2/M3 perception

**Result:** Full registered audit completed2,920.68s during executiongap. All12,362,313points/2,844TRAINimages; joint1,189→1,234/1,248, red618/green311/yellow305, towns288/126/554/266. Knownbackground2,668→8,197 exceeds fixed3,075.8limit; unknown7,480. REJECTED, no componentfollow-up. Original48,865sampledfeature/probabilityrows exact; everyoldpoint retained. Owner14275exited0; sourcepinsreleasedonlyafterclosure.
**Independent audit:** Separatematcherrecountsalllabels/background/unknowns/towns/phasesexactly.5,529/8,197FPsarenewcoordinates;only201coincidewitholdsampledTRAINbackground. MedianFPconfidence.850,p90.994;5,111red/370green/2,716yellow. Fixedfirstsample/coordinatepertown/classgalleryshowsfoliage/signs/building/roadedges. Coverage is notcausalproof. Report navigation_signal_plateau_20261022.md/json/png; reporter66452exited0.
**Next decision:** BroadenTRAINbackgroundsamplingfromnewproposaldistributionindependentofclassifieroutputs; whole-junctionfoldexclusioninbothminingandfit. Do notminefoldGusingaheadtrainedonheldoutfoldFthenclaimuncontaminatedFvalidation. Preservepositive/unknownlabels,classpriorandoriginalthresholds/gates. Newprotocolrequired beforecache/fit; no newtrainingyet.
**Host recovery:** Firstescalatedread-onlyowner/storagechecktimedoutinautomaticapprovalreviewbeforeexecution. Allowedone retry succeeded: onlyoriginalguardianPID7272,17.742GBfree; no worker/natives. Timeoutwasnotanunsafedecisionandwasnotbypassed. Existingguardbaseline/capsunchanged. M2/M3/M5unqualified,M6notrun,goalactive.
**Lesson:** Candidate recall can improve while its new background distribution invalidates a verifier trained on the older proposals; diagnose sampling coverage before changing thresholds or benchmarking again.

## 2026-10-22 20:38 - Expanded TRAIN negative cache complete; matched fold fits start
**Type:** result / decision / execution
**Stage:** Navigation M2/M3 perception

**Protocol:** Registered navigation_signal_background_protocol_20261022.md beforecache. Uniform64newknownbackgroundpointsperoriginalTRAINimage, seedSequence[42,sampleindex], no modelscoreselection. Unknowns/oldTRAINcoordinatesexcluded; all49,088supportedrowsretained. Extraexampleschangecoverage, nottotalBGclassprior1/2orperphase1/6. Samejoint100/400trees/seed42/hyperparameters/.5acceptance. Wholejunctionfoldsunchanged; no heldoutpredictionmining.
**Cache:** Owner64401complete163.06s,182,016extraTRAINpointsfrom2,844images, shape3×182,016×87FP16. No oldcoordinateoverlap;allknownbackground;foldcounts69,888/53,760/58,368. FeaturesSHAcd075f5f02271175d73d0a09d3589d825260a9e6f549be334ad5545ec1d11824;rowsSHA40d8e106f7c3300c2c9efa999ec3d38554197a30ef57eaea28391da57f40336b. Sourcepinsreleasedaftercacheexit0.
**Fit:** Start outputs/navigation_signal_background_fit_20261022/session29133:3CPUfits,300seach/1200sstage/100MBcap. Preflightindependentlyrechecksallnegativeeligibility/foldsand63fixedfirstlocation-phaseimages(all3featureversions). Workerappendsnewnegativesonlyfromnonheldoutfolds; receiptsverifycounts/gammaalignment.12focusedtests pass,lintclean. ScientificsourcebytesPINNEDthroughclosure.
**Qualification:** OriginalsampledOOFmetricsarediagnosticsonly; nonecanpromotethiscandidate. Laterseparatelyregisteredfull12,362,313pointauditmustprotect1,234joint/everytownphaseandBG≤3,075.8. No finalfit/componentVAL/nativejobnow. FreshhostonlyguardianPID7272beforelaunch,17.688GBfree,originalbudgetunchanged. M2/M3/M5unqualified,M6notrun,goalactive.
**Lesson:** Usemodel-independentsamplingandstrictgeographicfoldassignmenttoimprovenegativecoveragewithoutindirectheldout-labelleakage.

## 2026-10-22 20:47 - Expanded-background fits complete; full TRAIN comparison starts
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**What happened:** Three fixed whole-junction fits completed in 77.45 s. All 182,016 new negatives passed eligibility checks; 12,096 feature rows reproduced on 63 fixed images. Old sampled detection remains 1,144/1,248 and background 201; small town/phase regressions and point loss 1.2642→1.3515 are retained. No sampled qualification.
**Decision:** Launch the predeclared full audit on all 12,362,313 frozen coordinates, with exact original/extra feature checks and baseline recount. Require ≥1,234 joint, every town/phase protected, and background ≤3,075.8. No component follow-up unless every gate passes.
**Verified by:** Cache verification/assessment in navigation_signal_background_fit_20261022; audit source lint clean, only storage guardian on fresh host check. Owner 18923/output navigation_signal_background_full_20261022; 4,500 s / 600 MB cap. All registered source bytes pinned. Original budget unchanged, 17.67 GB free.
**Lesson:** Coverage-expanded supervision must be tested on the deployment proposal distribution; sampled parity is not evidence of improvement. M2/M3/M5 remain unqualified; M6 not run.
**Interview angle:** Why can unchanged sampled performance hide changes on the full proposal distribution?

## 2026-10-22 20:52 - Stage exact RGB context reuse while the full audit runs
**Type:** decision / verification
**Stage:** Navigation runtime preparation

**What happened:** Repeated 96×96 averaging dominates the candidate feature path. The staged extractor precomputes every aligned 3×3 integer average once per image, then gathers the original patches in bounded chunks. No deployed source was edited.
**Verified by:** Two focused tests pass: all spatial alignment phases, borders, singleton images, repeated points, extreme colors, every possible nine-pixel sum, exact patch bytes and FP32 features across chunk sizes.
**Decision:** Registered a 63-image TRAIN profile before measurement, using the prior fixed image list and every frozen candidate. Require exact features/probabilities; record original and staged times, with concurrent audit disclosed. Output navigation_signal_context_profile_20261022/session 68341; 600 s / 20 MB cap, original storage budget. The full background audit retains its original implementation.
**Lesson:** Reuse integer image operations before attempting approximate acceleration; equivalence and end-to-end cost are separate questions.
**Interview angle:** How can shared image preprocessing accelerate thousands of crops without changing their meaning?

## 2026-10-22 20:55 - Exact context reuse passes; compiled reduction staged
**Type:** result / decision
**Stage:** Navigation runtime preparation

**What happened:** Fixed 63-image profile completed in 146.95 s. Every patch byte, FP32/FP16 feature and probability matched across 236,284 TRAIN candidates. Median feature-plus-head time fell 975.8→343.4 ms/image; p95 1404.2→489.8 ms. Concurrent main CPU audit, decode/proposer/control excluded; not a real-time driving claim.
**Decision:** Keep the successful extractor staged; test a separate compiled sequential FP32 reduction that avoids materializing patches. One synthetic equivalence test passes across small/native images and edges; no actual-image profile yet. No registered runtime or active audit changes.
**Verified by:** navigation_signal_context_profile_20261022/assessment.json and 63 incremental receipts; owner 68341 exited 0. Current full TRAIN audit continues with original code.
**Lesson:** A measured 2.84× kernel-path improvement can still fall short of the complete system timing requirement.
**Interview angle:** Why distinguish exact acceleration from real-time qualification?

## 2026-10-22 20:56 - Fix staged JIT cache identity before real-image profiling
**Type:** bug / decision
**Stage:** Navigation runtime preparation

**What happened:** A second-process synthetic test exposed Numba cache reload failure: `ModuleNotFoundError: No module named <dynamic>`. Initial compilation had passed, so single-process testing missed the lifecycle defect.
**Cause:** The staged module loads dynamically; its cached environment has no importable module identity.
**How we handled it:** Disable disk caching for this staged implementation, preserve strict FP32/fastmath=False behavior, and measure compilation separately. Three extractor/reduction tests pass after correction. No running audit or deployed source changed.
**Next:** Registered compiled-reduction profile on the same 63 fixed TRAIN images; output navigation_signal_context_fused_profile_20261022, 600 s / 20 MB cap. Exact original features/probabilities required, no fit or VAL.
**Lesson:** JIT startup and second-process cache loading need separate checks; a warm numerical test does not prove restart behavior.
**Interview angle:** What failure only appeared when rerunning the staged compiled implementation?

## 2026-10-22 22:02 - Background coverage helps false detections but fails protection
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** Full audit complete in 3,388.32 s; all 12,362,313 points across 2,844 TRAIN images. Joint 1,234→1,232/1,248, background 8,197→4,848, unknown 7,480→7,497. Town03 red loses two detections (279→277); all four registered criteria fail. Exact 48,865 original sampled predictions and 182,016 extra feature rows verified. No component follow-up, no gate relaxation.
**Runtime evidence:** Separate compiled CPU profile passed every feature/probability on 236,284 points/63 TRAIN images; median feature-plus-head 972.8→49.1 ms, p95 1371.2→65.8 ms. Compilation 0.31 s. Staged only; decode/proposals/control excluded.
**Lifecycle:** After interruption, prior handles were missing and fresh anchored host inventory showed no active owners. Full audit terminal files show complete; no restart. Resume only original storage guardian with identical baseline; 17.21 GB free. GPU synthetic test result was unobserved, so reverified (1 pass); no scientific case repeated. Registered 63-image CUDA profile now owner 24670; full CPU report owner 19838.
**Decision:** Diagnose remaining TRAIN visual ambiguity; more negative coverage alone did not meet the gate. No driver promotion; M6 not run.
**Interview angle:** Why reject a model that reduces false detections by 41%?

## 2026-10-22 22:08 - GPU preprocessing parity passes; spatial-context comparison registered
**Type:** result / decision / mistake
**Stage:** Navigation perception and runtime preparation

**Result:** Host CUDA profile complete in 193.24 s: exact FP32/FP16 features on 236,284 points/63 TRAIN images, max probability error 6.66e-16, zero acceptance/phase changes. Median features plus head 22.87 ms, p95 30.63 ms; includes CPU preparation/transfer and returned probabilities, excludes proposer/control/three-camera scheduling. Peak 11.09 MB; compile/warm-up 0.37 s. Staged only.
**Metadata correction:** Copied profile request/assessment says concurrent_cpu_audit=true, but the main full audit had already completed before launch after interruption. Original files retained; actual concurrency was brief report generation, then cache preparation near the end. Timing values stand; no isolated-hardware or precise cross-run speedup claim.
**Decision:** TRAIN error gallery/recount supports testing spatial context: append 4×4 mean/std RGB grids at 16/32/96 pixels (288 extra features) to the same raw87. Same original+182,016 background rows, gamma versions, folds, labels, weights, 400 trees and original ≥1,234/BG≤3,075.8/full-town-phase gates. Registered protocol navigation_signal_spatial_context_protocol_20261022.md before cache/fit. Two direct-pixel descriptor tests pass.
**Execution:** Cache owner 85122/output navigation_signal_spatial_cache_20261022 (1,200 s/500 MB). All source bytes pinned. No fit or component VAL yet. Original budget guardian owner 29695.
**Lesson:** Spatially collapsed color moments can confuse unrelated textures; test richer arrangement without relaxing the held-out comparison or inventing privileged inputs.
**Interview angle:** What information does a spatial grid add beyond whole-window mean and variance?

## 2026-10-22 22:09 - Spatial cache audited; three fixed fits start
**Type:** milestone / decision
**Stage:** Navigation M2/M3 perception

**Result:** Cache complete 146.43 s: 233,110 fixed TRAIN rows, three RGB/gamma versions, 288 added spatial features. All extra negative eligibility and original whole-junction assignments verified. Independent direct-pixel moments reproduce 15,417 FP32 rows on the 63 fixed images; two focused descriptor tests pass.
**Execution:** Three fixed 400-tree/375-input fits start at navigation_signal_spatial_fit_20261022, owner 52141. Same supported rows, augmentation, per-phase weights, folds and hyperparameters; baseline artifacts retained. TRAIN sampled results remain diagnostics; original full gate protects 1,234 joint and every town/phase, BG≤3,075.8. No component VAL or all-TRAIN final fit.
**Checks:** Fresh host shows only original guardian PID 175563 before launch, 16.62 GB free. Original budget unchanged. New runner import/lint issue fixed before launch and help/import passes; all registered bytes pinned.
**Lesson:** Change one representation component with fixed data and evaluation before attributing improvement to more context.

## 2026-10-22 22:17 - Spatial fits complete with sampled recall tradeoff; full audit starts
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** Three 400-tree spatial fits complete in 508.46 s. Original sampled joint 1,144→1,138 and background 201→163; town joints 261/115/504/258, phases red576/green299/yellow263. Preserve losses; sampled results cannot qualify. Every saved model reproduces its own predictions exactly; fit example counts match original raw controls.
**Execution:** Separately predeclared full audit at navigation_signal_spatial_full_20261022, 1,200 s/600 MB. Same 12,362,313 coordinates; original plateau baseline ≥1,234 joint/everytownphase/BG≤3,075.8, not the newer lower recall. Exact compiled CPU raw extractor independently checked on original samples; all spatial original/extra cached vectors and OOF probabilities compared. No fit/VAL in audit.
**Checks:** Eleven focused extractor/supervision/boundary tests pass; lint and import checks pass. Fresh host inventory shows only original guardian before launch. Source pins released after fit closure, then new audit sources pinned.
**Decision:** Complete full-proposal comparison before judging representation. No driver promotion or M6 evaluation. Compact live watcher keeps historical failures accessible with --history.
**Lesson:** Fewer sampled false detections can accompany lost signal recall; broader proposal scoring and fixed subgroup protection decide the outcome.

## 2026-10-22 22:27 - TRAIN proposal positives have a measured coverage gap
**Type:** result / decision
**Stage:** Navigation M2/M3 perception

**Result:** No-inference TRAIN diagnostic complete in 21.49 s. There are 23,784 audited active bulb pixels; 3,467 are frozen candidate coordinates, but only 1,871 are in supported original supervision. The other 1,596 exact active candidates are absent from every original row (577 red/339 green/680 yellow). No known-background conflict. Whole TRAIN identity and original candidate hashes verified; no component VAL or score-based mining.
**Cause:** The original cache sampled at most four active anchors per housing in addition to old CNN proposals; the independent proposal generator selects additional genuine active pixels. This is a coverage gap, not proof that adding them improves the model.
**Decision:** Retain rows as diagnostic only (no_training_release=true); any fit expansion requires a prospective protocol and independent eligibility/fold check first. Full spatial audit still runs on its unchanged original supervision; no active source changes.
**Lesson:** When a proposal generator changes, audit positive as well as negative training coverage against the new coordinates.
**Interview angle:** How did you detect missing positive supervision without mining validation errors?

## 2026-10-23 17:06 - Recover terminal perception evidence and create the conference paper
**Type:** result / milestone / decision
**Stage:** Navigation research communication

**Recovered evidence:** Oct22 spatial full audit completed657.41s:1,228/1,248joint,3,221background,6,898unknown; all original town/phase/count recounts match. It fails all four original criteria and receives no component follow-up. Positive cache completed42.63s:1,596exact TRAIN positives/4,788verified features; next fit is prepared, not started. Earlier running banners are historical.
**Deliverable:** User-provided conference-template-a4.docx filled as an editable seven-page paper, with seven embedded figures, six tables and eleven references. Technical text/results filled; only author/affiliation/contact/acknowledgment values remain bold. Corrected and historical studies remain distinct; no qualified M6 claim. File: reports/conference_20261023/Auto_Thinker_Engine_Conference_Paper.docx.
**Implementation/QA:** Strict OOXML namespaces and length units normalized while preserving template styles/margins. First layout had two short spillover pages and crowded diagram labels; corrected to seven pages. Official temporary LibreOffice renderer used from RAM-backed /tmp, system/training packages unchanged. ZIP validity, all six bold fields, figures/tables, required metrics and page margins verified; pages visually inspected. Source hashes/primary reference URLs retained.
**State:** Fresh anchored host check finds no training/CARLA/guardian owner; no job script changes, training launches, reward/benchmark edits or pushes. M2/M3/M5 remain unqualified and M6 unrun.
**Lesson:** A research paper should preserve negative results, denominators and evidence boundaries instead of converting engineering demonstrations into safety claims.
**Interview angle:** How did you communicate the project’s successes without overstating autonomy?

## 2026-10-23 17:36 - User pauses execution at the actual completed state
**Type:** decision
**Stage:** Navigation M0–M6 closeout

**What happened:** User requested stopping at completed work, detailed project/interview reports and a Git checkpoint/merge/push. Goal explicitly set to paused; fresh anchored host inventories found no trainer, queue, evaluator, CARLA or guardian owner, so no STOP file was necessary.
**State:** Latest spatial full TRAIN audit remains rejected: 1,228/1,248 joint and 3,221 background. The 1,596-positive/4,788-feature cache is verified; prospective fit unstarted. M2/M3/M5 qualification remains open and M6 final driving is unrun.
**How we handled it:** Wrote reports/navigation_pause_20261023.json; preserved prior next action in the execution ledger; added pause notices to README, CLAUDE, handoff, plans, results index, runbook and read-only watcher. Historical outcomes and source archives retained.
**Verified by:** Goal tool returned paused; host process inventories empty; watcher shows PAUSED and labels retained guardian data as historical. No training, native evaluation, job-script edit, reward change or benchmark change in this closeout.
**Lesson:** A user-requested stop is an execution boundary, not evidence that scientific stage gates passed.
**Interview angle:** How did you hand off an unfinished research project without overstating completion?

## 2026-10-23 17:36 - Detailed project and interview reports verified for repository checkpoint
**Type:** milestone
**Stage:** Research documentation and reproducibility

**Deliverables:** reports/journey/summary.md explains goal, both architectures, timeline, datasets, corrected results, failed approaches, exact acceleration, hardest bugs, limits and prioritized future work. reports/interview_preparation_20261023.md includes short/long spoken explanations, 70 technical Q&As, five STAR stories, whiteboard tasks and honest resume/demo wording. README rewritten for current paused state and measured laptop setup.
**Verification:** Full test suite passed twice: 559 passed/3 skipped, latest 52.88 s with 64 upstream AMP deprecation warnings. All new-report local links resolve. Scoped Ruff passes; old unrelated legacy lint findings are not claimed fixed. Ten checkpoint formatting fixes preserve parsed behavior; watcher only adds pause/history labels.
**Vendor audit:** Tracked patches applied to clean temporary checkouts reproduce all eight modified CarDreamer/Dreamer files byte-for-byte; upstream gitlinks remain pinned. reports/vendor_patch_audit_20261023.json records parity. Intentional patched submodule working trees are documented, not unpublished vendor commits.
**Git scope:** Source, configurations, tests, protocols, compact results, figures and conference paper are checkpoint material. Raw recordings, datasets, weights and re-renderable page screenshots remain ignored. Local commit/merge and remote publication outcomes are recorded separately after the Git operations.
**Lesson:** A reproducible handoff needs the experiment's negative evidence and its dependency patches, not only a polished success narrative.
**Interview angle:** What does a software test pass prove, and what remains a native-driving qualification question?

**Checkpoint preflight:** All 428 changed/new Python files under src, scripts, tests and patches pass Ruff. All new/updated overview links resolve; git diff --check passes. The 902 candidate-path size/credential-pattern scan found no matches; raw datasets/checkpoints/recordings remain ignored. The updated read-only watcher was exercised manually and accurately shows the pause.

**Artifact whitespace:** Staged-file checking found required unified-diff context spaces and Matplotlib SVG path serialization. .gitattributes preserves those generated/patch bytes instead of stripping them or invalidating paper hashes; source/document whitespace checking remains enabled.

## 2026-10-23 17:39 - Local checkpoint merged; GitHub publication blocked by missing authentication
**Type:** milestone / blocker
**Stage:** Repository handoff

**Completed:** Scientific/source/documentation checkpoint 47e39bd created on experiment/camera-world-model-navigation; main fast-forward merged to it, preserving history. The conference paper, current README, detailed summary, interview guide, frozen suites, tests and corrected/negative evidence are included; raw outputs remain ignored.
**Blocker:** Actual noninteractive atomic push of both branches exited128: `fatal: unable to get password from user`. No HTTPS credential is available; neither remote branch was updated. This was a Git authentication failure, not automatic approval-review rejection.
**How we handled it:** Preserve the successful local merge, document the exact failed publication in reports/repository_checkpoint_20261023.md, and checkpoint this outcome. Both local branches are kept aligned. Office lock files created while the paper is open are ignored; the active document/lock was not deleted.
**Verified by:** Local ancestry/ref checks and scoped full validation; failed push output retained in the task transcript. No credential changes, force push, training launch or vendor gitlink changes.
**Lesson:** A local commit and merge do not imply remote publication; report the transport/authentication outcome separately.
**Interview angle:** How do you distinguish a reproducible local checkpoint from a published repository state?
