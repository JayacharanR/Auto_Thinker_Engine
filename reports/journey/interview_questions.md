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

- **Q:** What was the turning point in getting the car to drive?
  **A (short):** Diagnosing a stall instead of adding compute. After 150k steps the agent reached the intersection, started the turn and braked forever (0% success): CarDreamer's reward made standing still free. A 0.1/step time penalty took the fresh run to 100% eval success within 22.5k steps and 20/20 on a final 20-episode evaluation with no collisions.
  **Evidence:** journal "Stage 3 at 150k steps", "Stage 3 gate passed"; videos in outputs/eval_results/cnn_bev_tp01_150k

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

- **Q:** How do you choose a checkpoint for a self-supervised model?
  **A (short):** Not by its own training loss: in JEPA the targets come from an improving EMA teacher (or the online encoder with SIGReg), so prediction loss fell for two epochs and then rose while the representation kept improving. We keep the final weights of a fixed budget and compare methods with a downstream linear probe against a random-init control.
  **Evidence:** journal "First long-run attempt"; `scripts/train_phase2_jepa.py`

- **Q:** Why tube masking?
  **A (short):** If a patch is hidden in one frame but visible in the next, the model can copy it instead of reasoning. Masking the same spatial blocks in every frame forces it to infer content from context.
  **Evidence:** journal 2026-10-02 00:20; `src/jepa/masking.py`

- **Q:** Why turn off action conditioning in the predictor?
  **A (short):** The predictor was given the clip's own steering and speed. With spatial masking, that hands it exactly the signal the steering probe measures, so the encoder would not need to learn it.
  **Evidence:** journal 2026-10-02 00:20

- **Q:** Did the pretrained encoders help?
  **A (short):** On the right-turn task from the front camera, with two seeds: both frozen pretrained encoders (our JEPA on comma2k19 and Meta's V-JEPA2) solved it on both seeds, while the CNN trained from scratch solved it on one of two within 50k steps. V-JEPA2 reached 80% success fastest (21.7k steps on average). With one seed all three looked tied - the second seed changed the conclusion, so I report it as suggestive pending more seeds.
  **Evidence:** journal "Seed 123: both pretrained encoders learn"; outputs/comparison/.../comparison.md
  The clearest evidence is out of distribution: evaluated zero-shot at night, both pretrained-encoder agents still succeed 100% while the CNN agents leave the lane every time (0%); in heavy rain 95-100% vs 45%. Journal "Zero-shot robustness"; outputs/robustness/robustness.md and the videos.

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

- **Q:** How did you handle an unreliable simulator in multi-day experiments?
  **A (short):** CARLA 0.9.15 segfaulted on episode resets twice in one day. The trainer now checkpoints when the simulator connection is lost, and the queue restarts CARLA and resumes the arm (up to 6 attempts, one log per attempt). Rehearsed by killing CARLA with SIGSEGV mid-run: the checkpoint landed at exactly the crash step and the run finished after resuming.
  **Evidence:** journal "Comparison run 1"; `jobs/comparison.sh`, `scripts/train_cardreamer.py`

## Debugging stories

- **Q:** Why did valid MP4 recordings fail in browsers, and how did you verify the fix?
  **A (short):** OpenCV wrote `mp4v` video inside MP4 containers; the container and MIME label did not guarantee a browser-supported codec. Transcoded showcase copies to H.264 MP4 plus VP9 WebM, preserved source provenance and frame counts, and tested all 14 actual gallery clips plus all 14 direct MP4s in Firefox using native WebDriver BiDi with advancing playback time.
  **Evidence:** journal "Fixed browser playback of showcase recordings"; outputs/showcase_20261008/07_sources/build_showcase.py; validate_browser_playback.mjs; browser_playback_validation.json

- **Q:** Tell me about a hard bug.
  **A (short):** A patch meant to keep frozen encoders frozen made every parameter non-trainable after the first update, so the agent silently stopped learning; fp32 hid it, and the unit test only ran one step. fp16's GradScaler exposed it ("no inf checks recorded"). Fixed with a persistent per-parameter freeze marker and a test that checks the second update.
  **Evidence:** journal 2026-10-01 23:10

- **Q:** Any bug where the workaround everyone used was wrong?
  **A (short):** CARLA's Traffic Manager segfaulted the Python client; the previous setup simply disabled traffic. A gdb backtrace showed the crash in InMemoryMap::Load parsing the pre-built Traffic Manager map cache the server ships (Maps/TM/Town03_Opt.bin). Moving that cache aside makes the Traffic Manager build its map from the road graph, and traffic tasks work - which unblocked the whole traffic part of the roadmap.
  **Evidence:** journal "Traffic Manager segfault root-caused"; scripts/configure_carla.sh

- **Q:** What went wrong with the data pipeline?
  **A (short):** The download links were dead, the speed field name was wrong (it would have trained on zeros), telemetry timing was ~60 ms off, and per-sample HEVC decoding from frame 0 made training CPU-bound. Fixed with a one-time preprocessor aligned on recorded frame times and memory-mapped clips.
  **Evidence:** journal 2026-10-02 00:12

## Experiment design & evaluation

- **Q:** How did you turn experiment artifacts into a reproducible research presentation?
  **A (short):** Generated an editable 30-slide deck with graphs from recorded metrics, native architecture diagrams, primary-paper references and technical notes, then packaged 14 original driving videos with hashes and an offline gallery. The presentation separates camera encoder comparisons from privileged-BEV diagnostics, labels example clips, and reports the small final-stream gain, null reward ablation and late collapse with their limitations.
  **Evidence:** journal "Research presentation and portable driving showcase completed"; outputs/showcase_20261008/07_sources/build_showcase.py; validation.json; video_catalog.json; outputs/Auto_Thinker_Engine_Showcase_20261008.zip

- **Q:** How do you evaluate a driving agent beyond success rate?
  **A (short):** Per episode: return, route completion (fraction of planned waypoints passed), termination reason (collision / out of lane / destination / time-out), speed and wall clock. Route completion shows progress while success is still 0%; the comparison reports success, route completion, collision rate, steps to 80% success, and compute cost.
  **Evidence:** journal "Route completion metric"; `src/dreamer/carla_wrappers.py`

- **Q:** How do you make the encoder comparison fair?
  **A (short):** Same task, seeds, step budget, evaluation protocol, route branch (identical MLP), decoder target and adapter capacity; only the image encoder differs. Report mean +/- std over seeds, steps to 80% success, wall clock and peak VRAM.
  **Evidence:** `configs/experiment_contract.yaml`; `scripts/train_cardreamer.py` (`run_comparison`)

- **Q:** How did you make sure your encoder evaluation was not fooling you?
  **A (short):** Every probe runs against a random-init encoder of the same architecture, and selection uses only the gain over it. Splits are by driving segment, including the inner holdout that picks the ridge strength: a random-clip holdout once produced a fake +0.24 R2 gain for an untrained encoder because neighbouring clips are near-duplicates.
  **Evidence:** journal "Linear probe: 30x faster, and a leaky holdout found"; `src/eval/linear_probe.py`
  Later, a steering probe could not separate any encoder from random (R2 ~ 0 for all), so method selection moved to a speed probe where EMA beat random by +0.10 R2 and SIGReg fell below it; journal "Phase 2 results".

- **Q:** Did you change the benchmark's reward? Why, and how did you keep it fair?
  **A (short):** Yes, one term: CarDreamer's per-step time penalty from 0 to 0.1, after the agent learned that stopping mid-turn was free while a failed turn was penalised. It is applied identically to every arm, recorded in each run's spec, and verified by its effect (same-seed episodes score exactly 0.1 x length lower).
  **Evidence:** journal "Stage 3 at 150k steps", "Stage 3 retry with a time penalty"

## Trade-offs & what I would do differently

- **Q:** Did direct traffic fine-tuning work?
  **A (short):** No arm solved dense cross traffic after 50k steps: final success was CNN 30%, custom JEPA 0%, V-JEPA2 20%, with 70-100% collisions. The last three evaluations and late training episodes also remained poor, so the failure was not just one noisy final evaluation. It motivates testing a curriculum, rather than claiming pretrained encoders solve traffic.
  **Evidence:** journal "Direct dense-traffic fine-tuning finished"; outputs/comparison/carla_right_turn_hard_finetune
  Fresh 30-episode checks confirmed CNN 10 successes, custom JEPA 2, V-JEPA2 5, while simple-turn retention was 10/10, 9/10, 10/10. This points to unresolved negotiation with mostly retained turning skill, not a lost ability to drive the original route. Journal "Fresh traffic evaluation completed".

- **Q:** What did the fresh evaluation establish before the curriculum retry?
  **A (short):** The previous fine-tuned agents still completed almost every empty-intersection turn, but failed most dense-traffic episodes. The original custom-JEPA checkpoint also achieved only 5/20 on medium traffic, giving a measured starting point for the new curriculum. These observations justify testing easier traffic, but do not guarantee improvement.
  **Evidence:** journal "Fresh traffic evaluation completed; overnight medium training started"

- **Q:** How did you redesign the unsuccessful traffic run?
  **A (short):** Start again from the successful empty-intersection checkpoints, train on the built-in medium traffic task, and advance to hard traffic only after separate traffic and simple-turn retention checks pass. Preserve the best evaluated checkpoint, leave the reward and final hard benchmark unchanged, and cap the unattended run at eight hours. Baseline and validation outcomes make even a failed curriculum informative.
  **Evidence:** journal "User authorized an unattended overnight curriculum"; scripts/run_traffic_curriculum.py

- **Q:** How do you separate simulator failures from policy failures?
  **A (short):** A native CARLA crash leaves an incomplete evaluation, not a failed driving episode. Preserve its logs and retry the evaluation; only completed evaluation JSON files enter the comparison. A custom-JEPA evaluation crashed after 24 episodes and was rerun to a complete 30-episode result.
  **Evidence:** journal "CARLA crashed during fresh evaluation"

- **Q:** Has video pretraining solved traffic negotiation yet?
  **A (short):** No. At the 2026-10-04 19:21 snapshot, CNN finished 50k fine-tuning steps with 30% success and 70% collisions; custom JEPA was still running (~13.4k), with 0% success / 90% collisions at its latest 12.5k eval, and V-JEPA2 had not started. Wait for the remaining arms before changing the curriculum or reward.
  **Evidence:** journal "Onboarding snapshot"; outputs/comparison/carla_right_turn_hard_finetune

- **Q:** What confounds remain in the encoder comparison?
  **A (short):** CNN and frozen-video arms differ in architecture, resolution, temporal input and trainable capacity. The CNN feeds 5120 dimensions into the RSSM versus 2048 for frozen arms, changing the RSSM observation layer too; even the two frozen adapters have different parameter counts. The results support these encoder pipelines, while an RL random-init ViT control would help isolate pretraining.
  **Evidence:** journal "Audit: encoder comparison also changes RSSM input capacity"; saved seed-42 checkpoint shapes

- **Q:** What lifecycle risks remain after adding the queue lock?
  **A (short):** A rejected duplicate can still remove STOP and overwrite steps.json before taking the lock. The stop helper also targets trainers globally and omits standalone evaluation. These were identified by inspection and left unchanged during the running job.
  **Evidence:** journal "Audit: rejected duplicate queue can clear an active STOP request"; jobs/lib_queue.sh, jobs/comparison.sh

- **Q:** Which reported metrics still need an audit before presenting the results?
  **A (short):** Training and standalone evaluation compute route completion differently; compute statistics cover only the last process invocation after a resume. Partial-run progress may mean the last evaluated step rather than the checkpoint step (V-JEPA2 seed 42: 66,695 versus 72,500), and the manifest retains a config hash rather than the full effective config. Success and collision flags remain directly traceable to episode outcomes.
  **Evidence:** journal "Audit: metric accounting and documentation need reconciliation"; evaluator, summarize_run, manifest utility

- **Q:** What would you do differently?
  **A (short):** Save every submodule change as a patch immediately; test the second training step and resume paths from day one; validate loaders against real files before writing training code; make long jobs checkpoint on SIGTERM.
  **Evidence:** journal lessons 23:02, 23:10, 23:18, 00:12, 01:00

- **Q:** What did the overnight curriculum establish, and what should come next?
  **A (short):** Medium traffic remained unsolved after 30k steps per arm: selected CNN success was 11/30, custom JEPA and V-JEPA2 both 4/30; all failed promotion. Empty-turn retention was 10/10, 10/10, 8/10. The next proposed diagnostic gives one CNN the existing BEV traffic observation to test whether clearer traffic information supports yielding, before another expensive camera sweep.
  **Evidence:** journal "Overnight medium curriculum finished"; reports/traffic_curriculum_assessment_20261005.md

- **Q:** Why can traffic success and collision percentages sum to more than 100%?
  **A (short):** Destination and collision flags can be true on the same terminal step. The benchmark counts destination arrival independently, so V-JEPA2's 13/20 original medium successes include one collision: only 12/20 were collision-free. Preserve the recorded benchmark and report the joint outcomes; changing success or checkpoint selection requires an explicit protocol decision.
  **Evidence:** journal "Arrival success can overlap collision"; scripts/evaluate_agent.py; baseline_vjepa2_medium/evaluation_metrics.json

- **Q:** How did you test whether traffic visibility was limiting learning?
  **A (short):** The user authorized one bounded CNN diagnostic using the existing BEV observation and successful BEV checkpoint on medium traffic, with unchanged reward and actions. Its initial policy achieved 11/30 successes and 19/30 collisions; a 30k-step run will be followed automatically by 30 traffic episodes and ten empty-turn retention episodes. BEV supplies privileged traffic information, so this is a diagnostic rather than part of the camera-encoder comparison; its training result is still pending.
  **Evidence:** journal "User authorized and started the bounded CNN BEV traffic diagnostic"; scripts/run_bev_traffic_diagnostic.py; outputs/bev_traffic_diagnostic_queue

- **Q:** How did you resume training without discarding progress or duplicating jobs?
  **A (short):** First confirm with a host-level anchored process check that the old queue has exited, then inspect the saved checkpoint. The BEV run resumed from 24,969 steps with replay and optimizer state, reused its completed baseline, and retained the independently selected best snapshot. Archive prior logs and let the queue clear STOP only after acquiring its lock.
  **Evidence:** journal "Resumed the paused BEV diagnostic"; outputs/bev_traffic_diagnostic_queue/history; resumed trainer log

- **Q:** Did the BEV diagnostic improve traffic performance?
  **A (short):** It showed measured gains on two evaluation streams: original versus selected successes were 11/30 -> 15/30 and 9/30 -> 20/30, with full 10/10 empty-turn retention. Pooled collision-free success rose from 33.3% to 58.3%, but collisions remained 25/60 and the gate failed. This is one training seed; the proposed next experiment is a bounded continuation of this agent, not another three-arm sweep.
  **Evidence:** journal "BEV traffic diagnostic completed"; reports/bev_traffic_assessment_20261005.md

- **Q:** Why did the video say SAFE immediately before a collision?
  **A (short):** The vendor TTC utility considers vehicles ahead on the same road and lane, excluding many crossing vehicles. Its no-target value can produce SAFE on the HUD even when the collision sensor subsequently fires. This is a diagnostic limitation; the policy and reward in these runs do not consume TTC, and shields were disabled.
  **Evidence:** journal "TTC SAFE telemetry excludes crossing vehicles"; toolkit/carla_manager/utils.py; module-isolated TTC filter reproduction

- **Q:** How did you avoid accidentally reusing completed results when extending training?
  **A (short):** The controller originally skipped any directory with metrics.json, including a finished 30k run requested to continue to 60k. Target-aware completion checks and regression tests now verify a higher target resumes while the same completed target stays done. A separate continuation queue gives the new selected checkpoint fresh validation instead of reusing the prior candidate's cached metrics.
  **Evidence:** journal "Fixed completed-run detection"; scripts/run_bev_traffic_diagnostic.py; tests/test_bev_traffic_diagnostic.py

- **Q:** How did you preserve the previous experiment while continuing its learning?
  **A (short):** Archive the 30k latest checkpoint, the selected 27.5k checkpoint and their metadata before continuing the same replay and optimizer state. Keep the original baseline and old evaluation files, use a new queue for fresh checks, and verify --resume advances the step counter. This maintains traceability as the current best.pt changes.
  **Evidence:** journal "User authorized BEV continuation"; outputs/diagnostics/bev_traffic_medium_seed42/history/at_30000

- **Q:** Did the additional 30k steps improve traffic driving, and is the task solved?
  **A (short):** Yes, in this one-seed BEV diagnostic, collision-free arrivals rose from47/90 to67/90 across three checked medium streams and collisions fell from42/90 to23/90. The scheduled validation passed25/30 with full10/10 empty-turn retention, but the newest medium stream was17/30 and zero-shot hard traffic21/30, so reliable traffic handling remains unsolved. Stream IDs are not independent training replicas or guaranteed paired scenarios.
  **Evidence:** journal "BEV 60k continuation improved traffic handling"; reports/bev_60k_assessment_20261005.md; completed_assessment_audit.json

- **Q:** Why propose a fixed collision-cost ablation instead of another unchanged traffic sweep?
  **A (short):** The current collision component is proportional to post-step ego speed and can approach zero when the ego stops; observed crashes sometimes retained nearly+300 return. Most crashes already had large penalties, so this is a measured limitation rather than a proven root cause. A proposed unchanged-reward control and fixed-cost variant from identical weights, with equal budgets and fresh replay in both, would test the hypothesis under the original evaluation benchmark before changing all encoder runs.
  **Evidence:** journal "Low-speed collision penalty can leave a crash with near-success return"; carla_wpt_env.py:95; collision_reward_audit.json; reports/bev_60k_assessment_20261005.md

- **Q:** How did you turn the collision-reward finding into a controlled training experiment?
  **A (short):** After user approval, freeze the assessed60k CNN BEV weights and initialize two30k-step pilots with fresh replay and optimizers. Change only the variant's once-per-episode collision cost; use final fixed-budget checkpoints, identical evaluation streams and the original benchmark reward. Reference measurements, code/config hashes and automatic retention checks make improvements or failures traceable.
  **Evidence:** journal "User approved the controlled collision-cost pilot"; reports/collision_cost_ablation_v1.md; outputs/collision_cost_ablation_queue/experiment.json

- **Q:** What prevented the collision-cost experiment from silently changing its treatment or benchmark?
  **A (short):** The sensor stays latched and terminal flags can lag reward detection, so a resettable guard charges the fixed cost only once per episode. Evaluation explicitly overrides the checkpoint's inherited training cost back to zero, and resume/cache provenance checks reject a different reward or source. Tests exercise these cases against the real vendor reward without changing terminal metrics or feeding diagnostics to the policy.
  **Evidence:** journal "Guarded fixed cost against latched collision telemetry"; tests/test_collision_cost_ablation.py; patches/cardreamer_compat.patch

- **Q:** How did you verify the approved pilot was actually training?
  **A (short):** The source checkpoint was archived and checked before launch; the trainer then logged loading those 60k weights, preparing fresh replay and starting from step2,500. Advancing episode records reached3,124 steps, and the host showed one trainer with the intended control reward. Complete canonical reference checks were26/30 and25/30 medium arrivals, plus10/10 simple retention.
  **Evidence:** journal "Reference measured; control pilot verified learning beyond prefill"; outputs/collision_cost_ablation_queue/control_seed42.log; results.json

- **Q:** Why did the live bar briefly predict an unrealistically short training time?
  **A (short):** It omitted the active pilot when no measured rate was available, then counted the post-prefill step-counter jump as fast learning. The watcher now falls back to the planned duration and excludes step-zero prefill records; regression tests cover both cases. The corrected observed rate was6.3 steps/s and queue ETA about2h50m.
  **Evidence:** journal "Corrected queue ETA during startup and random prefill"; scripts/watch_queue.py; tests/test_traffic_curriculum.py

- **Q:** How did you resume the collision-cost pilot without restarting training or mixing reward versions?
  **A (short):** Confirm the old queue exited, verify its saved step, replay, optimizer state and code/reward provenance, then restart the same locked controller. The control resumed at 3,998 steps and advanced past 4,136 without repeating prefill; completed reference checks were reused and the fixed300 pilot stayed queued. Archive the earlier STOP and logs so the manual interruption remains distinguishable from a completed policy failure.
  **Evidence:** journal "Resumed the user-stopped collision-cost pilot"; outputs/collision_cost_ablation_queue/history/before_resume_20261005_211717; control_seed42.log

- **Q:** Did the fixed collision penalty improve traffic driving?
  **A (short):** No benefit is demonstrated yet: the control completed30k and matched the reference at51/60 collision-free medium arrivals, while the infrastructure-interrupted variant achieved48/60 after21,278 steps. All retained10/10 simple turns. The fixed cost eliminated positive returns in47 observed failed post-prefill training collisions, but that mechanical correction is not evidence of improved policy safety; equal-budget validation remains unfinished.
  **Evidence:** journal "Collision-cost pilot stopped early"; reports/collision_cost_assessment_20261006.md; assessment_audit.json

- **Q:** Why did the overnight reward experiment end before its budget, and how was recovery made safe?
  **A (short):** A timeout reading optional CPU temperature escaped the health monitor and triggered checkpointed trainer shutdown. The repair tolerates missing CPU telemetry while retaining GPU/RAM/disk guards, explicitly reports failed/unattempted stages and preserves the original experiment manifest. An explicit old/new runtime hash record permits only queue/controller repairs; changed rewards, training code or budgets still fail the provenance check.
  **Evidence:** journal "Fixed fatal CPU telemetry timeout"; tests/test_traffic_curriculum.py; tests/test_collision_cost_ablation.py; outputs/collision_cost_ablation_queue/runtime_recovery_20261006.json

- **Q:** How was the overnight run designed to avoid repeating an uninformative continuation?
  **A (short):** Finish the unequal-budget pilot before selecting a training reward, rank completed candidates using safe arrivals across known difficult streams, and preserve replay plus optimizer state in an independent branch. Validate in30k blocks, preserve the best checkpoint with retention/collision guards, stop on prolonged stagnation, and compare against the immutable starting policy on separate final streams. This tests an improvement hypothesis; it does not guarantee a gain or resolve the camera-encoder comparison.
  **Evidence:** journal "User approved bounded overnight continuation"; reports/overnight_bev_protocol_20261006.md; tests/test_overnight_bev_improvement.py

- **Q:** How did you verify the overnight controller had really resumed learning?
  **A (short):** The trainer explicitly resumed at21,278 with replay and optimizers, then completed new episodes through21,827 without random prefill. The host showed one owned trainer/CARLA instance and the intended nested recovery controller, while the sleep/lid inhibitor and resource monitor were active. The live bar reports additional work instead of counting old checkpoint steps as overnight progress.
  **Evidence:** journal "Overnight queue launched and resumed trainer advanced"; outputs/overnight_bev_20261006; outputs/collision_cost_ablation_queue/fixed300_seed42.log

- **Q:** How did the overnight continuation handle selected overnight continuation source?
  **A (short):** Completed the 30k pilot comparison and selected control at step30000, validation rank (23, 78, -10, -1). Cloned replay and optimizer state; original checkpoints preserved.
  **Evidence:** journal 2026-10-06 01:20; /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json

- **Q:** How did the overnight continuation handle overnight bev block1 completed?
  **A (short):** Reached step60000; medium safety rank (27, 82, -6, -1), simple safe arrivals 10/10. Accepted=True; best checkpoint remains step60000. These streams are checkpoint-selection diagnostics, not holdouts.
  **Evidence:** journal 2026-10-06 02:48; /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json

- **Q:** How did the overnight continuation handle overnight bev block2 completed?
  **A (short):** Reached step90000; medium safety rank (29, 89, -1, 0), simple safe arrivals 10/10. Accepted=True; best checkpoint remains step90000. These streams are checkpoint-selection diagnostics, not holdouts.
  **Evidence:** journal 2026-10-06 04:20; /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json

- **Q:** How did the overnight continuation handle overnight bev block3 completed?
  **A (short):** Reached step120000; medium safety rank (25, 80, -9, -1), simple safe arrivals 10/10. Accepted=False; best checkpoint remains step90000. These streams are checkpoint-selection diagnostics, not holdouts.
  **Evidence:** journal 2026-10-06 05:57; /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json

- **Q:** How did the overnight continuation handle overnight bev block4 completed?
  **A (short):** Reached step150000; medium safety rank (0, 1, -13, -4), simple safe arrivals 0/10. Accepted=False; best checkpoint remains step90000. These streams are checkpoint-selection diagnostics, not holdouts.
  **Evidence:** journal 2026-10-06 07:33; /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json

- **Q:** How did the overnight continuation handle overnight bev run completed?
  **A (short):** Finished 4 blocks; selected step90000. Baseline and selected canonical holdout results: /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/summary.md. Medium streams3031/4031, simple3031 and zero-shot hard5031 were not used for training checkpoint selection; no hard-task training occurred.
  **Evidence:** journal 2026-10-06 07:46; /home/deepakbalaji/Project/Auto_Thinker_Engine/outputs/overnight_bev_20261006/results.json

- **Q:** Did the overnight training deliver an improvement, and which model should be retained?
  **A (short):** Retain the selected90k checkpoint: it scored89/90 on selection streams,58/60 on separate final medium streams versus56/60 baseline,27/30 hard versus26/30, and preserved10/10 simple turns. The final150k model collapsed to1/90 medium and0/10 simple; best-checkpoint preservation prevented losing the useful model. The final gains are modest, from one training seed, and require replication.
  **Evidence:** journal "Overnight assessment: modest separate-stream gain"; reports/overnight_bev_assessment_20261008.md; reports/overnight_bev_audit_20261008.json

- **Q:** Why avoid another long continuation or larger collision penalty immediately?
  **A (short):** The completed fixed300 pilot matched control/reference at51/60 safe arrivals rather than improving driving, while unchanged-reward continuation peaked and then learned mostly to stop. Finite model/Adam tensors and nearly all expected updates argue against a broadly halted optimizer, but do not identify the collapse's cause. Reproduce the good checkpoint and investigate the final2.5k steps before a short protected continuation; recovery also needs scaler/RNG and replay provenance, and evaluation seeds are not independently seeded traffic scenes.
  **Evidence:** journal "Fixed-cost pilot completed without safety gain"; reports/overnight_bev_assessment_20261008.md; scripts/evaluate_agent.py; third_party/dreamerv3_torch/tools.py

- **Q:** Why redesign the project instead of continuing the right-turn agent?
  **A (short):** Long routes compound failures across encounters, and the best58/60 result used privileged BEV on one repeated turn. The approved redesign first measures a stronger expert and existing evaluator, then learns camera path/speed decisions from encounter data and tests whether JEPA improves label efficiency and predictive planning against strong baselines.
  **Evidence:** journal "Approved camera navigation redesign"; reports/navigation_plan.md; configs/navigation.yaml

- **Q:** What made the stronger expert and reused harness require adaptation?
  **A (short):** The pinned PDM-Lite forced full throttle after 8.5s stopped and deleted invalid actors; we removed the throttle override and made invalid scenes fail explicitly. A native map-reload crash also showed that the evaluator must run in a child process while a separate owner controls simulator cleanup.
  **Evidence:** journal "M0 native map reload failure and expert anti-stall audit"; patches/navigation_garage.patch; src/navigation/runtime.py

- **Q:** What did the first multi-camera navigation profile establish?
  **A (short):** The modified privileged expert, actual evaluator and three 512×288 cameras ran for 200 synchronized frames at 13.81 frames/s, with 32.46ms p95 expert/control latency. This validates integration and estimates throughput; it is deliberately truncated, does not prove camera-policy competence or long-route reliability, and says nothing about rule coverage. (Corrected 2026-10-08 23:50: an earlier version said the Town03 route had no stop-sign opportunities; the runtime world has 20 stop actors in Town03, so stop opportunities must be counted from runtime trigger/lane applicability, not OpenDRIVE signals.)
  **Evidence:** journal "M0 profile passes after adapter and route-generation corrections"; outputs/navigation_m0_profile_v2_20261008; reports/navigation_runbook.md

### Navigation integration follow-up (2026-10-08 23:32)

- **Q:** What does the first successful long drive prove, and what does it leave unproven?
  **A (short):** The privileged PDM-Lite reference completed 1.23 km with no recorded infractions while recording three synchronized views. It proves integration, not camera-policy learning or encounter reliability; the identical 200-frame ego prefix also does not establish surrounding-traffic determinism.
  **Evidence:** journal 2026-10-08 23:32, outputs/navigation_m0_long_20261008/audit.json.
- **Q:** How did route provenance checks prevent a silently easier navigation task?
  **A (short):** A 389m planned route became 161m after snapped endpoint export. The agent rejected the drift before driving; exporting the exact planning inputs fixed it, and incomplete/setup-failed probes now return failure explicitly.
  **Evidence:** journal 2026-10-08 23:32, src/navigation/routes.py, tests/test_navigation_harness.py.

- **Q:** Why did the stop-sign coverage conclusion change after simulator inspection?
  **A (short):** The OpenDRIVE XML reports no stop signals for Town03/04, but the loaded worlds contain20/8 stop actors. We now audit runtime lane/trigger encounters; actor presence and expert hazard flags alone do not establish a rule-opportunity denominator.
  **Evidence:** journal 2026-10-08 23:38, reports/navigation_m0_assessment_20261008.md.
- **Q:** How did you validate repeatability before proposing rejected-trajectory replay?
  **A (short):** Two fresh seed42 runs matched 200 frames of ego controls, 19 dynamic actors' poses/velocities and signal phases exactly. This validates the observed prefix only; hidden-state agreement and reactive branches still need testing.
  **Evidence:** journal 2026-10-08 23:38, outputs/navigation_m0_repeat_b_20261008/audit.json, src/navigation/replay.py.

- **Q:** How did you keep documentation consistent when a later simulator finding overturned an earlier assumption?
  **A (short):** After the runtime inventory showed 20 stop actors in Town03, I searched the earlier reports and the question bank for answers that relied on the OpenDRIVE-based assumption. I corrected each one in place with a dated note rather than silently rewriting it, and confirmed the edits changed only that claim and the text spacing, not any reported number.
  **Evidence:** journal 2026-10-08 23:58; reports/journey/interview_questions.md; reports/navigation_review_plan.md.

- **Q:** Why does each simulator client run in a child process under a separate owner?
  **A (short):** CARLA's C++ client can abort the whole Python process on a timeout, which skips `finally` blocks. When a map load timed out during the encounter inventory, the server was left running with no owner. The fix gives the server lifecycle to a parent process that survives a child abort, keeps STOP-file handling, and uses SIGTERM only on processes it started.
  **Evidence:** journal 2026-10-09 00:20; scripts/build_encounter_inventory.py; outputs/navigation_encounter_inventory_20261009 (failed, preserved).

- **Q:** How did you count traffic-rule opportunities so they match how the evaluator scores violations?
  **A (short):** I called the pinned evaluator's own trigger-volume geometry for red lights and stop signs on every lane, then kept only junction movements with at least a 30m approach that the harness planner actually takes. That gives 326 signal, 49 stop and 166 uncontrolled valid movements across five towns. It also showed that six Town03 stop actors only cover lanes inside the junction, an evaluator quirk to avoid by design rather than by discarding results.
  **Evidence:** journal 2026-10-09 00:25; src/navigation/encounters.py; outputs/navigation_encounter_inventory_v2_20261009.

- **Q:** How did you force a red-light opportunity without changing how violations are scored?
  **A (short):** A small predeclared scenario, built from the evaluator's own TrafficLightFreezer, holds the ego's light red until the car has stood still near the line for a few seconds (or a maximum hold passes), then runs a yellow and all-red clearance before green. Scoring is untouched: the pinned red-light criterion judges every crossing, and the same hold rule applies to the expert and every student.
  **Evidence:** src/navigation/scenarios.py; journal 2026-10-09 00:05 and 00:15; outputs/navigation_encounter_pilot_20261009.

- **Q:** Why report min-speed infractions separately instead of counting them as unsafe?
  **A (short):** The evaluator's min-speed check compares the ego's speed with surrounding traffic, so encounters that impose a wait (held reds, stop signs, a stopped lead car) trigger it by design. The headline counts every collision, rule, route and blocking infraction. Min-speed is still reported per encounter, alongside the strict any-infraction count, so a reader can apply either definition.
  **Evidence:** src/navigation/metrics.py (PROGRESS_ONLY_INFRACTIONS); journal 2026-10-09 00:15.

- **Q:** How did you make long simulator evaluations resumable without letting retries become outcome selection?
  **A (short):** Encounters run in chunks, each with a fresh owned server and numbered attempt directories. Only infrastructure failures are retried, at most once: a missing record or a simulator crash. Agent failures, collisions and timeouts are results. Routes already recorded in the raw evaluator file are never rerun, a STOP-interrupted route is dropped rather than scored, and every attempt is kept in the summary.
  **Evidence:** scripts/run_encounters.py; tests/test_navigation.py runner tests; journal 2026-10-09 00:30.

- **Q:** How did you speed up a synchronous simulator evaluation without changing its results?
  **A (short):** Profiling showed one CARLA server used about 2.6 of 24 cores and 15% of the GPU, so the loop was latency-bound. I ran independent episode chunks on three servers with separate ports, locks and resource-gated starts. Chunk composition and per-route seeds stay identical, so results should not depend on the worker count, and the overlapping encounters from the stopped single-server run test exactly that.
  **Evidence:** journal 2026-10-09 01:00; scripts/run_encounters.py (LaunchGate); src/navigation/runtime.py (slot_ports).

- **Q:** How did you make parallel simulation safe on a 32GB laptop?
  **A (short):** The first gate checked memory as soon as a server answered, before it loaded its map. Three heavy Town10HD servers then exhausted memory and the run was killed. Server starts now count as complete only once the chunk is driving, need 12GB RAM and 6GB VRAM free, and a watchdog pauses the newest chunk below 6GB so its route is simply reattempted. Killed attempts are annotated, so infrastructure loss never counts as a driving failure.
  **Evidence:** journal 2026-10-09 09:05; scripts/run_encounters.py (LaunchGate, memory_watchdog).

- **Q:** How did you tell a simulator crash from the simulator's normal shutdown noise?
  **A (short):** I first read "signal 11" in CARLA's log as a segfault, then found the same lines at the end of a clean, fully successful drive: UE4 prints them during normal termination. The reliable marker is the evaluator process dying, plus the route it was running, so the assessment now uses those and the journal entry was corrected.
  **Evidence:** journal 2026-10-09 09:10; scripts/assess_encounters.py (evaluator_failures).

- **Q:** How did you discover that a third of your red-light tests never actually forced a red light?
  **A (short):** The harness logs every scenario that starts or is skipped, because the upstream route runner only prints a warning when setup fails. Auditing all 300 trials showed that 35 light holds had been skipped: the copied landmark lookup found no light on many Town03/04 approaches. I fixed the lookup to use the evaluator's own red-light lane geometry, reported v1 over valid encounters only, and reran everything with one code version.
  **Evidence:** journal 2026-10-09 09:25; src/navigation/scenarios.py (pick_ego_light); reports/navigation_expert_ceiling_v1_20261009.json.

### M0 resumption and assessment audit (2026-10-09 23:50)

- **Q:** How did you safely resume an interrupted evaluation without rerunning completed outcomes?
  **A (short):** The stale display said 43 encounters, but raw records preserved52. We resumed the existing catalog and seeds under the declared retry policy, reduced concurrency to two workers, and retained the unknown termination cause explicitly.
  **Evidence:** journal 2026-10-09 23:50; outputs/navigation_expert_ceiling_v2_20261009.
- **Q:** How did an apparently conservative setup filter accidentally undercount collisions?
  **A (short):** Filtering invalid setups before summing incidents hid events outside valid final trials. The corrected v1 audit counts 7 raw-attempt collision events, separately showing 5 in valid final trials, and ties setup evidence to the exact final attempt.
  **Evidence:** scripts/assess_encounters.py; tests/test_encounter_assessment.py; reports/navigation_expert_ceiling_v1_audited_20261009.json.

### M0 shutdown recovery (2026-10-11 10:15)

- **Q:** How did you distinguish host shutdown from a simulator failure when resuming evaluation?
  **A (short):** System journals showed the evaluator services stopping during host shutdown at both interruption times. We preserved that evidence, retained all 64 completed records including the blocked-agent result, and used the existing external-interruption rule without spending simulator retries.
  **Evidence:** journal 2026-10-11 10:15; outputs/navigation_expert_ceiling_v2_20261009/shutdown_evidence_20261009.txt and shutdown_evidence_20261009_morning.txt.
- **Q:** Why did a live report call a still-driving trial a crash?
  **A (short):** It reused post-exit settlement logic before a record could exist. The assessor now verifies the owning evaluator PID/output argument in the host process namespace and labels that trial in progress.
  **Evidence:** scripts/assess_encounters.py; tests/test_encounter_assessment.py; journal 2026-10-11 10:15.

### M0 closeout (2026-10-11 14:11)

- **Q:** Why did ten apparently wrong traffic-light setups turn out to be valid without rerunning them?
  **A (short):** CARLA renumbered actors between launches. The selected lamps matched the frozen inventory's entry lanes and physical positions within 0.624 mm; the corrected audit checks physical identity while preserving the original report and all raw outcomes.
  **Evidence:** reports/navigation_expert_ceiling_v2_audited_20261011.json; scripts/assess_encounters.py; journal 2026-10-11 14:11.
- **Q:** What did M0 establish, and which failures should guide the camera student's data collection?
  **A (short):** The infrastructure and expert were calibrated on all 300 encounters: 291 rule-safe completions, 298 arrivals, six collisions, two blocks and one lane departure. Crossing/yield was weakest at 45/50; preserve failed experience for learning, but do not treat every expert action as a clean imitation target.
  **Evidence:** reports/navigation_m0_closeout_20261011.md; outputs/navigation_expert_ceiling_v2_20261009/evidence_audit_20261011.json.

### M1 data engine (2026-10-15 14:24)

- **Q:** Why did you need a new spatial split after finishing the expert benchmark?
  **A (short):** The calibration suite had consumed every usable stop-sign junction. With explicit user approval, we preserved its results and froze a new geometry-only split of 62 training / 17 held-out junctions before any student training; the new held-out suite needs its own expert reference.
  **Evidence:** configs/encounters/location_split_v1/locations.json; journal 2026-10-15 14:24.
- **Q:** How does the dataset survive crashes while preserving action alignment?
  **A (short):** Only fsynced, renamed 32-sample archives are visible to readers; incomplete shards stay excluded. A sampled camera frame is committed only after observing the requested actuator command at the next physics tick, and policy readers never deserialize privileged labels.
  **Evidence:** src/navigation/dataset.py, src/navigation/collection.py; tests/test_navigation_collection.py.
- **Q:** Why stop scaling after a successful pilot?
  **A (short):** The installed NVIDIA libraries were upgraded during the session but the old kernel module was still loaded, causing NVML failure. We preserved 1,460 audited pilot samples and the 96-sample noise test, made failed GPU queries block new simulation, and left the prepared 20k collection pending reboot.
  **Evidence:** reports/navigation_m1_pilot_20261015.md/json; journal 2026-10-15 14:24; src/navigation/runtime.py.

### M1 collection boundary and launch (2026-10-15 15:00)

- **Q:** How did you catch training-location leakage before scaling collection?
  **A (short):** The runtime guard rejected Town03 route94 at0.25 simulated seconds: its nominally nonjunction spawn coincided with held-out junction103 within0.02mm. We stopped through STOP and retained297 samples with an explicit do-not-train review; pixel integrity alone does not prove spatial eligibility. We also found that agent crashes can produce a zero harness exit code and now inspect the actual record.
  **Evidence:** journal 2026-10-15 15:00; reports/navigation_m1_collection_20261015.md; original collection's console/trace/data_audit.json and collection_review.json.
- **Q:** How do you fix a collection boundary error without tuning the held-out benchmark?
  **A (short):** Apply the same geometry-only1m clearance rule to every training route, retain287/300 with all family/town strata, and freeze the eligibility/map hashes in the new request. Keep the held-out catalogs, location assignments and metrics unchanged, retain the strict runtime guard and stop the whole queue on a violation. The fresh bounded run supplies data, not evidence of learned camera driving.
  **Evidence:** src/navigation/splits.py; scripts/collect_navigation.py; outputs/navigation_m1_20k_v2_20261015/geometry_eligibility.json and request.json.

### M1 assessment and first camera baseline (2026-10-15 16:55)

- **Q:** What did the20k collection establish, and which data were excluded?
  **A (short):** Pixel/action alignment and all20k stored ego positions passed audits;17,567 timestamps from63 episodes were eligible for imitation. Three collision episodes,five privileged-safety-stop episodes,one native timeout and the final intentional cap remained preserved but excluded from clean imitation. Rule coverage exists, but stop-sign diversity and yellow examples remain limited.
  **Evidence:** reports/navigation_m1_assessment_20261015.md/json; journal 2026-10-15 16:55.
- **Q:** Why reject a model that improved offline speed error?
  **A (short):** The best checkpoint reduced matched internal speed MAE to2.596m/s versus3.736 for a train-mean predictor, but stopped-target error was3.300m/s and light accuracy49.78% was below majority56.80%. The real route then failed blocked at1.91%; lower regression error did not establish driving competence.
  **Evidence:** reports/navigation_m2_student_smoke_20261015.md/json; failed route summary/trace.
- **Q:** How did you separate oracle supervision from camera inference?
  **A (short):** Only exact SensorPacket array/JSON fields enter a fresh process with simulator/teacher/dataset imports blocked. Camera RGB, map-route geometry, declared ego odometry and timestamps are inputs; privileged labels and world traces remain training/scoring only. This is audited code separation, not a claim of an adversarial OS sandbox or official sensors-track compliance.
  **Evidence:** src/navigation/policy_ipc.py,policy_worker.py,camera_agent.py; tests/test_navigation_student.py.
- **Q:** Why did the harness reject a valid camera rig before driving?
  **A (short):** Two upstream Track enums print the same MAP label but compare unequal across modules. Importing the canonical validator enum fixed setup without changing evaluator behavior; a test calls that pinned validator with the declared three-camera rig.
  **Evidence:** tests/test_navigation_harness.py; original setup console and corrected profile.
- **Q:** What protocol deviation did you disclose before any final test result?
  **A (short):** Town05 remained unseen, but its final routes were not frozen before the first student as intended. We recorded that timing deviation rather than implying preregistration; a geometry-only coverage suite and the new heldout_dev expert reference remain required before their student comparisons.
  **Evidence:** journal 2026-10-15 16:55; reports/navigation_m2_student_smoke_20261015.md; configs/navigation.yaml.

- **Q:** Why qualify the first traffic-light accuracy and future-planning claims?
  **A (short):** Next-route lamp states were labeled without visibility validity;523 known states were beyond the80m valid distance range and2,738 beyond40m, so label accuracy is not an audited visible-lamp benchmark. Future speed profiles were trained/logged, but v0 control used current speed and spatial path; full time-profile tracking and predictive ranking remain open.
  **Evidence:** journal 2026-10-15 17:01; reports/navigation_m2_student_smoke_20261015.md/json; policy_runtime.py/controller.py.

- **Q:** Why did the speed metric suggest little over-stopping while the car got stuck?
  **A (short):** It ignored the brake head, which can command a full stop despite positive target speed. Applying the deployed decision rule reveals349 conflicting positions, only43.53% hold agreement on teacher stops and10.85% hold on teacher-go/nonbraking positions. The failed real-route result is retained; we did not reselect its checkpoint after seeing it.
  **Evidence:** journal 2026-10-15 17:04; scripts/assess_navigation_student.py; reports/navigation_m2_student_smoke_20261015.json.

- **Q:** How did you address speed/brake conflicts without forcing the car to move?
  **A (short):** A separate semantic stop head learns the teacher's zero-speed intent and gates cruising speed, while ordinary deceleration stays in the controller. The bounded candidate uses the same data and location split; actual hold errors determine checkpoint selection, and closed-loop results must still establish whether it improves driving.
  **Evidence:** Journal2026-10-16 semantic-stop decision; src/navigation/student.py; reports/navigation_m2_stop_go_pilot_20261016.json.

- **Q:** Why did you not call coherent stop/go outputs an improvement in driving?
  **A (short):** On matched internal positions the new model reduced false holds7.56% to4.81%, but stop recall fell44.64% to42.50%. Removing speed/brake conflicts is guaranteed by its architecture; actual closed-loop rule compliance still has to improve.
  **Evidence:** Journal2026-10-16 stop-go result; reports/navigation_m2_stop_go_20261016.json.

- **Q:** How did you detect and recover from binary recorder corruption without discarding outcomes?
  **A (short):** Binary CARLA_RECORDER headers appeared in a console file because the pinned local evaluator appends.log to a prefix and reuses it across routes. I stopped with STOP, preserved the attempt, separated per-route recorder files from console logs, and reran the same fixed pilot with unchanged criteria.
  **Evidence:** Journal2026-10-16 recorder-prefix bug; src/navigation/harness.py; tests/test_navigation_harness.py.

- **Q:** Why is positive target speed plus a brake command not automatically a contradiction?
  **A (short):** A vehicle above its desired speed should decelerate even if the target is positive. All987 training and268 validation teacher examples with that combination were above target; the failed camera policy instead kept braking at zero speed with a positive target, which is a release failure.
  **Evidence:** Journal2026-10-16 target-semantics clarification; outputs/navigation_m2_stop_go_20261016/teacher_braking_audit.json.

- **Q:** What did the semantic-stop change improve, and why did you still reject it as a safe driver?
  **A (short):** In six frozen TRAIN cases it increased compliant arrivals2/6 to3/6 and reduced blocked failures3 to2, but still ran a red light and added a stop-sign violation. The expert was6/6 compliant; stronger progress alone does not pass M2, and this tiny familiar-location pilot is not a generalization benchmark.
  **Evidence:** Journal2026-10-16 closed-loop countermeasure result; reports/navigation_m2_stop_go_20261016.md and audited pilot JSON.

- **Q:** How did you prevent an analysis script mistake from changing the experimental conclusion?
  **A (short):** A surprising zero-stop result exposed a missing checkpoint load in an extra streaming probe. I marked and retained the invalid artifacts, loaded frozen weights strictly, and reran identical episodes; training, checkpoint selection and real driving outcomes were unaffected.
  **Evidence:** Journal2026-10-16 extra-diagnostic mistake; outputs/navigation_m2_stop_go_20261016/streaming_assessment.py and explicitly invalid predecessor files.

- **Q:** How can you collect useful expert corrections without leaking oracle inputs or calling assisted driving a learned-policy success?
  **A (short):** The camera process receives the same allowlisted sensor packet; a separate privileged collector labels the observed state and records every assistance decision. Student requests and executed controls stay separate from teacher targets, assisted rollouts are censored, and the old internal validation junctions stay excluded from aggregation.
  **Evidence:** Journal2026-10-16 DAgger collection decision; reports/navigation_m2_dagger_protocol_20261016.md; src/navigation/dagger.py.

- **Q:** What did DAgger add beyond repeating clean demonstrations?
  **A (short):** The fresh pilot supplied118 stationary teacher-go timestamps after5s, including states in which the student kept braking. It recorded1,168 privileged control ticks explicitly; six assisted arrivals were collection evidence, not model performance.
  **Evidence:** Journal2026-10-16 correction-data milestone; reports/navigation_m2_dagger_assessment_20261016.json.

- **Q:** How do you learn from assisted rollouts without treating their future as an expert or counterfactual trajectory?
  **A (short):** Train only instantaneous expert targets on the observed state, and mask every future-speed target from the mixed behavior. Keep the original clean demonstration supervision alongside those corrections and validate on unchanged reserved junctions.
  **Evidence:** Journal2026-10-16 correction fine-tune decision; src/navigation/correction_learning.py.

- **Q:** What prevented a completed DAgger training run from replacing a less unsafe parent?
  **A (short):** The parent was included as candidate step0 under the unchanged internal selection score. All ten fine-tuned evaluations were worse; latest stopped only23.75% of required-stop positions versus42.50%, so the retained model stayed bit-identical to the parent and no duplicate simulator evaluation was needed.
  **Evidence:** Journal2026-10-16 correction-fine-tune result; reports/navigation_m2_dagger_finetune_20261016.json.

- **Q:** Why can a correct simulator traffic-light label still be unsuitable as visible-image supervision?
  **A (short):** A route planner can label a light beyond the image, behind the car or occluded. Project the actual light housings with full camera extrinsics and match physical identity first; even overlap cannot prove visibility, facing direction or legibility, so those unknowns stay explicit.
  **Evidence:** Journal2026-10-16 rule-geometry decision; src/navigation/rule_geometry.py and its projection tests.

- **Q:** How did you fix false duplicate-job detection without weakening concurrency protection?
  **A (short):** A new probe matched the existing anchored probe_ process pattern and rejected itself. Excluding only the current PID lets it start while a regression test proves a different probe is still rejected; the failed pre-server attempt is retained.
  **Evidence:** Journal2026-10-16 resource-guard bug; src/navigation/runtime.py; tests/test_navigation_rule_geometry.py.

- **Q:** What did the observability audit rule out, and what did it leave uncertain?
  **A (short):** All766 known applicable-lamp labels have a housing inside some camera's frustum, so a complete lack of image coverage does not explain them. However717 have a largest effective-input housing under6px, and geometric overlap cannot certify readable, unobscured bulbs; four target-stop runs also exceed clip length. Neither measurement proves the cause of policy failures.
  **Evidence:** Journal2026-10-16 projection-audit result; reports/navigation_m2_rule_observability_20261016.md/json.

- **Q:** Why did a harmless static-camera setup failure leave CARLA alive, and how was the cleanup path corrected?
  **A (short):** A settled ego pose failed the simulator's spawn collision check after synchronous mode was enabled. The probe now spawns above the road before placing the physics-disabled car exactly, and restores world settings in finally even on setup failure; the failed server is retained under STOP while additional graceful cleanup permission is pending.
  **Evidence:** Journal2026-10-16 pixel-probe bug; scripts/probe_navigation_signal_pixels.py; outputs/navigation_m2_signal_pixels_retry_20261016/worker.log and cleanup_failed.json.

- **Q:** How can a correct metric still be presented misleadingly in a plot?
  **A (short):** A fixed axis ceiling can hide large requests without changing the stored numbers. Visual review caught an11m/s cap in the new diagnostic timeline; automatic upper bounds now show all values, with the original plot and corrected provenance retained.
  **Evidence:** Journal2026-10-16 chart-ceiling correction; outputs/navigation_m2_rule_audit_20261016/unclipped_plot_provenance.json.

- **Q:** How did paired rendering distinguish readable signal pixels from merely projected light housings?
  **A (short):** Render red, green and red again at the same fixed TRAIN pose and measure changes inside the projected housing, with the repeated red as a temporal-rendering control. Seven crops show clear bulb changes at three poses, while back-facing/distant housings do not; this is static pixel evidence, not original-traffic visibility or a driving result.
  **Evidence:** Journal2026-10-16 exact-frame probe milestone; outputs/navigation_m2_signal_pixels_retry3_20261016/assessment.json and signal_crop_pairs.png.

- **Q:** How did you test whether the camera network reads signal color rather than predicting a location prior?
  **A (short):** Load the selected checkpoint through the production constructor, verify identical parameters before/after, and branch the same recurrent history into paired red/green images with fixed navigation and ego motion. The state class stayed Red, and green probability barely moved even where bulb pixels clearly changed; repeated-red controls and full input hashes were retained.
  **Evidence:** Journal2026-10-16 frozen-student response result; outputs/navigation_m2_signal_response_20261016/assessment.json.

- **Q:** Why did a valid stop line fail nearest-waypoint lane matching?
  **A (short):** The closest sampled waypoint was already on a junction road, while the line was associated with its approach road. Segment projection across that boundary recovers the geometric association without loosening distance criteria or changing scoring.
  **Evidence:** Journal2026-10-16 geometry-preparation bug; src/navigation/signal_data.py; frozen signal-pair plan_v2.

- **Q:** How do static counterfactual images help perception without becoming benchmark gaming or oracle inputs?
  **A (short):** Freeze TRAIN-only locations and phase-response criteria before imaging, change the target lamp while keeping background states fixed, and retain unknown or failed crops. Geometry and simulator state supply training labels only; action targets and runtime oracle inputs remain absent, and those labels count in later supervision budgets.
  **Evidence:** Journal2026-10-16 phase-pair pilot decision; reports/navigation_m2_signal_pairs_protocol_20261016.md.

- **Q:** How did you prevent collecting images from the wrong simulator town?
  **A (short):** The collector checked both the live canonical map name and frozen OpenDRIVE hash before imaging. A packaged default-town mismatch was rejected, then explicit load_world was added and tested.
  **Evidence:** Journal2026-10-16 default-map bug; scripts/collect_navigation_signal_pairs.py.

- **Q:** Why are saved RGB files alone insufficient to declare a data-collection attempt valid?
  **A (short):** A failed summary write left images without committed phase/frame certification. Those images stayed excluded; each completed pose now commits metadata and pixel-file hashes atomically, while the entire corpus still needs its gate.
  **Evidence:** Journal2026-10-16 shared-renderer argument mistake; scripts/probe_navigation_signal_pixels.py.

- **Q:** Why did you retain a failed data gate despite seeing real signal changes in the images?
  **A (short):** The predeclared hue rule rejected near-white Town10HD greens. The original corpus remains failed and unused for fitting; annotation development is TRAIN-only, and a fresh versioned collection must meet unchanged coverage and integrity requirements.
  **Evidence:** Journal2026-10-16 v1 static-corpus result; reports/navigation_m2_signal_pairs_protocol_v2_20261016.md.

- **Q:** How did the annotation revision differ from gaming an evaluation threshold?
  **A (short):** It changes a TRAIN-only label-validity method to test repeatable local on/off changes instead of assuming a fixed rendered hue. The original failure remains, new images use a frozen versioned method, and no driving metric, held-out route, success rule or model result changes.
  **Evidence:** Journal2026-10-16 validator2 decision; frozen plan_v3 and unchanged old data_audit.json.

- **Q:** How did you recover a simulator crash without rerunning or mixing the already collected data?
  **A (short):** Validate completed-pose hashes, require the original worker/archive/plan hashes, archive only the failed town and launch the same worker for its missing poses. The recovery ledger shows the four completed towns were skipped and only Town10HD was retried.
  **Evidence:** Journal2026-10-16 native-crash recovery; scripts/resume_navigation_signal_pairs.py and corpus_v2 resume_attempts ledger.

- **Q:** What evidence separates successful data collection from a successful learned driver?
  **A (short):** The data gate checks source, images, phases, geographic coverage and integrity. It contains no driving controls or safety result; a camera policy still needs separate overrides-off closed-loop tests under the unchanged criteria.
  **Evidence:** Journal2026-10-16 fresh corpus milestone; outputs/navigation_m2_signal_corpus_v2_20261016/data_audit.json.

- **Q:** Why does the new signal component use spatial outputs instead of another globally pooled auxiliary classifier?
  **A (short):** Most housings occupy only a few input pixels, and the existing network barely changes its output when the bulb changes. A fully convolutional stride2 grid retains location and local bulb layout, with native RGB and sparse supervision; it still needs a separate downstream driving test.
  **Evidence:** Journal2026-10-16 localizer decision; reports/navigation_m2_signal_grid_protocol_20261016.md; src/navigation/signal_model.py.

- **Q:** Why was the completed spatial signal model still rejected for integration?
  **A (short):** Its selected model labels only32/72 exact centers correctly and fires on9.38% of declared background cells; a red-majority prior labels36/72. Later updates become overconfident and worse. Nearby hits alone do not justify connecting it to driving decisions.
  **Evidence:** Journal2026-10-16 sparse-center result; outputs/navigation_m2_signal_grid_20261016/review.json.

- **Q:** How did you improve a localization loss while proving the measured task stayed unchanged?
  **A (short):** Keep evaluation masks and points separate from the broader training target plane. Load the original archived builder and compare every RGB input, evaluation mask, split and sample record; only the training label density may differ.
  **Evidence:** Journal2026-10-16 regional-target decision; outputs/navigation_m2_signal_grid_20261016/region_evaluation_invariance.json.

- **Q:** Why did broader positive regions not justify integrating the second traffic-light model?
  **A (short):** Evaluation inputs and masks stayed identical, but center accuracy fell to28/72 and yellow recognition to0/18. Lower background activation alone does not establish useful localization or state recognition; both bounded fits were retained as negative results.
  **Evidence:** Journal2026-10-16 regional supervision result; outputs/navigation_m2_signal_grid_region_20261016/review.json.

- **Q:** What does high accuracy on exact traffic-light crops prove, and what does it omit?
  **A (short):** The fixed classifier gets61/72 states with oracle geometry, so the localized pixels contain useful appearance information. It omits RGB detection, ego-lane relevance, occlusion, temporal memory and driving; correlated TRAIN development crops are not an untouched test.
  **Evidence:** Journal2026-10-16 exact-crop diagnostic; outputs/navigation_m2_signal_oracle_crops_20261016/assessment.json.

- **Q:** Why was the externally pretrained detector rejected despite every matched state being correct?
  **A (short):** It finds only14/72 supported labels, including0/40 whose housings are under8 native pixels. Three-view inference alone takes about183ms, exceeding the10Hz control interval, and its COCO supervision must be declared separately from fair JEPA experiments.
  **Evidence:** Journal2026-10-16 COCO baseline result; outputs/navigation_m2_reference_detector_20261016/assessment.json and predictions.json.

- **Q:** How do you prevent a successful data pilot or perception diagnostic from becoming an overstated driving result?
  **A (short):** Keep static data gates, sparse perception metrics, oracle diagnostics and closed-loop outcomes explicitly separate. Preserve negative attempts, source/checkpoint hashes and unchanged scoring; improve RGB detection, applicability and memory before testing a new driver with overrides off.
  **Evidence:** Journal2026-10-16 M2 next-experiment decision; reports/navigation_m2_signal_perception_20261016.md/json.

- **Q:** How do you adapt a detector when only some visible traffic lights have trustworthy annotations?
  **A (short):** Use verified phase-responsive housings as positives and exclude projected uncertain lamp regions from negative sampling in both the proposal network and ROI classifier. Resize those regions with the image, preserve true positives, and evaluate RGB-only inference against unchanged whole-junction validation with a step0 control.
  **Evidence:** Journal2026-10-16 small-anchor detector decision; reports/navigation_m2_signal_detector_protocol_20261016.md; tests/test_navigation_signal_detector.py.

- **Q:** Why report30/72 when a later detector checkpoint finds45/72 lights?
  **A (short):** The selection criterion was frozen mean validation detector loss, which selects step50. Reporting step200 as selected after seeing higher recall would change the rule after the result; both checkpoints' losses, recall, state errors and background candidates remain visible.
  **Evidence:** Journal2026-10-16 small-anchor adaptation result; outputs/navigation_m2_signal_detector_20261016/review.json and validation.jsonl.

- **Q:** How can a resumable run accidentally skip its final evaluation?
  **A (short):** A checkpoint can contain the final optimizer step and an older validation if STOP interrupts scoring. Track the last fully completed validation step separately and rerun pending scheduled scoring before marking completion; preserve RNG and reject changed-worker resumes.
  **Evidence:** Journal2026-10-16 resume bookkeeping bug; scripts/train_navigation_signal_detector.py; validation-interruption RNG regression.

- **Q:** How can you test crop alignment without improving the measured detector by changing its boxes?
  **A (short):** Freeze the selected detector and its one-to-one matched predicted ROIs, then change only TRAIN augmentation of the crop-state classifier. Outward expansions retain the original lamp, and1/16 variant weights preserve the source-example budget; geometry remains absent from classifier inference.
  **Evidence:** Journal2026-10-16 crop-alignment decision; reports/navigation_m2_signal_crop_alignment_protocol_20261016.md.

- **Q:** What improved after crop augmentation, and why is it still not a safer-driving result?
  **A (short):** Correct states on the same30 predicted light boxes improve21 to24, while privileged exact-crop states improve61/72 to68/72. The detector still misses42/72 sparse labels, produces background candidates and has no route applicability or driving evaluation.
  **Evidence:** Journal2026-10-16 crop-augmentation result; outputs/navigation_m2_signal_crop_alignment_20261016/assessment.json.

- **Q:** How do you evaluate mixed-precision perception without hiding numerical regressions?
  **A (short):** Freeze checkpoint, classifier, images, thresholds and matching; warm up both modes and alternate inference order on identical batches. Report accuracy changes with paired latency and verify all model parameters/buffers remain unchanged; precision alone does not establish driving safety.
  **Evidence:** Journal2026-10-16 FP32/FP16 audit decision; scripts/assess_navigation_signal_precision.py.

- **Q:** Why did the faster FP16 detector still fail the intended timing budget?
  **A (short):** Its three-view mean106.5ms and p95112.5ms exceed a10Hz controller's100ms interval before other processing. It also changes one sparse detection and17 total predictions, so the optimization is not described as numerically identical or a safety improvement.
  **Evidence:** Journal2026-10-16 FP16 result; outputs/navigation_m2_signal_precision_20261016/assessment.json.

- **Q:** How can you reduce a two-stage detector's inference cost while keeping the tradeoff honest?
  **A (short):** Freeze weights, images, thresholds and classifier, then compare a smaller proposal budget against the original modes with identical matching and paired timing. Report every missed or changed light, retain the original result, and avoid promoting from latency alone.
  **Evidence:** Journal2026-10-16 proposal-budget decision; controlled --include-cap200 request and prediction artifacts.

- **Q:** Why reject a detector configuration that finally fits within100ms?
  **A (short):** The200-proposal FP16 mode averages85.6ms but loses7 of31 sparse detections and6 of25 correct joint states. That is a coverage regression; model-only latency also excludes preprocessing, classifier and controller, so it does not prove end-to-end10Hz.
  **Evidence:** Journal2026-10-16 proposal-budget result; outputs/navigation_m2_signal_proposal_budget_20261016/assessment.json.

- **Q:** What is actually improved at this milestone, and what must still be demonstrated?
  **A (short):** Selected housing detection improves on the matched adapted control, and crop augmentation raises correct states21/30 to24/30 on unchanged predicted boxes. These are correlated TRAIN development labels; misses/backgrounds, lane applicability, long stop/release memory and overrides-off driving still need separate evidence.
  **Evidence:** Journal2026-10-16 component-gain milestone; reports/navigation_m2_signal_detector_20261016.md/json and preserved source/data hashes.

- **Q:** How can hard-negative mining improve perception without leaking validation examples?
  **A (short):** Freeze the parent and selection rule, run mining only on TRAIN views, and require geometry exclusion plus stable phase-control pixels before labeling background. Preserve validation examples for scoring, retain the parent if recall/state/background guards fail, and count all extra labels in the supervision budget.
  **Evidence:** Journal2026-10-16 hard-negative decision; reports/navigation_m2_signal_hard_negative_protocol_20261016.md.

- **Q:** Why can a frozen non-target traffic light pass a hard-negative phase-invariance filter?
  **A (short):** Only the selected lamp changes state, so other real lamps remain visually stable. If their housings are absent from the projected inventory, geometry exclusion plus phase invariance mislabels them as background; enlarged context review vetoed118 candidates before training, and a semantic-release gate now prevents attachment.
  **Evidence:** Journal2026-10-16 mining-review bug; outputs/navigation_m2_signal_hard_negatives_20261016/training_release.json and context pages.

- **Q:** How did you distinguish a detector false positive from a mislabeled real traffic light?
  **A (short):** Enlarged RGB crops initially looked like lamps, but registered semantic pixels identified building features and traffic signs. I corrected the overconfident visual diagnosis and required full TRAIN-set semantic verification before releasing negatives; no inventory defect or new training gain was claimed.
  **Evidence:** Journal2026-10-16 corrected visual diagnosis; matched Town01/Town04 semantic probes.

- **Q:** Why did a successful camera-attribute setter still produce an invalid calibration control?
  **A (short):** str(ActorAttribute) returns a descriptive wrapper, not the numeric lens value; CARLA parsed it as zero. Copy as_float, verify actual spawned attributes and matched sensor optics, and retain the invalid attempt rather than using it as evidence.
  **Evidence:** Journal2026-10-16 diagnostic mistakes; src/navigation/signal_label_probe.py and optics_validity.json.

- **Q:** How did you certify mined negatives without giving the driving policy semantic sensors?
  **A (short):** On TRAIN poses only, capture registered RGB/semantic pairs with verified optics and exact frames, reject lamp/unknown pixels and mismatched appearances, then hash the allowed source-box indices. The110 released boxes supervise only a binary classification head; inference remains RGB-only, and validation/held-out examples never enter replay.
  **Evidence:** Journal2026-10-16 semantic release; outputs/navigation_m2_signal_negative_release_20261016/release.json.

- **Q:** How did you recover from launching before a newly added replay test had passed?
  **A (short):** The test found immutable tuple handling in ROI regression targets. I issued STOP during initial validation so no optimizer update occurred, retained the attempt, fixed the type handling and required the focused checks to pass before a fresh retry; the journal also corrects my premature test-count claim.
  **Evidence:** Journal2026-10-16 premature launch mistake; stopped hard_negative_fit output and replay regression.

- **Q:** Why reject a detector with lower loss and fewer background candidates?
  **A (short):** The protected refinement reduced approximate backgrounds681 to171 but known-light coverage30/72 to27/72, including Town03 falling6/20 to1/20. All trained candidates failed the predeclared recall floor, so selected step0 retained the original parent.
  **Evidence:** journal 2026-10-16 18:03 protected negative refinement; outputs/navigation_m2_signal_hard_negative_fit_retry_20261016/review.json.

- **Q:** How can you test confidence calibration without tuning on validation?
  **A (short):** Freeze one scalar rule using only TRAIN ground-truth ROI margins: add the smallest nonnegative foreground bias matching the parent's TRAIN acceptance count. Keep the.5 reporting threshold and all validation guards fixed; reuse an identical candidate's evaluation if the offset is zero.
  **Evidence:** journal 2026-10-16 18:03 TRAIN calibration decision; reports/navigation_m2_signal_train_calibration_protocol_20261016.md.

- **Q:** What did a zero calibration offset establish?
  **A (short):** The negative-trained head already accepts120/179 positive TRAIN GT ROIs versus117/179 for the parent, so our frozen TRAIN-only rule added no foreground bias. It still loses validation detections (27/72 versus30/72); this rejects that particular calibration remedy but does not isolate whether proposals, localization, sampling or visual generalization caused the loss.
  **Evidence:** journal 2026-10-16 18:04; outputs/navigation_m2_signal_train_calibration_20261016/assessment.json.

- **Q:** What progress came from the rejected hard-negative fit?
  **A (short):** We certified110 TRAIN negatives with synchronized semantic/RGB evidence, caught and corrected a mistaken visual diagnosis, and preserved the stronger parent automatically. A one-to-one audit found5 lost/2 gained known labels; a TRAIN-only calibration rule chose zero. The report reproducer verifies source/data/weight identity;300 CPU tests pass,2 skipped. The next experiment needs broader positive approach/view coverage, rather than a longer classifier-only fit.
  **Evidence:** journal 2026-10-16 18:09; reports/navigation_m2_signal_hard_negatives_20261016.md/json; scripts/summarize_navigation_signal_negatives.py.

- **Q:** How do you expand the corpus without moving validation locations?
  **A (short):** Freeze the original whole-junction roles and collect a separate TRAIN-only supplement from those13 groups. Select distinct incoming lanes and new distances using map geometry, reject old/duplicate poses and reserved-junction proximity, then explicitly append samples while keeping the original144 validation images/72 labels unchanged. Never run the seeded splitter on the smaller subset.
  **Evidence:** journal 2026-10-16 19:08; reports/navigation_m2_signal_expansion_protocol_20261016.md; scripts/prepare_navigation_signal_expansion.py.

- **Q:** What did the multi-approach data gate establish?
  **A (short):** All99 planned poses produced1,188 hash-verified native images,116 responsive light/view pairs across13 TRAIN junctions and36 incoming lanes. Appending them raises TRAIN positive phase labels179→642 while validation remains byte-identical at144 images/72 labels; it demonstrates broader audited supervision, not better driving.
  **Evidence:** journal 2026-10-16 19:12; outputs/navigation_m2_signal_expansion_20261016/data_audit.json and attachment_audit.json.

- **Q:** What if recall improves but the frozen selection loss worsens?
  **A (short):** Keep the selection rule and parent, while reporting the unselected candidate transparently. Here joint adaptation raised detections30→36/72, correct joint states24→26 and proposals58→69, with approximate backgrounds681→209; loss.126094 exceeded.118660, so selected0 stayed. Decompose loss and localization/classification behavior before declaring a new prospective experiment.
  **Evidence:** journal 2026-10-16 19:15; outputs/navigation_m2_signal_joint_expansion_20261016/validation.jsonl.

- **Q:** How do you investigate a selection-loss disagreement fairly?
  **A (short):** Freeze the exact parent and completed candidate, reproduce the original loss seed/batches and decompose its terms once. Use fixed-threshold GT ROI scores only as privileged diagnostics; keep the original selected checkpoint and never turn that inspected validation result into a new threshold search.
  **Evidence:** journal 2026-10-16 19:17; reports/navigation_m2_signal_loss_audit_protocol_20261016.md.

- **Q:** Why can better proposals accompany higher ROI box loss?
  **A (short):** FasterRCNN sums box residuals over positive ROIs but divides by all sampled ROIs; changing proposals can change the positive count. In our fixed audit the ROI box term rose.043388→.054930 while classifier/objectness fell. A separate frozen sampler-count audit checks per-positive residual rather than assuming physical box quality worsened.
  **Evidence:** journal 2026-10-16 19:20; reports/navigation_m2_signal_sampling_audit_protocol_20261016.md.

- **Q:** What explained this loss/recall disagreement?
  **A (short):** The joint model produced222 sampled positive ROIs versus171, within the same9,216 total sampled ROIs. Per-positive SmoothL1 fell2.338→2.280, but summing over more positives and dividing by all ROIs raised the reported box term. We retained the frozen selection and did not present this sampled-population comparison as physical localization accuracy.
  **Evidence:** journal 2026-10-16 19:22; outputs/navigation_m2_signal_joint_sampling_audit_20261016/assessment.json.

- **Q:** What did the approach-expansion continuation achieve?
  **A (short):** It added1,188 TRAIN images from36 incoming lanes while preserving validation byte-for-byte, then improved the unselected candidate's detections30→36/72 and proposals58→69 with fewer approximate backgrounds. We honored loss-based parent selection and measured why its regression term rose: more positive ROIs under all-ROI normalization. No driving improvement or policy promotion is claimed; the reproducible report verifies1,800 images and305 tests pass/two skip.
  **Evidence:** journal 2026-10-16 19:26; reports/navigation_m2_signal_expansion_20261016.md/json.

- **Q:** How can you adapt ROI classification cheaply while preserving localization?
  **A (short):** Freeze the detector and add a zero-initialized65,665-parameter nonlinear foreground-logit residual with a fixed±2 bound. Train cached TRAIN ROI features using contained positive context, inward-only verified negatives and soft teacher consistency on unlabelled proposals. Box outputs and zero-step logits are identical; the new experiment's prospective operational rule leaves old selections unchanged.
  **Evidence:** journal 2026-10-16 19:38; reports/navigation_m2_signal_roi_adapter_protocol_20261016.md; tests/test_navigation_signal_roi_adapter.py.

- **Q:** What improved with the cached ROI adapter, and why reject latest?
  **A (short):** Selected500 improved detection36→44/72, joint state26→29 and approximate backgrounds209→160, with unchanged frozen base tensors. Latest1000 lost Town03 coverage6→5/20 and fell to40 detections/26 joint, so the prospective guard rejected it. The65,665-parameter adapter used only TRAIN features and did not change the driver.
  **Evidence:** journal 2026-10-16 19:42; outputs/navigation_m2_signal_roi_adapter_20261016/review.json.

- **Q:** How do you isolate a state-head improvement from localization?
  **A (short):** Keep the selected detector, all44 matched boxes,578 predictions and confidence scores fixed. Fit one state model using only expanded TRAIN crops, infer every predicted ROI from RGB without labels, then score one-to-one; require more than29 joint-correct labels and preserve24/30 on the old detector's fixed ROIs. Oracle-crop accuracy remains a separate diagnostic.
  **Evidence:** journal 2026-10-16 19:46; reports/navigation_m2_signal_state_expansion_protocol_20261016.md.

- **Q:** Why accept a state head whose oracle-crop diagnostic worsens?
  **A (short):** Runtime uses predicted boxes. The frozen guard required improvement on those exact boxes plus old-model compatibility, and the candidate improved29→31 joint-correct labels and24→27/30 on old ROIs. Exact annotated crops fell68→66/72, which is disclosed as a separate privileged diagnostic rather than substituted for pipeline accuracy. No driving gain is claimed.
  **Evidence:** journal 2026-10-16 19:48; outputs/navigation_m2_signal_state_expansion_20261016/assessment.json.

- **Q:** What improved now, and why is safe long driving still unproven?
  **A (short):** The selected perception bundle improves the joint initialization36→44/72 detections and26→31 correct joint states, with approximate backgrounds209→160. It still misses28 labels and misstates13 detected lights; red performance and three towns are unchanged. Tracking, ego-route applicability and full stop/release behavior remain, and the camera driving policy was not changed or re-evaluated.
  **Evidence:** journal 2026-10-16 19:58; reports/navigation_m2_signal_roi_adapter_20261016.md/json and perception_bundle.json.

- **Q:** Why guard component-validation locations when reusing older TRAIN rollouts?
  **A (short):** Location roles can differ across experiments. Two previously valid driving-TRAIN correction routes use Town03/655, now reserved for static perception validation. Preserve the old data but exclude those491 timestamps from temporal TRAIN development. Causal inference gets only RGB/navigation/ego packets; GT projection association is scoring-only and does not solve applicability.
  **Evidence:** journal 2026-10-16 20:15; reports/navigation_m2_signal_temporal_protocol_20261016.md; tests/test_navigation_signal_tracking.py.

- **Q:** Why does high state accuracy on detected lights not establish signal compliance?
  **A (short):** Raw715/769 matched housing states were correct, but only213/440 known-state timestamps had any matching detection, including a10.9s gap. GT chose the applicable actor only for scoring; runtime applicability remains unresolved. Smoothing gained six correct labels, delayed transitions and did not meet10Hz. These familiar assisted TRAIN images and uncertain projected visibility cannot support a driving safety claim.
  **Evidence:** journal 2026-10-16 20:18; outputs/navigation_m2_signal_temporal_20261016/assessment.json.

- **Q:** Why collect complete signal approaches before the next policy fit?
  **A (short):** Our following rollout stops37.2m before its future light; the release rollout starts4.6m away already waiting. Neither shows approach, braking onset, wait and release together. Replay found weak distant detection and a small smoothing gain, while87/133 assigned-ID changes still had the old ID overlapping the same housing. Collect split-safe continuous episodes and verify visible/applicable labels rather than assuming longer memory solves missing supervision.
  **Evidence:** journal 2026-10-16 20:25; reports/navigation_m2_signal_temporal_20261016.md/json; scripts/summarize_navigation_signal_temporal.py.

- **Q:** How do you fill a missing approach/wait/release sequence without leaking validation?
  **A (short):** Use unchanged frozen TRAIN signal_stop routes, exclude both driving and perception validation junctions, recompute spatial clearance and check actual positions. Require moving approach, observed onset, stationary wait, same-light release and post-release frames before accepting a clean demo; failed episodes stay in the archive. The strict geometry leaves only Town01/Town03 pilot approaches, which is disclosed rather than bypassed.
  **Evidence:** journal 2026-10-18 14:12; reports/navigation_m2_approach_protocol_20261018.md; tests/test_navigation_approaches.py.

- **Q:** Why reject two rule-compliant expert pilot episodes from direct-signal training?
  **A (short):** Both stopped behind two background cars25–27m before the light reference and lacked a sustained direct-light hazard. They are valid queued-traffic data, but do not prove the missing direct lamp approach/braking sequence. Preserve them and run a predeclared no-lead TRAIN scene while keeping evaluation traffic unchanged.
  **Evidence:** journal 2026-10-18 14:16; outputs/navigation_m2_approach_pilot_20261018/data_audit.json.

- **Q:** How can an empty training road be used honestly?
  **A (short):** Declare it as a curriculum intervention, require guarded TRAIN expert collection, prevent background spawning and verify recorded actor counts. Keep all evaluation traffic/criteria unchanged and never present curriculum success as camera performance; the earlier queued pilot remains rejected for direct-signal training.
  **Evidence:** journal 2026-10-18 14:20; reports/navigation_m2_empty_approach_protocol_20261018.md; tests/test_navigation_approaches.py.

- **Q:** What did the no-lead pilot actually establish?
  **A (short):** An expert sequence contains49.3m approach,4.8s go motion, an observed stop onset,7s stationary wait and4.8s after same-light green release, with exact camera/world/control synchronization. It fills the continuous-data gap but proves no student driving or rule understanding; the earlier queued pilots remain separately recorded.
  **Evidence:** journal 2026-10-18 14:21; outputs/navigation_m2_empty_approach_pilot_20261018/data_audit.json.

- **Q:** Why did Town05 have33 runtime stops despite the earlier five-sign count?
  **A (short):** The earlier scan counted explicit XML signal nodes only; CARLA also exposes object-derived signs. We matched the five explicit mapped locations to runtime actors for the approved coverage requirement while retaining all33 actors and every stop infraction. No student outcome informed that correction.
  **Evidence:** journal 2026-10-18 14:25; outputs/navigation_final_geometry_20261018/assessment.json; reports/navigation_m6_registration_protocol_20261018.md.

- **Q:** How can pilot coverage metadata be misleading?
  **A (short):** A two-town pilot intentionally omits other towns; its absence list was misread as geometric ineligibility. The full frozen schedule showed24 episodes across four towns and corrected that interpretation before expansion outcomes or training. Original plans and the mistaken journal statement were preserved with an append-only correction.
  **Evidence:** journal 2026-10-18 14:28; outputs/navigation_m2_empty_approaches_20261018/plan.json.

- **Q:** Is adding waypoint coverage anchors benchmark gaming?
  **A (short):** They enforce the already declared five-stop-location coverage using map geometry before any driver outcome. The unfrozen endpoint-only attempt remains archived, the full1–3km route is revalidated and all native stops/criteria remain. No failed model trial or difficult sign is removed.
  **Evidence:** journal 2026-10-18 14:31; reports/navigation_m6_registration_v2_protocol_20261018.md.

- **Q:** What does the frozen Town05 suite establish before evaluation?
  **A (short):** Twelve1–3km routes, three seeds, five required mapped stop locations and84/36 planned stop/red opportunities are fixed without a model outcome. Native harness interpolation and exact signal-entry bindings reproduce. Actual encounters, all infractions and camera driving remain to be measured; late registration is disclosed.
  **Evidence:** journal 2026-10-18 14:35; reports/navigation_m6_registration_20261018.md/json.

- **Q:** Why can complete stop/release training still teach the wrong behavior?
  **A (short):** In the empty-road data every wait lasted7s, so a recurrent policy could release by elapsed time instead of reading green. Vary declared TRAIN wait durations and keep evaluation settings unchanged; test paired red/green cases with overrides disabled. Twelve repeated episodes cover four locations, not12 independent encounters.
  **Evidence:** journal 2026-10-18 14:39; outputs/navigation_m2_empty_approaches_20261018/data_audit.json.

- **Q:** How did you validate geography between saved10Hz images?
  **A (short):** Independently replay all8,364 recorded20Hz ego poses against the strict map/junction guard, then change future collection to check every control tick. Actual braking timestamps are separate from zero-speed target onset; variable TRAIN waits prevent a constant release timer from substituting for observed green.
  **Evidence:** journal 2026-10-18 14:43; reports/navigation_m2_empty_approach_positions_20261018.json.

- **Q:** How do you train complete driving episodes without losing causal history?
  **A (short):** Keep the GRU state/gradient from episode start, recompute visual chunks with checkpointing and supervise after a fixed causal burn-in. Tests compare outputs/gradients with full forwarding. New expert action targets are allowed, while unverified signal/hazard/distance labels receive no perception loss; default historical loss behavior stays identical.
  **Evidence:** journal 2026-10-18 14:48; tests/test_navigation_causal_learning.py; reports/navigation_m2_causal_memory_protocol_20261018.md.

- **Q:** How do you fairly compare causal-history training against a clipped parent?
  **A (short):** Score both from episode start on the same original validation label positions, retain clipped-history regression checks and protect stop recall/false holds before selection. Keep parent0 and exact source/data hashes; no new perception labels are inferred from expert controls. The new12 episodes span four locations with varied7/11/15s waits, not12 independent environments.
  **Evidence:** journal 2026-10-18 15:00; reports/navigation_m2_causal_memory_protocol_20261018.md; scripts/train_navigation_causal_student.py.

- **Q:** Why reject a model with fewer false holds after long-history training?
  **A (short):** False holds fell4.81→0.17%, but required-stop recall fell42.50→23.39% and go-speed MAE worsened1.980→2.576m/s on the same1312 validation labels. Every update failed parent protection, so the unchanged parent stays selected and is not redundantly driven again. The next hypothesis adds localized learned perception while freezing the parent, rather than another longer repeat.
  **Evidence:** journal 2026-10-18 15:09; outputs/navigation_m2_causal_history_20261018/validation.jsonl/review.json.

- **Q:** How do you add localized perception without overwriting the parent controller?
  **A (short):** Freeze every parent parameter and add a zero-initialized learned stop-logit residual, so initialization reproduces the parent exactly and path/cruising predictions cannot change. Compare RGB-fact and null-token residuals with identical training budgets and fixed validation protection; only closed-loop and counterfactual tests can support a driving claim.
  **Evidence:** journal 2026-10-18 sampler/residual decision; reports/navigation_m2_signal_residual_protocol_20261018.md; tests/test_navigation_signal_policy.py.

- **Q:** How are final candidates and low-label claims kept honest?
  **A (short):** Final candidates bind checkpoint, frozen suite, exact seeds, source hashes and qualification reports to one resumable result root; failed driver outcomes are not rerun as infrastructure failures. M4 must reinitialize full-label driving heads and account for shared perception labels, so the current parent/residual cannot silently become a5% label model.
  **Evidence:** journal 2026-10-18 evaluation safeguards; src/navigation/final_evaluation.py; reports/navigation_m2_signal_residual_protocol_20261018.md.

- **Q:** Why is token count insufficient when adapting a video transformer to rectangular cameras?
  **A (short):** The same count can describe different time/height/width grids;576 tokens here means4×9×16, while the old square metadata inferred9×8×8. Explicit three-axis interpolation preserves source behavior and prevents space becoming time. Causal clip and future-control tests protect the later JEPA study, but no research gain is established by these implementation tests.
  **Evidence:** journal 2026-10-18 rectangular JEPA bug; src/navigation/rectangular_jepa.py; tests/test_navigation_rectangular_jepa.py.

- **Q:** Why might pgrep show nothing while progress keeps updating?
  **A (short):** The sandbox process view can omit jobs launched on the host. Use the anchored pattern in the actual host namespace and corroborate request/heartbeat/locks before starting or stopping anything; here it confirmed one cache and one waiting orchestrator, with no duplicate trainer.
  **Evidence:** journal 2026-10-18 host process visibility correction.

- **Q:** Did a frozen driver with localized RGB facts fix stopping?
  **A (short):** No. At matched1000 updates required-stop recall fell42.50→34.29% with facts and31.07% without, so neither was selected despite fewer false holds. Retain the exact parent and all negative evidence, audit observation/reset/label differences, and complete the missing held-out expert reference instead of extending either fit.
  **Evidence:** journal 2026-10-18 matched residual result; outputs/navigation_m2_signal_experiment_20261018/review.json.

- **Q:** Why was the new expert reference294/300 rather than the raw296/300?
  **A (short):** Two native arrivals never instantiated their declared HardBreak scenario. Keep both as failed planned tests, count every collision, and report valid-only family rates separately; source/setup auditing changes the headline, not the benchmark.
  **Evidence:** journal 2026-10-19 new held-out reference; reports/navigation_heldout_expert_20261019.md.

- **Q:** Why change a training teacher's stopping gap?
  **A (short):** The teacher knows the light through simulator state, while the camera student must retain visual evidence. Existing TRAIN optical probes motivate testing an earlier stop; a bounded6→12m IDM trial leaves evaluation rules/driver defaults unchanged and must verify geometry/RGB response before any new label or learning claim.
  **Evidence:** journal 2026-10-19 stopping-gap decision; reports/navigation_m2_light_gap_protocol_20261019.md.

- **Q:** Why keep the earlier-stop pilot out of the next imitation fit?
  **A (short):** It stopped5.4m earlier but added no projected camera coverage at this location and made the housings smaller. Clean expert execution is not evidence of improved observation or student driving; keep its data and failed hypothesis without expanding automatically.
  **Evidence:** journal2026-10-19 stop-gap result; reports/navigation_m2_light_gap_20261019.json.

- **Q:** Does preserving tiny red/green pixels hardcode a traffic rule?
  **A (short):** The common frontend pools continuous RGB and opponent-channel extrema from every native-view cell without labels or thresholds; a learned spatial/route attention and recurrent head chooses actions. All encoder arms share it, and predeclared disabled-component/trained color-only ablations separate its contribution from pretraining.
  **Evidence:** journal2026-10-19 spatial study decision; reports/navigation_m4_spatial_protocol_20261019.md; src/navigation/optical_fields.py.

- **Q:** Why report actual rather than nominal label-efficiency fractions?
  **A (short):** Complete episodes cannot always fill a percentage cap exactly. One fixed nested order releases733/1257/4069/16639 labeled timestamps; every encoder uses those same groups and the actual fraction is the graph x-axis. Shared photometric fields use no extra labels, and no full-label head is inherited.
  **Evidence:** journal2026-10-19 shared field cache; outputs/navigation_spatial_data_20261019/manifest.json.

- **Q:** Does96.8% offline stopping recall establish improved driving?
  **A (short):** No. The fresh spatial model is selected at500 by the unchanged balanced validation score; later updates regress. Freeze it, retain all updates, then test native closed-loop driving with overrides disabled. Offline recall can hide wrong-light association or distribution shift.
  **Evidence:** journal2026-10-19 first spatial fit; outputs/navigation_spatial_resnet_f100_seed42_20261019/validation.jsonl.

- **Q:** When is a failed launcher safe to retry?
  **A (short):** Here path normalization failed before CARLA or any trial began. Preserve that attempt, fix only the path and use a fresh output; never erase/retry a model's unfavorable completed driving case.
  **Evidence:** journal2026-10-19 pilot path mistake.

- **Q:** Why can excellent offline stop recall produce a car that never starts?
  **A (short):** It may copy the teacher's already-stationary ego state instead of deciding from the scene. Here zeroing only neural ego speed changes false holds4.47→83.51%, while273/392 TRAIN stationary-go labels were excluded by burn-in. Preserve the failed driver, test the input separately, retain actual speed in the physical controller, and validate in closed loop.
  **Evidence:** journal2026-10-19 spatial speed shortcut; reports/navigation_spatial_speed_diagnosis_20261019.json.

- **Q:** What did the high-scoring spatial model do in the simulator?
  **A (short):** Zero arrivals in six fixed cases, each at0% route progress. Zero recorded collisions/violations came from immobility. Setup, synchronization, all videos and source/recorder audits passed, so it is a model failure, not a reason to replace the scenarios.
  **Evidence:** journal2026-10-19 spatial pilot0/6; outputs/navigation_spatial_resnet_pilot_v2_20261019/assessment.json.

- **Q:** Does completing1.23km mean the driving goal is achieved?
  **A (short):** No. Neural-speed omission fixes the demonstrated launch failure and reaches6/6 short destinations, but the long drive records5 vehicle collisions,2 red and1 stop violation. Retain the complete video and all events, withhold promotion and test the stronger matched representations/planning.
  **Evidence:** journal2026-10-19 full destination drive; reports/navigation_visual_decision_20261019.md/json.

- **Q:** What if an old integration route overlaps a newly frozen split?
  **A (short):** Actual ego traces exposed two heldout_dev locations here. Disclose the mistake, keep all300 cases and denominators, avoid tuning the sites and stop claiming fully unseen geography. Training arrays still exclude them; the independent Town05 final remains untouched.
  **Evidence:** journal2026-10-19 integration exposure; reports/navigation_integration_exposure_20261019.json.

- **Q:** Does a lower JEPA continuation loss prove better driving?
  **A (short):** No. EMA targets move, so the fixed TRAIN-only loss/scene-variance checks diagnose learning or collapse, while downstream matched label curves and native behavior establish usefulness. Select the registered final update rather than searching driving validation during pretraining.
  **Evidence:** journal2026-10-19 continuation; outputs/navigation_visual_study_20261019/jepa_pretraining.

- **Q:** Why did7.9GB of logical trace reduction not recover that disk space?
  **A (short):** The host already uses Btrfs compression. I verified hashes and a full old audit, then restored all1170 original paths/bytes/metadata after discovering the assumption. Report actual host free space, not apparent file-size reduction.
  **Evidence:** journal2026-10-19 storage mistake; outputs/navigation_trace_restore_20261019/summary.json.

- **Q:** Why reclassify a native Perfect expert episode as unsafe?
  **A (short):** The pinned collision criterion ignores near-stationary ego contacts. Preserved recorder data, independently matched to official CARLA queries, show two contacts in that episode. Keep native scores separately and correct the approved contact-free headline294→293/300, without fault filtering or rerunning the trial.
  **Evidence:** journal2026-10-19 collision correction; reports/navigation_collision_correction_20261019.md.

- **Q:** Do scripted prefixes prove autonomous rule understanding?
  **A (short):** They establish controlled initial histories and poses. Log the ignored proposal and executed setup command, hand over once, then never override again. Score the unassisted interval separately and keep the native full-route result; certify the actual physical light/visibility/actors first.
  **Evidence:** journal2026-10-19 physical rule prototypes; reports/navigation_rule_counterfactual_protocol_20261019.md.

- **Q:** How do you prevent a planner from gaining reward through wiggles?
  **A (short):** The common planning cost measures ordered progress projected onto declared navigation, not raw traveled distance. Shared15 physically bounded candidates and risk costs apply to every predictor; actual native route completion remains the external measurement.
  **Evidence:** journal2026-10-19 M5 primitives; tests/test_navigation_trajectory_candidates.py.

- **Q:** How did you choose the encoder candidate without cherry-picking its seed?
  **A (short):** Register the lowest mean full-label offline score over three seeds, then always use seed42. Comma-JEPA wins that selector; every seed stays reported and native driving must still be evaluated.
  **Evidence:** journal2026-10-19 matched66 study; reports/navigation_m4_selection_protocol_20261019.md.

- **Q:** Why did the memory probe fail despite the expert stopping correctly?
  **A (short):** The right camera still saw the applicable light throughout the hold. That demonstrates visible-light stopping, not memory. Keep it invalid and establish a new geometry before testing students, using training geography only.
  **Evidence:** journal2026-10-19 native setup probes; reports/navigation_memory_geometry_search_20261019.json.

- **Q:** How did you recover ordered20Hz actions from10Hz demonstrations?
  **A (short):** Decode official native control/pose packets, align the entire20Hz pose trace uniquely, then use recorder control atf+1 as acknowledgment of the action taken after RGBframef. All19461 shard acknowledgments across72sources match; do not interpolate teacher proposals.
  **Evidence:** outputs/navigation_executed_action_audit_v2_20261019/assessment.json.

- **Q:** Why merge two native lane records for one stop sign?
  **A (short):** Native eligibility spans the incoming and internal junction lanes during one physical encounter. Count distinct physically matched signs per mission as an explicit conservative lower bound; never double the denominator or invent exact per-visit rates.
  **Evidence:** tests/test_navigation_final_audit.py; native stop reference eligibility log.

- **Q:** Why repeat the encoder study without changing validation?
  **A (short):** Actual RGB poses revealed two overlapping Town02 junctions. Remove complete contaminated TRAIN episodes, keep all13validation episodes/arrays unchanged, and repeat equal-budget fits from fresh initialization. Preserve the old results and disclose the defect.
  **Evidence:** reports/navigation_m4_geography_repair_protocol_20261019.md.

- **Q:** Does54/58 mean93% detector recall?
  **A (short):** It is origin matching against nearby dynamic actors, including cases without certified visibility. FPN also produces382 unmatched hypotheses and very slow CPU inference. Measure visibility, false braking, tracking and GPU runtime before a driving claim.
  **Evidence:** reports/navigation_vehicle_perception_comparison_20261020.md.

- **Q:** Why do matching contact counts not establish deterministic simulator replay?
  **A (short):** All480RGBsamples and960controls passed audit, and each repeated option had the same contact count. But light elapsed timers differed beyond the frozen tolerance; hidden state and alternative futures remain unqualified. Keep that failure rather than relaxing the gate.
  **Evidence:** journal 2026-10-20 00:43; outputs/navigation_branch_audit_v3_20261020/assessment.json.

- **Q:** How did you recover training space without deleting experimental evidence?
  **A (short):** Verify exact SHA-identical checkpoint copies, replace only duplicate storage with independent copy-on-write clones, preserve paths/bytes/mtimes and test write isolation. Measure committed filesystem free space after sync:1.13GB recovered;25GBreserve stays unchanged.
  **Evidence:** journal 2026-10-20 00:43; outputs/navigation_checkpoint_cow_more_20261020/summary.json.

- **Q:** Why are six true validity flags insufficient to release a physical rule test?
  **A (short):** They can be copied or become stale after a case, partner, runtime or evidence changes. The native entry point now verifies exact paired case bytes, underlying physical evidence and all declared setup runtime sources; changing the partner invalidates the certificate. Expert success is never a setup-selection condition.
  **Evidence:** journal 2026-10-20 00:51; src/navigation/rule_suite.py; tests/test_navigation_rule_suite.py.

- **Q:** Can a bootstrap ensemble member miss every hazardous example?
  **A (short):** Yes. Our ordinary seed42 episode bootstrap omitted all three contact-positive episodes. Before fitting, stratify the TRAIN-only episode bootstrap by contact status, retain paired draws across models and leave validation untouched; disclose the sparse3episode evidence.
  **Evidence:** journal 2026-10-20 01:13; reports/navigation_dynamics_fit_protocol_20261020.md; src/navigation/dynamics_learning.py.

- **Q:** Why is a fitted steering angle insufficient to qualify candidate-control prediction?
  **A (short):** Longitudinal and open-loop timing errors compound. Our TRAIN-only model fits steering at.649rad but has2.49m mean/10.79m p95 four-second validation position error, and its acceleration intercept hits a bound. Keep it diagnostic and verify physical alternatives before planning claims.
  **Evidence:** journal 2026-10-20 01:27; outputs/navigation_ego_kinematics_20261020/assessment.json.

- **Q:** Can a controlled rule objective fail despite safe driving?
  **A (short):** Yes. Both close-amber expert references safely stop, failing the predeclared objective to clear during amber. Keep that objective failure separately from collision/legal infractions and report actual5m/s handoff speeds; do not weaken the objective after observing results.
  **Evidence:** journal 2026-10-20 01:29; reports/navigation_reference_rule_behavior_20261020.json.

- **Q:** How do you resume training without hiding duplicated work after interruption?
  **A (short):** Prove the process is gone, restore the full committed optimizer/RNG checkpoint, and retain the longer raw log separately. Our1800-step checkpoint required replaying85 known uncheckpointed logged updates; completed fits and all seven immutable queued requests were preserved.
  **Evidence:** journal 2026-10-20 01:48; outputs/navigation_interruption_20261020_0134/recovery.json and restarted_queue_identity.json.

- **Q:** How can you reduce driving-evaluation storage without hiding failures?
  **A (short):** Retain every native recorder, trace, contact and metric, and choose showcase-video routes identically before either model runs. Our measured filesystem compresses a99.68MBworldtrace to~12.53MBphysical, while MP4 remains~159MB; a fixed six-drive video subset projects~3.46GB for all72missions' native evidence plus showcases, with explicit margin and hard storage guards.
  **Evidence:** journal 2026-10-20 01:51; reports/navigation_local_storage_plan_20261020.md/json.

- **Q:** Why can72unique output names still be an invalid72-trial rule evaluation?
  **A (short):** Names can duplicate the same physical case and seed. Validate the complete sixfamily×twolocation×twoarm×threeseed identity matrix, bind actual native controls and camera sources to certified cases, and keep every missing or invalid case failed in72. Count orphan recordings' contacts too.
  **Evidence:** journal 2026-10-20 02:00; tests/test_navigation_rule_student_assessment.py; scripts/assess_navigation_rule_student.py.

- **Q:** Why did correctly acknowledged commands still give inaccurate ego forecasts?
  **A (short):** Acknowledgment establishes what command was accepted, not instantaneous mechanical response. TRAIN-only timing analysis favored a.1sdelay; the frozenexpertmodel with that delay improved4sVALpositionMAE2.49→1.86m but p95remained6.69m, so reactive native calibration is still required.
  **Evidence:** journal 2026-10-20 02:14; outputs/navigation_actuator_timing_20261020; outputs/navigation_vendor_ego_20261020.

- **Q:** How did you recover a queue whose first driver case crashed before producing a video?
  **A (short):** Kept all six outcomes and the crashed case as a failure, changed the assessor to flag missing evidence, and executed only the four previously unstarted frozen candidates. The original crash, audit error and five available videos remain retained.
  **Evidence:** 2026-10-20 journal, scripts/continue_navigation_corrected_pilots.py.

- **Q:** Why can a low-loss world model miss every collision?
  **A (short):** Rare positives concentrated in a few episodes permit appearance memorization and near-zero risk predictions; a good latent loss or Brier score alone does not prove action ranking. Our selected factual members missed44–49 of49 positive validation horizons and remain unqualified pending diverse paired action outcomes and calibration.
  **Evidence:** 2026-10-20 journal; outputs/navigation_dynamics_study_v2_20261020/results.json.

- **Q:** Why did a spatial JEPA decoder still produce spatially blind risks?
  **A (short):** Its risk recurrence started from a global token mean and never consumed the spatial decoder output. Swapping encoded camera/grid assignments changed selected JEPA collision probabilities by only floating-point noise. A separate untrained v2 gives risk a position-aware attention path; this structural repair still requires physical training/calibration.
  **Evidence:** 2026-10-20 journal; reports/navigation_spatial_dynamics_v2_protocol_20261020.md.

- **Q:** Why compare a collision predictor with an always-safe constant?
  **A (short):** Rare collisions make aggregate prediction errors deceptively low. Our TRAIN-only constant prior achievesBCE.04315, while six selected models score.1099–.2216 and mostly miss all49positive horizons from one episode; neither the constant nor these models qualifies as a planner.
  **Evidence:** 2026-10-20 journal; outputs/navigation_dynamics_diagnosis_v2_20261020/report.md.

- **Q:** Does a planned wall budget allow retrying the final missing case?
  **A (short):** No. We retained the CNN's censored fifth case and unexecuted sixth case as failures under the original900s budget, preserving all native artifacts. Queue recovery executed only entirely unstarted candidates; an incomplete attempt without verified budget evidence is rejected.
  **Evidence:** 2026-10-20 journal; outputs/navigation_corrected_pilot_continuation_20261020/cnn_scratch_plain/reassessment_v2.json.

- **Q:** How can perception and action disagree, and what was the controlled response?
  **A (short):** A CNN fact model stalled while its hazard heads were low; a red prediction alone did not imply a nearby visible signal because presence targets meanwithin80m. We froze the parent and registered a small learned stop residual using its structured predictions, compared with constant-TRAIN-mean features, before observing student rule tests.
  **Evidence:** 2026-10-20 journal; reports/navigation_semantic_residual_protocol_20261020.md.

- **Q:** How did you fix a flaky safety test without weakening the stop?
  **A (short):** STOP remains published before audit metadata. The test now joins the monitor before checking the later receipt, rather than racing its write; production helper bytes remain identical and all14activation checks pass.
  **Evidence:** 2026-10-20 journal; outputs/navigation_compact_validation_recovery_20261020.

- **Q:** Why were four additional native encoder controls still needed after five candidates had driven?
  **A (short):** Four of those candidates belonged to a separate M2experiment with cold-start or auxiliary-supervision changes. They were not the matched M4heads, so we preserved their results and queued the originally required corrected-data controls, reusing the settled comma primary.
  **Evidence:** reports/navigation_m4_matched_native_protocol_20261020.md.

- **Q:** Why freeze actuator plans before resimulation?
  **A (short):** A logged feedback-control sequence can encode responses to events that occurred after the prediction point. The new pilot saves all plans first, verifies native execution matches each plan until its first post-contact intervention, and keeps intervened tails unknown.
  **Evidence:** reports/navigation_fixed_branch_protocol_20261020.md; scripts/audit_navigation_fixed_branches.py.

### 2026-10-20 13:30 — Paired rule tests and own-actuator acknowledgment
- Why did15/72rule passes fail to demonstrate rule understanding? No location–seed pair passed both changed arms; memory behavior followed location rather than red/green history. All22contacts remain counted.
- Why version own actuator feedback? During scripted setup the policy's proposed control can differ from the applied ego command. Send only acknowledged ego steer/throttle/brake with frame IDs, identically to CV/RSSM/JEPA; preserve old input/control parity and independently verify native frames before planning.

### 2026-10-20 14:01 — First-event causal timing and observational action bias
- Why did the better stop composite remain rejected? Facts750added25stop hits but14falseholds; the frozen+1point false-hold cap still fails.
- What did observed alternatives show? Brake avoided contact twice; go/wait-go collided. JEPAranked brake riskier; both learned ensembles missed all14positive horizons. CVcaught them but falsely flagged every negative horizon; none is calibrated or driving-qualified.
- How can action conditioning learn the wrong association? All144pre-first-contact positive horizons contain braking, often followed by reaction controls; two TRAINcollision episodes are near-stationary rear impacts. This is association, not a proven causal mechanism. Keep rear impacts counted and collect diverse interventions.
- How does20Hzfirst-event supervision help? Hazard atf+t uses actions only througha[f+t−1], then its loss prefix ends; future reaction commands cannot explain an already-scored event. Construction tests do not replace empirical calibration or fair baselines.

### 2026-10-20 15:09 — A useful long drive, TRAINlabels and actual-pose geography
- What works now? A1.376kmfamiliarTRAINcamera drive arriveswith0contacts/0red0stop/0interventions, but5speedinfractions andonejunction limitclaims; expertcrashremainsfailed.
- Why not pick the best M4driverafterpilots? Theprimarywasfrozenbeforedriving; deploymentresultsdisagreewithofflinescore. PosthocResNetengineeringisexplicitlyseparate, notnewprimaryorstatisticalencoderclaim.
- Why are depth/semantic cameras honest here? They aredeclaredTRAINsupervision,separateallowlistedRGBpolicyinputs; source/roles/pixels audited andallfuturelabels charged. CARLA0.9.15enum14/7differsfromstaledocstable; engineUEuniqueIDsarenotRPCactorIDs.
- Why check actual poses? Route78passedcataloggeometrybutspawnmappedreservedTown02/242. Guardstoppedbeforelabels;retainfailureandcontinueonlyunstartedcaseswithoutnewbudget.

### 2026-10-20 15:21 — Quarantine source data without rewriting experiment history
- How did you retain the1mgeography requirement? Source101measures0.501m;quarantinethewholeepisodeandits22labels. Preserveoriginalfailed13sourceaudit,andpublishseparate11verifiedsourcemanifestwith2rejections.
- What doeszero ruleviolationmeanonacleandrive? Nothingaboutred/stophandlingwhenbothopportunitycountsarezero. The1.376kmcontact-freearrivalshowsnavigationonly;failedrulecounterfactualsremainreported.

### 2026-10-20 15:37 — Diagnose perception before training a replacement
- Why do fused box counts understate physical-object coverage? One object appears in several cameras. Our frozen FPN drops from926to666mask/view matches after fusion but from616to608cross-view object groups; most reduction is expected deduplication.
- Which gap is supported by labels? At30–60m encoded depth,414/591vehicle masks match raw detections but only273survive projection. Small-mask detection also fails; every included mask is a vehicle, so no pedestrian claim is possible.
- Why not trust a plausible depth plot? Axial depth and radial range lift the same pixel to different3Dlocations. Verify the installed sensor against known native boxes before fitting a geometry head; visible surfaces still are not actor centres.

### 2026-10-20 15:57 — Keep unknowns explicit and reject an unnecessary learned replacement
- What changed in the perception interface? Every raw box remains present even when its ground intersection is unknown. Separate presence/ground-valid masks prevent zero-padded XYfrom silently becoming free space. All429packets independently reproduce; original frozen forward parity also passes.
- Why reject a depth head that beats a constant? On the same921supported pairs, flat-ground encoded-depthMAEis1.53m versusboosting1.77m/ridge3.15m. Both lose the predeclared protection and remain undeployed; a lower error on a different coverage subset is not an improvement.
- What does the failed instrument check prove? Nothing about depth convention:0/6placements captured before signal11. Keep the failure and avoid a positive claim. Future cleanup can skip RPCs to an exited owned engine without killing jobs or hiding the original error.

### 2026-10-20 16:18 — Preserve causal data through native infrastructure failures
- What mistake did the bounded cleanup introduce? It left a reused client at1stimeout; the next map reload failed. Restore20safter cleanup and before reload. A later20sfailure shows this is not the whole lifecycle problem; isolate remaining cases in fresh owned servers and retain both failedattempts.
- How did continuation avoid outcome cherry-picking? It ran only original whollyunstarted cases with unchanged geometry/actions and original900s/600MBshared cap. Failedbrake and crossingwait remain in10denominator; no retry, new seed or route substitution.
- What new evidence is promising but not yet qualified? Frontgo/crossinggo collide; frontwait/crossingbrake do not. Native first contact and actual acknowledged controls support only the executed prefix. Audit geography, RGB and replay before fitting; do not label unexecuted or intervened tails safe.

### 2026-10-21 00:09 — Native containment and factual action data
- **Q:** How do you recover from an uncatchable simulator-client abort without hiding failures?
  **A (short):** A native core identified an uncaught LightManager callback timeout. We moved CARLA bindings into a fresh child per recording; two instruments verified parent survival and child exit0, while both engine signal11 exits remained recorded. This verifies containment, not engine stability.
  **Evidence:** journal 2026-10-21; outputs/navigation_native_lifecycle_diagnosis_20261020/client_last2_backtrace.txt; outputs/navigation_process_isolation_probe_20261020.
- **Q:** What does a completed 63-case collection prove?
  **A (short):** It proves 63 attempts settled. Only 60 captured, three failed without retries, and every recording still requires independent pixel/control/geography/first-event audit. TRAIN/calibration roles and every contact are retained before model fitting.
  **Evidence:** reports/navigation_intervention_expansion_protocol_20261020.md; outputs/navigation_intervention_expansion_20261020/assessment.json.

### 2026-10-21 00:15 — Correspondence and censoring
- **Q:** Why did a replay comparison report a772m difference when both cars matched?
  **A (short):** Static traffic-light actor IDs changed order across servers. An independent geometry-keyed comparison, retaining original tolerances and failed artifacts, matches all39prefixes exactly. Unique moving roles and exact static poses are required; ambiguous matches fail.
  **Evidence:** outputs/navigation_intervention_prefix_identity_20261021; tests/test_navigation_intervention_prefixes.py.
- **Q:** Why sum log-likelihood across each observed first-event prefix?
  **A (short):** A censored survival likelihood is the product of survival probabilities through observation, plus the event hazard when observed. Dividing each sequence by its variable duration changes example weighting and overweights early collisions; the untrained prototype is preserved and a corrected generation is required before fitting.
  **Evidence:** reports/navigation_first_event_cache_protocol_20261021.md; journal2026-10-21.

- **Q:** Why reject the new world models even though sequence likelihood improved?
  **A (short):** On18calibration episodes they missed all four contacts, and endpoint Brier worsened versus an action-only baseline (.177 versus .163). The registered gate required endpoint improvement too; later fits overfit, so we retained the failures and returned to perception/data diagnosis.
  **Evidence:** Journal 2026-10-21 00:44; reports/navigation_first_event_study_20261021.md.

- **Q:** Why use phase-paired images to label traffic-light pixels, and how do you avoid leakage?
  **A (short):** The lit bulb may occupy only one native pixel while the housing center is dark. Offline red/green/yellow interventions isolate stable emissive pixels, but the model receives only one RGB view; we excluded65oldTRAINposes under current reservations and preserved144validation images/72labels.
  **Evidence:** Journal 2026-10-21 00:50; reports/navigation_signal_pixel_audit_protocol_20261021.md; reports/navigation_signal_pixel_study_protocol_20261021.md.

- **Q:** How can a detector have low average background loss and still be unusable?
  **A (short):** Most background pixels are easy. Our bulb model had backgroundCE.07156but37,241false peaks across144images; we rejected it and registered hard-background weighting with explicit light-coverage protections.
  **Evidence:** Journal 2026-10-21 00:54; reports/navigation_signal_pixels_20261021.md.

- **Q:** Why reject a correction that removes nearly95%of false detections?
  **A (short):** It reducedcorrectlightdetections44→27,violatingpredeclaredtown/statecoverageprotections. We retainedtheexactparentandallfailedcheckpoints; a cleaner visualization cannot substitute for better detection.
  **Evidence:** Journal 2026-10-21 00:57; reports/navigation_signal_hard_pixels_20261021.md.

- **Q:** How can a two-stage light detector gain context without privileged inputs?
  **A (short):** FreezeRGB-derivedcandidatepoints, thenclassifylocalandwiderRGBcropsaroundeachpoint. Allvalidationproposalsareverified, TRAIN-onlyminingkeepsunknownlightregionsoutofnegatives, andrecallcannotexceedtheproposer'scoverage.
  **Evidence:** Journal 2026-10-21 01:01; reports/navigation_signal_context_protocol_20261021.md.

- **Q:** What justified collecting more signal data instead of extending the verifier fit?
  **A (short):** FrozenTRAINbalancedaccuracy82.5%versusVAL40.2%showedpoortransfer; only7of21eligible signaljunctions were represented. We preservedallreservedTown10locationsandpreparedadditionalstrictTRAINapproaches, keepingthe144imagevalidationdenominator.
  **Evidence:** Journal 2026-10-21 10:39; reports/navigation_signal_context_20261021.md.

- **Q:** How did you expand perception data without changing evaluation?
  **A (short):** We froze200staticphasecasesat21strictTRAINjunctionswithtwoilluminationconditions, preservingallreservedgeographyandoriginal144validationimages. Everycapturefailurestaysunknowninthe200denominator; onlyindependentlyauditedclosedcasescanenternewtraining.
  **Evidence:** Journal 2026-10-21 10:43; reports/navigation_signal_diversity_protocol_20261021.md.

- **Q:** How can you use data from a partially failed collection honestly?
  **A (short):** Keep the collection failed and every original case in its denominator. Our independent audit accepted164of200closed cases and left36unknown; only the verified subset entered a new fit with unchanged validation images and labels.
  **Evidence:** Journal 2026-10-21 10:52; outputs/navigation_signal_diversity_audit_20261021/assessment.json.

- **Q:** Did broader signal data improve the model?
  **A (short):** Not under the unchanged1000update recipe: latestlocalization rose56→61of72,butcorrectstatefell44→43andbackgroundpeaksincreased37k→88k. We keptthefailedresultanditsinitializationselection,andretainedtheaudited21locationcorpusforaseparatelyregisterednextmodel.
  **Evidence:** Journal 2026-10-21 10:55; reports/navigation_signal_diversity_20261021.md.

- **Q:** How do you test a pretrained verifier without giving it simulator state?
  **A (short):** ExtractfixedfeaturesfromRGBcropsaroundRGB-derivedproposals, fitonlyTRAINbinaryforegroundlabels, andretainoriginalphasepredictions. We verifyfrozenweights/BNbuffers, scoreeveryvalidationproposal, andcompareagainstafixedRGBstatisticsbaseline withnoVALtuning.
  **Evidence:** Journal 2026-10-21 11:15; reports/navigation_signal_pretrained_verifier_protocol_20261021.md.

- **Q:** Did ImageNet features solve traffic-light false positives?
  **A (short):** They removed 97.9% of background peaks but reduced joint light detections from 44 to 26, so the verifier was rejected. All 18 lost detections were below 16 pixels in housing height; we registered a paired tighter-crop experiment while keeping labels, validation and gates fixed.
  **Evidence:** Journal 2026-10-21; reports/navigation_signal_pretrained_20261021.md.

- **Q:** How did you distinguish insufficient training from poor generalization?
  **A (short):** The frozen ResNet verifier reached 99.7% balanced accuracy on TRAIN proposals but 72.9% on component validation. A paired crop-scale change still failed, so the next experiment selects regularization on whole TRAIN-junction folds rather than tuning against validation errors.
  **Evidence:** reports/navigation_signal_tight_20261021.md; reports/navigation_signal_regularization_protocol_20261021.md.

- **Q:** How did you tune regularization without tuning to the component benchmark?
  **A (short):** We kept entire TRAIN junctions in three internal folds and chose among five fixed penalties using location/class-balanced out-of-fold loss. The selected head improved joint detections from 26 to 30, but still failed the unchanged 44-light coverage requirement, so it was not deployed.
  **Evidence:** reports/navigation_signal_regularization_20261021.md; journal 2026-10-21.

- **Q:** Why try a compact RGB classifier after pretrained features?
  **A (short):** Inspection confirmed some valid signals occupy one or two colored pixels, which pooled semantic features can lose. We registered a nonlinear RGB baseline and a paired photometric-augmentation arm, selected using whole TRAIN junctions, without changing labels or validation gates.
  **Evidence:** reports/navigation_signal_error_gallery_20261021.png; reports/navigation_signal_rgb_boost_protocol_20261021.md.

- **Q:** What prevents training after a resource guardian silently exits?
  **A (short):** Every study requires a fresh guardian heartbeat at entry and while running. A stale heartbeat refused an empty preflight; we verified the missing host process, resumed the exact original budget baseline, and preserved both the failed record and original execution deadline.
  **Evidence:** Journal 2026-10-21 16:46–16:48; outputs/navigation_signal_rgb_boost_20261021/progress.json.

- **Q:** What did the nonlinear RGB experiment improve, and why was it still rejected?
  **A (short):** It retained41of72correct lights while reducing background detections from37,241to1,696. Three previously correct yellow lights fell below the unchanged foreground threshold, so coverage protection failed; ranking changes did not recover them.
  **Evidence:** reports/navigation_signal_rgb_boost_20261021.md; journal 2026-10-21.

- **Q:** How do you compare binary and state-aware traffic-light supervision fairly?
  **A (short):** Keep features, data, location folds and tree count fixed, and preserve half the training weight for background in both models. The joint model receives phase only as a training label; runtime inputs stay RGB, and internal TRAIN folds choose the head before component validation.
  **Evidence:** reports/navigation_signal_phase_boost_protocol_20261021.md.

- **Q:** Why reject a classifier that matched the parent's total accuracy and greatly reduced false positives?
  **A (short):** Its44correct lights included five extra green detections but five fewer yellow detections. The predeclared gate protects each state and town, so aggregate equality could not compensate for those regressions.
  **Evidence:** reports/navigation_signal_phase_boost_20261021.md.

- **Q:** How did you handle a plausible feature improvement that failed internal validation?
  **A (short):** Added color differences, chromaticities and contrast lost to raw RGB on whole-TRAIN-junction loss. We retained the raw control, verified bit-identical predictions, and did not evaluate the unselected representation on component validation.
  **Evidence:** reports/navigation_signal_opponent_20261021.md.

- **Q:** How can a second-stage classifier reuse a frozen detector without introducing privileged inputs?
  **A (short):** Gather the detector's learned RGB channels and predicted class probabilities at its original image points, retaining all proposals and freezing every weight. Check full-image forward parity and disclose that internal folds validate the new head, since the frozen upstream model already saw some TRAIN locations.
  **Evidence:** reports/navigation_signal_proposer_fusion_protocol_20261021.md; tests/test_navigation_signal_proposer_features.py.

- **Q:** How can a strict data-identity check reject unchanged data?
  **A (short):** Python tuples become JSON arrays, so a direct object comparison falsely rejected the same point records. We verified all records and indices, compared the complete JSON semantics instead, and added tests that still reject reordered points, roles and splits.
  **Evidence:** tests/test_navigation_signal_proposer_features.py; journal 2026-10-21.

- **Q:** What did the local RGB support audit reveal?
  **A (short):** Nearly60%ofpredicted positive TRAINpoints were more than2pixels from an audited active bulb, outside the detailed5×5input. Out-of-fold joint detections were1253/1365with that support but360/2006without; this motivated auditing context and local supervision, without changing validation labels or claiming causation.
  **Evidence:** reports/navigation_signal_train_support_20261021.md.

- **Q:** How did you distinguish weak traffic-light features from unsuitable local supervision?
  **A (short):** We audited all3,371 predicted-positive TRAIN points against paired physical phase images. 2,006 lacked an audited active pixel in the detailed5×5 crop, and1,351 of those had alternate-phase differences within red-repeat variation. This motivated excluding unsupported positives from fitting while retaining every original scoring example; conservative masks and rendering noise prevent a categorical unobservability claim.
  **Evidence:** reports/navigation_signal_support_geometry_20261022.md; journal2026-10-22; supported-supervision protocol.

- **Q:** Why can pointwise classification loss worsen while signal detection improves?
  **A (short):** Our local-support experiment stopped fitting phase labels on housing points without nearby active pixels. TRAIN point logloss worsened0.360→1.264, but correct housing detections rose1,141→1,144 and sampled background acceptance fell1,990→201. That passed a preregistered housing-level TRAIN check, with all excluded positives retained in scoring; full-image and component validation still had to follow.
  **Evidence:** outputs/navigation_signal_supported_20261022/assessment.json; navigation_signal_supported_protocol_20261022.md.

- **Q:** Did a large reduction in traffic-light false positives qualify the model?
  **A (short):** No. Supported local supervision reduced component background detections37,241→40, but correct housing-and-phase detections fell44→43/72, with yellow15→10 and Town10HD10→8. We rejected it under the unchanged subgroup protections despite a passing TRAIN-only check, retained the exact model and probabilities, and did not deploy or run M6.
  **Evidence:** reports/navigation_signal_supported_20261022.md/json; journal2026-10-22 supported-supervision result.

- **Q:** How did you decide whether to retrain signal classification or improve proposals?
  **A (short):** We split104TRAIN misses into58without a proposal and46filtered proposals. Existing out-of-fold predictions at audited active pixels recognized94of104, suggesting candidate placement deserved testing first. Those annotation-derived coordinates stayed diagnostic; the proposed runtime refinement uses only RGB neighborhood maxima and retains every original point.
  **Evidence:** reports/navigation_signal_appearance_20261022.md/json; navigation_signal_refinement_protocol_20261022.md.

- **Q:** Did RGB-only candidate refinement improve the learned signal detector?
  **A (short):** It raised component correct housing-and-phase detections43→55/72 without retraining, preserving every old point and adding nearby RGB channel peaks. Background detections rose40→127 but stayed99.66%below the original proposer. We still rejected deployment because yellow12/18was below the fixed parent15/18protection; the next TRAIN-only diagnostic targets seed-limited candidate coverage.
  **Evidence:** reports/navigation_signal_refinement_20261022.md/json/png.

- **Q:** How did you control the cost of an independent RGB candidate generator?
  **A (short):** The all-pixel rule exceeded15millionpoints after2,164completeimages and stopped before classifier inference. A separately registered one-representative-per-channel-plateau rule completed with12.36millionpoints and1,246/1,248TRAINhousing candidate coverage. That passed only an inventory gate; its full classifier false positives and runtime cost still required a bounded audit.
  **Evidence:** outputs/navigation_signal_global_inventory_20261022/progress.json; outputs/navigation_signal_plateau_inventory_20261022/assessment.json; navigation_signal_plateau_inference_protocol_20261022.md.

- **Q:** How do you expose point detections to a driving policy without inventing geometry?
  **A (short):** We staged calibrated camera rays and camera origins with full class probabilities and view/pixel identity. A point alone does not determine depth, housing size, stop-line distance or route applicability. All detections remain, stale frames are rejected, and route association/memory require separate learned-policy evidence.
  **Evidence:** patches/navigation_signal_point_tokens_candidate.py; tests/test_navigation_staged_signal_point_tokens.py; navigation_signal_runtime_staging_20261022.md.

- **Q:** Did porting the tree classifier to GPU make the whole driving pipeline faster?
  **A (short):** Only the measured classifier kernel:256rows took0.815ms versus3.519ms on the original CPUimplementation, with6.66e-16maximum probability difference and zero decision changes on4,096TRAINrows. Feature generation, transfers, proposer and three-camera control were excluded, so we made no full-pipeline claim and left the active scientific audit unchanged.
  **Evidence:** outputs/navigation_signal_tensor_profile_host_20261022/assessment.json; navigation_signal_runtime_staging_20261022.md.

- **Q:** Why did broader RGB proposals fail despite much higher TRAIN signal recall?
  **A (short):** Correct detections improved1,189→1,234/1,248, but background detections rose2,668→8,197 beyond the fixed3,075.8limit. Most false positives occurred at new coordinates; only201coincided with old sampled background examples. We retained the rejection and planned model-independent TRAIN negative sampling with strict junction exclusions, rather than loosening thresholds or using held-out predictions to mine training examples.
  **Evidence:** reports/navigation_signal_plateau_20261022.md/json/png.

- **Q:** Why test expanded-background supervision on the full proposal distribution?
  **A (short):** The old sampled check covered only 201 of 8,197 background detections in the failed full audit. Three new whole-junction fits retained the sampled total but still needed a fixed full-distribution comparison; sampled equality cannot qualify them.
  **Evidence:** Journal 2026-10-22 20:47; navigation_signal_background_full_protocol_20261022.md.

- **Q:** How can overlapping crop preprocessing be accelerated exactly?
  **A (short):** Precompute every possible aligned 3×3 integer average once and gather the same reduced windows, preserving padding and NumPy memory layout. Nine integer pixels cannot average to a half integer, so integer rounding reproduces the original mean-and-round operation; tests check full feature equality.
  **Evidence:** Journal 2026-10-22 20:52; patches/navigation_signal_context_fast_candidate.py.

- **Q:** Does a faster image feature pipeline make the driver real-time?
  **A (short):** No. Exact preprocessing reduced median feature-plus-head time from 976 to 343 ms on 63 TRAIN images, but that excludes candidate generation, image transfer and control and is still slow for three cameras at 10 Hz.
  **Evidence:** Journal 2026-10-22 20:55; navigation_signal_context_profile_20261022/assessment.json.

- **Q:** What did a second-process JIT test catch?
  **A (short):** Numba could compile the dynamic staged module but could not reload its cache because the module had no importable identity. Disabling the staging cache and measuring compile time separately fixed the lifecycle issue without changing numerical operations.
  **Evidence:** Journal 2026-10-22 20:56; patches/navigation_signal_context_fused_candidate.py.

- **Q:** Why reject a 41% reduction in background detections?
  **A (short):** The expanded-background head reduced false detections from 8,197 to 4,848, but the declared limit was 3,075.8 and it lost two previously correct red lights. Aggregate improvement did not satisfy safety-sensitive recall protection.
  **Evidence:** Journal 2026-10-22 22:02; navigation_signal_background_full_20261022/assessment.json.

- **Q:** What does spatial context add beyond global color moments?
  **A (short):** Two windows can have identical global color statistics but different arrangements of housing, bulb, foliage and edges. The prospective descriptor retains 4×4 local patterns at three scales while using only RGB; the fixed data and gates test whether that actually reduces confusion.
  **Evidence:** Journal 2026-10-22 22:08; navigation_signal_spatial_context_protocol_20261022.md.

- **Q:** How did you find missing positive supervision without using validation outcomes?
  **A (short):** Intersect the frozen TRAIN candidate coordinates with independently audited active bulb pixels and subtract every existing training coordinate. This found 1,596 additional exact positives without consulting model scores or component-validation labels; they remained diagnostic pending a prospective release.
  **Evidence:** Journal 2026-10-22 22:27; navigation_signal_positive_coverage_20261022/assessment.json.

- **Q:** How did you present positive results without overstating autonomous-driving capability?
  **A (short):** The paper separates weather/right-turn transfer, corrected offline scores, familiar-route navigation and rule/planner qualification. It reports missed hazards, recall regressions, unknown failures and the unrun final suite alongside the successful demonstrations, with source hashes and exact denominators.
  **Evidence:** Journal 2026-10-23 17:06; reports/conference_20261023/Auto_Thinker_Engine_Conference_Paper.docx and source_manifest.json.

- **Q:** How did you hand off an unfinished research project without overstating completion?
  **A (short):** Pause execution explicitly, record live ownership and prepared-but-unstarted work, and separate implemented milestones from failed safety/rule/planning gates. The handoff states that M6 driving was not run and preserves rejected models and unknown attempts.
  **Evidence:** Journal 2026-10-23 17:36, user pause; reports/navigation_pause_20261023.json; reports/journey/summary.md.

- **Q:** What does the final software test pass prove?
  **A (short):** The 559 passing tests support software contracts and regression checks; three environment-dependent tests remain skipped. They do not establish native driving safety, rule understanding, calibration on new situations or a completed final benchmark.
  **Evidence:** Journal 2026-10-23 17:36, documentation verification; reports/interview_preparation_20261023.md, Q69.

- **Q:** How do you distinguish a local checkpoint from a published repository state?
  **A (short):** A commit and fast-forward merge can succeed locally while remote authentication fails. The handoff records the local checkpoint, failed push and unpublished state separately; credentials are never committed, and a later authenticated push must respect remote history.
  **Evidence:** Journal 2026-10-23 17:39 IST, repository handoff; reports/repository_checkpoint_20261023.md.
