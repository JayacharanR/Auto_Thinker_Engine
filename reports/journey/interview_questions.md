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
