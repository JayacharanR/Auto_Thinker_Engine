# Camera world-model navigation: approved implementation plan

> **Paused by the user, 23 October 2026.** The plan below records the approved
> research scope and historical execution. No training, collection, native
> evaluation or guardian is live; do not resume until the user explicitly asks.
> M2/M3/M5 remain unqualified and M6 final driving has not run. See the
> [current project report](journey/summary.md),
> [interview guide](interview_preparation_20261023.md) and
> [pause record](navigation_pause_20261023.json). Earlier running/next paragraphs
> are historical, not current authorization.

Approved by the user on 2026-10-08 following the external review. Branch:
`experiment/camera-world-model-navigation`. Historical experiments retain their
original contracts and results. This is a new custom navigation benchmark, not
an official CARLA Leaderboard or Bench2Drive submission.

Current execution2026-10-20: all67 actual-geography-corrected encoder fits,
all4 paired cold-start/fact fits and all6 factual dynamics members are complete.
See [corrected M4 analysis](navigation_m4_repaired_20261020.md). Comma-JEPA wins
the predeclared offline selector; CARLA continuation is a negative result.
Native driving remains unqualified. First plain-ResNet pilot retains3/6arrivals,
2rule-safe,0strict,0recordercontacts,1red/1stop and a first-response IPC timeout.
Only the four unstarted frozen candidates continue; no failed case is retried.
Repaired memory references are available; source-bound certification and student
M3 remain pending. All factual models are explicitly unqualified: most miss
every positive collision horizon; zero positive red supervision remains.
User approved14GB additional net storage/10GB minimum free. The guard is active;
the exact resource-only runtime patch waits for pinned pilots/clock repair.
All raw evidence remains retained. Watch the [execution ledger](navigation_execution_20261018.json).

Historical status2026-10-18: user authorizes continuing the approved M0–M6 work,
including necessary runs; scientific/benchmark rules remain. Read the live
[execution ledger](navigation_execution_20261018.json). Varied-wait TRAIN data
complete5151 timestamps,12 accepted/2550 samples, all10296 control ticks verified.
[Whole-history fit](navigation_m2_causal_history_20261018.md) rejected all1000-update
checkpoints; selected parent unchanged. Matched frozen-parent RGB/null learned
stop-residual experiment is running (check live output). No new driving gain.
[Town05 suite](navigation_m6_registration_20261018.md) registered/native-audited;
late timing disclosed. Guarded final runner implemented, no final candidate/test.
M2 driving/M3/M4/M5/M6 final evidence remain OPEN. New heldout_dev expert reference
pending. Later-stage code primitives are not completed research comparisons.

Previous status 2026-10-16: M0 integration and the 300-encounter expert calibration are
complete; see [the closeout](navigation_m0_closeout_20261011.md). M1 data-engine
implementation and its small pilot passed on 2026-10-15; see
[the pilot report](navigation_m1_pilot_20261015.md). The
[20k collection passed](navigation_m1_assessment_20261015.md), with17,567 eligible
imitation timestamps. The [first CNN student](navigation_m2_student_smoke_20261015.md)
trained but failed its long-route diagnostic blocked at1.91%; M2 is not complete.
The [semantic-stop countermeasure](navigation_m2_stop_go_20261016.md) completes
3/6 TRAIN pilot cases compliantly versus2/6 for v0, with persistent blocking,
one red and one stop violation. It is not promoted; targeted correction data
and rule perception remain next. No longer repeat run is scheduled.
The [DAgger continuation](navigation_m2_dagger_20261016.md) collected1,729 audited
correction timestamps, but no fine-tuned checkpoint beat its protected parent.
Localized rule-label validity and long stop/release memory remain next; no new
learned driving result is claimed.
The [spatial/temporal audit](navigation_m2_rule_observability_20261016.md) is complete:
very small effective light housings and7.3–14.1s target-stop events motivate localized
all-view perception and longer context. Neither hypothesis proves the failure cause.
The [pixel/perception continuation](navigation_m2_signal_perception_20261016.md)
resolves cleanup and verifies the pixel probe. Fresh static supervision passes
its51-pose/612-image gate; two spatial CNN fits fail, exact privileged crops give
61/72 states, and a frozen COCO detector finds14/72 sparse labels. No policy is
promoted. Next: RGB small-object detection, temporal tracking/lane applicability
and complete stop/release memory. The
[two-stage follow-up](navigation_m2_signal_detector_20261016.md) now improves sparse
detection to30/72 (selected50), including13/40 under8px; fixed-box crop-state
accuracy improves21/30 to24/30. Still no driving integration: miss/background,
lane and temporal issues remain. FP16 latency107.5ms still exceeds10Hz interval;
the faster proposal cap loses7 detections and is rejected as default. Next:
The [hard-negative continuation](navigation_m2_signal_hard_negatives_20261016.md)
releases110 semantically checked negatives, but rejects all refined checkpoints:
approximate backgrounds681→171 accompany lost detection30/72→27/72. TRAIN-derived
calibration chooses0; keep the parent. Next broaden positive TRAIN approaches/views
before bounded joint proposal/head adaptation, then tracking and stop/release memory.
The [approach expansion](navigation_m2_signal_expansion_20261016.md) now completes
99 poses/1,188 new TRAIN images, with original validation byte-identical. Joint
fit improves detection30→36/72, proposals58→69 and backgrounds681→209, but fixed
loss retains the parent. Sampling audit finds171→222 positive ROIs with lower
per-positive box residual despite higher total box loss. No post-hoc promotion.
Next: ROI context/scale/limited feature adaptation with prospective selection,
then tracking/applicability and complete stop/release memory.
The [ROI/state continuation](navigation_m2_signal_roi_adapter_20261016.md) selects
ROI adapter500 (44/72 detections,160 approximate backgrounds) plus one expanded
state fit (31/72 joint). Latest1000 fails town protection. All detection gains
are at Town10HD development; red/yellow confusion,28 misses and13 state errors
remain. No driving promotion. Next ground temporal tracking/ego-route applicability
and full stop/release observability on existing TRAIN rollouts before another
bounded policy fit/pilot. Keep component-validation errors out of training.
The [fixed temporal replay](navigation_m2_signal_temporal_20261016.md) now completes
518 timestamps without fitting: raw715/769 vs smoothed721/769 matched states,
only213/440 known timestamps with a match. Route101 stops before reaching the
signal;123 begins already waiting, so complete approach/onset coverage is absent.
Visibility/applicability remain unverified,197ms triplet cadence misses10Hz,
and duplicate hypotheses contribute to assigned-ID changes. Tracker stays a
diagnostic. Next plan split-safe full approach/stop/release TRAIN collection and
verify label applicability/visibility before a new bounded policy fit. Exclude
older correction111/133 at component-validation655; preserve all original data.
All bounded jobs exited; check live state.
Town05 remains unused. Its intended pre-student final-suite registration was missed;
freeze geometry/coverage before any Town05 testing and disclose the timing deviation.

User-approved revision 2026-10-15: preserve dev_v1 as legacy calibration and use
`configs/encounters/location_split_v1/train` and `heldout_dev` for the learned
experiment. Reserve junctions before collection: 62 train/17 held-out, with no
cross-partition planned or actual collection trajectories. The new held-out
suite needs an expert reference; do not reuse the old 291/300 score for it.

## Milestones and gates

| Stage | Deliverable | Evidence required |
|---|---|---|
| M0 | Pinned PDM-Lite expert and Leaderboard/ScenarioRunner; synchronized three-camera rig; controlled CARLA lifecycle | Expert long development drive, scoring adversarial tests, camera throughput, repeatability, environment compatibility |
| M1 | Crash-safe encounter collection on Town01–04/10HD; expert, perturbed and student experience | 20k then at most 100k timestamps; coverage audit; atomic shards and failure provenance |
| M2 | Compact CNN perception, memory, separate route correction and speed imitation; DAgger | First camera long-route evaluation/video and 300-encounter development report with intervals |
| M3 | Paired rule counterfactuals | Full-system and learned-policy-only results; both engineered rule and emergency overrides off for the latter |
| M4 | In-domain JEPA and label-efficiency study | Five encoders; 5/10/25/100% nested labeled episode subsets; three seeds and equal downstream tuning budgets |
| M5 | Causal action-conditioned JEPA vs RSSM vs constant-velocity forecasts | Shared trajectory candidates/controller/costs; measured development improvement without reduced completion; counterfactual calibration |
| M6 | Frozen Town05 suite | 12 routes of 1–3km ×3 traffic seeds; all attempts, infractions and collisions; expert-relative comparisons |

The camera student is the first learned deliverable; it does not wait for JEPA
or learned predictive planning. Dreamer BEV remains a secondary arm after M2:
complete recovery state and investigate the 150k collapse before an approved
encounter/reward experiment. The2026-10-18 instruction authorizes necessary runs across the approved plan.
Estimate duration/resources from measurements and preserve every scientific gate.

## Architecture and research controls

Three 512×288 RGB views (yaw 0/−60/+60), navigation geometry and ego motion feed
temporal perception and separate path/speed heads. A conventional controller
tracks eight trajectory points over 4s. Explicit perception-based rule and
emergency overrides are logged; no oracle hazard inputs at camera evaluation.
The inference process accepts only SensorPacket and returns TrajectoryPlan or
ControlCommand. TrainingTargets never cross that boundary.

Reuse CARLA Garage's leaderboard_2 PDM-Lite integration and existing evaluator
criteria. Pin revisions and preserve tracked compatibility patches. Audit
teacher-side actor deletion and evaluator force-kill behavior; all candidates
use the same predeclared actor validity rules and evaluator semantics. The
existing Traffic Manager map-cache workaround must remain effective.

Keep spatial JEPA tokens. The existing predictor averages action embeddings
over time and uses within-clip masked targets; it is not a ready-made causal
world model. Future implementation needs past-only context, ordered control
tokens, future target grids, correct rectangular positions and leakage tests.
Compare 3×5 path/speed candidates using constant-velocity, RSSM and JEPA prediction.
Use three independently initialized/bootstrap-trained dynamics models.

Train/development towns: 01/02/03/04/10HD, separated by episode/location. Town05
is excluded from every learned fit, including unlabeled pretraining. Freeze 300
development encounters, 50 per configured family. Long-route integration runs
begin at M0 with the expert and M2 with the camera student. Freeze final routes
and seeds before student results, covering all five Town05 map stop signs and
at least 30 stop-sign/30 red opportunities. Validate runtime applicability, not
only OpenDRIVE counts. Do not replace expert failures with easier routes.

Count all collisions. Report safe completion, infractions/km, violations per
rule opportunity, blocked time, interventions, comfort and latency; Driving
Score is supplementary. Use paired comparisons, McNemar where appropriate,
and location/scenario-clustered intervals. 33/36 safe final missions with no
collisions/rule violations is a stretch target, not a safety guarantee.

Rule pairs: empty red/green; green plus stopped lead vehicle; red outside the
camera view at the stop line; yellow at varying braking distances; empty stop
junction stop-and-release; green with clearing cross traffic. Also test route
shortcuts, permanent stopping, rolling stops, stale frames and policy crashes.

Compare CARLA-adapted JEPA, unchanged comma2k19 JEPA, DINOv2-S, ImageNet ResNet18
and scratch CNN. Identical episode label subsets include auxiliary perception
labels. Report unequal pretraining sources/costs; match downstream budgets and
heads, not fictitious identical total compute. A random ViT remains an extra
same-architecture control. Cache only frozen backbones.

Collection records requested/executed actions, teacher corrections and safety
interventions. Safety-stopped branches are censored, not safe counterfactual
labels. Test alternatives by resimulating the prefix, checking state agreement,
then branching with live reactive traffic. CARLA recorder playback is for
visualization, not reactive branch validation.

## Laptop constraints

One CARLA server initially. 20Hz physics/cameras, 10Hz history, 5Hz planning;
profile before increasing concurrency. Original25GB disk floor is superseded
by the user-approved20October compact plan:14GB additional net usage and10GB
minimum free. Source-pinned work retains25GB until its queued activation;
see [the immutable approved plan](navigation_local_storage_plan_20261020.json).
100k three-view timestamps cost 14.75GB at 128² or 33.18GB at 256×144 for 384-d FP16
temporally pooled tokens alone; full temporal tokens multiply this by 4.
Keep one regenerable encoder cache at a time and measure compressed video size.
The historical 385 clips/s result does not predict larger multi-view throughput.

Checkpoint models, optimizers, AMP scaler if used, RNG, sampler and dataset/replay
provenance. Queues take locks and support STOP files; no broad process killing.
Log every material result, failure and decision. Defer learned creeping,
map-guided light crops, real-video transfer and large-model expansion.
