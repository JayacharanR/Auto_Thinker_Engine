# Camera world-model navigation: approved implementation plan

Approved by the user on 2026-10-08 following the external review. Branch:
`experiment/camera-world-model-navigation`. Historical experiments retain their
original contracts and results. This is a new custom navigation benchmark, not
an official CARLA Leaderboard or Bench2Drive submission.

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
encounter/reward experiment. No multi-hour training is authorized by the short
M0 implementation gate. Estimate durations from measurements before proposing
the next long stage.

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
profile before increasing concurrency. Preserve at least 25GB free disk.
100k three-view timestamps cost 14.75GB at 128² or 33.18GB at 256×144 for 384-d FP16
temporally pooled tokens alone; full temporal tokens multiply this by 4.
Keep one regenerable encoder cache at a time and measure compressed video size.
The historical 385 clips/s result does not predict larger multi-view throughput.

Checkpoint models, optimizers, AMP scaler if used, RNG, sampler and dataset/replay
provenance. Queues take locks and support STOP files; no broad process killing.
Log every material result, failure and decision. Defer learned creeping,
map-guided light crops, real-video transfer and large-model expansion.
