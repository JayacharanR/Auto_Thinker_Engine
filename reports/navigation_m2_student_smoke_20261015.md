# First camera student — 15 October 2026

**M2 has started, but its driving gate has not passed.** The initial CNN student
trained and ran behind the intended sensor boundary. Its first long-route
diagnostic failed because it remained stopped. Preserve this negative baseline;
there is no demonstrated improvement in learned long-route driving yet.

[Results graph](figures/navigation_m1_m2_20261015.png).

## Implemented and measured

The trainable scratch CNN sees three256×144 context views plus the original
512×288 front view for small signals. Spatial pooling retains a2×4 grid per
context view and3×8 front grid. Navigation/ego features join these in a192-unit
causal GRU. Separate heads predict lateral path corrections, immediate speed,
eight0.5–4s future speed targets, brake command and auxiliary light/hazard/
distance/presence labels. Training labels are privileged; inference inputs are not.

Training uses only the63 safe, unassisted eligible M1 episodes. Whole junction
components are split before clips:50 episodes/14,745 timestamps for training,
13 episodes/2,822 for internal validation. The first seeded coverage-valid split
was attempt1. All six families occur on both sides. The model consumes raw images;
the13.605GB cache is uint8 image data, not frozen encoder features.

The bounded experiment completed1,000 AdamW steps in441.06s, including preparation,
with batch4,16-frame clips, learning rate3e-4 and CUDA BF16. It saves model,
optimizer, Python/NumPy/Torch/CUDA RNG and sampler state; BF16 requires no scaler.
The three original training sources were subsequently archived only after their
hashes matched the launch request. The best checkpoint was step300 under the
declared internal balanced go/stop speed-plus-brake selection score. Latest1000
is retained; it is not the selected policy.

Checkpoint: `outputs/navigation_m2_student_smoke_20261015/best_policy.pt`.
SHA256: `9cc1c2f7f8d260883f427251027f969f4337cbf95267056a67aece895bd15681`.
Detailed matched-window assessment:
[navigation_m2_student_smoke_20261015.json](navigation_m2_student_smoke_20261015.json).

| Internal metric | Best camera policy | Simple matched diagnostic |
|---|---:|---:|
| Speed MAE | 2.596m/s | Train-mean speed3.736; always-stop4.002m/s |
| Go-target speed MAE | 2.139m/s | — |
| Stopped-target speed MAE | 3.300m/s | — |
| Brake accuracy | 61.46% | Train-majority brake50.15% |
| Applicable next-light classification accuracy | 49.78% | Train-majority red56.80% |
| Lateral path correction MAE | 0.03794m | Zero correction0.00365m |

These comparisons use the exact same1,352 supervised internal-validation positions
and burn-in. The earlier baselines.json uses all2,822 timestamps and must not be
mixed with these matched scores. Blanking every image raises speed MAE to3.714m/s;
this shows input sensitivity, not a trained navigation-only baseline. Blanked
light accuracy58.28% further cautions against claiming competent signal reading.
The state labels describe the next route-applicable lamp, not verified visible
lamps:523 known-state samples have no valid distance within80m, and2,738 are more
than40m away. No visibility/occlusion validity label was stored. This metric is
therefore a label-prediction diagnostic, not an audited visible-lamp perception
benchmark; some targets may be unavailable from the current images. Add explicit
visibility/localization validity before interpreting or emphasizing that loss.

An exact replay of the1,000-batch sampler gives40.29% stopped targets and47.84%
brake commands over32,000 supervised positions. The problem is not simply a lack
of stopped examples. Stop-sign episode diversity and yellow-light coverage remain
limited. Random1.6s training clips also do not establish reliable memory over
multi-second waits; inference carries its GRU state throughout the episode.

An action-level audit applies the actual controller's hold condition (predicted
speed below0.15m/s or brake probability at least0.5) on the same positions.
It holds only242/556 teacher-stopped positions (**43.53%**) and incorrectly holds
70/645 teacher-go/nonbraking positions (**10.85%**). Positive target speed and a
brake command conflict in349/1,352 positions. The original speed-only
`go_predicted_stop_fraction` of0.13% missed this brake-driven behavior and must not
be treated as an actuator-level collapse check. None of17 zero-speed teacher-go
positions are held in this internal sample, so the actual long-route deadlock
also exposes a student-state/generalization gap. These are teacher-label agreement
diagnostics, not closed-loop rule-compliance probabilities.

## Camera boundary and actual driving

The camera adapter derives only the declared map route and ego odometry/localization
from CARLA. A fresh inference process receives strictly allowlisted arrays/JSON;
it blocks simulator, teacher, collection and dataset imports. No pickle, other
actor state, ground-truth light state, hazard label or teacher action crosses IPC.
Parent-side world traces are scoring/audit evidence only. This is a code-level
separation with payload/import checks, not an adversarial OS sandbox or an official
Leaderboard sensors-track submission.

The controller follows route geometry plus the learned lateral correction using
pure pursuit and a speed controller, with an8m/s v0 cap. It has no forced crawl.
The future speed profile/four-second trajectory is supervised and logged, but
v0 actuation uses the immediate speed head and corrected spatial path; it does
not yet track the predicted time profile or rank trajectory candidates. Wiring
and validating that tracking remains part of M2; predictive ranking belongs to M5.
The optional learned-perception rule layer is explicit; **all driving tests below
have it disabled, with zero engineered interventions**. It has not been validated
as a reliable full system.

| Attempt | Outcome |
|---|---|
| Initial200-frame profile | Setup failure before driving: wrong upstream Track enum; preserved |
| Corrected200-frame profile | 200 aligned frames,15.96m displacement, no recorded collision; intentionally truncated |
| Existing1,233.361m Town03 integration route | Failed: blocked at1.91% completion;20.90m traveled;241.4 simulated seconds; zero recorded collision/red/stop violations |

The pinned validator uses the canonical Track enum, whereas its local base agent
defines a second enum with the same printed name. The adapter now uses the
validator's enum; a test against the actual upstream validator covers this.
No evaluator, route, reward or metric was changed to fix setup.

The full-route run processed4,828 aligned camera timestamps at27.92frames/s,
with12.57ms p95 packet/inference/IPC latency, excluding simulation, video and audit
recording. At the blocked position, the network requested roughly1m/s while its
independent brake probability exceeded0.5, so the controller held the brake.
This is a concrete speed/brake conflict; it is not evidence of a legitimate red
wait. Do not assign a specific traffic-rule cause without applicable-signal/teacher
diagnostics. Zero collisions while blocked is not successful or safe navigation.

The complete three-view MP4 decoded all2,414 frames: H.264/yuv420p,1536×288,10fps.
It is stored in
`outputs/navigation_m2_camera_route_20261015/episodes/navigation_m0_town03_opt_v2_route0_10_15_16_42_40/three_cameras.mp4`.
Keep these diagnostic route recordings evaluation-only; DAgger must collect fresh
student states on the approved training catalog, not import evaluation geography.

## Next implementation priorities

1. Separate semantic stop/go intent from positive cruising speed and ordinary
   overspeed braking. Supervise stop intent from the teacher's immediate zero-speed
   target; test contradictions and false braking at zero ego speed. Preserve the
   failed v0 and compare under the same internal location split.
2. Add focused signal supervision (applicable signal localization/state and stop
   line) rather than relying on whole-image state classification. Use training-only
   oracle labels; a map-guided crop would need an explicitly declared/ablated input
   and remains deferred. Mask unobservable targets, add visibility/occlusion
   validity, and report yellow scarcity and class-wise recall.
3. Train/evaluate longer contiguous sequences or carry state within episodes.
   Include complete stop-and-release events and evaluate streaming memory, not
   only short reset clips. Keep a centreline-only path baseline until learned
   corrections show a benefit. Complete and test time-profile tracking; the v0
   future heads are auxiliary for control.
4. Add DAgger corrections at student states on the287 eligible training routes.
   Preserve requested/executed controls and privileged intervention/censoring labels;
   no teacher facts in the camera process. The v0 blocked state illustrates the
   expert-state distribution gap.
5. Calibrate the untouched heldout_dev suite with the expert before its first
   student comparison. Freeze/hash the final Town05 suite from geometry and rule
   coverage before any Town05 testing. Its intended pre-student freeze was missed;
   report that timing deviation explicitly. No Town05 data have been used.

A longer repeat of the same training setup is not supported by these results:
validation got worse after the selected checkpoint. No additional training queue
is running. M1's initial gate passed; the100k, DAgger, M2 driving and later JEPA/
rule-understanding gates remain open. Full CPU verification:232 passed,2 skipped;
changed-file lint/format and whitespace checks pass.
