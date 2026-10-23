# Frozen-model prospective-action diagnostic

This analysis is registered after the six fixed-action rollouts and their contact
outcomes exist. It is a post-hoc engineering diagnosis. Every existing selected
JEPA/RSSM member predates these rollouts; no model selection, fitting, threshold
sweep or change to the recorded actions is allowed. All six observed options are
included: go, brake and wait-go twice, at one stationary TRAINlead location.

Require the independent closed fixed-plan audit: every native/pixel acknowledgment,
the original observable replay tolerances and exact before-outcome80-command plans
must verify. Extract exactly eight10Hz packets ending at the decision frame. No
future RGB, world state, teacher facts or event label enters prediction. All15
shared command sequences must reproduce from the archived ego parameters and
declared navigation, own speed and two actual prior ego commands.

Predict every candidate using the original three-member JEPA and RSSM exports.
The same clean frozen CARLA-JEPA encoder supplies past48×384tokens. CVuses the
already fixed COCO FPN detector, threshold.35, ground projection/fusion and tracker
on context samples1,4,7, all before/current decision. Use the cumulative maximum
of its eight proximity scores for a by-horizon event diagnostic. CVproximity and
balanced-training neural sigmoid outputs are both uncalibrated. The old neural
models do not consume FPN tracks, so this is not the final same-perception planner
comparison promised by M5. That confound and the missing red examples remain open.

Use CPU FP32 forward and FP16 encoder-token export, two threads and a900second
compute cap; explicitly disclose its numerical difference from the GPU BF16
feature generation. No GPU/server/global navigation ownership is requested. The
live storage guardian protects the owned output; anticipated artifacts<20MB.
Preserve raw predictions, descriptors, source hashes and all failed attempts.

After all predictions have been saved, score only the actually executed selected
option per rollout. The audit metadata is available to the analysis process but
never passed into encoder/detector/model forward. A first contact strictly after
the decision and inside the matching prefix supports an absorbing collision
event at every later horizon, even when post-contact braking censors future
latent/progress targets. A no-contact censored tail is unknown, never safe. Report
raw per-horizon squared error and misses at the unchanged.5diagnostic threshold;
these correlated horizon counts do not establish calibration/generalization.

All predictors also get the existing common collision-only cost: identical
frozen-ego progress, correction and acceleration terms, no learned progress-head
substitution. The untrained red term is disabled, not called observed-safe.
Any minimum-cost option is a prediction, not an executed or successful drive.
Only three of15options have ground truth, so no full-grid ranking success claim,
confidence interval, planner promotion, safety qualification or M6access follows.
