# TRAIN approach expansion and protected joint fit

Freeze before new captures or fits. The negative-only2050-parameter refinement
was rejected:30→27/72 known lights. Its TRAIN-derived bias was zero. Continue
from the ORIGINAL selected50 detector, never the rejected latest200.

Keep original corpus_v2,13/4 component roles,144 validation images/72 labels and
all driving-validation/held-out geography unchanged. A new TRAIN-only supplement
uses only the same13 TRAIN junctions in Town01–04. Town10HD's sole original
component junction is validation, so it is excluded. Never rerun the seeded
split on this subset. Freeze references to the original plan/audit/roles.

Select first3 distinct light/incoming-lane approaches per junction in catalog
route-ID order across the original geometry-eligible routes. Outgoing turns and
episode families do not count as new approaches. Render32/12/2m before the
evaluator line, using native three-view512×288/FOV100/yaw0,−60,+60. Exclude poses
within1m of any junction outside these13 TRAIN groups; deduplicate old/new poses
within.5m and.05deg yaw. At most117 poses/1404 RGB images. Geometry preparation
uses maps only, never model outcomes. Seed background phases independently.

Reuse exact-frame Red/Green/Yellow/Red-repeat renderer, physical-light identity,
frozen non-target phases, optical pose checks and original response-v2/yellow
annotation rules. No runtime camera change. Every image/source is hashed.
Collection cap900s, one owned server at a time, STOP. Preserve partial failures.
Data gate: all planned poses complete;≥10 responsive junctions across all4 towns;
≥45 new responsive head/view pairs;≥6 responsive junctions with two incoming
lanes. No gate relaxation if coverage fails. These are correlated labels, not
independent driving encounters. Any new training geometry/state labels count.

Only after passing the gate: append TRAIN samples to the original data and
assert validation indices/RGB/targets/72-label denominator unchanged. Preserve
the approved110 canonical negatives/440 original phase copies; no unverified
new negatives. Retain frozen backbone/FPN/ROI features and BN buffers, but train
RPN and binary box/class predictor jointly (1,194,265 parameters). Start original
parent, FP32 AdamW5e-4,weight decay1e-4,grad cap5,batch2,seed20261016. Each batch
has one supported-positive image and one uniform TRAIN image; for each position
choose original/supplement with50/50 probability. Keep existing ROI replay on
original verified-negative images.200 updates/900s cap, score0/50/100/200.

Keep the robust crop classifier fixed, confidence.5,1,000 RPN proposals and
all prior IoU/state definitions. Score RPN proposals separately atIoU.3/.5,
then final boxes with independent one-to-one matching. Parent must reproduce
30 detected/24 joint/681 approximate background candidates at0. Candidate needs
recall≥30,joint≥24,background≤681; among eligible checkpoints minimize the
pre-existing fixed-seed detector loss. Keep parent if none improves; show town
and tiny-object regressions. No threshold tuning, longer repeat or automatic
driver integration. Dataset/optimizer/RNG/source and separate validation_step
are saved. Model timing is not end-to-end planner speed or safety.

If inadequate, diagnose proposal versus classifier/feature coverage before more
training. Temporal tracking, ego-route applicability and full stop/release memory
remain necessary before another driving pilot. No driving benchmark/reward change,
heldout_dev/Town05 test, matched JEPA claim or multi-hour authorization.
