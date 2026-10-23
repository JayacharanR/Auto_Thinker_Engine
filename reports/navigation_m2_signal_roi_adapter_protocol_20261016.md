# ROI context residual classifier — prospective protocol

Freeze before feature extraction/fitting. Original selected50 remains the retained
detector. Expanded joint200 is an UNSELECTED development initialization, SHA256
a44f65acd4e3d2941c7f36e6dcc7e0f5d6a3853e3be6a7d1fbd691ed0370f1f9.
Its36/72 detections,26 joint states,209 approximate backgrounds and69/72 RPN
coverage are the new experiment's step0 reference, not a retroactive promotion.

Use the same explicit TRAIN supplement,642 positive phase labels and440 verified
negative phase copies. Preserve13/4 component roles,144 validation images/72
labels, all driving-validation/held-out geography and fixed robust state classifier.
No new simulator data, runtime sensor, driving benchmark/reward or threshold.
These are already inspected component-development locations, not an untouched test.

Cache only TRAIN ROI features from frozen joint200 (FP32 native512×288, original
internal resize and1000-proposal settings). Four variants per source label:
positive original plus three outward boxes, each side expanding at most25% of
source width/height, clipped to native bounds (all original housing pixels retained;
area at most2.25x, IoU with source at least4/9 before clipping). Negative original
plus three inward crops, each side at most10%; NEVER expand an approved negative
outside its semantically verified box. Variants are correlated, not new labels.

Also cache the first16 objectness-ranked RPN proposals from every TRAIN image
for teacher-consistency only. These receive frozen soft logits, not asserted
background/negative labels. No validation feature in the optimizer/cache.

Freeze all base tensors, including RPN, ROI features, box regression and old binary
classifier. Add a residual foreground-logit MLP: non-affine LayerNorm1024,
Linear1024→64, GELU, Linear64→1, bounded by2*tanh(raw/2). Zero last layer makes
step0 exactly the joint source; keep binary background logit and all box outputs.
65,665 trainable parameters. New architecture ID signal-detector-roi-residual-v1;
original checkpoint/schema is not overwritten. Cached and live forward must match.

Train1000 updates, batch64 supervised (16 original positives,16 supplement positives,
32 verified negatives; source variants sampled uniformly) plus64 unlabelled
proposal features. Loss is balanced supervised cross-entropy plus1.0*KL(old||new)
on proposals. FP32 AdamW1e-3,weight decay1e-4,grad cap1,seed20261016. No weight,
threshold, bound, batch or regularization grid.900s cumulative cap including cache
and scoring, STOP, full optimizer/RNG and separate validation_step.

Score at0/250/500/1000 on the same144 RGB images with existing confidence.5,
IoU.3/.5 and fixed crop classifier. Step0 reuses exact joint200 predictions only
after verifying source hashes and zero-adapter/base/box parity; no duplicate full
inference. Retain the raw detector loss as a diagnostic, not selection: the prior
audit demonstrated its positive-ROI-count dependence.

NEW prospective component selection, not a driving benchmark change: eligible
only if final one-to-one recall>=36, joint correct state>=26, backgrounds<=209,
each town recall at least starting4/8,6/20,12/12,14/32, and under8px>=15/40.
A trained candidate must strictly improve joint count or detection count; background
reduction alone is insufficient. Rank eligible improved candidates by joint count,
then recall, then fewer approximate backgrounds. Keep step0 initialization if
none improves. Report every checkpoint/loss/regression and explicitly distinguish
this rule from prior runs; do not reselect their checkpoints under the new rule.
Even a selected component is not automatically integrated/promoted to the driver.

No longer repeat or automatic further tuning. Tracking, applicable lane and full
stop/release memory remain necessary before a new driving pilot. External COCO
pretraining, geometry/semantic labels and tuning effort count in fair comparisons.
No JEPA, heldout_dev/Town05 or closed-loop safety claim.
