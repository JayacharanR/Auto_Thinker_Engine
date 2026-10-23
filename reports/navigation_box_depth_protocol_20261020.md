# Bounded box-only depth baselines — before fitting

The frozen detector's30–60m projection loss motivates a cheap geometry baseline
before another neural head. Fit **encoded camera depth**, not collision risk,
actor centres or certified physical3Dpositions. Native six-placement convention
probe failed with0/6observations after an engine signal11; preserve failure and
do not rerun it. Documentation/contributor evidence supports axial interpretation,
but installation-specific confirmation is absent. No controller adoption follows
from this supervised diagnostic.

Use exactly the verified9TRAIN/2calibration sources,351/78dense-label timestamps,
all three views and every existing frozen accepted prediction. Source roles,
geography and rawpixel hashes must verify. Both rejected sources remain excluded.
Charge all429supervised timestamps/1,287images; disclose COCOexternal supervision
and historical backbone/driver exposure at the calibration locations. This is a
posthoc engineering response to the preceding diagnostic, not untouched M4/M6
evidence or a new label-efficiency comparison.

Pair original predicted boxes, not ground-truth boxes, with visible vehicle,
person and two-wheeler masks at the original≥8pixel/IoU≥.3same-group
maximum-cardinality one-to-one matching. Target the actual observed bottom-row
pixel's decoded depth in metres, using the nearest observed pixel to the median
bottom x. No actor ID association, future action or teacher fact becomes a model
input. Any invalid/nonpositive target is recorded as unsupported, not relabelled.
Metrics are detection-conditional; undetected objects never become successes.

Inputs: the20RGB-derived shared observation features plus log normalized box
width and height, log positive box-bottom/horizon distance, and camera one-hot.
No rawsemantic/depth value, instanceID, town, route, frame, time, object pixel
count, ego/navigation or world state enters forward.

Fixed candidates, no sweep/continuation:

- Constant weighted TRAINmedian log depth.
- Standardized log-depth ridge regression, alpha1.
- Histogram gradient boosting,200iterations,learning rate.05,15leaves,L2=10,
  seed42, no early stopping. Both learned models use equal total weight per
  TRAINsource, then uniform object weight within source. CPUonly, two threads.
- Original flat-ground axial depth where its original60mgeometry is valid;
  report its missing-geometry coverage, never fill it from target labels.

Fit TRAINpairs first. Save all candidates' predictions for EVERY raw calibration
box before reading calibration masks/depth for pairing/scoring. Keep all fits and
predictions. Report per-location macro absolute-log-depth error, absolute metric
error/median/p95/relative error, original-projectable subset, and original
unprojectable subset. All depths are clipped to[.01,1000]m purely as the declared
sensor's numeric domain. No cost or uncertainty calibration is inferred.

Choose the learned candidate with lowest macro absolute-log error across the
two calibration locations. Permit an **offline depth diagnostic selection** only
if it improves original flat-ground error on their common-supported subset by
≥5%and beats the constant on all matched predictions at BOTH locations. Otherwise
select none. No automatic further fitting, perception/driver deployment or safety
gate promotion. Native geometry confirmation and independent broader scenarios
remain necessary.

Cap900s/100MB, same guardian14GBadditional/10GBminimum, one navigation owner.
Keep all failures. Cache/source changes during execution fail the stage. Every
new target must remain training/scoring-only; inference accepts only the raw
shared RGB observation arrays.
