# Regional training targets, unchanged component evaluation

The first grid is not ready: selected200 center accuracy32/72(44.44%),9.38%
background active cells@.5; latest1000 overfits with foreground CE4.66. Preserve
it and the majority-red baseline36/72. Original camera policy remains unchanged.

Change only the TRAIN target density: assign each supported housing region its
known phase instead of only one center cell. Unknown light boxes stay ignored.
Keep separate evaluation arrays with the original72 centers and background masks.
No changes to images,13/4 junction split, model37,772 parameters, seed20261016,
class/crop sampler,1,000 steps, batch16, AdamW LR1e-3, BF16/FP32,1,200s cap or
foreground CE+0.2 background CE checkpoint selection. Both runs start from scratch.

Before fitting, compare the archived original dataset builder against the new
region builder: all612 RGB arrays/evaluation masks, sample metadata, split and
validation indices must be exactly equal. Save the proof in the original run's
region_evaluation_invariance.json. Retain both checkpoint sets and sources.

This is a small component-training countermeasure, not a new driving benchmark,
learned-policy result or permission to promote a driving model. Regional labels
remain approximations from phase-supported world-box projections, not pixel
segmentation. A negative result also remains reportable; no automatic long run.
