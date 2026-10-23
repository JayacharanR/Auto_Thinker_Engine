# Expanded TRAIN state classifier on fixed ROI-adapter boxes

Freeze before fitting. ROI adapter selected500 improves detection36→44/72 but
joint correct state is29/72 (29/44 matched boxes); all15 matched state errors
remain. Keep that detector/boxes/scores/assignments fixed. This is a separate
state-component fit, not another adapter tuning run. Original detector30-box
state baseline24/30 is a compatibility safeguard.

Use all642 existing TRAIN GT phase crops (179 original,463 supplement), no new
labels or validation fitting. Reuse the prior state baseline exactly:16x32 RGB,
original plus15 outward full-housing-preserving augmentations, C1 balanced L2
lbfgs logistic regression,max_iter1000,seed20261016, two CPU threads.16 variants
per source remain correlated. Total source weight642; original and supplement
each total321, divided equally among their source crops and16 variants.

Fit ONE model, no hyperparameter/threshold/feature search. Fixed native144
validation RGB images/72 labels and all spatial guards remain. Score actual RGB
inside the detector's predicted boxes; geometry only assigns scoring labels.
Update state estimates for all578 predictions without labels, then independent
one-to-one audit. Candidate accepted only if joint correct states strictly>29
on unchanged44 detections AND original fixed30-box correctness>=24. Preserve old
state artifact/outputs on failure. Oracle GT state accuracy is reported separately
and never a pipeline accuracy or selection criterion.

Fresh source/request/artifact hashes,300s wall cap and STOP. Verify exported
coefficients match sklearn, box/score identities stay fixed, and original selected
detector/camera policy/cache/old state classifier remain unchanged. A local
perception-bundle manifest may reference selected ROI adapter plus accepted state
classifier; no policy/runtime integration or driving promotion. No heldout_dev/
Town05, matched JEPA claim, driving benchmark/reward change or multi-hour work.
