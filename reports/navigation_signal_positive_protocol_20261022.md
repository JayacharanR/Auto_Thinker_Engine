# Align positive supervision with the frozen proposal distribution

Registered after the spatial-context audit failed at 1,228/1,248 joint and 3,221
background detections; before any new cache or fit. Spatial context reduces false
detections but does not qualify. No prior criteria or results change.

The independent no-inference TRAIN coverage audit found 3,467 frozen candidate
coordinates that are exact audited active bulb pixels. Only 1,871 were already
supervised. Release the other 1,596 (577 red, 339 green, 680 yellow) only after an
independent check against original TRAIN images, exact bulb targets, original
candidate inventory, every existing sampled coordinate and whole-junction folds.
No score-based mining, component validation, unknown-region relabeling or new
simulator case. The diagnostic remains marked no_training_release; a separate
successful cache receipt releases these exact rows for this experiment only.

Cache the same 375 features (raw87 plus spatial288), original/gamma .7/1.4 versions,
on those 1,596 rows. Independently reproduce every raw vector with the original
patch implementation and every spatial vector with direct pixel moments. Keep
FP16 feature boundary, source/data hashes and class/fold labels. Cache cap 300 s /
30 MB. Output `outputs/navigation_signal_positive_cache_20261022`.

Append the new version-major positives after the unchanged original and added
negative rows. Keep every prior supported training row, label, negative, gamma
variant and whole-junction exclusion. Same phase-balanced weights recomputed on
the expanded labels (background mass 1/2; each phase 1/6), seed 42 and 400-tree
hyperparameters. Exactly three internal folds; 300 s each / 1,200 s stage / 100 MB.
No final fit or component prediction. Output
`outputs/navigation_signal_positive_fit_20261022`. Sampled scores are diagnostic.

After all folds complete, separately register full frozen-proposal inference.
Original gate stays ≥1,234 joint, every original plateau-baseline town/phase
protected, background ≤3,075.8. Do not use 1,228 as the acceptance baseline.
No threshold sweep, longer fit, selection on component labels or M4 primary change.
No native driver or final Town05 testing before the existing qualification gates.

Original storage guardian/baseline, STOP and retained-failure rules apply. This
adds TRAIN supervision and may fail; its value must be measured, not assumed.
