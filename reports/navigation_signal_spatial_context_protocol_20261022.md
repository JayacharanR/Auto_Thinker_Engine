# Spatial RGB context on the fixed expanded-background corpus

Register before feature caching or fitting. Expanded negative supervision reduced
background detections 8,197→4,848 but lost two red detections (1,234→1,232 total),
failing the existing gate. Fixed TRAIN galleries show sign edges, foliage and
building/road textures. The old descriptor preserves a 5×5 color patch but only
global moments of wider windows. Test whether spatial arrangement helps.

Append 288 deterministic RGB features to the unchanged 87: 4×4 grids of per-channel
mean and standard deviation at native 16×16, 32×32 and 96×96 centered windows.
Use edge padding, exact integer sums/squared sums and float64 moment computation,
then FP32 output/FP16 cache. No position, town, lane, actor, teacher or map metadata
enters features. Same descriptor for every point. No spatial or color threshold.

Use exactly the original 51,094 sampled rows plus the previously registered
182,016 uniformly selected background rows, retaining their order and all three
original/gamma(.7/1.4) versions. The same 2,006 unsupported positive rows remain
excluded only from fitting. Same original labels, 21 whole-junction folds, weights
(background mass 1/2 and each phase 1/6), seed 42, 400 trees/100 rounds, depth recipe,
learning rate .1, max leaves 15, min leaf 50, L2 10, max bins 255, no early stopping.
Larger input dimension does not mean matched hardware time; record fit costs.

Cache all 233,110 rows ×3×288 FP16 features, maximum 500 MB / 1,200 s. Verify the
original cache hashes/ordered sample identity, negative eligibility and whole-fold
assignment. Independently recompute direct-pixel descriptors on the same fixed
63 TRAIN images used in the prior cache audit, all original and added rows/all
versions. Preserve all source hashes. No component-VAL features are extracted.

Fit exactly three internal heads, max 300 s each / 1,200 s stage / 100 MB artifacts.
Keep original sampled probabilities as diagnostics only. The raw expanded-
background controls are their already completed models, not refitted or retuned.
No final fit or validation prediction is authorized by sampled results.

Then separately register a full audit on the exact 12,362,313 existing coordinates.
Original foreground-sum≥.5, phase argmax and score-ordered housing matching remain.
Require ≥1,234 joint detections, each town/phase non-regressing against the original
plateau baseline and background ≤3,075.8; do not weaken the baseline to 1,232.
Only a full pass permits separately registered final fit/component validation.
No threshold search, discarded failures, native driver or final Town05 evaluation.

Use the original storage budget/guardian. STOP/failure retained. Proposed outputs:
`outputs/navigation_signal_spatial_cache_20261022` and
`outputs/navigation_signal_spatial_fit_20261022`. This is a TRAIN engineering
comparison, not an unseen end-to-end feature-learning or M4 primary claim.
