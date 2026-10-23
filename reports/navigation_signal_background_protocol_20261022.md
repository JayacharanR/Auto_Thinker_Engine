# Expanded background distribution — prospective TRAIN study

The independent plateau proposer achieves 1,234/1,248 TRAIN joint detections but
8,197 known-background detections, failing the 3,075.8 limit. Of these false positives,
5,529 are newly proposed coordinates and only 201 coincide with the original sampled
background rows. This motivates testing training distribution coverage, not a new
acceptance threshold or more component-validation inspection.

**Cache.** For every original 2,844 TRAIN image, sample up to 64 coordinates uniformly
without replacement from its frozen plateau-union candidates where the original
background mask is true. Exclude every coordinate already in the original TRAIN-row
cache, including anchors. Seed each image with SeedSequence([42, sample_index]);
sort selected row indices for stable order. No model probabilities, phase hue,
ground-truth positive boxes, validation examples or held-out outcome selects negatives.
The mask is a collection-side supervised target and never an inference input.
Unknown signal regions are excluded, not relabeled background.

Cache original and existing gamma0.7/1.4 RGB87 features at exactly the old FP16
boundary. Retain sample/coordinate/fold metadata for every extra negative. Keep
the original 49,088 supported rows and their three versions unchanged. Assign new
rows to the existing whole-town/junction folds; no resplit. Verify every new sample
is original TRAIN and every held-out junction's rows are absent in each fold fit.

**Three fixed fits.** Same supported joint100 architecture, seed42, 400 trees,
learning rate0.1, 15 leaves, min leaf50, L2=10, max bins255, no early stopping.
Append the extra negatives and their two gamma variants only from that fold's
fitting junctions. Keep total sample-weight mass background1/2 and each phase1/6;
the additional rows change coverage, not the class prior. No model-score mining:
using an OOF head trained on foldF to select examples for a model evaluated onF
would introduce indirect leakage. Uniform geometric sampling avoids that issue.

Save all three models and original-row OOF probabilities; report sampled metrics
as diagnostics, without selecting a head or running component VAL. The only intended
qualification comparison is on ALL frozen 12,362,313 TRAIN proposal points using
these excluded-junction heads and the existing original labels/threshold0.5. That
full audit requires separate registration after verified caches and fits; its gate
is joint≥1,234, every town/phase non-regressing versus the failed plateau baseline,
and background≤3,075.8. No smaller sampled metric can qualify the candidate.

Cache output outputs/navigation_signal_background_cache_20261022: 1,200s /200MB cap.
Fit output outputs/navigation_signal_background_fit_20261022: three CPU fits,
300s each /1,200s stage /100MB cap. Existing original features are hash-referenced,
not copied. Same original 14GB net /10GB minimum storage guardian. STOP, preserve
failures/no retries; source bytes pinned while owners run. No all-TRAIN final fit,
component-validation prediction, deployment, native driving or M6 in these stages.
