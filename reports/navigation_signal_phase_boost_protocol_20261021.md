# State-aware RGB supervision versus a matched tree-count binary control

Registered before any fit. Prior gamma-augmented binary RGB model retained41/72
joint lights with1,696background peaks, but filtered three parent-correct yellow
lights. Original ordering did not recover them. Test whether preserving phase
distinctions during fitting improves foreground decisions; do not lower the .5gate.

Reuse exact original51,094TRAINrows/87RGBfeatures plus the two gamma feature arrays
from navigation_signal_rgb_boost_v2_20261021 (153,282fit rows). Same21locations,
three whole-location folds and predicted-point selection mask. No new data,
features, coordinates, metadata inputs, native cases, VAL augmentation or label edits.
Phase comes from the existing TRAIN image's declared target state ONLY to build
supervised labels and sample weights. All binary-background labels remain0.

Two fresh HistGradientBoosting arms, same seed42, learning_rate=.1,15leaves,
min_samples_leaf50,L2=10,max_bins255,early_stopping=False:
- `binary400`: original binary labels, balanced binary class weights,400iterations.
- `joint100`: background/red/yellow/green labels,100iterations with four trees per
  iteration; total400trees. Sample weights allocate total mass1/2to background and
  1/6to each phase. Preserve the binary foreground prior1/2; no implicit75%prior.
  Set class_weight=None and compute weights from each fold's training labels only.

Equal tree count is a capacity control, not equal parameter count or exact CPU work.
Record actual times, tree counts and row counts. Previous binary100study is retained;
this is a separate architecture comparison, not continuation of its fitted model.

Six internal fits. Aggregate each arm's OOF foreground probabilities on ORIGINAL
TRAIN features with the same macro location/class-balanced binary logloss. Select
lowest, tie to binary400. Selection does not use the original component VAL or
phase-specific validation errors. Only then fit the selected arm on all TRAIN and
assess original144VALimages/72housings once. Every model, OOF vector and source hash
is retained; no unselected-arm component evaluation or threshold search.

Inference: binary arm retains original proposer state; joint arm uses argmax among
its three RGB-predicted foreground states. Both accept sum foreground probability
>=.5, rank by foreground confidence, retain every original38,543proposal, and use
the same point-in-housing matching. For joint, save full four-class probabilities.
Same qualification: >=44joint lights, every town and state non-regressing versus
parent, >=90%fewer known-background peaks. No driving or M4primary promotion.

Run existing trainer with `--state-aware --out outputs/navigation_signal_phase_boost_20261021`.
This mode overrides the earlier augmentation-arm recipe only as declared here;
default behavior remains unchanged. CPU two threads,180s per fit,1800s whole stage,
100MB new artifacts. Hash-reference existing augmentation cache; do not copy it.
Same original storage guardian baseline14GBnet/10GBminimum, STOP/normal cleanup.
No benchmark/reward/interface/split/finalTown05 changes. Results are development
evidence on a repeatedly inspected component set, never a final driving result.
