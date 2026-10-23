# Relative-color RGB representation, paired against exact state-aware control

Register before fitting. Joint state supervision reached44/72total but regressed
yellow15→10 and Town10HD10→9, so it remains rejected. Test whether channel
relationships and local contrast are easier for a small tree classifier to use
than raw intensities alone. No claim this explains the observed failures.

Reuse exact original87RGBfeatures and two gamma variants, existing TRAIN labels,
21junctions and three whole-junction folds. No images/features are recollected.
Both arms use joint100 four-class boosting (100iterations×4trees), same sample
weights(background1/2,each phase1/6), settings, RNG and datasets as the last study.
- `joint100`: exact87feature control. Require bit-identical original TRAIN
  prediction vectors versus every retained joint100fold before accepting comparison.
- `opponent100`: append225 features from the original5x5RGB patch: pixelwise
  R-G,G-B,R-B(75), chromaticityR/S,G/S,B/S withS=max(R+G+B,1e-6)(75), and each
  channel minus its5x5mean(75). Total312features. Keep original87unchanged.
  These are deterministic RGB transformations, not teacher/map/state inputs.

Same tree count and tuning budget; more features may cost more CPU. Measure times.
Six internal fits; selection uses the unchanged macro TRAIN-location binary OOF
logloss, tie to raw joint100. Only the selected final fit reaches component VAL.
If raw wins, require exact original component probabilities as a repeated control.
No additional state weighting, threshold adjustment, candidate filtering/cap or
native deployment. Infer summed foreground probability>=.5 and RGB-predicted
foreground-state argmax. All38,543original proposals remain scored.

Unchanged gate: >=44joint, every town/state non-regressing, >=90%fewer background
peaks than37,241. Keep all144VALimages/72labels, prior failures and denominators.
This is development engineering on an inspected component set; no finalTown05
tuning, benchmark/reward/interface/split/M4primary change.

Run RGB trainer `--opponent-colors --out outputs/navigation_signal_opponent_20261021`.
This explicit mode overrides the previous arm recipe only as declared here.
Seven CPU fits, two threads,180s perchild/1800s stage/100MBnewartifacts. Reference
cached features without copies. Original guardian baseline14GBnet/10GBminimum,
source pinning, STOP and ordinary owned-child cleanup remain.
