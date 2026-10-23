# Frozen-proposer RGB feature fusion — prospective engineering protocol

Status: registered preparation; no cache extraction or classifier fit yet.
Host process/GPU readiness must be rechecked after automatic approval review is
available. The preceding host check was not executed because review hit its usage
limit. Do not infer host liveness solely from a stored status or bypass that check.

Staged cache command (not yet executed):
`.venv/bin/python scripts/cache_navigation_signal_proposer.py --out
outputs/navigation_signal_proposer_features_20261021`.
The command runs only after the host check and GPU readiness are available.

Hypothesis: the current compact verifier discards the native CNN proposer's learned
representation and confidence. Preserve it alongside raw RGB to test whether it
improves generalization. Prior state-aware model44/72joint still fails yellow and
Town10HD protection; opponent-color features lost internal selection. Retain both.

Exact frozen proposer checkpoint: outputs/navigation_signal_pixel_study_20261021/
bulb/step1000.pt, SHA256
27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193.
For every original point, gather32last-convolution channels and4softmax probabilities
from a FULL native512x288image forward. XY addresses the map only; it is never an
input feature. Do not infer features from separately padded crops. No weights,
normalization state, proposal coordinates, labels or source images are updated.

Inputs remain RGB. Retain the exact51,094TRAINpoints and original38,543VALproposals;
never regenerate, relocate, cap or filter the proposal set before verification.
TRAIN gets original/gamma.7/gamma1.4images with unchanged labels; use the same
full-image monotone transforms as the preceding augmented RGB study. Extract36
features for each version; concatenate with the corresponding87RGBmeasurements
to produce123features. Feature caches use FP16; record inference precision/batch
shape. Original and augmented images must reproduce the existing RGB features;
verify untransformed proposer probability maps against the original forward path.
Record discrepancies instead of assuming different execution batches are bit-exact.

Cache stage: hash verified original sources/labels/folds, exact frozen model before
and after, per-image ordered point identities, every feature array and completion
denominator. 1800seconds/150MB cap, original storage baseline. Complete and audit
the cache before fitting; partial/unverified caches do not enter fits.

Classifier study: raw87joint100 versus fused123joint100, same400trees,
sample weighting(background1/2, each phase1/6), gamma data, hyperparameters,
three TRAIN-junction folds and seed as navigation_signal_phase_boost_20261021.
Exact raw-fold prediction parity required. Same macro TRAIN-location binary OOF
loss selects the arm; tie to raw. Only selected arm fits all TRAIN and reaches
component VAL once. If raw selected, exact original final TRAIN/VAL parity required.
Seven owned CPU fits;180s each/1800s stage/100MBnewartifacts. Equal tree count is
not equal CPU work; record feature dimensions, fitted row counts and elapsed time.

Inference uses predicted foreground sum>=.5 and argmax of RGB-predicted phases,
with the original point-in-housing evaluation. Same>=44joint and every town/state
non-regression plus>=90%fewer background peaks than37,241. No new threshold choice,
reward, benchmark/split, allowed external input, M4primary or finalTown05change.
No automatic native driver deployment from this component study.

Limits: the proposer was fitted on the original eligible TRAIN subset and selected
using component validation before these experiments. Internal folds validate the
NEW downstream head, not an entirely unseen end-to-end feature-learning pipeline.
Frozen learned features may amplify that exposure; disclose it and use untouched
driving locations for any eventual generalization claim. Component VAL has been
inspected repeatedly; no test-performance or driving-safety claim follows here.
All failed models/data remain retained. Original guardian baseline14GBnet/10GBmin
and STOP-only owned-job shutdown remain. No source-pin changes during active work.
