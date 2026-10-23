# Locally supported signal supervision — TRAIN-only comparison

Registered before fitting. The fixed TRAIN audit finds 2,006 of 3,371 predicted
positive points have no audited active pixel within the detailed 5×5 RGB crop.
1,351 of these have alternate-phase local differences no greater than measured
red-repeat variation. This supports testing supervision observability; it does
not prove annotation corruption or absence of every possible visual cue.

Keep original RGB87 features, gamma .7/1.4 arrays, labels, rows, three junction
folds, seed42 and joint100 model/settings. Exclude only those 2,006 TRAIN points
from the fitting loss, including their two gamma copies. They become unknown for
training; they do not become background. All 45,494 background rows and 2,229
GT-only bulb anchors remain. Verify anchors against original active-pixel masks.
Exactly three new fold fits, 400 trees each; no hyperparameter or threshold search.
Use saved raw joint100 OOF predictions as the fixed control, without refitting it.

Evaluate BOTH heads on the SAME original 48,865 predicted TRAIN rows, including
every excluded positive. GT-only anchors never count in scoring. Report original
macro point logloss, foreground/state counts, and one-to-one housing localization
and phase counts, plus sampled background acceptance. The latter uses all original
TRAIN housing labels and identical score ordering. The sampled background count
is NOT a full-image false-positive count. Whole-location OOF applies to the new
head only; the upstream proposer has TRAIN exposure. Correlated phase images and
points do not establish driving success or independent confidence intervals.

Decision, fixed before fits: proceed toward a separately registered full-image
TRAIN verification only if joint housing count does not decline overall, by town
or by phase, AND sampled background acceptance strictly decreases. This internal
TRAIN diagnostic better matches one-detection-per-housing behavior than demanding
a correct phase from every point on a housing. It is not a change to the benchmark.
Original point logloss remains a reported diagnostic even if it worsens.

No component VAL predictions, final all-TRAIN fit, model deployment, native driving,
or final Town05 evaluation in this stage, even if the diagnostic passes. All original
component labels, thresholds, protected town/phase counts, M4 primary and gate
requirements remain unchanged. Repeatedly inspected component VAL is development
evidence, never untouched-test evidence.

Output: outputs/navigation_signal_supported_20261022. Three CPU fits, two threads,
180s per child, 900s stage, 100MB new artifacts. Existing feature caches are hash
referenced; all source/recipe identities archived. Original storage baseline,
14GB net cap and 10GB minimum apply; STOP and normal owned cleanup, no retries.
