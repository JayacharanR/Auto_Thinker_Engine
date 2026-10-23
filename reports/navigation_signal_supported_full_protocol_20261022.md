# Full TRAIN proposal audit of supported supervision

Registered after the three-fold study and before any full-image verification.
The TRAIN-only sampled comparison passed: joint1,141→1,144/1,248, background1,990→201,
no town/phase regression. Point logloss worsened .360094→1.264219 because the model
no longer learns phase from every housing point; this tradeoff stays disclosed.

No new fits. Use the exact saved control and supported joint100 fold models. For
each of the original2,844TRAIN images, evaluate ALL original frozen proposer peaks
(715,825 total), using the head whose training excluded that entire junction.
Recompute the same original87RGB features, round to the originalFP16 cache boundary,
then predict in float32 on CPU. Every original48,865predicted sampled row must have
bit-exact features and both OOF probability vectors at its matching coordinate.
GT-only anchors do not enter full-proposal scoring.

Preserve all original housing boxes, broad background/unknown masks, threshold.5,
foreground-score ordering and one-to-one point-in-housing matching. Count all known
background proposals and all unmatched unknowns; retain every proposal probability,
offset and source/model hash. Do not use ground-truth boxes to choose runtime crops.
Only the RGB point coordinate centers the native/context windows.

Same internal TRAIN continuation criterion: joint detection non-regression overall,
by town and by state, and strictly fewer known-background proposals. Report all
counts even if this fails. No selection sweep, trial retries, component-VAL scoring,
final fit, driving, or benchmark changes. Upstream proposer TRAIN exposure remains;
this is not end-to-end unseen-location evidence.

Output outputs/navigation_signal_supported_full_20261022. CPU2threads,900s wall cap,
100MB artifacts. BudgetWatch and original14GBnet/10GBminimum guardian; STOP/retain
failures. All registered source bytes pinned until closure; no new training.
