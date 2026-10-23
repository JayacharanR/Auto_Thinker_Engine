# Crop alignment robustness — before fitting

The fixed detector selection is step50 (loss.11866),30/72 detected labels and
21/72 detected-and-correct states. Do not substitute step200's higher recall45/72.
All boxes/confidences/assignments remain frozen for this experiment.

Hypothesis: tight native housings are only a few pixels wide; small proposal-box
errors disturb the linear classifier's normalized RGB layout. Test TRAIN-only
outward crop augmentation on the existing classifier architecture, without new
simulator data or changes to detector selection, camera pixels or driving inputs.

For each of179 TRAIN source crops, retain its exact box and generate15 outward
expansions, independently uniform integer margins: left/right0..max(1,ceil(.3*width)),
top/bottom0..max(1,ceil(.2*height)), clipped to frame. Every expanded box contains
the complete original housing, so cropping never deliberately removes its bulbs.
Seed20261016. All2,864 training crops come from the same13 TRAIN component groups.
Use the same16x32 bilinear RGB feature and class-balanced L2 logistic regression,
C1,lbfgs,max_iter1000. Weight each augmentation1/16 to retain total179 source-example
weight, rather than changing regularization by duplicating examples. No tuning,
validation-based checkpoint choice, threshold search or new model architecture.

Report exact GT validation crop states separately as a privileged diagnostic.
For deployment-like state measurement, reclassify ONLY the frozen detector's
predicted boxes. Use the geometry-only one-to-one assignments already audited at
IoU.3. Denominator stays72; the matched subset stays30. Record old and new results
on those identical predicted ROIs, including per-state confusion. Geometry enters
training labels/scoring, not classifier prediction. No lane or driving claim.

Archive source/request before fit and verify exported coefficients reproduce
predictions. CPU fit, no new GPU training; expected seconds with two BLAS threads.
This is a development test on already inspected TRAIN locations. A negative result
is retained; do not keep tuning on these72 labels.
