# Native bulb supervision versus housing-center control

Prospective component experiment, registered after the TRAIN-only pixel audit
and before any fit. Audit: outputs/navigation_signal_pixel_audit_20261021.
73 eligible static poses/seven locations, 92 responsive red/green housings and
90 yellow; all supported housings have nonempty fixed masks. Exclude65oldTRAIN
poses (63newly reserved; two actual poses inside reserved junctions). Keep the
same144original component-validation images/72labels. No validation pixel masks.
This set is previously inspected development data, not a new held-out test.

Two fresh seeds42 fits, center versus bulb supervision, identical37,772parameter
native stride1 CNN with21pixel receptive field (six3x3layers12/24/32/32/32/32,
dilation1/1/2/2/2/2). Both see one RGB view only. No map, coordinates, teacher,
phase-pair or traffic state at inference. Pair images and map are label-only.
No old detector weights or previously trained reserved-location checkpoint.

1000updates each, AdamW lr.001/wd.0001, clip1, BF16forward/FP32crossentropy,
batch12native64pixel crops, half class-balanced housing crops/half random
TRAINimages; fixed seed and same housing/class sampling distributions. A positive
anchor is a housing center or uniformly selected bulb pixel, respectively, so
crop positions are not identical. Foreground meanCE plus.2background meanCE,
as in the original grid study. Projected light-ignore zones unchanged. No hue
augmentation that would invalidate the label. Checkpoints0/100/250/500/1000,
all optimizer/RNG states retained. No losing-fit extension.

Both arms use the SAME prospective component selection loss: for each known
housing choose the most confident foreground pixel within its existing box and
score negative log probability of its true state; mean across known housings,
plus.2mean backgroundCE outside every projected housing. Lowest score wins,
earlier tie. Report every checkpoint, never reselect by a diagnostic score.

Operational diagnostic: threshold.5,8-connected foreground components, one peak
per component; score-order one-to-one matching of a predicted point inside an
existing known housing. Report all72known labels, correctly localized+classified,
each town/class, original center accuracy, background-active pixel fraction and
peaks in known-background regions. Other light-ignore regions are unknown, not
false positives. This is point localization, NOT bounding-box AP/IoU and cannot
be compared numerically to the old44/72bounding-box detector result.

Bulb-label countermeasure is useful for further development only if its selected
checkpoint improves joint point localization over selected center control, with
no town joint-count regression and no increase in background active fraction or
background peak count. This is not a driving gate or replacement benchmark.
Measure three-view GPU inference latency separately; only a later independent
native pilot can establish driving or signal applicability. No automatic pilot.

Maximum1200s per fit,2800s stage including preparation,250MBnew artifacts;
original shared guardian baseline and STOP. No new native recording/data.
No original M4primary, M3suite, M6suite, runtime input or benchmark change.
