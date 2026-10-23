# One TRAIN-only classifier calibration check

The protected hard-negative fit rejects all trained checkpoints; latest200 has
27/72 detections,25 joint states and171 approximate-background candidates versus
parent30/72,24 joint and681. Keep both raw outcomes and selected parent0.

One new candidate, not a threshold search: use the fixed completed latest200 and
the original parent50. Freeze all tensors except a possible foreground logit-bias
addition. Extract shared frozen ROI features from the same179 supported TRAIN GT
boxes and440 verified negative phase copies. No validation ROI/image participates
in feature extraction or scalar fitting; negatives are diagnostic only.

Let K be the number of TRAIN positive GT ROIs accepted by the parent at.5. Find
the smallest NONNEGATIVE foreground-bias addition that makes the candidate accept
at least K of those179 TRAIN ROIs. If it already does, add zero. Use kth-largest
candidate positive margin and1e-6 tie epsilon. No temperature tuning, negative
validation examples, held-out outcomes or choice among multiple scalars.

At zero offset, reuse the raw200 evaluation only after verifying exact weights;
do not rerun identical inference. Otherwise evaluate the single calibrated model
on the unchanged144 images once at the unchanged.5 confidence/IoU criteria. Check
one-to-one recall>=30,joint states>=24,approximate-background<=681 and loss below
parent.118660469. Save separately as a candidate, never overwrite original runs.
The parameter rule uses TRAIN data; it is not a new benchmark threshold or policy
override. Preserve any tradeoff or negative result; no further automatic tuning.
GPU/CPU diagnostic cap300s, STOP. Camera driver and held-out suites unchanged.
