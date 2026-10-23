# Sampling normalization audit supplement

The first fixed-model audit reproduced both saved totals in21.49s. ROI box loss
increases.043388→.054930; classifier and RPN objectness losses fall. GT ROI binary
acceptance23→29/72 remains weak; keep it privileged and reuse its exact evidence.

One new diagnostic quantity warrants one more loss pass: torchvision's ROI box
loss sums only positive examples but divides by all sampled ROIs. A better RPN
can change that positive count. Freeze the same parent50 and joint200, original
validation/seed/batches, no candidate or threshold change. Wrap the ROI sampler
to count positive/all samples without changing its returned tensors or RNG.
Report total positive/all samples and sum(loss_box_reg * sampled_count) divided
by total positives. This per-positive SmoothL1 is a descriptive training-space
residual, not physical box accuracy, a new selection metric or a promotion rule.

Fresh output, source/archive/prior-audit hashes,300s cap, STOP. Verify old loss
totals within1e-6. Reuse prior fixed GT-ROI evidence; do not repeat its features
or full RGB detections. No training, calibration, longer fit or driver integration.
