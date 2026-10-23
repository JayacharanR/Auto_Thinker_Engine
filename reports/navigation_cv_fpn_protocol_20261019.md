# Stronger frozen road-user comparator — before its native predictions

The frozen SSDLite check reports15/58 nearby actor origins matched at a10m gate,
with many visibly present lead-car snapshots missed. Retain that complete failed
diagnostic. Register a separate, untuned COCO FasterRCNN-ResNet50-FPN-v2 arm using
the existing official checkpoint, all91 classes, native512 minimum resize and
1024 maximum, FP32 CPU for this geometric diagnostic. Keep the same0.35 threshold,
road-user classes, ground-contact approximation,2m cross-view merge, every165
snapshot,10m origin-matching gate and separate same-frame GT scoring. No CARLA
fine-tuning, label borrowing, threshold sweep or selected successful subset.

This is an engineering comparison on familiar strict-TRAIN prototype snapshots,
registered after the SSDLite failure was seen. It is not untouched evaluation,
M4 encoder evidence or proof of depth/visibility accuracy. Use the same full
failure and timing report. The heavier detector must separately establish
deployment latency, tracking and new-scenario availability before CV planning;
an empty prediction list never proves a road is clear. No simulator facts enter
the inherited detector inference method.
