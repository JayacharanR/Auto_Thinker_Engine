# Frozen vehicle perception diagnostic

Same165 source timestamps/native RGB hashes, fixed0.35 threshold and10m ground-origin matching gate. Nearby geometry is not a visibility-certified detection denominator.

| Frozen COCO model | Matched nearby origins | Median matched origin error | Unmatched detections | CPU p95,3views |
|---|---:|---:|---:|---:|
| SSDLite320 COCO | 15/58 | 1.61m | 10 | 82.0ms |
| FasterRCNN FPNv2 COCO | 54/58 | 1.41m | 382 | 10782.9ms |

No CARLA fine-tuning or threshold sweep. Unmatched detections include hypotheses about people/vehicles absent from the dynamic-actor reference; they are not exhaustively annotated false positives. FPN has209 unmatched person,87 truck,43 car,40 bus and3 bicycle hypotheses. Static props/parked objects need an independent audit. Stronger geometric availability does not establish safe planning; GPU latency, temporal tracking, false braking and new-scenario calibration remain open. Preserve SSDLite failure.
