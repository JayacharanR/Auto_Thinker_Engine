# Frozen FPN against dense calibration masks — before inference

Diagnose the current CVperception before building or fitting a replacement.
Use only the verified dense-label manifest's two predeclared calibration sources:
Town03/1221 andTown04/916, all78timestamps/234views. No source replacement, frame
selection, threshold sweep, new local supervision or checkpoint update. The
existing detector has COCOexternal labels and no CARLAfine-tuning. Historical
camera encoders/teacher heads have TRAINexposure here; this is not untouched
encoder evaluation or the final benchmark.

Run the exact original FrozenRoadUsersFPN.detect method, FP32GPU, native512minimum/
1024maximum resize, score.35, fixed road-user classes,60mground projection and
2mcross-view merge. Three unscored warm-ups, then every original frame once.
A read-only forward hook preserves raw COCOoutputs before the inherited filters;
the hook may affect diagnostic timing. Save each original RGB-only prediction
before loading any instance/depth target. No labels or world state enter forward.

From raw instance labels, group visible pixels by low16Unrealactor ID and semantic
road-user group: vehicle14–17, person12/13, two-wheeler18/19. Compare same-view
visible-mask bounding boxes atIoU≥.3 using maximum-cardinality one-to-one matching.
Primary visible-instance denominator requires≥8actualpixels; report every smaller
mask separately. Use broad road-user groups for matching, retain raw COCOclass
labels for later size-error diagnosis. Do not use a ground-truth actor inventory
as a visibility denominator. UnrealIDlow16can alias, and mask boxes are visible
surfaces rather than amodal boxes; this does not establish standard detectionAP.

Report raw, projectable and fused stages: included mask/view pairs, matched pairs,
predictions and unmatched predictions. Also report unique renderer actor/group
coverage across the three views within each timestamp, so expected cross-view
deduplication is not mistaken for lost physical-object recall. Never count a
prediction as two reference objects. Unmatched predictions are against this
included-mask denominator, not exhaustive false-positive proof.

Independently inspect metric depth at observed object-bottom pixels. Retain BOTH
axial and radial interpretations when lifting pixels into the declared rig frame;
report their surface heights and distances from the old flat-ground estimate.
Do not assert that a visible surface is the actor centre, choose a favorable
interpretation, or call these differences certified3Dlocalization accuracy.
No geometry interpretation or sigma is changed in this diagnostic.

Two CPUthreads, one GPUowner/global navigation lock,900swhole-stage cap and live
approved14GBadditional/10GBreserve guardian. Estimated output<30MB. No simulator,
training, new policy inputs, rule/benchmark/score changes or model promotion.
All raw source/descriptor/manifest hashes and predictions remain retained.
The result should identify whether a next experiment needs detection coverage,
geometry or tracking/risk work; its two repeated locations cannot establish safe
driving, calibrated collision probability or learned traffic-rule behavior.
