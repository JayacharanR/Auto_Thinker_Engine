# Full TRAIN audit of spatial-context heads

Registered while the three fixed fits run, before reading their results. Use all
12,362,313 frozen plateau-union points on the same 2,844 TRAIN images. Only the
registered spatial-context feature representation and its three trained heads
change. Every image uses the head excluding its whole junction. No new fitting.

Preserve original foreground-sum≥.5, phase argmax and original score-order matching.
Compare against the original plateau baseline: ≥1,234/1,248 joint, every town/phase
non-regressing, known background ≤3,075.8. The newer 1,232 result does not lower
this bar. No component follow-up if any criterion fails.

For speed, use the separately verified compiled CPU raw extractor: exact FP32
features/probabilities on 236,284 points across 63 fixed TRAIN images. The GPU
implementation is not used. Independently recompute original CPU raw features at
all 48,865 original sampled proposal rows. Also require exact spatial-cache
features and new OOF probabilities on those rows and all 182,016 added backgrounds.
The spatial cache already passed 15,417 direct-pixel descriptor comparisons.

Archive/hash all actual implementations and cache identities; preserve original
full baseline probabilities and recount every baseline measurement. Store all new
coordinates/probabilities/offsets and per-image receipts. No threshold changes,
component validation, final fit, deployment or final Town05 driving.

Output `outputs/navigation_signal_spatial_full_20261022`; 1,200 s / 600 MB cap,
two CPU threads, original storage guardian/baseline. STOP/errors retained, no retry.
Record measured feature/head cost but exclude proposal generation/decode/control;
this is a TRAIN engineering comparison, not real-time driving qualification.
