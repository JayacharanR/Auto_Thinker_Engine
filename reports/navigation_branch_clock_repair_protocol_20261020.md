# Clock/recorder lifecycle repair — before new calibration rollouts

Retain the first two failed collections and the completed v3pilot480timestamps,
8unfiltered contacts and its failed elapsed-phase replay check. No model trained
on that calibration-only data. Do not change its tolerances or replace outcomes.

A new exact-geometry six-rollout pilot uses the same Town01/6, seed62001,4s prefix,
three4s action branches/two repeats,512×288camera geometry and post-contact stop.
Before actors, reset each traffic-light group once in the paused synchronous
world, set all lights green/frozen, then acknowledge with one common tick.
Record every phase elapsed time and reject nonzero startup offsets>.051s rather
than ignoring a failed comparator. Use the current actual-validation geography
guard; the same declared Town01route6must remain the eligible first location.

After stopping the recorder, explicitly tick once to acknowledge the server's
asynchronous stop/flush, retaining that cleanup frame and all raw sensor events.
The measured dataset and action horizon remain80RGB samples/16020Hzcontrols.
Require a complete recorder with at least161frames and stable file identity
before deriving its final hash or declaring completion. Retain any extra native
frames/events; do not clip contacts from raw attempt accounting. A separate
closed-file audit still verifies all pixel hashes and actual next-tick actions.

Unchanged.1m/.5deg/.1mps/.051s replay tolerances apply to all measured prefixes
and repeat branches, including signal timers. Failed physical comparability
remains a failed calibration gate. This does not establish reactive cross traffic,
red-risk supervision or prediction accuracy. Collection is calibration-only and
cannot silently join the first registered dynamics fit. All successful and failed
attempts remain, with source/guard/phase/control/recorder identities and25GBreserve.
