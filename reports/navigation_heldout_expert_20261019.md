# Held-out location expert reference

The new location_split_v1 heldout_dev suite completes300/300 records on17
reserved junctions, native seed42+route_id, two resource-gated servers.
Verified headline: **294/300 rule-safe (98.0%)**,57/300 strict-safe,
297/300 arrivals with valid setup. Native raw arrivals299/300 and raw rule-safe
296/300 include two missing HardBreak scenarios (116,189); neither can pass
in the audited headline. Keep every planned case in the denominator.

Two collision events (102,177), one red violation (187), one blocked driver
(140), zero stop violations. Stop-sign50 trials reuse only two locations.
Rule-safe junction-cluster bootstrap95%:95.75–99.44%; independence-based Wilson:
95.71–99.08%. These do not establish99.3% encounter reliability or a safety guarantee.
Strict-safe retains speed-relative-to-traffic penalties; rule-safe follows the
existing predeclared progress-penalty distinction, and all infractions remain.

|Family|Verified rule-safe / planned|Missing setups|Collisions|Red|
|---|---:|---:|---:|---:|
|crossing_yield|48/50|0|2|0|
|following_stopped_vehicle|47/50|2|0|1|
|route_junction|50/50|0|0|0|
|signal_release_memory|49/50|0|0|0|
|signal_stop|50/50|0|0|0|
|stop_sign|50/50|0|0|0|

Town10HD evaluator aborted once on266; only that route was charged a crash,
eight following routes had not been reached. Existing infrastructure retry
settled the same frozen routes; all raw attempts remain.33 attempt source
archives verify against their stored hashes.100 held-red physical-light
identity audits pass;198 scripted scenarios started,100 routes require no
scripted setup, two setups missing. No route was replaced or easier seed chosen.

This is a privileged expert reference, separate from the earlier dev_v1
291/300 calibration. It is not a camera-model result. No student has been
selected using these held-out results. M2–M6 driving/research remain open.

Reproduce: `.venv/bin/python scripts/assess_encounters.py --out
outputs/navigation_heldout_expert_20261018 --report
reports/navigation_heldout_expert_audited_20261019.json --wait-for-run`.
