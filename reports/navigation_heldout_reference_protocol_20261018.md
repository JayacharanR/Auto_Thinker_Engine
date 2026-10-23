# New-location held-out reference and candidate evaluation

Use the unchanged frozen configs/encounters/location_split_v1/heldout_dev catalog:
300 encounters,50 per family at17 reserved locations across all five development
towns. This is distinct from dev_v1; its legacy291/300 cannot substitute.
Run the same privileged modified PDM-Lite expert, seed42+route_id, chunks10,
native criteria/scenarios/background. No curriculum overrides or collection.
Use up to two resource-gated owned servers, minimum6GB free VRAM/12GB available
RAM before each start, STOP at critical6GB RAM and25GB disk reserve. Estimate
completion from the live run; no scientific gain is inferred from progress.

No simulator-crash retry for driver failures. Retain every collision, timeout,
blocked case and scenario setup failure; existing infrastructure-only retry
policy allows one rerun with both attempts retained. Keep missing execution in
the300 denominator. Record source/config/catalog hashes and exact actor setup.
Do not replace difficult routes or recalibrate criteria after expert outcomes.
Report rule-safe and strict-safe completion, every infraction, each family's
denominator, Wilson intervals explicitly marked as independence-based, and
junction-clustered intervals. Compare student results pairwise to this reference
under identical seeds/catalog, with exact McNemar tests and clustered intervals.

scripts/run_encounters.py now accepts an optional immutable exported camera
checkpoint with overrides OFF by default. Its request includes checkpoint SHA,
architecture and override setting; each chunk verifies the checkpoint identity.
The expert default/request/route seeds remain compatible with existing runs.
Parent/selected model development uses TRAIN and fixed internal validation only;
heldout_dev outcomes are reports of frozen candidates, never targets to refit or
resplit. M2 camera driving and M3–M6 gates remain separate from expert calibration.
