# Frozen spatial split

User approved on 2026-10-15 before student training. `locations.json` and its hash
declare 62 training and 17 held-out junctions; `train/` and `heldout_dev/` each
contain a frozen 300-encounter catalog, with separate hashes and XML manifests.

The generator selects geometry only, seed 20261015, first qualifying partition
at attempt 3. Planned routes may not cross the opposite partition. Training
collection also checks the actual ego junction. No Town05 input is permitted.

Never overwrite these files after student results. The old `../dev_v1` remains
the legacy expert-calibration suite, which shares locations with the new train
partition. Its 291/300 score is not an expert reference for this held-out suite.
The new held-out stop family uses two physical junctions across 50 trials;
cluster intervals by location and disclose this reuse.

Build commands and measured collection evidence are in
`reports/navigation_m1_pilot_20261015.md` at the repository root.
