# M1 data engine and pilot — 15 October 2026

The collection engine and small-pilot gate passed. At pilot closeout (14:24 IST),
20k collection was blocked by a kernel/userspace NVIDIA mismatch. The host reboot
later resolved it. For the subsequent collection, boundary bug and corrected live
run, see [the collection note](navigation_m1_collection_20261015.md). The commands
below describe the original request; use the current runbook before launching.
Pilot data need the new spatial-clearance audit before any training use.
No camera model is trained.

## Approved location split

The original development suite used every usable stop-sign junction and almost
every other junction, leaving no rule-complete training set with strict location
separation. The user approved a new location split before student training.
`dev_v1` and its expert results remain intact as legacy calibration; they must not
be described as results on the new held-out suite.

`configs/encounters/location_split_v1` freezes **62 training and 17 held-out
junctions**. The builder uses geometry, control availability and seed 20261015,
accepting the first partition meeting its declared coverage constraints, attempt 3.
It never reads driving outcomes. Both partitions have 300 encounters, 50 per family.
Every planned route's crossed junctions must belong to its partition; collection
also rejects actual ego trajectories entering a held-out junction.

| Town | Training junctions | Held-out junctions |
|---|---:|---:|
| Town01 | 9 | 3 |
| Town02 | 6 | 2 |
| Town03 | 20 | 5 |
| Town04 | 20 | 5 |
| Town10HD | 7 | 2 |

Held-out stop trials reuse **two junctions**, versus five on the training side;
clustered analysis is essential. The new held-out suite needs its own expert
reference before student comparisons. Town05 remains excluded from learning.

## Data and recovery contract

Each episode contains `data/episode.json`, atomic `shard_*.zip` files and a
post-evaluation `outcome.json`. A shard covers 32 timestamps, or 3.2 simulated
seconds at 10 Hz, from 20 Hz physics/cameras. Interrupted `.part` files are ignored;
committed archives survive a later crash. File and directory fsync precede/follow
the atomic rename. A new simulator attempt never overwrites an earlier episode.

Policy records contain raw three-view RGB, calibration, a fixed map-route vector,
ego motion and timestamps. The route projector uses navigation geometry without
expert hazard decisions. Privileged world/teacher facts and expert/requested/
executed controls occupy separate label members. The policy reader never opens
those labels. A sampled observation commits only after `get_control()` at the
next physics frame confirms which command actually executed.

RGB streams use lossless `libx264rgb`, CRF 0, in per-view Matroska members. Decoded
pixels must match the original RGB hashes; there is no YUV conversion or burned-in
annotation. The loader holds one small shard cache. Original PNG smoke data remain
readable. Training modes explicitly select imitation or prediction: unsafe,
unknown, truncated and safety-stopped episodes are not clean imitation targets.
Prediction data retain censoring flags; safety-stopped branches are never labeled
safe counterfactuals.

## Measured pilot

| Run | Samples | Result |
|---|---:|---|
| Original PNG writer smoke | 468 | Stop-sign arrival; exact RGB and actuator alignment |
| Lossless-video repack of same smoke | 468 | Exact pixel hashes; about 770 → 331 KB/timestamp |
| Five-town/six-family expert pilot | 1,460 | All six arrived; zero recorded safety/rule infractions; zero actuator mismatches |
| Bounded-noise/sample-cap test | 96 | Exactly three shards; 18 perturbed control ticks; zero feedback mismatches |

The capped test intentionally stopped at 67.07% route completion and is marked
censored and ineligible for clean imitation. It is not a successful driving trial.
Noise pulses add signed steering 0.04–0.08 for at most 0.3 s per 2 s, only on clear
approaches at 0.5–8 m/s outside junctions. New teacher hazards/braking cancel the
pulse and trigger a privileged collection brake, logged as an intervention.
This guard is collection machinery, not evidence that a camera policy is safe.

All committed pilot RGB frames and label pairs passed decoding, hashes, chronology,
control-feedback and outcome audits. The broader pilot projects **8.9 GB for 20k**
timestamps and **44.6 GB for 100k**, excluding videos, traces, caches and models.
These are scene-dependent estimates. Reassess storage after 20k before approving
100k; the writer preserves at least 25 GB free disk.

Validation: full CPU suite 219 passed/two skipped before the final queue-resume
test; all 14 collection regressions passed afterward. Changed-file lint/format
checks pass. Tests include crash-prefix recovery, exact RGB/frame order, corruption,
privileged-label isolation, feedback timing, eligibility, deterministic location
splits, duplicate-free resume and failed GPU readiness.

## Next concrete run

The resumable collector cycles family/town strata without consulting outcomes,
using two expert episodes then one noisy episode. It counts committed shards,
caps the final episode at exactly the remaining sample budget, preserves every
attempt, verifies source/config hashes on resume and audits all data on exit.
One server runs at a time. An invocation stops after 90 minutes plus cleanup;
it does not authorize a multi-hour training job.

After reboot, verify NVIDIA readiness and run:

```bash
nvidia-smi
systemd-inhibit --what=sleep:idle:handle-lid-switch --why='M1 bounded collection' \
  .venv/bin/python -u scripts/collect_navigation.py \
  --out outputs/navigation_m1_20k_20261015 --target-samples 20000 --max-wall 5400
```

Progress: `python3 scripts/watch_navigation.py --out outputs/navigation_m1_20k_20261015`.
Stop: `touch outputs/navigation_m1_20k_20261015/STOP`. The STOP file is intentionally
not cleared automatically; remove it only when deliberately resuming. Rerun the
same collection command to resume with the same source/config request.

At 14:05, host packages updated `nvidia-utils` to 615.78.08-2 while kernel module
615.71.09 remained loaded. NVML reports a driver/library mismatch. A native client
timeout occurred before the first noise probe frame; a causal link is plausible,
not established. The initial failed probe and successful capped retry are retained.
New simulation now refuses failed GPU-memory queries. The agent did not modify
driver packages, reboot the host or launch the 20k stage.

Artifacts: [machine-readable report](navigation_m1_pilot_20261015.json),
`outputs/navigation_m1_pilot_20261015/data_audit.json`,
`outputs/navigation_m1_noisy_cap_retry_20261015/data_audit.json`, and
`configs/encounters/location_split_v1/locations.json`.
