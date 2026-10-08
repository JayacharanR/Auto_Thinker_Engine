# Navigation experiment runbook

This branch adds a separate camera-navigation experiment. The existing Dreamer
results, reward contract, job scripts and best90k checkpoint remain unchanged.
See [the approved plan](navigation_plan.md) and [configuration](../configs/navigation.yaml).

## Install and inspect

From the project root, using the existing Python3.10 environment:

```bash
.venv/bin/python scripts/setup_navigation.py --install
.venv/bin/python scripts/run_navigation.py doctor
```

Source and small additional Python packages go under `outputs/navigation_vendor`.
The source revision and package versions are pinned in
`configs/navigation_vendor.json`; the installer does not change `.venv`.
The tracked patch removes PDM-Lite's forced-throttle escape, rejects invalid
actor geometry instead of deleting actors, resolves speed-limit paths relative
to source, handles `_Opt` map names, and fixes a setup-failure reporting call.
This modified privileged expert is not claimed to reproduce published scores.

The optional harness tests require this installation; the core navigation tests
do not. Importing CARLA's Python package does not require a running simulator.

## Short profile and full expert route

The checked-in M0 route is a development mission on Town03_Opt. It is not part
of the final Town05 benchmark. Its original OpenDRIVE path is 1228.73m; the
harness interpolation measured 1233.36m. The route adapter rejects length drift
above 2% or 10m, whichever is larger. The route manifest hashes the XML and map.

```bash
.venv/bin/python scripts/run_navigation.py expert \
  --out outputs/navigation_profile_new --max-steps 200 --max-wall 300

.venv/bin/python scripts/run_navigation.py expert \
  --out outputs/navigation_expert_new --max-wall 900
```

Use a new output directory for every attempt. For a sustained run, prefix the
command with `systemd-inhibit --what=sleep:idle:handle-lid-switch --why=navigation`.
The runtime refuses competing navigation runs and active historical trainers.
It owns one server on port 2100 with Traffic Manager 8100 and uses SIGTERM only
for owned processes. A separate evaluator child contains native client crashes.
The existing CARLA TM cache workaround must already be applied.

The installed CARLA defaults to Town03_Opt despite the command-line map argument.
The adapter loads a different requested map **before** creating Traffic Manager;
all five training towns have passed bounded camera probes through this path.
Reloading with an active Traffic Manager crashed during initial integration.
A map mismatch is a failure, never silently accepted as the requested town.

Watch without changing the run:

```bash
python3 scripts/watch_navigation.py --out outputs/navigation_expert_new
python3 scripts/watch_navigation.py --out outputs/navigation_expert_new --once
```

Stop cleanly:

```bash
touch outputs/navigation_expert_new/STOP
```

Ctrl-C in the watcher only stops watching. The STOP file is checked during
startup and scenario ticks. The owner also polls it; after a 60s grace period it
may terminate its hung evaluator child with SIGTERM. No force-kill fallback is
used. A cleanup failure is preserved in `cleanup_failed.json`.

## Evidence per attempt

- `manifest.json`, `source_hashes.json`, `source/`: config, seed, route and code provenance.
- `leaderboard.json`: original upstream criteria and Driving Score records.
- `summary.json`: all collision events, completion and stricter safe/compliant completion.
- `progress.json`: current measured state; the bar uses evaluator route progress.
- `episodes/*/trace.jsonl`: controls, ego pose/speed, privileged expert diagnostics.
- `episodes/*/world_trace.jsonl`: audit-only actors, velocities and traffic-light phases.
- `runtime_sign_inventory.json`: actual loaded traffic actors, independent of XML signals.
- `episodes/*/three_cameras.mp4`: synchronized left/front/right H.264 video.
- `episodes/*/profile.json`: throughput and expert latency, with scope stated.
- `*.log`: CARLA recorder evidence; `carla.log`: simulator stdout.
- `error.json`/`stopped.json`: explicit failure or truncation reason.

A 200-step profile is deliberately truncated and cannot count as successful
arrival. Never mix it into final mission statistics. Zero collisions in a
profile, or even one completed route, is not a reliability claim. OpenDRIVE
signals are insufficient for rule coverage: the runtime contains 20 stop actors
in Town03 and 8 in Town04 despite their XML declaring none. Presence alone does
not establish an applicable encounter or compliant full stop on a given route.

Audit preserved evidence and compare two 200-frame same-seed runs:

```bash
.venv/bin/python scripts/assess_navigation.py --out outputs/navigation_m0_long_20261008
.venv/bin/python scripts/assess_navigation.py \
  --out outputs/navigation_m0_repeat_b_20261008 \
  --compare outputs/navigation_m0_repeat_a_20261008
```

New routes use the exact serialized original start/destination for planning.
Do not reuse the deprecated manifests listed in `configs/routes/README.md`.
`--rebuild-summary` preserves the initial derived summary and leaves raw records
untouched; it also fixes execution status for historical setup failures.

## Encounter inventory (rule applicability)

```bash
.venv/bin/python scripts/build_encounter_inventory.py \
  --out outputs/navigation_encounter_inventory_NEW
```

Starts one owned server per training town in turn, loads the map without a
Traffic Manager, and records the pinned evaluator's own red-light/stop-sign lane
applicability and every junction movement. A valid movement needs at least a 30 m
approach, and the harness planner must take it. Each town runs in a child
process, so a native client abort cannot orphan the server. It never drives,
never loads Town05 and freezes nothing. Current result:
`outputs/navigation_encounter_inventory_v2_20261009`.

## Development encounter catalog (frozen dev_v1)

`configs/encounters/dev_v1` holds 300 encounters (50 per family) over the five
training towns, selected by `scripts/build_encounter_catalog.py` with a fixed
seed, without driving outcomes. Do not rebuild or edit it: the builder refuses
to overwrite it, and `catalog.sha256` is checked by the runner. Scenarios:
NavigationSignalHold (held red, declared release rule), HardBreakRoute,
SignalizedJunctionLeftTurn and NonSignalizedJunctionLeft/RightTurn. Traffic seed
= `--seed` + route id.

```bash
systemd-inhibit --what=sleep:idle:handle-lid-switch --why=navigation \
  .venv/bin/python scripts/run_encounters.py \
  --out outputs/navigation_expert_ceiling_NEW --seed 42
python3 scripts/watch_navigation.py --out outputs/navigation_expert_ceiling_NEW
touch outputs/navigation_expert_ceiling_NEW/STOP   # clean stop; rerun the same command to resume
```

Chunks of 10 routes each get a fresh owned server. Only a missing record or a
simulator crash is retried, once, and every attempt is kept. Headline per family:
`rule_safe` (completed with no collision, rule, route or blocking infraction).
Min-speed infractions and the strict any-infraction count are reported beside
it. Intervals must be clustered by junction/stop location before any claim.

## Current research boundary

M0 integrates the privileged teacher, cameras and evaluator. It does not yet
train a camera student or claim a JEPA navigation improvement. Remaining gates
include encounter coverage/expert calibration, then M1 data and M2 imitation.
Multi-town probes and a 200-frame observable-world repeatability check now pass;
branching determinism is still unproven. Final Town05 routes must be frozen before model
selection. Multi-hour collection/training budgets are proposed from measured
throughput and require user approval as agreed.
