# Navigation M0 assessment — 2026-10-08

The long-route infrastructure works on branch
`experiment/camera-world-model-navigation`. **The driver is a privileged modified
PDM-Lite expert; no camera student has been trained.** The approved research plan
is in [navigation_plan.md](navigation_plan.md), commands in
[navigation_runbook.md](navigation_runbook.md), and machine-readable evidence in
[the assessment JSON](navigation_m0_assessment_20261008.json).

## Measured results

| Check | Result | Scope |
|---|---|---|
| Town03 long expert drive | 1233.361m, 100% completion, zero recorded infractions, Driving Score 100 | One development route, one traffic seed; not an official benchmark score |
| Elapsed time | 255.4s simulated; 262.895s evaluator wall time | Excludes server startup |
| Camera stream | 5108 aligned frames; three 512×288 RGB cameras; 19.37 frames/s | 20Hz simulator; RGB views are not measured depth |
| Expert latency | 33.84ms p95 | Excludes rendering and video encoding |
| Video | 2554 H.264/yuv420p frames; 1536×288 left/front/right mosaic | Fully decoded and count checked; not independently browser-tested |
| Training-town compatibility | Town01,02,03,04,10HD `_Opt` variants each passed 60-frame probes | Probes are intentionally truncated, not successful missions |
| Repeatability | 200 frames matched exactly across two fresh same-seed runs | Ego controls; poses/velocities of 19 dynamic actors including ego; lights and phase times |

Long-drive evidence: `outputs/navigation_m0_long_20261008`. Video:
`episodes/navigation_m0_town03_opt_v2_route0_10_08_23_16_39/three_cameras.mp4`.
Repeatability: `outputs/navigation_m0_repeat_b_20261008/audit.json`.
Source archives, raw evaluator records, route hashes, control traces and CARLA
recorder logs remain in each attempt directory. All observed collisions count;
fault attribution never filters the headline.

Validation: 189 repository tests passed, one GPU-only test skipped in the CPU
test environment. Navigation accounts for 23 tests. New-code lint/format checks,
source-hash/video audits and tracked whitespace checks passed. No training or
simulator process remains active after the bounded probes.

## What changed

- Pinned CARLA Garage's `leaderboard_2` expert/evaluator to
  `48e6a6dbcb133220333bd65e242fbc2b492bb15b`; isolated extra dependencies.
- Removed the expert's forced-throttle escape and actor deletion; documented
  compatibility fixes as a verified tracked patch.
- Added rotated three-view cameras with exact-frame assertions, camera geometry,
  typed future policy inputs, owned server/evaluator lifecycle, STOP handling,
  progress watcher, provenance archives and H.264 recordings.
- Used upstream route, collision, red-light and stop-sign criteria; exercised
  red stop-line crossing, full versus rolling stops, permanent stopping and
  route-shortcut behavior in adversarial tests.
- Added independent raw-record/video/source auditing and observable-world replay
  comparison. Full camera-policy process isolation is still future work.

## Corrections and retained failures

The first native map reload crashed CARLA. Loading a map before creating Traffic
Manager now works in all training towns. An incorrect adapter tick signature and
an upstream cleanup thread also caused an initial failed attempt; both are fixed.

Intermediate-keypoint export initially expanded a route; snapped endpoint export
later shortened a Town10HD mission from 389m to 161m. The length guard blocked the
latter before driving. Generation now plans from the exact serialized original
start/destination, tested by round-trip interpolation. Deprecated manifests and
attempts remain; replacements address infrastructure geometry, never expert
driving outcomes.

My summary parser initially missed upstream `Perfect`, reporting the successful
long drive as a failure. A regression test and preserved-summary rebuild fixed
that. Another wrapper error reported a zero-frame setup failure as evaluation
complete; setup failures/incomplete profiles now fail explicitly. Raw evaluator
records were not rewritten.

**The review's map-stop assumption needs correction.** Runtime actor counts are:

| Training town | Runtime stop actors | Runtime light actors |
|---|---:|---:|
| Town01 | 0 | 36 |
| Town02 | 0 | 24 |
| Town03 | 20 | 38 |
| Town04 | 8 | 43 |
| Town10HD | 3 | 15 |

Town03/04 declare no OpenDRIVE stop signals but do contain runtime stop actors.
The earlier runbook statement that Town03 cannot test stop signs was incorrect.
Presence is not an encounter denominator: lane applicability, trigger crossing
and actual red/stop opportunities must be audited. Expert hazard flags are only
diagnostics. Town05 remains untouched.

## Remaining gates and implementation sequence

1. Build the encounter catalog from runtime trigger geometry and route/lane
   applicability. Freeze development locations and rule-family coverage before
   student results; include all expert failures. Measure the expert ceiling.
2. Implement M1 atomic data shards with separate policy inputs and privileged
   labels; retain requested/executed actions, interventions and failed/censored
   branches. Audit a small collection before 20k timestamps, then 100k.
3. Train the compact CNN recurrent camera student with separate path and speed
   heads, evaluate long routes early, and add DAgger and overrides-off rule pairs.
4. Proceed to JEPA label-efficiency comparisons and predictive planning only
   after those baselines. Keep Dreamer recovery/collapse diagnosis separate.

M0 integration checks have passed; **encounter-level expert calibration is still
pending**. Exact 10s observable-state agreement does not prove hidden simulator
state agreement or determinism after alternative actions. A single clean route
cannot establish reliability. No multi-hour collection/training was launched.

At the observed full-drive rate, 300 encounters of 20–40 simulated seconds require
roughly 1.7–3.4 hours of steady-state simulation before resets, setup and retries.
20k timestamps at 10Hz need roughly 34 minutes of steady-state simulation; 100k about
2.9 hours. These are lower-bound estimates, not queue duration promises. Request
the agreed budget approval before a multi-hour stage, using a concrete frozen
catalog and command after the small collection audit.

## Addendum 2026-10-09 — runtime encounter inventory

`scripts/build_encounter_inventory.py` applied the pinned evaluator's own
red-light/stop-sign geometry to every junction movement in the five training
towns (`outputs/navigation_encounter_inventory_v2_20261009`).

| Town | Valid / all movements | Signal | Stop | Uncontrolled | Stop actors with a valid movement |
|---|---:|---:|---:|---:|---:|
| Town01 | 72 / 72 | 72 | 0 | 0 | 0 / 0 |
| Town02 | 44 / 48 | 44 | 0 | 0 | 0 / 0 |
| Town03 | 156 / 220 | 91 | 23 | 42 | 11 / 20 |
| Town04 | 221 / 256 | 84 | 24 | 113 | 8 / 8 |
| Town10HD | 48 / 82 | 35 | 2 | 11 | 1 / 3 |
| **Total** | **541 / 678** | **326** | **49** | **166** | **20 / 31** |

Six Town03 stop actors trigger only on junction-internal connector lanes. The
evaluator would require a full stop inside the junction there, so the catalog
must route around them by design. Five more stop actors have approaches under
30 m. With 49 valid stop movements at 20 locations, the planned 50 stop-sign
encounters need a predeclared location-reuse rule and location-clustered
intervals. The first attempt's Town10HD client abort, and its orphaned server,
are preserved and fixed (see the journal). This is still an inventory: no
encounter has been selected, frozen or driven.

## Addendum 2026-10-09 — frozen development catalog and pilot

`configs/encounters/dev_v1` (catalog sha256 `716c3600…`) has 300 encounters:

| Family | Distinct movements | Distinct junctions | Towns | Scenario |
|---|---:|---:|---|---|
| signal_stop | 50 | 35 | all five, 10 each | NavigationSignalHold (release within 40 m) |
| signal_release_memory | 50 | 35 | all five, 10 each | NavigationSignalHold (start 12 m before line, 10 s hold) |
| stop_sign | 43 | 7 | Town03 29, Town04 19, Town10HD 2 | none (runtime stop actors) |
| crossing_yield | 45 | 36 | 5/5/15/15/10 | pinned junction-flow scenarios |
| following_stopped_vehicle | 50 | 47 | all five, 10 each | HardBreakRoute |
| route_junction | 49 | 46 | 8/8/12/12/10 | none |

Stop-sign encounters cover all 20 usable stop actors, 2–3 times each, but only
7 junctions, so clustered intervals are essential there. A six-encounter Town03
pilot confirmed that every scenario starts and does what it is meant to.
A runner smoke test (22 Town10HD encounters, stopped deliberately) exposed one
CARLA segfault, handled by the declared one-time retry, and two runner bugs,
both fixed before the expert-ceiling run. No student exists yet; these runs are
privileged expert infrastructure checks.
