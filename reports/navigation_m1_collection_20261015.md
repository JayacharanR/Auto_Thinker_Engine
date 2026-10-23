# M1 initial collection — 15 October 2026

Completion update: the corrected run finished20k at15:39:02 IST and passed its
integrity/spatial audits. See [the final collection assessment](navigation_m1_assessment_20261015.md)
and [first student result](navigation_m2_student_smoke_20261015.md).
The launch history below remains evidence; no collection service is running.

The host reboot resolved the NVIDIA mismatch: kernel and userspace both report
615.78.08. The corrected 20k collection started at 14:57 IST under user service
`navigation-m1-20k-v2-20261015.service`, output
`outputs/navigation_m1_20k_v2_20261015`. Check live status; this note records launch,
not completion. One CARLA server, 90-minute invocation budget plus cleanup, with
automatic full pixel/label audit on exit. No camera student has been trained.

## Preserved boundary failure

The first collection, `outputs/navigation_m1_20k_20261015`, ran for about 64 s
and committed 297 timestamps in 13 shards. Three encounters completed rule-safe;
Town03 route 94 aborted at 0.25 simulated seconds before any steering perturbation.
The runtime guard rejected held-out junction 103. The queue was stopped with STOP,
and the full data integrity audit passed. This verifies stored pixels/actions,
not spatial eligibility. Keep the output and exclude it from training.

The offline planner labels the spawn as road 24/lane 1, outside a junction, but
its first point is only 0.0000153 m from a waypoint in held-out junction 103.
Stationary ego rounding changes CARLA's nearest-road lookup to road 123/lane 1
inside junction 103. Checking junction IDs on one planner polyline missed the
coincident endpoints. Two earlier samples had committed; none of this run is
included in the fresh collection. Town02 route 46 also fails the new geometric
clearance despite completing without a runtime guard error.

## Countermeasures

Before launching simulation, training collection now compares 1m-spaced planned
route points with 0.5m-spaced held-out junction waypoints in 3D. Any route with
less than 1m clearance is excluded from the collection schedule. This conservative
geometry-only rule is applied to every training route without reading outcomes.
The map and inventory hashes must match the frozen reference. It is a numerical
buffer, not proof of complete visual/geographical independence; the runtime guard
remains active and now writes explicit violation evidence and stops the whole queue.

The audit excludes 13/300 training routes: Town01 4/19/20; Town02 46/59/83;
Town03 94; Town04 187/188/206/210/221/229. Retained counts: crossing 45, following 50,
route/junction 45, signal release/memory 50, signal stop 47, stop sign 50. Every
original family/town stratum remains represented. Frozen catalogs, 62/17 location
assignments, held-out routes and evaluation metrics are unchanged. The 300-trial
held-out suite has not been driven and still needs its own expert reference.

The eligibility manifest is frozen in the collection request; source/config or
eligibility changes refuse resume. A fresh output avoids mixing code versions.
Harness records saying "Agent crashed" now count as collector errors even when
the evaluator process exits zero; three consecutive errors stop collection.
Collisions and other ordinary driving failures remain retained outcomes.

Evidence: `outputs/navigation_m1_geometry_audit_20261015.json`, original console/
trace/data audit, and the new run's `geometry_eligibility.json` and `request.json`.
All 16 collection regressions pass; changed-file lint/format and whitespace checks
pass. Earlier pilot data must pass this spatial eligibility rule before training.

## Live progress and next gate

```fish
cd ~/Project/Auto_Thinker_Engine; and python3 scripts/watch_navigation.py --out outputs/navigation_m1_20k_v2_20261015
```

Stop through `touch outputs/navigation_m1_20k_v2_20261015/STOP`. Do not start a
duplicate. The bar counts committed three-view timestamps. After the run, inspect
`data_audit.json`, coverage, clean-imitation versus prediction-only data, all
collisions/interventions and disk usage before beginning M2's camera CNN student.
Do not scale to 100k or start multi-hour training without the agreed approval.
