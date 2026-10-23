# Town05 final-suite registration, before further student fitting

This implements the approved12-route/three-seed M6 plan. The intended
pre-student registration was missed; camera students and component development
already exist. Disclose that timing deviation in the frozen manifest. No
Town05 student outcome has been used for selection. Geometry registration
does not provide Town05 training/pretraining examples or permission to tune.

First inventory the runtime Town05_Opt light and stop actors using the pinned
evaluator's applicability geometry. No driver or policy is loaded. Keep map and
source hashes. Verify the expected five mapped stop locations against actual
runtime actors; fail registration on a discrepancy instead of silently weakening
coverage. Source inventory and all failed infrastructure attempts are preserved.

Choose12 shortest-path routes of1–3km from serialized endpoints using the
pinned1m global route planner and fixed seed20261018. The first ten require
each of the five explicit mapped stop locations twice; last two have no forced stop location.
Every route requires a signal opportunity with a >=40m approach and >=25m exit
inside the mission; put the original NavigationSignalHold at that approach,
with unchanged signal_stop parameters. These are predeclared custom missions,
not an official Leaderboard submission. No vehicle is driven during selection.

Count planned sign opportunities conservatively using route samples within2m
of a pinned-evaluator affected-lane point, at least20m after start and20m before
end. Verify all five locations and >=10 route/sign opportunities (>=30 across
three seeds).12 held-red setups give36 planned red opportunities. Planned
coverage is not actual encountered coverage; runtime setup and encounter counts
must be audited for every final result, including early failures.

Use seeds3031/4031/5031 and the declared camera rig/canonical background, same
controls/criteria for expert and candidates. Hash XML, map, scenario parameters,
selection source and analysis before any new student fit. Register each candidate
checkpoint once before final evaluation. Count every collision; report route
completion, all infractions, opportunity denominators, interventions, comfort,
latency and confidence intervals. Driving Score remains supplementary. The
33/36 safe stretch target is not a statistical safety guarantee.

Never replace a mission because the expert/student fails. Only a documented
infrastructure defect established without using a student outcome can justify
a new version, preserving the original. Do not tune on final results. Expert
calibration on the separate heldout_dev suite still remains required.

Geometry-only clarification, before any final route/student outcome: the five
explicit XML signals are not the full runtime stop population. The inventory
found33 runtime actors; CARLA also exposes object-derived landmarks. Uniquely
match each explicit signal's physical landmark to its runtime actor (<1m),
require those five locations for planned coverage and count ALL additional
runtime stop infractions. The initial script's exact-five-runtime-actors check
was an incorrect assumption and is corrected before route selection. No actor
or criterion is removed and the approved five-location requirement is preserved.
