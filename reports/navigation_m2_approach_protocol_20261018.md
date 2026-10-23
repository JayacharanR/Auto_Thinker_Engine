# Continuous TRAIN signal approaches

The user authorized execution through M6 on18 October. Preserve the approved
milestone gates, all existing results and every validation reservation. This is
TRAIN collection and data-eligibility checking, not a reward or driving benchmark
change. No existing route/scenario is edited. Use original frozen signal_stop
routes with their original NavigationSignalHold parameters and identical cameras.

Start with a two-route infrastructure/data pilot: first eligible signal_stop
route in Town01 and Town03 after the whole-junction exclusions. Then expand
outcome-blind to the first two distinct eligible signal junctions in each town,
with three predeclared seeds per route. Never replace a failed episode with a
successful one. The original driving-internal validation and four static
component-validation junctions are excluded from planned and actual trajectories;
the approved global heldout_dev and Town05 remain unused. Town10HD has no
signal_stop location left after these reservations; disclose that absence.

New guard recomputes geometry-only route clearance against all reservations and
checks each actual20Hz camera packet. Fail closed on a partition violation.
Save exact frames, camera/world timestamps, actual expert actuation, upcoming
light actor ID/state and original planner/API distances. Actor states/geometry
are training/scoring-only. Visibility remains unknown; no new state-label release
is implied by collection. Preserve failures/collisions and all committed shards.

Before data enters an imitation supplement, require the original clean expert
outcome gate AND a continuous-coverage gate: same upcoming actor observed at
>=40m API/planner distance, >=1s of actual/teacher-go motion before braking,
braking onset after episode start, an actual stationary light-hazard stop,
>=1s stationary wait, same-actor green teacher-go release, >=3s recorded after
release and actual resumed motion. Never stitch separate episodes together.
Unknown/occluded/back-facing lamps do not become negative perception labels.

Report every episode's coverage/eligibility/censoring and all infractions. These
are TRAIN-data requirements, not revised driving success criteria. Keep previous
camera policy, original image cache and static component artifacts unchanged.
Freeze sources/guard/plan/seeds before outcomes, fresh directories, STOP-only
stopping, >=25GB free and one CARLA server. Pilot capped900s, expansion5400s per
invocation, each episode300s; failed infrastructure attempts remain documented.
