# Town05 registration v2 — geometry coverage anchors

Before any Town05 policy outcome, replace the inefficient endpoint-only route
generator with geometry-constrained mission waypoints. The unfrozen v1 attempt
is preserved with its two partial routes and STOP file. It tried thousands of
random pairs without a qualifying1–3km route through the next mandatory stop;
no benchmark was frozen and no driver was tested. This is a route-generation
implementation correction, preserving the approved five-location/12-route/
three-seed coverage and all metrics.

Keep all rules in navigation_m6_registration_protocol_20261018.md, including the
explicit-XML/runtime-stop correction, timing disclosure and every scoring actor.
For each required mapped stop location, choose the lexicographically first
inventory movement whose independently replanned serialized start/end pair
passes the pinned affected-lane opportunity check. This geometric anchor is
chosen once, without outcomes. Stop89 has a27m local approach: the standalone
dev-movement30m validity filter is not a final-route exclusion. Keep that sign
and count its infractions normally.

For each of the first ten routes, interpolate a mission through seeded start,
the required movement's start/end, and seeded destination. Last two have seeded
start/destination only. Use the pinned harness's same per-segment interpolation,
verify the FULL route length1–3km and all directional opportunities after
serialization, and keep the original held-red setup requirement. Route0/1 cover
first sorted stop actor,2/3 next, and so on. Seed remains20261018. No performance
outcome, model, image, actor deletion or scoring change enters selection.

Same900s cap/STOP and fresh directory configs/routes/town05_final_v2. Archive
all sources, exact input hashes, anchors and selected route geometry. Freeze
only after12 routes, all five explicit mapped locations and >=30 planned stop/
red opportunities. Actual runtime encounter counts remain to be audited at M6.
