# Factual alternative-action pilot — before collection

Separate M5 TRAIN collection, never a canonical encounter or camera driving
claim. Select the first strict-TRAIN following-stopped-vehicle approach by route
ID that passes the existing1m geography guard and avoids all M3 prototype
junctions. That is Town01 route6/junction26. Freeze this choice before outcomes.
Keep driving/component validation, heldout_dev and Town05 out of this dataset.

One owned CARLA server, synchronous20Hz, original three512x288 calibrated views,
fixed clear weather, no background actors except a declared stationary Lincoln
15m along the route. Fixed geometry/seed,1s settling,4s common route-following
prefix at requested3m/s; then4s with target speed5m/s (go),0(brake), or0for2s
then5(wait/go). These are declared action interventions, not teacher targets.
Run all three options twice, preserving every attempt. Compare observable
prefixes and repeated factual branches before claiming deterministic replay.
No Traffic Manager playback or frozen nonreactive moving traffic: the only other
actor is a stationary braked vehicle in this initial pilot. This cannot establish
cross-traffic forecasting, rule understanding or final M5 planning improvement.

Record20Hz requested/executed controls and full world snapshots; at10Hz record
frame-exact lossless RGB, navigation, odometry and collection-only facts using
the existing atomic shard writer and next-tick actuator acknowledgment. Preserve
native CARLA recorder and every ego contact without speed/fault filtering. A
collision sensor triggers a privileged collection stop after first observed
contact; explicitly mark that intervention and censor any counterfactual claim
beyond it. Actual observations/actions remain factual prediction data; a safety
stop is never a clean or collision-free imitation outcome.

Compare go/brake alternatives at the same verified prefix. Use actual executed
ordered commands and observed outcomes, not hypothetical ground-truth labels
for actions never taken. Register each physical command sequence and timing;
kinematic action candidates still need measured controller calibration. No
model is trained on these examples before the raw pilot quality/censoring and
repeatability report. This pilot is480 prospective10Hz timestamps, not a claim
that sufficient world-model training data have already been collected. Keep25GB
disk reserve, source/request/route hashes, bounded1800s wall and STOP handling.
