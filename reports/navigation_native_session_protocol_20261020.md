# Native callback lifecycle check — before execution

The saved client1611008core shows TimeoutException escaping
LightManager::QueryLightsStateToServer → its SetEpisode callback → Episode::Listen
on a native streaming thread, followed by std::terminate. The main Python thread
is sleeping. This directly identifies the aborting client path; it does not prove
the cause of every engine signal11or establish that the proposed fix works.
Read-only GDBtrace: outputs/navigation_native_lifecycle_diagnosis_20261020.
The [pinned0.9.15 LightManager source](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/client/LightManager.cpp)
registers a blocking light-state query callback; the
[Episode listener](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/client/detail/Episode.cpp)
dispatches it from the stream. No native binary has been patched or recompiled.

Test a new private ClientLease which releases its only Client reference while
the owned engine is still alive, before CarlaServer.close. Episode functions
return plain data only: no world/actor/manager handle may escape. Run exactly two
braked-camera instrument cycles at existing strict TRAINroute6: new engine and
new scoped episode per cycle,3RGBcameras,20settlingticks/10camera-warmticks/20measured
ticks, no motion policy, no other vehicle, no native case/benchmark replay.
Store pixel hashes, camera frame IDs, actual ego poses and teardown receipts;
no RGBdataset or model supervision. All instrument data is excluded from fitting.
No exposure at heldout/VAL/M3/Town05. Verify actual geography with the existing
role guard. All captured contacts (if any) are retained as instrument evidence.

Ownership/order unit assertions cannot substitute this native test. Publish
both runs even on failure; no repeat/replacement. Require no client abort and
each server to terminate under its normal owner. Record an abnormal exit as
infrastructure failure rather than hiding it. No new data collection/ensemble
fit until the lifecycle evidence is reviewed. Later native episodes still need
an outer process supervisor for faults that kill Python from C++.

One global owner,300s/20MBcap, same original14GBnet/10GBminimum guardian.
STOPonly via the owned root. No policy/reward/scoring/input/benchmark changes,
model update, actor-visibility claim or driver promotion. Previous ten-attempt
pilot remains8verified/2unknown with all source owners and whole audit FAILED.
