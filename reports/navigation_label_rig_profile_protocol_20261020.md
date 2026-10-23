# TRAIN-only aligned-label collection profile — before native data

Profile the training data engine without changing the policy contract, benchmark,
reward, model selection or any existing fit. The forward policy still receives
only the original three RGB cameras, declared map/ego values and its separately
approved own-actuator feedback when explicitly versioned. Simulator label cameras
are TRAIN supervision/audit only; they are never supplied to this profile's route
controller or any policy worker. No new supervised fit is launched by this pilot.

Use the existing strict-TRAIN route6/Town01 junction26 diagnostic location, which
is excluded from the camera rule test locations. One ego and a stationary lead
25m along the route, clear noon, no Traffic Manager background. Fixed numerical
route tracker requests2m/s, with separately logged post-contact stopping only.
This is instrumentation, not teacher imitation or a safe-driving demonstration.

Run one RGB-only baseline then one nine-camera rig (RGB, instance segmentation,
depth at every front/left/right pose), same fresh-world setup. Each mode uses
20settling ticks,10camera warm-up ticks and120measured20Hz physics ticks. Store
all three views every second tick:60timestamps/mode at10Hz. The first baseline
order, warmups and storage recipe are fixed before any frame exists. Do not
repeat or select a favorable run. Retain both modes even if a slower outcome
disagrees with expectations; fixed order is a declared throughput confound.

Preserve512×288 raw RGB and BGRA label bytes, frame/timestamp and image transforms.
Require all nine streams at the requested frame, common timestamps and per-view
pose alignment within1mm/.01degrees. Small20timestamp exclusive shards close,
decompress and verify every original pixel hash before atomic publication.
Keep any truncated uncommitted part on failure. No labels are put into a legacy
policy packet. Store the raw two instance channels losslessly; do not assume
render-component identifiers equal CARLA actor IDs.

CARLA0.9.15's installed CityObjectLabel enum and pinned C++ source use Car14,
TrafficLight7, Roads1. The older class table visible on the versioned sensor-doc
page differs. Check the installed enum before simulation and preserve it. Raw
instance red channel supplies class, green/blue supply instance identity; raw
depth decodes R+256G+65536B, scaled by1000/(2^24−1). Independently preserve raw
bytes, semantic counts and car-pixel median depths for geometry inspection.
These checks do not prove lane applicability, stop-line localization or actor
identity; visible car pixels do not by themselves establish full visibility.

The profile waits until the original unstarted camera integration owner is
terminal and its actual process/server absent. A failed infrastructure predecessor
remains failed; it is never rerun by this profile. A stopped predecessor halts the
profile. Single native owner, global navigation lock, same14GBadditional/10GBfree
storage guardian. Total native wall cap600s including startup; own STOP lifecycle,
no kill9, no job-shell editing, no duplicate server. Retain all recorder contacts
and actual world/control traces. Maximum350MBartifact budget, no deletion.

Report totalframes/s, p95tick time including collection/storage, native contact
counts and complete raw roundtrips for both modes. Static/slow single-location
compression and throughput are not a general traffic estimate. Larger collection
must separately register diverse scenarios, source/geography split, label budgets
and measured capacity before fitting. The current M4 matrix stays unchanged.

Primary references: [CARLA0.9.15 sensor formats](https://carla.readthedocs.io/en/0.9.15/ref_sensors/),
[pinned0.9.15 CityObjectLabel](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/rpc/ObjectLabel.h),
[instance-camera implementation](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/InstanceSegmentationCamera.cpp).
