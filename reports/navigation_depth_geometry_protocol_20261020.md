# Native depth geometry check — before execution

Confirm the installed camera's metric-depth interpretation using a known parked
vehicle before any learned geometry fit. This is instrumentation at the existing
strict TRAIN Town01 route6 location, not an evaluation suite, a driver test or
new model supervision. Exclude this diagnostic from all later model fitting.

Six fixed placements: front/left/right camera, bearing0° and40° relative to that
camera, optical-axis displacement25m. Keep the same ego parked throughout, no
background TrafficManager. Target physics disabled; target identity is isolated
by comparing instance masks before/after placement, not by equating Unreal and
CARLA RPC IDs. Require one new visible vehicle instance and≥100pixels. Retain
baseline/placed raw RGB, instance and depth, physical frame/timestamp, native
sensor poses, target pose and bounding-box transform/extents. Record the episode
with CARLA's recorder and report all contacts.

Lift every isolated target pixel under BOTH hypotheses: axial z-buffer depth and
radial range, using half-pixel centres and the measured sensor transform. Compare
the points with the target's actual oriented3D bounding box, allowing0.30m mesh/
box tolerance fixed before execution. This tests depth convention, not surface
accuracy or actor-centre prediction. Pixel observations are highly correlated.
The axial convention check passes only if≥95%of points in each of six placements
are inside the padded box AND≥80%of points at each40°placement lie outside the
box under the radial hypothesis. Publish every placement; no replacements or
threshold sweep. A failed placement remains failed, with no automatic rerun.

The documented channel decode is from the [official0.9.15 sensor reference](https://carla.readthedocs.io/en/0.9.15/ref_sensors/#depth-camera).
The CARLA contributor's [depth interpretation explanation](https://github.com/carla-simulator/carla/issues/56#issuecomment-347683108)
supports the axial hypothesis; the native check verifies this installation rather
than treating a historical comment as current physical evidence.

One owned simulator/global lock,300swhole execution cap,100MBartifact cap,
existing14GBadditional/10GBminimum storage guardian. Stop via this owned root's
STOPfile. Snapshot code/config/geometry before launching. No controller, input
contract, reward, benchmark or qualification gate changes. An instrument check
cannot qualify perception, planning or M6 driving.
