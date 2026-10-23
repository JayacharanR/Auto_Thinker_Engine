# Constant-velocity perception check — before paired-world results

Freeze the existing COCO SSDLite320 model and its previously chosen0.35 threshold,
three-view ground-ray approximation,2m cross-view fusion and odometry tracker.
Do not tune these on these results. Evaluate every preserved1Hz snapshot in the
seven expert-only native setup probes, with exact matching world frame and the
declared unchanged camera calibration. No student controls, training updates,
new driving or Town05 frames enter this diagnostic.

Report raw detections, rejected ground contacts, localization error versus
nearby scripted actor origins, unmatched actors/detections and runtime. A nearby
actor is not necessarily visible; this is geometric tracking availability, not
visibility-certified object-detection recall or AP. Occluded/truncated/behind
actors remain explicit. Actor origins differ from the visible bumper/contact
point; do not label that diagnostic as accurate depth ground truth.

Detector input is only SensorPacket RGB/calibration/navigation/ego odometry.
World actors are read separately after inference by this scoring script. No
ground-truth boxes/positions/velocities enter the detector or tracker. Associate
diagnostically by minimum total origin distance with a fixed10m gate, report
unmatched actors explicitly, and retain per-frame predictions before matching.
This adds no local CARLA training labels to the M4 encoder study; it is a separate
M5 perception diagnostic with external COCO labels disclosed. Passing runtime
alone does not qualify CV planning or prove absence of traffic when no actor is
detected. Training a later detector needs its own declared label accounting.
