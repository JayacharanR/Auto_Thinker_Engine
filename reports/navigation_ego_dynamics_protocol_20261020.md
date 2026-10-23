# Ego actuator model — before fitting

A shared camera-planning candidate must map to an ordered executable80control
sequence. Fit one small kinematic ego-actuation baseline on the TRAIN episodes
of the registered69source dynamics manifest only. Validation episodes remain
scoring-only. Every input action is the uniquely pose-aligned native command
acknowledged at the next20Hz physics tick, not the expert proposal.

Use planar position/yaw/speed from TRAIN world traces as supervised physics
measurements; never pass simulator poses, actors or future states to deployment.
The exported model contains only fitted constant parameters. Fit acceleration
as a bounded linear throttle/brake/drag/intercept model and steering as a bicycle
with fixed2.8m wheelbase and a fitted maximum angle in[.3,1.3]rad. No validation
search or driving-outcome selection. Use least squares once, with both endpoint
speeds>.5m/s, exclude the first1s and.2s before/.5s after observed contacts.
These are physical-identification filters, not removal of collisions from any
headline metric; every source/contact remains retained and forecast error on
all eligible windows is separately disclosed.

Report one-step errors and factual4second open-loop trajectories on all windows
with80known controls, by TRAIN versus the original dynamics-validation split.
Initialization uses only the current pose/odometry. Execute the given ordered
controls through the exported model; future ground truth is scoring-only. Do not
claim deterministic replay, calibrated collision risk or accurate alternative
actions merely from this fitted model. The JEPA/RSSM/CV comparison shares its
candidate-control construction and conventional tracker. Subsequent simulator
branches must establish actual control/trajectory error, using the registered
replay agreement checks and live reactive actors.
