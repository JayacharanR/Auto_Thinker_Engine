# Causal signal replay — prospective TRAIN diagnostic

Freeze before inference. Use the selected ROI-adapter500/expanded-state bundle
without fitting, calibration, threshold search or driving integration. This is
an offline component diagnostic, not a benchmark change or a policy promotion.

Start from the six previously audited DAgger episodes and their same-frame
projection audit. Require their original provenance, global TRAIN partition and
internal driving-TRAIN guard. Additionally exclude every junction assigned to
static component validation: Town01/87, Town03/655, Town04/335, Town10HD/468.
Thus retain routes86,101,123,143 (1,238 timestamps); preserve but exclude111/133
(491 timestamps). Audit teacher stop/release events on all four retained routes.
Run RGB inference on both complete signal-bearing episodes101/123 (518
timestamps,1,554 native images). Selection uses pre-existing label availability,
never detector output. Do not access heldout_dev or Town05. These familiar,
assisted TRAIN rollouts do not measure independent encounters/generalization.

Inference receives SensorPacket only. Housing detector confidence remains.5;
16x32 bilinear RGB crops and exported logistic coefficients retain the previous
state-head behavior. No annotated crop or actor identity enters perception.
Add a fixed causal, per-camera angular tracker: approved ego heading compensates
yaw; geometric one-to-one greedy association ignores predicted color. Gates:
center1–4 degrees, based on twice housing angular size; size ratio.4–2.5.
Exponential probability smoothing time constant.3s; at least two consecutive
observations and maximum probability.6 to report a state. A gap over.2s gives
Unknown, over.5s expires the identity. No cross-camera physical-ID claim,
triangulated depth, stop-line estimate, applicability confirmation or control.

Score raw argmax and tracked argmax on the SAME current-frame, one-to-one
IoU>=.3 projected applicable-housing matches (GT clipped to the image). Separately count confident tracked
states/unknowns, consecutive-match identity changes, detection gaps and state
transition delays. Projection overlap is NOT certified visibility/state truth;
these labels remain diagnostics and are not released for fitting. Report all
states, failures and denominators. Never choose a track using GT at runtime.
Any GT association is scoring-only; ego-route applicability remains unresolved.

For every pre-existing teacher zero-speed run report duration, causes, start
censoring, release availability and applicable-light evidence at start/release.
A teacher halt is not automatically caused by a light, and assisted actual speed
is not camera driving. Check whether complete history fixes missing evidence;
do not assume longer memory helps. Fixed parameters get one replay only.

Fresh output, archived sources/input hashes, STOP checks and600s inference cap.
No new simulator, collection, weight update, long training or job-script change.
Run focused CPU tests before replay; preserve parent policy/checkpoints/cache.
