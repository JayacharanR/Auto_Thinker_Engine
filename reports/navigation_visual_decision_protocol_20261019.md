# Visual decision feasibility test — before any update

The first spatial ResNet seed42/full-label model scores96.79% required-stop
recall/4.47% false holds on original validation but remains stationary in the
first native pilot cases. Preserve the entire six-case run, including all failures.
A post-hoc artificial ego-speed probe changes false holds4.47→83.51% at zero
speed and stop recall96.79→0% at5m/s. This demonstrates strong sensitivity,
not a physical counterfactual or proof that velocity alone causes every failure.
The all66-fit M4 matrix has NOT started; do not spend its budget on this known
feasibility failure. Keep the first M4 protocol, cache and model unchanged.

Test one new M2 architecture: spatial-visual-decision-v2. The entire learned
visual/path/speed/stop recurrent model receives zero in its ego-speed coordinate.
It retains three RGB views, photometric fields, past video/recurrent memory,
navigation, heading and yaw rate. The physical trajectory controller receives
the original measured actual speed, exactly as before. No artificial speed is
reported in SensorPacket/telemetry, no stop/go rule is added and no override
forces movement. Explicitly disclose this omission in the policy metadata.

Fresh ResNet18 downstream seed42, same common initialization, same fixed59 TRAIN
episodes/16639 labels, same cache/labels and original13 validation episodes/1312
scored positions. Exactly2000 updates, identical sampler draws, AdamW3e-4/grad1,
BF16 and full causal history/burnin8. Select with the unchanged score at exactly
0/500/1000/1500/2000. No inherited driving/perception head or auxiliary labels.
This ablation isolates the neural speed input against the original spatial fit;
it is not an encoder comparison or a claim of learned traffic rules.

After the fit, run the unchanged six-case native TRAIN engineering pilot with
overrides off. Keep every outcome, collision and failure. Run the original
1233m integration route if the model demonstrates movement/completion in the
pilot; a model that remains stationary does not warrant a duplicate long timeout.
M2–M6 remain open; no heldout_dev/Town05 evaluation, route/scoring/reward change,
or teacher-label modification. If feasible, prospectively register the corrected
common architecture for every M4 arm before any further matrix fitting.
