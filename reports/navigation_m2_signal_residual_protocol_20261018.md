# Frozen-parent localized signal experiment — prospective

The completed causal-history fit failed every checkpoint. Exact sampler replay
finds47.35% stop labels, so a shortage of stop examples is not established.
Keep its failed results and the original selected80bee4 parent. No longer repeat.

Compare two matched small learned stop-residual arms: actual frozen RGB signal
facts versus the same architecture with only its null token. Freeze ALL parent
parameters, path, cruising speed, comfort brake and auxiliary perception heads.
Initialize the nine-output residual to zero, exactly preserving parent behavior.
Learn changes to stop logits only, with object attention and a64-unit causal GRU;
parent192-unit recurrent features plus declared route/ego inputs condition it.
There is no if-red-then-stop runtime rule or oracle applicability association.

Use the selected frozen housing detector/RGB state bundle, native512x288 ALL
three views read from lossless shards. Infer every fourth10Hz frame (2.5Hz),
hold ONLY past observations with explicit0/.1/.2/.3s age. Up to100 outputs/view,
all retained plus a null token. Token17: normalized box4, confidence1, RGB state
probabilities3, ego-relative bearing sin/cos2, elevation/width/height3, view one-hot3,
age1. No actor, map-light location, GT facing, stop-line or teacher fact is input.
The learned attention must resolve applicability; it remains unverified until
driving/counterfactual evidence. Cache and deployment use the identical packer.
No heuristic tracker identity, EMA or threshold search in this experiment.

Use original M1 cache/split, excluding the three newly reserved component TRAIN
locations, plus accepted varied-wait TRAIN supplement only. Original13 validation
episodes and1312 scored label positions unchanged. Cache full-episode parent
features and predictions without gradients; never use validation targets to
construct features. New supplement has imitation stop/speed targets only, no
unverified perception supervision. Preserve original component development
history: those four locations are not an untouched test of the inherited parent.

Each arm: seed42, identical1000 AdamW updates at1e-3, grad cap1, FP32,
whole-episode causal histories, burn-in8, half original family-balanced episodes
and half clean approaches. BCE current stop weight2 and masked future stop1;
keep original parent stop-probability distillation weight.1 on original episodes,
none on supplement. No speed/path loss: these parameters cannot change. Same
sampler draws/initial weights/budget for both arms; no extension of losing arms.
Evaluate exactly0/250/500/750/1000 on unchanged causal validation positions.
Count all evaluations and frames. Selector: required-stop recall>=parent,
each family's stop recall>=parent-.02 (skip zero-stop families), false holds
<=parent+.02, go-speed MAE<=parent+.2m/s and strictly lower existing balanced
stop/false-hold score. Keep parent0 unless all protections pass. Apply the same
selector to both arms. No safety/JEPA claim from offline metrics.

Cache cap3600s; fit cap1800s per arm, full optimizer/RNG/sampler checkpoint,
STOP and25GB disk reserve. Archive all code/protocol/artifact/manifest hashes.
Measure actual detector/cache throughput before estimating runtime. If a fit
passes, run the unchanged fixed TRAIN pilot with overridesOFF and the1.23km
route; preserve every failure/collision. Do not touch heldout_dev/Town05 to tune.
M3 paired counterfactuals and new heldout_dev expert calibration remain required.

M4 fairness: this RGB detector has external COCO initialization and642 TRAIN
housing labels. Disclose/charge that shared perception cost, match it across
encoder arms and ablate it. Low-label comparisons must reinitialize driving
heads and count all labels; inheriting this full-label driver would leak labels.
