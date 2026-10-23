# Signal runtime preparation — staged, not deployed

The current detector remains unqualified:55/72component correct detections and
127known-background detections, but yellow12/18versus the immutable parent15/18.
The full independent-plateau TRAIN classifier audit is still running on its
registered CPU implementation. Nothing in this staging changes that experiment.

**Point interface.** Existing policy/tracker code assumes a housing bounding box
and at most300detections. A bulb point does not provide housing width, height,
depth, stop-line location or route applicability. The staged16-value token carries
only calibrated unit-ray direction3, camera origin3, unconditionalBG/red/yellow/
green probabilities4, view identity3, normalized nativeXY2 and a null indicator1.
All observations remain; padding is dynamic and501detections are tested. No top-K,
invented box dimensions, depth or map-based applicability. Only exactcurrent-frame
observations are accepted; stale/futureframes and extra actor fields are rejected.
Future policy memory must be trained explicitly; old rays are not passed off as
current geometry. Three tests verify reprojection for all camera rotations and
pitched/rolled cameras, padding/null behavior and invalid/stale fields.
Source: `patches/navigation_signal_point_tokens_candidate.py`.

**Exact-model numeric export.** The staged tensor representation retains every
existing tree feature, FP64threshold, branch, leaf and baseline. It does not fit
or select another model. Tests against a separately trained real classifier probe
adjacentFP32values around split thresholds; CPU output agreement≤2e-13.
Actual savedhead SHAed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191:
first4,096TRAINfeatures tested, CPUmaxprobabilitydifference5.55e-16,GPU6.66e-16,
zero foreground-acceptance or phaseargmaxchanges. This is tested-input parity,
not bitwise equality or validation of every possible input.

| Same256row kernel,2warmups/10repeats | Median |
|---|---:|
| Original scikit-learn CPU,2threads | 3.519ms |
| Staged tensor CPU,2threads | 8.483ms |
| Staged tensor GPU,inputs resident | 0.815ms |

GPUkernel is4.32×faster than originalCPUfor this batch. CPUtensor is slower and
is not an improvement. GPUpeak allocated tensor memory4.77MB; RTX5070TiLaptop,
Torch2.11.0+cu128/CUDA12.8. A separateCPUaudit ran concurrently. This excludesRGB
feature generation, CPU/GPUtransfers, the CNNproposer, three-camera scheduling and
control. It is NOT an end-to-end runtime speedup, deployment claim or reason to
change the active scientific audit. Source:
`patches/navigation_signal_tree_tensor_candidate.py`.

**Retained failure.** The first profile stopped after0.64s when sandboxCUDA was
unavailable; noGPUexecution. HostCUDApreflight then succeeded. Separatehostoutput
completed2.14s with the original300sabsolute deadline preserved; originalfailed
output retained. CPU reference calculations were recomputed, with no scientific
fit or candidate outcome retry. Both outputs keep source/recipe hashes; the
successful profile retains export arrays and all4,096reference/CPU/GPUprobabilities.

Evidence: outputs/navigation_signal_tensor_profile_host_20261022/assessment.json;
failed outputs/navigation_signal_tensor_profile_20261022/progress.json;
reports/navigation_signal_tensor_profile_protocol_20261022.md. These staged modules
have no deployed-policy imports. Applying them requires completed perception gates,
full pipeline numerical/runtime verification and learned route/stop-memory training.
