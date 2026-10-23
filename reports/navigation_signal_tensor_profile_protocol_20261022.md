# Staged numeric-tree export: TRAIN parity and runtime profile

Independent engineering profile, no learnedmodel or evaluation changes. The full
plateau TRAIN audit remains CPU/scikit-learn with its registered source untouched.
Profile a staged tensor implementation of the existing supported finalRGB87head
SHAed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191.
No inference module is connected to a deployed driving policy.

Export every node feature, FP64threshold, branch and FP64leaf, plus baseline logits.
All100×4trees run the same numeric comparisons. Missing/nonfinite inputs are rejected;
realRGBfeatures are finite. Leaf self-loops preserve variable-depth paths. Float64
summation and softmax can differ by rounding; require maxabsoluteprobabilitydifference
≤2e-13 and zero acceptance(.5foreground)/phaseargmaxchanges. Synthetic tests cover
actual learned trees and adjacentFP32values around split boundaries, not just mirrors
of the tensor implementation.

Use first4,096original TRAINrows from the hash-verified RGBfeaturecache; noVALinputs.
One scikit reference over allrows, CPUtensorparity and GPUsinglebatch256parity over
allrows. CPUandGPUtiming uses the same first256rows,2warmups/10repeats; synchronize
GPUbefore/after each measurement. Report device/versions/memory and concurrentCPU
audit. This is kernel timing with inputs already resident, not fullthreecamera
latency or a deployment claim. Save all4,096reference/CPU/GPUprobabilities, feature
row identities/exportarrays/sourcehashes. Do not select a scientific model based
on timings. No parityclaim beyond tested inputs.

Output outputs/navigation_signal_tensor_profile_20261022;300s/30MBcap, original
storageguardian. STOP/retainfailures. Existingnavigation sourcebytes remainpinned;
only stagedpatches/newprofile/tests are written. No training,nativejob,benchmark,
inputcontract, M4primary or candidateclassification change.
