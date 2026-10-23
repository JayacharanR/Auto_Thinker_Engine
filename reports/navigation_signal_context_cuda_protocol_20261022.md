# CUDA features and numeric-tree profile

The compiled CPU profile passed exact features/probabilities on 236,284 points
across the fixed 63 TRAIN images. Median feature-plus-head time was 49.1 ms,
versus 972.8 ms for the original path. Compilation took 0.31 s. This remains staged.

Separately test the CUDA feature kernel and existing numeric-tree export on the
same 63 TRAIN images and every frozen candidate coordinate. Sequential FP32 sums
and variances preserve operation order, FP fusion disabled, correctly rounded
square root. Original CPU integer averaging/normalization stays unchanged and
is included in timing along with host-to-device copies. Keep the FP16 feature
boundary. GPU trees operate in batches of 256 and final probabilities return
to CPU; include these operations in measured head time.

Require exact patch bytes and FP32/FP16 feature vectors, probability error ≤2e-13
and zero acceptance/phase changes. No thresholds, model, candidates or benchmark
change. Warm the native-sized feature kernel and one tree batch before timing,
report compilation/warm-up separately. Dynamic candidate count must not create
per-image compiled kernels. Alternate original/staged order by image index.

Output `outputs/navigation_signal_context_cuda_profile_20261022`; 600 s / 20 MB
cap, original storage guardian, two CPU threads. Host CUDA preflight required.
Archive/hash all used sources; no fit, component VAL, native driving or deployment.
Concurrent CPU audit disclosed. Decode, CNN proposer, candidate generation, IPC,
tracking, three-camera scheduling, policy and simulator are excluded. No full
runtime claim. Every receipt/failure retained; STOP respected.
