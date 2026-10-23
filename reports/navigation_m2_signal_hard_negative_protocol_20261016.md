# TRAIN-only hard negatives and protected classifier refinement

Freeze before mining. Parent is selected step50 from
outputs/navigation_m2_signal_detector_20261016, SHA256
8df27ee218056e392cc5ddb0f50fef9263a1e8b81517b1d5ecbeba8038a6e49b.
State classifier remains robust_rgb_linear.npz from the crop-alignment experiment.
Parent FP32 has30/72 detections,24/72 detected-and-correct states and681 approximate
background candidates across144 component-development images. Do not use the
illustrated validation mistakes as training examples.

Mine only the117 red-phase images inside the13 component-TRAIN groups of the
audited static corpus (no internal-driving validation, heldout_dev or Town05).
Run frozen parent FP32 with1,000 proposals, confidence.5 and unchanged512x288 RGB.
Take at most2 candidates/image, descending confidence, pairwise IoU<=.3. Require
intersection/min(area candidate,area ignored)<=.05 against EVERY padded housing
region. Round outward only for pixel comparisons. On the same pose/view, require
both red-vs-green and red-vs-yellow mean absolute crop change<=5 and<=2*red-repeat
MAE+1. Keep all geometry and photometric statistics. These are conservative
approximate negatives, not exhaustive human ground truth.

Data gate before fitting:>=20 distinct candidate boxes,>=6 TRAIN junctions,
>=3 TRAIN towns, every source/role/hash/geometric/phase check valid. No validation
image may enter the manifest or mining inference. Every accepted box transfers to
the same four frozen phase images; these are repeated views, not new independent
negative objects. View a fixed seeded contact sheet and retain examples/unknowns.
If the gate fails, do not fit or lower it retroactively. Mining cap300s, STOP.

If data gate passes, refine only the2050 binary ROI classification parameters.
Freeze RPN, box regression, backbone/FPN and ROI features/buffers. Preserve parent
weights and the same RGB-only inference architecture. On TRAIN images with mined
negatives, append4 copies of each supported GT ROI and each mined negative to
the standard sampled ROIs. Positives have class1 and zero box-regression target
(proposal equals GT); negatives have class0. Additional training regions resize
with the image. No replay injection occurs during validation or inference.

One bounded fit:200 updates,batch2,same positive/uniform image sampler,
FP32 AdamW5e-4,weight decay1e-4,grad cap5,seed20261016,900s cumulative cap.
Evaluate0/50/100/200 on the unchanged144 images,72 labels, confidence.5 and fixed
seed validation loss. Protect parent0: a candidate is eligible only if IoU.3
recall>=30, joint correct-state>=24 and approximate-background candidates<=681;
among eligible candidates minimize the pre-existing detector loss. Keep parent
if none improves. This is a new predeclared component-selection guard, not a
change to the driving benchmark, reward, existing result or detector thresholds.
Report every checkpoint, one-to-one audit and all regressions. No automatic
longer repeat or driving integration. Full optimizer/RNG/source/input provenance.

Count geometry and mined-background supervision in later label budgets. Four
already-inspected validation junctions remain development data, not an untouched
test. A static perception gain does not establish lane applicability, memory,
rule understanding or closed-loop safety. No held-out suite is used here.
