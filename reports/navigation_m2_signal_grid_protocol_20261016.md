# Native RGB signal-grid component — before fitting

Use only the new audited static corpus_v2 after its unchanged coverage/integrity
gate passes. It contains51 poses,612 native RGB images,63 supported head/view
pairs (62 also support yellow) over16 responsive junctions in5 training towns.
Original v1 corpus remains failed, unused for fitting. Original camera driving
policy, dataset/cache, validation locations, controller and benchmark unchanged.

Architecture: six3x3 convolution layers with12/24/32/32/32/32 channels and SiLU;
only layer2 strides2; a1x1 four-class head predicts background/red/yellow/green.
Receptive field21 native pixels. No global pooling, absolute position, navigation,
map/light actor, paired image or override input. Runtime can scan each full RGB
image; crop boxes are training supervision only. This exports a perception
component, never a driving-policy checkpoint.

Labels: one phase label at each supported projected housing's visible-crop center.
All other projected lamp regions (all actors, including unknown/back-facing heads)
are ignored, with a2-cell margin. Remaining image cells are approximate background.
Native128x128 training crops, half positive class-balanced anchors and half random
TRAIN-image crops. No actor/image coordinates enter the network itself.

Internal component split: seed20261016, first seeded ceil20% of17 sorted whole
junctions.13 training/4 validation; all poses, cameras and phases stay together.
Val: Town01_87,Town03_655,Town04_335,Town10HD_468.468 train/144 validation images;
179 training/72 validation unique grid centers. Training centers:90 red,45 green,
44 yellow; validation36 red,18 green,18 yellow. These are development validation
locations inside the approved TRAIN partition. Earlier annotation-method work
inspected some of those TRAIN locations; this is not an untouched test or a claim
of Town10HD zero-shot generalization. Original internal-driving validation,
heldout_dev and Town05 are absent from all phase data.

One seed,1,000 updates, batch16, AdamW LR1e-3, weight decay1e-4, BF16 CUDA training
(no scaler), FP32 evaluation, gradient norm cap5 and1,200s wall cap. Loss/selection:
mean foreground cross-entropy +0.2*mean background cross-entropy; ignore255 labels.
Evaluate initial random model and every200 updates. Save optimizer and complete
Python/NumPy/Torch/CUDA/sampler RNG, source/data provenance, latest and selected
component. STOP and bounded resume supported. No hyperparameter search.

Report strict center class accuracy,72 correlated supported centers, per-class
counts, correct state within4 native pixels at probability>=.5, and active
background-cell fractions at.5/.9. These are sparse annotation diagnostics, not
full detection AP, lane applicability, rule behavior or driving reliability.
Do not promote the camera student from these numbers. If the component fails,
retain the negative result; no longer repeat is automatically authorized.
