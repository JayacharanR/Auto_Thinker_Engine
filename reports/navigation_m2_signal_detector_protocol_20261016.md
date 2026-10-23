# Small-anchor housing detector — frozen before fitting

M2 component experiment only, using the completed static corpus_v2. Keep its
13/4 whole-junction split,468/144 native512x288 RGB images and179/72 supported
housing/phase labels. No new collection, driving policy, reward, sensor, held-out
partition or evaluation-benchmark change. Development annotations were inspected
earlier; this is neither untouched testing nor a fair encoder comparison.

Hypothesis: a two-stage detector can separate housing localization from the
normalized crop-state task, which scored61/72 with oracle crops. Generic COCO
anchors do not match the smallest traffic-light housings. Test domain adaptation
of the existing declared COCO supervised detector with smaller vertical anchors.

Initialize official torchvision FasterRCNN-ResNet50-FPN-v2 strictly from the
previous hash-verified COCO file. Binary background/housing predictor copies
COCO classes0/10 and their box regressors. Anchor square-root areas8/16/32/64/128
on successive FPN levels; height/width ratios1/2/4 (three per level as before).
Freeze backbone/FPN and ROI feature head, including normalization buffers.
Train RPN and binary ROI predictor only. ROI sampling64/image, positive fraction.25;
RPN and NMS defaults otherwise unchanged. Confidence.5 and detection cap100/image.
Default internal800/1333 resize interpolates unchanged camera pixels.

Supported native housing boxes are positives. Every projected housing, including
unknown/back-facing/unresponsive heads, gets the original four-native-pixel ignore
margin. Preserve positive assignments; exclude negative anchors/proposals when
intersection / min(candidate area,ignored area)>=.3. Apply ignores in BOTH RPN
and ROI sampling, after transforming them with the image. A projected unresponsive
lamp is not certified background. Remaining background is approximate geometry
supervision, not exhaustive human annotation.

One seed20261016,200 updates, two images/update (one uniform supported-positive
TRAIN image plus one uniform TRAIN image), FP32 AdamW LR5e-4/weight decay1e-4,
gradient norm cap5. No label/class or horizontal-flip augmentation. Whole experiment
wall cap900s; STOP between batches. Preserve optimizer/Python/NumPy/Torch/CUDA/
sampler RNG, source hashes, request, latest and selected full component checkpoints.
Resume requires identical worker/data/request hashes and remaining cumulative cap.

Evaluate0/50/100/200. Selection: mean sum of four standard detector losses over
all144 validation images, with a fixed RNG seed20261017 restored afterwards.
Frozen normalization remains in eval even when detector training samplers are used
to compute validation loss. Selection is independent of displayed recall/state
counts. Compare adapted step0 with trained candidates; preserve the earlier
91-class frozen baseline separately because class/anchor changes affect step0.

For every validation inference, report supported-housing recall at IoU.3/.5,
detected-and-correct state using the SAME saved linear crop classifier on predicted
boxes, and counts by native height/town. Also count predictions outside every
projected/ignore region as approximate-background candidates; sparse labels do not
support full AP or an exact false-positive rate. No GT box, lamp state, map geometry
or phase pair enters inference. Report the model's three-view latency excluding
the first cold batch; it is likely too slow for10Hz without later optimization.

Do not integrate solely because recall rises: check background behavior, crop-state
alignment, applicability and latency. Save the negative result if this fails.
References: [official finetuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html),
[official model](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html).
Installed torchvision0.26.0 source signatures were checked directly; the website
tracks a newer release.
