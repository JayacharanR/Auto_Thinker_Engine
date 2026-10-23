# Spatial representation study and new camera baseline — prospective

The whole-history and both frozen-parent residual fits failed all protections.
Keep them and80bee4 selected parent. Stop-gap pilot changes the actual stationary
reference distance5.7→11.1m but adds no camera frustum coverage at Town03/134;
front housings shrink7.6→6.2px. Do not expand that teacher change or train from
its diagnostic labels. Standard teacher6m and every benchmark remain unchanged.

Proceed with the approved stronger encoder comparison as a new M2 driving
countermeasure and M4 study; M2 is not marked complete or skipped. New fresh
heads/architecture constitute a combined countermeasure versus historical v2;
only encoder comparisons WITHIN this experiment isolate encoder choice.

Data: original47 strict TRAIN episodes plus12 accepted varied-wait episodes,
16,639 labeled timestamps. Original13 internal-validation episodes/2822 frames
and1312 scored label positions remain unchanged. Preserve component reserves,
heldout_dev17 locations and Town05. In-domain JEPA uses only these TRAIN video
inputs, never internal validation/heldout_dev/Town05, with no action labels.

Five encoders: scratch trainable CNN; frozen ImageNet ResNet18; frozen DINOv2-S;
unchanged project comma2k19 JEPA; CARLA-continued JEPA. Input all three256x144
context views,8 past frames (episode-start repetition). Retain4x4 spatial tokens
per view, temporally pool within the past clip, then shared192-unit causal GRU.
2D backbones average their eight per-frame grids; JEPA temporally attends before
pooling. DINO pads10 pixels on bottom/right with normalized mean color to fit
14px patches; it does not crop/distort the physical camera content. Explicit
rectangular T/H/W positional interpolation for project JEPA.

Common high-resolution component: deterministic continuous RGB/opponent max
and RGB mean fields on a9x16 grid of32x32 cells from ALL native512x288 cameras.
Channels:R,G,B,R-G,G-R,R-B,G-B,min(R,G)-B maxima, plus three RGB means.
It preserves tiny saturated pixels that averaging can erase. No thresholds,
light labels, GT boxes/map-light inputs or if-red-stop rule. A learned route/ego
conditioned cross-attention reads visual/context/color tokens and predicts
separate path correction, cruising speed and stop intent with causal memory.
This is feature engineering, not evidence the model learned a rule. No existing
642-label detector/state model or full-label driver head is inherited.

Nested complete-episode annotation budgets5/10/25/100%, one seeded order42,
actual label counts/fractions and family coverage reported. No partial episodes,
resplitting, search for a favorable subset or labels from outside the selected
episodes (including auxiliary perception labels). Common photometric fields
consume zero annotation labels. External ImageNet supervised pretraining,
DINO's data scale and all pretraining/compute differences are disclosed.

Each arm/fraction: three seeds42/123/456, freshly initialized downstream heads,
2000 updates, AdamW3e-4, grad cap1, BF16 CUDA/no scaler, original-family-balanced
whole-episode sampling from ONLY the selected labeled episodes, burn-in8.
Full causal GRU gradients; scratch visual chunks recomputed to fit memory.
Same sampler draws/update/label exposure budgets per seed across encoders.
Evaluate exactly0/500/1000/1500/2000 on the SAME1312 original validation positions
from episode-start history. Selector remains existing balanced missed-stop/
false-hold score +.02*go-speed MAE. Record all scores/checkpoints, controller
counts, path/speed errors and per-family behavior; no legacy-v2 guard is weakened
or retrospectively reselected. Every newly selected model is a research candidate,
not a promoted working driver. Profile before estimating whole-study duration.

In-domain JEPA: start from the fixed comma checkpoint;10000 unlabeled updates,
8 clips x3 views,256x144, half spatial patches masked consistently through time;
EMA target. Predictor remains action-unconditioned at M4. No hyperparameter
selection on driving validation. Use a fixed TRAIN-only unsupervised holdout
for collapse/loss diagnostics, train on the rest; report exact split and
within-clip/scene variance. Future causal action conditioning is separately M5.
Bound pretraining6h, downstream1h per fit, cache2h; measured times must justify
any extended/repeated budget. Preserve model/optimizer/RNG/sampler and manifests.

One regenerable frozen encoder cache at a time, FP16 and explicit quantization
also used at deployment; keep source manifests/hashes after cache eviction.
Native photometric cache stays shared. Reconstruct evicted supplement RGB in
RAM from preserved lossless shards; never delete source video or original M1
cache.25GB reserve and STOP/owned process guards throughout.

Predeclared ablations at100%/seed42, same2000 updates: color fields disabled for
each encoder; a trained nav/ego/color-only model with visual tokens zeroed.
Inference blanking alone is a separate sensitivity diagnostic, not that trained
baseline. Report all66 fits and failures. Offline curves are not driving gains.
Evaluate unchanged six-case engineering pilot on frozen candidates; development
and M3 paired rule outcomes retain failed/collision cases and clustered intervals.
M2 camera long-drive/native encounter evidence and M3/M5/M6 gates remain open.
Do not tune on heldout_dev or consume Town05 before final qualification.

Official weight provenance: https://github.com/facebookresearch/dinov2 and
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html.
Use installed timm0.9.16/torchvision0.26 APIs, strict load and archived SHA.
