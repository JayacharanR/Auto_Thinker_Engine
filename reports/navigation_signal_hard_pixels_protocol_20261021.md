# Rare-background countermeasure, fixed before fitting

The completed center/bulb study remains failed and unchanged. Bulb labels give
56localized/44joint of72known lights, but37,241known-background peaks and1.654%
active background. Fixed full-view examples show many isolated facade/foliage/
road-edge responses. Mean backgroundCE.07156can hide these rare high-confidence
mistakes. Test explicit hard-background weighting, not a threshold adjustment.

One new engineering fit starts from the exact selected bulb1000checkpoint
SHA27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193.
Same audited876TRAIN/144VALimages, no label/geometry/split change and no native
data. NativeRGBonly architecture/37,772parameters unchanged. New optimizer
AdamWlr.0001/wd.0001, seed42,1000updates,batch12,clip1,BF16/FP32loss. Original
foregroundCE+.2meanbackgroundCE, PLUS mean of the hardest1%background-pixelCE
within each image. Ignore255regions completely. No mining from validation.
This changes the training objective and is not a continuation of the settled
old experiment. The old failed gate and checkpoints remain intact.

Evaluate0/100/250/500/1000with the unchanged threshold.5/8connected one-peak
point metric and72label denominator. New prospective selection: preserve parent
jointcount44, every town joint7/15/12/10 and every state22red/15yellow/7green.
Among protected trained candidates, rank smallest background-peak count,
then smallest background-active fraction, then most joint; earlier tie.
Initialization retained if no protected improvement. Countermeasure pass needs
at least90% reduction in BOTHbackgroundpeaks and active fraction with those
coverage protections. No claim of driving readiness from this component gate.

Retain every checkpoint/optimizer/RNG and all scheduled predictions. Maximum
1200sfit/1500stage/150MBadditional, original storageguardian and STOP. One recipe,
no extension/sweep/automatic native pilot. No reward, allowed-input, M4primary,
M3/M6suite or scientific driving gate changes. Development has already been
inspected; this is not an independent test or causal ablation of all past work.
