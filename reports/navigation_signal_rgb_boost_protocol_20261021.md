# Nonlinear native RGB verification with TRAIN-only photometric augmentation

Registered before feature extraction or fitting. Fixed-order gallery shows some
supported housings only 2x4 pixels. Direct source PNG inspection and exact phase
RGB values confirm real emissive changes (including a single red pixel changing
[194,185,175] to [255,199,183]); no image-loader corruption was found. Previous
linear RGB features underfit TRAIN (69.4% predicted-point balanced accuracy),
while pooled ResNet features overfit locations. Test a nonlinear RGB baseline.

Two arms share the same 87 RGB features, same 51,094 original TRAIN points,
original state proposer, binary labels and exact original component VAL. Use
HistGradientBoostingClassifier: learning_rate=.1, max_iter=100, max_leaf_nodes=15,
min_samples_leaf=50, l2_regularization=10, class_weight=balanced,
early_stopping=False, random_state=42, max_bins=255. Fixed iteration budget,
not a claim of convergence. Native RGB statistics only, no metadata inputs.

Arm `plain` fits the original features. Arm `gamma` adds two versions of each
TRAIN image: uint8 round(255*(RGB/255)^gamma), gamma=.7 and1.4. Apply to full RGB
BEFORE extracting native32 / area-averaged96 patches; recompute all87 features.
Same channelwise monotone lookup preserves channel ordering and spatial geometry;
it approximates photometric variation, not physically complete new illumination.
Reproduce original untransformed RGB features exactly against retained FP16 cache.
No spatial augmentation, synthetic labels, hue swaps, new locations or VAL transforms.

Use the exact previous three whole-TRAIN-junction folds (21 locations) and
predicted-point selection mask; all original and augmented versions of a location
stay in the same fold. Six fits produce OOF predictions on ORIGINAL TRAIN features.
Choose the arm with lower macro junction/class-balanced OOF logloss, tie to plain.
Only then fit that arm on all TRAIN data and assess original VAL once. No threshold
search: gate .5; retain all original38,543proposals and original predicted states.
Same coverage protection (44joint and each town/state), >=90% background reduction.

The nonlinear head and augmentation are new engineering choices; the paired arms
isolate augmentation within this recipe, not a causal pretraining result. Prior
linear and semantic failures remain retained. No benchmark, reward, allowed-input,
M4-primary or final Town05 change. No automatic driving or final model promotion.

Output `outputs/navigation_signal_rgb_boost_20261021`, runner
`scripts/train_navigation_signal_rgb_boost.py`. CPU two threads; 180s per owned fit
child, 1800s whole stage including feature extraction, 100MB artifacts. Reuse original
feature caches; retain augmented features, every fitted model and OOF vector.
STOP / ordinary owned-child cleanup only. Same original storage guardian baseline,
14GB net cap /10GB minimum. Source hashes pinned from request through closure.
