# Prospective fixed crop-scale comparison

Registered before new feature extraction or fitting. The prior frozen ImageNet
verifier lost 18 of 44 jointly correct lights, all in housings under 16 native
pixels tall. This motivates, but does not prove, a scale hypothesis. The small
component validation set has been repeatedly inspected; it is development data.

Run the existing `train_navigation_signal_pretrained.py --tight-crops` recipe at
`outputs/navigation_signal_tight_20261021`, with exactly one ResNet change:
replace native 32-pixel / averaged 96-pixel context views with native 8-pixel and
16-pixel views centered on the same proposal. Both are bilinearly resized to 128.
Retain local central 2x2 pooling and second-view global pooling, ImageNet
normalization, frozen ResNet18 weights, 1,024 features, BF16 extraction / FP16 cache.
The smaller windows come from the original native 32-pixel crop, with the same
edge padding and integer proposal center. No source-coordinate or map feature.

Everything else is paired: original bulb proposer, all 38,543 VAL proposals,
2,844 TRAIN images, 51,094 mined TRAIN points and binary labels, random mining
seed 42, fixed balanced L2 logistic regression C=1, max_iter=1000, tolerance 1e-4,
TRAIN-only scaler and 0.5 foreground threshold. The RGB statistics arm repeats
as an implementation control. Independently require exact TRAIN rows/labels,
RGB feature arrays, validation offsets and repeated RGB validation probabilities
against the prior retained study before accepting a scale comparison.

Same 144 VAL images / 72 housing labels; score every proposal, no top-K cap,
original phase predictions retained. Same total 44 joint lights and every town
and state protected; >=90% fewer background peaks than parent 37,241. Convergence
and export parity also required. Report both arms regardless of result. No
threshold sweep, retries, extension, best-of-many selection, native deployment,
M4 primary change, reward change, benchmark relaxation or final Town05 tuning.

Existing recipe caps remain: 1,800 seconds features, 600 seconds per owned fit
child, 3,000 seconds stage, 400 MB artifacts. Original storage guardian baseline,
14 GB net cap and 10 GB minimum reserve remain unchanged. STOP and ordinary
owned-child cleanup only. Record source hashes before launch and verify afterward;
preserve all failed runs and caches. Source must stay fixed until terminal.
