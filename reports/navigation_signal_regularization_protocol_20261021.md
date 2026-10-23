# TRAIN-junction selection of signal verifier regularization

Register before any internal-fold fit. The frozen 32/96-pixel ResNet verifier has
99.7% TRAIN versus 72.9% component-VAL balanced accuracy on predicted points.
The paired 8/16-pixel change failed protection. Test stronger L2 regularization
using existing larger-crop features; no more crop sizes or native data collection.

Use exact retained cache `outputs/navigation_signal_pretrained_20261021`:
51,094 TRAIN rows / 21 junctions, same 1,024 ResNet features. Entire junctions
are indivisible. Sort town/junction keys by SHA256("42:" + key), assign round-robin
to three internal folds, and persist assignments before fitting. No original
component VAL or reserved route enters any internal fold. Three fits at each
fixed C in [0.0001, 0.001, 0.01, 0.1, 1.0]; same balanced binary L2 logistic,
max_iter=1000/tol=1e-4 and scaler fitted ONLY on that fold's training portion.

Store every head and out-of-fold prediction. Select C by minimum mean of the
21 junction-balanced binary log losses: each junction weights positive/background
equally, then junctions equally. Selection scores predicted TRAIN proposals only,
excluding extra GT-only bulb anchors; all original rows remain available to fit.
Tie chooses smaller C. Require convergence of every fold, finite complete OOF
probabilities and both classes in every selection junction. If a fold fails,
retain the failed study; no silent incomplete-candidate selection or retries.

After writing the TRAIN-only selection receipt, fit exactly one final head on
all original TRAIN rows and evaluate the original 144 VAL images / 72 housings
once. No VAL-based C choice or threshold sweep. Keep foreground threshold 0.5,
all original proposals, original states and unchanged parent coverage protections
(44 joint, each town/state protected) plus >=90% background reduction. Original
C=1 outcome remains a control, not a replacement selector. If C=1 is selected,
require exact old probabilities. Preserve every failure; no automatic deployment.

Run `scripts/train_navigation_signal_regularized.py --out
outputs/navigation_signal_regularized_20261021`. CPU only, two threads; 180 seconds
per owned fit child, 1,800 seconds whole stage, 60 MB artifacts. Cached features
are referenced and hash-verified, not copied. Original guardian baseline, 14 GB
net cap / 10 GB minimum remain. STOP then ordinary owned-child cleanup only.
Archive source and data identities before launch; verify unchanged at closure.

This is development model selection; component VAL has been inspected repeatedly.
No M4 primary, reward, allowed input, route split, benchmark gate or Town05 change.
