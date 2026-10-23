# TRAIN-only spatial support diagnosis

The detailed RGB representation covers only a5×5patch around a proposal. Of3371
predicted positive TRAINpoints,2006(59.5%) have no conservatively audited active
bulb pixel in that window. All belong to known responsive housing boxes, but box
membership alone does not establish that a small local crop can reveal its phase.
This motivates a supervised-support/context audit before another representation fit.
No labels, thresholds, model selection or component-validation predictions changed.

| Support | Points | Foreground accepted | Correct foreground state |
|---|---:|---:|---:|
| Audited bulb within 5×5 | 1365 | 1259 | 1253 |
| Audited bulb outside 5×5 | 2006 | 633 | 360 |
| All predicted positive TRAIN points | 3371 | 1892 | 1613 |

Predictions are the frozen raw-joint100model's out-of-fold outputs. Every point's
head excluded its whole TRAINjunction. Extra GT-only anchors are excluded; all
3371rows are original predicted positives. The upstream frozen proposer saw some
TRAINlocations, so this does not test unseen end-to-end representation learning.
These correlated point counts are not independent-driving success rates or CIs.

Support is the minimum Chebyshev distance to a conservative phase-response pixel
inside the same matched housing. All3371matchedhousings have at least one such
pixel. The mask can omit real low-response pixels; absence within a window does
not prove the crop contains no usable information. Radius/location/scale are
correlated, so the contrast in recall does not prove a causal context effect.

| Detailed RGB window | Positive points with an audited active pixel |
|---|---:|
| 5×5 | 1365 /3371 |
| 9×9 | 1569 /3371 |
| 17×17 | 1974 /3371 |
| 25×25 | 2329 /3371 |
| 31×31 | 2509 /3371 |

Even17×17would cover only58.6%ofthese positive points; blindly enlarging5×5is not
a demonstrated fix. Large projected boxes can put positive point labels far from
the active bulb. Distinguish sufficient visual context from unsupported local phase
supervision using TRAIN-only inspection before registering the next fit. Do not
relabel component VAL or drop its missed lights. No new training or native data.

Evidence: reports/navigation_signal_train_support_20261021.json and
scripts/diagnose_navigation_signal_train_support.py. Same originalTRAINgeography,
all old failures retained; M2/M3/M5unqualified andM6notrun.
