# Full TRAIN comparison after expanded background supervision

Registered before full-proposal inference. Three fixed expanded-background fold
fits completed77.45s, after all182,016extra negatives passed independent eligibility/
geography checks and12,096features reproduced across63fixed images/three versions.
Old sampled diagnostics: totaljoint1,144unchanged, background201unchanged, small
town/phase regressions. These diagnostics do not qualify or select a candidate.

Use every frozen12,362,313plateau-union coordinate in the original2,844TRAINimages.
Only the learned head changes; no proposals, labels, thresholds or ranking recipe
change. Classify each image using the expanded-background head whose training
excluded its whole junction. Same RGB87/FP16boundary, foreground≥.5, phaseargmax,
score-order one-to-one housing matches and known-background/unknown masks.

Recompute all48,865original sampled features and new-head OOFprobabilities exactly;
also compare all182,016extra negatives' original-version feature vectors to the
registered cache. Recount the prior full model from its retained probabilities
on the identical coordinates; compare every original baseline count. Store all
new probabilities/coordinates/offsets and incremental per-image receipts.

Predeclared gate from the background study: joint≥1,234/1,248, every town and phase
non-regressing versus that exact plateau baseline, knownbackground≤3,075.8.
The previousbaseline8,197background is retained as failed. No threshold search,
subgroup exception, component-validation prediction or further fit in this stage.
If any gate fails, no component follow-up. Passing would only authorize a separately
registered all-TRAIN finalfit/component check, not runtime or driving promotion.

Output outputs/navigation_signal_background_full_20261022. CPU2threads,4,500s wall
cap/600MBartifactcap (same395.6MBprobabilities+49.4MBcoordinates). The larger time cap
is registered before execution because this head must score ALLcoordinates, whereas
the prior audit reused probabilities for1.77millionoldpoints. No prior deadline is
extended. Original14GBnet/10GBminimum storageguardian and reserve remain unchanged.
STOP/failure retained/no retry. Registered source bytes pinned until closure.

Use original CPU preprocessing/scikit inference; staged GPU export is not substituted.
This is correlated TRAIN component evidence with proposer TRAINexposure, not a
driving reliability claim. Original M4primary remains; M2/M3/M5unqualified,M6notrun.
