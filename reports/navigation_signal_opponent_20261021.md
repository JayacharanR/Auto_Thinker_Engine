# Relative-color features do not beat the raw-RGB control

All seven fits finish in94.91seconds. Whole TRAIN-junction OOF selection retains
the raw87feature joint classifier: macro binary logloss0.360094 versus0.368144 for
the312feature opponent-color representation. The unselected representation is
NOT evaluated on component VAL. No threshold change or posthoc model substitution.

| Representation | Features | Mean TRAIN-location OOF loss | Selection |
|---|---:|---:|---|
| Original raw RGB |87|0.360094|Retained|
| RGB plus relative color/contrast |312|0.368144|Rejected|

Same augmented data, labels, whole-location folds, sample weights,400trees and
settings. The added225features comprise channel differences, chromaticities and
local contrast. They are deterministic RGB transformations, not metadata inputs.
Raw-fold fitting took5.22/5.25/5.31s; larger representation19.16/21.44/20.94s.
Equal tree count did not mean equal CPU cost. Existing feature caches were reused.

Independent audit reproduces all six OOF vectors and macro losses, verifies
whole-location separation,400tree/feature-count receipts and selection before the
final fit. Every raw-control fold class-probability vector matches the previous
study bit-for-bit. Final raw-control TRAIN and component-VAL vectors also match
exactly, and metrics match their independently recounted predecessor. This is
repeatability evidence, not a new driving result.

The retained model therefore remains45/72localized,44joint,1,171background peaks.
It still FAILS protection: yellow10/18 versus parent15/18; Town10HD9/32versus10/32.
Neither aggregate44joint nor96.86%fewer background detections justifies promotion.
See [state-aware results and graph](navigation_signal_phase_boost_20261021.md).

Ten focused feature/state/group/research-boundary tests passed. Original failed
studies, source archives, caches and denominators remain. Same original storage
baseline14GBnet/10GBminimum; no new native cases, benchmark/reward/interface changes,
M4primary reselection or finalTown05trial. M2/M3/M5 remain unqualified.

Next investigate a compact verifier that retains the frozen proposer CNN's learned
visual features and probability evidence, alongside RGB. The present second stage
discards those features. Register exact extraction, TRAIN-only location selection,
capacity controls and limits before fitting; keep failed representations and all
state/town gates. No claim of improvement until tested.

[Protocol](navigation_signal_opponent_protocol_20261021.md).
Evidence: outputs/navigation_signal_opponent_20261021.
