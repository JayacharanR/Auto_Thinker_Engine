# Continuous expert data and causal history — prospective camera experiment

Before fitting, keep the selected camera-stop-go-v2 parent, original M1 cache,
original internal validation junctions and all labels/positions immutable.
Town05_final_v2 is registered; no Town05 image/outcome enters learning. Retain
the canonical-background and empty-background collection results separately.

Use only clean expert episodes that pass the frozen continuous-approach gate
from the new24-episode four-town VARIED-WAIT collection. Keep the fixed7s
collection as a separate control, not this fit's new supplement. Require the
independent20Hz position report to bind the exact collection audit/guard.
Failed/censored/queued episodes
remain excluded from this supplement. Every new episode is TRAIN; no resplit.
The original driving validation is unchanged. Additionally exclude static
component-validation geography from the ORIGINAL TRAIN sampling pool in new
fits. The parent may historically have seen that geography; disclose that
pretraining history and make no untouched-component-test claim.

One main comparison: selected parent versus a bounded fine-tune with clean
continuous direct-signal sequences and episode-start causal memory. Architecture,
controller, input rig and driving criteria stay unchanged. No detector/tracker
integration, new handcrafted stop rule or threshold calibration. This is a
combined data/history countermeasure; any gain cannot identify either cause
alone. Require a later matched ablation before a mechanism claim.

Draw whole TRAIN episodes, half original family-balanced episodes and half new
approaches. Keep hidden state from episode start; backpropagate through the
complete GRU history, checkpoint visual forward chunks to fit memory. Supervise
after a fixed8-frame causal burn-in so the first half of a long wait is not
discarded. Keep original loss terms/weights for original examples. New examples
supervise expert path/speed/brake/stop targets only: unverified visibility/state,
distance and hazard labels do not receive new perception supervision. Never
imitate executed controls from assisted student data.

Fresh AdamW5e-5,grad cap1,seed42,BF16 autocast on CUDA (no scaler needed),
1000 updates and3600s cumulative cap including cache/scoring, STOP and >=25GB
reserve. Save full optimizer/RNG/sampler, dataset/source hashes and selected vs
latest checkpoints. No source-changing resume or automatic longer repeat.

Evaluate parent0 and250/500/1000 using episode-start causal burn-in on the SAME
pre-existing internal validation label positions (the second half of each fixed
32-frame block). Keep the original clipped-history metrics as a separate
regression diagnostic. New primary operational selector requires causal stop
recall>=parent, false holds<=parent+.02, go-speed MAE<=parent+.2m/s and a strictly
better existing balanced-stop/false-hold score. Clipped-history stop recall may
drop at most.02 and false holds rise at most.02. This is a new prospective
development selection, not a changed driving benchmark or retroactive reselect.

After a passing selected checkpoint, evaluate a fresh fixed TRAIN camera pilot
with overrides OFF, plus the unchanged1.23km route. Keep failed videos and all
collisions. Parent0 retained if no update passes; do not rerun identical weights.
M2 is not complete until real camera behavior is measured. M3–M6 research/test
gates remain separate and the new heldout_dev expert reference remains pending.
