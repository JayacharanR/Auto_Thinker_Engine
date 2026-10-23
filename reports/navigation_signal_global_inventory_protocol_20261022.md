# TRAIN inventory of independent RGB proposals

Registered before generating candidates. Local refinement remains seed-limited:
39of59remaining TRAIN misses have no candidate, and49of59can be recognized by the
existing verifier at annotated active pixels absent from the candidate set.
Oracle coordinates remain diagnostic only and never become runtime inputs.

Test ONE fixed global RGB proposal rule: each native pixel that is a channel-wise
maximum within5×5 and whose channel local range(max−min)is≥16/255. Union R/G/B,
retain all qualifying plateau pixels, no intensity minimum, hue test, top-K,
image-position restriction or class-label input. Border extension is nearest.
This16level local-contrast rule is a model-proposal hyperparameter fixed before
measurement; it is not the classifier acceptance threshold or a benchmark change.
No threshold/window search in this inventory.

Generate on all2,844original TRAIN images; union with all1,770,643previous refined
points, retaining them exactly. Save every coordinate and per-image offsets. Count
proposal growth, timings and extra candidate coverage of the original1248TRAIN
housings and59misses. Inspect whether the49recognized oracle cases have any exact
recognized oracle coordinate in the new RGB set. Oracle membership is diagnosis
only, not a measured new model prediction or a driving result.

No classifier inference/fit/componentVAL/native driving. This inventory determines
whether a full classifier audit is worth its compute cost: proceed only if at least
one previously no-candidate housing gains a coordinate AND at least one missed
housing gains a recognized oracle coordinate; ≤15milliontotalunionpoints and
≤120MBinventoryartifacts. These resource criteria prevent an unbounded dense scan;
if exceeded, retain the failure and do not truncate candidates or silently change
the rule. Model inference and downstream gates require a separate protocol.

Output outputs/navigation_signal_global_inventory_20261022; CPU,600s stagecap,
120MBartifactcap, original14GBnet/10GBminimum guardian. Source/model identities
retained; STOP/failure retained/no retries. Previous component55/72stillFAILSyellow.
