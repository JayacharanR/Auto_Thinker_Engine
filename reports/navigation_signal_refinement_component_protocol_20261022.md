# Conditional component check of RGB-only refinement

Registered before any refined component predictions. Run only if the full TRAIN
refinement audit passes its predeclared recall/town/phase/background criteria and
48,865 sampled-row parity checks. Otherwise leave this protocol unexecuted.

No fitting or parameter changes. Use the exact previously fitted supported head
ed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191.
For all144original component images, preserve all38,543original proposer coordinates
and probabilities, adding only the registered5×5RGBchannel maxima with geometric
tie-breaking. Deduplicate coordinates, without truncation or score filtering before
the original learned foreground threshold.5. Infer only new-coordinate features;
recompute ALL original38,543RGB87features and model probabilities for exact parity.

Use original72housing labels, broad background/unknown masks and score-ordered
one-to-one point matching. Record all coordinate/probability arrays and offsets,
before/after counts, model/source hashes and timings. Compare to both supported
baseline43joint/40background and immutable bulb parent44joint/37,241background.
Qualification remains≥44joint, every original parent town and phase protected,
and≥90%fewer known-background detections versus37,241. Unknowns remain separate.
No threshold sweep, extra classifier fit, reserved-location training or benchmark
change. This component set is repeatedly inspected development evidence.

Output outputs/navigation_signal_refinement_component_20261022. CPU2threads,
300s wallcap/40MB artifactcap, same original14GBnet/10GBminimum guardian. STOP and
retain failures, no retries. No automatic runtime deployment or native/finalTown05
driving, even if the component gate passes. Runtime speed and point-to-policy
integration remain separate work requiring verification.
