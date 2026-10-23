# RGB-only local proposal refinement — TRAIN-only audit

Registered before candidate-generation or classifier evaluation. The preceding
TRAIN appearance audit decomposes104missed housings into58with no proposal and
46with all proposals filtered. Existing OOF predictions at audited active pixels
recognize94ofthese104. Those oracle coordinates are diagnostic only and prohibited
as runtime candidates. No new phase head or photometric fit is justified yet.

Keep the exact three supported-supervision OOF heads. For each original RGB proposal,
keep its point and add the native5×5 neighborhood's maximum in each ofR,G,B. Ties:
nearest squared pixel distance, then y, then x. Merge duplicate coordinates. All
original coordinates remain; no top-K, intensity gate, phase label, map input or
GT crop. No search over neighborhood sizes or color transforms. This modifies
the model's proposal generator, not the benchmark labels or acceptance threshold.

Evaluate every resulting point on all2,844TRAINimages with the existing same-junction
excluded head, same87RGBfeatures rounded through originalFP16 cache boundary and
foreground acceptance≥.5. Reuse frozen probabilities for original points, compute
only new-coordinate features. Recompute every original sampled predicted row's RGB
features and probabilities as parity checks. Keep all point/probability arrays and
offsets. Same score-order point-in-housing matching, original1,248TRAINlabels, broad
background and unknown masks. Unknowns are reported, not counted as correct negatives.

Compare to the original supported TRAIN pipeline(1,144joint,724background). Continue
only if totaljoint strictly improves, no town or phase loses joint detections, and
known-background detections remain≤10% of the frozen rawRGBcontrol30,758 (≤3,075.8).
This permits an explicitly declared false-positive/recall tradeoff while retaining
90%background reduction relative to that control. Report all count changes and
per-image inference timings; no driving-rate claim without a runtime profile.

No new fit, component VAL, native driving, deployment or finalTown05test in this
stage. Full component gate remains the original44joint/town/phase protections and
90%reduction versus37,241parentbackground. No threshold relaxation. TRAIN signals
and repeated phases are correlated; the upstream CNN proposer has TRAIN exposure.

Output outputs/navigation_signal_refinement_train_20261022. CPU2threads,900s wallcap,
120MBartifactcap, original14GBnet/10GBminimum storageguardian. Source bytes pinned
until terminal; failures retained/no retries. Original proposals/labels untouched.
