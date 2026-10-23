# Exact context extraction profile

Register before measuring the staged implementation on real TRAIN images.
This is an implementation-equivalence and cost study, not a model selection or
driving evaluation. It does not change the running full TRAIN audit.

Use the same 63 fixed images already listed in
`outputs/navigation_signal_background_fit_20261022/cache_verification.json`:
the first red, green and yellow image at each of 21 TRAIN locations. Use every
frozen plateau-union coordinate for each image, with no top-K or score filter.

The staged extractor precomputes all aligned 3×3 integer averages in the padded
RGB image, then gathers the same native 32×32 and reduced 96×96 windows. Preserve
the original interleaved memory layout and NumPy feature reduction. Process
256 points at a time to bound temporary memory. Require exact patch bytes and
FP32/FP16 feature equality at every point, plus exact probabilities from the
unchanged supported-supervision final head. This head's previous qualification
failure remains; using it for equivalence does not promote it.

Time original and staged feature extraction, including per-image preparation,
once per image, alternating order by image index. Keep both probability timings
and total feature-plus-head timings. Report medians and p95 with all per-image
values. Main CPU audit is concurrent; no isolated throughput claim. Exclude image
decode, candidate generation, neural proposer, IPC, calibration, tracking, policy,
simulator and video. Do not call this a 10 Hz driving pipeline.

Output `outputs/navigation_signal_context_profile_20261022`, 600 s / 20 MB cap,
two CPU threads, original storage guardian and budget. Source/model/cache hashes
pinned and archived. STOP/error retained; no fitting, component VAL, native driving
or deployed source modification. A pass permits later integration review only.
