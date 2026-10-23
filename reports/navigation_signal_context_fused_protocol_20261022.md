# Compiled exact RGB reductions

The preceding context profile passed exact patch, feature and probability checks
on 236,284 points across 63 fixed TRAIN images. Its median feature-plus-head time
was 343 ms, versus 976 ms for the original. Neither is a full runtime profile.

Separately stage a Numba reduction that gathers the same pixels and accumulates
FP32 means/variances in the original sequential order. Fast math is disabled;
no approximate moments, lookup tables, thresholds, candidates or model changes.
Numba disk caching is disabled because dynamic staging imports have no stable
module identity. Measure one synthetic compilation/warm-up separately; include
per-image integer-average preparation and normalization in inference timing.

Use the same fixed 63 TRAIN images, all frozen candidate coordinates, original
supported-supervision head and exact equality checks as
`navigation_signal_context_profile_protocol_20261022.md`. Measure original and
compiled extraction once per image, alternating order. Require exact FP32/FP16
features and exact probabilities. Retain every receipt and failure. Concurrent
CPU audit is disclosed; no fitting, component VAL, native driving or deployment.

Output `outputs/navigation_signal_context_fused_profile_20261022` with 600 s /
20 MB cap, two CPU threads, original storage budget. Registered source bytes are
pinned. A successful profile is evidence for future integration review only.
