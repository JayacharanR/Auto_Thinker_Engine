# TRAIN signal appearance and proposal diagnosis

The existing supported head misses104of1,248TRAIN housing labels. The dominant
next question is candidate placement, rather than a demonstrated need for another
phase-classifier fit.

| TRAIN outcome | Housings | Correct with an existing audited active-pixel query | Median box width × height |
|---|---:|---:|---:|
| Correct detection | 1,144 | 1,144 | 4 × 8px |
| No original proposal in housing | 58 | 57 | 3 × 5px |
| All original proposals filtered | 46 | 37 | 4 × 11px |

There are no wrong-phase accepted housing matches in this TRAIN result. Of58no-
proposal cases,33areyellow,18red including repeated-red images,7green. The filtered
46include30red including repeats,10yellow,6green. All46have an existing audited
active-pixel query in the retained TRAIN cache, so the94/104recognizable count is
not inferred from new fits or fabricated points.

**Oracle limitation:** active-pixel locations come from TRAIN annotations. They
are diagnostic queries of existing out-of-fold predictions, not legal runtime
proposals and not a driving metric. No such coordinates may enter the new RGB-only
candidate generator. Repeated phases and points are correlated; only the verifier
head is excluded by junction, while the upstream proposer has TRAIN exposure.

Median brightest active-pixel intensity is180.3forcorrect,198.2forno-proposal and
177.2forfiltered cases; median active saturation.643/.552/.622respectively. These
aggregate descriptions do not show a clean dimness-only explanation or a causal
photometric fix. They do not justify further blind phase-head fitting.

Next registered experiment preserves every original proposal and adds the RGB
channel maxima within its native5×5neighborhood. No intensity/score threshold,
labels, phase state, map or GT crop chooses these new points. The same learned
foreground threshold.5and original housing metrics remain. See
[navigation_signal_refinement_protocol_20261022.md](navigation_signal_refinement_protocol_20261022.md).

Evidence: `navigation_signal_appearance_20261022.json` stores all1,248housing records,
source/input hashes, exact phase/location strata and all oracle-query limitations.
No component-validation diagnosis, new fit or label change occurred.
