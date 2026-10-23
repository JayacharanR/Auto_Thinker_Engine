# Static phase supervision: validator revision2, before fresh collection

The first51-pose corpus completed in73.00s,612 images. Its v1 data gate FAILED:
56 responsive head/view candidates,55 yellow,15 junctions in4 towns. Preserve
`outputs/navigation_m2_signal_corpus_retry2_20261016/data_audit.json` unchanged.
No training consumed it. The failed gate will not be relabelled as passed.

TRAIN-only inspection explains the missing Town10HD candidates: nine clearly
phase-responsive crops at the one selected junction have red/green MAE8.95–26.46,
with repeated-red variation0.54–2.03. Their rendered green-over-red amplitude
is only18–22, below v1's hard30 criterion. Bright near-white bulbs and which bulb
is lit contain evidence that v1's absolute-hue assumption rejects.

Revision2 replaces that assumption with paired local on/off evidence. Retain
mean RGB change>=2 and>=5*max(repeated-red MAE,.25). Require both a red-on and a
separate green-on mean intensity gain>=20, plus maximum channel change>=25.
A global brightness shift cannot satisfy both opposite-signed changes; unstable
repeats and static/back-facing housings still fail. Synthetic regression tests
cover near-white greens, global drift, unstable repeats and the unchanged v1
failure. Original-scene visibility remains unknown. Yellow's additional chroma
criterion remains unchanged and unknown yellow candidates remain unlabelled.
This is a new training-annotation heuristic, not a change to any driving metric.

Freeze fresh plan `navigation_m2_signal_pairs_plan_v3_20261016.json`, SHA
`92335457bb8521593450eff76007a119e663a49b01c9820689b049b7c32c911e`.
Same51 poses,17 junctions,5 TRAIN towns and3 viewpoints, with new background seeds
(offset100000). Same612-image integrity gate,5 responsive towns/>=10 junctions,
900s cap and all prior guards. Collect fresh PNGs; do not change the earlier gate.
No student, reward, benchmark or runtime camera setting is changed.

Earlier engineering attempts also remain: default-map mismatch was rejected before
imaging; a missing renderer geometry-path argument lost the first Town01 final
summary. Corrected explicit map loading and argument, plus per-pose atomic
metadata/image hashes, prevent those failures from silently producing labels.
