# Corrected common encoder study — prospective for remaining arms

Preserve navigation_m4_spatial_protocol_20261019.md and its speed-conditioned
ResNet prototype. That prototype's0/6 arrivals and artificial ego-speed probe
invalidate using its offline score as evidence of visual decisions. Its planned
66-fit matrix never started. Do not erase or pool that result with this study.

Use spatial-visual-decision-v2 consistently for all five encoders and all66 cells.
The learned model omits instantaneous ego speed; the unchanged physical controller
uses actual measured speed. Heading/yaw rate, RGB, calibration, navigation and
past memory remain declared inputs. This architecture is frozen after the single
ResNet100%/seed42 feasibility fit and its pilot began, and before any other encoder
or label-budget fit. Disclose this timing and baseline-development history.
Reuse that exact fresh model (not a second favorable attempt); preserve its
unfavorable40.0% stop recall/7.56% false holds/3.200m/s go MAE and every pilot result.

Everything else is the original spatial protocol: fixed59 TRAIN episodes/16639
labels and13 internal-validation episodes/1312 scored positions; nested actual
733/1257/4069/16639 labels; five encoders × four budgets × three seeds;2000 updates,
identical sampler draws, same fresh common heads, AdamW3e-4/BF16/grad1, burn-in8,
same registered validation schedule/selector. The first-eight-frame launch
supervision limitation is disclosed; do not silently repair it in one arm.
All five color-disabled100%/seed42 ablations and the trained nav/heading/yaw/color
baseline remain included. No full-label driving/perception head is inherited.

Continue JEPA exactly as navigation_jepa_continuation_protocol_20261019.md,
TRAIN-only/no action labels. Cache only frozen backbones, one regenerable cache
at a time, preserve recipes/hashes and all trained models/results;25GB reserve.
Report all cells/failures and per-encoder downstream/pretraining costs. Offline
label curves are not safety evidence. M2 native long-drive/encounter, M3 physical
counterfactual, M5 planning and M6 frozen Town05 gates remain separate and open.
No reward, route, scenario, score, heldout_dev or Town05 selection change.

Geographic exposure audit before matrix launch: the historical M0 integration
route, reused for the visual-policy demonstration, actually traversed globally
heldout_dev Town03 junctions1135/103 (142/105 recorded ego-center junction ticks).
It was explicitly an engineering integration route, but its geography is not
untouched after this test. The shared architecture/protocol were fixed before
its final outcomes. Keep all300 frozen heldout_dev cases and denominators; do
not tune on the exposed locations or claim all17 locations remained unseen.
TRAIN/pretraining arrays are unchanged and contain neither location. Town05
student outcomes remain unused. reports/navigation_integration_exposure_20261019.json.
