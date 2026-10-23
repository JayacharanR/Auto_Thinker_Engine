# M1 20k collection assessment — 15 October 2026

**The initial M1 data gate passed.** Collection completed at15:39:02 IST in
2,480.25 seconds (41.34 minutes). This is a training-data result from a privileged
expert and bounded perturbations, not a learned driving benchmark score.

Source: `outputs/navigation_m1_20k_v2_20261015`; machine-readable assessment:
[navigation_m1_assessment_20261015.json](navigation_m1_assessment_20261015.json).
The earlier 297-sample boundary-error run remains excluded from training.

| Measure | Result |
|---|---:|
| Committed three-camera timestamps | 20,000 |
| Atomic shards | 665 |
| Shard storage | 8.888 GB |
| Collected episodes | 73 |
| Eligible imitation episodes / timestamps | 63 / 17,567 |
| Excluded from clean imitation | 10 episodes / 2,433 timestamps |
| Recorded arrivals / rule-safe arrivals | 71 / 68 |
| Recorded collisions | 3 vehicle collisions |
| Recorded red-light / stop-sign violations | 0 / 0 |
| Actual perturbation / privileged safety-stop ticks | 252 / 7 |
| Action-feedback mismatches | 0 |
| Sampled ego positions checked for partition membership | 20,000 |
| Held-out partition violations | 0 |
| Distinct observed training junctions | 43 |

All committed RGB pixels decoded and matched their hashes; observation/label/
world frame IDs, 10Hz continuity and next-physics-tick action acknowledgements
passed. Source hashes matched before M2 changes. The independent offline spatial
audit uses the hash-verified maps and every stored ego pose. Original shards,
outcomes, console logs and failures remain unchanged.

## Coverage and exclusions

| Encounter family | Episodes | All timestamps | Eligible imitation timestamps |
|---|---:|---:|---:|
| Crossing/yield | 15 | 1,999 | 1,498 |
| Following/stopped vehicle | 15 | 5,883 | 5,553 |
| Route/junction | 15 | 3,736 | 3,317 |
| Signal release/memory | 12 | 2,588 | 2,553 |
| Signal stop | 10 | 3,466 | 3,150 |
| Stop sign | 6 | 2,328 | 1,496 |

All five towns are represented (14–15 episodes each). There are49 expert-mode
and24 noisy-mode episodes. The clean data comprises12,414 expert-mode and5,153
noisy-mode timestamps. All three recorded collisions occurred in crossing/yield:
Town03 route90, Town04 route184 and Town10HD route256. Two occurred without noise;
the noisy-mode collision also had zero actual perturbation ticks. Noise is not
established as their cause. Every collision remains in the headline count.

Five episodes with privileged safety stops are censored despite eventual safe
arrival. Episode57 (Town02 route44) lost its evaluator on a30s CARLA timeout;
its atomic prefix remains, with no final outcome/profile and no claim about
unobserved incidents. Episode72 was intentionally truncated after35 timestamps
to hit the exact20k budget. These seven censored episodes and the three collision
episodes are excluded from clean imitation. Failed data remain explicitly labeled
prediction experience; they are not safe counterfactual labels.

## Limits and next action

This supports the first camera imitation baseline, not the100k gate or a world
model planning claim. Only four clean stop-sign episodes remain. Yellow appears
in187 clean timestamps versus10,745 red,2,771 green and3,864 without a next light.
About51.45% of clean teacher targets are stopped. Repeated video frames are not
independent traffic-rule encounters. Student rollouts and reactive counterfactual
branches are still missing;252 perturbed physics ticks are limited off-policy
coverage.

The initial camera student uses an internal split of the clean training geography:
50 training episodes/14,745 timestamps and13 validation episodes/2,822 timestamps.
Shared junction components remain together, with all six families on both sides.
The new300-encounter heldout_dev suite still needs its own expert reference;
Town05 remains unused. No result here replaces that reference or the legacy M0
expert ceiling.

Measured shard cost projects44.44GB for100k timestamps before traces, videos,
caches and checkpoints. The first trainable-CNN raw image cache uses13.605GB.
With about40GiB disk free afterward, expanding blindly to100k while retaining
that cache would violate the25GB reserve. Prioritize targeted correction/stop
data and an audited camera baseline before more bulk collection.

See [the first student assessment](navigation_m2_student_smoke_20261015.md) and
[initial results graph](figures/navigation_m1_m2_20261015.png).
