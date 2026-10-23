# M0 closeout — 11 October 2026

**M0 infrastructure and expert calibration are complete.** The modified privileged
PDM-Lite expert recorded **291/300 rule-safe completions (97%)**, **298/300 arrivals**
and **6 collision events**. This is a reference for the camera student, not a
camera-driving result or a safety guarantee. No camera training has begun.

## Frozen development results

| Encounter family | Rule-safe / valid | Arrivals | Collision events |
|---|---:|---:|---:|
| Held red, stop and release | 50/50 | 50/50 | 0 |
| Signal release/memory | 47/50 | 48/50 | 1 |
| Crossing/yield | 45/50 | 50/50 | 5 |
| Stopped lead vehicle | 49/50 | 50/50 | 0 |
| Route/junction | 50/50 | 50/50 | 0 |
| Stop sign | 50/50 | 50/50 | 0 |
| **Total** | **291/300** | **298/300** | **6** |

“Rule-safe” excludes only the separately reported minimum-speed criterion, which
penalizes these mandated waits relative to surrounding traffic. Strict completion
with **no infraction of any kind is 63/300**. All raw criteria remain available;
there were no recorded red-light or stop-sign violations. Collision totals include
every raw attempt, irrespective of fault, setup validity or retry.

The nine unsuccessful rule-safe trials are six collision episodes, two blocked
episodes (54 and 190), and one lane-departure episode (72). Crossing/yield failures
include four encounters at Town10HD junction 895 and one at Town03 junction 1441.
The layout collision is route 103. Failures remain in the frozen suite.

## Evidence and corrections

- All **200 declared scenarios started**; the other 100 encounters require no
  scenario insertion. All **100 signal episodes recorded red and green states**.
- Across 300 final episodes: **201,215 aligned camera timestamps**, continuous
  frame IDs and 20 Hz timestamp spacing, matching profile/video-header counts,
  registered route lengths and route-specific seeds.
- Checked archived source hashes in **38 final-result chunks**: one version of
  each driving source file. Median episode throughput was **26.42 frames/s**;
  this excludes startup/reset overhead and is not a training-throughput estimate.
- Verified all video headers as H.264/yuv420p, 1536×288; fully decoded all six
  collision videos. Approximate collision-time bookmarks are available for review.
- One evaluator abort on route 298 was retried under the declared infrastructure
  rule. Five earlier attempts were interrupted by documented host shutdowns.
  No memory-watchdog pauses were recorded in the final continuation.

The first automatic report incorrectly rejected ten Town10HD holds because it
compared actor IDs across simulator sessions. CARLA renumbered the same lamps.
The corrected audit uses the catalog's hash-verified reference inventory, the
declared entry lane and a unique physical-location match within 1 cm. Across
all 100 holds the largest difference was **0.624 mm**. Actual world traces confirm
the selected actors' positions. Tests also reject a numerically identical ID at
the wrong position. The original report and raw evidence are preserved; no
route, seed, driving action or evaluator criterion was changed to recover passes.

Assessment checks: **40 passed, one optional-import test skipped**; changed-file
lint/format checks passed. Confidence intervals in the JSON resample junctions.
Unanimous samples give null bootstrap bounds, not zero risk. The 50 stop trials
cover only seven junctions; repeated locations are not independent evidence.

## Implications for M1

The teacher is usable as an initial reference, but copying all its actions would
also copy failures. Prioritize crossing-gap decisions, blocked release behavior
and path tracking near curbs. Approximate collision bookmarks show braking in
all five vehicle-collision cases; they do not establish fault or root cause.

Next implementation work is the crash-safe data engine: separate allowed camera/
navigation/ego inputs from privileged labels, store requested and executed actions,
tag collisions and interventions, and split collection locations from the frozen
development locations. Preserve unsuccessful experience for predictive learning;
do not silently label it as a clean expert demonstration. Start with a small
audited shard before the 20k-timestamp collection stage and the compact CNN
path/speed student. Multi-hour collection/training still needs the agreed budget
approval. Town05 remains untouched.

## Artifacts

- [Corrected assessment JSON](navigation_expert_ceiling_v2_audited_20261011.json)
- [Original automatic assessment](navigation_expert_ceiling_v2_20261009.json)
- [Trace/source/light evidence audit](../outputs/navigation_expert_ceiling_v2_20261009/evidence_audit_20261011.json)
- [Collision bookmarks and episode paths](../outputs/navigation_expert_ceiling_v2_20261009/collision_review_20261011.json)
- [Archived evidence-audit script](../outputs/navigation_expert_ceiling_v2_20261009/audit_m0_evidence_20261011.py)
- [Approved roadmap](navigation_plan.md) and [runbook](navigation_runbook.md)
