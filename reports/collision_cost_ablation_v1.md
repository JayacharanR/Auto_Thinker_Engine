# Approved collision-cost pilot — 2026-10-05

The user approved the experiment proposed in `reports/bev_60k_assessment_20261005.md`: test whether a fixed cost addresses the measured weak penalty for low-speed collisions, while controlling for the benefit of additional training. This is a CNN BEV diagnostic, separate from the camera encoder comparison. Improvement is a hypothesis, not guaranteed.

| Setting | Control | Fixed-cost pilot |
|---|---|---|
| Initial weights | Same immutable copy of assessed 60k checkpoint | Same |
| Replay and optimizer at pilot start | Fresh | Fresh |
| Training seed / task / observation / actions | 42 / medium / BEV / discrete | Same |
| Budget | 30k new steps including 2.5k prefill | Same |
| Train ratio / evaluation interval | 512 / 2.5k | Same |
| Early stopping / selected checkpoint | Disabled / final 30k checkpoint | Same |
| Existing time cost / speed-scaled collision penalty | 0.1 / 30 × ego speed | Same |
| Added collision event cost | 0 | **300, charged once per episode** |
| Final benchmark reward | Original reward, fixed cost explicitly zero | Same |

The vendor collision sensor remains latched until reset, and terminal handling can observe it one step later than reward handling. Charging the fixed cost on every flagged reward step could accidentally charge twice. The implementation applies it on first collision detection and resets its guard at environment reset. Existing collision-speed penalties and terminal flags remain unchanged. `collision_events.jsonl` captures reward components, speed, first detection and terminal flags without exposing them as agent inputs.

Queue: `outputs/collision_cost_ablation_queue`. Pilot directories: `outputs/diagnostics/collision_cost_ablation_v1/{control,fixed300}`. Controller: `scripts/run_collision_cost_ablation.py`; uses existing launcher, queue lock, STOP, health guards and retry support. No `jobs/*.sh` changes.

The queue first measures the immutable 60k reference, then trains/evaluates the control, then trains/evaluates the variant. Each candidate gets 30 medium episodes on each predeclared stream **2029 and 2030**, plus ten simple-turn retention episodes on 2029, with no shield. Evaluation appends explicit canonical reward overrides after those inherited from checkpoint training. Different policies and episode lengths need not produce paired identical traffic scenarios.

Approximately three hours expected; four-hour controller cap allows startup/evaluation/retry overhead. The cap stops cleanly and preserves the current checkpoint; it does not mark an unfinished pilot complete. No automatic hard training. Report success and collision flags, collision-free arrivals, timeouts and canonical returns; do not compare modified training returns directly across pilots. Checkpoint selection uses fixed budgets, avoiding reward-based selection differences.

Provenance: the queue saves `source_60k.pt`, its SHA256, the immutable experiment protocol, code/config/patch snapshots, per-evaluation checkpoint hashes and requested settings. Resume rejects a pilot checkpoint with a different source/reward/observation contract. Default `collision_fixed=0.0` preserves existing benchmarks; the tracked `patches/cardreamer_compat.patch` contains the vendor changes. Controlled reward tests cover stationary/moving collisions, one-time application, reset, non-collision and legacy defaults; runner tests cover canonical evaluation and fresh/resumed pilot contracts.

On completion, `results.json` and `comparison_summary.md` contain the three candidates' checks. Favor a repeatable reduction in collisions without increased stalling or lost retention, then replicate any promising result with a second training seed before adopting a common, versioned camera-encoder protocol.

Live progress (fish):

```fish
cd ~/Project/Auto_Thinker_Engine; and python3 scripts/watch_queue.py --out outputs/collision_cost_ablation_queue
```

Clean stop: `touch outputs/collision_cost_ablation_queue/STOP`.

2026-10-06 assessment: the control completed, but a CPU telemetry timeout stopped
the variant at 21,278 steps. See `reports/collision_cost_assessment_20261006.md`
for canonical partial evaluations and the prepared controller recovery. The
original experiment remains unfinished; the variant still needs 8,722 steps.
