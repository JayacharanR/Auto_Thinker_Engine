# Overnight BEV assessment — 2026-10-08

**The run completed and preserved a better checkpoint, but the independent
evaluation gain is modest. Keep the selected90k checkpoint, not the collapsed
150k latest checkpoint.** The large improvement on checkpoint-selection streams
did not fully carry over to the separate final evaluation streams. This remains
a one-training-seed CNN BEV diagnostic, not evidence for a JEPA encoder advantage.

## Completion and selected model

Queue `outputs/overnight_bev_20261006` started2026-10-06 00:45:33 and finished
07:46:32 IST: **7h00m59s**, all32 stages completed, `results.json completed:true`.
It finished the fixed-cost pilot's remaining8,722 steps, compared candidates,
and performed all four30k continuation blocks: **120,000 additional steps**.
No trainer, evaluator, controller or CARLA process was active at the Oct8 review.

Source selection chose the unchanged-reward control, preserving its replay and
Adam state in a separate run directory. The control's initial run counter was
30k; the selected checkpoint's counter is90k, so the useful selected model adds
**60k overnight steps**. These counters follow an original60k source, giving
approximately150k cumulative environment-step budget along this model's lineage,
including pilot prefill. They are not a new training-from-scratch90k total.

Recommended model: `outputs/overnight_bev_20261006/best.pt`, SHA256
`5fe91136ac62dc0fc3ee9753627d221e95463cce7dbdd47fdeb1d850d25c759e`.
It is identical to `checkpoints/step90000.pt` and includes Adam state.
The immutable comparison baseline is `baseline.pt`, matching the original
completed control30k checkpoint. Latest150k is retained as failure evidence.

## Separate final evaluations

These streams were specified before training and were not used to select the
checkpoint. All evaluations used the canonical time cost0.1/fixed collision
cost0, greedy policy evaluation and no shield. Safe arrival means destination
arrival without collision; historical independent terminal flags are unchanged.

| Test | Starting control30k | Selected90k | Interpretation |
|---|---:|---:|---|
| Medium3031, safe arrivals | 30/30 | 29/30 | One fewer arrival |
| Medium4031, safe arrivals | 26/30 | 29/30 | Three more arrivals |
| **Pooled medium** | **56/60 (93.3%)** | **58/60 (96.7%)** | **+2 episodes, +3.3 percentage points** |
| Medium collisions | 3/60 | 2/60 | One fewer collision |
| Medium lane departures | 1/60 | 0/60 | One fewer lane departure |
| Simple3031 retention | 10/10 | 10/10 | Turning skill preserved |
| Zero-shot hard5031, safe arrivals | 26/30 (86.7%) | 27/30 (90.0%) | One more arrival |
| Hard collisions | 4/30 | 3/30 | One fewer collision |

No arrival/collision overlap or timeouts occurred in these final evaluations.
Mean canonical medium return rose295.7→330.5; medium speed slowed slightly
11.31→10.79km/h. Hard return rose254.7→299.3, with speed11.38→11.16km/h.
Slower speed alone does not demonstrate anticipatory yielding.

The evidence supports retaining the selected model as a promising candidate.
It does **not** establish a repeatable, large generalization gain: medium improved
by only two episodes, hard by one, and medium3031 regressed slightly. Both
policies were already strong on the final medium streams. Further evaluation
streams and another training seed are needed before calling traffic solved.

## Selection curve and the late collapse

These are the reused diagnostic streams2028/2029/2030,30 episodes each, with
simple2029 retention10 episodes. They select checkpoints and are not holdouts.

| Run counter | Added overnight steps | Medium safe arrivals | Collisions | Lane failures | Timeouts | Simple safe arrivals | Accepted |
|---|---:|---:|---:|---:|---:|---:|---|
| Source30k | 0 | 78/90 (86.7%) | 10 | 1 | 1 | 10/10 | Baseline |
| 60k | 30k | 82/90 (91.1%) | 6 | 1 | 1 | 10/10 | Yes |
| **90k** | **60k** | **89/90 (98.9%)** | **1** | **0** | **0** | **10/10** | **Best** |
| 120k | 90k | 80/90 (88.9%) | 9 | 1 | 0 | 10/10 | No |
| 150k | 120k | 1/90 (1.1%) | 13 | 4 | 72 | 0/10 | No |

At150k, all ten simple-turn episodes timed out, with near-zero mean speed
(0.000049km/h) and zero route progress. Medium speed was0.91–1.30km/h, dominated
by72 timeouts. Low collision counts here would be misleading if success and
progress were ignored. The preserved best model still passed all final checks.

Ten-episode training evaluation remained strong until147.5k:100% arrival with
10% collisions (one overlap), then fell to0% at150k. This locates the observed
collapse within the final2,500 steps; it does not identify the triggering update.
The120k standalone regression was already evidence that longer training could
lose quality. The final60k overnight steps did not produce an accepted model;
the fourth block exposed a severe stability failure.

All audited checkpoint weights **and Adam tensors are finite**. Adam counters
from source30k to latest150k grew by59,961 world-model updates,59,992 actor
updates and59,988 value updates, close to the expected60k at ratio512. The
optimizer did not broadly stop learning. Intermittent non-finite AMP gradient
norms occurred, including an infinite actor gradient norm logged at149k, but
these observations do not prove that an overflow caused the behavioral collapse.
The exact cause remains open; a finite policy can still learn to stop.

Code inspection adds two recovery limitations: checkpoints save Adam but do not
save AMP GradScaler or RNG state, and milestone snapshots do not separately
archive replay. Restoring the90k checkpoint from the current150k run therefore
needs a deliberate replay plan; simply resuming that run directory loads the
bad150k latest checkpoint. The stored NPZ episode times permit investigation of
replay up to the chosen milestone, but exact uninterrupted-run reproduction is
not available from the current checkpoint format.

## Completed fixed-cost ablation

All three final-budget candidates now have canonical30-episode medium checks on
2029 and2030, plus10 simple turns. The old partial result48/60 at21,278 steps is
superseded by the completed30k pilot, without rewriting its historical report.

| Candidate | Medium2029 | Medium2030 | Pooled safe arrivals | Collisions | Lane failures | Simple |
|---|---:|---:|---:|---:|---:|---:|
| Original60k reference | 26/30 | 25/30 | 51/60 | 9 | 0 | 10/10 |
| Control +30k | 23/30 | 28/30 | 51/60 | 8 | 1 | 10/10 |
| Fixed collision cost300 +30k | 23/30 | 28/30 | 51/60 | 9 | 0 | 10/10 |

On the additional source-selection medium2028 check, reference/control/fixed300
gave22/30,27/30 and27/30 safe arrivals. Over all three selection streams, control
and fixed300 both had78/90, but control had10 collisions versus12. The source
ranking therefore chose control; overnight training used fixed cost0 throughout.
**The fixed penalty corrected accounting but did not demonstrate better driving.**
Increasing it again is not supported by this comparison.

## Reliability, limitations and next action

Two native CARLA client failures recovered automatically: one block2 evaluation
and one block3 training invocation. Retries preserved completed checks and
resumed the trainer from its periodic checkpoint. The optional CPU telemetry
repair did not abort the run again. Recorded code/config/patch hashes all match
the evaluated implementation at this review.

Evaluation seeds reset Python/NumPy/Torch RNG once at evaluator startup; the
Traffic Manager seed remains2024. The printed per-episode seed label increments
without actually reseeding per episode. Thus streams3031/4031 are separate RNG
streams, **not guaranteed new, paired or independent traffic scenarios**. This
also limits claims based on pooled90-episode selection results and small final
differences. No seed or benchmark behavior was changed in this assessment.

Recommended next action: preserve and re-evaluate the90k candidate on more
predeclared streams before spending another night on training. Investigate the
147.5k–150k change with actions, actor/value targets and AMP update telemetry.
If more training is approved, use a short bounded continuation from the good
model, more frequent retention checks and immediate rollback on collapse; test
reduced actor/critic learning rate as a controlled hypothesis rather than another
reward sweep. Save scaler/RNG state and replay provenance before relying on
resume for that comparison. Defer hard-task fine-tuning until stability and
repeatability are established. Multi-hour training or benchmark/reward changes
still require user authorization. No new training was started in this review.

## Evidence and validation

- `reports/overnight_bev_audit_20261008.json`: tracked counts and hashes; validates
  **36 evaluation sets /900 recorded episodes** against summaries, requested and
  manifested seeds/tasks/checkpoints and canonical reward requests. Checks ten
  checkpoint files, including aliases, for finite weights and optimizer tensors.
- `outputs/overnight_bev_20261006/{results.json,source_selection.json,best_validation.json,summary.md,status.txt}`:
  original stage decisions, completed results and immutable model snapshots.
- `outputs/overnight_bev_20261006/assessment_20261008.png|pdf`: selection/final
  results plot. Large models, replay and plots remain Git-ignored.
- `outputs/collision_cost_ablation_queue/comparison_summary.md`: completed pilot
  results; `outputs/diagnostics/overnight_bev_20261006/eval.jsonl`: collapse timing.
- CPU test suite: **166 passed, one GPU-only test skipped**. Changed code lint,
  diff whitespace and reverse-apply checks for both vendor patches passed.

Reproduce the read-only audit and figure:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/auto-thinker-matplotlib \
  .venv/bin/python scripts/assess_overnight_bev.py \
  --audit reports/overnight_bev_audit_20261008.json \
  --plot outputs/overnight_bev_20261006/assessment_20261008.png
```
