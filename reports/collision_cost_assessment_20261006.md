# Collision-cost pilot assessment — 2026-10-06

**The experiment is unfinished.** The unchanged-reward control completed its 30k pilot budget, but the fixed-cost variant stopped at **21,278/30,000 steps** after the controller's optional `sensors` command timed out. Its checkpoint and optimizers were saved. The control shows no pooled success gain over the starting policy. Short evaluations of the saved partial fixed-cost candidate show no demonstrated driving improvement either; unequal training budgets prevent a final ablation conclusion.

## Canonical evaluation results

All candidates were evaluated on medium streams 2029 and 2030, 30 episodes each, and simple retention on 2029, ten episodes. Fixed cost was explicitly restored to zero and the time cost remained 0.1. No shield was used. Collision-free arrivals below are supplementary joint-outcome counts; the historical benchmark's independent destination and collision flags are unchanged.

| Candidate | Pilot steps | Collision-free arrivals, 2029 | Collision-free arrivals, 2030 | Pooled safe arrivals | Collisions | Lane failures | Simple retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Starting policy, 60k source | 0 | 26/30 | 25/30 | **51/60 (85%)** | 9/60 | 0/60 | 10/10 |
| Unchanged-reward control | 30,000 | 23/30 | 28/30 | **51/60 (85%)** | 8/60 | 1/60 | 10/10 |
| Fixed cost 300, **partial** | 21,278 | 24/30 | 24/30 | **48/60 (80%)** | 11/60 | 1/60 | 10/10 |

No timeouts occurred. The partial variant has **49/60 destination arrivals**, but one arrival in stream 2030 also collided; only 48 are collision-free. Its stream 2030 raw arrival/collision counts are 25/30 and 6/30. The stream 2029 counts are 24/30 arrivals, five collisions and one lane departure. This overlap is another reason to inspect joint outcomes rather than treat destination arrival as automatically safe.

Pooled canonical medium return: starting policy **276.6**, control **270.6**, partial variant **247.0**. Mean speeds were 11.86, 11.59 and 10.95 km/h. Empty-turn retention stayed 10/10 for all three, although the learned pilots took more steps on average: reference 150.6, control 164.6, variant 171.3. Thus the policies still turn successfully; traffic safety is the unresolved part. Mean speeds alone do not establish better yielding.

Control performance varied from 76.7% to 93.3% across streams, despite a final ten-episode training evaluation of 100%. The higher final training score did not become a pooled standalone gain. These are one training seed and two evaluation streams, not independent training replicas or guaranteed paired traffic scenes. Nearby evaluation seeds also do not guarantee independent scenarios; simulator traffic and episode lengths affect the stream. Differences of a few episodes should not be treated as decisive.

## The reward change worked mechanically; policy benefit is unproven

Recorded variant traces contain **71 fixed-cost applications**, all exactly **-300**, with no duplicate charge for a recorded episode in its single training invocation. This includes training and training-evaluation events. Subsequent terminal rows can show a small speed component because the fixed event cost was already charged on first detection; inspecting only the final step would miss that cost.

Among **47 completed training episodes after prefill that collided without arriving**, none had positive cumulative return; the largest return was **-145.6**. This supports the intended correction to reward accounting. It does not establish that the policy anticipates traffic better. Its canonical standalone outcomes currently trail the reference and control, and it has 8,722 fewer pilot steps than the control.

Both checkpoint weight sets are finite. Control checkpoint: 30,000 steps, Adam counters world model 13,735 / actor 13,750 / value 13,750. Partial variant: 21,278 steps, counters 9,382 / 9,389 / 9,389. The model skipped few updates; there is no evidence of broad numerical collapse. Control `metrics.json` reports only the last resumed invocation's 3,750 attempted updates and approximately 21 minutes, not its full training cost.

## Why the queue ended

After the user-directed pause, the queue resumed at 21:17:38. The control suffered a native CARLA timeout at 22:15:26, retried from its checkpoint, and finished at 22:37:01; its checks ended 22:41:34. That recovered crash may change collected experience and random trajectory, so even equal nominal budgets are not identical uninterrupted runs.

The fixed-cost trainer ran from 22:41:34 to 23:34:44. A `subprocess.TimeoutExpired` from **`sensors` after ten seconds** escaped `Queue.check_health`. The controller then requested STOP and signaled its own trainer, which saved at 21,278. Last recorded health was normal: GPU 53°C, approximately 7.8 GB VRAM, 20.2 GB available RAM and 85 GB free disk. This was an infrastructure interruption, not a failed learning gate or an overheating stop.

The controller printed its final banner while `results.json` retained `completed: false`; no final comparison summary existed. Its exceptional exit omitted a `FAIL` record for the active variant. Stale SKIP records from the earlier pause also prevented explicit SKIP records in the latest invocation. These reporting defects made the ended queue easier to mistake for a completed experiment.

## Repairs completed during this review

- CPU telemetry timeout or an unavailable `sensors` executable now records a warning and `cpu=?`; GPU, RAM, disk and time-budget guards still execute. GPU safety remains active when the CPU read fails.
- Controller exceptions explicitly record the failed stage. The ablation saves an error in results and marks unattempted stages using only the current invocation's status records.
- The original experiment manifest and original source snapshots remain unchanged. An explicit runtime-recovery manifest records exact original/current code hashes and a reviewable patch. Only queue/controller files may differ; changes to reward/training code, budgets or other experiment fields are refused.
- Recovery preflight verified finite source weights and accepted the recorded runtime repair while leaving the original STOP file and manifest intact. **No further training was launched.**

Validation: **29 targeted tests passed**, including CPU telemetry timeout/unavailable-tool recovery, retained GPU overheating STOP, exception reporting, pause/resume, canonical evaluation and recovery provenance. Lint and diff checks passed. The partial candidate's three short evaluations ended 00:17:29; a host anchored process check afterward found no active trainer/evaluator/controller/CARLA.

## Next step

Finish the existing fixed-cost pilot's **remaining 8,722 steps** with its saved replay and optimizers, using the repaired controller. Expect approximately **25 minutes of training plus five minutes of validation/startup**, based on the measured six steps/s. Reuse reference and completed control checks; retain the fixed 30k budget and original evaluation protocol. This completes the already specified experiment rather than introducing another reward sweep.

After equal-budget results are available, test any promising candidate on the previously difficult medium stream 2028 before hard-task training. That is a known diagnostic stream, not a new held-out set. If the fixed cost still gives no repeatable benefit, retain the 60k reference and investigate failure timing with collision/brake/actor traces before further reward or encoder sweeps. Increasing the penalty again is not justified by these results. Any adoption in the camera comparison must use a common, explicitly versioned reward for all encoders.

Prepared continuation command, with explicit runtime recovery:

```bash
.venv/bin/python scripts/run_collision_cost_ablation.py --max-hours 4 \
  --runtime-recovery outputs/collision_cost_ablation_queue/runtime_recovery_20261006.json
```

Long-run launch should use the existing detached `systemd-inhibit` convention. The command above is documentation; it was not executed as training in this review.

## Evidence

- Original queue: `outputs/collision_cost_ablation_queue/{results.json,status.txt,launcher.log}`; original completed reference/control episode JSONs under `evaluations/`.
- Partial candidate snapshot and short evaluations: `outputs/collision_cost_ablation_assessment_20261006/`, including `results.json`, `assessment_audit.json`, `assessment_curves.png|pdf`, and derivation/controller scripts.
- Training histories/checkpoints/collision traces: `outputs/diagnostics/collision_cost_ablation_v1/{control,fixed300}/`.
- Recorded runtime repair: original queue's `runtime_recovery_20261006.json|patch`, `runtime_protocol.json` and `recovery_snapshot/`. The original `experiment.json` and `source_snapshot/` are preserved.
