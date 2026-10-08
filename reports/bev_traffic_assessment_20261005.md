# Completed CNN BEV traffic diagnostic — 2026-10-05

There is measured improvement in medium traffic, but reliable yielding is still
unsolved. The selected BEV agent improves on its original policy in both completed
evaluation streams. It retains the empty-intersection turn and remains below the
80% traffic-success / <=20% collision promotion gate.

## Results

| Evaluation stream | Original successes | Selected successes | Original collisions | Selected collisions |
|---|---:|---:|---:|---:|
| 2026, scheduled validation | 11/30 (36.7%) | 15/30 (50.0%) | 19/30 | 15/30 |
| 2027, new post-run check | 9/30 (30.0%) | 20/30 (66.7%) | 21/30 | 10/30 |
| Pooled episode counts | 20/60 (33.3%) | 35/60 (58.3%) | 40/60 (66.7%) | 25/60 (41.7%) |

All successes in these evaluations are collision-free. No lane violations or
timeouts occurred in the medium evaluations. Simple-turn retention: 10/10
successes, zero collisions/lane violations. Mean traffic return, original ->
selected: stream 2026, 65.7 -> 109.3; stream 2027, 28.7 -> 198.6.

The new evaluation ran from 16:52:43 to 16:56:10, without training, reward changes
or a safety shield. It checked the original and preselected 27.5k checkpoint on
30 episodes each. It did not select a new candidate from its own results.

These are two evaluation streams for ONE training seed (42), with Traffic Manager
seed 2024 unchanged. Scenarios are not paired: policy-dependent episode lengths
and random-number consumption change traffic. Pooled counts are descriptive;
they are not two independent training replications. The observed gain does not
establish statistical significance or broad generalization.

## Training and recovery

The diagnostic completed 30k adaptation steps, including 2.5k random prefill.
Started 11:31:46, paused by manual STOP at 12:36:34 with checkpoint at 24,969,
resumed 16:24:02, completed training 16:39:28 and final checks 16:42:09.
Active queue time excluding the pause was approximately 83 minutes; training
stage time including simulator startup and in-training evaluation was ~79 minutes.
No early stop or hard-stage training occurred. Final host process checks found
no matching training or CARLA process before the extra evaluation.

Training evaluation success at 5k–30k, every 2.5k:
40%, 0%, 30%, 30%, 30%, 40%, 30%, 40%, 40%, 70%, 70%.
The best candidate was saved at 27.5k: both final evaluations had 70% success /
30% collisions, and 27.5k won the return tie-break. CPU checkpoint inspection
verified best.pt=27,500 and latest.pt=30,000. Final model weights are finite;
saved Adam state has 13,739 model updates and 13,749 actor/value updates against
13,750 attempted updates, with no evidence of widespread numerical collapse.
metrics.json reports only the final resumed invocation (2,516 attempted updates,
~908 seconds), not the full run; use checkpoint counters and queue timestamps.

Sampled collision-free training arrivals were 38/126 (30.2%) at steps >2.5k through 20k,
18/35 (51.4%) at >20k through 25k, and 12/35 (34.3%) at >25k through 30k.
Training episodes therefore do not show a smooth improvement; the two late
evaluation increases and the new stream check are encouraging but insufficient
to declare convergence.

## What remains wrong

Final ten training-evaluation episodes contain three collisions. In all three,
the discrete action trace has zero brake commands in the last ten steps (one
second) before impact; acceleration occupies ~85–87% of each failed episode.
This does not mean the agent never brakes: it brakes in other episodes, and the
preceding evaluation contains collision episodes with late braking. It indicates
that learned timing is unreliable, rather than proving a particular root cause.
The first selected-checkpoint validation video also shows continued motion into
a collision. One video is illustrative, not a representative behavior sample.

The current TTC diagnostic is unsuitable for crossing traffic. In CarDreamer
`toolkit/carla_manager/utils.py`, `is_vehicle_ahead` rejects every target with a
different road_id OR lane_id before testing proximity; it also requires targets
ahead of the ego. `get_ttc` returns zero for no eligible target, and the evaluator
can display "SAFE" while recording no finite positive TTC. The remaining
distance / relative-speed formula is not a general path-intersection predictor.
A module-isolated reproduction includes a same-road lead vehicle and excludes
a nearby vehicle on another road. The failure-video HUD illustrates the problem.

This TTC is telemetry and a disabled shield input in these runs. The agent's BEV
observation and current reward do not consume it, so repairing TTC alone will
not change the learned policy or explain all collisions. Current collision counts
remain valid. A diagnostic replacement should consider crossing paths/swept
vehicle boxes and distinguish unavailable data from no predicted conflict.

## Recommended next step

Continue ONLY the current BEV diagnostic from latest.pt=30k to a total 60k, using
the saved optimizer and replay, unchanged reward/actions/traffic and preserved
best.pt. Additional 30k steps should take roughly 1.5 hours at observed throughput,
plus validation; cap it and review at 60k. Require separate traffic validation
>=80% success / <=20% collisions and simple retention >=90% / <=10% collisions
before considering hard traffic. The continuation is a proposal, not launched.

The rationale is specific: the BEV policy improved on two streams and reached
70% only near the end of its current budget. The original
[CarDreamer paper, section 5.1](https://arxiv.org/html/2405.09111v2#S5.SS1)
reports about 150k–200k steps for denser traffic with full observability and
intention sharing. Those settings and implementation differ from this project;
their results are context, not a promised target or proof that a longer run wins.

Repair crossing-traffic risk telemetry and add traces of actions, nearby actors,
speed and reward components to diagnose remaining failures. If the bounded
continuation stalls, compare observed collision outcomes with the world model's
predicted reward/continuation under braking versus accelerating before selecting
a reward, exploration or imagination-horizon change. Current imag_horizon=15
means a 1.5 s explicit rollout, with critic bootstrapping beyond it; it is a
candidate to test, not a demonstrated cause of short-sighted behavior.

Once a reliable BEV traffic baseline is established, return to the camera encoder
question with controlled observation coverage and spatial/motion features. BEV
includes privileged nearby-vehicle information and intended paths; camera_route
contains the ego route/speed and front camera. Pretraining cannot directly supply
an actor observation that the camera did not capture. A wider-camera experiment
or training-only BEV auxiliary prediction would be a separately agreed protocol,
applied fairly across arms. Do not equate the BEV result with an encoder-pretraining
effect or replace the current front-camera benchmark silently.

## Saved evidence

- `outputs/bev_traffic_diagnostic_queue/results.json`, `status.txt`, evaluations.
- `outputs/bev_traffic_holdout_20261005_seed2027/`: complete new-stream metrics/logs.
- `outputs/bev_traffic_diagnostic_queue/completed_assessment_audit.json` and plot
  `completed_assessment_curves.png`.
- `outputs/bev_traffic_diagnostic_queue/action_audit_last20.json`,
  `validation_failure_strip.jpg`, `ttc_filter_audit.json`.
- `outputs/diagnostics/bev_traffic_medium_seed42/`: checkpoints, eval/episode histories.

No new training, reward or benchmark changes were made during this assessment.
