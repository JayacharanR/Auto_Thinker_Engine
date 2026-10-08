# Overnight traffic curriculum assessment — 2026-10-05

The medium-first curriculum did not solve traffic. CNN showed a modest measured
gain; both frozen-video arms scored worse after fine-tuning. No arm passed the
promotion gate, so no hard-stage training occurred. Keep the original successful
checkpoints; these candidates do not replace them as the project's reference agents.

## Completed run and evaluation

Controller: 01:40:30–06:10:27 IST (4 h 30 min, including waiting). Active schedule
began at 01:48:23. Each arm completed 30,000 medium-task steps, including 2,500
random prefill steps; 27,500 steps were collected after prefill. No early stops.
One custom-JEPA CARLA failure (exit 250) recovered automatically. All nine
baseline/validation/retention evaluations completed. Host-level anchored process
inspection confirmed no matching queue, trainer, evaluator or CARLA process during
this morning's review. The watcher reports 100%, including intentionally skipped stages.

Destination-reached success, preserving the existing benchmark definition:

| Arm | Original medium, 20 episodes | Selected medium, 30 episodes | Selected collisions | Simple retention, 10 episodes | Selected step |
|---|---:|---:|---:|---:|---:|
| CNN | 3/20 (15%) | 11/30 (36.7%) | 19/30 (63.3%) | 10/10 | 30,000 |
| Custom JEPA | 5/20 (25%) | 4/30 (13.3%) | 26/30 (86.7%) | 10/10 | 27,500 |
| V-JEPA2 | 13/20 (65%) | 4/30 (13.3%) | 26/30 (86.7%) | 8/10 | 7,500 |

There were no out-of-lane terminations. V-JEPA2's two simple-task failures were
timeouts, without collisions. Promotion required medium success >=80%, collision
<=20%, and simple success >=90%, collision <=10%; all failed medium, and V-JEPA2
also failed retention. Validation seed 2026; unchanged Traffic Manager seed 2024.

For equal episode counts, the first 20 selected-policy validation episodes yielded
CNN 6/20, custom JEPA 3/20 and V-JEPA2 3/20. This preserves the observed direction
of change. These are not paired scenarios: policy-dependent episode lengths and
random-number consumption change subsequent traffic. One training seed and one
validation stream cannot establish a reliable improvement or causal encoder effect.

## Learning histories and integrity

Training evaluations contained ten episodes each. Peak success: CNN 40%, custom
JEPA 30%, V-JEPA2 50%. Final 30k evaluations: 40%, 10%, 30%, respectively. No arm
crossed 80%. Last-10k training episode outcomes:

| Arm | Successes | Collisions | Episodes |
|---|---:|---:|---:|
| CNN | 20 (29.9%) | 47 (70.1%) | 67 |
| Custom JEPA | 11 (15.3%) | 61 (84.7%) | 72 |
| V-JEPA2 | 8 (10.4%) | 69 (89.6%) | 77 |

CPU checkpoint inspection confirmed best.pt steps match best_eval.json for all
arms; latest.pt is 30k for all. V-JEPA2 validation used the selected 7.5k snapshot,
not the final 30k snapshot. Final model tensors are all finite. Losses logged at
sampled intervals are finite, but NaN/Inf gradient-norm aggregates recur after
startup. Actual Adam step counts versus 13,750 attempted updates:

| Arm | World model | Actor | Value |
|---|---:|---:|---:|
| CNN | 13,740 | 13,748 | 13,747 |
| Custom JEPA | 13,735 | 13,747 | 13,748 |
| V-JEPA2 | 13,739 | 13,747 | 13,747 |

The optimizer uses unscale, clipping, scaler.step and scaler.update. GradScaler
skips optimizer updates when gradients contain NaN/Inf, as documented in
[PyTorch's AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
The observed step deficits are <=0.11%; this audit does not support widespread
numerical collapse as the cause of poor traffic performance. Logged aggregates
cannot reconstruct each overflow. Custom-JEPA metrics.json covers only its final
resumed invocation (1,250 reported updates, ~425 seconds); queue timestamps and
saved Adam state establish the full run. This is the previously recorded resume
accounting limitation.

## Metric caveat

V-JEPA2's original medium baseline includes one episode with BOTH destination
reached and collision. Thus 13/20 destination arrivals means 12/20 (60%)
collision-free arrivals. Its selected 7.5k training evaluation likewise reports
50% success and 60% collision, with one overlapping event. Independent flags
and destination-priority termination labeling cause this, not a missing episode.
All selected medium validation successes are collision-free. Report both flags;
do not quietly redefine historical success or selection. A collision-free success
metric and safer selection criterion would require an agreed protocol change.

## Recommended next experiment

Run one bounded CNN BEV diagnostic on the SAME medium traffic, reward, actions
and train ratio, initialized from the existing solved BEV checkpoint:
`outputs/logs/cnn_bev_tp01_seed42/latest.pt`. Use a separate diagnostic output
directory, a 30k-step cap, preserved checkpoints and 30-episode validation.
Observed CNN throughput suggests about 80 minutes of training plus evaluations,
but BEV throughput is not measured here. This is a proposed observation/control
diagnostic, requiring agreement before launch; it is not a replacement benchmark.

BEV exposes nearby background vehicles and their planned paths; the current
camera-route vector contains ego speed/route only. The traffic actors explicitly
ignore other vehicles and lights, so the ego must find gaps rather than expect
cooperative traffic. Saved first-episode failure videos show continued motion
into the traffic encounter; this limited sample does not prove the cause.

If BEV reaches the gate, prioritize traffic visibility and spatial/motion features
in the camera pipelines; any later actor-state augmentation must be a separately
labeled diagnostic. If BEV also fails, inspect reward decomposition at impact,
braking/exploration, and the world model's collision prediction before extending
training. The collision penalty scales with post-step ego speed and the reward
encourages forward progress; their effect on yielding is an open hypothesis,
not an established root cause. Leave the hard benchmark and reward unchanged
while diagnosing. Do not run another unchanged three-arm overnight sweep.

## Evidence

- Queue results: `outputs/overnight_traffic_queue/results.json` and `status.txt`.
- Complete episode metrics: `outputs/overnight_traffic_queue/evaluations/`.
- Histories/checkpoints: `outputs/comparison/traffic_curriculum_v1/medium/`.
- Derived audit: `outputs/overnight_traffic_queue/assessment_audit.json`.
- Plot: `outputs/overnight_traffic_queue/assessment_curves.png`.
- Failure video frame strips: `outputs/overnight_traffic_queue/validation_failure_*.jpg`.
- Observation, reward and traffic behavior: `src/dreamer/carla_wrappers.py`,
  CarDreamer `configs/common.yaml`, `carla_wpt_env.py`, and
  `toolkit/carla_manager/world_manager.py`.

Assessment generated from existing complete evaluations; no additional simulator
runs, training runs, reward changes or benchmark changes were made in this review.
