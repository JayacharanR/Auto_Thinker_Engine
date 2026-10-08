# BEV medium-traffic continuation assessment — 2026-10-05

The additional 30k steps improved this CNN BEV agent across all three checked medium-traffic evaluation streams. Collision-free arrivals increased from **47/90 (52.2%) at the 30k budget to 67/90 (74.4%) at 60k**; collisions fell from **42/90 (46.7%) to 23/90 (25.6%)**. This run produced useful progress, but medium traffic is still unreliable on the newest stream. The scheduled promotion gate passed; that alone does not establish robust traffic handling.

These are repeated evaluations of **one training seed**, not three independently trained agents. BEV supplies privileged traffic information, so these results do not rank the camera CNN, custom JEPA and V-JEPA2 pipelines.

## Completed run and checkpoint verification

- Controller: `outputs/bev_traffic_continuation_60k_queue`; training directory: `outputs/diagnostics/bev_traffic_medium_seed42`.
- Started 17:21:56 IST; training finished 18:45:41; scheduled validation/retention finished 18:48:22. Total queue duration approximately 86 minutes, including approximately 84 minutes for the training stage. No retries, crashes or early stopping in this continuation.
- Resumed from 30k to 60k adaptation steps with replay/optimizer state. CNN, BEV, discrete actions, medium task, seed 42, train ratio 512, time penalty 0.1; reward unchanged. No hard-task training.
- Current `best.pt` and `latest.pt` both verified at 60,000 steps with finite weights. The archived 30k-budget candidate was selected at 27,500 steps and remains under `history/at_30000/best.pt`.
- Final invocation: 15,000 attempted updates, 5,006.6 seconds, 6.01 environment steps/s; trainer peak VRAM 3,945.3 MB excludes CARLA. These invocation statistics are not the entire resumed experiment's compute cost.
- Checkpoint Adam counters across the full adaptation: world model 28,729; actor 28,747; value 28,746, versus 28,750 attempted updates. There is no evidence of a general numerical collapse.
- All assessment evaluations completed by 19:55:15. Host-level anchored process check then found no active queue, trainer, evaluator or CARLA process. No new training was launched during this assessment.

## Standalone evaluation results

Each medium cell contains **collision-free arrivals / collisions / timeouts**, over 30 episodes. Destination flags are preserved exactly; no arrival/collision overlap occurred in these standalone evaluations.

| Evaluation stream | 30k-budget candidate | 60k candidate |
|---|---:|---:|
| 2026, scheduled validation | 15 / 15 / 0 | **25 / 5 / 0** |
| 2027, additional stream | 20 / 10 / 0 | **25 / 5 / 0** |
| 2028, additional stream | 12 / 17 / 1 | **17 / 13 / 0** |
| Total, 90 episodes | **47 / 42 / 1** | **67 / 23 / 0** |

The gain is **22.2 percentage points** in success and **21.1 points** fewer collisions. Stream 2028 improved too, but its 56.7% success / 43.3% collision rate still fails the medium gate. The scheduled 2026 validation passed the existing gate: at least 80% arrivals, no more than 20% collisions, plus the simple-turn retention requirement. Preserve that historical `gate_passed: true`; do not reinterpret it as evidence that every traffic stream passes.

| Additional task, current 60k candidate | Collision-free arrivals | Collisions | Lane failures |
|---|---:|---:|---:|
| Empty-intersection retention, stream 2026 | **10/10** | 0/10 | 0/10 |
| Hard traffic, zero-shot, stream 2028 | **21/30 (70%)** | 9/30 | 0/30 |

The hard-task result is encouraging transfer, but does not pass an 80% arrival / 20% collision target. Its higher observed success than medium stream 2028 does not establish that the nominal hard task is easier: these are small, policy-dependent traffic samples. No lane failures occurred in any of these checks.

The evaluation seed initializes a stream; simulator timing, traffic-manager settings and differing episode lengths do not guarantee paired identical scenarios across policies. Pooled counts are descriptive. Additional training seeds and more independent scenarios are needed for a generalization claim. Stream 2026 was also used during learning/checkpoint selection and is less independent than the added post-run streams.

## What the agent appears to have learned

Training evaluations remained noisy. The last three were 80%, 80%, 90% arrival success; the final 90% was only nine of ten episodes, whereas standalone medium success was 25/30, 25/30 and 17/30. Training episodes improved modestly: collision-free arrivals were 38/71 in steps 30–40k, 38/68 in 40–50k and 40/67 in 50–60k. These training episodes include exploratory behavior and should not be substituted for standalone evaluation.

In the final ten training-evaluation episodes, episode-average acceleration share decreased from 72.4% at 30k to 68.7% at 60k and coasting increased from 15.9% to 20.9%. Braking share decreased from 11.8% to 10.4%; the data do not support a claim that it simply brakes more overall. The sole collision in the final 60k training evaluation had six brake commands in its final ten steps, unlike the three final-30k collisions, which had none. This small sample suggests changed action timing; it does not demonstrate reliable anticipation or yielding.

## Measured reward limitation

`third_party/CarDreamer/car_dreamer/carla_wpt_env.py:95` computes the collision component as:

```text
r_collision = -30 * abs(ego_speed_m_per_s), if collision
```

Reward is calculated after the physics step. The intrinsic collision penalty approaches zero when the ego is nearly stationary, including cases where a crossing vehicle strikes a braking ego. Increasing only the multiplier cannot remove the zero-speed limit. A crash still terminates the episode and loses future rewards, so this is not a claim that stopping in a collision has no cost at all.

During the 30–60k continuation, **9 of 90 training collision episodes had positive cumulative return**. **Three had terminal total reward above -1**. One ended without arrival at 94% route completion with **+299.685 episode return and -0.855 terminal total reward**. The collision in the final 60k training evaluation earned **+296.515 return**, with terminal total reward **-0.372**. Those terminal values include all reward components, not just `r_collision`.

Most collisions already received a substantial penalty: median terminal total reward was **-161.36**. Thus this limitation is a plausible contributor, not an established explanation for all failures. Positive undiscounted episode return also does not by itself prove that the actor prefers a crash: its learned, discounted value estimates need separate investigation.

The earlier TTC telemetry limitation remains: same-road/same-lane filtering can miss crossing vehicles, and the HUD can show SAFE before a collision. TTC is not an input to this policy or reward, and evaluations have no safety shield. Correcting that display would improve diagnosis without automatically improving policy behavior.

## Recommended next experiment — proposed, not applied

Run a bounded, controlled **fixed collision-cost ablation** before committing to hard-task training or another three-encoder traffic sweep. It tests the measured low-speed penalty limitation while allowing extra training itself to explain improvement.

1. Preserve the current 60k weights and all original reward/evaluation files. Initialize both experimental arms from those same weights, with fresh replay and reset optimizers in separate directories. Fresh replay avoids training the new reward model on old reward labels; resetting both arms makes that reset comparable.
2. Control: existing collision reward. Variant: on a collision, add a fixed cost of 300, giving `r_collision = -300 - 30 * abs(ego_speed_m_per_s)`. Keep all other reward terms, task, BEV input, actions, train ratio and training seed unchanged. The 300 is a pilot value informed by the observed near-300 collision returns; it is not a proven optimum or a safety guarantee.
3. Give each arm **30k additional environment steps**, including identical fresh-replay prefill. Use the final fixed-budget checkpoints for the comparison; do not let different early-stop times or reward-based checkpoint tie-breaks confound the ablation.
4. Before training, verify the new component on controlled zero-speed and moving collision cases, verify that non-collision rewards are unchanged, and check both resolved configs. Record the reward version and vendor patch for reproducibility. Record collision reward components and ego speed when available so subsequent diagnoses can distinguish penalty size from late braking.
5. Evaluate both arms on the **original canonical benchmark reward**, without a shield. Predeclare streams 2029 and 2030, 30 medium episodes each, plus ten simple-retention episodes; report destination, collision-free arrival, collision and timeout counts separately. Preserve historical metric definitions and select no winner by modified training return. Keep the 60k reference available for comparison.
6. Expect roughly **three hours for two sequential pilots plus checks**, based on the completed continuation; fresh-replay overhead makes this approximate. Do not advance automatically to hard training. Look for a repeatable collision reduction without stalling or losing empty-turn skill, then replicate a promising result with another training seed.

This changes the training reward and needs user approval. **No reward code, benchmark definition or new training run has been changed or started in this assessment.** If successful, adopting the reward in the camera encoder experiment would require a clearly versioned, common protocol for all arms. BEV improvement alone cannot establish a pretraining benefit.

## Evidence and artifacts

- Scheduled checks: `outputs/bev_traffic_continuation_60k_queue/results.json` and its `evaluations/` directory.
- New medium and hard stream: `outputs/bev_60k_assessment_seed2028/results.json`.
- Cross-budget completion: `outputs/bev_medium_budget_comparison_20261005/results.json`; earlier stream 2027: `outputs/bev_traffic_holdout_20261005_seed2027/results.json`.
- Derived counts and provenance: `outputs/bev_traffic_continuation_60k_queue/completed_assessment_audit.json`.
- Standalone plot: `outputs/bev_traffic_continuation_60k_queue/completed_assessment_curves.png` and `.pdf`.
- Episode/action audit: `behavior_audit_30k_vs60k.json`; all matched training collisions: `collision_reward_audit.json`, in the same queue directory. Audit matches used episode timestamps and recorded NPZ rewards/actions.
- The earlier `reports/bev_traffic_assessment_20261005.md` remains the historical 30k assessment. This report adds the third medium stream and the completed continuation; it does not replace the earlier two-stream result.
