# Autonomous Driving Implementation Plan

**Project:** Auto Thinker Engine  
**Last reviewed:** 2026-09-10  
**Primary target:** research-grade closed-loop autonomous driving in CARLA 0.9.15  
**Secondary target:** a later, separately safety-engineered real-vehicle pilot

This document is the implementation roadmap for reaching a defensible
autonomous-driving result. It is deliberately stricter than “the training
script runs”: a driving agent is not autonomous until it can control a vehicle
over multiple routes and conditions, avoid collisions, obey traffic rules,
recover from disturbances, and produce reproducible evaluation evidence.

## 1. Direct answer: how much is left?

The project has completed the environment bring-up and the first end-to-end
Dreamer/CARLA integration check. It has **not** yet demonstrated autonomous
driving as a final capability.

These are planning estimates, not code-coverage measurements:

| Capability | Current state | Approximate completion | Work remaining |
|---|---|---:|---|
| CARLA runtime and GPU launch | CARLA 0.9.15, A4000 container, Gate 0 smoke test passed | 90% | Reproducible fresh-install setup and traffic-task validation |
| CNN Dreamer integration | 10,000-step run completed and checkpoint written | 60% | 500,000-step baseline, resume, robust evaluation and metric artifacts |
| JEPA pretraining | Model/data/training code exists; no validated checkpoint | 30% | Dataset verification, full training, collapse checks and linear probe |
| Encoder transfer comparison | Runner supports arms, but comparison is not validated | 20% | Custom-JEPA checkpoint, V-JEPA2 feasibility, matched seeds and analysis |
| Multi-route simulated driving | One simple right-turn task is the validated path | 15% | Routes, maps, traffic, weather, disturbances and generalization tests |
| Safety supervisor | No independent action shield or emergency fallback | 5% | TTC/collision/lane/traffic-rule checks and safe override policy |
| Production-quality evaluation | Basic episode metrics exist; final JSON contract is incomplete | 35% | Stable manifests, confidence intervals, replay, latency and failure taxonomy |
| Real-vehicle autonomy | No vehicle interface, sensor stack, calibration or safety case | 0–5% | A separate hardware, testing and regulatory program |

For the **research-grade CARLA target**, approximately **8–15 engineering
weeks** remain for one person working full-time, assuming the existing GPU,
container and simulator remain available. Training wall-clock time is separate
and may add days or weeks on a single RTX A4000.

For a **real vehicle**, the remaining work is not a small extension of this
repository. A controlled-track prototype would typically require **months** of
additional hardware, integration, validation and safety work. Public-road
deployment requires a formal safety case, redundancy, compliance and regulatory
approval; this repository must not be treated as road-ready software.

## 2. Define the final state before implementing it

### 2.1 Research-grade CARLA autonomy — the immediate project goal

The final research result should satisfy all of the following:

1. The agent receives only the declared onboard observations and route goal.
2. It selects bounded steering and throttle/brake actions in closed loop.
3. It completes routes on unseen seeds, not just the training episode.
4. It avoids collisions, stays within the drivable lane and obeys applicable
   stop signs and traffic lights.
5. It operates under multiple maps, weather settings, traffic densities and
   controlled disturbances.
6. An independent safety supervisor can override unsafe policy actions.
7. Every run records configuration, seed, checkpoint, route, actions, latency,
   failures and metrics.
8. Results are reproducible across at least three random seeds.

Proposed release thresholds, to be frozen before the final comparison:

| Metric | Suggested CARLA release threshold |
|---|---:|
| Route completion | ≥ 90% on the simple-task validation suite |
| Collision rate | ≤ 5% of episodes and zero tolerated collision in the safety suite |
| Off-road / lane departure | ≤ 5% of episodes |
| Traffic-rule violations | ≤ 1% on applicable scenarios |
| Safety-supervisor emergency interventions | Report separately; trend must decrease during training |
| Evaluation coverage | ≥ 100 episodes per core task and ≥ 3 seeds |
| Control loop | Stable at the simulator tick rate with no missed-deadline failures |
| Process reliability | No orphan CARLA process, hang, NaN checkpoint or unrecoverable run |

These thresholds are engineering gates, not claims that the current reward
function already measures every item. The metric and sensor contracts must be
implemented before the thresholds are used.

### 2.2 Real-vehicle autonomy — a separate later program

The repository currently has no interface to a physical vehicle. A real-vehicle
target additionally requires:

- a defined sensor suite, time synchronization and calibrated extrinsics;
- vehicle-state estimation and a validated drive-by-wire interface;
- redundant braking, steering and compute watchdogs;
- an independent emergency-stop and minimum-risk maneuver;
- hardware-in-the-loop and closed-course testing with a safety driver;
- data governance, cybersecurity, hazard analysis and regulatory review.

The CARLA result is evidence for simulation research only. It is not evidence
that a physical vehicle is safe to operate.

## 3. Target system architecture

```text
Camera / optional BEV / vehicle state / route goal
                 │
                 ▼
        Timestamp + calibration + normalization
                 │
                 ▼
    Visual encoder + temporal frame buffer / feature state
                 │
                 ▼
       Dreamer RSSM world model and latent state
                 │
                 ├── reward / route-progress / termination heads
                 ├── uncertainty and OOD monitors
                 ▼
          Policy or hierarchical local planner
                 │
                 ▼
     Independent safety supervisor / action shield
                 │
                 ▼
     Rate-limited steering + throttle/brake controller
                 │
                 ▼
          CARLA vehicle or later vehicle adapter
```

The current project covers much of the encoder/RSSM/policy branch and part of
the CARLA adapter. It does not yet cover the independent safety branch,
uncertainty monitoring, broad scenario evaluation or a physical vehicle
adapter.

## 4. Critical path and dependencies

```text
P0 Contract and test hardening
          │
          ▼
P1 CNN baseline ───────────────┐
          │                    │
          ▼                    ▼
P2 JEPA pretraining       P4 Scenario + safety stack
          │                    │
          ▼                    │
P3 Encoder comparison ────────┘
          │
          ▼
P5 Multi-route CARLA release evaluation
          │
          ├── Research result complete
          ▼
P6 Hardware-in-the-loop / sim-to-real (optional, separate safety program)
```

P1 is the minimum learning baseline and can proceed without JEPA. P2 can run
in parallel once the comma2k19 data path is verified. P3 must wait for a real
custom-JEPA checkpoint. P4 should begin early, but its final thresholds depend
on P1 and the route/evaluation contract.

## 5. Phase-by-phase implementation plan

### P0 — Freeze the contract and make runs auditable

**Goal:** remove ambiguity before spending long GPU runs.

Implementation:

- Make `configs/experiment_contract.yaml` the single source of truth for task,
  action type, image resolution, episode limit, training budget and metrics.
- Make the active runner and YAML agree on continuous actions, observation keys,
  image size, prefill, train ratio and evaluation count.
- Fail fast when a requested checkpoint, dataset, CARLA map or model dependency
  is missing. The custom-JEPA arm must not silently train from random weights.
- Add a run manifest containing Git/submodule revisions, config hash, seed,
  CARLA version, map, Docker image, GPU, checkpoint path and environment
  variables.
- Add a stable `metrics.json` schema with train/eval split, episode records,
  success, collision, route completion, lane departure, traffic violations,
  wall-clock time, step throughput, action latency and peak VRAM.
- Add checkpoint resume tests, including optimizer state and replay metadata.
- Remove or clearly mark the legacy `scripts/train_phase3_arm.py` path so two
  incompatible Phase 3 implementations are not treated as equivalent.

**Exit criteria:** a 100-step run can be reproduced from a manifest, missing
dependencies fail before CARLA starts, and the run produces a valid metrics
record even when an episode terminates early.

### P1 — Complete the CNN Dreamer baseline

**Goal:** establish a competent closed-loop baseline before comparing encoders.

Implementation:

- Run the existing validated runner with normal prefill and the intended
  `500,000` environment-step budget; the previous 10,000-step run was only an
  integration check and used `CARLA_PRETRAIN=1`.
- Run at least seeds `42`, `123` and `456` with identical environment and
  reward settings.
- Implement true resume from `latest.pt`, including Dreamer optimizer state,
  replay state or a documented replay warm-start policy.
- Replace the current minimal final metrics dictionary with the P0 JSON
  contract. Preserve the evaluation return/length currently visible in logs.
- Make optional video logging non-fatal and avoid requiring `moviepy` when video
  prediction is disabled.
- Evaluate at fixed checkpoints rather than only at the end so learning curves
  and steps-to-threshold can be computed.
- Verify that success means route completion and is not accidentally inferred
  from truncation alone.

**Exit criteria:** the CNN baseline completes all three seeds, produces valid
  checkpoints and metrics, and meets the simple right-turn gate on held-out
  seeds. If it does not, improve the environment/reward/control contract
  before starting a representation comparison.

### P2 — Run and validate JEPA pretraining

**Goal:** produce a trustworthy custom visual representation for transfer.

Implementation:

- Verify the comma2k19 download format, segment count, timestamps, frame rate,
  steering units and speed units. Start with one development chunk before the
  full dataset.
- Add a dataset manifest and deterministic train/validation split by drive
  segment, not by adjacent frames from the same segment.
- Run the ViT-Small JEPA configuration with EMA target updates, masked-token
  loss, telemetry alignment and mixed precision.
- Save `latest`, `best` and resumable optimizer checkpoints; retain the config
  and data manifest beside each checkpoint.
- Monitor representation variance, norm, masked/unmasked leakage and NaN/Inf
  rates. Stop or flag collapse rather than accepting a low loss blindly.
- Run the steering linear probe against both the trained encoder and a
  random-initialized encoder using exactly the same data and probe budget.
- Run the representation visualization only after the probe and checkpoint
  reload tests pass.

**Exit criteria:** a checkpoint reloads cleanly, no information leak is found,
representations do not collapse, and the trained probe beats the random control
by a predefined margin on held-out drives. A JEPA loss curve alone is not
enough evidence.

### P3 — Validate the controlled encoder comparison

**Goal:** determine whether visual pretraining improves driving, not merely
whether it changes the loss.

Implementation:

- Use one active Dreamer runner and one environment/reward/action contract for
  CNN, custom-JEPA and V-JEPA2 arms.
- Require the Phase 2 checkpoint for `custom_jepa`; record whether the encoder
  is frozen or fine-tuned.
- Validate temporal frame stacking and warm-up behavior at episode reset. The
  first four frames must have an explicitly documented padding policy.
- Add a dedicated V-JEPA2 memory and latency probe before full training. The
  ViT-L model is likely tight alongside CARLA on a 16 GB A4000; defer it or use
  a smaller approved model if the measured budget is unsafe.
- Match seeds, route order, action semantics, reward, training steps,
  evaluation episodes, checkpoint cadence and data logging across arms.
- Report mean, standard deviation and confidence intervals across seeds, plus
  parameter count, update time, peak VRAM and steps-to-threshold.
- Separate representation quality, policy learning quality and environment
  reliability in the analysis.

**Exit criteria:** all feasible arms complete the same smoke, short-training
and evaluation gates. A missing checkpoint, OOM or unvalidated arm is reported
as incomplete rather than omitted from the comparison.

### P4 — Expand from one right turn to an autonomy scenario suite

**Goal:** test driving behavior rather than memorization of one route.

Implementation:

- Add deterministic route suites for straight driving, left/right turns,
  intersections, lane following, stop signs, traffic lights, lane merge,
  following and obstacle interaction using the existing CarDreamer tasks where
  appropriate.
- Validate Traffic Manager only for tasks that require it. Keep the current
  bypass for tasks without traffic and isolate native CARLA failures by task.
- Add multiple maps, weather, time-of-day, traffic density and spawn seeds.
- Add controlled perturbations: initial pose offsets, sensor noise, brief
  frame drops, action delay and mild route deviations.
- Add sensor/event instrumentation for collision type, off-road state, lane
  departure, red-light/stop-sign violation, route completion, stuck state and
  timeout reason.
- Ensure train routes and evaluation routes are disjoint by route seed and,
  where possible, by map/scenario family.

**Exit criteria:** the selected policy is evaluated on a frozen scenario matrix
with per-scenario results. It must not pass by succeeding only on the original
right-turn task.

### P5 — Implement the independent safety and control layer

**Goal:** prevent unsafe policy actions and make failures diagnosable.

Implementation:

- Add action bounds, acceleration/brake limits, steering-rate and jerk limits.
- Add a collision/near-collision monitor using CARLA sensors and a
  conservative emergency-brake action.
- Add lane-boundary and drivable-area checks, with a safe slow/stop fallback.
- Add route-aware stop-sign and traffic-light checks independent of the learned
  reward.
- Add a watchdog for stale observations, missing policy output, NaNs, excessive
  inference latency and CARLA tick failure.
- Record every supervisor intervention and distinguish policy failure from
  safety override success.
- Add uncertainty/OOD signals from ensemble disagreement, RSSM prediction
  error, or a simpler calibrated state/action monitor before using them for
  hard overrides.
- Test the shield against adversarial unsafe actions and ensure it cannot be
  disabled accidentally by a model checkpoint.

**Exit criteria:** unsafe scripted actions are overridden reliably, emergency
  stop is bounded and repeatable, supervisor latency fits the control deadline,
  and all interventions appear in the evaluation artifact.

### P6 — Freeze and publish the CARLA autonomy result

**Goal:** create a reproducible research release.

Implementation:

- Freeze code, submodule revisions, Docker image, CARLA build and configs.
- Run the final scenario matrix on at least three seeds with no manual changes.
- Save checkpoint, manifest, per-episode JSONL, aggregate JSON, videos for
  failures, route traces and system telemetry.
- Generate a report with confidence intervals, failure examples, throughput,
  peak VRAM, interventions and known limitations.
- Add regression tests for every previously fixed CARLA and Dreamer failure.
- Verify fresh-clone setup using the runbook and a clean output directory.

**Exit criteria:** another operator can reproduce the result from the frozen
  manifest and obtain the same class of behavior without editing source files.

### P7 — Optional sim-to-real / physical vehicle program

**Goal:** only after the CARLA target is stable, investigate controlled physical
  deployment.

This phase is intentionally separate from the current research milestone.

1. Define the vehicle, compute platform, sensor suite, drive-by-wire API and
   emergency controls.
2. Build a hardware-in-the-loop adapter with recorded-sensor replay before
   connecting actuators.
3. Calibrate cameras, clocks, coordinate frames and vehicle dynamics; quantify
   end-to-end latency.
4. Collect real driving data and measure the CARLA-to-real observation gap.
5. Add domain adaptation or fine-tuning only with an offline validation gate.
6. Test in simulation, then static hardware, then a closed course with a
   safety driver and independent emergency stop.
7. Run hazard analysis, fault injection, cybersecurity review and required
   regulatory/compliance processes before any public-road use.

**Exit criteria:** a qualified safety process, not just a successful neural
network run, authorizes each increase in test scope.

## 6. File-level implementation map

| Area | Files to change or verify | Required result |
|---|---|---|
| Contract/config | `configs/experiment_contract.yaml`, `configs/phase1_dreamer_baseline.yaml`, `configs/phase3_transfer_arms.yaml` | One consistent action, observation, reward, budget and metric contract |
| Active training | `scripts/train_cardreamer.py` | Resume, fail-fast checkpoints, exact metrics, evaluation and manifests |
| CARLA lifecycle | `jobs/slurm_run.sh`, `scripts/smoke_test_carla.py`, CarDreamer world/monitor files | Reliable startup, traffic-task isolation and clean shutdown |
| Observation/action | `scripts/train_cardreamer.py`, `src/dreamer/cardreamer_encoder_hook.py` | Stable image/action/terminal shapes for online and replay paths |
| JEPA data/training | `src/data/`, `scripts/train_phase2_jepa.py`, `src/jepa/` | Verified dataset, resumable pretraining and collapse evidence |
| Transfer arms | `src/dreamer/encoder_adapter.py`, active runner, V-JEPA2 dependency path | Correct frozen/trainable behavior and measured memory/latency |
| Metrics | `src/eval/metrics.py`, `src/utils/logging_utils.py` | Per-episode JSONL, aggregate JSON and confidence intervals |
| Scenarios | `third_party/CarDreamer/car_dreamer/`, new scenario/config modules | Frozen multi-route/multi-condition evaluation matrix |
| Safety | New `src/safety/` package and runner integration | Action shield, watchdog, emergency stop and intervention logs |
| Evaluation | `src/eval/`, new report/aggregation scripts | Reproducible gates, failure taxonomy and plots |
| Deployment | New vehicle adapter outside the CARLA-only path | Hardware-in-loop and safety-controlled physical interface |

## 7. Test and acceptance matrix

| Gate | Test | Evidence required |
|---|---|---|
| G0 Runtime | Doctor + 100-step CARLA smoke | Imports, CUDA, CARLA, throughput, clean process tree |
| G1 Interface | Reset, action, terminal and replay-shape test | Correct spaces/keys/dtypes for both online and replay paths |
| G2 Learning | Prefill plus repeated Dreamer updates | No crash, NaN, gradient disablement or stale-state failure |
| G3 Baseline | 500k CNN run on three seeds | Checkpoints, curves, metrics and held-out route success |
| G4 JEPA | Pretrain + reload + random probe control | No collapse/leakage and positive probe delta |
| G5 Transfer | Short run for every encoder arm | Same contract, memory budget and checkpoint provenance |
| G6 Scenario | Frozen route/weather/traffic matrix | Per-scenario success, collisions, violations and timeouts |
| G7 Safety | Scripted unsafe actions and injected faults | Reliable override, emergency stop and bounded latency |
| G8 Reproducibility | Fresh clone/container rerun | Manifest-driven reproduction without manual source edits |
| G9 Physical pilot | Replay/HIL/closed-course sequence | Qualified safety approval at each stage; never skip gates |

## 8. Immediate next actions

1. Treat the existing 10,000-step CNN run as an integration artifact, not the
   final baseline.
2. Implement P0 manifests and stable metrics before launching 500,000-step
   jobs.
3. Run the CNN baseline with the normal prefill and three seeds.
4. Verify/download one comma2k19 development chunk and execute a short JEPA
   run with collapse monitoring.
5. Make `custom_jepa` fail fast without `outputs/checkpoints/phase2/best.pt`.
6. Validate the custom-JEPA short transfer run before attempting V-JEPA2.
7. Build the multi-route scenario matrix and safety supervisor in parallel.
8. Freeze the CARLA autonomy thresholds and evaluation seeds before the final
   comparison.

## 9. Known blockers and decisions

- **Single A4000 memory:** V-JEPA2 ViT-L may not fit safely with CARLA and
  Dreamer. Measure first; do not silently change the model and claim a matched
  comparison.
- **Action semantics:** the project now defaults to the continuous contract;
  the discrete compatibility path must not be mixed into comparison results.
- **Reward versus safety:** reward shaping is not a safety guarantee. Traffic
  rules and emergency behavior need independent checks.
- **Legacy implementations:** `scripts/train_phase3_arm.py` and deprecated
  wrappers should be retired or explicitly labeled; only one runner should be
  used for published results.
- **Map installation state:** the current optimized-map CARLA configuration
  is an installation-level change and must become reproducible in the image or
  launch configuration.
- **Metrics semantics:** collision, success, truncation, route completion and
  supervisor intervention need one documented definition before thresholds are
  compared.
- **Phase 4 joint JEPA+RSSM:** the design in `reports/future_work.md` is an
  optional research extension after the baseline and scenario gates. It is not
  required to establish the first autonomous CARLA policy.

## 10. Definition of complete

The project may claim **research-grade autonomous driving in CARLA** only when
G0–G8 pass, the final policy meets the frozen scenario thresholds, and the
artifacts are reproducible from a clean environment.

It may claim a **physical-vehicle prototype** only after G9 passes through a
documented hardware-in-the-loop and closed-course safety process. It should not
claim road-ready autonomy from simulation results alone.

