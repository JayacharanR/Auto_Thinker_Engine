# Auto Thinker Engine — Progress, Bugs, and Implementation Plan

**Last updated:** 2026-09-04  
**Project:** `Auto_Thinker_Engine`  
**Hardware path tested:** RTX A4000, CARLA 0.9.15, `auto-thinker:carla-gpu`

## Executive status

The CARLA runtime and Gate 0 smoke path are working. The project can launch
CARLA headlessly, connect from Python, spawn a vehicle and camera, and step the
simulator. The measured final smoke throughput was **83.9 steps/sec for 100
steps** on the A4000 path.

The CNN training run has **not completed** and no Dreamer checkpoint has been
generated yet. Several infrastructure and integration bugs were fixed, but the
latest training attempt reached environment creation and exposed the first
Dreamer action-space mismatch. A narrow trainer patch for that mismatch is
applied, but it still needs runtime validation. The next expected integration
blocker is the observation-key mismatch documented below.

Current phase status:

| Phase | Status | Evidence / next gate |
|---|---|---|
| Phase 0 — environment | **Passed** | Strict doctor passed; CARLA Gate 0 smoke passed at 83.9 steps/sec. |
| Phase 1 — CNN Dreamer baseline | **In progress / blocked before training** | Six 10k attempts stopped at CARLA integration failures; the latest reached the action wrapper. |
| Phase 2 — JEPA pretraining | **Not started** | Requires comma2k19 data verification and a stable baseline runtime. |
| Phase 3 — encoder comparison | **Not started** | Depends on Phase 1 and Phase 2 checkpoints. V-JEPA2 remains deferred on 16 GB VRAM. |

## What the project is intended to do

The project combines:

1. A DreamerV3 world-model RL agent running in CARLA.
2. JEPA self-supervised visual pretraining on comma2k19 driving video.
3. A controlled Phase 3 comparison of three visual encoders:
   - trainable Dreamer CNN;
   - frozen custom JEPA ViT-S;
   - frozen pretrained V-JEPA2 ViT-L.

The implementation intentionally uses `dreamerv3-torch` as the PyTorch
Dreamer backbone and CarDreamer only for CARLA task definitions, route logic,
observations and rewards. The original JAX CarDreamer Dreamer path cannot accept
the project’s PyTorch encoder modules without a framework-level rewrite.

## Directory understanding

No `README.d` file was present in the project. `README.md` is the main project
description and was treated as the intended README.

| Path | Role |
|---|---|
| `README.md` | Architecture, phases, setup, commands and design decisions. |
| `run.py` | Root dispatcher for `doctor`, `smoke`, `train`, `phase2`, `probe` and `compare`. |
| `SERVER_RUNBOOK_A4000.md` | Ubuntu 22.04 + RTX A4000 installation and operational runbook. |
| `DOCKER_CARLA_RUNBOOK.md` | Docker-specific CARLA/NVIDIA/Vulkan setup and Gate 0 commands. |
| `jobs/slurm_run.sh` | CARLA lifecycle wrapper, readiness checks, smoke gate and training dispatch. |
| `scripts/smoke_test_carla.py` | Vehicle/camera/100-step CARLA Gate 0 test. |
| `scripts/train_cardreamer.py` | Phase 1/3 DreamerV3 training entry point and encoder-arm comparison loop. |
| `scripts/train_phase2_jepa.py` | Planned comma2k19 JEPA pretraining entry point. |
| `src/jepa/` | JEPA encoder, target EMA, predictor and masking implementation. |
| `src/dreamer/` | Encoder adapters, frame stacking and Dreamer encoder hook. |
| `src/data/` | comma2k19 loading and transforms. |
| `configs/` | Experiment, Phase 1, Phase 2 and Phase 3 configuration contracts. |
| `third_party/CarDreamer/` | CARLA environments and task definitions. |
| `third_party/dreamerv3_torch/` | PyTorch DreamerV3 submodule plus the expected compatibility changes. |
| `outputs/` | Ignored logs, replay episodes, checkpoints and videos. |
| `reports/` | Technical writeups and future-work documents. |

## Runtime discovered and validated

The host project `.venv` is a broken/root-owned symlink in the current
environment, and the host shell does not expose the required Python 3.10
runtime. The working execution path is the already available
`auto-thinker:carla-gpu` Docker image/container:

- Python: `/opt/auto-thinker/venv/bin/python`, Python 3.10.12.
- CARLA: `/opt/carla` inside the container; CARLA 0.9.15.
- GPU: NVIDIA RTX A4000, CUDA visible and usable from PyTorch.
- Graphics: NVIDIA graphics capabilities and Vulkan ICD are available.
- Shared memory: 2 GiB was used for CARLA/Unreal stability.
- Project mount: repository mounted as `/work`.

The older CUDA-only runtime is not suitable for Unreal/CARLA because CARLA
needs the graphics-capable NVIDIA runtime, Vulkan libraries and off-screen
rendering support.

The strict doctor checks passed for the required Python imports, CUDA, CARLA
API and project dependencies. The CARLA launcher must still run as a non-root
user or use the runbook’s explicit `CARLA_RUN_USER` path.

## Implemented fixes

### 1. CARLA process-group cleanup and startup readiness

Updated [`jobs/slurm_run.sh`](../jobs/slurm_run.sh) to:

- launch CARLA through `setsid --wait` in a dedicated process group;
- terminate the full process group, including Unreal’s shipping binary;
- detect early CARLA exit and print the server log;
- probe readiness with a fresh short-lived CARLA client on every attempt;
- wait up to `CARLA_STARTUP_TIMEOUT` (default 240 seconds);
- run the actual smoke test only after `get_world()` succeeds;
- pass the configurable `CARLA_CLIENT_TIMEOUT` to the smoke test.

This fixed the previous cleanup failure where killing `CarlaUE4.sh` left
`CarlaUE4-Linux-Shipping` orphaned on port 2000.

### 2. Configurable smoke-test timeout

Updated [`scripts/smoke_test_carla.py`](../scripts/smoke_test_carla.py) so the
client timeout is configurable via the function and CLI `--timeout` argument,
instead of always using a hard-coded value.

### 3. CARLA world loading and map reuse

Updated CarDreamer’s
[`world_manager.py`](../third_party/CarDreamer/car_dreamer/toolkit/carla_manager/world_manager.py)
to:

- read `CARLA_WORLD_TIMEOUT`, defaulting to 60 seconds;
- reuse an already-loaded map when the requested map matches after removing the
  `_Opt` suffix;
- avoid an unnecessary `load_world()` during task creation;
- print the current/requested map transition for diagnosis.

The packaged CARLA build was reliable with `Town03_Opt`, while direct startup
with `Town03` crashed. The installed CARLA
`software/CarlaUE4/Config/DefaultEngine.ini` was changed locally so the
startup/default/transition maps use `Town03_Opt`. This is an installation-level
change, not a project Git file, and must be repeated for a fresh CARLA install.

### 4. Traffic Manager crash avoidance

The CarDreamer right-turn-simple task does not request traffic-flow
parameters, but `WorldManager` unconditionally called CARLA Traffic Manager.
On this CARLA build, `get_trafficmanager()` caused a native Python/CARLA
segmentation fault, including when alternate Traffic Manager ports were tried.

`world_manager.py` now disables Traffic Manager when the task does not request
traffic parameters or when `CARLA_DISABLE_TRAFFIC_MANAGER` is true. It also
guards traffic-dependent methods and synchronous-mode handling. Tasks that
actually require background traffic still need a dedicated validation pass.

### 5. Monitor thread shutdown

Updated CarDreamer’s [`monitor.py`](../third_party/CarDreamer/car_dreamer/toolkit/monitor/monitor.py)
so the Flask monitor thread is daemonized and `stop()` uses a bounded join.
This addresses the controlled environment test reaching `env_closed` while
the Python process remained alive because the Flask thread was non-daemon.
A clean process-exit retest after this final patch is still recommended.

### 6. Discrete action wrapper selection

The latest patch in [`scripts/train_cardreamer.py`](../scripts/train_cardreamer.py)
detects Gym `Discrete` action spaces and applies Dreamer’s `OneHotAction`
wrapper. Continuous `Box` spaces continue to use `NormalizeActions`.

This is necessary because `NormalizeActions` accesses `.low` and `.high`,
which do not exist on `Discrete`. The patch was applied after the latest failed
run, but post-patch runtime validation was interrupted before it could be
confirmed end-to-end.

## Reproduced bugs and root causes

### Bug A — CARLA client timed out during startup

Symptoms appeared in `cnn_10000_seed42_console.log` through
`cnn_10000_seed42_fixed3.log` as 20-second, 60-second and 180-second
`get_world()` timeouts.

Root causes were combined:

- Unreal took longer than the original probe window to initialize.
- The original launcher reused a client that had been created during the
  loading window and could remain stuck.
- Cleanup killed only the shell wrapper, leaving Unreal alive and making later
  attempts interact with stale state.

Status: fixed in the launcher, with Gate 0 subsequently passing.

### Bug B — CARLA server process orphaned after failed attempts

The old launcher tracked `CarlaUE4.sh`, but the real Unreal process was its
child. On failure, the child remained alive and kept the RPC port occupied.

Status: fixed by dedicated process-group launch and group cleanup.

### Bug C — Wrong/default map and slow map switching

The task `carla_right_turn_simple` requests `Town03`. A server started with
another map caused CarDreamer to call `load_world()`, which repeatedly timed
out or caused instability. The packaged build was stable with the optimized
`Town03_Opt` startup map, and the manager now treats `Town03` and `Town03_Opt`
as the same logical task map.

Status: startup path fixed for the current installation; fresh installations
need the optimized-map default or an equivalent CARLA launch configuration.

### Bug D — Traffic Manager native crash

The task did not need background traffic, but CarDreamer initialized Traffic
Manager unconditionally. The CARLA native client segfaulted while obtaining a
Traffic Manager handle.

Status: bypassed for tasks without traffic configuration. Medium/hard/random
traffic tasks remain unvalidated.

### Bug E — Flask monitor prevented process exit

The environment reset/close path completed, but the non-daemon Flask server
thread kept the process alive.

Status: bounded daemon-thread shutdown patch applied; clean-exit verification
is pending.

### Bug F — Continuous normalizer applied to a discrete task

`third_party/CarDreamer/car_dreamer/configs/tasks.yaml` defines
`carla_right_turn_simple` with five discrete steering values. CarDreamer
therefore returns `gym.spaces.Discrete`, but the custom trainer always applied
`NormalizeActions`, causing:

```text
AttributeError: 'Discrete' object has no attribute 'low'
```

Status: discrete-to-one-hot wrapper patch applied. The Dreamer actor
distribution must also be set to `onehot`; see remaining blockers.

### Bug G — Configuration says continuous while the selected task is discrete

`configs/experiment_contract.yaml`, the Phase 1 config and the Phase 3 config
describe a continuous `[steer, throttle/brake]` action. The selected
CarDreamer right-turn-simple task actually uses a discrete action composed from
discrete acceleration and steering values.

This is a scientific-contract issue, not only a Python exception. The action
space must be made identical across all comparison arms and documented as
either discrete or continuous.

Status: unresolved decision. Short-term validation can use the actual discrete
task. The controlled experiment should later either add/use a continuous
right-turn task or update every contract/config/metric description to discrete.

### Bug H — Observation key mismatch is expected next

The CarDreamer reset test returned observation keys:

```text
['birdeye_wpt', 'camera', 'collision']
```

The custom encoder hook defaults to `obs_key="image"`, and Dreamer defaults
also refer to an `image` CNN key. Once action creation proceeds past the new
one-hot wrapper, training is expected to fail unless the camera observation is
adapted to the project’s `image` contract or all encoder/decoder keys are
changed consistently.

Status: unresolved; this must be fixed before claiming that training has
started.

### Bug I — Dreamer actor distribution still defaults to continuous

The action wrapper changes the exposed action space to a one-hot Box with a
`discrete` marker, but `load_dreamer_config()` starts from Dreamer’s default
actor configuration, whose distribution is `normal`. A normal actor emits
continuous values; `OneHotAction.step()` requires a valid one-hot vector.

Status: unresolved. For the discrete task, set the actor distribution to
`onehot` (and the corresponding non-learned standard-deviation setting) before
constructing `Dreamer`, then run a one-episode action test.

### Bug J — Training/evaluation environment sharing

`train_cardreamer.py` currently creates one CarDreamer environment and wraps the
same object in both `train_envs` and `eval_envs`. Evaluation and training can
therefore share CARLA actors, world settings and episode state.

Status: unresolved. Create separate environment instances, or make evaluation
strictly sequential with explicit reset/state ownership.

### Bug K — Requested step count needs verification

The custom loop uses `while agent._step < config.steps + config.eval_every` and
then simulates `eval_every` steps per iteration. With a 10,000-step request and
10,000-step evaluation interval, this may run an extra interval rather than
honoring `--steps` as the exact learner/environment budget.

Status: unresolved. Define whether `steps` includes prefill and evaluation,
then add an assertion/log showing the actual environment and update counts.

### Bug L — Required experiment metrics are incomplete

The experiment contract requires episode reward/length, success rate, collision
rate, wall-clock time and peak VRAM. The current custom final-metrics block
primarily extracts reward/length/success-like values from Dreamer internals and
does not yet guarantee all required metrics are emitted for every arm.

Status: unresolved. Add a shared metrics collector and write a stable JSON
record per arm/seed.

## Attempt history

The following logs are in `outputs/`:

| Log | Result |
|---|---|
| `cnn_10000_seed42_console.log` | CARLA client timed out at 20 seconds. |
| `cnn_10000_seed42_fixed.log` | Same startup/readiness/orphan-process family of failure. |
| `cnn_10000_seed42_fixed2.log` | 60-second world-load timeout. |
| `cnn_10000_seed42_fixed3.log` | 180-second world-load timeout. |
| `cnn_10000_seed42_fixed4.log` | CARLA was on Town04; task attempted Town04 → Town03 and terminated. |
| `cnn_10000_seed42_fixed5.log` | Map reuse succeeded; Traffic Manager initialization then crashed/terminated. |
| `cnn_10000_seed42_fixed6.log` | Map reuse and Traffic Manager bypass succeeded; trainer failed on `Discrete.low`. |

No current CARLA or training job should be assumed active after the interrupted
turn. No `latest.pt` or Phase 2 checkpoint was observed in `outputs/`.

## Validation already completed

### Passed

- Strict runtime doctor in the A4000 container.
- Python, PyTorch, NumPy, YAML, Flask, OpenCV and CARLA imports.
- CUDA availability from PyTorch.
- CARLA server launch in off-screen low-quality mode.
- CARLA world readiness probe.
- Vehicle and camera spawn.
- 100 synchronous CARLA steps with random controls.
- Final Gate 0 throughput: 83.9 steps/sec.
- CarDreamer task creation and reset after map reuse and Traffic Manager bypass.

### Pending verification

- Post-patch discrete `OneHotAction` environment path.
- `camera` → `image` observation adaptation.
- Dreamer `onehot` actor construction and one-hot action execution.
- Full prefill and one learner update.
- Clean exit after the monitor daemon-thread patch.
- CNN 10,000-step run and checkpoint creation.

## Ordered implementation plan

### P0 — Finish the first CNN integration run

1. Validate the current trainer file without writing root-owned `__pycache__`
   files.
2. Add the observation adapter so CarDreamer’s `camera` is exposed as the
   expected `image` key, while deciding whether `birdeye_wpt` and `collision`
   should be retained for the decoder/metrics.
3. Set the Dreamer actor distribution to `onehot` for the actual discrete task.
4. Create independent train and evaluation environments.
5. Run a short one-episode/prefill test and verify action, observation, reward,
   terminal and cleanup behavior.
6. Run the configured CNN validation:

   ```bash
   PYTHON_BIN=/opt/auto-thinker/venv/bin/python \
   CARLA_ROOT=/opt/carla \
   RUN_MODE=cnn STEPS=10000 SEED=42 \
   bash jobs/slurm_run.sh
   ```

7. Confirm that `outputs/logs/cnn_seed42/latest.pt`, replay episodes, metrics
   and CARLA logs are created and that the process tree is empty at exit.

### P1 — Make the experiment contract executable

- Resolve discrete-versus-continuous action semantics.
- Load shared values from `configs/experiment_contract.yaml` rather than
  duplicating important values in Python and multiple YAML files.
- Make `--steps` exact and report prefill, environment steps, update steps and
  wall-clock time separately.
- Add deterministic seed reporting and a run manifest.
- Record success, collision, reward, episode length, wall-clock and peak VRAM
  consistently for every arm/seed.
- Add tests for action conversion, observation adaptation and checkpoint resume.

### P2 — Complete Phase 1 baseline

- Run at least the 10,000-step integration validation.
- If stable, run the intended 500,000-step CNN baseline.
- Evaluate on the configured number of episodes.
- Store the baseline checkpoint and metrics as the comparison reference.
- Keep CARLA and all replay/checkpoint outputs on persistent storage.

### P3 — Implement and validate Phase 2 JEPA

- Verify/download comma2k19; the full dataset is approximately 100 GB.
- Start with one development chunk before the full dataset.
- Run the ViT-Small context/EMA target/predictor training at 224x224.
- Monitor representation variance and stop/report collapse.
- Save best/latest checkpoints under `outputs/checkpoints/phase2/`.
- Run the steering linear probe against both trained and random-init encoders.
- Generate representation visualizations only after the probe path is valid.

### P4 — Implement controlled Phase 3 comparison

- Ensure CNN, custom JEPA and V-JEPA2 receive the same action/environment
  semantics, reward, seed policy, evaluation protocol and training budget.
- Validate frame stacking for four-frame JEPA/V-JEPA2 clips.
- Load the Phase 2 checkpoint for `custom_jepa` and fail loudly if it is absent
  when the arm is intended to be a pretrained arm.
- Defer V-JEPA2 until memory use is measured; a 16 GB A4000 is tight with
  CARLA and a ViT-L model sharing the GPU.
- Run matched seeds, save one manifest/metrics record per run, then aggregate
  convergence, success, collision, wall-clock and VRAM results.

### P5 — Harden the operational workflow

- Re-test a fresh clone with initialized submodules.
- Make the CARLA optimized-map configuration reproducible instead of relying
  only on a manually edited installation file.
- Add a preflight that checks Vulkan, CARLA writability, non-root execution,
  available VRAM, disk space and required dataset/checkpoint paths.
- Add automatic stale-port detection that reports the owning process without
  killing unrelated jobs.
- Run the existing test suite and add regression tests for each fixed bug.

## Important configuration inconsistencies to resolve

1. **Action type:** experiment configs say continuous; the selected task is
   discrete.
2. **Observation names:** project contract says `image`; CarDreamer returns
   `camera`, `birdeye_wpt` and `collision`.
3. **Resolution:** the project describes CNN 64/128 and JEPA/V-JEPA2 224, but
   the current training entry point uses a common default image size and relies
   on the encoder hook to resize.
4. **Traffic:** simple right-turn does not need Traffic Manager, while harder
   traffic tasks do. The bypass must not silently change the intended task.
5. **Metric ownership:** the contract requires driving metrics that are not yet
   guaranteed by the Dreamer logger.
6. **Phase dependency:** Phase 3 custom JEPA needs a real Phase 2 checkpoint;
   Phase 3 should not be presented as complete before that artifact exists.

## Current source-control state

The root worktree currently reports modifications in:

- `jobs/slurm_run.sh`
- `scripts/smoke_test_carla.py`
- `scripts/train_cardreamer.py`
- `third_party/CarDreamer` submodule
- `third_party/dreamerv3_torch` submodule

The dirty third-party submodules contain the local CARLA lifecycle/monitor
changes and the expected Dreamer compatibility changes. Preserve these changes
when preparing a reproducible commit. The CARLA `DefaultEngine.ini` change is
outside the project worktree and must be documented or automated separately.

## Definition of “training started successfully”

The CNN implementation should not be called started merely because CARLA
launches. The minimum evidence is:

- a compatible observation and action space printed by the trainer;
- one successful reset and action step through the Dreamer policy;
- prefill episodes written to disk;
- at least one Dreamer world-model/actor update;
- a checkpoint written to `outputs/logs/<arm>_seed<seed>/latest.pt`;
- metrics and a clean CARLA/container process exit.

Only after those checks pass should the project proceed to the 500,000-step
baseline, Phase 2 pretraining and the Phase 3 three-arm comparison.
