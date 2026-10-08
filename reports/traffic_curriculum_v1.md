# Overnight traffic curriculum, 2026-10-05

The direct simple-to-hard fine-tuning experiment failed to teach reliable
traffic negotiation. All three seed-42 arms used their 50k budget without an
80% success crossing. Final training evaluations were CNN 30% success / 70%
collisions, custom JEPA 0% / 100%, and V-JEPA2 20% / 80%. Complete evaluation
histories also stayed poor, so extending those runs unchanged is unsupported.

The user authorized an overnight training attempt. The new experiment starts
from the original successful simple-turn checkpoints and introduces the built-in
medium task before the hard task. This is a curriculum pilot, with one training
seed, and does not establish a causal pretraining advantage.

## Protocol

- Arm order: custom JEPA, V-JEPA2, CNN; training seed 42.
- Original checkpoints: `outputs/comparison/carla_right_turn_simple_camera_route/`.
- Medium traffic: built-in 8–16 m flow spacing; maximum 30k training steps.
- Hard traffic: built-in 6–8 m flow spacing; maximum 25k steps after promotion.
- Front camera + route, discrete actions, train ratio 512, time penalty 0.1.
  Reward terms and final hard-task benchmark are unchanged.
- Each stage has fresh replay and optimizers, initialized with the preceding
  agent's weights. Every 2.5k steps: ten evaluation episodes. Training ends early
  after three evaluations at >=90% success.
- The best completed training evaluation checkpoint is copied to `best.pt`;
  `latest.pt` remains the interruption/recovery checkpoint. Selection prefers
  higher success, fewer collisions, then higher return.
- Original medium baseline: 20 episodes. Medium candidate validation: 30
  episodes; simple-turn retention: ten. All use evaluation stream seed 2026;
  the Traffic Manager seed stays at the existing default. This is validation
  used for a promotion decision, not an untouched final test set.
- Promotion requires medium success >=80%, collision <=20%, simple-turn success
  >=90% and collision <=10%. An arm failing the gate skips hard training.
- After hard training, evaluate the selected candidate on 30 hard episodes
  and ten simple turns; save metrics and a video of the first episode.
- Controller wall-clock budget: eight hours from launch, including initial
  waiting/evaluations/retries. Consequently later stages can be interrupted by
  the cap, and per-arm realized budgets may differ; report them in the morning.

## Operation

Implementation: `scripts/run_traffic_curriculum.py`. It waits for an existing
CARLA job to finish before starting. It takes its queue lock before clearing
STOP or changing metadata, retries failed training launches up to four times
and evaluations up to three times, and restarts CARLA via the existing launcher.
STOP signals only this launcher's main trainer/evaluator, leaving unrelated
clients and DataLoader workers alone. Training checkpoints on SIGTERM.

The detached controller runs under `systemd-inhibit`; temperature >=92 C for
90 seconds, available RAM <2 GB or free disk <5 GB writes STOP. Checkpoints,
baselines, candidates and older direct-hard runs are kept in separate folders.

```fish
cd ~/Project/Auto_Thinker_Engine
python3 scripts/watch_queue.py --out outputs/overnight_traffic_queue
```

Ctrl-C stops the watcher. To stop training:

```bash
touch outputs/overnight_traffic_queue/STOP
```

Restart after confirming the old controller has exited:

```bash
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
  --why=overnight-traffic .venv/bin/python scripts/run_traffic_curriculum.py \
  --max-hours 8 >> outputs/overnight_traffic_queue/run.log 2>&1 < /dev/null &
```

Outputs:

- `outputs/overnight_traffic_queue/{status.txt,health.log,results.json,plan.json}`
- `outputs/overnight_traffic_queue/evaluations/`: metrics and videos.
- `outputs/comparison/traffic_curriculum_v1/{medium,hard}/<arm>_seed42/`:
  replay, training evals, latest and best checkpoints.

The progress watcher's full-schedule ETA includes hard stages that may be
skipped and can exceed the controller's eight-hour cap. The cap is authoritative.

## Morning decision

Compare the original medium baseline with the selected medium candidate and
its simple-turn retention. Look for actual success gains and collision reduction,
not return alone. For promoted arms, inspect hard success, collisions and
retention. If medium also stays flat, diagnose perception, exploration and
collision timing before changing rewards or spending another equal budget.

Improvement is a hypothesis: the curriculum increases the chance of learning
useful traffic behavior, while the gates prevent blindly promoting failed
policies. Simulator failures and partial stages must be reported separately.
