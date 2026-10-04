#!/usr/bin/env bash
# Roadmap steps 1 and 2, one after the other:
#   1. Zero-shot robustness evaluation of the trained right-turn agents
#      (jobs/robustness.sh -> outputs/robustness/robustness.md, videos).
#   2. Traffic curriculum: fine-tune each arm's seed-42 right-turn agent on
#      carla_right_turn_hard (dense cross traffic), 50k steps max, evals every
#      5k, early stop after 3 evals at >= 90% success (jobs/comparison.sh ->
#      outputs/comparison/carla_right_turn_hard_finetune/comparison.md).
# Both resume if started again. Stopping step 1 (touch
# outputs/robustness_queue/STOP) also skips step 2; stop step 2 with
# touch outputs/traffic_queue/STOP.
#
#   setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why=training \
#     bash jobs/roadmap_1_2.sh > outputs/roadmap_1_2.log 2>&1 &
# Watch:  python3 scripts/watch_queue.py --out outputs/robustness_queue   (then traffic_queue)

set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

bash jobs/robustness.sh
if [[ -f outputs/robustness_queue/STOP ]]; then
  echo "Robustness evaluation was stopped; not starting the traffic run."
  exit 0
fi

TASK=carla_right_turn_hard \
INIT_RUNS=outputs/comparison/carla_right_turn_simple_camera_route \
RUN_DIR=outputs/comparison/carla_right_turn_hard_finetune \
OUT=outputs/traffic_queue \
SEEDS=42 AGG_SEEDS=42 STEPS=50000 EARLY_STOP=3 EXTRA_SET="--set eval_every=5000" \
  bash jobs/comparison.sh
