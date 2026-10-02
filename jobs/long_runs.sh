#!/usr/bin/env bash
# Long-run queue for the laptop (one GPU, so steps run one after another):
#   1. Phase 2 JEPA, EMA regulariser       (~1.4 h, 30k steps)
#   2. Phase 2 JEPA, SIGReg regulariser    (~2.2 h, 30k steps)
#   3. Linear steering probe for both      (~2 min, encode once + ridge)
#   4. Pick the larger probe R2 gain over a random encoder
#      -> outputs/checkpoints/phase2/best.pt (used by the custom_jepa arm)
#   5. Stage 3: resume the CNN/BEV run from its checkpoint with
#      train_ratio 512 (CarDreamer's setting) up to DREAMER_STEPS (~3 h)
#
# Every step resumes if the script is started again, and a failed step does
# not stop the later ones. Progress: outputs/long_runs/status.txt; full logs
# in outputs/long_runs/; machine health every 30 s in outputs/long_runs/health.log.
#
# Crash protection: a health monitor stops the trainers (SIGTERM, both save a
# checkpoint) and halts the queue if the GPU stays >= GPU_TEMP_LIMIT C for 90 s,
# available RAM < MIN_RAM_GB, or free disk < MIN_DISK_GB. Start detached with
# system sleep blocked:
#   mkdir -p outputs/long_runs
#   setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="training" bash jobs/long_runs.sh > outputs/long_runs/run.log 2>&1 &
# Stop cleanly: touch outputs/long_runs/STOP   (finishes nothing new, stops trainers)

set -uo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PY="$PROJECT_ROOT/.venv/bin/python"
OUT="outputs/long_runs"
mkdir -p "$OUT"
export CARLA_ROOT="${CARLA_ROOT:-$HOME/software/CARLA_0.9.15}"
DREAMER_STEPS="${DREAMER_STEPS:-150000}"
DREAMER_LOGDIR="${DREAMER_LOGDIR:-outputs/logs/cnn_bev_seed42}"
CONFIG="configs/phase2_jepa_laptop.yaml"
GPU_TEMP_LIMIT="${GPU_TEMP_LIMIT:-92}"
MIN_RAM_GB="${MIN_RAM_GB:-2}"
MIN_DISK_GB="${MIN_DISK_GB:-5}"
STOP_FILE="$OUT/STOP"
rm -f "$STOP_FILE"

status() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$OUT/status.txt"; }

stop_trainers() {
  # Anchored: only a Python interpreter running one of these scripts matches,
  # never a shell whose command line merely mentions them.
  pkill -TERM -f '^[^ ]*python[0-9.]* [^ ]*scripts/(train_phase2_jepa|train_cardreamer|probe_phase2)\.py'
}

health_monitor() {
  local hot=0 gpu ram disk cpu
  while true; do
    gpu="$(nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw \
           --format=csv,noheader,nounits 2>/dev/null | head -1)"
    ram="$(awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"
    disk="$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc 0-9)"
    cpu="$(sensors 2>/dev/null | awk '/Package id 0/ {print $4; exit}')"
    echo "$(date '+%H:%M:%S') gpu[temp,MiB,util,W]=${gpu// /} cpu=${cpu:-?} ram_avail=${ram}G disk_free=${disk}G" \
      >> "$OUT/health.log"
    if [[ "${gpu%%,*}" =~ ^[0-9]+$ ]] && (( ${gpu%%,*} >= GPU_TEMP_LIMIT )); then
      hot=$((hot + 1))
    else
      hot=0
    fi
    local reason=""
    (( hot >= 3 )) && reason="GPU at ${gpu%%,*} C for 90 s"
    awk -v r="$ram" -v m="$MIN_RAM_GB" 'BEGIN { exit !(r < m) }' && reason="RAM available ${ram} GB"
    (( disk < MIN_DISK_GB )) && reason="disk free ${disk} GB"
    if [[ -n "$reason" ]]; then
      echo "health monitor: $reason" > "$STOP_FILE"
    fi
    if [[ -f "$STOP_FILE" ]]; then
      status "HEALTH STOP: $(cat "$STOP_FILE"); stopping trainers"
      stop_trainers
      return
    fi
    sleep 30
  done
}

run_step() {  # name, command...
  local name="$1"; shift
  if [[ -f "$STOP_FILE" ]]; then
    status "SKIP  $name (stopped: $(cat "$STOP_FILE"))"
    return
  fi
  status "START $name"
  if "$@" > "$OUT/$name.log" 2>&1; then
    status "OK    $name"
  else
    status "FAIL  $name (exit $?, see $OUT/$name.log)"
  fi
}

phase2() {  # regulariser
  local dir="outputs/checkpoints/phase2_$1" resume=()
  [[ -f "$dir/latest.pt" ]] && resume=(--resume)
  "$PY" scripts/train_phase2_jepa.py --config "$CONFIG" --regularizer "$1" \
    --checkpoint-dir "$dir" "${resume[@]}"
}

select_phase2() {
  "$PY" - <<'PY'
import json, shutil
from pathlib import Path

results = {}
for reg in ("ema", "sigreg"):
    path = Path(f"outputs/probe_results/probe_phase2_{reg}_best_seed42.json")
    if path.is_file():
        r = json.loads(path.read_text())["results"]
        results[reg] = {"trained_r2": r["trained"]["r2"], "random_r2": r["random"]["r2"],
                        "gain_r2": r["trained"]["r2"] - r["random"]["r2"],
                        "trained_mae": r["trained"]["mae"], "random_mae": r["random"]["mae"]}
if not results:
    raise SystemExit("no probe results")
winner = max(results, key=lambda k: results[k]["gain_r2"])
dest = Path("outputs/checkpoints/phase2/best.pt")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(f"outputs/checkpoints/phase2_{winner}/best.pt", dest)
summary = {"winner": winner, "criterion": "trained R2 - random R2", "results": results}
Path("outputs/probe_results/selection.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
}

status "=== long-run queue started (Dreamer target ${DREAMER_STEPS} steps) ==="
health_monitor &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null' EXIT
run_step phase2_ema phase2 ema
run_step phase2_sigreg phase2 sigreg
for reg in ema sigreg; do
  if [[ -f "outputs/checkpoints/phase2_$reg/best.pt" ]]; then
    run_step "probe_$reg" "$PY" scripts/probe_phase2.py \
      --checkpoint "outputs/checkpoints/phase2_$reg/best.pt"
  fi
done
run_step select_phase2 select_phase2
run_step dreamer_resume env RUN_MODE=cnn STEPS="$DREAMER_STEPS" \
  TRAIN_ARGS="--resume --logdir $DREAMER_LOGDIR --set train_ratio=512" bash jobs/slurm_run.sh
status "=== long-run queue finished ==="
