#!/usr/bin/env bash
# Shared machinery for long-job queues (sourced, not executed).
#
# The sourcing script sets PROJECT_ROOT and OUT (its output directory), then
# calls start_queue "<title>" and run_step <name> <command...> for each job.
# Provides: status log (OUT/status.txt), health monitor (OUT/health.log) that
# stops trainers cleanly and halts the queue if the GPU stays >= GPU_TEMP_LIMIT C
# for 90 s, available RAM < MIN_RAM_GB or free disk < MIN_DISK_GB, and a STOP
# file (touch OUT/STOP) for manual stops. Trainers save a checkpoint on SIGTERM.

mkdir -p "$OUT"
export PYTHONUNBUFFERED=1  # logs show progress as it happens
GPU_TEMP_LIMIT="${GPU_TEMP_LIMIT:-92}"
MIN_RAM_GB="${MIN_RAM_GB:-2}"
MIN_DISK_GB="${MIN_DISK_GB:-5}"
STOP_FILE="$OUT/STOP"
rm -f "$STOP_FILE"

status() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$OUT/status.txt"; }

TRAINER_PATTERN='^[^ ]*python[0-9.]* [^ ]*scripts/(train_phase2_jepa|train_cardreamer|probe_phase2)\.py'

stop_trainers() {
  # SIGTERM only the main trainer processes. The pattern is anchored to a
  # Python interpreter (never a shell that merely mentions the script), and
  # children of a trainer (DataLoader workers share its command line) are
  # skipped: the trainer shuts them down itself after checkpointing.
  local pid parent
  for pid in $(pgrep -f "$TRAINER_PATTERN"); do
    parent="$(ps -o ppid= -p "$pid" | tr -d ' ')"
    if [[ -n "$parent" ]] && tr '\0\n' '  ' < "/proc/$parent/cmdline" 2>/dev/null \
        | grep -Eq "$TRAINER_PATTERN"; then
      continue
    fi
    kill -TERM "$pid" 2>/dev/null
  done
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
      [[ -s "$STOP_FILE" ]] || echo "manual STOP file" > "$STOP_FILE"
      status "STOP: $(cat "$STOP_FILE"); stopping trainers"
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

start_queue() {  # title
  rm -f "$STOP_FILE"
  status "=== $1 started ==="
  health_monitor &
  MONITOR_PID=$!
  trap 'kill "$MONITOR_PID" 2>/dev/null' EXIT
}
