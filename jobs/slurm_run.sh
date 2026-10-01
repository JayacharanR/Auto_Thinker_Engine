#!/usr/bin/env bash
#SBATCH --job-name=driving-world-model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#
# Headless terminal runner. Submit with, for example:
#   CARLA_ROOT=/scratch/$USER/software \
#   RUN_MODE=cnn STEPS=10000 sbatch jobs/slurm_run.sh
#
# Supported RUN_MODE values:
#   phase2, smoke, cnn, custom_jepa, vjepa2, comparison

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN_MODE="${RUN_MODE:-cnn}"
TASK="${TASK:-carla_right_turn_simple}"
SEED="${SEED:-42}"
STEPS="${STEPS:-10000}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_HOST="${CARLA_HOST:-localhost}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/comma2k19}"
PHASE2_CONFIG="${PHASE2_CONFIG:-configs/phase2_jepa_pretrain.yaml}"
PHASE3_CONFIG="${PHASE3_CONFIG:-configs/phase3_transfer_arms.yaml}"

mkdir -p outputs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  echo "Create the environment first, for example: uv sync --frozen" >&2
  exit 2
fi

if [[ "$RUN_MODE" == "phase2" ]]; then
  "$PYTHON_BIN" run.py doctor \
    --require-cuda \
    --require-data \
    --data-dir "$DATA_DIR"
  exec "$PYTHON_BIN" run.py phase2 --config "$PHASE2_CONFIG"
fi

if [[ -z "${CARLA_ROOT:-}" ]]; then
  echo "ERROR: CARLA_ROOT must point to a CARLA 0.9.15 installation." >&2
  exit 2
fi

find_carla_server() {
  local candidate="$1"
  if [[ -f "$candidate" && "$(basename "$candidate")" == "CarlaUE4.sh" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if [[ -f "$candidate/CarlaUE4.sh" ]]; then
    printf '%s\n' "$candidate/CarlaUE4.sh"
    return 0
  fi
  find "$candidate" -maxdepth 3 -type f -name CarlaUE4.sh -print -quit
}

CARLA_SERVER="$(find_carla_server "$CARLA_ROOT")"
if [[ -z "$CARLA_SERVER" ]]; then
  echo "ERROR: Could not find CarlaUE4.sh below CARLA_ROOT=$CARLA_ROOT" >&2
  echo "Set CARLA_ROOT to the simulator directory or its containing directory." >&2
  exit 2
fi
if [[ ! -x "$CARLA_SERVER" ]]; then
  echo "ERROR: CARLA server is not executable: $CARLA_SERVER" >&2
  echo "Run: chmod +x '$CARLA_SERVER'" >&2
  exit 2
fi
CARLA_ROOT="$(dirname "$CARLA_SERVER")"
export CARLA_ROOT
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$PROJECT_ROOT/third_party/CarDreamer:$PROJECT_ROOT/third_party/dreamerv3_torch${PYTHONPATH:+:$PYTHONPATH}"

# CARLA/Unreal needs the graphics-capable NVIDIA runtime, not only CUDA. This
# validates that the container toolkit mounted a real NVIDIA Vulkan ICD and
# also disables audio for headless execution.
source "$PROJECT_ROOT/scripts/carla_runtime_env.sh"

if [[ ! -w "$CARLA_ROOT" ]]; then
  echo "ERROR: CARLA_ROOT is not writable by $(id -un 2>/dev/null || id -u): $CARLA_ROOT" >&2
  echo "Mount the CARLA 0.9.15 directory read-write and use its owning UID." >&2
  exit 2
fi

# Unreal/CARLA refuses uid 0. The preferred fix is to run the container/job as
# a normal user. For containers that must start as root, CARLA_RUN_USER may
# name an existing non-root account; only the CARLA server is then launched
# through runuser, while the Python client remains in the current environment.
CARLA_LAUNCH_PREFIX=()
if [[ "$(id -u)" -eq 0 ]]; then
  if [[ -z "${CARLA_RUN_USER:-}" ]]; then
    echo "ERROR: CARLA refuses to run as root (uid 0)." >&2
    echo "Run the container as a non-root user, or set CARLA_RUN_USER to an existing non-root account." >&2
    echo "Example: CARLA_RUN_USER=student RUN_MODE=smoke bash jobs/slurm_run.sh" >&2
    exit 2
  fi
  if ! id "$CARLA_RUN_USER" >/dev/null 2>&1; then
    echo "ERROR: CARLA_RUN_USER does not exist: $CARLA_RUN_USER" >&2
    exit 2
  fi
  if [[ "$(id -u "$CARLA_RUN_USER")" -eq 0 ]]; then
    echo "ERROR: CARLA_RUN_USER must resolve to a non-root account: $CARLA_RUN_USER" >&2
    exit 2
  fi
  if ! command -v runuser >/dev/null 2>&1; then
    echo "ERROR: root container requires runuser to launch CARLA as $CARLA_RUN_USER." >&2
    echo "Ask the image administrator to install util-linux or run the container as that user." >&2
    exit 2
  fi
  if ! runuser -u "$CARLA_RUN_USER" -- test -r "$CARLA_SERVER"; then
    echo "ERROR: CARLA server is not readable by $CARLA_RUN_USER: $CARLA_SERVER" >&2
    exit 2
  fi
  if ! runuser -u "$CARLA_RUN_USER" -- test -w "$CARLA_ROOT"; then
    echo "ERROR: CARLA_ROOT is not writable by $CARLA_RUN_USER: $CARLA_ROOT" >&2
    echo "Extract CARLA into that user's persistent directory or grant ownership before retrying." >&2
    exit 2
  fi
  CARLA_LAUNCH_PREFIX=(runuser -u "$CARLA_RUN_USER" --)
fi

"$PYTHON_BIN" run.py doctor --require-cuda --require-carla

CARLA_LOG="${CARLA_LOG:-$PROJECT_ROOT/outputs/carla_${CARLA_PORT}.log}"
echo "[job] Starting CARLA on $CARLA_HOST:$CARLA_PORT"
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required so CARLA can be cleaned up with its Unreal child process." >&2
  exit 2
fi
# Keep CARLA and its Unreal child in a dedicated process group. The previous
# launcher tracked only CarlaUE4.sh; when the first readiness probe timed out,
# cleanup could kill the wrapper while leaving CarlaUE4-Linux-Shipping orphaned
# on the RPC port.
setsid --wait -- "${CARLA_LAUNCH_PREFIX[@]}" "$CARLA_SERVER" \
  -RenderOffScreen \
  -nosound \
  -quality-level="${CARLA_QUALITY_LEVEL:-Low}" \
  -carla-rpc-port="$CARLA_PORT" \
  >"$CARLA_LOG" 2>&1 &
CARLA_PID=$!

cleanup() {
  if [[ -n "${CARLA_PID:-}" ]] && kill -0 "$CARLA_PID" 2>/dev/null; then
    echo "[job] Stopping CARLA (pid $CARLA_PID)"
    # Negative PID targets the complete process group, including Unreal.
    kill -TERM -- "-$CARLA_PID" 2>/dev/null || true
    kill -TERM "$CARLA_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$CARLA_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$CARLA_PID" 2>/dev/null || true
    wait "$CARLA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
startup_deadline=$((SECONDS + ${CARLA_STARTUP_TIMEOUT:-240}))
while (( SECONDS < startup_deadline )); do
  if ! kill -0 "$CARLA_PID" 2>/dev/null; then
    echo "ERROR: CARLA exited before becoming ready. Log: $CARLA_LOG" >&2
    tail -80 "$CARLA_LOG" || true
    exit 1
  fi

  # Use a fresh short-lived client for every probe. A client created while
  # Unreal is still loading can remain stuck in get_world() even after CARLA
  # becomes ready; the old wrapper then incorrectly exhausted its retries.
  if "$PYTHON_BIN" - "$CARLA_HOST" "$CARLA_PORT" <<'PY'
import carla
import sys

client = carla.Client(sys.argv[1], int(sys.argv[2]))
client.set_timeout(5.0)
try:
    world = client.get_world()
except Exception:
    raise SystemExit(1)
print(f"[job] CARLA world ready: {world.get_map().name}", flush=True)
PY
  then
    ready=1
    break
  fi

  echo "[job] CARLA world not ready; retrying in 5s"
  sleep 5
done

if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: CARLA world did not become ready within ${CARLA_STARTUP_TIMEOUT:-240}s. Log: $CARLA_LOG" >&2
  tail -80 "$CARLA_LOG" || true
  exit 1
fi

if ! "$PYTHON_BIN" run.py smoke \
    --host "$CARLA_HOST" \
    --port "$CARLA_PORT" \
    --timeout "${CARLA_CLIENT_TIMEOUT:-30}"; then
  echo "ERROR: CARLA smoke test failed after the world became ready. Log: $CARLA_LOG" >&2
  tail -80 "$CARLA_LOG" || true
  exit 1
fi

case "$RUN_MODE" in
  smoke)
    echo "[job] Smoke test passed; no training requested."
    ;;
  cnn|custom_jepa|vjepa2)
    "$PYTHON_BIN" run.py train \
      --arm "$RUN_MODE" \
      --task "$TASK" \
      --seed "$SEED" \
      --steps "$STEPS" \
      --config "$PHASE3_CONFIG"
    ;;
  evaluate|eval)
    "$PYTHON_BIN" run.py evaluate \
      --checkpoint "${CHECKPOINT:-outputs/logs/cnn_seed42/latest.pt}" \
      --task "$TASK" \
      --arm "${ARM:-cnn}" \
      --episodes "${EPISODES:-5}" \
      --seed "$SEED" \
      --output-dir "${OUTPUT_DIR:-outputs/eval_results}"
    ;;
  comparison)
    "$PYTHON_BIN" run.py compare \
      --task "$TASK" \
      --steps "$STEPS" \
      --config "$PHASE3_CONFIG"
    ;;
  *)
    echo "ERROR: unsupported RUN_MODE=$RUN_MODE" >&2
    exit 2
    ;;
esac
