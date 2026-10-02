#!/usr/bin/env bash
# Long-run queue for the laptop (one GPU, so steps run one after another):
#   1. Phase 2 JEPA, EMA regulariser       (~1.4 h, 30k steps)
#   2. Phase 2 JEPA, SIGReg regulariser    (~2.2 h, 30k steps)
#   3. Linear steering probe for both      (~50 min)
#   4. Pick the larger probe R2 gain over a random encoder
#      -> outputs/checkpoints/phase2/best.pt (used by the custom_jepa arm)
#   5. Stage 3: resume the CNN/BEV run from its checkpoint with
#      train_ratio 512 (CarDreamer's setting) up to DREAMER_STEPS (~3 h)
#
# Every step resumes if the script is started again, and a failed step does
# not stop the later ones. Progress: outputs/long_runs/status.txt; full logs
# in outputs/long_runs/. Start detached:
#   setsid nohup bash jobs/long_runs.sh > outputs/long_runs/run.log 2>&1 &
# Stop: pkill -f jobs/long_runs.sh; pkill -f train_phase2_jepa; pkill -f slurm_run.sh
# (the Dreamer trainer saves latest.pt on SIGTERM).

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

status() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$OUT/status.txt"; }

run_step() {  # name, command...
  local name="$1"; shift
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
