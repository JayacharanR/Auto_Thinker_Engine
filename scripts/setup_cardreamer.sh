#!/usr/bin/env bash
# Environment Setup Script
#
# Run this on the target hardware to set up CARLA + dreamerv3-torch + CarDreamer tasks.
# Prerequisites: CARLA 0.9.15 installed, Python 3.10, uv
#
# Usage:
#   bash scripts/setup_cardreamer.sh /path/to/carla

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CARDREAMER_DIR="$PROJECT_ROOT/third_party/CarDreamer"
DREAMER_TORCH_DIR="$PROJECT_ROOT/third_party/dreamerv3_torch"
DREAMER_BASE_COMMIT="6ef8646d807cd10ce0c88e10a7e943211e7fc44c"
DREAMER_COMPAT_PATCH="$PROJECT_ROOT/patches/dreamerv3_torch_compat.patch"
CARDREAMER_COMPAT_PATCH="$PROJECT_ROOT/patches/cardreamer_compat.patch"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
UV_BIN="${UV_BIN:-uv}"

echo "============================================"
echo "  Environment Setup"
echo "============================================"

# --- Step 1: CARLA path ---
CARLA_ROOT_INPUT="${1:-${CARLA_ROOT:-}}"
if [ -z "$CARLA_ROOT_INPUT" ]; then
    echo "ERROR: CARLA path required."
    echo "Usage: bash scripts/setup_cardreamer.sh /path/to/carla"
    echo ""
    echo "Download CARLA 0.9.15 from:"
    echo "  https://github.com/carla-simulator/carla/releases/tag/0.9.15/"
    exit 1
fi

if [ ! -e "$CARLA_ROOT_INPUT" ]; then
    echo "ERROR: CARLA path not found: $CARLA_ROOT_INPUT"
    exit 1
fi

find_carla_server() {
    local candidate="$1"
    if [ -f "$candidate" ] && [ "$(basename "$candidate")" = "CarlaUE4.sh" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [ -f "$candidate/CarlaUE4.sh" ]; then
        printf '%s\n' "$candidate/CarlaUE4.sh"
        return 0
    fi
    find "$candidate" -maxdepth 3 -type f -name CarlaUE4.sh -print -quit
}

CARLA_SERVER="$(find_carla_server "$CARLA_ROOT_INPUT")"
if [ -z "$CARLA_SERVER" ]; then
    echo "ERROR: Could not find CarlaUE4.sh below: $CARLA_ROOT_INPUT"
    echo "Pass either the CARLA directory or the directory containing CarlaUE4.sh."
    exit 1
fi
if [ ! -x "$CARLA_SERVER" ]; then
    echo "ERROR: CARLA server is not executable: $CARLA_SERVER"
    echo "Run: chmod +x '$CARLA_SERVER'"
    exit 1
fi
CARLA_ROOT="$(dirname "$CARLA_SERVER")"

echo "[1/6] CARLA root: $CARLA_ROOT"
echo "      CARLA server: $CARLA_SERVER"
export CARLA_ROOT
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}"

# --- Step 2: Init submodules ---
echo "[2/6] Checking git submodules..."
cd "$PROJECT_ROOT"
if [ -d "$CARDREAMER_DIR/car_dreamer" ] && [ -f "$DREAMER_TORCH_DIR/models.py" ]; then
    echo "  Submodule directories are already populated; skipping network update."
else
    echo "  Populating missing submodule directories..."
    git submodule update --init --recursive
fi

if [ ! -d "$CARDREAMER_DIR/car_dreamer" ]; then
    echo "ERROR: CarDreamer submodule is missing: $CARDREAMER_DIR"
    exit 1
fi
if [ ! -f "$DREAMER_TORCH_DIR/models.py" ]; then
    echo "ERROR: dreamerv3-torch submodule is missing: $DREAMER_TORCH_DIR"
    exit 1
fi

# Apply a tracked compatibility patch to a submodule checkout. A tree that
# already contains the patch is accepted unchanged; any other modified tree is
# rejected rather than patched on top of unknown edits.
apply_compat_patch() {
    local repo="$1" patch="$2" name="$3"
    if [ ! -f "$patch" ]; then
        echo "ERROR: Missing compatibility patch: $patch"
        exit 1
    fi
    if ! git -C "$repo" rev-parse --verify HEAD >/dev/null 2>&1; then
        echo "ERROR: $name is not a usable git checkout."
        exit 1
    fi
    if git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "  $name compatibility patch already applied."
    elif git -C "$repo" apply --check "$patch" >/dev/null 2>&1; then
        echo "  Applying project compatibility patch to $name..."
        git -C "$repo" apply "$patch"
    else
        echo "ERROR: $name is neither the pinned base nor the patched tree."
        echo "       Current commit: $(git -C "$repo" rev-parse HEAD)"
        echo "       To discard local edits and retry: git -C '$repo' checkout -- ."
        exit 1
    fi
}

# The public upstream dreamerv3-torch repository no longer contains the old
# gitlink used by earlier project commits. The superproject pins its reachable
# main tip; the patch restores the custom-encoder and CPU-optimizer interfaces.
apply_compat_patch "$DREAMER_TORCH_DIR" "$DREAMER_COMPAT_PATCH" "dreamerv3-torch"
# CarDreamer: world timeout, _Opt map reuse, lazy Traffic Manager, daemon monitor.
apply_compat_patch "$CARDREAMER_DIR" "$CARDREAMER_COMPAT_PATCH" "CarDreamer"
echo "  dreamerv3-torch commit: $(git -C "$DREAMER_TORCH_DIR" rev-parse HEAD)"

# --- Step 3: Install project with uv ---
echo "[3/6] Installing project dependencies via uv (locked)..."
cd "$PROJECT_ROOT"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Project Python environment not found: $PYTHON_BIN"
    echo "Run: uv sync --frozen --extra dev --extra carla"
    exit 1
fi
"$UV_BIN" sync --frozen --extra dev --extra carla

# --- Step 4: Prepare CarDreamer task definitions ---
echo "[4/6] Preparing CarDreamer task definitions..."
# CarDreamer's package metadata pins old Gym/Numpy versions. The main project
# declares compatible runtime dependencies and imports this submodule through
# PYTHONPATH, so do not run `flit install` here.
if [ ! -d "$CARDREAMER_DIR/car_dreamer" ]; then
    echo "  ERROR: CarDreamer submodule is missing."
    exit 1
fi
export PYTHONPATH="$CARDREAMER_DIR:${PYTHONPATH:-}"

# --- Step 5: Install CARLA Python API ---
echo "[5/6] Installing CARLA Python API..."
CARLA_WHEEL=""
CARLA_DIST="$CARLA_ROOT/PythonAPI/carla/dist"
if [ -d "$CARLA_DIST" ]; then
    CARLA_WHEEL=$(find "$CARLA_DIST" -name "carla-*cp310*.whl" -print -quit)
fi
if ! "$PYTHON_BIN" -c 'import pip' >/dev/null 2>&1; then
    # The uv environment may not include pip. It is useful for diagnostics,
    # but it is not required for uv's package installation path.
    if ! "$UV_BIN" pip install --python "$PYTHON_BIN" pip; then
        echo "  WARNING: pip could not be installed; continuing with uv pip."
    fi
fi
if [ -n "$CARLA_WHEEL" ]; then
    "$UV_BIN" pip install --python "$PYTHON_BIN" "$CARLA_WHEEL"
    echo "  Installed local wheel: $CARLA_WHEEL"
elif "$PYTHON_BIN" -c 'import carla' >/dev/null 2>&1; then
    echo "  CARLA API already installed by uv sync."
else
    echo "  No cp310 wheel in the simulator archive; using the PyPI package."
    "$UV_BIN" pip install --python "$PYTHON_BIN" "carla==0.9.15"
fi

# --- Step 6: Verify setup ---
echo "[6/6] Verifying setup..."

echo -n "  dreamerv3-torch: "
if [ -f "$DREAMER_TORCH_DIR/models.py" ]; then
    echo "OK (PyTorch nn.Module)"
else
    echo "MISSING — run: git submodule update --init --recursive"
fi

echo -n "  CarDreamer tasks: "
"$PYTHON_BIN" -c 'import car_dreamer; print("OK")' 2>/dev/null || echo "NOT IMPORTABLE (task definitions may not be needed for unit tests)"

echo -n "  CARLA API: "
"$PYTHON_BIN" -c 'import carla; print("OK")' 2>/dev/null || echo "NOT IMPORTABLE (need CARLA server for full training)"

echo -n "  PyTorch: "
"$PYTHON_BIN" -c 'import torch; print(f"OK (CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()})")' 2>/dev/null || echo "MISSING"

echo -n "  Unit tests: "
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m pytest tests/ -q --no-header 2>/dev/null | tail -1 || echo "FAILED"

# --- Done ---
echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Architecture:"
echo "  DreamerV3 backbone: dreamerv3-torch (PyTorch) — third_party/dreamerv3_torch/"
echo "  CARLA tasks:        CarDreamer (framework-agnostic) — third_party/CarDreamer/"
echo "  Encoder hook:       src/dreamer/cardreamer_encoder_hook.py"
echo ""
echo "Environment variables to set in your shell:"
echo "  export CARLA_ROOT=\"$CARLA_ROOT\""
echo "  export PYTHONPATH=\"\${CARLA_ROOT}/PythonAPI/carla:\${PYTHONPATH}\""
echo ""
echo "To start training:"
echo "  1. Start CARLA server:"
echo "     \$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low"
echo ""
echo "  2. Train:"
echo "     python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple"
echo ""
echo "  3. Full comparison:"
echo "     python scripts/train_cardreamer.py --comparison --task carla_right_turn_simple"
