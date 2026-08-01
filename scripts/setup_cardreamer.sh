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

echo "============================================"
echo "  Environment Setup"
echo "============================================"

# --- Step 1: CARLA path ---
CARLA_ROOT="${1:-${CARLA_ROOT:-}}"
if [ -z "$CARLA_ROOT" ]; then
    echo "ERROR: CARLA path required."
    echo "Usage: bash scripts/setup_cardreamer.sh /path/to/carla"
    echo ""
    echo "Download CARLA 0.9.15 from:"
    echo "  https://github.com/carla-simulator/carla/releases/tag/0.9.15/"
    exit 1
fi

if [ ! -d "$CARLA_ROOT" ]; then
    echo "ERROR: CARLA directory not found: $CARLA_ROOT"
    exit 1
fi

echo "[1/6] CARLA root: $CARLA_ROOT"
export CARLA_ROOT
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}"

# --- Step 2: Init submodules ---
echo "[2/6] Initializing git submodules..."
cd "$PROJECT_ROOT"
git submodule update --init --recursive

# --- Step 3: Install project with uv ---
echo "[3/6] Installing project dependencies via uv..."
cd "$PROJECT_ROOT"
uv sync --extra dev

# --- Step 4: Install CarDreamer (task definitions only) ---
echo "[4/6] Installing CarDreamer (CARLA task definitions)..."
cd "$CARDREAMER_DIR"
# CarDreamer's car_dreamer package provides CARLA env wrappers/tasks.
# We do NOT use CarDreamer's JAX-based DreamerV3 — only the task defs.
pip install flit 2>/dev/null || true
flit install --symlink 2>/dev/null || flit install --pth-file 2>/dev/null || {
    echo "  WARNING: flit install failed. Adding CarDreamer to PYTHONPATH instead."
    export PYTHONPATH="$CARDREAMER_DIR:${PYTHONPATH:-}"
}

# --- Step 5: Install CARLA Python API ---
echo "[5/6] Installing CARLA Python API..."
CARLA_EGG=$(find "$CARLA_ROOT/PythonAPI/carla/dist" -name "carla-*cp310*.whl" 2>/dev/null | head -1)
if [ -n "$CARLA_EGG" ]; then
    pip install "$CARLA_EGG"
    echo "  Installed: $CARLA_EGG"
else
    CARLA_EGG=$(find "$CARLA_ROOT/PythonAPI/carla/dist" -name "carla-*cp310*.egg" 2>/dev/null | head -1)
    if [ -n "$CARLA_EGG" ]; then
        echo "  Found .egg: $CARLA_EGG"
        echo "  Add to PYTHONPATH or install with easy_install"
    else
        echo "  WARNING: No CARLA Python 3.10 package found in $CARLA_ROOT"
        echo "  You may need to build from source or download the correct version."
    fi
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
python -c 'import car_dreamer; print("OK")' 2>/dev/null || echo "NOT IMPORTABLE (task definitions may not be needed for unit tests)"

echo -n "  CARLA API: "
python -c 'import carla; print("OK")' 2>/dev/null || echo "NOT IMPORTABLE (need CARLA server for full training)"

echo -n "  PyTorch: "
python -c 'import torch; print(f"OK (CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()})")' 2>/dev/null || echo "MISSING"

echo -n "  Unit tests: "
cd "$PROJECT_ROOT"
uv run pytest tests/ -q --no-header 2>/dev/null | tail -1 || echo "FAILED"

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
echo "     \$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low"
echo ""
echo "  2. Train:"
echo "     python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple"
echo ""
echo "  3. Full comparison:"
echo "     python scripts/train_cardreamer.py --comparison --task carla_right_turn_simple"
