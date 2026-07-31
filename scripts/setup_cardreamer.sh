#!/usr/bin/env bash
# CarDreamer Setup Script
#
# Run this on the target hardware to set up CarDreamer integration.
# Prerequisites: CARLA 0.9.15 installed, Python 3.10, uv
#
# Usage:
#   bash scripts/setup_cardreamer.sh /path/to/carla

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CARDREAMER_DIR="$PROJECT_ROOT/third_party/CarDreamer"

echo "============================================"
echo "  CarDreamer Setup"
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

echo "[1/5] CARLA root: $CARLA_ROOT"
export CARLA_ROOT
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}"

# --- Step 2: Init submodule ---
echo "[2/5] Initializing CarDreamer submodule..."
cd "$PROJECT_ROOT"
git submodule update --init --recursive

# --- Step 3: Install CarDreamer ---
echo "[3/5] Installing CarDreamer..."
cd "$CARDREAMER_DIR"
pip install flit 2>/dev/null || true
flit install --symlink 2>/dev/null || flit install --pth-file

# --- Step 4: Install DreamerV3 dependencies ---
echo "[4/5] Installing DreamerV3 dependencies..."
cd "$CARDREAMER_DIR/dreamerv3"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "setup.py" ]; then
    pip install -e .
else
    echo "WARNING: No requirements.txt or setup.py found in dreamerv3/"
    echo "         You may need to install DreamerV3 dependencies manually."
fi

# --- Step 5: Install CARLA Python API ---
echo "[5/5] Installing CARLA Python API..."
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

# --- Done ---
echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Environment variables to set in your shell:"
echo "  export CARLA_ROOT=\"$CARLA_ROOT\""
echo "  export PYTHONPATH=\"\${CARLA_ROOT}/PythonAPI/carla:\${PYTHONPATH}\""
echo ""
echo "To start training:"
echo "  1. Start CARLA server:"
echo "     \$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low"
echo ""
echo "  2. Run smoke test:"
echo "     python -c 'import car_dreamer; print(\"CarDreamer OK\")'"
echo "     python -c 'import carla; print(\"CARLA API OK\")'"
echo ""
echo "  3. Train:"
echo "     python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple"
