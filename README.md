# Vision World Model Driving Agent

A multi-phase autonomous driving agent combining **self-supervised JEPA pretraining** (on comma2k19 real driving data) with **DreamerV3 world-model RL** (in CARLA simulator), culminating in a **three-way encoder comparison** that measures the value of pretrained visual representations for driving.

## Architecture Overview

```
comma2k19 video + telemetry         CARLA (via CarDreamer tasks)
        │                                    │
  Phase 2: JEPA                    Phase 1 & 3: DreamerV3
  self-supervised                  (dreamerv3-torch, PyTorch)
  pretraining                              │
        │                           ┌──────┼──────┐
        ▼                           ▼      ▼      ▼
  ViT-S encoder ──────────────► [Arm 2] [Arm 1] [Arm 3]
  (frozen weights)               JEPA    CNN    V-JEPA2
                                   │      │       │
                                   └──────┼───────┘
                                          ▼
                                  EncoderAdapter (same capacity)
                                          │
                                          ▼
                               dreamerv3-torch RSSM
                               + Actor-Critic (PyTorch)
                                          │
                                          ▼
                                   Driving Actions
```

## Project Structure

```
├── configs/                     # YAML configs for all phases
│   ├── phase1_dreamer_baseline.yaml
│   ├── phase2_jepa_pretrain.yaml
│   └── phase3_transfer_arms.yaml
├── src/
│   ├── jepa/                    # ViT encoder, EMA target, predictor, masking
│   ├── dreamer/                 # Encoder adapters + dreamerv3-torch hook
│   ├── data/                    # comma2k19 dataset loader + transforms
│   ├── eval/                    # Linear probe, driving metrics, comparison
│   └── utils/                   # Logging, checkpointing, seeding
├── scripts/
│   ├── setup_cardreamer.sh      # CarDreamer + CARLA setup (run first)
│   ├── train_cardreamer.py      # Phase 1 & 3: DreamerV3 training
│   ├── train_phase2_jepa.py     # Phase 2: JEPA pretraining on comma2k19
│   ├── probe_phase2.py          # Linear probe evaluation
│   ├── visualize_representations.py  # PCA/t-SNE maneuver clustering
│   └── download_comma2k19.py    # Dataset download + verification
├── third_party/
│   ├── dreamerv3_torch/         # Git submodule (NM512/dreamerv3-torch, PyTorch)
│   └── CarDreamer/              # Git submodule (CARLA task definitions only)
├── tests/                       # Unit tests (73 passing)
├── outputs/                     # Checkpoints, logs, videos (gitignored)
└── reports/                     # Phase writeups + future work
```

## Phases

| Phase | Goal | Key Output |
|-------|------|-----------|
| **0** | Environment setup | CARLA + dreamerv3-torch smoke test |
| **1** | DreamerV3 baseline | CNN agent via dreamerv3-torch training loop |
| **2** | JEPA pretraining | ViT-Small encoder on 33h driving video |
| **3** | Three-way comparison | CNN vs custom-JEPA vs V-JEPA2 |

## Setup

### Requirements
- Python 3.10 (pinned for CARLA compatibility)
- CUDA-capable GPU with ≥16GB VRAM (A4000-class)
- 64GB RAM
- CARLA 0.9.15

### Installation

```bash
# Clone repository with submodules
git clone --recursive <this-repo>
cd Auto_Thinker_Engine

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (Python 3.10 will be auto-installed by uv)
uv sync

# Install optional dependencies
uv sync --extra dev      # pytest, ruff
uv sync --extra wandb    # Weights & Biases
uv sync --extra viz      # UMAP, seaborn

# Set up CarDreamer + CARLA (on target hardware)
bash scripts/setup_cardreamer.sh /path/to/carla
```

### Dataset Download

```bash
# Download comma2k19 (~100 GB, 10 chunks × ~10 GB)
uv run python scripts/download_comma2k19.py --output-dir data/comma2k19

# Download single chunk for development (~10 GB)
uv run python scripts/download_comma2k19.py --output-dir data/comma2k19 --chunks 1

# Verify downloaded data (checks CAN field completeness)
uv run python scripts/download_comma2k19.py --output-dir data/comma2k19 --verify
```

## Running

### Phase 1: DreamerV3 Baseline
```bash
# Start CARLA server first:
# $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -quality-level=Low

uv run python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple
```

### Phase 2: JEPA Pretraining
```bash
uv run python scripts/train_phase2_jepa.py --config configs/phase2_jepa_pretrain.yaml

# Evaluate encoder quality
uv run python scripts/probe_phase2.py --checkpoint outputs/checkpoints/phase2/best.pt

# Visualize representation clusters
uv run python scripts/visualize_representations.py \
    --checkpoint outputs/checkpoints/phase2/best.pt \
    --data-dir data/comma2k19
```

### Phase 3: Three-way Comparison
```bash
# Run each arm with matched seeds
uv run python scripts/train_cardreamer.py --arm cnn --seed 42
uv run python scripts/train_cardreamer.py --arm custom_jepa --seed 42
uv run python scripts/train_cardreamer.py --arm vjepa2 --seed 42

# Or run full comparison (all arms × all seeds)
uv run python scripts/train_cardreamer.py --comparison
```

### Tests
```bash
uv run pytest tests/ -v
```

## Key Design Decisions

### DreamerV3 Backbone: dreamerv3-torch (PyTorch)
The project initially attempted to use CarDreamer's JAX-based DreamerV3. Code review
identified a **framework mismatch**: our PyTorch encoders cannot be inserted into
JAX's JIT-compiled function graph. We pivoted to `NM512/dreamerv3-torch`, a well-tested
PyTorch implementation. CarDreamer's CARLA task definitions (reward functions, route
logic) are still used — they're framework-agnostic Python.

### Temporal Input: Frame Stacking (Option A)
For temporal encoders (custom_jepa, vjepa2), we maintain a rolling buffer of
the last 4 CARLA frames. This preserves the scientific claim that **temporal
video pretraining transfers to control**, rather than degenerating to 2D patches.
Clips always satisfy `T >= tubelet_size` to avoid 3D convolution errors.

### Resolution: Per-arm
- JEPA/V-JEPA2: 224×224 (matching pretrained positional embeddings)
- CNN: 64×64 (dreamerv3-torch default, saves VRAM)

Temporal PE interpolation handles frame-count mismatches between Phase 2
pretraining (16 frames) and Phase 3 RL (4-frame clips).

## Architecture Details

### JEPA (Phase 2)
- **Context Encoder**: ViT-Small (384-dim, 12 layers, 6 heads)
- **Target Encoder**: EMA copy of context encoder (momentum τ=0.996)
- **Predictor**: Lightweight transformer (192-dim, 6 layers)
- **Training**: Smooth L1 loss on masked patch predictions, no pixel reconstruction
- **Action Conditioning**: V-JEPA-AC style injection of steering/speed

### DreamerV3 (Phases 1 & 3) — via dreamerv3-torch
- **RSSM**: PyTorch implementation (32×32 categorical latents)
- **Actor-Critic**: Learned, symlog returns
- **Reward**: CarDreamer's built-in task rewards (right-turn, traffic-light, etc.)
- **Encoder Swap**: `patch_dreamerv3_encoder()` replaces `MultiEncoder` (both nn.Module)

### Phase 3 Arms
| Arm | Encoder | Params | Frozen | Input |
|-----|---------|--------|--------|-------|
| CNN | DreamerV3 default | ~2M | No | Single frame (64×64) |
| Custom JEPA | Phase 2 ViT-S | ~22M | Yes | 4-frame clip (224×224) |
| V-JEPA2 | Meta ViT-L | ~300M | Yes | 4-frame clip (224×224) |

## Licenses
- **This project**: MIT
- **CARLA**: MIT
- **dreamerv3-torch**: MIT
- **CarDreamer**: Apache 2.0
- **comma2k19**: MIT
- **V-JEPA2**: CC-BY-NC-4.0 (non-commercial)
