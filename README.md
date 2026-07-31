# Vision World Model Driving Agent

A multi-phase autonomous driving agent combining **self-supervised JEPA pretraining** (on comma2k19 real driving data) with **DreamerV3 world-model RL** (in CARLA simulator via [CarDreamer](https://github.com/ucd-dare/CarDreamer)), culminating in a **three-way encoder comparison** that measures the value of pretrained visual representations for driving.

## Architecture Overview

```
comma2k19 video + telemetry         CARLA (via CarDreamer)
        │                                    │
  Phase 2: JEPA                    Phase 1 & 3: DreamerV3
  self-supervised                  (CarDreamer's tested loop)
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
                                    CarDreamer RSSM
                                    + Actor-Critic
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
│   ├── dreamer/                 # Encoder adapters + CarDreamer encoder hook
│   ├── data/                    # comma2k19 dataset loader + transforms
│   ├── eval/                    # Linear probe, driving metrics, comparison
│   └── utils/                   # Logging, checkpointing, seeding
├── scripts/
│   ├── setup_cardreamer.sh      # CarDreamer + CARLA setup (run first)
│   ├── train_cardreamer.py      # Phase 1 & 3: CarDreamer-based training
│   ├── train_phase2_jepa.py     # Phase 2: JEPA pretraining on comma2k19
│   ├── probe_phase2.py          # Linear probe evaluation
│   ├── visualize_representations.py  # PCA/t-SNE maneuver clustering
│   └── download_comma2k19.py    # Dataset download + verification
├── third_party/
│   └── CarDreamer/              # Git submodule (ucd-dare/CarDreamer)
├── tests/                       # Unit tests
├── outputs/                     # Checkpoints, logs, videos (gitignored)
└── reports/                     # Phase writeups + future work
```

## Phases

| Phase | Goal | Key Output |
|-------|------|-----------|
| **0** | Environment setup | CARLA + CarDreamer smoke test |
| **1** | DreamerV3 baseline | CNN agent via CarDreamer's training loop |
| **2** | JEPA pretraining | ViT-Small encoder on 33h driving video |
| **3** | Three-way comparison | CNN vs custom-JEPA vs V-JEPA2 |

## Setup

### Requirements
- Python 3.10 (pinned for CARLA/CarDreamer compatibility)
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

### Phase 1: DreamerV3 Baseline (via CarDreamer)
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

### Phase 3: Three-way Comparison (via CarDreamer)
```bash
# Run each arm with matched seeds
uv run python scripts/train_cardreamer.py --arm cnn --seed 42
uv run python scripts/train_cardreamer.py --arm custom_jepa --seed 42 \
    --jepa-checkpoint outputs/checkpoints/phase2/best.pt
uv run python scripts/train_cardreamer.py --arm vjepa2 --seed 42

# Or run full comparison (all arms × all seeds)
uv run python scripts/train_cardreamer.py --run-comparison
```

### Tests
```bash
uv run pytest tests/ -v
```

## Key Design Decisions

### CarDreamer Integration (not custom RSSM)
The original plan specified CarDreamer to avoid reimplementing RSSM/actor-critic.
Code review confirmed: custom RSSM introduced bugs that CarDreamer solves.
Our contribution layer (encoder adapter) hooks into CarDreamer at the encoder level.

### Temporal Input: Frame Stacking (Option A)
For temporal encoders (custom_jepa, vjepa2), we maintain a rolling buffer of
the last 4 CARLA frames. This preserves the scientific claim that **temporal
video pretraining transfers to control**, rather than degenerating to 2D patches.

### Resolution: 224×224 Everywhere
Phase 2 JEPA pretrains at 224×224. Phase 3 uses the same resolution.
This ensures positional embeddings match without interpolation.

## Architecture Details

### JEPA (Phase 2)
- **Context Encoder**: ViT-Small (384-dim, 12 layers, 6 heads)
- **Target Encoder**: EMA copy of context encoder (momentum τ=0.996)
- **Predictor**: Lightweight transformer (192-dim, 6 layers)
- **Training**: Smooth L1 loss on masked patch predictions, no pixel reconstruction
- **Action Conditioning**: V-JEPA-AC style injection of steering/speed

### DreamerV3 (Phases 1 & 3) — via CarDreamer
- **RSSM**: CarDreamer's tested implementation (32×32 categorical latents)
- **Actor-Critic**: CarDreamer's 4-layer MLPs
- **Reward**: CarDreamer's built-in task rewards (right-turn, traffic-light, etc.)
- **Training**: CarDreamer's replay buffer + training loop

### Phase 3 Arms
| Arm | Encoder | Params | Frozen | Input |
|-----|---------|--------|--------|-------|
| CNN | DreamerV3 default | ~2M | No | Single frame |
| Custom JEPA | Phase 2 ViT-S | ~22M | Yes | 4-frame clip |
| V-JEPA2 | Meta ViT-L | ~300M | Yes | 4-frame clip |

## Licenses
- **This project**: MIT
- **CARLA**: MIT
- **CarDreamer**: Apache 2.0
- **comma2k19**: MIT
- **V-JEPA2**: CC-BY-NC-4.0 (non-commercial)
