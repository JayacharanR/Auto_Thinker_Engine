# Vision World Model Driving Agent

A multi-phase autonomous driving agent combining **self-supervised JEPA pretraining** (on comma2k19 real driving data) with **DreamerV3 world-model RL** (in CARLA simulator), culminating in a **three-way encoder comparison** that measures the value of pretrained visual representations for driving.

## Project Structure

```
├── configs/                     # YAML configs for all phases
│   ├── phase1_dreamer_baseline.yaml
│   ├── phase2_jepa_pretrain.yaml
│   └── phase3_transfer_arms.yaml
├── src/
│   ├── envs/                    # CARLA wrapper + EPDMS reward function
│   ├── jepa/                    # ViT encoder, EMA target, predictor, masking
│   ├── dreamer/                 # RSSM world model + encoder adapters
│   ├── data/                    # comma2k19 dataset loader + transforms
│   ├── eval/                    # Linear probe, driving metrics, comparison
│   └── utils/                   # Logging, checkpointing, seeding
├── scripts/
│   ├── smoke_test_carla.py      # Gate 0: CARLA sanity check
│   ├── train_phase1.py          # DreamerV3 baseline training
│   ├── train_phase2_jepa.py     # JEPA pretraining on comma2k19
│   ├── probe_phase2.py          # Linear probe evaluation
│   ├── train_phase3_arm.py      # Three-way comparison (--arm flag)
│   └── download_comma2k19.py    # Dataset download script
├── tests/                       # Unit tests
├── outputs/                     # Checkpoints, logs, videos (gitignored)
└── reports/                     # Phase writeups
```

## Phases

| Phase | Goal | Key Output |
|-------|------|-----------|
| **0** | Environment setup | CARLA smoke test passing |
| **1** | DreamerV3 baseline | RL agent completing right-turn task |
| **2** | JEPA pretraining | ViT-Small encoder trained on 33h driving video |
| **3** | Three-way comparison | CNN vs custom-JEPA vs V-JEPA2 |

## Setup

### Requirements
- Python 3.10 (pinned for CARLA/CarDreamer compatibility)
- CUDA-capable GPU with ≥16GB VRAM (A4000-class)
- 64GB RAM
- CARLA 0.9.15

### Installation

```bash
# Clone repository
git clone <this-repo>
cd driving-world-model

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (Python 3.10 will be auto-installed by uv)
uv sync

# Install optional dependencies
uv sync --extra dev      # pytest, ruff
uv sync --extra wandb    # Weights & Biases

# Install CARLA Python API
pip install <path_to_carla>/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
```

### Dataset Download

```bash
# Download comma2k19 (~100 GB, 10 chunks × ~10 GB)
uv run python scripts/download_comma2k19.py --output-dir data/comma2k19

# Download single chunk for development (~10 GB)
uv run python scripts/download_comma2k19.py --output-dir data/comma2k19 --chunks 1
```

## Running

### Phase 0: Smoke Test
```bash
# Start CARLA server first:
# ./CarlaUE4.sh -RenderOffScreen -quality-level=Low

uv run python scripts/smoke_test_carla.py
```

### Phase 1: DreamerV3 Baseline
```bash
uv run python scripts/train_phase1.py --config configs/phase1_dreamer_baseline.yaml
```

### Phase 2: JEPA Pretraining
```bash
uv run python scripts/train_phase2_jepa.py --config configs/phase2_jepa_pretrain.yaml

# Evaluate encoder quality
uv run python scripts/probe_phase2.py --checkpoint outputs/checkpoints/phase2/best.pt
```

### Phase 3: Three-way Comparison
```bash
# Run each arm with matched seeds
uv run python scripts/train_phase3_arm.py --arm cnn --seed 42
uv run python scripts/train_phase3_arm.py --arm custom_jepa --seed 42
uv run python scripts/train_phase3_arm.py --arm vjepa2 --seed 42

# Or run all seeds at once
uv run python scripts/train_phase3_arm.py --arm cnn --run-all-seeds
```

### Tests
```bash
uv run pytest tests/ -v
```

## Architecture

### JEPA (Phase 2)
- **Context Encoder**: ViT-Small (384-dim, 12 layers, 6 heads)
- **Target Encoder**: EMA copy of context encoder (momentum τ=0.996)
- **Predictor**: Lightweight transformer (192-dim, 6 layers)
- **Training**: Smooth L1 loss on masked patch predictions, no pixel reconstruction
- **Action Conditioning**: V-JEPA-AC style injection of steering/speed

### DreamerV3 (Phases 1 & 3)
- **RSSM**: 32×32 categorical latents + 512-dim GRU deterministic state
- **Actor-Critic**: 4-layer MLPs with SiLU activation
- **Reward**: EPDMS-inspired 5-term function (collision, TTC, lane-keeping, progress, traffic rules)

### Phase 3 Arms
| Arm | Encoder | Params | Frozen |
|-----|---------|--------|--------|
| CNN | DreamerV3 default | ~2M | No |
| Custom JEPA | Phase 2 ViT-S | ~22M | Yes |
| V-JEPA2 | Meta ViT-L | ~300M | Yes |

## Licenses
- **This project**: MIT
- **CARLA**: MIT
- **CarDreamer**: Apache 2.0
- **comma2k19**: MIT
- **V-JEPA2**: CC-BY-NC-4.0 (non-commercial)
