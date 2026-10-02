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
uv sync --extra dev --extra carla  # pytest, ruff, CARLA Python API
uv sync --extra wandb    # Weights & Biases
uv sync --extra viz      # UMAP, seaborn

# Set up CarDreamer + CARLA (on target hardware)
bash scripts/setup_cardreamer.sh /path/to/carla
```

For the A4000 Docker path, use the checked-in [CARLA Docker runbook](DOCKER_CARLA_RUNBOOK.md).
It builds a non-root Python 3.10 image and starts the container with the
graphics-capable NVIDIA runtime, a 2 GiB `/dev/shm`, Vulkan ICD validation,
dummy audio, and CARLA 0.9.15 mounted at `/opt/carla`. A CUDA-only container
with `NVIDIA_DRIVER_CAPABILITIES=compute,utility` is not sufficient for CARLA.

### Laptop (native, no Docker)

Tested on an RTX 5070 Ti Laptop GPU (12 GB, Blackwell sm_120) under Arch/CachyOS.
Blackwell needs the CUDA 12.8 torch build pinned in `pyproject.toml`.

```bash
uv sync --extra dev --extra carla --extra viz
git submodule update --init --recursive
# CARLA 0.9.15 (8.4 GB; the tiny.carla.org link may return 403, use the origin):
mkdir -p ~/software/CARLA_0.9.15 && cd ~/software
curl -L -o CARLA_0.9.15.tar.gz \
  https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz -C CARLA_0.9.15 && cd -
export CARLA_ROOT=~/software/CARLA_0.9.15
bash scripts/setup_cardreamer.sh "$CARLA_ROOT"   # applies patches/*.patch to the submodules
bash scripts/configure_carla.sh                    # Town03_Opt as the startup map
uv run python run.py doctor --require-cuda --require-carla
RUN_MODE=smoke bash jobs/slurm_run.sh              # starts, checks and stops CARLA
```

`jobs/slurm_run.sh` runs without Slurm: it launches CARLA headless, waits for
it, runs the requested mode and always stops CARLA. CARLA at Low quality uses
about 2 GB of VRAM; a Dreamer run (batch 16 x 64, AMP) about 4 GB. Train on AC
power. Long runs: start them detached, e.g. `setsid nohup bash jobs/slurm_run.sh &`.

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
# First working car: bird's-eye view + discrete actions (the launcher manages CARLA)
RUN_MODE=cnn STEPS=100000 TRAIN_ARGS="--logdir outputs/logs/cnn_bev_seed42" \
  bash jobs/slurm_run.sh

# Short integration run with config overrides; resume an interrupted run
RUN_MODE=cnn STEPS=2000 TRAIN_ARGS="--set prefill=500 --set eval_every=1000" bash jobs/slurm_run.sh
RUN_MODE=cnn STEPS=100000 TRAIN_ARGS="--resume --logdir outputs/logs/cnn_bev_seed42" \
  bash jobs/slurm_run.sh
```

Dreamer settings come from `configs/laptop.yaml` (`--profile`, `--set key=value`).
Monitoring and diagnosis (read-only, safe during training):

```bash
python3 scripts/watch_run.py                                   # live progress bar of the latest run
uv run python scripts/inspect_episode.py outputs/logs/cnn_bev_seed42 --eval -n 1
#   -> frames before the episode ended + decoded action shares (straight/left/right)
```

Long jobs (Phase 2 EMA + SIGReg, probes, encoder selection, Stage 3 resume) are
queued in `jobs/long_runs.sh`; it resumes if restarted, logs progress to
`outputs/long_runs/status.txt` and machine health to `outputs/long_runs/health.log`,
and stops the trainers cleanly (checkpoints saved) if the GPU overheats or RAM or
disk run low:

```bash
mkdir -p outputs/long_runs
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why=training \
  bash jobs/long_runs.sh > outputs/long_runs/run.log 2>&1 &
touch outputs/long_runs/STOP     # stop cleanly; rerun the command above to resume
```
Each run writes `latest.pt`, `episodes.jsonl` (one line per episode),
`eval.jsonl` and `metrics.json` to its log directory.

### Phase 2: JEPA Pretraining
```bash
# Decode the videos once (128 px, 10 Hz, memory-mapped arrays)
uv run python scripts/preprocess_comma2k19.py --src data/comma2k19 --out data/comma2k19_128

# Same budget for both collapse-prevention methods
uv run python scripts/train_phase2_jepa.py --config configs/phase2_jepa_laptop.yaml \
    --regularizer ema --checkpoint-dir outputs/checkpoints/phase2_ema
uv run python scripts/train_phase2_jepa.py --config configs/phase2_jepa_laptop.yaml \
    --regularizer sigreg --checkpoint-dir outputs/checkpoints/phase2_sigreg

# Linear steering probe vs a random encoder; keep the larger gain as phase2/best.pt
uv run python scripts/probe_phase2.py --checkpoint outputs/checkpoints/phase2_ema/best.pt
uv run python scripts/probe_phase2.py --checkpoint outputs/checkpoints/phase2_sigreg/best.pt

# Visualize representation clusters
uv run python scripts/visualize_representations.py \
    --checkpoint outputs/checkpoints/phase2/best.pt \
    --data-dir data/comma2k19
```

### Phase 3: Three-way Comparison
Front camera + route vector, continuous actions. Frozen encoders run once per
environment step and replay stores their embeddings, so training never runs
the ViT.
```bash
# One arm (2k-step check)
RUN_MODE=custom_jepa OBS=camera_route ACTION=continuous STEPS=2000 bash jobs/slurm_run.sh

# All arms x seeds; finished runs are reused and interrupted runs resume
RUN_MODE=comparison OBS=camera_route ACTION=continuous STEPS=200000 bash jobs/slurm_run.sh
# -> outputs/comparison/<task>_camera_route/comparison.md (mean ± std per arm)
```

### Tests
```bash
uv run pytest tests/ -v
```

### Terminal / HPC launcher

The project does not need a graphical terminal session. Use the root launcher
to run the existing scripts through one stable command:

```bash
python run.py doctor
python run.py smoke --host localhost --port 2000
python run.py train --arm cnn --task carla_right_turn_simple --steps 10000
python run.py phase2 --config configs/phase2_jepa_pretrain.yaml
python run.py compare --steps 500000
```

For a Slurm allocation, copy CARLA to persistent storage and submit the
headless job template. The default is a deliberately short 10,000-step CNN
validation run:

```bash
CARLA_ROOT=/scratch/$USER/software \
RUN_MODE=cnn STEPS=10000 \
sbatch jobs/slurm_run.sh
```

Other modes are `phase2`, `smoke`, `custom_jepa`, `vjepa2`, and `comparison`.
Keep CARLA, datasets, checkpoints, and `outputs/` outside temporary job-local
storage when allocations can be preempted.

For the complete runbook tailored to the Ubuntu 22.04 + RTX A4000 container,
see [SERVER_RUNBOOK_A4000.md](SERVER_RUNBOOK_A4000.md).
For the exact Docker build, `docker run`, and Gate 0 smoke commands, see
[DOCKER_CARLA_RUNBOOK.md](DOCKER_CARLA_RUNBOOK.md).

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
