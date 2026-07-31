"""
Deterministic seeding for reproducible experiments.

Sets seeds for torch, numpy, random, and optionally CARLA traffic manager.
All Phase 3 comparison arms MUST use the same seed for valid comparison.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Recorded in run configs for Phase 3
              seed-matched comparisons.
        deterministic_cudnn: If True, forces deterministic cuDNN algorithms.
            This may reduce throughput by ~10-15% but ensures exact
            reproducibility. Recommended for Phase 3 comparison runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU (future-proofing)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN to auto-tune for faster training
        # (non-deterministic but ~10-15% faster)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_carla_traffic_manager(traffic_manager, seed: int) -> None:
    """
    Set CARLA traffic manager seed for reproducible traffic patterns.

    Args:
        traffic_manager: CARLA TrafficManager instance.
        seed: Must match the seed used in seed_everything() for this run.
    """
    traffic_manager.set_random_device_seed(seed)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a seeded torch Generator for DataLoader worker reproducibility.

    Usage:
        g = get_generator(seed)
        DataLoader(dataset, generator=g, worker_init_fn=worker_init_fn)

    Args:
        seed: Seed value.

    Returns:
        Seeded torch.Generator instance.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker init function for reproducible data loading.

    Each worker gets a deterministic seed derived from the base seed
    and its worker ID, ensuring different but reproducible sequences
    per worker.

    Usage:
        DataLoader(dataset, worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
