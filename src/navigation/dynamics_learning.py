"""Factual dynamics batching; future grids are separate loss targets only."""

import numpy as np
import torch


def member_population(windows, seed):
    train = np.flatnonzero(~windows["is_validation"])
    episodes = np.unique(windows["episode_id"][train])
    rng = np.random.default_rng(seed)
    collision = (
        (windows["outcomes"][train, :, 0] == 1) & (windows["cost_valid"][train, :, 0] > 0)
    ).any(1)
    positive_episodes = np.unique(windows["episode_id"][train][collision])
    negative_episodes = np.setdiff1d(episodes, positive_episodes)
    # Stratify the whole-episode bootstrap by TRAIN contact status. An ordinary
    # bootstrap omits all three positive episodes for registered seed42.
    strata = [group for group in (positive_episodes, negative_episodes) if len(group)]
    chosen = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
    groups = [train[windows["episode_id"][train] == episode] for episode in chosen]
    positive = np.concatenate([g[(windows["outcomes"][g, :, 0] == 1).any(1)] for g in groups])
    return groups, positive, chosen


def sample_batch(groups, positives, rng, batch=16):
    if batch != 16:
        raise ValueError("Registered16window batch required")
    uniform = [int(rng.choice(groups[int(rng.integers(len(groups)))])) for _ in range(batch // 2)]
    if len(positives):
        other = rng.choice(positives, size=batch // 2, replace=True).tolist()
    else:
        other = [int(rng.choice(groups[int(rng.integers(len(groups)))])) for _ in range(batch // 2)]
    return np.array(uniform + other, dtype=np.int64)


def tensor_batch(windows, features, indices, device):
    # Explicit split prevents a training caller from passing any future grid or
    # oracle outcome into the forward input mapping.
    rows = np.asarray(indices)
    present = (
        torch.from_numpy(np.asarray(features[windows["input_index"][rows]]).copy())
        .to(device)
        .float()
    )
    future = (
        torch.from_numpy(np.asarray(features[windows["future_indices"][rows]]).copy())
        .to(device)
        .float()
    )
    inputs = dict(
        past_tokens=present,
        ordered_controls=torch.from_numpy(windows["controls"][rows].copy()).to(device),
        conditioning=torch.from_numpy(windows["conditioning"][rows].copy()).to(device),
    )
    mask = windows["cost_valid"][rows].copy()
    mask[:, :, 1] = (
        0  # No positive red examples; prospective protocol forbids learning/deploying red risk.
    )
    targets = dict(
        future_tokens=future,
        latent_valid=torch.from_numpy(windows["latent_valid"][rows].copy()).to(device),
        outcomes=torch.from_numpy(windows["outcomes"][rows].copy()).to(device),
        cost_valid=torch.from_numpy(mask).to(device),
    )
    return inputs, targets
