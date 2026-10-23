"""Training-only unlabeled CARLA continuation; explicit clip and mask geometry."""

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


def location_key(episode):
    return episode["town"], tuple(episode["junction_ids"])


def unlabeled_split(manifest):
    episodes = [e for e in manifest["episodes"] if e["role"] == "train"]
    if any(e["partition"] != "train" or "Town05" in e["town"] for e in episodes):
        raise ValueError("Unlabeled pretraining requires strict TRAIN geography")
    locations = sorted(
        {location_key(e) for e in episodes},
        key=lambda key: hashlib.sha256(f"20261019:{key}".encode()).hexdigest(),
    )
    held = set(locations[: max(1, len(locations) // 5)])
    train = [e for e in episodes if location_key(e) not in held]
    validation = [e for e in episodes if location_key(e) in held]
    if not train or not validation:
        raise ValueError("Nonempty disjoint TRAIN-only unsupervised split required")
    train_locations = {(e["town"], j) for e in train for j in e["junction_ids"]}
    if train_locations & {(e["town"], j) for e in validation for j in e["junction_ids"]}:
        raise ValueError("Unsupervised location leakage")
    return train, validation


def tube_masks(batch, rng, device):
    """Mask72/144 spatial cells, identically at all four temporal tubelets."""
    order = np.stack([rng.permutation(144) for _ in range(batch)])
    context, masked = [], []
    for selection, target in ((order[:, :72], context), (order[:, 72:], masked)):
        ids = np.concatenate([selection + t * 144 for t in range(4)], axis=1)
        target.append(torch.as_tensor(np.sort(ids, axis=1), device=device, dtype=torch.long))
    return context[0], masked[0]


def sample_clips(raw, episodes, rng, count=8):
    clips, sources = [], []
    for _ in range(count):
        e = episodes[int(rng.integers(len(episodes)))]
        frames = raw.rgb(e)
        endpoint = int(rng.integers(min(7, len(frames) - 1), len(frames)))
        indices = np.maximum(0, np.arange(endpoint - 7, endpoint + 1))
        clips.append(np.asarray(frames[indices]).transpose(1, 2, 0, 3, 4).copy())
        sources.append(dict(episode=e["id"], endpoint=endpoint))
    # Each episode clip supplies three camera views; no action/rule labels read.
    return np.concatenate(clips, axis=0), sources


def normalized_video(clips, device):
    x = torch.as_tensor(clips, device=device).float() / 255
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
    return (x - mean) / std


def prediction_loss(context, teacher, predictor, video, visible, masked):
    with torch.no_grad():
        full = teacher(video)
        target = full.gather(1, masked[..., None].expand(-1, -1, full.shape[-1]))
    encoded = context(video, visible)
    predicted = predictor(encoded, visible, masked)
    return F.smooth_l1_loss(predicted.float(), target.float()), full


@torch.no_grad()
def ema_update(context, target, momentum=0.996):
    for source, destination in zip(context.parameters(), target.parameters(), strict=True):
        destination.lerp_(source, 1 - momentum)


def variance_diagnostics(tokens):
    values = tokens.float()
    # Report positional variation separately from variation between scenes.
    within = values.var(1, unbiased=False).mean()
    pooled = values.mean(1).reshape(-1, 3, values.shape[-1])
    between = pooled.var(0, unbiased=False).mean()
    return dict(within_clip_variance=float(within), between_scene_variance=float(between))
