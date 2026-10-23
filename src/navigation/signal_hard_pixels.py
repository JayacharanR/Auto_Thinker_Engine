"""Training-only hard-background objective for the native bulb component."""

import math

import torch
from torch.nn import functional as F


def hard_pixel_loss(logits, targets):
    """Keep original foreground loss; add per-image hardest1% background CE."""
    errors = F.cross_entropy(logits.float(), targets.long(), reduction="none", ignore_index=255)
    foreground = (targets > 0) & (targets < 4)
    background = targets == 0
    positive = errors[foreground].mean() if foreground.any() else errors.sum() * 0
    negative = errors[background].mean() if background.any() else errors.sum() * 0
    hard = []
    for error, mask in zip(errors, background):
        values = error[mask]
        if values.numel():
            hard.append(values.topk(max(1, math.ceil(0.01 * values.numel()))).values.mean())
    rare = torch.stack(hard).mean() if hard else errors.sum() * 0
    return positive + 0.2 * negative + rare


def protected_signal_metrics(candidate, parent):
    return (
        candidate["joint"] >= parent["joint"]
        and all(candidate["towns"][t]["joint"] >= v["joint"] for t, v in parent["towns"].items())
        and all(
            candidate["classes"][c]["joint"] >= v["joint"] for c, v in parent["classes"].items()
        )
    )
