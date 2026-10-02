"""
Linear Probe for evaluating JEPA encoder quality.

Freezes the pretrained encoder and fits a single linear map on top to
regress the steering angle of held-out comma2k19 clips.

CRITICAL: Always run the identical probe on an UNTRAINED (randomly
initialized) encoder of the same architecture as a control.
Report BOTH numbers — the delta is the evidence, not the trained
number alone.

Each encoder encodes every clip once (deterministic clips, no augmentation,
patch tokens mean-pooled); the probe is then closed-form ridge regression on
standardised features, with the ridge strength chosen on a held-out part of
the training clips. Clips from one drive are near-duplicates, so that held-out
part is whole segments, never random clips (which would reward memorising
drives). This is exact, reproducible and takes seconds, instead of
re-encoding the dataset for every probe epoch.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

RIDGE_ALPHAS = tuple(10.0 ** k for k in range(-2, 6))


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode every clip once.

    Returns:
        (N, D) mean-pooled encoder features, (N,) targets (the clip's mean
        normalised steering angle) and (N,) segment ids of the clips.
    """
    encoder.eval()
    features, targets, groups = [], [], []
    use_amp = str(device).startswith("cuda")
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = encoder(video)
        if out.dim() == 3:
            out = out.mean(dim=1)  # pool patch tokens
        features.append(out.float().cpu().numpy())
        targets.append(batch["telemetry"][:, :, 0].mean(dim=1).numpy())
        groups.extend(batch["segment_path"])
    return np.concatenate(features), np.concatenate(targets), np.asarray(groups)


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge regression (intercept not penalised)."""
    x_mean, y_mean = x.mean(0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    gram = xc.T @ xc + alpha * np.eye(x.shape[1])
    w = np.linalg.solve(gram, xc.T @ yc)
    return w, float(y_mean - x_mean @ w)


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    train_groups: Optional[np.ndarray] = None,
    alphas=RIDGE_ALPHAS,
    holdout: float = 0.15,
    seed: int = 0,
) -> dict[str, float]:
    """
    Fit ridge regression on training features and score it on validation.

    The ridge strength is chosen by R2 on a ``holdout`` fraction of the
    training groups (segments; each clip is its own group if none are given),
    then the probe is refitted on all training clips.

    Returns:
        Dict with validation 'mae', 'r2', 'loss' (MSE) and the chosen 'alpha'.
    """
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    train_x, val_x = (train_x - mean) / std, (val_x - mean) / std

    if train_groups is None:
        train_groups = np.arange(len(train_x))
    unique = np.unique(train_groups)
    held = np.random.RandomState(seed).permutation(unique)[: max(1, int(len(unique) * holdout))]
    in_holdout = np.isin(train_groups, held)
    fit_idx, sel_idx = np.where(~in_holdout)[0], np.where(in_holdout)[0]

    def score(alpha: float) -> float:
        w, b = _ridge(train_x[fit_idx], train_y[fit_idx], alpha)
        return r2_score(train_y[sel_idx], train_x[sel_idx] @ w + b) if len(sel_idx) > 1 else 0.0

    alpha = max(alphas, key=score)
    w, b = _ridge(train_x, train_y, alpha)
    prediction = val_x @ w + b
    return {
        "mae": float(mean_absolute_error(val_y, prediction)),
        "r2": float(r2_score(val_y, prediction)),
        "loss": float(np.mean((prediction - val_y) ** 2)),
        "alpha": float(alpha),
    }


def probe_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """Encode both splits once, then fit and score the linear probe."""
    train_x, train_y, train_groups = extract_features(encoder, train_loader, device)
    val_x, val_y, _ = extract_features(encoder, val_loader, device)
    results = fit_linear_probe(train_x, train_y, val_x, val_y, train_groups)
    results["num_train"], results["num_val"] = len(train_y), len(val_y)
    return results


def run_probe_comparison(
    trained_encoder: nn.Module,
    encoder_class: type,
    encoder_kwargs: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[dict] = None,
    device: str = "cuda",
    logger: Optional[object] = None,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """
    Run the critical trained vs random-init probe comparison.

    This is the actual proof of JEPA pretraining quality. The trained
    encoder should meaningfully beat the random-init control on steering
    prediction. If it doesn't, pretraining didn't learn useful representations.

    Args:
        trained_encoder: The pretrained JEPA encoder.
        encoder_class: Class to instantiate for random baseline.
        encoder_kwargs: Kwargs for random encoder instantiation.
        train_loader / val_loader: Deterministic loaders (no augmentation).
        config: Probe config (kept for interface compatibility).
        device: Torch device.
        logger: Optional logger.
        seed: Seed of the random-init control encoder.

    Returns:
        Dict with 'trained' and 'random' result dicts, each containing
        'mae', 'r2', 'loss' and 'alpha'.
    """
    print("=" * 60)
    print("LINEAR PROBE COMPARISON: Trained vs Random-Init Encoder")
    print("=" * 60)

    print("\n--- Probing TRAINED encoder ---")
    trained_results = probe_encoder(trained_encoder, train_loader, val_loader, device)

    print("--- Probing RANDOM-INIT encoder (control) ---")
    torch.manual_seed(seed)
    random_encoder = encoder_class(**encoder_kwargs).to(device)
    random_results = probe_encoder(random_encoder, train_loader, val_loader, device)

    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"  Trained encoder MAE:   {trained_results['mae']:.4f}")
    print(f"  Random  encoder MAE:   {random_results['mae']:.4f}")
    print(f"  Delta MAE:             {random_results['mae'] - trained_results['mae']:.4f}")
    print(f"  Trained encoder R²:    {trained_results['r2']:.4f}")
    print(f"  Random  encoder R²:    {random_results['r2']:.4f}")
    print(f"  Delta R²:              {trained_results['r2'] - random_results['r2']:.4f}")

    if logger is not None:
        for name, results in (("trained", trained_results), ("random", random_results)):
            logger.log_scalar(f"probe/{name}_val_mae", results["mae"], 0)
            logger.log_scalar(f"probe/{name}_val_r2", results["r2"], 0)

    if trained_results["mae"] >= random_results["mae"]:
        print("\n  ⚠️  Trained encoder does NOT beat random-init on steering MAE.")
        print("  Possible causes:")
        print("    1. Representation collapse (check collapse monitor logs)")
        print("    2. Insufficient pretraining")
        print("    3. Wrong EMA momentum / SIGReg weight, or predictor capacity")

    return {"trained": trained_results, "random": random_results}
