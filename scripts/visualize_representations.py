"""
PCA/t-SNE/UMAP Visualization of Encoder Representations.

Generates maneuver-cluster visualizations for Phase 2 evaluation.
This was flagged as missing in the code review (#11).

Usage:
    python scripts/visualize_representations.py \
        --checkpoint outputs/checkpoints/phase2/best.pt \
        --data-dir data/comma2k19 \
        --output-dir outputs/visualizations
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jepa.encoder import ViTEncoder
from src.data.comma2k19_dataset import Comma2k19Dataset
from src.data.transforms import VideoTransform


def extract_representations(
    encoder: ViTEncoder,
    dataset: Comma2k19Dataset,
    device: str,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract encoder representations and telemetry from dataset.

    Returns:
        representations: (N, D) encoder outputs
        speeds: (N,) speed values
        steering: (N,) steering angle values
    """
    encoder.eval()

    representations = []
    speeds = []
    steering_angles = []

    indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in indices:
            sample = dataset[int(idx)]
            video = sample["video"].unsqueeze(0).to(device)  # (1, C, T, H, W)
            telemetry = sample["telemetry"]  # (T, A)

            # Encode
            features = encoder(video)  # (1, N, D) or (1, D)
            if features.dim() == 3:
                features = features.mean(dim=1)  # (1, D)

            representations.append(features.cpu().numpy().squeeze())

            # Extract telemetry (mean over time)
            if telemetry.shape[-1] >= 1:
                steering_angles.append(telemetry[:, 0].mean().item())
            else:
                steering_angles.append(0.0)

            if telemetry.shape[-1] >= 2:
                speeds.append(telemetry[:, 1].mean().item())
            else:
                speeds.append(0.0)

    return (
        np.stack(representations),
        np.array(speeds),
        np.array(steering_angles),
    )


def categorize_maneuvers(
    steering: np.ndarray,
    speed: np.ndarray,
) -> np.ndarray:
    """
    Assign maneuver labels based on telemetry.

    Categories:
        0: Straight + cruising
        1: Left turn
        2: Right turn
        3: Stopped/slow
        4: Acceleration/high speed
    """
    labels = np.zeros(len(steering), dtype=int)

    # Thresholds (in degrees, approximately)
    steer_threshold = 5.0  # degrees
    slow_threshold = 2.0  # m/s
    fast_threshold = 25.0  # m/s

    for i in range(len(steering)):
        if speed[i] < slow_threshold:
            labels[i] = 3  # Stopped
        elif abs(steering[i]) < steer_threshold:
            if speed[i] > fast_threshold:
                labels[i] = 4  # High speed
            else:
                labels[i] = 0  # Straight cruising
        elif steering[i] > steer_threshold:
            labels[i] = 1  # Left turn
        else:
            labels[i] = 2  # Right turn

    return labels


def plot_2d_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
    label_names: list[str] = None,
):
    """Plot a 2D embedding colored by maneuver labels."""
    if label_names is None:
        label_names = ["Straight", "Left turn", "Right turn", "Stopped", "High speed"]

    colors = ["#4A90D9", "#E8744A", "#48B685", "#9B59B6", "#F5A623"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=colors[i % len(colors)],
                label=f"{name} (n={mask.sum()})",
                alpha=0.6,
                s=15,
                edgecolors="none",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def run_pca(representations: np.ndarray, labels: np.ndarray, output_dir: str):
    """Generate PCA visualization."""
    from sklearn.decomposition import PCA

    print("  Running PCA...")
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(representations)

    var_explained = pca.explained_variance_ratio_
    title = f"PCA of JEPA Representations\n(PC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%})"

    plot_2d_embedding(
        embedding, labels, title,
        os.path.join(output_dir, "pca_maneuver_clusters.png"),
    )


def run_tsne(representations: np.ndarray, labels: np.ndarray, output_dir: str):
    """Generate t-SNE visualization."""
    from sklearn.manifold import TSNE

    print("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embedding = tsne.fit_transform(representations)

    plot_2d_embedding(
        embedding, labels,
        "t-SNE of JEPA Representations",
        os.path.join(output_dir, "tsne_maneuver_clusters.png"),
    )


def run_umap(representations: np.ndarray, labels: np.ndarray, output_dir: str):
    """Generate UMAP visualization (if umap-learn is installed)."""
    try:
        import umap
    except ImportError:
        print("  UMAP not installed. Install with: uv sync --extra viz")
        return

    print("  Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(representations)

    plot_2d_embedding(
        embedding, labels,
        "UMAP of JEPA Representations",
        os.path.join(output_dir, "umap_maneuver_clusters.png"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize encoder representations with PCA/t-SNE/UMAP"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Phase 2 encoder checkpoint")
    parser.add_argument("--data-dir", default="data/comma2k19",
                        help="Path to comma2k19 dataset")
    parser.add_argument("--output-dir", default="outputs/visualizations",
                        help="Output directory for plots")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Maximum samples to visualize")
    parser.add_argument("--methods", nargs="+", default=["pca", "tsne"],
                        choices=["pca", "tsne", "umap"],
                        help="Visualization methods to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    print("Loading encoder checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    encoder = ViTEncoder(
        img_size=224, patch_size=16, tubelet_size=2,
        embed_dim=384, depth=12, num_heads=6, num_frames=16,
    ).to(device)
    encoder.load_state_dict(ckpt["context_encoder_state_dict"])
    print(f"  Loaded from: {args.checkpoint}")

    # Load dataset
    print("Loading dataset...")
    transform = VideoTransform(spatial_size=224, is_train=False)
    dataset = Comma2k19Dataset(
        root_dir=args.data_dir,
        split="val",
        transform=transform,
        num_frames=16,
    )
    print(f"  Dataset size: {len(dataset)}")

    # Extract representations
    print("Extracting representations...")
    representations, speeds, steering = extract_representations(
        encoder, dataset, device, args.max_samples
    )
    print(f"  Shape: {representations.shape}")

    # Categorize maneuvers
    labels = categorize_maneuvers(steering, speeds)
    unique, counts = np.unique(labels, return_counts=True)
    print("  Maneuver distribution:")
    names = ["Straight", "Left turn", "Right turn", "Stopped", "High speed"]
    for u, c in zip(unique, counts):
        print(f"    {names[u]}: {c}")

    # Run visualizations
    print("\nGenerating visualizations...")
    if "pca" in args.methods:
        run_pca(representations, labels, args.output_dir)
    if "tsne" in args.methods:
        run_tsne(representations, labels, args.output_dir)
    if "umap" in args.methods:
        run_umap(representations, labels, args.output_dir)

    print(f"\nDone! Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
