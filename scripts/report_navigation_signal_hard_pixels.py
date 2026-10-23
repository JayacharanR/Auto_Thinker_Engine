#!/usr/bin/env python3
"""Preserve the failed coverage protection and verify unchanged parent selection."""

import json
import sys
from pathlib import Path

import matplotlib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402


def main():
    root = ROOT / "outputs/navigation_signal_hard_pixels_20261021"
    result = json.loads((root / "assessment.json").read_text())
    history = json.loads((root / "history.json").read_text())
    assert result["selected_step"] == 0 and not result["countermeasure_pass"]
    parent = torch.load(result["parent_checkpoint"], map_location="cpu", weights_only=False)
    selected = torch.load(result["selected_checkpoint"], map_location="cpu", weights_only=False)
    assert parent["model"].keys() == selected["model"].keys()
    assert all(torch.equal(v, selected["model"][k]) for k, v in parent["model"].items())
    parent_root = Path(result["parent_checkpoint"]).parent
    assert json.loads((root / "predictions0.json").read_text()) == json.loads(
        (parent_root / "predictions1000.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [r["metrics"]["joint"] for r in history], "o-", color="#006f9e")
    axes[0].axhline(44, color="black", linestyle="--", label="Parent coverage floor")
    axes[0].set_title("Correct point localization + state /72")
    axes[0].legend()
    axes[1].plot(steps, [r["metrics"]["background_peaks"] for r in history], "o-", color="#ad542f")
    axes[1].axhline(3724.1, color="black", linestyle="--", label="90% reduction target")
    axes[1].set_title("Known-background peaks /144images")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("New-objective optimizer updates")
        ax.grid(alpha=0.2)
    fig.suptitle("Fewer false alarms, but lost light coverage: reject correction")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_hard_pixels_20261021.png", dpi=150)
    plt.close(fig)
    rows = []
    for row in history:
        m = row["metrics"]
        rows.append(
            f"| {row['step']} | {m['localized']}/72 | {m['joint']}/72 | "
            f"{m['background_peaks']:,} | {100 * m['background_active_fraction']:.4f}% | "
            f"{'yes' if row['coverage_protected'] else 'no'} |"
        )
    table = "\n".join(rows)
    (
        ROOT / "reports/navigation_signal_hard_pixels_20261021.md"
    ).write_text(f"""# Hard-background correction rejected: coverage regresses

The registered1000update fit completed24.01seconds. Every trained checkpoint
fails parent light-coverage protection. Select initialization0; all parent model
tensors and frozen predictions independently match exactly. The parent itself
remains unqualified. No component or driving promotion, no native rollout.

| Updates | Localized | Joint state | Background peaks | Active background | Protected |
|---|---:|---:|---:|---:|---|
{table}

![Complete trade-off](navigation_signal_hard_pixels_20261021.png)

Latest1000reduces backgroundpeaks37,241→1,957(94.7%) and activefraction
1.6541%→0.0849%(94.9%). But correct lights fall44→27. Bytown joint counts
Town01 7→5,Town03 15→8,Town04 12→12,Town10HD10→2. Red22→14,yellow15→6,
green7→7. Never trade these losses for a misleading cleaner-looking demo.

Only the TRAINloss changed: original foreground+.2meanbackgroundCE plus
the mean hardest1%backgroundCE per image. The model remains a single-view
RGB37,772parameter/21pixel-receptive-fieldCNN; same876TRAIN/144VAL and72known
housings, source/split/thresholds unchanged. All checkpoints, optimizer/RNG and
predictions retained. The old paired experiment remains failed as originally
recorded. Seventeen focused signal/research-boundary tests pass; no full-suite
claim. Protocol [hard background](navigation_signal_hard_pixels_protocol_20261021.md).

Next: test wider RGB housing context or a separately learned candidate verifier
using TRAIN-only mined background patches. Current evidence shows a precision/
coverage trade-off; it does not prove context is the cause. Freeze the new method
and budget before fitting, preserve the same validation denominator and all
past failures, and do not tune the finalTown05suite. Signal applicability and
stop/release memory still require separate camera-driving evidence.

M2/M3/M5 remain unqualified; M6has not run. Approved actuatorfeedbackv2 and
original storagebaseline remain active. No reward, benchmark, M4primary or
allowed-input changes. Evidence: outputs/navigation_signal_hard_pixels_20261021.
""")


if __name__ == "__main__":
    main()
