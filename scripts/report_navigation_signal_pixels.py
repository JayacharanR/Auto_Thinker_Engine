#!/usr/bin/env python3
"""Report all registered native-pixel checkpoints, independently recount peaks."""

import json
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402

from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa:E402


def main():
    root = ROOT / "outputs/navigation_signal_pixel_study_20261021"
    result = json.loads((root / "assessment.json").read_text())
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    rows = []
    for arm, color in (("center", "#9a5b34"), ("bulb", "#006f9e")):
        history = json.loads((root / arm / "history.json").read_text())
        steps = [r["step"] for r in history]
        metrics = [r["metrics"] for r in history]
        for ax, field in zip(axes, ("selection_loss", "joint", "background_peaks")):
            ax.plot(steps, [m[field] for m in metrics], "o-", label=arm, color=color)
        chosen = result["arms"][arm]
        predictions = json.loads(
            (root / arm / f"predictions{chosen['selected_step']}.json").read_text()
        )
        count = dict(localized=0, joint=0, background_peaks=0)
        for sample in predictions:
            index = sample["sample"]
            available = set(range(len(data.boxes[index])))
            for peak in sample["peaks"]:
                eligible = [
                    i
                    for i in sorted(available)
                    if data.boxes[index][i][0] <= peak["x"] < data.boxes[index][i][2]
                    and data.boxes[index][i][1] <= peak["y"] < data.boxes[index][i][3]
                ]
                if eligible:
                    label = eligible[0]
                    available.remove(label)
                    count["localized"] += 1
                    count["joint"] += peak["cls"] == data.boxes[index][label][4]
                else:
                    count["background_peaks"] += bool(data.background[index, peak["y"], peak["x"]])
        assert all(count[k] == chosen["selected_metrics"][k] for k in count)
        for step, metric in zip(steps, metrics):
            rows.append(
                f"| {arm} | {step} | {metric['selection_loss']:.4f} | "
                f"{metric['localized']}/72 | {metric['joint']}/72 | "
                f"{metric['background_peaks']:,} |"
            )
    for ax, title in zip(
        axes,
        (
            "Common selection loss ↓",
            "Localized + correct state /72 ↑",
            "Known-background peaks /144 images ↓",
        ),
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Bulb supervision helps localization; background rejection still fails")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_pixels_20261021.png", dpi=160)
    plt.close(fig)
    # Examples are fixed by first known red housing per validation town, not success.
    selected = result["arms"]["bulb"]["selected_step"]
    predictions = json.loads((root / "bulb" / f"predictions{selected}.json").read_text())
    pred_map = {row["sample"]: row["peaks"] for row in predictions}
    examples = {}
    for index in data.validation:
        sample = data.samples[index]
        if sample["phase"] == "red" and data.boxes[index]:
            examples.setdefault(sample["town"], index)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    for ax, (town, index) in zip(axes.flat, sorted(examples.items())):
        ax.imshow(data.images[index].transpose(1, 2, 0))
        peaks = pred_map[index]
        if peaks:
            ax.scatter(
                [p["x"] for p in peaks],
                [p["y"] for p in peaks],
                s=8,
                facecolors="none",
                edgecolors="magenta",
                linewidths=0.6,
            )
        for x0, y0, x1, y1, _ in data.boxes[index]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan"))
        ax.set_title(f"{town}: case{data.samples[index]['case_id']} {data.samples[index]['view']}")
        ax.axis("off")
    fig.suptitle("All predicted peaks (magenta); known housing labels (cyan). No filtering.")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_pixel_errors_20261021.png", dpi=150)
    plt.close(fig)
    table = "\n".join(rows)
    (
        ROOT / "reports/navigation_signal_pixels_20261021.md"
    ).write_text(f"""# Native bulb labels: better localization, rejected background behavior

The bulb-supervised model localizes56/72known lights and correctly classifies44,
but generates37,241known-background peaks across144images (258.6/image).
It fails both registered background-protection checks; no driving deployment.
The center control selects its untrained step0 by the frozen common loss.
Do not change that selection after seeing the point-localization result.

![All scheduled checkpoints](navigation_signal_pixels_20261021.png)

| Arm | Updates | Common loss ↓ | Localized | Localized + state | Background peaks ↓ |
|---|---:|---:|---:|---:|---:|
{table}

Fresh37,772parameter native-stride1 CNNs,1000updates each, same876TRAINimages
from73poses/sevenlocations. Original144VALimages/72known housings unchanged.
The paired-label audit checked1800original images and excluded65oldTRAINposes:
63at newly reserved locations, two actual poses inside reservedTown02junctions
242/132 despite nominal298labels. All92red/green and90yelloweligible housings
have stable phase-response targets. No validation pixel labels were constructed.

Both arms use single-view RGB only, no coordinates/map/phase-pair input at
inference. The audit's paired phases/geometry are offline label construction.
Center and bulb arms use the same architecture/optimizer/updates and sampling
distribution; positive crop positions differ with the supervision. Original
four validation junctions were inspected previously, so this is development.

Bulb selected1000: town joint countsTown01 7/8,Town03 15/20,Town04 12/12,
Town10HD10/32; red22/36,yellow15/18,green7/18. Background-active fraction1.654%.
Original housing-center accuracy22/72is not the bulb task. This point-in-box
metric is NOTbounding-boxIoU/AP and cannot be compared directly to old44/72
box detections. Background labels outside projected light regions are approximate.
All predictions in other ignored light regions remain unknown, not false positives.

Execution28.53s including preparation; center15.28s/bulb5.37s with cold-start
overhead differences. Three-viewGPUforward p95center3.00ms/bulb3.17ms,
20timed calls after5warmups; excludesCPUpeak extraction/simulator/IPC.
Every scheduled checkpoint includes optimizer/RNG and explicitBF16(no scaler).
Independent closed-prediction recount reproduces selected localized/joint/background
counts. Fivefocused tests and dataset preflight passed, not a full-suite claim.

![Predetermined complete validation views](navigation_signal_pixel_errors_20261021.png)

Next investigate TRAIN-only hard background pixels with the fixed bulb labels.
The mean background loss can hide rare confident peaks among mostly easy pixels;
this is a testable loss-weighting hypothesis, not a proven cause. A separately
registered countermeasure must protect light coverage and measure background
rejection. No automatic extension, threshold retuning, native pilot or M6run.
The old M4primary, M3rule failures and all scientific gates remain unchanged.

Evidence: outputs/navigation_signal_pixel_audit_20261021 and
outputs/navigation_signal_pixel_study_20261021. Protocols:
[label audit](navigation_signal_pixel_audit_protocol_20261021.md),
[paired study](navigation_signal_pixel_study_protocol_20261021.md).
""")


if __name__ == "__main__":
    main()
