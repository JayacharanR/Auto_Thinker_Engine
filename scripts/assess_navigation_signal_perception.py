#!/usr/bin/env python3
"""Reproduce the dated M2 perception comparison from preserved bounded runs."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/navigation_models/matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def assess():
    artifacts = {}

    def read(relative):
        path = ROOT / relative
        artifacts[relative] = digest(path)
        return json.loads(path.read_text())

    base = "outputs/navigation_m2_"
    corpus = read(base + "signal_corpus_v2_20261016/data_audit.json")
    previous = read(base + "signal_corpus_retry2_20261016/data_audit.json")
    assert corpus["static_label_data_gate_passed"]
    assert not previous["static_label_data_gate_passed"]
    center = read(base + "signal_grid_20261016/review.json")
    region = read(base + "signal_grid_region_20261016/review.json")
    oracle = read(base + "signal_oracle_crops_20261016/assessment.json")
    detector = read(base + "reference_detector_20261016/assessment.json")
    predictions = read(base + "reference_detector_20261016/predictions.json")
    response = read(base + "signal_response_20261016/assessment.json")
    for run, report in (("signal_grid", center), ("signal_grid_region", region)):
        path = ROOT / f"{base}{run}_20261016/best_component.pt"
        assert digest(path) == report["component_sha256"]
    assert (
        digest(ROOT / (base + "signal_oracle_crops_20261016/oracle_rgb_linear.npz"))
        == oracle["artifact_sha256"]
    )
    assert (
        digest(ROOT / "outputs/navigation_models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth")
        == (detector["weights_sha256"])
    )
    preserved = {
        base
        + "stop_go_20261016/best_policy.pt": (
            "80bee4e338eec8f55f57d96ad4cf0bf27c736e47bc641e2b1dd4e8bb0cfdc3e4"
        ),
        base
        + "cache_20k_20261015/manifest.json": (
            "b9e09499fa0c0158a62849369f44a2c942d90aa9c2dcdff1b7dffdd73e6710ae"
        ),
    }
    for relative, expected in preserved.items():
        assert digest(ROOT / relative) == expected
    assert response["model_parameters_equal_checkpoint_before_and_after"]
    assert response["checkpoint_sha256"] == next(iter(preserved.values()))
    assert all(r["validation"]["supported_points"] == 72 for r in (center, region))
    assert oracle["validation_crops"] == detector["counts"]["centers"] == 72
    assert sum(map(sum, oracle["confusion_matrix"])) == 72

    labels = {
        (row["town"], row["case_id"], row["view"], row["head"]): row for row in corpus["labels"]
    }
    height_groups, towns = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for row in predictions["rows"]:
        sample = row["sample"]
        for match in row["supported_annotation_matches"]:
            label = labels[sample["town"], sample["case_id"], sample["view"], match["head"]]
            height = label["crop_xyxy"][3] - label["crop_xyxy"][1]
            bucket = "<8 px" if height < 8 else "8–15 px" if height < 16 else "≥16 px"
            for group in (height_groups[bucket], towns[sample["town"]]):
                group[0] += int(match["best_iou"] >= 0.3)
                group[1] += 1
    assert sum(row[1] for row in height_groups.values()) == 72
    assert sum(row[0] for row in height_groups.values()) == detector["counts"]["detected_iou3"]

    summary = {
        "status": "M2 remains open; no component or driving-policy promotion",
        "corpus_v2": {k: v for k, v in corpus.items() if k not in ("labels", "image_sha256")},
        "previous_corpus_gate_retained_failed": not previous["static_label_data_gate_passed"],
        "center_supervision": center,
        "region_supervision": region,
        "oracle_crop_state": oracle,
        "reference_detector": detector,
        "detector_native_crop_height_correct_total": dict(height_groups),
        "detector_by_town_correct_total": dict(towns),
        "unchanged_parent_and_cache_hashes": preserved,
        "evidence_sha256": artifacts,
        "analysis_source_sha256": digest(Path(__file__)),
        "limitations": [
            "Only four component-validation junctions within TRAIN; 72 correlated labels",
            "Some TRAIN validation locations inspected during annotation development",
            "Sparse housing geometry is not a complete detection benchmark",
            "Oracle crops use privileged geometry and are not deployable RGB perception",
            "COCO supervised pretraining is not a matched-compute JEPA ablation",
            "No heldout_dev, Town05 or new driving success result",
        ],
    }
    destination = ROOT / "reports/navigation_m2_signal_perception_20261016.json"
    atomic_json(destination, summary)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    values = [
        50,
        100 * center["validation"]["center_class_accuracy"],
        100 * region["validation"]["center_class_accuracy"],
    ]
    bars = axes[0].bar(
        ["Always red", "Center targets", "Region targets"],
        values,
        color=["#8892a2", "#3789b9", "#6e66ad"],
    )
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set(
        ylim=(0, 100),
        ylabel="Exact center accuracy (%)",
        title="Full-image scratch localizers\nSame 72 sparse centers",
    )
    matrix = np.array(oracle["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for (row, col), count in np.ndenumerate(matrix):
        axes[1].text(
            col,
            row,
            str(count),
            ha="center",
            va="center",
            color="white" if count > matrix.max() / 2 else "black",
        )
    classes = ["Red", "Yellow", "Green"]
    axes[1].set(
        xticks=range(3),
        xticklabels=classes,
        yticks=range(3),
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
        title="Privileged exact-crop RGB classifier\n61/72 (84.7%); no localization",
    )
    buckets = ["<8 px", "8–15 px", "≥16 px"]
    numbers = [height_groups[k] for k in buckets]
    bars = axes[2].bar(buckets, [100 * a / b for a, b in numbers], color="#c17e37")
    axes[2].bar_label(bars, labels=[f"{a}/{b}" for a, b in numbers], padding=3)
    axes[2].set(
        ylim=(0, 100),
        ylabel="Sparse housing recall at IoU ≥0.3 (%)",
        xlabel="Native annotated housing crop height",
        title="Frozen COCO detector\nConfidence ≥0.5; no threshold tuning",
    )
    fig.suptitle("M2 perception diagnostics — TRAIN locations, not driving results", fontsize=14)
    figures = ROOT / "reports/figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "navigation_m2_signal_perception_20261016.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "report": str(destination),
                "status": summary["status"],
                "detector_height_breakdown": dict(height_groups),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    assess()
