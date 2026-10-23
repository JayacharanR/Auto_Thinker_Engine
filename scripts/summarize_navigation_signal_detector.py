#!/usr/bin/env python3
"""Rebuild the dated M2 report/figures without mutating any experiment input."""

import json
import os
import sys
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


def main():
    evidence = {}

    def read(path):
        path = ROOT / path
        evidence[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    base = Path("outputs/navigation_m2_signal_detector_20261016")
    review = read(base / "review.json")
    matches = read(base / "one_to_one_audit.json")
    crop_root = Path("outputs/navigation_m2_signal_crop_alignment_20261016")
    crops = read(crop_root / "assessment.json")
    precision_root = Path("outputs/navigation_m2_signal_proposal_budget_20261016")
    precision = read(precision_root / "assessment.json")
    original_precision = read("outputs/navigation_m2_signal_precision_20261016/assessment.json")
    corpus_root = Path("outputs/navigation_m2_signal_corpus_v2_20261016")
    corpus = read(corpus_root / "data_audit.json")
    assert review["selected_step"] == matches["selected_step"] == 50
    assert review["frozen_parameters_and_buffers_unchanged"]
    assert precision["parameters_and_buffers_unchanged"]
    assert digest(ROOT / base / "best_component.pt") == review["component_sha256"]
    assert digest(ROOT / crop_root / "robust_rgb_linear.npz") == crops["artifact_sha256"]
    for path, expected in matches["evidence_sha256"].items():
        assert digest(ROOT / path) == expected
    for path, expected in corpus["image_sha256"].items():
        assert digest(ROOT / corpus_root / path) == expected
    for directory in [
        base,
        crop_root,
        precision_root,
        Path("outputs/navigation_m2_signal_precision_20261016"),
    ]:
        request = read(directory / "request.json")
        for path, expected in request["source_hashes"].items():
            assert digest(ROOT / directory / "source" / path) == expected, path
    crop_request = read(crop_root / "request.json")
    assert digest(ROOT / base / "one_to_one_audit.json") == crop_request["fixed_assignments_sha256"]
    assert (
        digest(ROOT / base / "predictions_step50.json")
        == crop_request["fixed_detector_predictions_sha256"]
    )
    parent_sha = "80bee4e338eec8f55f57d96ad4cf0bf27c736e47bc641e2b1dd4e8bb0cfdc3e4"
    cache_sha = "b9e09499fa0c0158a62849369f44a2c942d90aa9c2dcdff1b7dffdd73e6710ae"
    preserved = {
        "outputs/navigation_m2_stop_go_20261016/best_policy.pt": parent_sha,
        "outputs/navigation_m2_cache_20k_20261015/manifest.json": cache_sha,
    }
    for path, expected in preserved.items():
        assert digest(ROOT / path) == expected
    summary = dict(
        status="M2 open; component improvement, no policy promotion or driving result",
        detector=review,
        detector_history=matches["results"],
        crop_alignment=crops,
        precision=precision,
        earlier_precision=original_precision,
        unchanged_camera_parent_and_cache=preserved,
        evidence_sha256=evidence,
        summary_source_sha256=digest(Path(__file__)),
        limitations=[
            "Four already inspected TRAIN development junctions;72 correlated labels",
            "COCO supervised pretraining; not a fair JEPA/encoder ablation",
            "Sparse geometry is not full detection AP or exhaustive background truth",
            "Runtime crop/state never receives oracle geometry; labels/scoring do",
            "No lane applicability, temporal-tracking or closed-loop claim",
            "No heldout_dev/Town05 testing; final-registration timing issue remains",
        ],
    )
    atomic_json(ROOT / "reports/navigation_m2_signal_detector_20261016.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    positions = np.arange(2)
    before = [100 * 61 / 72, 100 * 21 / 30]
    after = [100 * 68 / 72, 100 * 24 / 30]
    old = axes[0].bar(positions - 0.18, before, 0.36, color="#8791a0", label="Exact-crop training")
    new = axes[0].bar(
        positions + 0.18, after, 0.36, color="#3286b3", label="TRAIN crop augmentation"
    )
    axes[0].bar_label(old, labels=["61/72", "21/30"], padding=3)
    axes[0].bar_label(new, labels=["68/72", "24/30"], padding=3)
    axes[0].set(
        xticks=positions,
        xticklabels=[
            "Oracle GT ROI\n(privileged diagnostic)",
            "Fixed predicted ROI\n(30 matched boxes)",
        ],
        ylabel="State accuracy (%)",
        ylim=(0, 108),
        title="State task: unchanged source labels and detector boxes",
    )
    axes[0].legend(loc="lower left")
    modes = precision["results"]
    for name, color in zip(["fp32", "fp16", "fp16_cap200"], ["#287ca6", "#9263a3", "#c87b2f"]):
        result = modes[name]
        x = 1000 * result["mean_three_view_seconds"]
        y = result["one_to_one"]["iou3"]
        axes[1].scatter(x, y, s=80, color=color)
        axes[1].annotate(
            f"{name}\n{y}/72",
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axes[1].axvline(100, ls="--", color="gray", label="10Hz interval (model only)")
    axes[1].set(
        xlim=(65, 210),
        ylim=(0, 40),
        xlabel="Mean three-view GPU inference (ms)",
        ylabel="One-to-one detected labels /72",
        title="Precision/proposal budget: speed versus coverage",
    )
    axes[1].legend(loc="lower right")
    fig.suptitle("M2 controlled component results — no driving safety claim", fontsize=13)
    fig.savefig(
        ROOT / "reports/figures/navigation_m2_signal_alignment_latency_20261016.png", dpi=170
    )
    plt.close(fig)
    print(
        json.dumps(
            {
                "selected_step": 50,
                "detected": "30/72",
                "correct_state": "24/72",
                "corpus_images_verified": len(corpus["image_sha256"]),
                "camera_policy_unchanged": True,
                "all_archives_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
