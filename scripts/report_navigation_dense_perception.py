#!/usr/bin/env python3
"""Describe the frozen dense-mask diagnostic without rerunning or tuning inference."""

import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_dense_perception import original_packet  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.storage_budget import require_budget  # noqa: E402

RUN = ROOT / "outputs/navigation_dense_perception_diagnosis_20261020"
STAGES = ("raw", "projectable", "fused")


def read(path):
    return json.loads(path.read_text())


def main():
    require_budget()
    assessment = read(RUN / "assessment.json")
    records = [json.loads(line) for line in (RUN / "scored_records.jsonl").read_text().splitlines()]
    predictions = [
        json.loads(line) for line in (RUN / "predictions.jsonl").read_text().splitlines()
    ]
    if (
        len(records) != 78
        or len(predictions) != 78
        or digest(RUN / "scored_records.jsonl") != assessment["scored_records_sha256"]
        or digest(RUN / "predictions.jsonl") != assessment["prediction_sha256"]
        or read(RUN / "progress.json")["status"] != "complete"
    ):
        raise ValueError("A complete unchanged original diagnostic is required")
    counts = defaultdict(Counter)
    for row in records:
        for view in row["views"].values():
            matches = {
                stage: {m["reference"] for m in view["stages"][stage]["matches"]}
                for stage in STAGES
            }
            for index, obj in enumerate(view["visible_objects"]):
                depth, pixels = obj["median_encoded_depth_m"], obj["visible_pixels"]
                depth_bin = (
                    "<15"
                    if depth < 15
                    else "15-30"
                    if depth < 30
                    else "30-60"
                    if depth < 60
                    else ">=60"
                )
                size_bin = (
                    "8-31"
                    if pixels < 32
                    else "32-127"
                    if pixels < 128
                    else "128-511"
                    if pixels < 512
                    else ">=512"
                )
                for key in (
                    "all",
                    "group:" + obj["group"],
                    "depth:" + depth_bin,
                    "pixels:" + size_bin,
                    "town:" + row["town"],
                    "truncated:" + str(obj["image_truncated"]),
                ):
                    counts[key]["visible"] += 1
                    for stage in STAGES:
                        counts[key][stage] += int(index in matches[stage])
    result = dict(
        original_assessment_sha256=digest(RUN / "assessment.json"),
        generating_source_sha256=digest(__file__),
        breakdown=dict(sorted(counts.items())),
        all_observed_groups=sorted(
            {
                o["group"]
                for row in records
                for view in row["views"].values()
                for o in view["visible_objects"]
            }
        ),
        tiny_instance_views=assessment["tiny_instance_views"],
        posthoc_descriptive_breakdown=True,
        no_model_updates=True,
        independent_locations=2,
    )
    atomic_json(RUN / "descriptive_analysis.json", result)
    shutil.copyfile(__file__, RUN / "descriptive_report_source.py")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    all_counts = assessment["counts"]
    coverage = [
        100 * all_counts[s]["matched_object_groups"] / all_counts[s]["visible_object_groups"]
        for s in STAGES
    ]
    bars = axes[0].bar(STAGES, coverage, color=["#3b82b5", "#ec9e38", "#8057a3"])
    axes[0].bar_label(bars, fmt="%.1f%%")
    axes[0].set(
        title="Cross-view object coverage", ylabel="Matched object groups (%)", ylim=(0, 100)
    )
    for i, stage in enumerate(STAGES):
        labels = ["<15", "15-30", "30-60", ">=60"]
        values = [
            100 * counts["depth:" + d][stage] / counts["depth:" + d]["visible"] for d in labels
        ]
        axes[1].bar(np.arange(4) + (i - 1) * 0.25, values, 0.25, label=stage)
    axes[1].set(
        xticks=np.arange(4),
        xticklabels=labels,
        ylim=(0, 105),
        xlabel="Median encoded visible-surface depth (m)",
        ylabel="Mask/view matches (%)",
        title="Descriptive distance breakdown",
    )
    axes[1].legend(fontsize=8)
    geometry = [d for row in records for d in row["depth_ground_diagnostics"]]
    for key, label in (
        ("flat_to_axial_visible_surface_xy_m", "Axial hypothesis"),
        ("flat_to_radial_visible_surface_xy_m", "Radial hypothesis"),
    ):
        values = np.sort([d[key] for d in geometry])
        axes[2].plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    axes[2].set(
        xlim=(0, 20),
        ylim=(0, 1),
        xlabel="Flat-ground / visible-surface gap (m)",
        ylabel="Fraction of observations",
        title="Geometry diagnostic, not 3D accuracy",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen COCO FPN · 78 timestamps · 2 familiar calibration locations")
    figure = ROOT / "reports/navigation_dense_perception_20261020.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    # Predetermined midpoint of each source: never select an attractive prediction.
    manifest = read(ROOT / "outputs/navigation_dense_labels_verified_20261020/manifest.json")
    sources = [s for s in manifest["sources"] if s["item"]["role"] == "calibration"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True)
    for row_index, source in enumerate(sources):
        ref = source["label_sources"][len(source["label_sources"]) // 2]
        packet = original_packet(ref)
        selected = next(
            r
            for r in records
            if r["route_id"] == source["item"]["route_id"] and r["frame"] == ref["frame"]
        )
        pred = next(
            r["prediction"]
            for r in predictions
            if r["route_id"] == selected["route_id"] and r["frame"] == selected["frame"]
        )
        for column, view in enumerate(("front", "left", "right")):
            ax = axes[row_index, column]
            ax.imshow(packet.rgb[view])
            for box, color in [
                *[(o["box"], "#34d399") for o in selected["views"][view]["visible_objects"]],
                *[(o["box"], "#fb923c") for o in pred["raw"] if o["view"] == view],
            ]:
                x0, y0, x1, y1 = box
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=0.8
                    )
                )
            ax.set_title(f"{source['item']['town']} · {view} · frame {packet.frame}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Fixed source midpoints · green: visible masks ≥8px · orange: raw detections")
    fig.savefig(ROOT / "reports/navigation_dense_perception_examples_20261020.png", dpi=150)
    plt.close(fig)
    report = """# Frozen camera perception diagnosis — 20 October 2026

The fixed detector loses substantial coverage before tracking: both small-object
detection and flat-ground projection need attention. Cross-view merging removes
many duplicate view predictions but only eight additional matched object groups.
This is a diagnosis, not a trained improvement or driving qualification.

![Coverage and geometry](navigation_dense_perception_20261020.png)

| Stage | Mask/view matches / 1,584 | Cross-view object groups / 1,134 |
|---|---:|---:|
| Raw detector | 1,110 (70.1%) | 786 (69.3%) |
| Projectable | 926 (58.5%) | 616 (54.3%) |
| Fused | 666 (42.0%) | 608 (53.6%) |

All 78 timestamps / 234 views from the two original calibration sources were
processed once with unchanged FP32 frozen FPN weights, score 0.35, IoU match 0.3,
60 m projection and 2 m merge. Predictions were written before reading targets.
Three warmups; measured mean 109.7 ms / p95 115.0 ms, peak allocated GPU 0.939 GB.
The diagnostic hook adds overhead. No model fitting or threshold sweep occurred.

The depth breakdown is descriptive and was calculated after the main diagnostic:

| Median encoded visible-surface depth | Mask/view pairs | Raw matches | Projectable matches |
|---|---:|---:|---:|
| <15 m | 253 | 223 (88.1%) | 214 (84.6%) |
| 15–30 m | 465 | 432 (92.9%) | 431 (92.7%) |
| 30–60 m | 591 | 414 (70.1%) | 273 (46.2%) |
| ≥60 m | 275 | 41 (14.9%) | 8 (2.9%) |

Thus the all-distance denominator includes objects outside the declared 60 m
projection range; it must not be used to claim a near-field recall of only 58.5%.
Nevertheless, 141 raw matched mask/view pairs in the 30–60 m encoded-depth bin
are lost at projection. Ground-plane assumptions, thin silhouettes and box
bottom errors are plausible contributors, not separately established causes.
Raw detection matches 21.2% of the 430 masks with 8–31 pixels, versus 99.1% of the
327 masks with ≥512 pixels. All 1,584 included masks are **vehicles**: there are
no pedestrian or two-wheeler examples here. A further 140 masks below eight
pixels remain in the raw records and outside the primary matching denominator.

The flat-ground estimate differs from the axial visible-bottom surface hypothesis
by median 1.05 m / p95 6.26 m across 1,004 diagnostic observations; the radial
alternative is 2.51 m / 14.88 m. These are different hypotheses about visible
surfaces, not certified actor-centre localization error. Neither is selected or
substituted into the current controller. Do not tune tracking uncertainty from
these distances or from the single observed action-branch outcome.

![Predetermined midpoint examples](navigation_dense_perception_examples_20261020.png)

The next useful countermeasure is to validate metric depth and preserve explicit
unknown/bearing-only detections instead of silently treating projection failure
as an empty scene. Any learned geometry must fit only the nine verified TRAIN
sources and select on the two predeclared calibration locations. Declare and
charge all new dense supervision. Keep the same shared perceived input for CV,
RSSM and JEPA; do not attribute a perception improvement to the dynamics model.
Collect independent alternative-action encounters before fitting first-event
risk: these short expert sources contain no contacts and cannot supply that
missing action coverage. Add pedestrian/two-wheeler coverage before claims about
those classes. These recommendations change no frozen test, reward or scoring.

Evidence: `outputs/navigation_dense_perception_diagnosis_20261020` retains the
request, exact initial source, predictions, matching records and assessment.
The verified manifest SHA is
`9165e84ebe15a78d9aafb8ec6139f7e081b68b348609a98b93fb60eff0742163`.
Both locations are familiar TRAIN geography with earlier encoder/driver exposure;
snapshots are correlated and low16 Unreal instance IDs may alias. Unmatched
predictions are relative to included masks, not exhaustive false positives.
The original 13-source collection denominator, both rejected sources and failed
whole-corpus audit remain unchanged. No M2/M3/M5/M6 safety gate has passed.
"""
    (ROOT / "reports/navigation_dense_perception_20261020.md").write_text(report)
    atomic_json(ROOT / "reports/navigation_dense_perception_20261020.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
