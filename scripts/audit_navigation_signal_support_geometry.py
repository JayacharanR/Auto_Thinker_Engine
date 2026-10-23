#!/usr/bin/env python3
"""Read-only TRAIN housing/context audit; no fitting or benchmark changes."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402


def main():
    def read(path):
        return json.loads(path.read_text())

    destination = ROOT / "reports/navigation_signal_support_geometry_20261022"
    if destination.with_suffix(".json").exists():
        raise FileExistsError("Retain the original audit receipt")
    previous = ROOT / "reports/navigation_signal_train_support_20261021.json"
    rows_file = ROOT / "outputs/navigation_signal_pretrained_20261021/train_rows.json"
    rows, support = read(rows_file), read(previous)
    data = DiverseSignalDataset(
        ROOT,
        ROOT / "outputs/navigation_signal_pixel_audit_20261021",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021",
    )
    train = set(data.train)
    pairs = defaultdict(dict)

    def identity(sample):
        return tuple(
            sample.get(k)
            for k in ("source", "task_id", "town", "junction_id", "case_id", "view", "illumination")
        )

    for i in data.train:
        key, phase = identity(data.samples[i]), data.samples[i]["phase"]
        if phase in pairs[key]:
            raise ValueError("Ambiguous phase-pair identity")
        pairs[key][phase] = i
    details, gallery, selected = [], [], set()
    for old in support["details"]:
        row = rows[old["row"]]
        i, x, y, cls = row["sample"], row["x"], row["y"], old["cls"]
        assert i in train and row["role"] == "train" and row["label"] == 1
        sample = data.samples[i]
        boxes = [b for b in data.boxes[i] if b[0] <= x < b[2] and b[1] <= y < b[3] and b[4] == cls]
        assert boxes
        distances = []
        for x0, y0, x1, y1, _ in boxes:
            yy, xx = np.where(data.targets["bulb"][i, y0:y1, x0:x1] == cls)
            distances.append(
                int(np.maximum(abs(xx + x0 - x), abs(yy + y0 - y)).min()) if len(xx) else None
            )
        assert distances[0] == old["active_pixel_chebyshev_radius"]
        finite = [d for d in distances if d is not None]
        radius = min(finite) if finite else None
        box = boxes[0]
        x0, y0, x1, y1, _ = box
        paired = pairs[identity(sample)]
        assert set(paired) == {"red", "green", "yellow", "red_repeat"}
        # Native pixels only. Compare alternate phase at the SAME camera and pose.
        # This measures visibility, never infers a new class label or deletes data.
        local = {
            p: data.images[j, :, max(0, y - 2) : y + 3, max(0, x - 2) : x + 3].astype(np.int16)
            for p, j in paired.items()
        }
        deltas = {
            p: int(np.abs(local[sample["phase"]] - local[p]).max())
            for p in ("red", "green", "yellow")
            if p != sample["phase"] and not (sample["phase"] == "red_repeat" and p == "red")
        }
        noise = int(np.abs(local["red"] - local["red_repeat"]).max())
        item = dict(
            **old,
            x=x,
            y=y,
            box=box,
            matching_boxes=len(boxes),
            all_matching_radii=distances,
            nearest_matching_radius=radius,
            width=x1 - x0,
            height=y1 - y0,
            local_max_abs_alternate_phase=deltas,
            local_red_repeat_max_abs=noise,
            alternate_phase_identical=any(d == 0 for d in deltas.values()),
            every_alternate_within_repeat_noise=all(d <= noise for d in deltas.values()),
        )
        details.append(item)
        bucket = (
            "0-2" if radius <= 2 else "3-15" if radius <= 15 else "16-47" if radius <= 47 else "48+"
        )
        selection_key = (cls, bucket)
        if selection_key not in selected:
            selected.add(selection_key)
            gallery.append(item)
    assert len(details) == 3371
    counts = {}
    for name, subset in (
        ("all", details),
        ("within5", [r for r in details if r["nearest_matching_radius"] <= 2]),
        ("outside5", [r for r in details if r["nearest_matching_radius"] > 2]),
    ):
        counts[name] = dict(
            points=len(subset),
            exact_alternate_patch=sum(r["alternate_phase_identical"] for r in subset),
            all_alternates_within_repeat_noise=sum(
                r["every_alternate_within_repeat_noise"] for r in subset
            ),
            multiple_matching_boxes=sum(r["matching_boxes"] > 1 for r in subset),
            width_quantiles=np.quantile([r["width"] for r in subset], [0, 0.5, 0.9, 1]).tolist(),
            height_quantiles=np.quantile([r["height"] for r in subset], [0, 0.5, 0.9, 1]).tolist(),
            radius_quantiles=np.quantile(
                [r["nearest_matching_radius"] for r in subset], [0, 0.5, 0.9, 1]
            ).tolist(),
        )
    overlap_changes = sum(
        r["nearest_matching_radius"] != r["active_pixel_chebyshev_radius"] for r in details
    )
    coverage = {
        str(r): sum(d["nearest_matching_radius"] <= r for d in details)
        for r in (2, 4, 8, 12, 15, 31, 47, 63, 127)
    }
    result = dict(
        train_only=True,
        no_fit_or_label_change=True,
        posthoc_diagnosis=True,
        source_sha256=digest(__file__),
        support_sha256=digest(previous),
        rows_sha256=digest(rows_file),
        counts=counts,
        coverage_by_radius=coverage,
        first_box_vs_any_matching_box_changes=overlap_changes,
        locations=dict(Counter(r["location"] for r in details)),
        gallery_selection="First existing row per state and radius bin; independent of prediction",
        gallery_rows=[r["row"] for r in gallery],
        details=details,
        caveats=[
            "Paired images have renderer noise; masks are conservative",
            "Point counts are correlated; no independent driving confidence interval",
            "5x5 equality does not imply equality of the full 87-feature vector",
            "Audit loads original dataset but all diagnostic rows and pairs are TRAIN",
        ],
    )
    fig, axes = plt.subplots(len(gallery), 4, figsize=(12, len(gallery) * 2.6), squeeze=False)
    for axes_row, item in zip(axes, gallery):
        i, x, y = item["sample"], item["x"], item["y"]
        paired = pairs[identity(data.samples[i])]
        x0, y0, x1, y1, _ = item["box"]
        # Show the entire annotated housing plus proposal, even for very large boxes.
        left, right = max(0, x0 - 8), min(512, x1 + 8)
        top, bottom = max(0, y0 - 8), min(288, y1 + 8)
        for ax, phase in zip(axes_row, ("red", "green", "yellow", "red_repeat")):
            j = paired[phase]
            ax.imshow(
                data.images[j, :, top:bottom, left:right].transpose(1, 2, 0),
                interpolation="nearest",
            )
            ax.add_patch(
                Rectangle(
                    (x0 - left, y0 - top),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=0.7,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x - 2.5 - left, y - 2.5 - top),
                    5,
                    5,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            ax.plot(x - left, y - top, "+", color="white", markersize=4)
            ax.set_title(
                f"{phase}: row {item['row']} / class {item['cls']}\n"
                f"{item['location']}, {x1 - x0}x{y1 - y0}, r={item['nearest_matching_radius']}",
                fontsize=8,
            )
            ax.axis("off")
    fig.suptitle("TRAIN-only fixed examples: projected housing (cyan), detailed 5x5 crop (white)")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(destination.with_suffix(".png"), dpi=110)
    plt.close(fig)
    atomic_json(destination.with_suffix(".json"), result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("details", "locations")}, indent=2)
    )


if __name__ == "__main__":
    main()
