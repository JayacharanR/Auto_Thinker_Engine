#!/usr/bin/env python3
"""Fixed-order visual diagnosis of closed signal predictions; never alters labels."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa:E402


def matched(peaks, boxes):
    remaining, result = set(range(len(boxes))), {}
    for peak in peaks:
        for i in sorted(remaining):
            x0, y0, x1, y1, _ = boxes[i]
            if x0 <= peak["x"] < x1 and y0 <= peak["y"] < y1:
                result[i] = peak
                remaining.remove(i)
                break
    return result


def main():
    def read(p):
        return json.loads(p.read_text())

    data = DiverseSignalDataset(
        ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021",
    )
    from src.navigation.signal_pretrained_verifier import retain_original_states

    base = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb"
    peaks = {r["sample"]: r["peaks"] for r in read(base / "predictions1000.json")}
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    offsets = read(cache / "cache.json")["validation_offsets"]
    probabilities = np.load(
        ROOT / "outputs/navigation_signal_regularized_20261021/validation_probability.npy"
    )
    selected, seen = [], set()
    for n, index in enumerate(data.validation):
        parent = matched(peaks[index], data.boxes[index])
        current = matched(retain_original_states(
            peaks[index], probabilities[offsets[n]:offsets[n + 1]]
        ), data.boxes[index])
        for i, box in enumerate(data.boxes[index]):
            key = (data.samples[index]["town"], box[4])
            if key in seen or i not in parent or parent[i]["cls"] != box[4]:
                continue
            if i in current and current[i]["cls"] == box[4]:
                continue
            seen.add(key)
            selected.append(dict(sample=index, box=box, role="VAL lost", point=parent[i]))
    train_seen = set()
    for index in data.train:
        key = (data.samples[index]["town"], data.samples[index]["junction_id"])
        if key in train_seen:
            continue
        eligible = [box for box in data.boxes[index] if box[4] == 1]
        if eligible:
            train_seen.add(key)
            selected.append(dict(sample=index, box=eligible[0], role="TRAIN first red", point=None))
    rows = (len(selected) + 5) // 6
    fig, axes = plt.subplots(rows, 6, figsize=(18, rows * 3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, selected):
        index, box = row["sample"], row["box"]
        x0, y0, x1, y1, cls = box
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        left, top = max(0, min(448, cx - 32)), max(0, min(224, cy - 32))
        image = data.images[index, :, top:top + 64, left:left + 64].transpose(1, 2, 0)
        ax.imshow(image, interpolation="nearest")
        ax.add_patch(Rectangle((x0-left, y0-top), x1-x0, y1-y0, fill=False,
                               edgecolor="cyan", linewidth=.8))
        if row["point"]:
            ax.plot(row["point"]["x"]-left, row["point"]["y"]-top, "+", color="white")
        sample = data.samples[index]
        ax.set_title(f"{row['role']} {sample['town'].replace('_Opt', '')}/{sample['junction_id']}"
                     f"\nclass {cls}; {x1-x0}×{y1-y0} px; sample {index}", fontsize=9)
    fig.suptitle("First lost light per VAL town/state; first red housing per TRAIN location")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_error_gallery_20261021.png", dpi=130)
    plt.close(fig)
    atomic_json(ROOT / "reports/navigation_signal_error_gallery_20261021.json", dict(
        selection="first lost parent-correct per VAL town/state; first red per TRAIN location",
        posthoc_diagnosis_only=True, selected=selected, no_label_or_selection_change=True,
    ))
    print(f"Saved {len(selected)} fixed-order crops; no fitting or label changes.")


if __name__ == "__main__":
    main()
