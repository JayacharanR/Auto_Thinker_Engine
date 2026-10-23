#!/usr/bin/env python3
"""Independent full TRAIN recount and fixed false-positive gallery; no refit."""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_phase_supervision import joint_candidates  # noqa: E402


def main():
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    root = ROOT / "outputs/navigation_signal_background_full_20261022"
    previous = ROOT / "outputs/navigation_signal_refinement_train_20261022"
    destination = ROOT / "reports/navigation_signal_background_full_20261022"
    if destination.with_suffix(".json").exists():
        raise FileExistsError("Retain original report")
    assessment = read(root / "assessment.json")
    if read(root / "progress.json")["status"] != "complete":
        raise ValueError("Closed full TRAIN audit required")
    for name, sha in assessment["artifact_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError("Closed scientific artifact changed")
    data = DiverseSignalDataset(
        ROOT,
        ROOT / "outputs/navigation_signal_pixel_audit_20261021",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021",
    )
    xy, p = (np.load(root / file, mmap_mode="r") for file in ("points.npy", "probability.npy"))
    with np.load(root / "offsets.npz") as a:
        offsets, indices = a["offsets"], a["samples"]
    if not np.array_equal(indices, data.train):
        raise ValueError("Original ordered TRAIN indices required")
    old_xy = np.load(previous / "points.npy", mmap_mode="r")
    with np.load(previous / "offsets.npz") as a:
        old_offsets = a["offsets"]
    rows = read(ROOT / "outputs/navigation_signal_pretrained_20261021/train_rows.json")
    sampled_bg = {(r["sample"], r["x"], r["y"]) for r in rows if r["label"] == 0}
    predictions, false_positive, gallery, seen = {}, [], [], set()
    for n, i in enumerate(indices):
        probabilities, points = p[offsets[n] : offsets[n + 1]], xy[offsets[n] : offsets[n + 1]]
        fg = probabilities[:, 1:].sum(1)
        accepted = np.flatnonzero(fg >= 0.5)
        # Match only accepted candidates; rejected candidates cannot affect original scorer.
        predictions[i] = joint_candidates(
            [dict(x=int(x), y=int(y)) for x, y in points[accepted]], probabilities[accepted]
        )
        old = old_xy[old_offsets[n] : old_offsets[n + 1]]
        old_codes = set((old[:, 1].astype(np.int64) * 512 + old[:, 0]).tolist())
        for j in accepted:
            x, y = map(int, points[j])
            if not data.background[i, y, x]:
                continue
            # Known background excludes all projected signal ignore zones; never relabel unknowns.
            sample = data.samples[i]
            cls = int(probabilities[j, 1:].argmax() + 1)
            row = dict(
                sample=int(i),
                x=x,
                y=y,
                cls=cls,
                foreground=float(fg[j]),
                town=sample["town"],
                junction=sample["junction_id"],
                independent_new_point=y * 512 + x not in old_codes,
                in_original_sampled_background=(int(i), x, y) in sampled_bg,
            )
            false_positive.append(row)
            key = (sample["town"], cls)
            if key not in seen:
                seen.add(key)
                gallery.append(row)
    # Independently follow the original score-order matching on the TRAIN denominator.
    totals = dict(
        labels=0, localized=0, joint=0, background_peaks=0, unknown_peaks=0, towns={}, classes={}
    )
    for i in data.train:
        available, matched = set(range(len(data.boxes[i]))), {}
        for point in predictions[i]:
            eligible = [
                k
                for k in sorted(available)
                if data.boxes[i][k][0] <= point["x"] < data.boxes[i][k][2]
                and data.boxes[i][k][1] <= point["y"] < data.boxes[i][k][3]
            ]
            if eligible:
                k = eligible[0]
                available.remove(k)
                matched[k] = point["cls"]
            elif data.background[i, point["y"], point["x"]]:
                totals["background_peaks"] += 1
            else:
                totals["unknown_peaks"] += 1
        for k, box in enumerate(data.boxes[i]):
            for group in (
                totals,
                totals["towns"].setdefault(
                    data.samples[i]["town"], dict(labels=0, localized=0, joint=0)
                ),
                totals["classes"].setdefault(str(box[4]), dict(labels=0, localized=0, joint=0)),
            ):
                group["labels"] += 1
                group["localized"] += int(k in matched)
                group["joint"] += int(matched.get(k) == box[4])
    for key in ("labels", "localized", "joint", "background_peaks", "unknown_peaks", "classes"):
        if totals[key] != assessment["metrics"][key]:
            raise ValueError("Independent full TRAIN recount differs")
    for town, values in totals["towns"].items():
        if any(assessment["metrics"]["towns"][town][k] != v for k, v in values.items()):
            raise ValueError("Independent town recount differs")
    if len(false_positive) != totals["background_peaks"]:
        raise ValueError("Known-background interpretation differs")
    summary = dict(
        total=len(false_positive),
        new_coordinate=sum(r["independent_new_point"] for r in false_positive),
        in_original_sampled_background=sum(
            r["in_original_sampled_background"] for r in false_positive
        ),
        predicted_phases=dict(Counter(str(r["cls"]) for r in false_positive)),
        score_quantiles=np.quantile(
            [r["foreground"] for r in false_positive], [0, 0.5, 0.9, 1]
        ).tolist(),
    )
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, gallery):
        i, x, y = row["sample"], row["x"], row["y"]
        left, top = max(0, min(448, x - 32)), max(0, min(224, y - 32))
        ax.imshow(
            data.images[i, :, top : top + 64, left : left + 64].transpose(1, 2, 0),
            interpolation="nearest",
        )
        ax.plot(x - left, y - top, "+", color="cyan")
        ax.set_title(
            f"{row['town']}/{row['junction']} class{row['cls']}\n"
            f"sample{i}; p={row['foreground']:.3f}; new={row['independent_new_point']}",
            fontsize=8,
        )
    fig.suptitle("First known-background detection per TRAIN town/predicted phase")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination.with_suffix(".png"), dpi=120)
    plt.close(fig)
    atomic_json(
        destination.with_suffix(".json"),
        dict(
            TRAIN_only=True,
            independent_recount_exact=True,
            no_component_val=True,
            assessment=assessment,
            recounted=totals,
            false_positive_summary=summary,
            false_positives=false_positive,
            gallery=gallery,
            gallery_selection="First sample/coordinate order per TRAIN town/predicted phase",
            source_sha256=digest(__file__),
            original_scorer_source_sha256=digest(
                ROOT / "scripts/train_navigation_signal_context.py"
            ),
            no_fit_or_threshold_change=True,
        ),
    )
    print(json.dumps(dict(recount=totals, false_positive_summary=summary), indent=2))


if __name__ == "__main__":
    main()
