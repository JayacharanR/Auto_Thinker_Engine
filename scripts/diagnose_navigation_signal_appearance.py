#!/usr/bin/env python3
"""TRAIN-only proposal/verification failure and active-bulb appearance diagnosis."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402


def main():
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    destination = ROOT / "reports/navigation_signal_appearance_20261022.json"
    if destination.exists():
        raise FileExistsError("Retain original diagnosis")
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    study = ROOT / "outputs/navigation_signal_supported_20261022"
    full = ROOT / "outputs/navigation_signal_supported_full_20261022"
    if read(full / "progress.json")["status"] != "complete":
        raise ValueError("Terminal full TRAIN audit required")
    rows, proposals = read(cache / "train_rows.json"), read(cache / "train_proposals.json")
    anchor_p = np.load(study / "oof_supported.npy")
    full_p = np.load(full / "supported_probability.npy", mmap_mode="r")
    with np.load(full / "offsets.npz") as a:
        offsets, indices = a["offsets"], a["samples"]
    data = DiverseSignalDataset(
        ROOT,
        ROOT / "outputs/navigation_signal_pixel_audit_20261021",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021",
    )
    if not np.array_equal(indices, data.train):
        raise ValueError("All original TRAIN images required")
    by_sample = defaultdict(list)
    train = set(data.train)
    for j, row in enumerate(rows):
        if row["sample"] not in train or row["role"] != "train":
            raise ValueError("TRAIN rows only")
        if row["label"]:
            by_sample[row["sample"]].append(j)
    details = []
    for position, i in enumerate(data.train):
        sample, boxes = data.samples[i], data.boxes[i]
        p, peaks = full_p[offsets[position] : offsets[position + 1]], proposals[str(i)]
        if len(peaks) != len(p):
            raise ValueError("Original proposal ordering required")
        fg = p[:, 1:].sum(1)
        available, matched = set(range(len(boxes))), {}
        for j in sorted(
            np.flatnonzero(fg >= 0.5), key=lambda j: (-fg[j], peaks[j]["y"], peaks[j]["x"])
        ):
            k = next(
                (
                    k
                    for k in sorted(available)
                    if boxes[k][0] <= peaks[j]["x"] < boxes[k][2]
                    and boxes[k][1] <= peaks[j]["y"] < boxes[k][3]
                ),
                None,
            )
            if k is not None:
                available.remove(k)
                matched[k] = j
        for k, box in enumerate(boxes):
            x0, y0, x1, y1, cls = box
            inside = [
                j for j, peak in enumerate(peaks) if x0 <= peak["x"] < x1 and y0 <= peak["y"] < y1
            ]
            yy, xx = np.where(data.targets["bulb"][i, y0:y1, x0:x1] == cls)
            if not len(xx):
                raise ValueError("Audited TRAIN housing lacks active pixels")
            rgb = data.images[i].transpose(1, 2, 0)[yy + y0, xx + x0].astype(np.float32)
            if rgb.shape != (len(xx), 3):
                raise ValueError("Active-pixel indexing mismatch")
            brightest = rgb[int(rgb.mean(1).argmax())]
            max_channel, min_channel = rgb.max(1), rgb.min(1)
            radii = [
                int(np.maximum(abs(xx + x0 - peaks[j]["x"]), abs(yy + y0 - peaks[j]["y"])).min())
                for j in inside
            ]
            # Oracle uses existing OOF probabilities only; never a runtime input.
            anchors = [
                j
                for j in by_sample[i]
                if x0 <= rows[j]["x"] < x1
                and y0 <= rows[j]["y"] < y1
                and data.targets["bulb"][i, rows[j]["y"], rows[j]["x"]] == cls
            ]
            oracle = [
                j
                for j in anchors
                if anchor_p[j, 1:].sum() >= 0.5 and anchor_p[j, 1:].argmax() + 1 == cls
            ]
            matched_cls = int(p[matched[k], 1:].argmax() + 1) if k in matched else None
            outcome = (
                "correct"
                if matched_cls == cls
                else "wrong_phase"
                if matched_cls is not None
                else "no_proposal"
                if not inside
                else "all_filtered"
            )
            details.append(
                dict(
                    sample=i,
                    box_index=k,
                    box=box,
                    town=sample["town"],
                    junction=sample["junction_id"],
                    phase=sample["phase"],
                    view=sample["view"],
                    outcome=outcome,
                    proposal_count=len(inside),
                    nearest_proposal_active_radius=min(radii) if radii else None,
                    proposal_with_local_support=any(r <= 2 for r in radii),
                    best_proposal_foreground=float(fg[inside].max()) if inside else None,
                    oracle_active_points=len(anchors),
                    oracle_can_recognize=bool(oracle),
                    active_pixels=len(xx),
                    width=x1 - x0,
                    height=y1 - y0,
                    brightest_rgb=brightest.astype(int).tolist(),
                    brightest_mean=float(brightest.mean()),
                    median_active_saturation=float(
                        np.median((max_channel - min_channel) / np.maximum(1, max_channel))
                    ),
                )
            )
    if len(details) != 1248 or sum(r["outcome"] == "correct" for r in details) != 1144:
        raise ValueError("Original TRAIN housing count or retained model result changed")
    counts = Counter(r["outcome"] for r in details)
    strata = {}
    for outcome in sorted(counts):
        group = [r for r in details if r["outcome"] == outcome]
        strata[outcome] = dict(
            houses=len(group),
            oracle_recognized=sum(r["oracle_can_recognize"] for r in group),
            oracle_present=sum(r["oracle_active_points"] > 0 for r in group),
            local_proposal_present=sum(r["proposal_with_local_support"] for r in group),
            median_brightest_intensity=float(np.median([r["brightest_mean"] for r in group])),
            median_active_saturation=float(
                np.median([r["median_active_saturation"] for r in group])
            ),
            median_width=float(np.median([r["width"] for r in group])),
            median_height=float(np.median([r["height"] for r in group])),
            states=dict(Counter(r["phase"] for r in group)),
            locations=dict(Counter(f"{r['town']}/{r['junction']}" for r in group)),
        )
    atomic_json(
        destination,
        dict(
            TRAIN_only=True,
            no_fit_or_label_change=True,
            posthoc_diagnosis=True,
            oracle_is_not_deployment_metric=True,
            source_sha256=digest(__file__),
            counts=dict(counts),
            strata=strata,
            details=details,
            evidence={
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    full / "assessment.json",
                    full / "supported_probability.npy",
                    full / "offsets.npz",
                    study / "oof_supported.npy",
                    cache / "train_rows.json",
                    cache / "train_proposals.json",
                )
            },
        ),
    )
    print(json.dumps(dict(counts=counts, strata=strata), indent=2))


if __name__ == "__main__":
    main()
