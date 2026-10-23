#!/usr/bin/env python3
"""Independent component recount and TRAIN residual proposal diagnosis."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_signal_context import score  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_phase_supervision import joint_candidates  # noqa: E402


def main():
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    train = ROOT / "outputs/navigation_signal_refinement_train_20261022"
    component = ROOT / "outputs/navigation_signal_refinement_component_20261022"
    destination = ROOT / "reports/navigation_signal_refinement_20261022.json"
    if destination.exists():
        raise FileExistsError("Retain original report")
    for root in (train, component):
        if read(root / "progress.json")["status"] != "complete":
            raise ValueError("Both owners must be terminal")
        for name, sha in read(root / "assessment.json")["artifact_sha256"].items():
            if digest(root / name) != sha:
                raise ValueError("Closed audit artifact changed")
    data = DiverseSignalDataset(
        ROOT,
        ROOT / "outputs/navigation_signal_pixel_audit_20261021",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021",
    )
    with np.load(component / "offsets.npz") as a:
        offsets, samples = a["offsets"], a["samples"]
    if not np.array_equal(samples, data.validation):
        raise ValueError("Original component ordering changed")
    p = np.load(component / "probability.npy", mmap_mode="r")
    xy = np.load(component / "points.npy", mmap_mode="r")
    predictions = {}
    for n, i in enumerate(samples):
        peaks = [dict(x=int(x), y=int(y)) for x, y in xy[offsets[n] : offsets[n + 1]]]
        predictions[i] = joint_candidates(peaks, p[offsets[n] : offsets[n + 1]])
    counted = score(predictions, data)
    assessment = read(component / "assessment.json")
    for key in ("labels", "localized", "joint", "background_peaks", "unknown_peaks", "classes"):
        if counted[key] != assessment["metrics"][key]:
            raise ValueError("Independent component recount differs")
    for town, values in counted["towns"].items():
        if any(assessment["metrics"]["towns"][town][k] != v for k, v in values.items()):
            raise ValueError("Independent town recount differs")
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    rows = read(cache / "train_rows.json")
    oracle_p = np.load(ROOT / "outputs/navigation_signal_supported_20261022/oof_supported.npy")
    by_sample = defaultdict(list)
    for j, row in enumerate(rows):
        if row["label"]:
            by_sample[row["sample"]].append(j)
    with np.load(train / "offsets.npz") as a:
        offsets, samples = a["offsets"], a["samples"]
    if not np.array_equal(samples, data.train):
        raise ValueError("Original TRAIN ordering changed")
    p = np.load(train / "probability.npy", mmap_mode="r")
    xy = np.load(train / "points.npy", mmap_mode="r")
    records = []
    for n, i in enumerate(samples):
        coordinates, probability = xy[offsets[n] : offsets[n + 1]], p[offsets[n] : offsets[n + 1]]
        for k, box in enumerate(data.boxes[i]):
            x0, y0, x1, y1, cls = box
            inside = (
                (coordinates[:, 0] >= x0)
                & (coordinates[:, 0] < x1)
                & (coordinates[:, 1] >= y0)
                & (coordinates[:, 1] < y1)
            )
            indices = np.flatnonzero(inside)
            foreground = probability[indices, 1:].sum(1)
            accepted = indices[foreground >= 0.5]
            if len(accepted):
                best = sorted(
                    accepted,
                    key=lambda j: (-probability[j, 1:].sum(), coordinates[j, 1], coordinates[j, 0]),
                )[0]
                outcome = "correct" if probability[best, 1:].argmax() + 1 == cls else "wrong_phase"
            else:
                outcome = "all_filtered" if len(indices) else "no_candidate"
            if outcome == "correct":
                continue
            oracle = [
                j
                for j in by_sample[i]
                if x0 <= rows[j]["x"] < x1
                and y0 <= rows[j]["y"] < y1
                and data.targets["bulb"][i, rows[j]["y"], rows[j]["x"]] == cls
                and oracle_p[j, 1:].sum() >= 0.5
                and oracle_p[j, 1:].argmax() + 1 == cls
            ]
            existing = {tuple(point) for point in coordinates[inside]}
            if any((rows[j]["x"], rows[j]["y"]) in existing for j in oracle):
                raise ValueError(
                    "Recognized oracle already present in a missed housing; investigate"
                )
            records.append(
                dict(
                    sample=int(i),
                    box_index=k,
                    box=box,
                    outcome=outcome,
                    town=data.samples[i]["town"],
                    junction=data.samples[i]["junction_id"],
                    phase=data.samples[i]["phase"],
                    recognized_oracle=bool(oracle),
                    oracle_coordinates=[[rows[j]["x"], rows[j]["y"]] for j in oracle],
                    refined_candidates_inside=len(indices),
                )
            )
    if len(records) != 59:
        raise ValueError("Original refined TRAIN missed-housing count changed")
    summary = dict(
        missed=len(records),
        outcomes=dict(Counter(r["outcome"] for r in records)),
        recognized_missing_oracle=sum(r["recognized_oracle"] for r in records),
        phases=dict(Counter(r["phase"] for r in records)),
    )
    atomic_json(
        destination,
        dict(
            independent_component_recount_exact=True,
            component=counted,
            component_assessment=assessment,
            train_assessment=read(train / "assessment.json"),
            residual_diagnosis_TRAIN_only=True,
            oracle_diagnostic_not_runtime=True,
            residual_summary=summary,
            residuals=records,
            source_sha256=digest(__file__),
            no_new_fit_or_component_reselection=True,
        ),
    )
    print(json.dumps(dict(component=counted, residual_summary=summary), indent=2))


if __name__ == "__main__":
    main()
