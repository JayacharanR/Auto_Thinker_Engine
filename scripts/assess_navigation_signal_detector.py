#!/usr/bin/env python3
"""Audit preserved detector predictions with independent one-to-one box matching."""

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
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_reference_detector import iou  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def match_boxes(truth, predicted, threshold):
    """Maximum-cardinality bipartite matching; state/color never influences matches."""
    edges = [
        sorted(
            [p for p, box in enumerate(predicted) if iou(t, box) >= threshold],
            key=lambda p: iou(t, predicted[p]),
            reverse=True,
        )
        for t in truth
    ]
    owners = {}

    def assign(t, seen):
        for p in edges[t]:
            if p in seen:
                continue
            seen.add(p)
            if p not in owners or assign(owners[p], seen):
                owners[p] = t
                return True
        return False

    for t in sorted(range(len(truth)), key=lambda t: len(edges[t])):
        assign(t, set())
    return sorted((t, p) for p, t in owners.items())


def audit_predictions(rows, labels):
    result = dict(denominator=0, iou3=0, iou5=0, iou3_state_correct=0)
    assignments = []
    by_town = defaultdict(lambda: [0, 0])
    for index, row in enumerate(rows):
        sample, predicted = row["sample"], row["predictions"]
        truth = [
            label
            for label in labels[sample["town"], sample["case_id"], sample["view"]]
            if sample["phase"] != "yellow" or label["yellow_color_supported"]
        ]
        boxes = [label["crop_xyxy"] for label in truth]
        matches = match_boxes(boxes, [p["box"] for p in predicted], 0.3)
        result["denominator"] += len(boxes)
        result["iou3"] += len(matches)
        result["iou5"] += len(match_boxes(boxes, [p["box"] for p in predicted], 0.5))
        expected = {"red": 0, "red_repeat": 0, "yellow": 1, "green": 2}[sample["phase"]]
        result["iou3_state_correct"] += sum(predicted[p]["state"] == expected for _, p in matches)
        by_town[sample["town"]][0] += len(matches)
        by_town[sample["town"]][1] += len(boxes)
        assignments.extend(
            dict(row=index, head=truth[t]["head"], prediction=p, truth=expected) for t, p in matches
        )
    return dict(**result, by_town=dict(by_town)), assignments


def main():
    output = ROOT / "outputs/navigation_m2_signal_detector_20261016"
    corpus = ROOT / "outputs/navigation_m2_signal_corpus_v2_20261016"
    review = json.loads((output / "review.json").read_text())
    request = json.loads((output / "request.json").read_text())
    assert digest(output / "best_component.pt") == review["component_sha256"]
    assert digest(corpus / "data_audit.json") == request["corpus_audit_sha256"]
    for relative, expected in request["source_hashes"].items():
        assert digest(output / "source" / relative) == expected, relative
    labels = defaultdict(list)
    for label in json.loads((corpus / "data_audit.json").read_text())["labels"]:
        if label["static_phase_response_supported"]:
            labels[label["town"], label["case_id"], label["view"]].append(label)
    history = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    results, selected_rows, assignments = [], None, None
    evidence = {}
    for entry in history:
        path = output / f"predictions_step{entry['step']}.json"
        rows = json.loads(path.read_text())["rows"]
        counts, matched = audit_predictions(rows, labels)
        assert counts["denominator"] == 72
        results.append(dict(**entry, one_to_one=counts))
        evidence[str(path.relative_to(ROOT))] = digest(path)
        if entry["step"] == review["selected_step"]:
            selected_rows, assignments = rows, matched
    assert selected_rows is not None
    atomic_json(
        output / "one_to_one_audit.json",
        dict(
            results=results,
            selected_step=review["selected_step"],
            assignments=assignments,
            source_sha256=digest(Path(__file__)),
            evidence_sha256=evidence,
            scope="Supplemental sparse-box matching audit; no new selection/gate or driving claim",
        ),
    )
    figures = ROOT / "reports/figures"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    steps = [r["step"] for r in results]
    axes[0].plot(steps, [r["selection_score"] for r in results], "o-")
    axes[0].axvline(review["selected_step"], color="gray", ls="--", label="Selected step50")
    axes[0].set(
        xlabel="Updates",
        ylabel="Fixed-seed validation detector loss",
        title="Frozen selection rule",
    )
    axes[0].legend()
    axes[1].plot(steps, [r["one_to_one"]["iou3"] for r in results], "o-", label="Detected IoU≥.3")
    axes[1].plot(
        steps,
        [r["one_to_one"]["iou3_state_correct"] for r in results],
        "o-",
        label="Also correct state",
    )
    axes[1].set(
        ylim=(0, 72),
        xlabel="Updates",
        ylabel="One-to-one labels /72",
        title="Sparse development recall",
    )
    axes[1].legend()
    axes[2].plot(
        steps,
        [r["counts"]["approximate_background_candidates"] / 144 for r in results],
        "o-",
        color="#b77332",
    )
    axes[2].set(
        xlabel="Updates", ylabel="Candidates/image", title="Centers outside projected lamp regions"
    )
    fig.suptitle("M2 small-anchor adaptation — four TRAIN development junctions, no driving result")
    fig.savefig(figures / "navigation_m2_signal_detector_20261016.png", dpi=170)
    plt.close(fig)
    # Illustrations: largest supported red housing per junction, chosen by geometry only.
    examples = {}
    for row in selected_rows:
        s = row["sample"]
        if s["phase"] != "red":
            continue
        candidates = labels[s["town"], s["case_id"], s["view"]]
        if not candidates:
            continue
        area = max(
            (b["crop_xyxy"][2] - b["crop_xyxy"][0]) * (b["crop_xyxy"][3] - b["crop_xyxy"][1])
            for b in candidates
        )
        key = s["town"], s["junction_id"]
        if key not in examples or area > examples[key][0]:
            examples[key] = area, row
    canvas = Image.new("RGB", (1024, 624), "#19212d")
    for n, (_, row) in enumerate(examples.values()):
        s = row["sample"]
        image = Image.open(corpus / s["path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for label in labels[s["town"], s["case_id"], s["view"]]:
            draw.rectangle(label["crop_xyxy"], outline="cyan", width=1)
        for predicted in row["predictions"]:
            draw.rectangle(
                predicted["box"], outline=["red", "yellow", "lime"][predicted["state"]], width=1
            )
        x, y = (n % 2) * 512, (n // 2) * 312
        canvas.paste(image, (x, y + 24))
        ImageDraw.Draw(canvas).text(
            (x + 6, y + 5),
            f"{s['town']} junction{s['junction_id']} | cyan: annotation | color: predicted state",
            fill="white",
        )
    canvas.save(figures / "navigation_m2_signal_detector_examples_20261016.png")
    print(
        json.dumps(
            dict(
                selected_step=review["selected_step"],
                results=[dict(step=r["step"], **r["one_to_one"]) for r in results],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
