#!/usr/bin/env python3
"""Recount frozen verifier results and diagnose housing size; no reselection."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_signal_context import score  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa:E402
from src.navigation.signal_pretrained_verifier import retain_original_states  # noqa:E402


def size_counts(predictions, data):
    bins = {key: dict(labels=0, localized=0, joint=0) for key in ("<8", "8–15", ">=16")}
    for index in data.validation:
        available, matches = set(range(len(data.boxes[index]))), {}
        for peak in predictions[index]:
            eligible = [
                i for i in sorted(available)
                if data.boxes[index][i][0] <= peak["x"] < data.boxes[index][i][2]
                and data.boxes[index][i][1] <= peak["y"] < data.boxes[index][i][3]
            ]
            if eligible:
                i = eligible[0]
                available.remove(i)
                matches[i] = peak
        for i, box in enumerate(data.boxes[index]):
            height = box[3] - box[1]
            row = bins["<8" if height < 8 else "8–15" if height < 16 else ">=16"]
            row["labels"] += 1
            row["localized"] += int(i in matches)
            row["joint"] += int(i in matches and matches[i]["cls"] == box[4])
    return bins


def main():
    root = ROOT / "outputs/navigation_signal_pretrained_20261021"
    def read(p):
        return json.loads(p.read_text())
    result, cache = read(root / "assessment.json"), read(root / "cache.json")
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    original = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {row["sample"]: row["peaks"] for row in read(original)}
    assert len(data.validation) == 144
    assert score(peaks, data) == result["baseline"]
    sizes = {"parent": size_counts(peaks, data)}
    offsets = cache["validation_offsets"]
    sources = {str(original.relative_to(ROOT)): digest(original)}
    for arm in ("rgb", "resnet"):
        path = root / arm / "validation_probability.npy"
        p = np.load(path)
        assert len(p) == offsets[-1] == 38543
        predictions = {
            index: retain_original_states(peaks[index], p[offsets[n]:offsets[n + 1]])
            for n, index in enumerate(data.validation)
        }
        assert score(predictions, data) == result["arms"][arm]["metrics"]
        sizes[arm] = size_counts(predictions, data)
        sources[str(path.relative_to(ROOT))] = digest(path)
    report = dict(
        posthoc_diagnosis_only=True, no_reselection=True, sizes=sizes,
        source_hashes=sources, full_metric_recount=True,
        study_assessment_sha256=digest(root / "assessment.json"),
    )
    atomic_json(ROOT / "reports/navigation_signal_pretrained_20261021.json", report)
    metrics = [result["baseline"]] + [result["arms"][a]["metrics"] for a in ("rgb", "resnet")]
    names = ["Parent", "RGB verifier", "ResNet verifier"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(names, [m["joint"] for m in metrics])
    axes[0].axhline(44, color="red", linestyle="--", label="Parent coverage")
    axes[0].set(title="Correct localization + state /72", ylim=(0, 72))
    axes[0].legend()
    axes[1].bar(names, [m["background_peaks"] for m in metrics])
    axes[1].set(title="Known-background peaks /144 images")
    for n, arm in enumerate(("parent", "rgb", "resnet")):
        rows = list(sizes[arm].values())
        axes[2].bar(
            np.arange(3) + (n - 1) * .25,
            [100 * r["joint"] / max(r["labels"], 1) for r in rows],
            width=.25, label=names[n],
        )
    axes[2].set(xticks=range(3), xticklabels=list(sizes["parent"]), ylim=(0, 100),
                title="Posthoc joint recall by housing height", xlabel="Native pixels")
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Both fixed verifier heads fail coverage protection")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_pretrained_20261021.png", dpi=150)
    plt.close(fig)
    table = "\n".join(
        f"| {name} | {m['localized']}/72 | {m['joint']}/72 | {m['background_peaks']:,} |"
        for name, m in zip(names, metrics)
    )
    size_table = "\n".join(
        f"| {size} | {row['labels']} | {row['joint']} | "
        f"{sizes['rgb'][size]['joint']} | {sizes['resnet'][size]['joint']} |"
        for size, row in sizes["parent"].items()
    )
    report_path = ROOT / "reports/navigation_signal_pretrained_20261021.md"
    report_path.write_text(f"""# Frozen visual verifiers reduce false alarms but lose lights

Both predeclared heads finish in an 87.22-second study and fail coverage protection.
No model is promoted or deployed. ResNet removes 97.9% of background peaks but
loses 18 of the parent's 44 correctly localized and classified lights.

| Model | Localized | Correct localization + state | Background peaks |
|---|---:|---:|---:|
{table}

![Frozen verifier comparison](navigation_signal_pretrained_20261021.png)

Same original 38,543 proposals, 144 validation images and 72 housing labels.
All proposals are scored, without a top-K cap. The binary gate is fixed at 0.5;
original proposal states are retained. A point inside a housing is the localization
metric; these are not bounding-box AP or driving results. Every town and state
must retain parent coverage as well as reducing background peaks by 90%.

TRAIN uses 2,844 images at 21 eligible locations: 51,094 patches, including 5,600
foreground and 45,494 background. Ambiguous projected lights are ignored. The
87-feature RGB head and 1,024-feature frozen ImageNet ResNet18 head each use
TRAIN-only standardization and balanced L2 logistic regression (C=1). They converge
in 133 and 81 iterations, without warnings. Exported probabilities reproduce the
fitted model within 1e-12. Frozen encoder parameters and batch-normalization buffers
are independently unchanged. All source, data, feature and head hashes are retained.

This changes representation, task and data relative to the older scratch context
study, so it does not isolate a causal pretraining benefit. ImageNet supervision
is declared. Inputs remain current RGB crops only, without map, phase-pair, teacher,
absolute-coordinate or metadata inputs. No reserved Town10HD signal location is used.

## Frozen size diagnosis (posthoc; no selection change)

Full metric recount exactly reproduces the stored assessment. Housing-height bins
below describe the same closed predictions; no threshold or model is reselected.

| Housing height (pixels) | Labels | Parent joint | RGB joint | ResNet joint |
|---|---:|---:|---:|---:|
{size_table}

Size is correlated with location, distance and appearance in this small component
set; these counts cannot establish that crop scale causes the losses. Validation
has been inspected repeatedly and is development evidence, not an untouched test.
The frozen proposer also misses 16 housings before any verifier is applied.

Next: use these size counts to decide whether a prospective tighter-scale feature
study is warranted. Keep TRAIN mining, validation, fixed gates and all failed runs.
M2/M3/M5 remain unqualified; no M6 driving or final Town05 tuning has occurred.

[Protocol](navigation_signal_pretrained_verifier_protocol_20261021.md).
Evidence: outputs/navigation_signal_pretrained_20261021. Neither isolated crop
throughput nor this offline fit establishes full runtime latency or reliability.
""")
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
