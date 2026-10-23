#!/usr/bin/env python3
"""Frozen scale comparison and TRAIN/VAL diagnosis; no fit or reselection."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_navigation_signal_pretrained import size_counts  # noqa:E402
from scripts.train_navigation_signal_context import score  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_context import label_proposals  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa:E402
from src.navigation.signal_pretrained_verifier import retain_original_states  # noqa:E402


def binary_metrics(p, labels):
    supported = labels >= 0
    p, y = np.asarray(p)[supported], np.asarray(labels)[supported]
    positive = y > 0
    recall = float((p[positive] >= .5).mean())
    specificity = float((p[~positive] < .5).mean())
    return dict(
        supported=int(supported.sum()), unknown=int((~supported).sum()),
        positives=int(positive.sum()), recall=recall, specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2,
        auc=float(roc_auc_score(positive, p)),
    )


def main():
    def read(path):
        return json.loads(path.read_text())

    original = ROOT / "outputs/navigation_signal_pretrained_20261021"
    tight = ROOT / "outputs/navigation_signal_tight_20261021"
    result = read(tight / "assessment.json")
    assert result["complete"] and not result["arms"]["resnet"]["component_pass"]
    assert all(read(tight / "paired_parity.json").values())
    assert digest(original / "rgb/head.npz") == digest(tight / "rgb/head.npz")
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    source = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {row["sample"]: row["peaks"] for row in read(source)}
    val_labels = np.concatenate([
        label_proposals(peaks[i], data.boxes[i], data.background[i]) for i in data.validation
    ])
    rows = read(original / "train_rows.json")
    # Remove GT-only anchors for a less biased TRAIN-versus-VAL diagnostic.
    train_peaks = read(original / "train_proposals.json")
    proposal_points = {
        int(i): {(p["x"], p["y"]) for p in points} for i, points in train_peaks.items()
    }
    predicted = np.array([
        (r["x"], r["y"]) in proposal_points[r["sample"]] for r in rows
    ])
    diagnosis, sources = {}, {}
    for name, root, arm in (
        ("RGB", original, "rgb"),
        ("ResNet 32/96", original, "resnet"),
        ("ResNet 8/16", tight, "resnet"),
    ):
        cache = read(root / "cache.json")
        for split in ("train", "validation"):
            path = root / f"{split}_features.npz"
            assert digest(path) == cache[f"{split}_features_sha256"]
        with np.load(root / "train_features.npz") as a:
            train_labels = a["labels"]
        with np.load(root / arm / "head.npz") as a:
            train_p = a["train_probability"]
        val_p = np.load(root / arm / "validation_probability.npy")
        offsets = cache["validation_offsets"]
        predictions = {
            i: retain_original_states(peaks[i], val_p[offsets[n]:offsets[n + 1]])
            for n, i in enumerate(data.validation)
        }
        metrics = score(predictions, data)
        assert metrics == read(root / "assessment.json")["arms"][arm]["metrics"]
        diagnosis[name] = dict(
            train=binary_metrics(train_p, train_labels),
            train_predicted_only=binary_metrics(train_p[predicted], train_labels[predicted]),
            validation=binary_metrics(val_p, val_labels),
            metrics=metrics, sizes=size_counts(predictions, data),
        )
        for file in ("assessment.json", f"{arm}/head.npz", f"{arm}/validation_probability.npy"):
            path = root / file
            sources[str(path.relative_to(ROOT))] = digest(path)
    atomic_json(ROOT / "reports/navigation_signal_tight_20261021.json", dict(
        diagnosis=diagnosis, posthoc_diagnosis_only=True, no_reselection=True,
        source_hashes=sources, exact_rgb_control=True, full_metric_recount=True,
    ))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(diagnosis)
    for n, split in enumerate(("train_predicted_only", "validation")):
        axes[0].bar(np.arange(3) + (n - .5) * .35,
                    [100 * d[split]["balanced_accuracy"] for d in diagnosis.values()],
                    width=.35, label=split.replace("_", " "))
    axes[0].set(xticks=range(3), xticklabels=names, ylim=(0, 100),
                title="Foreground balanced accuracy (%)")
    axes[0].legend()
    axes[1].bar(names, [d["metrics"]["joint"] for d in diagnosis.values()])
    axes[1].axhline(44, color="red", linestyle="--", label="Parent protection")
    axes[1].set(ylim=(0, 72), title="Correct localization + state /72")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Tighter crops alone do not pass")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_tight_20261021.png", dpi=150)
    plt.close(fig)
    table = "\n".join(
        f"| {name} | {d['metrics']['joint']}/72 | {d['metrics']['background_peaks']:,} | "
        f"{100*d['train_predicted_only']['balanced_accuracy']:.1f}% | "
        f"{100*d['validation']['balanced_accuracy']:.1f}% | "
        f"{100*d['validation']['recall']:.1f}% |"
        for name, d in diagnosis.items()
    )
    report = ROOT / "reports/navigation_signal_tight_20261021.md"
    report.write_text(f"""# Tighter frozen-feature crops are insufficient

The paired crop-scale study completes in 89.46 seconds. ResNet 8/16-pixel crops
give 33/72 localized lights, 28 with correct state, and 2,250 background peaks.
The 32/96-pixel version gave 29/26/789. Both fail the unchanged parent coverage
requirement of 44 joint lights plus every town and state. No promotion or deployment.

| Head | Joint | Background | TRAIN balanced acc.* | VAL balanced acc. | VAL recall |
|---|---:|---:|---:|---:|---:|
{table}

![Paired scale comparison and diagnostic](navigation_signal_tight_20261021.png)

*TRAIN diagnostic excludes GT-only bulb anchors; the fitted head used the same
51,094 examples as registered. TRAIN backgrounds were sampled, whereas all VAL
proposals were retained. Balanced accuracy adjusts the positive/background class
frequency but does not eliminate differences within those classes. This is a
posthoc diagnosis, not a threshold search or new selection rule. VAL has 73 known
positive proposals, not 73 independent lights; duplicates and shared static cases
make image-level counts correlated. There are 72 labeled housings.

Paired TRAIN points/labels, RGB arrays and VAL offsets match exactly. The repeated
RGB probabilities and entire head archive match bit-for-bit. The same frozen
ResNet18 weights, 1,024 features, pooling, balanced linear head, C=1 and threshold
0.5 are retained; only the two crop sizes change. ResNet converges in 92 iterations,
with no warnings; all parameters/buffers stay frozen. Saved linear probabilities
match the fitted head within 1e-12. Independent closed-prediction recount matches
every reported metric. All registered source hashes are unchanged during the run.

The result does not support a crop-only fix. A large TRAIN-to-VAL gap points toward
representation/generalization limits; high TRAIN accuracy alone cannot justify a
longer fit. Before further training, inspect the errors and augmentation coverage
on TRAIN-only location folds. Keep this component VAL set for declared development
comparison and leave held-out development routes / final Town05 untouched.

No new native data or driving results. Eleven focused tests passed, including
crop alignment/boundary padding, frozen encoder behavior and research boundaries.
The original 14 GB net storage cap / 10 GB minimum and baseline remain unchanged.
M2/M3/M5 are unqualified; M6 has not run. Full runtime cost remains unmeasured.

[Protocol](navigation_signal_tight_verifier_protocol_20261021.md),
[prior study](navigation_signal_pretrained_20261021.md).
Evidence: outputs/navigation_signal_tight_20261021.
""")
    print(json.dumps({k: {s: v[s] for s in ("train", "train_predicted_only", "validation")}
                      for k, v in diagnosis.items()}, indent=2))


if __name__ == "__main__":
    main()
