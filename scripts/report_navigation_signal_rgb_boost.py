#!/usr/bin/env python3
"""Recount the closed RGB augmentation study, including every remaining lost light."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inspect_navigation_signal_errors import matched  # noqa:E402
from scripts.train_navigation_signal_context import score  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa:E402
from src.navigation.signal_pretrained_verifier import retain_original_states  # noqa:E402


def main():
    def read(path):
        return json.loads(path.read_text())

    root = ROOT / "outputs/navigation_signal_rgb_boost_v2_20261021"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    result, selection = read(root / "assessment.json"), read(root / "selection.json")
    rows, dataset = read(cache / "train_rows.json"), read(cache / "dataset.json")
    labels = np.array([row["label"] for row in rows])
    phases = np.array(
        [dataset["samples"][r["sample"]]["phase"].replace("red_repeat", "red") for r in rows]
    )
    with np.load(ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz") as a:
        folds, groups, mask = a["fold"], a["groups"], a["predicted"]
    phase_diagnosis = {}
    for candidate in selection["candidates"]:
        arm = candidate["arm"]
        p = np.load(root / f"oof_{arm}.npy")
        for fi in range(3):
            directory = root / f"{arm}_fold{fi}"
            fit = read(directory / "fit.json")
            assert fit["iterations"] == 100 and not fit["component_validation_read"]
            assert fit["excluded_fold"] == fi
            assert fit["original_train_examples"] == int((folds != fi).sum())
            assert fit["fitted_examples"] == int((folds != fi).sum()) * (3 if arm == "gamma" else 1)
            assert np.array_equal(
                p[folds == fi], np.load(directory / "train_probability.npy")[folds == fi]
            )
        losses = []
        for group in sorted(set(groups)):
            assert len(set(folds[groups == group])) == 1
            at = mask & (groups == group)
            value = float(
                -0.5
                * (
                    np.log(np.clip(p[at & (labels == 1)], 1e-12, 1 - 1e-12)).mean()
                    + np.log1p(-np.clip(p[at & (labels == 0)], 1e-12, 1 - 1e-12)).mean()
                )
            )
            assert np.isclose(value, candidate["by_location"][group], rtol=0, atol=1e-14)
            losses.append(value)
        assert np.isclose(np.mean(losses), candidate["macro_logloss"], rtol=0, atol=1e-14)
        phase_diagnosis[arm] = {}
        for phase in ("red", "green", "yellow"):
            at = mask & (labels == 1) & (phases == phase)
            phase_diagnosis[arm][phase] = dict(
                points=int(at.sum()), recall=float((p[at] >= 0.5).mean())
            )
    chosen = min(selection["candidates"], key=lambda r: (r["macro_logloss"], r["arm"] != "plain"))
    assert chosen == selection["selected"] and chosen["arm"] == result["selected_arm"]
    assert read(root / "final/worker_identity.json")["started_at"] >= selection["selected_at"]
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    parent_path = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {r["sample"]: r["peaks"] for r in read(parent_path)}
    offsets = read(cache / "cache.json")["validation_offsets"]
    p = np.load(root / "final/validation_probability.npy")
    predictions, parent_order, lost = {}, {}, []
    for n, index in enumerate(data.validation):
        probabilities = p[offsets[n] : offsets[n + 1]]
        predictions[index] = retain_original_states(peaks[index], probabilities)
        parent_order[index] = [q for q, v in zip(peaks[index], probabilities) if v >= 0.5]
        old, new = (
            matched(peaks[index], data.boxes[index]),
            matched(predictions[index], data.boxes[index]),
        )
        for bi, box in enumerate(data.boxes[index]):
            if bi not in old or old[bi]["cls"] != box[4]:
                continue
            if bi in new and new[bi]["cls"] == box[4]:
                continue
            inside = [
                dict(q, foreground_probability=float(v))
                for q, v in zip(peaks[index], probabilities)
                if box[0] <= q["x"] < box[2] and box[1] <= q["y"] < box[3]
            ]
            lost.append(
                dict(
                    sample=index,
                    town=data.samples[index]["town"],
                    box=box,
                    parent=old[bi],
                    current=new.get(bi),
                    inside=inside,
                )
            )
    assert score(predictions, data) == result["metrics"]
    assert score(peaks, data) == result["baseline"]
    assert score(parent_order, data) == result["metrics"]
    assert len(lost) == 3 and not result["component_pass"]
    atomic_json(
        ROOT / "reports/navigation_signal_rgb_boost_20261021.json",
        dict(
            independent_oof_recount=True,
            independent_component_recount=True,
            original_order_diagnostic_unchanged=True,
            lost_parent_correct=lost,
            train_oof_phase_diagnosis=phase_diagnosis,
            no_reselection=True,
            assessment_sha256=digest(root / "assessment.json"),
            selection_sha256=digest(root / "selection.json"),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["Parent", "ResNet C=1", "ResNet regularized", "RGB gamma"]
    axes[0].bar(names, [44, 26, 30, 41])
    axes[0].axhline(44, color="red", linestyle="--", label="Parent coverage")
    axes[0].set(title="Joint localization + state /72", ylim=(0, 72))
    axes[0].legend()
    axes[1].bar(names, [37241, 789, 1919, 1696])
    axes[1].set(title="Known-background detections /144 images")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Nonlinear RGB closes much of the gap; three yellow lights still lost")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_rgb_boost_20261021.png", dpi=150)
    plt.close(fig)
    report = ROOT / "reports/navigation_signal_rgb_boost_20261021.md"
    report.write_text("""# Nonlinear RGB improves coverage, but misses the gate by three lights

Seven fixed-budget CPU fits complete in 79.65 seconds including feature extraction.
Whole TRAIN-junction folds select gamma augmentation: macro binary logloss
0.360422 versus plain 0.390922. The selected final head gives 47/72 localized lights,
41 with correct state, and 1,696 known-background peaks. Parent: 56/44/37,241.
Background peaks fall 95.45%, but three parent-correct yellow lights are lost.
The unchanged coverage gate FAILS; no deployment or driving promotion.

![Component comparison](navigation_signal_rgb_boost_20261021.png)

Town joint counts: Town01 7/8, Town03 15/20, Town04 12/12 and Town10HD 7/32.
The first three equal the parent; Town10HD loses three. State joint counts:
red22/36 and green7/18 equal parent; yellow12/18 versus parent15/18.

## What changed and what was checked

Same 51,094 TRAIN points, 21 locations, 87 native-RGB features, frozen proposer,
144 VAL images /72 housings and all38,543proposals. A nonlinear histogram-boosted
classifier replaces the linear head. Paired plain/gamma arms share one fixed
100-iteration recipe. Gamma .7/1.4 transforms full TRAIN images before context
averaging; final fit uses153,282original/augmented points. Channel order and pixel
geometry are preserved; this is approximate photometric variation, not new native
weather. Original untransformed features reproduce the existing FP16 cache exactly.

Three internal folds keep entire TRAIN junctions together. Augmented views follow
their source fold. The selector scores original predicted points only, excluding
GT-only anchors, and uses no original component VAL. Only the chosen gamma arm
is fitted on all TRAIN data and evaluated on component VAL. No threshold search,
plain-arm component assessment, or reselection. Every model, OOF vector and source
hash is retained. Serialized predictions match exactly. An independent recount
reproduces all six OOF vectors, group/class aggregation, selection before final fit,
and all final metrics. Fixed iterations are not a convergence claim.

## Remaining failure diagnosis

The three lost yellow housings are samples362/374/382, all Town10HD. Each has
one original proposal inside it, with foreground probability0.300916/0.108676/
0.137065, below the fixed0.5gate. All are filtered out; changing proposal ranking
to original parent order gives identical counts. This posthoc diagnostic rules
out ranking as the explanation here; it does not change selection or thresholds.

TRAIN OOF positive recall also remains incomplete. Plain/gamma recall respectively:
red51.97%/58.16% (1697points), green49.29%/55.18% (850), yellow47.82%/54.00% (824).
These are correlated proposal points, not independent encounter success rates.
The previous ResNet regularization selector had lower internal binary logloss
(.299255) but only30joint VAL lights; a lower average loss does not guarantee
every state's coverage. No primary research model is reselected on these results.

Next: inspect phase-aware foreground supervision and the remaining appearance gap
using TRAIN-only folds, with a prospectively registered objective/architecture.
Keep the three errors in the validation denominator; do not lower the gate to
pass them. Native pixel inspection confirmed genuine single-pixel emissive
responses, not loader corruption or justification for relabeling.

## Execution and limits

Original output navigation_signal_rgb_boost_20261021 retains a failed empty
preflight after the storage guardian disappeared during an execution gap. It read
no data and performed no fit. Host liveness was checked, the same immutable
guardian baseline resumed, and v2 retained the original1800second deadline.
No scientific trial was repeated, no allowance reset and no artifact deleted.
The100MBstage cap / original14GBnet cap /10GBminimum remain. Seven focused
augmentation/fold/research-boundary tests passed. No native data were collected.

Component validation has been repeatedly inspected; this is engineering evidence,
not a held-out driving result. Point-in-housing metrics are not bounding-box AP.
No reserved TRAIN location was borrowed, no phase/teacher/world metadata enters
inference, and no final Town05 trial has run. M2/M3/M5 remain unqualified.

[Protocol](navigation_signal_rgb_boost_protocol_20261021.md).
Evidence: outputs/navigation_signal_rgb_boost_v2_20261021.
""")
    print(
        "Verified six OOF vectors and final metrics; all three losses are filtered yellow points."
    )


if __name__ == "__main__":
    main()
