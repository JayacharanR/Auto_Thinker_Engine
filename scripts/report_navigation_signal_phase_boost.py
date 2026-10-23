#!/usr/bin/env python3
"""Closed state-aware comparison audit; aggregate gains never override state protection."""

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
from src.navigation.signal_phase_supervision import joint_candidates  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa:E402


def main():
    def read(path):
        return json.loads(path.read_text())

    root = ROOT / "outputs/navigation_signal_phase_boost_20261021"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    result, selection = read(root / "assessment.json"), read(root / "selection.json")
    with np.load(cache / "train_features.npz") as a:
        labels = a["labels"]
    with np.load(ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz") as a:
        folds, groups, mask = a["fold"], a["groups"], a["predicted"]
    times = {}
    for candidate in selection["candidates"]:
        arm = candidate["arm"]
        p = np.load(root / f"oof_{arm}.npy")
        times[arm] = []
        for fold in range(3):
            directory = root / f"{arm}_fold{fold}"
            fit = read(directory / "fit.json")
            assert fit["trees"] == 400 and not fit["component_validation_read"]
            assert fit["excluded_fold"] == fold
            assert fit["fitted_examples"] == 3 * int((folds != fold).sum())
            expected = np.load(directory / "train_probability.npy")
            assert np.array_equal(p[folds == fold], expected[folds == fold])
            if arm == "joint100":
                full = np.load(directory / "train_class_probability.npy")
                assert np.array_equal(full[:, 1:].sum(1), expected)
                assert np.allclose(full.sum(1), 1)
            times[arm].append(fit["elapsed_seconds"])
        losses = []
        for group in sorted(set(groups)):
            assert len(set(folds[groups == group])) == 1
            at = mask & (groups == group)
            loss = float(
                -0.5
                * (
                    np.log(np.clip(p[at & (labels == 1)], 1e-12, 1 - 1e-12)).mean()
                    + np.log1p(-np.clip(p[at & (labels == 0)], 1e-12, 1 - 1e-12)).mean()
                )
            )
            assert np.isclose(loss, candidate["by_location"][group], rtol=0, atol=1e-14)
            losses.append(loss)
        assert np.isclose(np.mean(losses), candidate["macro_logloss"], rtol=0, atol=1e-14)
    expected = min(
        selection["candidates"], key=lambda r: (r["macro_logloss"], r["arm"] != "binary400")
    )
    assert expected == selection["selected"] and result["selected_arm"] == "joint100"
    assert read(root / "final/worker_identity.json")["started_at"] >= selection["selected_at"]
    assert read(root / "final/fit.json")["trees"] == 400
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    path = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {r["sample"]: r["peaks"] for r in read(path)}
    offsets = read(cache / "cache.json")["validation_offsets"]
    p = np.load(root / "final/validation_class_probability.npy")
    assert np.array_equal(p[:, 1:].sum(1), np.load(root / "final/validation_probability.npy"))
    actual = score(
        {
            i: joint_candidates(peaks[i], p[offsets[n] : offsets[n + 1]])
            for n, i in enumerate(data.validation)
        },
        data,
    )
    assert actual == result["metrics"] and score(peaks, data) == result["baseline"]
    assert not result["component_pass"]
    audit = dict(
        independent_oof_recount=True,
        independent_component_recount=True,
        all_seven_fits_400_trees=True,
        selected_before_final_fit=True,
        fold_fit_seconds=times,
        assessment_sha256=digest(root / "assessment.json"),
    )
    atomic_json(ROOT / "reports/navigation_signal_phase_boost_20261021.json", audit)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, (name, metrics) in enumerate(
        (("Parent", result["baseline"]), ("State-aware", result["metrics"]))
    ):
        axes[0].bar(
            np.arange(3) + (n - 0.5) * 0.35,
            [metrics["classes"][str(c)]["joint"] for c in (1, 2, 3)],
            width=0.35,
            label=name,
        )
    axes[0].set(
        xticks=range(3),
        xticklabels=["Red /36", "Yellow /18", "Green /18"],
        title="State coverage: aggregate equality hides regression",
        ylim=(0, 36),
    )
    axes[0].legend()
    axes[1].bar(
        [c["arm"] for c in selection["candidates"]],
        [c["macro_logloss"] for c in selection["candidates"]],
    )
    axes[1].set(title="TRAIN-location OOF selection", ylabel="Macro binary log loss ↓")
    fig.suptitle("44/72 joint lights, but state/town protection still fails")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_phase_boost_20261021.png", dpi=150)
    plt.close(fig)
    (
        ROOT / "reports/navigation_signal_phase_boost_20261021.md"
    ).write_text("""# State-aware supervision matches the total, but loses yellow coverage

The seven-fit study completes in48.71seconds. TRAIN-location folds select joint100
(macro binary OOF logloss0.360094) over binary400(0.402281). The selected head
localizes45/72lights and correctly classifies44, with1,171background detections
(96.86%below parent37,241). Aggregate44matches the parent, but the component gate
FAILS: yellow15→10 and Town10HD10→9. No deployment, threshold change or promotion.

| State | Parent joint | State-aware joint | Labels |
|---|---:|---:|---:|
| Red |22|22|36|
| Yellow |15|10|18|
| Green |7|12|18|

![State protections and TRAIN selection](navigation_signal_phase_boost_20261021.png)

Town joint counts: Town01 7/8, Town03 16/20, Town04 12/12, Town10HD 9/32.
No state or town can compensate for another in the unchanged protection gate.
The result improves green classification, but does not solve yellow observability.

Same original51,094TRAINpoints/87RGBfeatures plus the exact two gamma variants,
153,282rows in each final fit; no data copies, new native collection or VAL transforms.
State labels are derived only from existing TRAIN phase supervision. Foreground
and background each receive half the training weight; each state receives one sixth.
Runtime model inputs remain RGB-derived features, without phase/teacher/world metadata.

Both arms use400trees:400binary iterations versus100four-class iterations, with
the same leaf limits and settings. This matches tree count, not exact CPU work or
parameter count; all actual fit times are retained in the JSON audit. Original
binary100/gamma and all prior failed models remain unchanged. Six internal folds
predict original TRAIN points; only the TRAIN-selected final head reaches the
144image/72housing component evaluation. Foreground gate stays0.5; joint state
is argmax of predicted foreground-state probabilities. All38,543proposals scored.

Independent audit reproduces all six OOF vectors, whole-location separation,
sample counts, macro loss, selection before final fitting, four-class probability
normalization/marginalization, all seven400tree receipts, and every final metric.
Serialized predictions exactly reproduce the fitted model. Ten focused
phase-label/weight/inference/augmentation/group/research-boundary tests passed.

Next test an explicit RGB opponent-color representation with TRAIN-only selection
before another component comparison. Bright yellow differs from gray/white by
channel relationships; raw-channel tree splits may represent those inefficiently.
This is a hypothesis, not an explanation established by these outcomes. Preserve
the five yellow regressions, original denominators and all protected locations.

No native planner or M6 driving. M2/M3/M5 remain unqualified. Original storage
baseline14GBnet/10GBminimum stays active. Component VAL has been inspected repeatedly;
this remains engineering evidence, not a final-test claim or M4 primary reselection.
[Protocol](navigation_signal_phase_boost_protocol_20261021.md).
Evidence: outputs/navigation_signal_phase_boost_20261021.
""")
    print(
        "All seven tree counts, OOF selection and final metrics verified; state protection FAILS."
    )


if __name__ == "__main__":
    main()
