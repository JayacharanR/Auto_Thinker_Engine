#!/usr/bin/env python3
"""Independently recount closed TRAIN-fold selection and the single VAL assessment."""

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


def main():
    def read(path):
        return json.loads(path.read_text())

    root = ROOT / "outputs/navigation_signal_regularized_20261021"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    request, result = read(root / "request.json"), read(root / "assessment.json")
    selected = read(root / "selection.json")
    with np.load(root / "folds.npz") as a:
        folds, groups, mask = a["fold"], a["groups"], a["predicted"]
    with np.load(cache / "train_features.npz") as a:
        labels = a["labels"]
    assert digest(cache / "train_features.npz") == request["train_features_sha256"]
    assert len(set(groups)) == 21 and len(groups) == 51094 and mask.sum() == 48865
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1
    for ci, row in enumerate(selected["candidates"]):
        p = np.load(root / f"oof_c{ci}.npy")
        for fi in range(3):
            directory = root / f"c{ci}_fold{fi}"
            receipt = read(directory / "fit.json")
            assert receipt["converged"] and receipt["excluded_fold"] == fi
            assert receipt["train_examples"] == int((folds != fi).sum())
            with np.load(directory / "head.npz") as a:
                assert np.array_equal(p[folds == fi], a["train_probability"][folds == fi])
        losses = []
        for group in sorted(set(groups)):
            at = mask & (groups == group)
            terms = [-np.log(np.clip(p[at & (labels == 1)], 1e-12, 1 - 1e-12)).mean(),
                     -np.log1p(-np.clip(p[at & (labels == 0)], 1e-12, 1 - 1e-12)).mean()]
            value = float(np.mean(terms))
            assert np.isclose(value, row["by_location"][group], rtol=0, atol=1e-14)
            losses.append(value)
        assert np.isclose(np.mean(losses), row["macro_logloss"], rtol=0, atol=1e-14)
    expected = min(selected["candidates"], key=lambda r: (r["macro_logloss"], r["C"]))
    assert expected == selected["selected"] and result["selected_C"] == expected["C"]
    assert read(root / "final/fit.json")["converged"]
    assert read(root / "final/worker_identity.json")["started_at"] >= selected["selected_at"]
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    original = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {r["sample"]: r["peaks"] for r in read(original)}
    with np.load(cache / "validation_features.npz") as a:
        offsets = a["offsets"]
    p = np.load(root / "validation_probability.npy")
    metrics = score({
        i: retain_original_states(peaks[i], p[offsets[n]:offsets[n + 1]])
        for n, i in enumerate(data.validation)
    }, data)
    assert metrics == result["metrics"] and score(peaks, data) == result["baseline"]
    assert not result["component_pass"] and not selected["component_validation_used"]
    atomic_json(ROOT / "reports/navigation_signal_regularization_20261021.json", dict(
        independent_oof_recount=True, independent_component_recount=True,
        whole_locations_verified=21, convergence_verified_fits=16,
        selected_before_final_fit=True, selected_C=result["selected_C"],
        selection_sha256=digest(root / "selection.json"),
        assessment_sha256=digest(root / "assessment.json"), component_pass=False,
    ))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    candidates = selected["candidates"]
    axes[0].semilogx([r["C"] for r in candidates],
                    [r["macro_logloss"] for r in candidates], "o-")
    axes[0].set(xlabel="C (smaller = stronger L2 penalty)", ylabel="Macro log loss ↓",
                title="Whole TRAIN-junction out-of-fold selection")
    axes[1].bar(["Parent", "C=1", "TRAIN-selected"], [44, 26, metrics["joint"]])
    axes[1].axhline(44, color="red", linestyle="--")
    axes[1].set(ylim=(0, 72), title="Component VAL joint lights /72")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Regularization improves the head but still fails coverage")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_regularization_20261021.png", dpi=150)
    plt.close(fig)
    rows = "\n".join(f"| {r['C']:g} | {r['macro_logloss']:.6f} |" for r in candidates)
    report = ROOT / "reports/navigation_signal_regularization_20261021.md"
    report.write_text(f"""# TRAIN-selected regularization remains unqualified

All 16 CPU fits converge in a 42.49-second study. Whole-junction internal folds
select C=0.0001 before the final fit or component-validation assessment. The final
head localizes 33/72 lights, classifies 30 correctly and leaves 1,919 background
peaks. That improves the previous ResNet head's 26 joint lights but still loses
14 of the parent's 44. The 94.85% background reduction does not compensate for
lost coverage: FAIL, no deployment or driving promotion.

| Fixed C | Mean TRAIN-location OOF balanced log loss ↓ |
|---|---:|
{rows}

![TRAIN selection and component result](navigation_signal_regularization_20261021.png)

## Selection and independent checks

Same larger-crop frozen ResNet features and original 51,094 TRAIN points. Three
folds keep all frames/views/phases of each of 21 TRAIN junctions together. Each
fold fits its own scaler on that fold's training portion only. Selection scores
48,865 predicted TRAIN points, excluding GT-only extra anchors; each location
and each class receive equal weight. All original rows remain available for fit.
Original component VAL is not used to select C. Penalty values, tie rule and
fitting recipe were registered before execution; threshold stays at 0.5.

The selected penalty is at the tested boundary. This does not establish the best
possible penalty, and no posthoc range extension is included in this result.
The component set has been inspected repeatedly; this remains development
evidence, not an untouched test. Internal-fold scores also share the predeclared
proposer and are not claims about entirely unseen end-to-end systems.

Independent recount verifies all 15 out-of-fold arrays against their head
predictions, whole-location separation, class/location aggregation, selector,
all 16 convergence receipts, and selection timestamp before final fitting.
The single saved component prediction vector reproduces all reported metrics.
Final town joint counts: Town01 7/8, Town03 8/20, Town04 12/12, Town10HD 3/32.
Final state joint counts: red 16/36, green 6/18, yellow 8/18.

No new recordings or feature cache copies. Original storage guardian baseline,
14 GB net cap and 10 GB minimum remain. All source hashes and failed candidates
are retained. M2/M3/M5 remain unqualified; M6 has not run.

Next: assess TRAIN-only appearance augmentation and feature adaptation using the
same whole-location internal folds before another component comparison. The
strong TRAIN fit and remaining location gap do not justify a longer unchanged
fit. Reserved signal locations, original M4 primary and final Town05 remain fixed.

[Protocol](navigation_signal_regularization_protocol_20261021.md),
[preceding scale diagnosis](navigation_signal_tight_20261021.md).
Evidence: outputs/navigation_signal_regularized_20261021.
""")
    print("Verified all 16 fits, TRAIN-only selection and final component recount; gate FAILED.")


if __name__ == "__main__":
    main()
