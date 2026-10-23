#!/usr/bin/env python3
"""Audit closed fusion selection and point metrics; retain every failed gate."""

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

    root = ROOT / "outputs/navigation_signal_proposer_fusion_20261021"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    old = ROOT / "outputs/navigation_signal_phase_boost_20261021"
    result, selection = read(root / "assessment.json"), read(root / "selection.json")
    if read(root / "progress.json")["status"] != "complete":
        raise ValueError("Only closed complete study can be reported")
    with np.load(cache / "train_features.npz") as a:
        labels = a["labels"]
    with np.load(ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz") as a:
        folds, groups, predicted = a["fold"], a["groups"], a["predicted"]
    times = {}
    for candidate in selection["candidates"]:
        arm = candidate["arm"]
        probabilities = np.load(root / f"oof_{arm}.npy")
        times[arm] = []
        for fi in range(3):
            name = f"{arm}_fold{fi}"
            fit = read(root / name / "fit.json")
            assert fit["trees"] == 400 and fit["iterations"] == 100
            assert not fit["component_validation_read"] and fit["serialization_parity"]
            assert fit["excluded_fold"] == fi
            assert fit["fitted_examples"] == 3 * int((folds != fi).sum())
            assert fit["input_features"] == (87 if arm == "joint100" else 123)
            full = np.load(root / name / "train_class_probability.npy")
            p = full[:, 1:].sum(1)
            assert np.allclose(full.sum(1), 1)
            assert np.array_equal(p[folds == fi], probabilities[folds == fi])
            if arm == "joint100":
                assert np.array_equal(full, np.load(old / name / "train_class_probability.npy"))
            times[arm].append(fit["elapsed_seconds"])
        losses = []
        for group in sorted(set(groups)):
            assert len(set(folds[groups == group])) == 1
            at = predicted & (groups == group)
            p = np.clip(probabilities, 1e-12, 1 - 1e-12)
            loss = float(
                -0.5
                * (np.log(p[at & (labels == 1)]).mean() + np.log1p(-p[at & (labels == 0)]).mean())
            )
            assert np.isclose(loss, candidate["by_location"][group], rtol=0, atol=1e-14)
            losses.append(loss)
        assert np.isclose(np.mean(losses), candidate["macro_logloss"], rtol=0, atol=1e-14)
    chosen = min(
        selection["candidates"], key=lambda r: (r["macro_logloss"], r["arm"] != "joint100")
    )
    assert chosen == selection["selected"] and chosen["arm"] == result["selected_arm"]
    assert read(root / "final/worker_identity.json")["started_at"] >= selection["selected_at"]
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    original = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {r["sample"]: r["peaks"] for r in read(original)}
    offsets = read(cache / "cache.json")["validation_offsets"]
    full = np.load(root / "final/validation_class_probability.npy")
    assert len(full) == 38543
    assert np.array_equal(full[:, 1:].sum(1), np.load(root / "final/validation_probability.npy"))
    metrics = score(
        {
            i: joint_candidates(peaks[i], full[offsets[n] : offsets[n + 1]])
            for n, i in enumerate(data.validation)
        },
        data,
    )
    assert metrics == result["metrics"] and score(peaks, data) == result["baseline"]
    if chosen["arm"] == "joint100":
        assert np.array_equal(full, np.load(old / "final/validation_class_probability.npy"))
    atomic_json(
        ROOT / "reports/navigation_signal_proposer_fusion_20261021.json",
        dict(
            independent_oof_recount=True,
            independent_component_recount=True,
            raw_control_fold_predictions_exact=True,
            selected_before_final_fit=True,
            selected_arm=chosen["arm"],
            component_pass=result["component_pass"],
            fold_fit_seconds=times,
            assessment_sha256=digest(root / "assessment.json"),
            no_driving_claim=True,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(
        [c["arm"] for c in selection["candidates"]],
        [c["macro_logloss"] for c in selection["candidates"]],
    )
    axes[0].set(title="TRAIN-location OOF selection", ylabel="Macro binary log loss ↓")
    for n, (name, m) in enumerate((("Parent", result["baseline"]), ("Selected", metrics))):
        axes[1].bar(
            np.arange(3) + (n - 0.5) * 0.35,
            [m["classes"][str(c)]["joint"] for c in (1, 2, 3)],
            width=0.35,
            label=name,
        )
    axes[1].set(
        xticks=range(3),
        xticklabels=["Red /36", "Yellow /18", "Green /18"],
        title="Protected state coverage",
        ylim=(0, 36),
    )
    axes[1].legend()
    fig.suptitle(
        "Frozen proposer + RGB: "
        + ("component pass" if result["component_pass"] else "gate failed")
    )
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_proposer_fusion_20261021.png", dpi=150)
    plt.close(fig)
    rows = "\n".join(f"| {r['arm']} | {r['macro_logloss']:.6f} |" for r in selection["candidates"])
    state_rows = "\n".join(
        f"| {name} | {result['baseline']['classes'][str(c)]['joint']} | "
        f"{metrics['classes'][str(c)]['joint']} | {metrics['classes'][str(c)]['labels']} |"
        for c, name in ((1, "Red"), (2, "Yellow"), (3, "Green"))
    )
    outcome = (
        "PASSES the component gate" if result["component_pass"] else "FAILS the component gate"
    )
    report = ROOT / "reports/navigation_signal_proposer_fusion_20261021.md"
    report.write_text(f"""# Frozen-proposer representation fused with RGB

The seven-fit comparison completes in{result["elapsed_seconds"]:.2f}seconds and selects
{chosen["arm"]} using whole-TRAIN-junction OOF loss. The selected model produces
{metrics["localized"]}/72localized lights, {metrics["joint"]}joint correct states and
{metrics["background_peaks"]:,}known-background peaks. It {outcome}.
No native driver deployment or M6 evaluation follows automatically.

| Arm | Macro TRAIN-location OOF logloss ↓ |
|---|---:|
{rows}

![Selection and protected state coverage](navigation_signal_proposer_fusion_20261021.png)

| State | Parent joint | Selected joint | Labels |
|---|---:|---:|---:|
{state_rows}

Same51,094TRAINpoints×3original/gamma versions, original38,543VALproposals,
144VALimages/72housings and all town/state/background protections. Raw87RGB
features versus123features including32frozenCNNchannels and4probabilities.
Classifier recipe,400trees, labels, sample weights, RNG, internal folds and selector
are matched. Only selected final head reaches component VAL. Every model and
prediction is retained; no threshold search or unselected-arm VAL evaluation.

Independent recount reproduces all six OOF vectors, group/class loss, selection
before final fitting, all final component metrics and exact raw-control folds.
All predictions use summed foreground probability>=.5 and RGB-predicted state.
Feature augmentation rows stay aligned within their original TRAIN location.
Actual fit times are in the JSON audit; equal trees do not mean equal CPU work.

## Cache provenance and limits

Original cache attempt failed at zero features because tuple-valued points were
compared directly with JSON arrays. All2988sample records and both index lists
were semantically identical. Corrected full-JSON identity preserves every field
and order; a regression test rejects point/order/split changes. Failed attempt
retained; v2 kept the original absolute1800second deadline. No scientific fit was
repeated and no source image, label, proposal or benchmark changed.

v2extracts in107.03s. Every8676same-batch extractor/originalforward comparison is
exact; model weights frozen; original/augmentedRGBfeatures exactly reproduced.
Newbatch1BF16+FP16cached probabilities differ from older proposal scores by up to
.013758, with zero point-class disagreements. Original proposal identities/scores
are preserved. This is disclosed numerical variation, not old-batch parity.

Separate14.20saudit verifies all hashes/shapes/ranges and67fixedfull-image original
model-hook forwards:21TRAINlocations×3versions plus4VALtowns, all exact. It is
sampled independent recomputation, not recomputation of every point. Fitting
requires matching complete cache/audit receipts. Ninefocusedextractor/audit/fusion/
research-boundary tests pass; source bytes were pinned through each active stage.

The proposer was trained on some of these TRAIN locations and selected on this
component validation before this study. Internal folds test the NEW downstream
head, not unseen end-to-end feature learning. Component validation has been
repeatedly inspected. No final driving/generalization or M4primary claim follows.
Original storage guardian baseline14GBnet/10GBminimum and all failed artifacts
remain. M2/M3/M5 driving gates remain unqualified; M6 has not run.

[Protocol](navigation_signal_proposer_fusion_protocol_20261021.md).
Evidence roots: outputs/navigation_signal_proposer_features_v2_20261021,
outputs/navigation_signal_proposer_audit_20261021,
outputs/navigation_signal_proposer_fusion_20261021.
""")
    print(
        json.dumps(
            dict(
                selected=chosen["arm"],
                metrics=metrics,
                component_pass=result["component_pass"],
                independent_recount=True,
            )
        )
    )


if __name__ == "__main__":
    main()
