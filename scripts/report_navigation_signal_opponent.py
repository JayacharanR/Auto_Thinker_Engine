#!/usr/bin/env python3
"""Audit a rejected representation without evaluating the unselected arm on VAL."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(path):
        return json.loads(path.read_text())

    root = ROOT / "outputs/navigation_signal_opponent_20261021"
    prior = ROOT / "outputs/navigation_signal_phase_boost_20261021"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    selection, result = read(root / "selection.json"), read(root / "assessment.json")
    with np.load(cache / "train_features.npz") as a:
        labels = a["labels"]
    with np.load(ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz") as a:
        folds, groups, mask = a["fold"], a["groups"], a["predicted"]
    timings = {}
    for row in selection["candidates"]:
        arm = row["arm"]
        p = np.load(root / f"oof_{arm}.npy")
        timings[arm] = []
        for fold in range(3):
            name = f"{arm}_fold{fold}"
            fit = read(root / name / "fit.json")
            assert fit["trees"] == 400 and fit["excluded_fold"] == fold
            assert not fit["component_validation_read"] and fit["serialization_parity"]
            assert fit["input_features"] == (87 if arm == "joint100" else 312)
            saved = np.load(root / name / "train_class_probability.npy")
            assert np.array_equal(saved[:, 1:].sum(1)[folds == fold], p[folds == fold])
            if arm == "joint100":
                assert np.array_equal(saved, np.load(prior / name / "train_class_probability.npy"))
            timings[arm].append(fit["elapsed_seconds"])
        values = []
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
            assert np.isclose(value, row["by_location"][group], rtol=0, atol=1e-14)
            values.append(value)
        assert np.isclose(np.mean(values), row["macro_logloss"], rtol=0, atol=1e-14)
    selected = min(
        selection["candidates"], key=lambda r: (r["macro_logloss"], r["arm"] != "joint100")
    )
    assert selected == selection["selected"] and result["selected_arm"] == "joint100"
    assert read(root / "final/worker_identity.json")["started_at"] >= selection["selected_at"]
    for file in ("train_class_probability.npy", "validation_class_probability.npy"):
        assert np.array_equal(np.load(root / "final" / file), np.load(prior / "final" / file))
    assert result["metrics"] == read(prior / "assessment.json")["metrics"]
    assert not result["component_pass"]
    report = dict(
        independent_oof_recount=True,
        original_raw_fold_predictions_exact=True,
        final_raw_train_and_validation_exact=True,
        unselected_arm_val_not_evaluated=True,
        selected_arm="joint100",
        fold_seconds=timings,
        component_pass=False,
    )
    (ROOT / "reports/navigation_signal_opponent_20261021.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (
        ROOT / "reports/navigation_signal_opponent_20261021.md"
    ).write_text("""# Relative-color features do not beat the raw-RGB control

All seven fits finish in94.91seconds. Whole TRAIN-junction OOF selection retains
the raw87feature joint classifier: macro binary logloss0.360094 versus0.368144 for
the312feature opponent-color representation. The unselected representation is
NOT evaluated on component VAL. No threshold change or posthoc model substitution.

| Representation | Features | Mean TRAIN-location OOF loss | Selection |
|---|---:|---:|---|
| Original raw RGB |87|0.360094|Retained|
| RGB plus relative color/contrast |312|0.368144|Rejected|

Same augmented data, labels, whole-location folds, sample weights,400trees and
settings. The added225features comprise channel differences, chromaticities and
local contrast. They are deterministic RGB transformations, not metadata inputs.
Raw-fold fitting took5.22/5.25/5.31s; larger representation19.16/21.44/20.94s.
Equal tree count did not mean equal CPU cost. Existing feature caches were reused.

Independent audit reproduces all six OOF vectors and macro losses, verifies
whole-location separation,400tree/feature-count receipts and selection before the
final fit. Every raw-control fold class-probability vector matches the previous
study bit-for-bit. Final raw-control TRAIN and component-VAL vectors also match
exactly, and metrics match their independently recounted predecessor. This is
repeatability evidence, not a new driving result.

The retained model therefore remains45/72localized,44joint,1,171background peaks.
It still FAILS protection: yellow10/18 versus parent15/18; Town10HD9/32versus10/32.
Neither aggregate44joint nor96.86%fewer background detections justifies promotion.
See [state-aware results and graph](navigation_signal_phase_boost_20261021.md).

Ten focused feature/state/group/research-boundary tests passed. Original failed
studies, source archives, caches and denominators remain. Same original storage
baseline14GBnet/10GBminimum; no new native cases, benchmark/reward/interface changes,
M4primary reselection or finalTown05trial. M2/M3/M5 remain unqualified.

Next investigate a compact verifier that retains the frozen proposer CNN's learned
visual features and probability evidence, alongside RGB. The present second stage
discards those features. Register exact extraction, TRAIN-only location selection,
capacity controls and limits before fitting; keep failed representations and all
state/town gates. No claim of improvement until tested.

[Protocol](navigation_signal_opponent_protocol_20261021.md).
Evidence: outputs/navigation_signal_opponent_20261021.
""")
    print("OOF selection and exact raw-control replication verified; unselected VAL untouched.")


if __name__ == "__main__":
    main()
