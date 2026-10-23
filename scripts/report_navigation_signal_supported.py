#!/usr/bin/env python3
"""Independently recount saved component probabilities and report all stages."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_signal_supported_full import add_counts, count_image  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa: E402


def main():
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    sampled = ROOT / "outputs/navigation_signal_supported_20261022"
    full = ROOT / "outputs/navigation_signal_supported_full_20261022"
    final = ROOT / "outputs/navigation_signal_supported_final_20261022"
    destination = ROOT / "reports/navigation_signal_supported_20261022"
    if destination.with_suffix(".json").exists():
        raise FileExistsError("Retain the original report")
    a, b, c = [read(p / "assessment.json") for p in (sampled, full, final)]
    if any(read(p / "progress.json")["status"] != "complete" for p in (sampled, full, final)):
        raise ValueError("All three source owners must have completed")
    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    source = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
    peaks = {r["sample"]: r["peaks"] for r in read(source)}
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021/cache.json"
    offsets = read(cache)["validation_offsets"]
    probabilities = np.load(final / "final/validation_class_probability.npy")
    total = dict(towns={})
    for n, i in enumerate(data.validation):
        counted = count_image(
            peaks[i], probabilities[offsets[n] : offsets[n + 1]], data.boxes[i], data.background[i]
        )
        add_counts(total, counted)
        add_counts(total["towns"].setdefault(data.samples[i]["town"], {}), counted)
    for key in ("labels", "localized", "joint", "background_peaks", "unknown_peaks"):
        if total[key] != c["metrics"][key]:
            raise ValueError("Independent component recount disagrees")
    for field in ("towns", "classes"):
        for key, result in c["metrics"][field].items():
            if any(total[field][key][m] != result[m] for m in ("labels", "localized", "joint")):
                raise ValueError("Independent subgroup recount disagrees")
    requirements = dict(
        aggregate=total["joint"] >= c["baseline"]["joint"],
        towns=all(
            total["towns"][k]["joint"] >= v["joint"] for k, v in c["baseline"]["towns"].items()
        ),
        phases=all(
            total["classes"][k]["joint"] >= v["joint"] for k, v in c["baseline"]["classes"].items()
        ),
        background=total["background_peaks"] <= 0.1 * c["baseline"]["background_peaks"],
    )
    if all(requirements.values()) != c["component_pass"]:
        raise ValueError("Independent component gate disagrees")
    result = dict(
        sampled=a,
        full_train=b,
        component=c,
        independent_component_recount_exact=True,
        independently_recounted=total,
        requirements=requirements,
        report_source_sha256=digest(__file__),
        sources={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                sampled / "assessment.json",
                full / "assessment.json",
                final / "assessment.json",
                final / "final/model.pkl",
                final / "final/validation_class_probability.npy",
                source,
                cache,
            )
        },
    )
    atomic_json(destination.with_suffix(".json"), result)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(["Control", "Supported"], [1141, 1144], color=["#627d98", "#19927a"])
    axes[0].set_ylim(0, 1248)
    axes[0].set_title("TRAIN OOF correct housing + phase")
    for x, y in enumerate((1141, 1144)):
        axes[0].text(x, y + 15, f"{y}/1248", ha="center")
    axes[1].bar(["Control", "Supported"], [30758, 724], color=["#627d98", "#19927a"])
    axes[1].set_yscale("log")
    axes[1].set_title("All TRAIN background detections (log)")
    for x, y in enumerate((30758, 724)):
        axes[1].text(x, y * 1.08, f"{y:,}", ha="center")
    positions = np.arange(3)
    axes[2].bar(positions - 0.2, [22, 7, 15], width=0.4, label="Original parent", color="#627d98")
    axes[2].bar(positions + 0.2, [22, 11, 10], width=0.4, label="Supported", color="#19927a")
    axes[2].set_xticks(positions, ["Red /36", "Green /18", "Yellow /18"])
    axes[2].set_title("Component VAL: yellow gate FAILS")
    axes[2].legend(fontsize=8)
    fig.suptitle("Supported local supervision: fewer background detections, still unqualified")
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=140)
    plt.close(fig)
    destination.with_suffix(".md").write_text(f"""# Supported local signal supervision

The change substantially reduces background detections, but **fails the unchanged
component gate**. It is retained as an experiment and is not deployed.

| Evidence | Control | Supported supervision |
|---|---:|---:|
| Sampled TRAIN correct housing + phase /1,248 | 1,141 | 1,144 |
| Sampled TRAIN accepted background proposals | 1,990 | 201 |
| All TRAIN correct housing + phase /1,248 | 1,141 | 1,144 |
| All TRAIN background detections | 30,758 | 724 |
| All TRAIN unmatched unknown detections | 4,524 | 2,391 |
| TRAIN macro point logloss (lower is better) | 0.360094 | 1.264219 |
| Component correct housing + phase /72 | 44 | 43 |
| Component background detections | 37,241 | 40 |
| Component unmatched unknown detections | 1,246 | 98 |

TRAIN control is the previous RGB joint100 head, evaluated out of fold. Component
control is the original native bulb proposer used by the immutable protection gate.
They are different controls. Relative to the previous RGB joint100 component result
(44joint/1,171background), the new model has43joint/40background.

The full TRAIN audit includes all715,825original proposals across2,844images; it is
not the sampled-background result. Every48,865original sampled row reproduces its
RGB features and both OOF probability vectors exactly. Independent sampled metric
recounts also match. All original TRAIN housing labels and excluded positives remain
in scoring. GT-only anchors never contribute detections.

| Component subgroup | Original correct | Supported correct | Labels |
|---|---:|---:|---:|
| Red | 22 | 22 | 36 |
| Green | 7 | 11 | 18 |
| Yellow | 15 | 10 | 18 |
| Town01 | 7 | 7 | 8 |
| Town03 | 15 | 16 | 20 |
| Town04 | 12 | 12 | 12 |
| Town10HD | 10 | 8 | 32 |

![TRAIN and component results](navigation_signal_supported_20261022.png)

**Method.** The paired-phase TRAIN audit showed that housing membership frequently
assigned phase to points whose detailed5×5crop contained no audited active bulb.
Exclude2,006suchpositivepoints from the fitting loss (unknown, never background),
keep45,494backgroundrows and2,229verifiedGT-onlyanchors. Fit49,088retainedrows and
two originalgamma variants:147,264examples. Same87RGBfeatures, joint100classifier,
400trees,seed42,weights and hyperparameters. Three whole-junction fold fits select
continuation by a predeclared housing-level TRAIN criterion; full-proposal audit
confirms it. One final fit then assesses original144componentimages/72labels once.
Point logloss worsens because excluded local labels remain in scoring; it is not
hidden or redefined after results.

**Timing.** Three-fold stage{a["elapsed_seconds"]:.2f}s; all-proposal audit
{b["elapsed_seconds"]:.2f}s; final fit/evaluation{c["elapsed_seconds"]:.2f}s.
Final classifier fit7.05s. Ten focused tests pass; serialized model predictions are
exact. A separate score-order matcher independently reproduces all component counts,
towns, phases and gate failures from retained probabilities, without refitting.
Final model SHA256: `{c["model_sha256"]}`.

**Limits and next action.** These are correlated component images and points, not
driving success rates. The upstream proposer has TRAIN exposure, while only the
new verifier is cross-validated by location. The component set has been repeatedly
inspected and is development evidence. The99.89%background reduction does not offset
missing yellow lights or qualify safety. No threshold relaxation, benchmark edits,
runtime integration, native driving, or M6 evaluation occurred.

Next investigate TRAIN-supported bulb appearance/phase generalization and the
remaining signal-recall limitation before a prospectively registered correction.
Preserve Town10HD reserved locations and the original yellow/town protection gates;
do not mine them as training data or relax thresholds to rescue this result.
Original M4 primary remains unchanged. M2/M3/M5 remain unqualified; M6 not run.

Protocols: [fold comparison](navigation_signal_supported_protocol_20261022.md),
[full TRAIN audit](navigation_signal_supported_full_protocol_20261022.md),
[conditional component evaluation](navigation_signal_supported_final_protocol_20261022.md).
Evidence and hashes are retained in the accompanying JSON and all three output roots.
""")
    print(
        json.dumps(
            dict(
                independent_component_recount_exact=True,
                requirements=requirements,
                component_pass=c["component_pass"],
                joint=total["joint"],
                background=total["background_peaks"],
            )
        )
    )


if __name__ == "__main__":
    main()
