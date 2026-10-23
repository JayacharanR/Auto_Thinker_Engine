#!/usr/bin/env python3
"""Summarize all200capture outcomes and all fixed data-countermeasure checkpoints."""

import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    collection = ROOT / "outputs/navigation_signal_diversity_20261021"
    audit = ROOT / "outputs/navigation_signal_diversity_audit_20261021"
    fit = ROOT / "outputs/navigation_signal_diverse_fit_20261021"
    observed = json.loads((collection / "assessment.json").read_text())
    verified = json.loads((audit / "assessment.json").read_text())
    result = json.loads((fit / "assessment.json").read_text())
    history = json.loads((fit / "history.json").read_text())
    assert observed["closed_case_receipts"] == verified["verified_cases"] == 164
    assert verified["unknown_cases"] == 36 and not verified["quarantined_cases"]
    assert result["selected_step"] == 0 and not result["countermeasure_pass"]
    old = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb"
    old0 = torch.load(old / "step0.pt", map_location="cpu", weights_only=False)
    new0 = torch.load(fit / "step0.pt", map_location="cpu", weights_only=False)
    assert old0["model"].keys() == new0["model"].keys()
    assert all(torch.equal(v, new0["model"][k]) for k, v in old0["model"].items())
    assert json.loads((old / "predictions0.json").read_text()) == json.loads(
        (fit / "predictions0.json").read_text()
    )
    # Independent one-to-one recount using the unchanged original validation boxes.
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.train_navigation_signal_context import score
    from src.navigation.signal_pixel_study import NativeSignalDataset

    data = NativeSignalDataset(ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021")
    for row in history:
        predictions = json.loads((fit / f"predictions{row['step']}.json").read_text())
        recount = score({r["sample"]: r["peaks"] for r in predictions}, data)
        assert all(
            recount[k] == row["metrics"][k]
            for k in ("labels", "localized", "joint", "background_peaks", "towns", "classes")
        )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    steps = [r["step"] for r in history]
    for source, label, style in (
        (history, "21location TRAIN", "o-"),
        (json.loads((old / "history.json").read_text()), "7location TRAIN", "s--"),
    ):
        for ax, key in zip(axes, ("selection_loss", "joint", "background_peaks")):
            ax.plot(steps, [r["metrics"][key] for r in source], style, label=label)
    for ax, title in zip(
        axes,
        (
            "Common loss ↓",
            "Correct point localization + state /72 ↑",
            "Known-background peaks /144images ↓",
        ),
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("More TRAIN diversity, same recipe: no protected improvement")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_diversity_20261021.png", dpi=150)
    plt.close(fig)
    rows = []
    for row in history:
        m = row["metrics"]
        rows.append(
            f"| {row['step']} | {m['selection_loss']:.4f} | {m['localized']}/72 | "
            f"{m['joint']}/72 | {m['background_peaks']:,} | "
            f"{100 * m['background_active_fraction']:.3f}% |"
        )
    table = "\n".join(rows)
    (
        ROOT / "reports/navigation_signal_diversity_20261021.md"
    ).write_text(f"""# More signal data audited; the unchanged training recipe still fails

The new164verified phase cases broaden TRAINcoverage from7to21signaljunctions.
The fixed1000update model does not pass: common loss selects initialization0.
Latest1000localizes61/72known lights,43withcorrectstate, but produces88,379
known-background peaks. Originalselectedbulb1000was56/44/37,241. Keep both failed
studies; no trained-model promotion, extension or driving rollout.

| Updates | Common loss ↓ | Localized | Joint state | Backgroundpeaks | Active background |
|---|---:|---:|---:|---:|---:|
{table}

![Same-recipe data comparison](navigation_signal_diversity_20261021.png)

## Native collection and independent audit

Frozen100poses/21junctions/36approaches inTown01–04, twoilluminationconditions,
200plannedcases/2400images. Eightisolatednativeworkers settled in319.84seconds.
FirstTown01clear-noon fails immediate nativeweather-statecheck beforeimages;
36cases remain UNKNOWN, no retry or replacement. Exact mismatched weather values
were not saved; cause is open. Other seven children exit0, each engineexit143;
first engine exits−11. The original owner's abnormal-exit counter8 is preserved
but is not eightSIGSEGVs or a proof of stable shutdown. Collection remainsFAILED.

After exactownerexit, separate7.08secondaudit verifies164cases/1968images:
originalsource/plan/RGBhashes, physicalsignal identity, independent camera-box
reprojection, phase/backgroundlamp states, orderedframes, actualstaticpose/rig
tolerances and≥1mreserved-geographyclearance. No additionalquarantine;36unknown.
All21locations represented;100overcast/64clear cases. Supported active-pixel
targets221red/green and219yellow, allnonempty. Empty/unsupported housings stay
unknown. Acceptedmanifest SHA:
f400408ae0ce7f53bbf16326592742b4df0ccdc4e79c3f16cc864cb3a7a8bdac.

Town10HD has no eligible TRAINsignaljunction under currentreservations. No
borrowing reservedlocations. The two lighting distributions are imbalanced by
the failed36cases. Staticphysics-offpairs omit originaltraffic and prove no
driving behavior or applicability. Originalfailure/source/nativeartifacts retained.

## Matched recipe and selection

2844TRAINimages versus prior876; same144VALimages/72labels/ignoreregions
byte-identical. Samefreshseed42/native21pixelRF37772parameterCNN/AdamW.001/
1000updates/class-balancedcrops/foregroundCE+.2meanbackgroundCE. Noarchitecture,
loss, threshold, encoder or state-feedback change. Newdatasetphasecopies:
626red/309yellow/313green, including repeatedred as originallydefined.
Execution22.49seconds, fitting/evaluation6.38seconds. Everycheckpoint/RNG/
optimizer/prediction retained. Initialmodeltensors andvalidationpredictions exactly
match priorseed42initialization; independentclosed-predictionrecount reproduces
allfivecheckpoints'localized/joint/background/town/statecounts.

Latest foreground loss2.661 versus old1.478 and backgroundCE.1391versus.07156.
Thus newdata alone does not resolve labelconfidence/backgroundconfusion under
thisbudget. Morelocalizedpoints are not a safety improvement. Town10joint9versus
old10; all other townjointcounts7/15/12unchanged. Original commonloss selection
is respected; no posthoclatest promotion. Componentvalidation was alreadyinspected,
not an untouchedtest. Pointmetric isNOTbounding-boxAP.

Next inspect the missedstates/falsealarmrepresentation and use a separately
registered semantic foreground verifier or architecture change. Existing pretrained
RGBbackbones may offer a different prior from the failedscratchcontextCNN; this
is a hypothesis, not a result. Keepallpreviousfailures and baselinebudgets explicit.
M2/M3/M5unqualified;M6notrun. No benchmark/reward/allowedinput/M4primary change.

Protocols: [capture](navigation_signal_diversity_protocol_20261021.md),
[fit](navigation_signal_diverse_fit_protocol_20261021.md). Evidence roots:
outputs/navigation_signal_diversity_20261021,
outputs/navigation_signal_diversity_audit_20261021,
outputs/navigation_signal_diverse_fit_20261021.
""")


if __name__ == "__main__":
    main()
