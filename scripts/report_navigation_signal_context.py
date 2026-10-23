#!/usr/bin/env python3
"""Report every context checkpoint and its frozen TRAIN/validation diagnosis."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "outputs/navigation_signal_context_20261021"
    history = json.loads((root / "history.json").read_text())
    diagnosis = json.loads(
        (ROOT / "outputs/navigation_signal_context_diagnosis_20261021/assessment.json").read_text()
    )
    assessment = json.loads((root / "assessment.json").read_text())
    assert assessment["selected_step"] is None and not assessment["countermeasure_pass"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [r["metrics"]["joint"] for r in history], "o-")
    axes[0].axhline(44, linestyle="--", color="black", label="Parent44/72")
    axes[0].set_title("Correct point localization + state")
    axes[0].legend()
    axes[1].plot(steps, [r["metrics"]["background_peaks"] for r in history], "o-")
    axes[1].set_title("Known-background peaks /144images")
    for role in ("train", "validation"):
        axes[2].plot(
            steps, [r[role]["balanced_accuracy"] for r in diagnosis["rows"]], "o-", label=role
        )
    axes[2].set_title("Patch/proposal balanced accuracy")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
    fig.suptitle("RGB context verifier: familiar-patch learning, poor location generalization")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/navigation_signal_context_20261021.png", dpi=150)
    plt.close(fig)
    rows = []
    for h, d in zip(history, diagnosis["rows"]):
        m = h["metrics"]
        rows.append(
            f"| {h['step']} | {m['localized']}/72 | {m['joint']}/72 | "
            f"{m['background_peaks']:,} | {d['train']['balanced_accuracy']:.3f} | "
            f"{d['validation']['balanced_accuracy']:.3f} |"
        )
    table = "\n".join(rows)
    (
        ROOT / "reports/navigation_signal_context_20261021.md"
    ).write_text(f"""# RGB context verifier rejected: no protected checkpoint

All1000updates finished in30.13seconds including25.59seconds of cache preparation.
None preserves the original44/72jointlight detections or town/state coverage.
Select no verifier. The latest1000has14localized/12joint and2575backgroundpeaks;
the original proposer remains56localized/44joint/37241backgroundpeaks, unqualified.

| Updates | Localized | Joint | Backgroundpeaks | TRAINbalancedacc | VALbalancedacc |
|---|---:|---:|---:|---:|---:|
{table}

![All scheduled checkpoints and learning gap](navigation_signal_context_20261021.png)

The verifier has126,804parameters and sees only32pixel native plus96pixel context
from the sameRGBimage. It scores all38,543frozen validation proposals. TRAINmining
used202,518proposals:909known-positive,192,283known-background,9,326unknown;
16negatives/image and audited bulb anchors yield15,678cachedpatches
(14,016background/831red/412yellow/419green). Same876TRAIN/144VALimages,72known
validation housings. All seven TRAINlocations are outside current reservations.

Frozen diagnosis shows TRAINbalancedaccuracy82.47% versusVAL40.25%; the TRAIN
sample includes extra known-bulb anchors. A separate predicted-point-only check
gives75.35%TRAINbalancedaccuracy, still aboveVAL. Original-statebinaryrejection
is posthocdiagnostic only: latestkeeps28/72jointwith4135backgroundpeaks, also
losing parentcoverage. No threshold retuning, reselection or fit extension.

Whole-housing proposal state labels can include a dark/inactive pixel rather than
the active bulb; the failed context classifier cannot be taken as proof that
largerRGBcontext is intrinsically useless. Coverage/label diversity and local
phase information remain possible causes. Do not automatically train longer.

Geography audit finds21eligible signaljunctions acrossTown01–04but only7currently
represented in these staticTRAINpairs. AllTown10HDsignaljunctions are reserved
by existing roles; the remaining allowed134/664/841/895arestop/unsignalized.
Retain those reservations and disclose this coverage gap. Next freeze additional
eligible TRAINapproaches/illumination, independently audit captures and avoid
changing the original144validationimages/72labels orTown05finalsuite.

Thirteenfocused crop/unknown/boundary tests pass; all scheduled checkpoint/RNG/
optimizer and probability files retained. No driving or rule-understanding claim.
Protocol: [RGBcontext](navigation_signal_context_protocol_20261021.md).
Evidence: outputs/navigation_signal_context_20261021 and
outputs/navigation_signal_context_diagnosis_20261021. M2/M3/M5unqualified;M6notrun.
""")


if __name__ == "__main__":
    main()
