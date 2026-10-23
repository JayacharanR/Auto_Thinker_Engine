#!/usr/bin/env python3
"""Report the completed, rejected first-event study without changing selection."""

import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    study = ROOT / "outputs/navigation_first_event_study_20261021"
    audit = ROOT / "outputs/navigation_first_event_assessment_20261021"
    features = ROOT / "outputs/navigation_intervention_features_20261021"
    assessment = read(audit / "assessment.json")
    members, rows = read(study / "members.json"), read(features / "records.json")
    report = ROOT / "reports/navigation_first_event_study_20261021"
    names = ["Constant", "Action only", "Calibrated CV", "JEPA ensemble", "RSSM ensemble"]
    scores = [
        assessment["baselines"]["constant"],
        assessment["baselines"]["action_only"],
        assessment["baselines"]["cv"],
        assessment["primary_metrics"]["jepa"]["calibration"],
        assessment["primary_metrics"]["rssm"]["calibration"],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    ax = axes[0, 0]
    values = [s["endpoint_brier"] for s in scores]
    ax.barh(names, values, color=["#6d7785", "#438f72", "#6d7785", "#407cad", "#a26898"])
    ax.invert_yaxis()
    ax.set(
        xlabel="Four-second Brier score (lower is better)",
        title="18 calibration options; four contacts, two locations",
    )
    for i, value in enumerate(values):
        ax.text(value + 0.002, i, f"{value:.4f}", va="center")
    ax.set_xlim(0, 0.215)
    for ax, kind in zip((axes[0, 1], axes[1, 0]), ("jepa", "rssm")):
        for member in members:
            if member["kind"] != kind:
                continue
            steps = [r["step"] for r in member["history"]]
            train = [r["train"]["sequence_nll"] for r in member["history"]]
            val = [r["calibration"]["sequence_nll"] for r in member["history"]]
            (line,) = ax.plot(steps, val, marker="o", label=f"seed {member['seed']}")
            ax.plot(steps, train, linestyle="--", color=line.get_color(), alpha=0.7)
        ax.axhline(
            assessment["baselines"]["constant"]["sequence_nll"], color="black", linestyle=":"
        )
        ax.set(
            title=f"{kind.upper()}: calibration solid / TRAIN dashed",
            xlabel="Optimizer updates",
            ylabel="Sequence NLL (lower is better)",
        )
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    indices = [i for i, r in enumerate(rows) if r["source_role"] == "calibration"]
    truth = np.array([rows[i]["positive_first_event"] for i in indices])
    for kind, marker in (("jepa", "o"), ("rssm", "x")):
        predicted = np.load(study / f"{kind}_ensemble.npz")["cumulative"][indices, -1]
        ax.scatter(np.arange(18), predicted, marker=marker, label=kind.upper())
    ax.scatter(
        np.flatnonzero(truth),
        np.ones(int(truth.sum())),
        color="#c74440",
        marker="v",
        label="Observed contact",
    )
    ax.axhline(0.5, color="black", linestyle=":", label="Fixed 0.5 threshold")
    ax.set(
        ylim=(-0.03, 1.06),
        xlabel="Original calibration option order",
        ylabel="Predicted contact probability",
        title="Both ensembles miss all four contacts",
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "First-event countermeasure rejected — no camera driving or M5 qualification", fontsize=14
    )
    fig.savefig(report.with_suffix(".png"), dpi=170)
    plt.close(fig)
    all15 = np.load(audit / "all15_predictions.npz")
    selected = all15["selected_indices"]
    parity = []
    for i, member in enumerate(members):
        logits = all15["logits"][i, np.arange(68), selected].astype(np.float64)
        new = -np.expm1(np.cumsum(-np.logaddexp(0, logits), axis=-1))
        old = np.load(Path(member["output"]) / "selected_predictions.npz")["cumulative"]
        parity.append(
            dict(
                kind=member["kind"],
                seed=member["seed"],
                maximum_cumulative_difference=float(abs(new - old).max()),
            )
        )
    record = dict(
        **assessment,
        selected_members=[
            dict(
                kind=m["kind"],
                seed=m["seed"],
                selected_step=m["selected_step"],
                trainable_parameters=m["trainable_parameters"],
                elapsed_seconds=m["elapsed_seconds"],
            )
            for m in members
        ],
        all15_batch_shape_parity=parity,
    )
    report.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    table = "\n".join(
        f"| {name} | {score['endpoint_brier']:.5f} | "
        + (f"{score['sequence_nll']:.4f}" if "sequence_nll" in score else "not a survival model")
        + f" | {score['misses']}/4 | {score['false_alarms']}/14 |"
        for name, score in zip(names, scores)
    )
    text = f"""# First-event models: completed countermeasure, both ensembles rejected

The new JEPA and RSSM ensembles do not beat the simple baselines on endpoint
prediction, and both miss all four calibration contacts at the frozen0.5threshold.
No native planner was deployed. M2/M3/M5 remain unqualified; M6 has not run.

| Model | Endpoint Brier ↓ | Sequence NLL ↓ | Misses | False alarms |
|---|---:|---:|---:|---:|
{table}

The raw CV proximity score flags all18cases:0misses/14false alarms. It is not a
probability. Its monotone sigmoid calibration was fitted on TRAIN only and frozen
before calibration targets were opened. The action-only model also uses TRAIN
only, with no visual/world/route identity. Lower sequenceNLL does not override
worse endpointBrier or justify a safety claim.

![Complete comparison and all scheduled learning curves](navigation_first_event_study_20261021.png)

Data:73original attempts,68 independently verified recordings,5unknown failures.
TRAIN50episodes/eightlocations/12contacts; calibration18/twolocations/4contacts.
All10,880native controls and5,440RGBtimestamps in eligible sources were audited.
39expansion prefixes match exactly after a separate static-light identity
correction; original failed audit and its actor-order error remain retained.
Scripted actors are nonreactive; all lights are green. No red-cost model is trained.

Both learned arms receive the same complete eight-timestamp RGB-derived object
history, current48x384JEPAgrid, declared navigation/ego state, two actual prior ego
commands, and prospective80-command plans. CV receives the same object history
and own-action geometry. No simulator traffic or teacher state enters inference.
The futureRGBtargets remain separate, censored at contact/intervention; missing
4.0sRGBis unknown. Every accepted detector box, including unlocated boxes, is
retained. Shared features completed in197.34s without model updates.

Six independent fits,400updates each, same50examples/uniformbatch12draws, fixed
optimizer and selection. The first-contact likelihood sums each observed prefix;
it fixes the variable-duration weighting in the earlier untrained prototype.
JEPA has453,613parameters andRSSM622,701; equal updates are not equal parameter
count or measured hardware compute. Training took58.90s total including loading.
Each arm selected steps100/0/0 for seeds42/1051/2060. Later training likelihood
improves while calibration worsens. No losing fit is extended or reselected.

All15prospective candidates were subsequently predicted by all six frozen members.
Only the originally executed options have observed outcomes; alternative outcomes
remain unknown. BF16different-batch inference differs by up to0.0625logit; maximum
cumulative-probability difference is{max(p["maximum_cumulative_difference"] for p in parity):.6f}.
Original selected predictions remain the primary metrics. Member disagreement
is not calibrated uncertainty, and two calibration locations do not support a
general reliability claim. The frozen backbone's historical exposure was already
disclosed; this is not a new clean encoder comparison or a change to M4's primary.

Next: diagnose ground geometry/object observability and address the independent
traffic-light localization/stop-release failures before another driving pilot.
Keep the rejected models, every checkpoint, RNG/optimizer state, original failed
attempts and frozen evaluation suite. No Town05 tuning or final driving.

Evidence: outputs/navigation_interventions_verified_20261021,
outputs/navigation_intervention_features_20261021,
outputs/navigation_first_event_baselines_20261021,
outputs/navigation_first_event_study_20261021 and
outputs/navigation_first_event_assessment_20261021. Protocol:
[registered study](navigation_first_event_study_protocol_20261021.md).
"""
    report.with_suffix(".md").write_text(text)
    shutil.copyfile(__file__, audit / "report_source.py")
    print(report.with_suffix(".md"))


if __name__ == "__main__":
    main()
