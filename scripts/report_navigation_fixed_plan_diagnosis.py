#!/usr/bin/env python3
"""Publish all six frozen-plan diagnostics with their correlated-scenario limits."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
import numpy as np  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.with_suffix(".md").exists() or args.out.with_suffix(".png").exists():
        raise FileExistsError("Preserve every published diagnostic")
    assessment = json.loads((args.root / "assessment.json").read_text())
    predictions = json.loads((args.root / "predictions.json").read_text())
    if len(predictions) != 6:
        raise ValueError("All six registered rollouts are required")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4), sharey=True)
    colors = dict(cv="#8c54b5", jepa="#2376af", rssm="#267b50")
    times = np.arange(0.5, 4.01, 0.5)
    lines = [
        "# Frozen-plan collision diagnostic",
        "",
        "Both repeats produce the same contact pattern: go hits at1.30s, "
        "brake has no contact over4s, and wait-go hits at3.80s. All4unfiltered "
        "contact starts are counted. Independent replay/control audits pass.",
        "",
        "The original three-member JEPA and RSSM exports both miss all14positive "
        "horizons at the fixed.5diagnostic threshold. CVmisses none, but exceeds.5on "
        "all34negative horizons. CVproximity and balanced neural sigmoid scores are "
        "uncalibrated; neither output supports a safety probability claim.",
        "",
        "| Executed option | Observed contact | CV4s score | JEPA4s score | RSSM4s score |",
        "|---|---|---:|---:|---:|",
    ]
    for ax, option, title in zip(
        axes,
        ("go", "brake", "wait_go"),
        ("Go · contact at1.30s", "Brake · no contact", "Wait-go · contact at3.80s"),
    ):
        subset = [r for r in predictions if r["episode"].endswith("_" + option)]
        # endswith('_go') also matches wait_go: use the exact suffix after repeatN_.
        subset = [r for r in subset if r["episode"].split("_", 1)[1] == option]
        if len(subset) != 2:
            raise ValueError("Both original repeats required for each option")
        means = {}
        for name in colors:
            values = np.array(
                [
                    r["predictions"][name]["collision_mean"][r["executed_candidate_index"]]
                    for r in subset
                ]
            )
            means[name] = values.mean(0)
            ax.plot(times, means[name], label=name.upper(), color=colors[name], linewidth=2)
            ax.fill_between(times, values.min(0), values.max(0), color=colors[name], alpha=0.15)
        if option != "brake":
            ax.axvline(
                1.3 if option == "go" else 3.8, color="#bc303a", linestyle="--", linewidth=1.5
            )
        ax.set(title=title, xlabel="Seconds after decision", ylim=(-0.04, 1.04), xlim=(0.5, 4))
        ax.grid(alpha=0.2)
        contact_label = (
            "None in4s" if option == "brake" else ("1.30s" if option == "go" else "3.80s")
        )
        lines.append(
            f"| {option} | {contact_label} "
            f"| {means['cv'][-1]:.4f} | {means['jepa'][-1]:.4f} | {means['rssm'][-1]:.7f} |"
        )
    axes[0].set_ylabel("Uncalibrated risk / proximity score")
    axes[-1].legend(loc="center right", frameon=False)
    fig.suptitle(
        "Frozen models vs observed alternative actions · one TRAINscenario, two repeats",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    lines += [
        "",
        "JEPAreverses the observed braking/go risk ordering. The shared frozen-ego "
        "collision-only cost selects center-path stop for CVin all6contexts, while both "
        "learned models select left-offset/faster. That latter option was not executed, "
        "so its actual outcome is unknown; no fabricated full-grid ranking result.",
        "",
        f"![Frozen-plan scores]({args.out.name}.png)",
        "",
        "These are post-hoc engineering diagnostics, not final M5qualification: one "
        "stationary TRAINlead, three observed options, two correlated repeats, CPUFP32 "
        "rather than the original GPUBF16forward, and different CV/learned perception "
        "pipelines. Red costs remain unsupported. No update, tuning or promotion.",
        "",
        "A subsequent source audit finds157positive training/validation horizons in only "
        "4episodes: two noisy and two expert. Two of3TRAINcollision episodes are rear "
        "impacts with near-stationary ego. All144positive horizons with an input before "
        "the first event include braking;136also include controls after that event. "
        "This supports a distribution/causal-model concern but does not prove the cause "
        "of the inverted predictions. Keep rear impacts in every headline count.",
        "",
        "Next: collect diverse action-intervention outcomes at strictly separate TRAIN "
        "locations and stage20Hzfirst-event prediction. Mask losses after first contact "
        "or privileged intervention so reaction controls cannot inform a scored earlier "
        "event. Preserve existing models/targets. No new fit until data/splits, matched "
        "baselines and calibration protocol are registered.",
        "",
        f"Raw evidence: `{args.root}/assessment.json`, `predictions.json`; "
        "`outputs/navigation_action_confound_audit_20261020/assessment.json`.",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(assessment["totals"]))


if __name__ == "__main__":
    main()
