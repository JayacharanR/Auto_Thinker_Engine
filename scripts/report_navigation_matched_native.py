#!/usr/bin/env python3
"""Report the completed frozen matched M4pilot matrix, including unknown failures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
import numpy as np  # noqa:E402

from src.navigation.dagger import digest
from src.navigation.recorder_collisions import recorder_contacts


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.with_suffix(".json").exists() or args.out.with_suffix(".md").exists():
        raise FileExistsError("Preserve all published comparison results")
    if read(args.root / "progress.json")["status"] != "complete":
        raise ValueError("Publish only the complete original planned matrix")
    request = read(args.root / "request.json")
    source = read(args.root / "assessment.json")
    rows = []
    for entry in source["results"]:
        summary = entry["native_summary"]
        rows.append(
            dict(
                name=entry["candidate"]["name"],
                candidate=entry["candidate"],
                planned=6,
                observed=entry["native_cases_observed"],
                unknown=entry["missing_unknown_failures"],
                arrived=sum(e["completed"] for e in summary["episodes"]),
                strict=summary["safe_completed"],
                rule_safe=summary["rule_safe_completed"],
                contacts=entry["unfiltered_contact_starts"],
                contact_lower_bound=entry["lower_bound"],
                red=sum(e["red_violations"] for e in summary["episodes"]),
                stop=sum(e["stop_violations"] for e in summary["episodes"]),
                native_summary=summary,
                control=entry["candidate"]["name"].endswith("_no_color"),
            )
        )
    original = Path(request["reuse_completed_comma_primary"])
    if not original.is_absolute():
        original = Path.cwd() / original
    old = original / "comma_f100_seed42"
    summary = read(old / "summary.json")
    contacts = [recorder_contacts(p) for p in sorted((old / "recordings").glob("*.log"))]
    old_request = read(original / "request.json")
    candidate = old_request["candidates"][0]
    if digest(candidate["checkpoint"]) != candidate["sha256"]:
        raise ValueError("Original selected comma checkpoint identity changed")
    rows.insert(
        4,
        dict(
            name="comma_f100_seed42",
            candidate=candidate,
            planned=6,
            observed=summary["count"],
            unknown=6 - summary["count"],
            arrived=sum(e["completed"] for e in summary["episodes"]),
            strict=summary["safe_completed"],
            rule_safe=summary["rule_safe_completed"],
            contacts=sum(r["ego_contact_starts"] for r in contacts),
            contact_lower_bound=any(not r["complete"] for r in contacts),
            red=sum(e["red_violations"] for e in summary["episodes"]),
            stop=sum(e["stop_violations"] for e in summary["episodes"]),
            native_summary=summary,
            control=False,
        ),
    )
    if len(rows) != 6 or any(r["observed"] + r["unknown"] != 6 for r in rows):
        raise ValueError("All five matched heads and separate control keep six planned cases")
    for row in rows:
        if digest(row["candidate"]["checkpoint"]) != row["candidate"]["sha256"]:
            raise ValueError("A registered checkpoint changed after execution")
    original_sources = read(old / "source_hashes.json")
    differences = [
        p
        for p, h in request["inference_source_sha256"].items()
        if p in original_sources and original_sources[p] != h
    ]
    report = dict(
        rows=rows,
        main_planned=30,
        main_native_observed=sum(r["observed"] for r in rows if not r["control"]),
        main_unknown_failures=sum(r["unknown"] for r in rows if not r["control"]),
        primary="comma_f100_seed42",
        primary_reselected=False,
        no_driver_promoted=True,
        matched_request_sha256=digest(args.root / "request.json"),
        matched_assessment_sha256=digest(args.root / "assessment.json"),
        original_primary_summary_sha256=digest(old / "summary.json"),
        original_primary_recorders=contacts,
        original_primary_source_differences=differences,
        source_difference_explanation="Three recording/final-registration modules differ "
        "across the "
        "approved compact-storage activation. All nonfinal videos remain full; runtime free-space "
        "floor25→10GB. Policy/physics/scoring equations unchanged. "
        "Later feedback patch applied only "
        "AFTER every matrix drive; none used own-actuatorv2.",
        reporting_source_sha256=digest(__file__),
        note="One training seed/checkpoint, six familiar TRAINengineering cases. Paired data "
        "but highly correlated geography; no statistical encoder superiority or M6qualification.",
    )
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    labels = {
        "cnn_f100_seed42": "Scratch CNN",
        "resnet_f100_seed42": "ImageNet ResNet18",
        "dino_f100_seed42": "DINOv2-S",
        "carla_f100_seed42": "CARLA JEPA",
        "comma_f100_seed42": "Comma JEPA (primary)",
        "comma_f100_seed42_no_color": "Comma JEPA, no-color control",
    }
    lines = [
        "# Corrected encoder study: matched native pilot results",
        "",
        "The frozen full-label seed42heads have all completed their original bounded pilots. "
        "Keep all30main planned cases:27have native outcomes and3remain unknown failures. "
        "The original comma primary is reused, not rerun or reselected. "
        "A separate six-case no-color control follows the five-encoder comparison.",
        "",
        "| Encoder | Arrived /6 | Rule-safe /6 | Strict /6 | All contacts "
        "| Red / stop | Unknown failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['name']]} | {row['arrived']} | {row['rule_safe']} | {row['strict']} "
            f"| {row['contacts']} | {row['red']} / {row['stop']} | {row['unknown']} |"
        )
    lines += [
        "",
        "No model passes all six cases safely. ResNet6/6arrivals/0contacts is useful "
        "engineering evidence but still includes one red and one stop-sign violation. "
        "The original comma primary completes4/6with10unfilteredcontactstarts,2red/1stop. "
        "The no-color control completes2/6with0observedcontacts and1red; it does not "
        "establish an improvement over the primary or justify replacing it.",
        "",
        "CNNhit the original900s wall cap with4native outcomes and2unexecutedcases; "
        "DINOhas5native outcomes and1unexecutedcase. These stay failures in the denominator. "
        "Observed zero contacts never certifies an unexecuted case safe. No case retry, "
        "replacement or extra wall budget. The earlier cold-start/fact M2variants remain "
        "separate experiments and are excluded from this matched table.",
        "",
        f"![Matched native outcomes]({args.out.name}.png)",
        "",
        "The label-efficiency study uses55TRAINepisodes/15712labels, the unchanged13VAL/2822 "
        "timestamps and matched downstream update budgets. Comma has the best mean "
        "full-label offline score(.3755) but does not yield the best native result. "
        "This selection/deployment gap is a result to report, not a license to reselect "
        "from evaluation outcomes. Differing external pretraining histories remain disclosed.",
        "",
        "All pilots used the original v1policy boundary. The user-approved compact "
        "recording/storage activation occurred between the original comma run and remaining "
        "heads: final-only recording gates leave these nonfinal videos full, and disk "
        "minimum changed25→10GB. Policy/physics/scoring equations remain the same; three "
        "archived recording/registration module identities differ. The actuator-v2extension "
        "was applied only after the entire matrix finished.",
        "",
        "One checkpoint seed and six familiar correlated TRAINcases cannot establish "
        "population safety or significant encoder superiority. All full raw attempts, "
        "contact recorders, failed videos, source archives and selectors remain retained. "
        "The300encounter camera gate, robust rule behavior and M5planner evidence remain open.",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines))
    x = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(x - 0.18, [r["arrived"] for r in rows], 0.36, label="Arrived", color="#539ac4")
    axes[0].bar(x + 0.18, [r["rule_safe"] for r in rows], 0.36, label="Rule-safe", color="#267b50")
    axes[0].set(ylim=(0, 6.8), ylabel="Missions /6", title="Fixed six-case TRAINpilot")
    axes[1].bar(x, [r["contacts"] for r in rows], color="#b74b4b")
    axes[1].set(ylabel="Unfiltered contact starts", title="Every observed contact retained")
    for ax in axes:
        ax.set_xticks(
            x, ["CNN", "ResNet", "DINO", "CARLA\nJEPA", "Comma\nJEPA", "No-color\ncontrol"]
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in [
                    "main_planned",
                    "main_native_observed",
                    "main_unknown_failures",
                    "no_driver_promoted",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
