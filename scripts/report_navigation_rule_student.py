#!/usr/bin/env python3
"""Summarize a settled M3 assessment without changing any outcome or scoring rule."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa:E402
import numpy as np  # noqa:E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Report prefix")
    args = parser.parse_args()
    if any(args.out.with_suffix(suffix).exists() for suffix in (".md", ".json", ".png")):
        raise FileExistsError("Preserve previous rule reports")
    source = json.loads(args.assessment.read_text())
    rows = source["rows"]
    if source["status"] != "complete" or len(rows) != source["planned_trials"]:
        raise ValueError("Report only the complete fixed-denominator assessment")
    groups, pairs, family = defaultdict(list), defaultdict(dict), defaultdict(list)
    for row in rows:
        groups[(row["family"], row["location"], row["arm"])].append(row)
        key = (row["family"], row["location"], row["seed"])
        if row["arm"] in pairs[key]:
            raise ValueError("Duplicate scenario arm/seed")
        pairs[key][row["arm"]] = bool(row["passed"])
        family[row["family"]].append(row)
    if any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("Both unchanged scenario arms are required in every pair")
    paired_passes = sum(all(pair.values()) for pair in pairs.values())
    contact_trials = sum(r["behavior"]["all_attempt_contact_starts"] > 0 for r in rows)
    report = dict(
        assessment=str(args.assessment),
        assessment_sha256=digest(args.assessment),
        reporting_source_sha256=digest(__file__),
        planned_trials=source["planned_trials"],
        evidence_complete=source["evidence_complete_trials"],
        passed=source["passed_trials"],
        failed=source["failed_or_unverified_trials"],
        gate_passed=source["gate_passed"],
        unfiltered_contact_starts=source["unfiltered_all_attempt_contact_starts"],
        contact_trials=contact_trials,
        native_red_violations=sum(r["behavior"]["native_red"] for r in rows),
        native_stop_violations=sum(r["behavior"]["native_stop"] for r in rows),
        both_arms_passed=paired_passes,
        physical_pairs_times_seeds=len(pairs),
        family_counts=source["family_counts"],
        groups=[
            dict(
                family=f,
                location=loc,
                arm=arm,
                passed=sum(r["passed"] for r in group),
                planned=len(group),
            )
            for (f, loc, arm), group in sorted(groups.items())
        ],
        rule_overrides=False,
        source_checkpoint_sha256=source["checkpoint_sha256"],
        note="Scripted setup component tests, not autonomous route success. Repeated locations and "
        "seeds are correlated; no independent-binomial confidence claim. No checkpoint promotion.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Frozen camera primary: rule counterfactual results",
        "",
        "The original CNN-facts primary passes **15/72** override-disabled component trials. "
        "All72 have complete evidence; **22 unfiltered contact starts across9 trials**, "
        "15native red-light violations and5native stop-sign violations are retained. "
        "The independent actuation audit verifies72recorders, zero control differences and "
        "zero missing acknowledgments. The model fails the M3gate and is not promoted.",
        "",
        "| Rule family | Passed / planned |",
        "|---|---:|",
    ]
    for name, counts in sorted(source["family_counts"].items()):
        lines.append(f"| {name} | {counts['passed']} / {counts['planned']} |")
    lines += [
        "",
        f"**Both counterfactual arms pass in {paired_passes}/{len(pairs)} "
        "location–seed pairs.** Passing one arm is insufficient to demonstrate that the "
        "agent responds correctly to the changed rule or obstruction.",
        "",
        "The memory result is location dependent: memory0holds for all3red-history trials "
        "but fails all3green departures; memory1fails all3red holds but passes all3green "
        "departures. This is consistent with a location-dependent stop/go tendency, "
        "not demonstrated use of light history. Stop-and-release scores0/12; failures "
        "include both remaining stopped and missing a compliant physical stop. "
        "Obstacle-arm contacts account for all22observed contact starts.",
        "",
        "These trials use certified scripted setup followed by unchanged camera controls. "
        "They do not establish long-route success. There are only two locations per "
        "family, with repeated seeds; do not treat72trials as72independent locations. "
        "The separately preregistered structured stop correction was frozen before these "
        "outcomes and does not train on or reselect from this suite.",
        "",
        "## Physical-arm grid",
        "",
        "Each cell shows passes over the original3seeds. "
        "Arm numbers refer to the immutable certified case definitions.",
        "",
        f"![Counterfactual arm grid]({args.out.name}.png)",
        "",
        f"Source: `{args.assessment}` (SHA256 `{report['assessment_sha256']}`).",
        "",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines))
    labels = sorted({(r["family"], r["location"]) for r in rows})
    matrix = np.array(
        [[sum(r["passed"] for r in groups[(f, loc, arm)]) for arm in (0, 1)] for f, loc in labels]
    )
    fig, ax = plt.subplots(figsize=(8.3, 7.4))
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks([0, 1], ["Arm 0", "Arm 1"])
    ax.set_yticks(range(len(labels)), [f"{f.replace('_', ' ')} · {loc}" for f, loc in labels])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value}/3", ha="center", va="center", fontweight="bold", color="#17212b")
    ax.set_title(
        "CNN-facts rule tests · overrides OFF\n15/72 trials; 0/36 pairs pass both arms", pad=16
    )
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in ("groups", "family_counts")}))


if __name__ == "__main__":
    main()
