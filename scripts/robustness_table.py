#!/usr/bin/env python3
"""
Summarise the zero-shot robustness evaluation (jobs/robustness.sh).

Reads <results>/<arm>_seed<n>/<condition>/evaluation_metrics.json and writes
<results>/robustness.md (success, collision and off-lane rates per condition
and encoder; mean over seeds with each seed's value) plus robustness.json and
a grouped bar chart robustness.png.

Usage:
    python scripts/robustness_table.py --results outputs/robustness
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ["cnn", "custom_jepa", "vjepa2"]
CONDITION_ORDER = ["baseline", "night", "rain", "sunset", "traffic"]
COLOURS = {"cnn": "#1f77b4", "custom_jepa": "#d62728", "vjepa2": "#2ca02c"}


def load(results: Path) -> dict:
    """{(arm, condition): {seed: summary}}"""
    data = defaultdict(dict)
    for path in sorted(results.glob("*_seed*/*/evaluation_metrics.json")):
        m = re.match(r"(.+)_seed(\d+)", path.parent.parent.name)
        if m:
            data[(m.group(1), path.parent.name)][int(m.group(2))] = (
                json.loads(path.read_text())["summary"])
    return data


def ordered(values, order):
    return sorted(set(values), key=lambda v: (order.index(v) if v in order else 99, v))


def cell(per_seed: dict, key: str) -> str:
    if not per_seed:
        return "-"
    values = [s[key] for _, s in sorted(per_seed.items())]
    seeds = " / ".join(f"{v:.0%}" for v in values)
    return f"**{sum(values) / len(values):.0%}** ({seeds})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("outputs/robustness"))
    args = parser.parse_args()

    data = load(args.results)
    if not data:
        print(f"No evaluations found under {args.results}")
        return 1
    arms = ordered([a for a, _ in data], ARM_ORDER)
    conditions = ordered([c for _, c in data], CONDITION_ORDER)
    seeds = sorted({s for per_seed in data.values() for s in per_seed})

    lines = ["# Zero-shot robustness (trained on carla_right_turn_simple, clear weather)", "",
             f"Seeds: {', '.join(map(str, seeds))}. Cells: mean over seeds (per-seed values).", ""]
    for key, title in (("success_rate", "Success rate"), ("collision_rate", "Collision rate"),
                       ("out_of_lane_rate", "Out-of-lane rate")):
        lines += [f"## {title}", "", "| Condition | " + " | ".join(arms) + " |",
                  "|---" * (len(arms) + 1) + "|"]
        for cond in conditions:
            lines.append(f"| {cond} | " + " | ".join(cell(data.get((a, cond), {}), key)
                                                   for a in arms) + " |")
        lines.append("")
    (args.results / "robustness.md").write_text("\n".join(lines))
    (args.results / "robustness.json").write_text(json.dumps(
        {f"{a}|{c}": {str(s): v for s, v in per_seed.items()} for (a, c), per_seed in data.items()},
        indent=2))
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        for j, cond in enumerate(conditions):
            per_seed = data.get((arm, cond), {})
            if not per_seed:
                continue
            values = [s["success_rate"] for s in per_seed.values()]
            x = j + (i - (len(arms) - 1) / 2) * width
            ax.bar(x, sum(values) / len(values), width * 0.9, color=COLOURS.get(arm),
                   label=arm if j == 0 else None)
            ax.scatter([x] * len(values), values, color="black", s=10, zorder=3)
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set(ylabel="success rate", ylim=(0, 1.05),
           title="Zero-shot robustness (bars: mean over seeds, dots: seeds)")
    ax.legend(frameon=False, ncol=len(arms))
    fig.tight_layout()
    fig.savefig(args.results / "robustness.png", dpi=120)
    print(f"\nSaved {args.results / 'robustness.md'} and robustness.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
