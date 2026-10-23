#!/usr/bin/env python3
"""Reproduce corrected67-fit label curves and freeze the primary before driving."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/auto_thinker_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_navigation_visual_study import fit_cells, verify_fit  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402

NAMES = dict(
    cnn_scratch="CNN scratch",
    imagenet_resnet18="ImageNet ResNet18",
    dinov2_small="DINOv2-S",
    jepa_comma2k19="Comma JEPA",
    jepa_carla="CARLA JEPA",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "outputs/navigation_repaired_study_v4_20261020"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    progress = json.loads((args.study / "progress.json").read_text())
    if progress["status"] != "complete" and not args.partial:
        raise ValueError("Full study must finish; label partial snapshots explicitly")
    declared = fit_cells() + [
        dict(encoder="jepa_random", fraction=1.0, seed=42, color_enabled=True)
    ]
    records = []
    for path in sorted((args.study / "completed").glob("*.json")):
        saved = json.loads(path.read_text())
        cell = {k: saved[k] for k in ("encoder", "fraction", "seed", "color_enabled")}
        if cell not in declared:
            raise ValueError("Unregistered cell")
        records.append(verify_fit(Path(saved["fit"]), cell))
    if len({(r["encoder"], r["fraction"], r["seed"], r["color_enabled"]) for r in records}) != len(
        records
    ):
        raise ValueError("Duplicate cell")
    if not args.partial and len(records) != 67:
        raise ValueError("Missing registered fits")
    rows = []
    for encoder in NAMES:
        for fraction in (0.05, 0.1, 0.25, 1.0):
            group = sorted(
                [
                    r
                    for r in records
                    if r["encoder"] == encoder and r["fraction"] == fraction and r["color_enabled"]
                ],
                key=lambda r: r["seed"],
            )
            if len(group) != 3:
                continue
            metrics = {
                k: [r["metrics"][k] for r in group]
                for k in (
                    "selection_score",
                    "controller_stop_recall",
                    "controller_false_hold_fraction",
                    "go_speed_mae_mps",
                    "path_correction_mae_m",
                )
            }
            rows.append(
                dict(
                    encoder=encoder,
                    nominal_fraction=fraction,
                    labels=group[0]["labels"],
                    actual_fraction=group[0]["labels"] / 15712,
                    seeds=[r["seed"] for r in group],
                    means={k: float(np.mean(v)) for k, v in metrics.items()},
                    minima={k: float(np.min(v)) for k, v in metrics.items()},
                    maxima={k: float(np.max(v)) for k, v in metrics.items()},
                    individual=group,
                    fit_wall_seconds=sum(r["elapsed_s"] for r in group),
                )
            )
    ablations = [r for r in records if not r["color_enabled"] or r["encoder"] == "nav_color_only"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure = args.out.with_suffix(".png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for encoder, label in NAMES.items():
        group = sorted([r for r in rows if r["encoder"] == encoder], key=lambda r: r["labels"])
        if not group:
            continue
        x = [r["labels"] for r in group]
        for ax, key, title in zip(
            axes,
            ("selection_score", "controller_stop_recall", "controller_false_hold_fraction"),
            ("Balanced stop score ↓", "Stop recall ↑", "False holds ↓"),
        ):
            y = [r["means"][key] for r in group]
            line = ax.plot(x, y, "o-", label=label)[0]
            ax.fill_between(
                x,
                [r["minima"][key] for r in group],
                [r["maxima"][key] for r in group],
                color=line.get_color(),
                alpha=0.1,
            )
            ax.set(title=title, xlabel="Actual labeled timestamps", xscale="log")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{'PARTIAL ' if args.partial else ''}Offline label efficiency • three seeds "
        "• range is not a confidence interval",
        fontsize=11,
    )
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    primary = None
    if not args.partial:
        ranked = sorted(
            [r for r in rows if r["nominal_fraction"] == 1.0],
            key=lambda r: (r["means"]["selection_score"], r["encoder"]),
        )
        selected = ranked[0]
        candidate = next(r for r in selected["individual"] if r["seed"] == 42)
        primary = dict(
            encoder=selected["encoder"],
            fit=candidate["fit"],
            policy_sha256=candidate["policy_sha256"],
            selection="Lowest mean full-label internal score over all3 seeds; fixed seed42 weights",
            driving_qualified=False,
        )
    atomic_json(
        args.out,
        dict(
            status="partial" if args.partial else "complete",
            completed_cells=len(records),
            planned_cells=67,
            request_sha256=digest(args.study / "request.json"),
            script_sha256=digest(__file__),
            selection_protocol_sha256=digest(
                ROOT / "reports/navigation_m4_selection_protocol_20261019.md"
            ),
            groups=rows,
            component_ablations=ablations,
            primary_candidate=primary,
            figure=str(figure),
            driving_improvement_established=False,
            random_vit_diagnostic_pending=False,
            random_vit_diagnostic=[r for r in records if r["encoder"] == "jepa_random"],
            actual_geography_repaired=True,
            repair_protocol_sha256=digest(
                ROOT / "reports/navigation_m4_geography_repair_protocol_20261019.md"
            ),
            note=(
                "Fixed original13 validation episodes/1312 positions; "
                "offline familiar validation. All cells/seeds/failures retained."
            ),
        ),
    )
    table = [
        "| Encoder | Labels | Score | Stop recall | False holds | Go MAE | Fit wall(s),3 seeds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["means"]
        table.append(
            f"| {NAMES[row['encoder']]} | {row['labels']} | {m['selection_score']:.4f} | "
            f"{m['controller_stop_recall']:.1%} | {m['controller_false_hold_fraction']:.1%} | "
            f"{m['go_speed_mae_mps']:.3f} | {row['fit_wall_seconds']:.1f} |"
        )
    args.out.with_suffix(".md").write_text(
        f"# {'Partial ' if args.partial else ''}Corrected encoder study —20 October2026\n\n"
        f"{len(records)}/67 registered fits verified. "
        "Offline results do not establish safer driving.\n\n"
        f"![Label curves]({figure.name})\n\n"
        + "\n".join(table)
        + "\n\nThree-seed ranges are descriptive, not confidence intervals. "
        "Actual label fractions are3.49/9.65/24.55/100%, "
        "549/1516/3857/15712timestamps. Corrected55TRAIN episodes; unchanged13VAL/2822timestamps. "
        "Prior contaminated study is preserved separately. "
        "ImageNet external labels/DINO data scale and pretraining/latencies are disclosed; "
        "downstream updates match, total compute does not. "
        "Common photometric fields use no local annotations; no full-label head is inherited. "
        "All component controls and the random-ViT diagnostic are in JSON. "
        "Native driving qualification remains pending.\n"
    )
    print(json.dumps(dict(completed_cells=len(records), primary_candidate=primary)), flush=True)


if __name__ == "__main__":
    main()
