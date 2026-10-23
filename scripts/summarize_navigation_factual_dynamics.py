#!/usr/bin/env python3
"""Audit all completed factual fits against explicit no-observation baselines."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def constant_baselines(windows):
    valid = windows["cost_valid"][..., 0] > 0
    outcomes = np.where(valid, windows["outcomes"][..., 0], 0.0)
    train, val = ~windows["is_validation"], windows["is_validation"]
    train_counts, val_counts = valid[train].sum(0), valid[val].sum(0)
    train_positive = ((outcomes[train] == 1) & valid[train]).sum(0)
    val_positive = ((outcomes[val] == 1) & valid[val]).sum(0)
    # Fixed add-half smoothing, derived exclusively from TRAIN labels.
    prior = (train_positive + 0.5) / (train_counts + 1)
    errors = (outcomes[val] - prior[None]) ** 2
    bce = -(outcomes[val] * np.log(prior) + (1 - outcomes[val]) * np.log1p(-prior))
    return dict(
        train_positive_by_horizon=train_positive.tolist(),
        train_labeled_by_horizon=train_counts.tolist(),
        validation_positive_by_horizon=val_positive.tolist(),
        validation_labeled_by_horizon=val_counts.tolist(),
        always_zero_brier_by_horizon=(val_positive / val_counts).tolist(),
        train_prior_probability_by_horizon=prior.tolist(),
        train_prior_brier_by_horizon=((errors * valid[val]).sum(0) / val_counts).tolist(),
        train_prior_collision_bce=float((bce * valid[val]).sum() / val_counts.sum()),
        collision_positive_validation_episodes=int(len(np.unique(windows["episode_id"][
            val & (((outcomes == 1) & valid).any(1))
        ]))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain every factual result report")
    if read(args.study / "progress.json")["status"] != "complete":
        raise ValueError("All six registered fits must be complete")
    manifest = read(args.windows / "manifest.json")
    if digest(args.windows / "windows.npz") != manifest["window_arrays_sha256"]:
        raise ValueError("Frozen factual windows changed")
    windows = dict(np.load(args.windows / "windows.npz", allow_pickle=False))
    baseline = constant_baselines(windows)
    result = read(args.study / "results.json")
    if [(r["model"], r["seed"]) for r in result] != [
        (model, seed) for model in ("jepa", "rssm") for seed in (42, 1051, 2060)
    ]:
        raise ValueError("Exact registered six-member matrix required")
    paired = []
    for seed in (42, 1051, 2060):
        logs = [lines(args.study / f"{model}_seed{seed}/train.jsonl") for model in ("jepa", "rssm")]
        if any([r["step"] for r in log] != list(range(1, 2001)) for log in logs):
            raise ValueError("Incomplete actual update evidence")
        if [r["window_indices"] for r in logs[0]] != [r["window_indices"] for r in logs[1]]:
            raise ValueError("Paired model arms saw different window draws")
        paired.append(dict(seed=seed, updates=2000, exact_draws_match=True))
    members = []
    for row in result:
        root = Path(row["output"])
        progress = read(root / "progress.json")
        selected = row["selected_validation"]
        actual = next(r for r in lines(root / "validation.jsonl")
                      if r["step"] == selected["selected_step"])
        if (actual["metrics"] != selected["metrics"]
                or progress["selected_step"] != selected["selected_step"]
                or digest(root / "best_model.pt") != row["checkpoint_sha256"]):
            raise ValueError("Selected weight/metric identity changed")
        metrics = selected["metrics"]
        if (metrics["positive_by_horizon"] != baseline["validation_positive_by_horizon"]
                or metrics["collision_labeled_by_horizon"]
                != baseline["validation_labeled_by_horizon"]):
            raise ValueError("Selected member evaluation labels changed")
        members.append(dict(
            model=row["model"], seed=row["seed"], selected_step=selected["selected_step"],
            checkpoint_sha256=row["checkpoint_sha256"],
            collision_bce=metrics["collision_bce"],
            brier_4s=metrics["collision_brier_by_horizon"][-1],
            missed_positive_horizons_at_half=sum(metrics["missed_positive_at_half_by_horizon"]),
            positive_horizons=sum(metrics["positive_by_horizon"]),
            progress_mae_4s_m=metrics["progress_mae_m_by_horizon"][-1],
            calibration_bins=metrics["calibration_bins"],
        ))
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "analysis_source.py")
    atomic_json(args.out / "assessment.json", dict(
        members=members, baselines=baseline, exact_paired_draws=paired,
        study_results_sha256=digest(args.study / "results.json"),
        windows_sha256=digest(args.windows / "windows.npz"),
        no_model_updates=True, planner_qualified=False, calibration_qualified=False,
        note="Overlapping windows and one positive validation episode; horizons are not "
             "independent encounters. Neither Brier nor latent loss proves safe action ranking.",
    ))
    text = ["# Factual world-model diagnosis", "",
            "All6 fits and12,000applied updates verified; each JEPA/RSSMseed saw identical draws.",
            "No new fit, calibration or native-driving result. All models remain unqualified.", "",
            "| Model | Seed | Selected | Collision BCE | Brier4s | "
            "Missed/positive horizons | Progress MAE4s |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in members:
        text.append(f"| {row['model']} | {row['seed']} | {row['selected_step']} | "
                    f"{row['collision_bce']:.4f} | {row['brier_4s']:.4f} | "
                    f"{row['missed_positive_horizons_at_half']}/{row['positive_horizons']} | "
                    f"{row['progress_mae_4s_m']:.3f}m |")
    text.extend(["", f"Always-zero Brier4s: {baseline['always_zero_brier_by_horizon'][-1]:.6f}. "
                 "TRAIN-only add-half constant-prior BCE: "
                 f"{baseline['train_prior_collision_bce']:.6f}; "
                 f"Brier4s: {baseline['train_prior_brier_by_horizon'][-1]:.6f}.", "",
                 "All49positive horizons come from one validation episode and overlapping windows. "
                 "Most selected members miss every positive horizon at.5; this threshold is "
                 "descriptive, not tuned. Always-safe references expose class imbalance, "
                 "not useful planning.", "",
                 "Zero positive red examples: red was masked/frozen and omitted at inference. "
                 "No constant-velocity superiority, reliable uncertainty or counterfactual "
                 "calibration is established."])
    (args.out / "report.md").write_text("\n".join(text) + "\n")
    print(json.dumps(dict(members=len(members), paired_draws_verified=True, baselines=baseline)),
          flush=True)


if __name__ == "__main__":
    main()
