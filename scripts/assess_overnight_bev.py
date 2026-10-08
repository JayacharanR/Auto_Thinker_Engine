#!/usr/bin/env python3
"""Audit a completed overnight BEV queue using CPU checkpoints and raw episodes.

No CARLA launch, policy updates or checkpoint writes. Output is a compact JSON
evidence file; --plot optionally produces a standalone results figure.
"""

import argparse
import math
from functools import lru_cache
from pathlib import Path

if __package__:
    from .run_collision_cost_ablation import CANONICAL_REWARD, digest
    from .run_traffic_curriculum import ROOT, read_json, write_json
else:
    from run_collision_cost_ablation import CANONICAL_REWARD, digest
    from run_traffic_curriculum import ROOT, read_json, write_json


@lru_cache
def checksum(path):
    return digest(path)


def counts(measurement):
    summary = measurement["summary"]
    n = summary["num_episodes"]
    return {"episodes": n, "safe_arrivals": measurement["collision_free_arrivals"],
            "raw_arrivals": round(summary["success_rate"] * n),
            "collisions": round(summary["collision_rate"] * n),
            "lane_failures": round(summary["out_of_lane_rate"] * n),
            "timeouts": measurement["timeouts"],
            "arrival_collision_overlap": measurement["arrival_collision_overlap"],
            "mean_reward": summary["mean_reward"], "mean_speed_kmh": summary["mean_speed_kmh"]}


def check_evaluation(queue, name, measurement, checkpoint):
    dest = queue / "evaluations" / name
    request = read_json(dest / "evaluation_request.json")
    data = read_json(dest / "evaluation_metrics.json")
    assert request["checkpoint_sha256"] == checksum(checkpoint), name
    assert request.get("reward", request.get("reward_overrides")) == CANONICAL_REWARD, name
    assert request["shield"] is False, name
    prefix, seed = name.rsplit("_seed", 1)
    task = prefix.rsplit("_", 1)[-1]
    assert request["seed"] == int(seed), name
    assert request["task"] in (task, f"carla_right_turn_{task}"), name
    manifest = read_json(dest / "manifest.json")
    assert manifest["seed"] == int(seed) and manifest["task"] == f"carla_right_turn_{task}", name
    assert Path(manifest["checkpoint"]["path"]) == checkpoint, name
    records = data["episodes"]
    n = len(records)
    assert n == request["episodes"] == measurement["summary"]["num_episodes"], name
    assert data["summary"] == measurement["summary"], name
    for key, flag in (("success_rate", "success"), ("collision_rate", "collision"),
                      ("out_of_lane_rate", "out_of_lane")):
        assert math.isclose(data["summary"][key], sum(e[flag] for e in records) / n), name
    assert measurement["collision_free_arrivals"] == sum(
        e["success"] and not e["collision"] for e in records), name
    assert measurement["arrival_collision_overlap"] == sum(
        e["success"] and e["collision"] for e in records), name
    assert measurement["timeouts"] == sum(e["time_exceeded"] for e in records), name
    assert math.isclose(data["summary"]["mean_reward"],
                        sum(e["total_reward"] for e in records) / n), name
    return {"request_sha256": checksum(dest / "evaluation_request.json"),
            "episodes_sha256": checksum(dest / "evaluation_metrics.json"), **counts(measurement)}


def check_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["agent_state_dict"]
    assert all(torch.isfinite(v).all() for v in weights.values()), path
    optimizers = {}
    for name, optimizer in checkpoint["optims_state_dict"].items():
        counters = []
        for state in optimizer["state"].values():
            for value in state.values():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all(), (path, name)
            if "step" in state:
                counters.append(int(state["step"]))
        optimizers[name] = {"min_step": min(counters), "max_step": max(counters)}
    return {"sha256": checksum(path), "step": checkpoint["step"],
            "finite_weights_and_optimizer_tensors": True, "optimizer_counters": optimizers,
            "run_spec": checkpoint["run_spec"],
            "actor_lr": checkpoint["dreamer_config"]["actor"]["lr"],
            "critic_lr": checkpoint["dreamer_config"]["critic"]["lr"]}


def audit(queue):
    results = read_json(queue / "results.json")
    assert results["completed"] is True and "error" not in results
    selection = read_json(queue / "source_selection.json")
    baseline = queue / "baseline.pt"
    best = queue / "best.pt"
    winner = selection["choices"][selection["winner"]]
    assert checksum(baseline) == checksum(Path(winner["checkpoint"]))
    assert checksum(best) == checksum(queue / "checkpoints" / f"step{results['best_step']}.pt")
    latest = Path(read_json(queue / "plan.json")["run_dir"]) / "latest.pt"
    assert checksum(latest) == checksum(Path(results["blocks"][-1]["checkpoint"]))
    report = {"queue": str(queue), "completed": True, "selected_source": selection["winner"],
              "best_step": results["best_step"], "checkpoints": {}, "evaluations": {},
              "training_additional_steps": results["blocks"][-1]["step"] - results["baseline_step"],
              "selection_baseline": {k: counts(v) for k, v in winner["evaluations"].items()},
              "selection_blocks": [], "holdouts": {}}
    for path in (baseline, best, *(Path(b["checkpoint"]) for b in results["blocks"]), latest):
        report["checkpoints"][str(path)] = check_checkpoint(path)
    pilots_dir = ROOT / "outputs/collision_cost_ablation_queue"
    pilots = read_json(pilots_dir / "results.json")
    assert pilots["completed"] is True
    report["completed_pilots"] = {}
    for variant, choice in selection["choices"].items():
        checkpoint = Path(choice["checkpoint"])
        report["checkpoints"][str(checkpoint)] = check_checkpoint(checkpoint)
        report["evaluations"][f"choose_{variant}_medium_seed2028"] = check_evaluation(
            queue, f"choose_{variant}_medium_seed2028",
            choice["evaluations"]["medium_seed2028"], checkpoint)
        report["completed_pilots"][variant] = {}
        for kind, measurement in pilots[variant]["evaluations"].items():
            assert choice["evaluations"][kind] == measurement
            name = f"{variant}_{kind}"
            report["evaluations"][name] = check_evaluation(
                pilots_dir, name, measurement, checkpoint)
            report["completed_pilots"][variant][kind] = counts(measurement)
    for block in results["blocks"]:
        compact = {"step": block["step"], "accepted": block["accepted"], "evaluations": {}}
        for kind, measurement in block["evaluations"].items():
            name = f"block{block['block']}_{kind}"
            report["evaluations"][name] = check_evaluation(
                queue, name, measurement, Path(block["checkpoint"]))
            compact["evaluations"][kind] = counts(measurement)
        report["selection_blocks"].append(compact)
    for candidate, checkpoint in (("baseline", baseline), ("selected", best)):
        report["holdouts"][candidate] = {}
        for kind, measurement in results[f"holdout_{candidate}"].items():
            name = f"holdout_{candidate}_{kind}"
            report["evaluations"][name] = check_evaluation(queue, name, measurement, checkpoint)
            report["holdouts"][candidate][kind] = counts(measurement)
    protocol = read_json(queue / "experiment.json")
    report["recorded_code_matches_current"] = {
        rel: digest(ROOT / rel) == saved for rel, saved in protocol["code_sha256"].items()}
    report["queue_status_sha256"] = checksum(queue / "status.txt")
    report["source_selection_sha256"] = checksum(queue / "source_selection.json")
    return report


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [(report["checkpoints"][str(Path(report["queue"]) / "baseline.pt")]["step"],
               report["selection_baseline"])]
    stages += [(b["step"], b["evaluations"]) for b in report["selection_blocks"]]
    x = [step / 1000 for step, _ in stages]
    safe = [sum(d["safe_arrivals"] for k, d in ev.items() if k.startswith("medium"))
            for _, ev in stages]
    simple = [ev["simple_seed2029"]["safe_arrivals"] for _, ev in stages]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(x, [n / 90 * 100 for n in safe], "o-", label="Medium selection (90 episodes)")
    axes[0].plot(x, [n / 10 * 100 for n in simple], "s--", label="Simple retention (10 episodes)")
    for step, count in zip(x, safe):
        axes[0].annotate(f"{count}/90", (step, count / 90 * 100), xytext=(0, 8),
                         textcoords="offset points", ha="center")
    axes[0].axvline(report["best_step"] / 1000, color="green", alpha=0.35,
                    label="Selected checkpoint")
    axes[0].set(xlabel="Continuation run counter (thousand steps)", ylabel="Safe arrivals (%)",
                ylim=(-5, 113), title="More training eventually regressed")
    axes[0].legend(fontsize=8, loc="lower left")
    for offset, candidate in ((-0.18, "baseline"), (0.18, "selected")):
        ev = report["holdouts"][candidate]
        nums = [sum(d["safe_arrivals"] for k, d in ev.items() if k.startswith("medium")),
                ev["hard_seed5031"]["safe_arrivals"], ev["simple_seed3031"]["safe_arrivals"]]
        totals = [60, 30, 10]
        bars = axes[1].bar([i + offset for i in range(3)],
                           [n / total * 100 for n, total in zip(nums, totals)],
                           width=0.34, label=candidate)
        axes[1].bar_label(bars, labels=[f"{n}/{t}" for n, t in zip(nums, totals)], fontsize=8)
    axes[1].set(xticks=range(3), xticklabels=["Medium", "Hard (zero-shot)", "Simple"],
                ylabel="Safe arrivals (%)", ylim=(0, 113),
                title="Separate final evaluation streams")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("One CNN BEV training seed; canonical reward, no shield")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/overnight_bev_20261006"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    report = audit(args.out.resolve())
    write_json(args.audit, report)
    if args.plot:
        plot(report, args.plot)
    print(f"Audited {len(report['evaluations'])} evaluations and "
          f"{len(report['checkpoints'])} finite optimizer-bearing checkpoints; "
          f"best step={report['best_step']}")


if __name__ == "__main__":
    main()
