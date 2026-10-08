#!/usr/bin/env python3
"""Two fixed-budget BEV pilots; compare under the original benchmark reward."""

import argparse
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

if __package__:
    from .run_traffic_curriculum import ROOT, Queue, read_json, write_json
else:
    from run_traffic_curriculum import ROOT, Queue, read_json, write_json

VARIANTS = {"control": 0.0, "fixed300": 300.0}
CANONICAL_REWARD = "--env-set reward.scales.time=0.1 --env-set reward.scales.collision_fixed=0.0"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def runtime_recovery_matches(saved, current, recovery, original_digest):
    """Allow a recorded queue repair while keeping the experiment itself fixed."""
    if not recovery or recovery.get("original_protocol_sha256") != original_digest:
        return False
    old_hashes, new_hashes = saved["code_sha256"], current["code_sha256"]
    if set(old_hashes) != set(new_hashes):
        return False
    changed = {path for path in old_hashes if old_hashes[path] != new_hashes[path]}
    return (
        changed <= {"scripts/run_traffic_curriculum.py", "scripts/run_collision_cost_ablation.py"}
        and recovery.get("from_code_sha256") == old_hashes
        and recovery.get("to_code_sha256") == new_hashes
        and {k: v for k, v in saved.items() if k != "code_sha256"}
        == {k: v for k, v in current.items() if k != "code_sha256"}
    )


def plan(args):
    stages = []
    for variant in ("reference", *VARIANTS):
        if variant != "reference":
            stages.append({"name": f"{variant}_seed{args.seed}", "hours": args.steps / 21000,
                           "logdir": str(args.run_dir / variant), "target": args.steps})
        stages.extend({"name": f"{variant}_medium_seed{seed}", "hours": 0.04}
                      for seed in args.eval_seeds)
        stages.append({"name": f"{variant}_simple_seed{args.eval_seeds[0]}", "hours": 0.03})
    return stages


class Ablation(Queue):
    def prepare(self):
        """Freeze the source and protocol before any training or evaluation."""
        source_files = (
            "scripts/run_collision_cost_ablation.py", "scripts/train_cardreamer.py",
            "scripts/evaluate_agent.py", "scripts/run_traffic_curriculum.py",
            "src/dreamer/carla_wrappers.py", "configs/laptop.yaml",
            "jobs/slurm_run.sh", "run.py", "configs/phase3_transfer_arms.yaml",
            "patches/cardreamer_compat.patch", "patches/dreamerv3_torch_compat.patch",
        )
        self.source = self.out / "source_60k.pt"
        if not self.source.exists():
            temp = self.source.with_suffix(".pt.tmp")
            shutil.copyfile(self.args.init_from, temp)
            temp.replace(self.source)
        protocol = {
            "source_sha256": digest(self.source), "source_step": 60000,
            "steps_per_pilot": self.args.steps, "seed": self.args.seed,
            "eval_seeds": self.args.eval_seeds, "variants": VARIANTS,
            "task": "carla_right_turn_medium", "obs": "bev", "action": "discrete",
            "train_ratio": 512, "eval_every": 2500, "prefill": 2500,
            "fresh_replay_and_optimizers": True, "early_stop_evals": 0,
            "selection": "final fixed-budget checkpoint",
            "canonical_evaluation_args": CANONICAL_REWARD,
            "run_dir": str(self.args.run_dir),
            "code_sha256": {rel: digest(ROOT / rel) for rel in source_files},
        }
        saved = read_json(self.out / "experiment.json")
        if saved and saved != protocol:
            recovery = read_json(self.args.runtime_recovery) if self.args.runtime_recovery else None
            if not runtime_recovery_matches(
                    saved, protocol, recovery, digest(self.out / "experiment.json")):
                raise ValueError("Existing ablation protocol differs; use a separate directory "
                                 "or an explicit recorded runtime recovery")
        import torch
        source = torch.load(self.source, map_location="cpu", weights_only=False)
        spec = source["run_spec"]
        assert source["step"] == 60000, "Expected the assessed 60k checkpoint"
        assert (spec["arm"], spec["obs"], spec["action"]) == ("cnn", "bev", "discrete")
        assert all(torch.isfinite(value).all() for value in source["agent_state_dict"].values())
        del source
        # Never replace the original provenance when applying a queue repair.
        write_json(self.out / ("runtime_protocol.json" if saved and saved != protocol
                               else "experiment.json"), protocol)
        # Keep the exact effective code/config/patches, including uncommitted changes.
        for rel in source_files:
            snapshot = ("recovery_snapshot" if saved and saved != protocol else "source_snapshot")
            dest = self.out / snapshot / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, dest)

    def run(self, name, env, capture_dir=None):
        env = dict(env)
        if env.get("RUN_MODE") == "evaluate":
            # Evaluator inherits checkpoint training overrides first. Last
            # overrides restore the canonical reward for both pilots.
            env["EVAL_ARGS"] = env.get("EVAL_ARGS", "") + " " + CANONICAL_REWARD
        return super().run(name, env, capture_dir=capture_dir)

    def train_pilot(self, variant):
        args = self.args
        name, logdir = f"{variant}_seed{args.seed}", args.run_dir / variant
        fixed_cost = VARIANTS[variant]
        expected_overrides = ["reward.scales.time=0.1",
                              f"reward.scales.collision_fixed={fixed_cost}"]
        latest = logdir / "latest.pt"
        if latest.exists():
            import torch
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            spec = checkpoint["run_spec"]
            if (spec.get("env_overrides") != expected_overrides
                    or spec.get("init_from") != str(self.source)
                    or spec.get("seed") != args.seed
                    or (spec.get("arm"), spec.get("obs"), spec.get("action"))
                    != ("cnn", "bev", "discrete")
                    or checkpoint["step"] > args.steps):
                raise ValueError(f"Resume checkpoint does not match pilot protocol: {latest}")
            del checkpoint
        completed = read_json(logdir / "metrics.json")
        if completed:
            if (completed.get("env_steps") != args.steps
                    or completed.get("env_overrides") != expected_overrides
                    or completed.get("init_from") != str(self.source)):
                raise ValueError(f"Completed pilot does not match protocol: {logdir}")
            self.status("DONE", name)
            return logdir / "latest.pt"
        for attempt in range(1, 5):
            resume = ("--resume" if (logdir / "latest.pt").is_file()
                      else f"--init-from {self.source}")
            env = self.environment(f"{name}_try{attempt}")
            env.update(RUN_MODE="cnn", TASK="carla_right_turn_medium", SEED=str(args.seed),
                       OBS="bev", ACTION="discrete", STEPS=str(args.steps),
                       TRAIN_ARGS=f"{resume} --logdir {logdir} --set train_ratio=512 "
                       "--set eval_every=2500 --early-stop-evals 0 "
                       f"--env-set {expected_overrides[0]} --env-set {expected_overrides[1]}")
            # Final checkpoint only; no reward-based best-checkpoint selection.
            if self.run(name, env):
                metrics = read_json(logdir / "metrics.json")
                if not metrics or metrics["env_steps"] != args.steps:
                    raise RuntimeError(f"Pilot ended without completing its budget: {name}")
                return logdir / "latest.pt"
            if self.stop.exists():
                return None
            (self.out / f"{name}.log").rename(self.out / f"{name}.try{attempt}.log")
            if attempt < 4:
                self.status("RETRY", f"{name} attempt {attempt + 1}/4")
                time.sleep(10)
        return None

    def evaluate_candidate(self, variant, checkpoint):
        evaluations = {}
        for task, seed, episodes in (
            *(("medium", seed, 30) for seed in self.args.eval_seeds),
            ("simple", self.args.eval_seeds[0], 10),
        ):
            self.args.eval_seed = seed
            name = f"{variant}_{task}_seed{seed}"
            dest = self.out / "evaluations" / name
            request = {"checkpoint_sha256": digest(checkpoint), "seed": seed,
                       "task": f"carla_right_turn_{task}", "episodes": episodes,
                       "reward_overrides": CANONICAL_REWARD, "shield": False}
            saved = read_json(dest / "evaluation_request.json")
            if saved and saved != request:
                raise ValueError(f"Evaluation cache has different provenance: {dest}")
            if (dest / "evaluation_metrics.json").exists() and saved is None:
                raise ValueError(f"Evaluation cache has no provenance: {dest}")
            write_json(dest / "evaluation_request.json", request)
            summary = self.evaluate(name, "cnn", checkpoint, request["task"], episodes)
            if summary is None:
                return None
            data = read_json(dest / "evaluation_metrics.json")
            records = data["episodes"]
            if len(records) != episodes:
                raise ValueError(f"Incomplete evaluation: {name}")
            evaluations[f"{task}_seed{seed}"] = {
                "summary": summary,
                "collision_free_arrivals": sum(
                    e["success"] and not e["collision"] for e in records),
                "arrival_collision_overlap": sum(e["success"] and e["collision"] for e in records),
                "timeouts": sum(e["time_exceeded"] for e in records),
            }
        return evaluations

    def execute(self):
        self.status("=== collision-cost ablation started ===")
        results = read_json(self.out / "results.json") or {}
        results["completed"] = False
        results.pop("error", None)
        try:
            self.prepare()
            while not self.idle():
                if self.check_health():
                    return
                print("Waiting for existing CARLA client...", flush=True)
                time.sleep(10)
            for variant in ("reference", *VARIANTS):
                if self.check_health():
                    return
                checkpoint = self.source if variant == "reference" else self.train_pilot(variant)
                if checkpoint is None:
                    return
                measurements = self.evaluate_candidate(variant, checkpoint)
                if measurements is None:
                    return
                results[variant] = {"checkpoint": str(checkpoint), "evaluations": measurements}
                write_json(self.out / "results.json", results)
            results["completed"] = True
            self.write_comparison(results)
        except Exception as error:
            results["error"] = f"{type(error).__name__}: {error}"
            self.status("ERROR", results["error"])
            raise
        finally:
            write_json(self.out / "results.json", results)
            # Old SKIP/FAIL records from a paused invocation do not account
            # for unattempted work in the current invocation.
            status = (self.out / "status.txt").read_text().rsplit(
                "=== collision-cost ablation started ===", 1)[-1]
            for step in plan(self.args):
                if not any(f" {state} {step['name']}" in status
                           for state in ("START", "DONE", "OK", "FAIL", "SKIP")):
                    self.status("SKIP", f"{step['name']} (earlier stage failed or STOP)")
            self.status("=== collision-cost ablation finished ===")

    def write_comparison(self, results):
        lines = ["# Collision-cost pilot comparison", "",
                 "Canonical reward, no shield; one training seed. Final budget checkpoints.",
                 "", "| Candidate | Evaluation | Safe arrivals | Collisions | Timeouts | Return |",
                 "|---|---|---:|---:|---:|---:|"]
        for variant in ("reference", *VARIANTS):
            for name, data in results[variant]["evaluations"].items():
                summary = data["summary"]
                count = summary["num_episodes"]
                collisions = round(summary["collision_rate"] * count)
                lines.append(f"| {variant} | {name} | {data['collision_free_arrivals']}/{count} "
                             f"| {collisions}/{count} | {data['timeouts']} "
                             f"| {summary['mean_reward']:.1f} |")
        (self.out / "comparison_summary.md").write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/collision_cost_ablation_queue"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/diagnostics/collision_cost_ablation_v1"))
    parser.add_argument("--init-from", type=Path,
                        default=Path("outputs/diagnostics/bev_traffic_medium_seed42/best.pt"))
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seeds", type=int, nargs=2, default=[2029, 2030])
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--runtime-recovery", type=Path,
                        help="Explicit old/new code hashes for a queue-only recovery")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("out", "run_dir", "init_from"):
        setattr(args, key, getattr(args, key).resolve())
    if args.steps <= 2500 or args.max_hours <= 0 or len(set(args.eval_seeds)) != 2:
        parser.error("Require steps >2500, positive max-hours, and two distinct evaluation seeds")
    if not args.init_from.is_file():
        parser.error(f"Missing source checkpoint: {args.init_from}")
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return
    queue = Ablation(args, steps=plan(args))
    signal.signal(signal.SIGTERM, lambda *_: queue.request_stop("controller SIGTERM"))
    queue.execute()


if __name__ == "__main__":
    main()
