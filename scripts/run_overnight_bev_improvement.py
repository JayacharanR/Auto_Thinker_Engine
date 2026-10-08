#!/usr/bin/env python3
"""Finish the controlled pilot, then continue a validated BEV policy overnight.

Preserves the original pilot, replay and optimizer state. No benchmark changes.
STOP is scoped to this queue; the recovery child receives its own STOP file.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .run_collision_cost_ablation import CANONICAL_REWARD, digest
    from .run_traffic_curriculum import ROOT, Queue, read_json, write_json
else:
    from run_collision_cost_ablation import CANONICAL_REWARD, digest
    from run_traffic_curriculum import ROOT, Queue, read_json, write_json

VALIDATION_SEEDS = (2028, 2029, 2030)
HOLDOUT_SEEDS = (3031, 4031)


def safe_count(measurement):
    return measurement["collision_free_arrivals"]


def retained(measurements):
    simple = measurements["simple_seed2029"]
    return safe_count(simple) >= 9 and simple["summary"]["collision_rate"] <= 0.1


def rank(measurements):
    """Favor the weakest stream, then pooled safety; reward is not a selector."""
    traffic = [measurements[f"medium_seed{s}"] for s in VALIDATION_SEEDS]
    return (min(map(safe_count, traffic)), sum(map(safe_count, traffic)),
            -sum(round(m["summary"]["collision_rate"] * 30) for m in traffic),
            -sum(round(m["summary"]["out_of_lane_rate"] * 30) for m in traffic))


def improvement(candidate, incumbent):
    """Do not adopt a one-episode gain or sacrifice a stream / turning skill."""
    if not retained(candidate):
        return False
    new, old = rank(candidate), rank(incumbent)
    return (new[0] >= old[0] and new[1] >= old[1] + 3
            and new[2] >= old[2] and new[3] >= old[3]
            and all(safe_count(candidate[f"medium_seed{s}"])
                    >= safe_count(incumbent[f"medium_seed{s}"]) - 2
                    for s in VALIDATION_SEEDS))


def checkpoint_step(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec = checkpoint["run_spec"]
    if (spec["arm"], spec["obs"], spec["action"], spec["task"]) != (
            "cnn", "bev", "discrete", "carla_right_turn_medium"):
        raise ValueError(f"Wrong continuation contract: {path}")
    if not checkpoint.get("optims_state_dict"):
        raise ValueError(f"Missing optimizer state: {path}")
    if not all(torch.isfinite(v).all() for v in checkpoint["agent_state_dict"].values()):
        raise ValueError(f"Non-finite weights: {path}")
    return int(checkpoint["step"]), spec


def clone_training_state(source, destination):
    """Clone completed replay and checkpoint; never mutate the assessed source."""
    if destination.exists():
        raise FileExistsError(f"Refusing to replace continuation state: {destination}")
    if not list((source / "train_eps").glob("*.npz")):
        raise ValueError(f"No replay to resume: {source}")
    destination.mkdir(parents=True)
    shutil.copytree(source / "train_eps", destination / "train_eps")
    shutil.copyfile(source / "latest.pt", destination / "latest.pt")


def make_steps(args):
    steps = [{"name": "finish_fixed_pilot", "hours": 0.52,
              "logdir": str(ROOT / "outputs/diagnostics/collision_cost_ablation_v1/fixed300"),
              "start": 21278, "target": 30000}]
    steps.extend({"name": f"choose_{variant}_medium_seed2028", "hours": 0.04}
                 for variant in ("reference", "control", "fixed300"))
    # Filled with the selected source's absolute counter before continuation.
    for block in range(1, args.blocks + 1):
        steps.append({"name": f"continue_block{block}", "hours": args.block_steps / 21000,
                      "logdir": str(args.run_dir), "start": (block - 1) * args.block_steps,
                      "target": block * args.block_steps})
        steps.extend({"name": f"block{block}_medium_seed{s}", "hours": 0.04}
                     for s in VALIDATION_SEEDS)
        steps.append({"name": f"block{block}_simple_seed2029", "hours": 0.03})
    for candidate in ("baseline", "selected"):
        steps.extend({"name": f"holdout_{candidate}_medium_seed{s}", "hours": 0.04}
                     for s in HOLDOUT_SEEDS)
        steps.append({"name": f"holdout_{candidate}_simple_seed3031", "hours": 0.03})
        steps.append({"name": f"holdout_{candidate}_hard_seed5031", "hours": 0.04})
    return steps


class Overnight(Queue):
    def run(self, name, env, capture_dir=None):
        env = dict(env)
        if env.get("RUN_MODE") == "evaluate":
            env["EVAL_ARGS"] += " " + CANONICAL_REWARD
        return super().run(name, env, capture_dir)

    def finish_pilot(self):
        previous = ROOT / "outputs/collision_cost_ablation_queue"
        results = read_json(previous / "results.json")
        if results and results.get("completed"):
            self.status("DONE", "finish_fixed_pilot")
            return results
        self.status("START", "finish_fixed_pilot")
        remaining_hours = max(0.1, (self.deadline - time.monotonic()) / 3600)
        command = [sys.executable, "-u", str(ROOT / "scripts/run_collision_cost_ablation.py"),
                   "--max-hours", str(min(4, remaining_hours)), "--runtime-recovery",
                   str(previous / "runtime_recovery_20261006.json")]
        with (self.out / "finish_fixed_pilot.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if self.check_health():
                        (previous / "STOP").touch()
                    time.sleep(2)
            except BaseException:
                (previous / "STOP").touch()
                child.wait(timeout=120)
                self.status("FAIL", "finish_fixed_pilot (controller exception)")
                raise
        results = read_json(previous / "results.json")
        completed = child.returncode == 0 and results and results.get("completed")
        self.status("OK" if completed else "FAIL", f"finish_fixed_pilot (exit {child.returncode})")
        if not completed:
            raise RuntimeError("Original fixed-budget ablation did not finish; "
                               "continuation refused")
        return results

    def measure(self, name, checkpoint, task, seed, episodes):
        dest = self.out / "evaluations" / name
        request = {"checkpoint_sha256": digest(checkpoint), "task": task,
                   "seed": seed, "episodes": episodes, "reward": CANONICAL_REWARD,
                   "shield": False}
        saved = read_json(dest / "evaluation_request.json")
        if saved and saved != request:
            raise ValueError(f"Evaluation provenance changed: {dest}")
        if saved is None and (dest / "evaluation_metrics.json").exists():
            raise ValueError(f"Evaluation provenance missing: {dest}")
        write_json(dest / "evaluation_request.json", request)
        self.args.eval_seed = seed
        summary = self.evaluate(name, "cnn", checkpoint, f"carla_right_turn_{task}", episodes)
        if summary is None:
            raise RuntimeError(f"Evaluation interrupted or failed: {name}")
        records = read_json(dest / "evaluation_metrics.json")["episodes"]
        if len(records) != episodes:
            raise ValueError(f"Incomplete evaluation: {name}")
        return {"summary": summary,
                "collision_free_arrivals": sum(
                    e["success"] and not e["collision"] for e in records),
                "arrival_collision_overlap": sum(e["success"] and e["collision"] for e in records),
                "timeouts": sum(e["time_exceeded"] for e in records)}

    def choose_source(self, pilots):
        choices = {}
        for variant in ("reference", "control", "fixed300"):
            source_dir = ROOT / ("outputs/diagnostics/bev_traffic_medium_seed42"
                                 if variant == "reference" else
                                 f"outputs/diagnostics/collision_cost_ablation_v1/{variant}")
            checkpoint = Path(pilots[variant]["checkpoint"])
            # Reference uses an archived source; ensure its live replay checkpoint agrees.
            if digest(source_dir / "latest.pt") != digest(checkpoint):
                raise ValueError(f"Assessed weights and resumable state differ: {source_dir}")
            measurements = dict(pilots[variant]["evaluations"])
            measurements["medium_seed2028"] = self.measure(
                f"choose_{variant}_medium_seed2028", checkpoint, "medium", 2028, 30)
            choices[variant] = {"checkpoint": str(checkpoint), "source_dir": str(source_dir),
                                "evaluations": measurements, "rank": rank(measurements)}
        eligible = [v for v in choices if retained(choices[v]["evaluations"])]
        if not eligible:
            raise RuntimeError("No candidate retained safe turning; further training refused")
        # Stable order retains the reference on exact ties.
        winner = max(eligible, key=lambda v: rank(choices[v]["evaluations"]))
        return {"winner": winner, "choices": choices}

    def train_block(self, block, target, spec):
        name, logdir = f"continue_block{block}", self.args.run_dir
        current, actual_spec = checkpoint_step(logdir / "latest.pt")
        if actual_spec != spec:
            raise ValueError("Continuation checkpoint changed its run contract")
        if current >= target:
            self.status("DONE", name)
            return
        overrides = " ".join(f"--env-set {v}" for v in spec["env_overrides"])
        for attempt in range(1, 5):
            env = self.environment(f"{name}_try{attempt}")
            env.update(RUN_MODE="cnn", TASK=spec["task"], SEED=str(spec["seed"]),
                       OBS="bev", ACTION="discrete", STEPS=str(target),
                       TRAIN_ARGS=f"--resume --logdir {logdir} --set train_ratio=512 "
                       f"--set eval_every=2500 --early-stop-evals 0 {overrides}")
            if self.run(name, env):
                if checkpoint_step(logdir / "latest.pt")[0] != target:
                    raise RuntimeError(f"Training exited without reaching {target}")
                return
            if self.stop.exists():
                raise RuntimeError(f"Training stopped: {name}")
            shutil.copyfile(self.out / f"{name}.log", self.out / f"{name}.try{attempt}.log")
            if attempt < 4:
                self.status("RETRY", f"{name} attempt {attempt + 1}/4")
                time.sleep(10)
        raise RuntimeError(f"Training exhausted retries: {name}")

    def execute(self):
        self.status("=== overnight BEV improvement started ===")
        results = read_json(self.out / "results.json") or {"completed": False, "blocks": []}
        results.pop("error", None)
        results["completed"] = False
        write_json(self.out / "results.json", results)
        try:
            self.prepare()
            pilots = self.finish_pilot()
            choice = read_json(self.out / "source_selection.json")
            if choice is None:
                choice = self.choose_source(pilots)
                write_json(self.out / "source_selection.json", choice)
            else:
                for variant in choice["choices"]:
                    self.status("DONE", f"choose_{variant}_medium_seed2028")
            selected = choice["choices"][choice["winner"]]
            source_dir = Path(selected["source_dir"])
            baseline = self.out / "baseline.pt"
            if not baseline.exists():
                shutil.copyfile(selected["checkpoint"], baseline)
            if not self.args.run_dir.exists():
                clone_training_state(source_dir, self.args.run_dir)
            base_step, spec = checkpoint_step(baseline)
            stages = make_steps(self.args)
            for stage in stages:
                if stage["name"].startswith("continue_block"):
                    stage["start"] += base_step
                    stage["target"] += base_step
            write_json(self.out / "steps.json", stages)
            best = self.out / "best.pt"
            if not best.exists():
                shutil.copyfile(baseline, best)
            best_record = read_json(self.out / "best_validation.json") or {
                "checkpoint": str(best), "step": base_step,
                "evaluations": selected["evaluations"], "blocks_without_gain": 0}
            write_json(self.out / "best_validation.json", best_record)
            results["selected_source"] = choice["winner"]
            results["baseline_step"] = base_step
            write_json(self.out / "results.json", results)
            self.status("SELECT", f"{choice['winner']} step={base_step} "
                        f"validation_rank={rank(selected['evaluations'])}; "
                        "replay/optimizers preserved")
            self.journal("decision", "Selected overnight continuation source",
                         f"Completed the 30k pilot comparison and selected {choice['winner']} "
                         f"at step{base_step}, validation rank {rank(selected['evaluations'])}. "
                         "Cloned replay and optimizer state; original checkpoints preserved.")
            for block in range(1, self.args.blocks + 1):
                if any(b["block"] == block for b in results["blocks"]):
                    for stage in stages:
                        if (stage["name"] == f"continue_block{block}"
                                or stage["name"].startswith(f"block{block}_")):
                            self.status("DONE", stage["name"])
                    continue
                if self.check_health():
                    raise RuntimeError("STOP before next training block")
                # Reserve an hour for checks/retries rather than start work past the cap.
                if self.deadline - time.monotonic() < self.args.block_steps / 6 + 3600:
                    self.status("GATE", "remaining time reserved for final evaluation")
                    break
                target = base_step + block * self.args.block_steps
                self.train_block(block, target, spec)
                snapshot = self.out / "checkpoints" / f"step{target}.pt"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                if not snapshot.exists():
                    shutil.copyfile(self.args.run_dir / "latest.pt", snapshot)
                measurements = {f"medium_seed{s}": self.measure(
                    f"block{block}_medium_seed{s}", snapshot, "medium", s, 30)
                                for s in VALIDATION_SEEDS}
                measurements["simple_seed2029"] = self.measure(
                    f"block{block}_simple_seed2029", snapshot, "simple", 2029, 10)
                accepted = improvement(measurements, best_record["evaluations"])
                if accepted:
                    temp = best.with_suffix(".pt.tmp")
                    shutil.copyfile(snapshot, temp)
                    temp.replace(best)
                    best_record = {"checkpoint": str(best), "step": target,
                                   "evaluations": measurements, "blocks_without_gain": 0}
                else:
                    best_record["blocks_without_gain"] += 1
                write_json(self.out / "best_validation.json", best_record)
                results["blocks"].append({"block": block, "step": target,
                                          "checkpoint": str(snapshot), "accepted": accepted,
                                          "evaluations": measurements})
                write_json(self.out / "results.json", results)
                self.status("GATE", f"block{block} rank={rank(measurements)} accepted={accepted} "
                            f"best_step={best_record['step']}")
                self.journal("result", f"Overnight BEV block{block} completed",
                             f"Reached step{target}; medium safety rank {rank(measurements)}, "
                             "simple safe arrivals "
                             f"{safe_count(measurements['simple_seed2029'])}/10. "
                             f"Accepted={accepted}; best checkpoint remains "
                             f"step{best_record['step']}. "
                             "These streams are checkpoint-selection diagnostics, not holdouts.")
                if (best_record["blocks_without_gain"] >= 3
                        or safe_count(measurements["simple_seed2029"]) < 8
                        or rank(measurements)[0] < 15):
                    self.status("GATE", "stagnation or serious regression; "
                                "retain best and evaluate")
                    break
            for candidate, checkpoint in (("baseline", baseline), ("selected", best)):
                results[f"holdout_{candidate}"] = {}
                for task, seed, episodes in (
                        *(("medium", s, 30) for s in HOLDOUT_SEEDS),
                        ("simple", 3031, 10), ("hard", 5031, 30)):
                    name = f"holdout_{candidate}_{task}_seed{seed}"
                    results[f"holdout_{candidate}"][f"{task}_seed{seed}"] = self.measure(
                        name, checkpoint, task, seed, episodes)
                    write_json(self.out / "results.json", results)
            results["completed"] = True
            results["best_checkpoint"] = str(best)
            results["best_step"] = best_record["step"]
            self.write_summary(results)
            self.journal("result", "Overnight BEV run completed",
                         f"Finished {len(results['blocks'])} blocks; "
                         f"selected step{best_record['step']}. "
                         f"Baseline and selected canonical holdout results: {self.out}/summary.md. "
                         "Medium streams3031/4031, simple3031 and zero-shot hard5031 were not used "
                         "for training checkpoint selection; no hard-task training occurred.")
        except BaseException as error:
            results["error"] = f"{type(error).__name__}: {error}"
            self.status("ERROR", results["error"])
            self.journal("blocker", "Overnight BEV queue interrupted",
                         f"{results['error']}. "
                         "Saved state, STOP reason and completed checks remain "
                         f"under {self.out}; completed=False, unfinished work is not a result.")
            raise
        finally:
            write_json(self.out / "results.json", results)
            current_status = (self.out / "status.txt").read_text().rsplit(
                "=== overnight BEV improvement started ===", 1)[-1]
            for stage in make_steps(self.args):
                if not any(f" {state} {stage['name']} " in current_status + " "
                           or f" {state} {stage['name']}\n" in current_status
                           for state in ("OK", "DONE", "FAIL", "SKIP")):
                    self.status("SKIP", f"{stage['name']} (not completed; gate, error or STOP)")
            self.status("=== overnight BEV improvement finished ===")

    def prepare(self):
        files = ("scripts/run_overnight_bev_improvement.py", "scripts/train_cardreamer.py",
                 "scripts/evaluate_agent.py", "scripts/run_traffic_curriculum.py",
                 "scripts/run_collision_cost_ablation.py", "src/dreamer/carla_wrappers.py",
                 "configs/laptop.yaml", "configs/experiment_contract.yaml",
                 "jobs/slurm_run.sh", "patches/cardreamer_compat.patch")
        protocol = {"additional_steps": self.args.block_steps * self.args.blocks,
                    "block_steps": self.args.block_steps, "max_hours": self.args.max_hours,
                    "validation_seeds": VALIDATION_SEEDS, "holdout_seeds": HOLDOUT_SEEDS,
                    "hard_holdout_seed": 5031, "train_ratio": 512, "eval_every": 2500,
                    "fresh_prefill": False, "canonical_reward": CANONICAL_REWARD,
                    "code_sha256": {rel: digest(ROOT / rel) for rel in files}}
        # Normalize tuples to JSON lists before comparing resumable provenance.
        protocol = json.loads(json.dumps(protocol))
        saved = read_json(self.out / "experiment.json")
        if saved is not None and saved != protocol:
            raise ValueError("Overnight protocol changed; use a separate queue directory")
        write_json(self.out / "experiment.json", protocol)
        for rel in files:
            dest = self.out / "source_snapshot" / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, dest)

    def journal(self, kind, title, happened):
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        entry = (f"\n## {timestamp} - {title}\n**Type:** {kind}\n"
                 "**Stage:** Approved overnight BEV continuation\n\n"
                 f"**What happened:** {happened}\n"
                 "**Cause:** Stage outcome during approved continuation; "
                 "any newly reported exception needs a separate root-cause assessment.\n"
                 f"**How we handled it:** See {self.out}/results.json and status.txt; "
                 "canonical reward evaluation, original source preserved.\n"
                 "**Verified by:** Standalone episode JSONs, frozen checkpoints and queue logs.\n"
                 "**Lesson:** Preserve the best validated state when continuing "
                 "an imperfect policy.\n"
                 "**Interview angle:** How did the overnight continuation handle this stage?\n")
        with (ROOT / "reports/journey/journal.md").open("a") as stream:
            stream.write(entry)
        with (ROOT / "reports/journey/interview_questions.md").open("a") as stream:
            stream.write(f"\n- **Q:** How did the overnight continuation handle {title.lower()}?\n"
                         f"  **A (short):** {happened}\n"
                         f"  **Evidence:** journal {timestamp}; {self.out}/results.json\n")

    def write_summary(self, results):
        lines = ["# Overnight BEV continuation results", "",
                 f"Source: {results['selected_source']}; "
                 f"{len(results['blocks'])} completed blocks; "
                 f"validation-selected checkpoint step{results['best_step']}.", "",
                 "Canonical reward, no shield. One training seed; BEV diagnostic, "
                 "not an encoder comparison. Holdouts were not used for checkpoint selection.", "",
                 "| Policy | Evaluation | Safe arrivals | Collisions | Lane failures |",
                 "|---|---|---:|---:|---:|"]
        for candidate in ("baseline", "selected"):
            for name, measurement in results[f"holdout_{candidate}"].items():
                summary = measurement["summary"]
                count = summary["num_episodes"]
                lines.append(f"| {candidate} | {name} | {safe_count(measurement)}/{count} "
                             f"| {round(summary['collision_rate'] * count)}/{count} "
                             f"| {round(summary['out_of_lane_rate'] * count)}/{count} |")
        (self.out / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/overnight_bev_20261006"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/diagnostics/overnight_bev_20261006"))
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-steps", type=int, default=30000)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.out, args.run_dir = args.out.resolve(), args.run_dir.resolve()
    if args.blocks < 1 or args.block_steps < 2500 or args.max_hours <= 0:
        parser.error("Require positive budgets and block-steps >=2500")
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(make_steps(args), indent=2))
        return
    # Independent global preflight; anchored patterns cannot match this shell's text.
    busy = subprocess.run(["pgrep", "-af", r"^bash jobs/|^[^ ]*python[0-9.]* (-u )?[^ ]*scripts/"
                           r"(train_|evaluate_|probe_|run_bev_traffic_diagnostic|run_traffic_curriculum|"
                           r"run_collision_cost_ablation|run_overnight_bev_improvement)|CarlaUE4-Linux"],
                          capture_output=True, text=True)
    others = [line for line in busy.stdout.splitlines()
              if int(line.split()[0]) != os.getpid() and "defunct" not in line]
    if others:
        raise SystemExit("Existing CARLA job/controller; refusing duplicate launch:\n"
                         + "\n".join(others))
    queue = Overnight(args, steps=make_steps(args))
    signal.signal(signal.SIGTERM, lambda *_: queue.request_stop("controller SIGTERM"))
    queue.execute()


if __name__ == "__main__":
    main()
