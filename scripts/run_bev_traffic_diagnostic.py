#!/usr/bin/env python3
"""One bounded CNN BEV medium-traffic diagnostic, with automatic evaluation."""

import argparse
import json
import signal
import time
from pathlib import Path

if __package__:
    from .run_traffic_curriculum import Queue, promotion_gate, read_json, write_json
else:
    from run_traffic_curriculum import Queue, promotion_gate, read_json, write_json


def plan(args):
    return [
        {"name": "baseline_cnn_bev_medium", "hours": 0.06},
        {"name": "cnn_bev_medium_seed42", "hours": args.steps / 21000,
         "logdir": str(args.run_dir), "target": args.steps},
        {"name": "validate_cnn_bev_medium", "hours": 0.06},
        {"name": "retention_cnn_bev_medium", "hours": 0.04},
    ]


class Diagnostic(Queue):
    def train_bev(self):
        args = self.args
        name = "cnn_bev_medium_seed42"
        logdir = args.run_dir
        completed = read_json(logdir / "metrics.json")
        if completed and (completed.get("env_steps", 0) >= args.steps
                          or completed.get("early_stopped_at") is not None):
            self.status("DONE", name)
            return logdir / "best.pt"
        for attempt in range(1, 5):
            resume = ("--resume" if (logdir / "latest.pt").is_file()
                      else f"--init-from {args.init_from}")
            env = self.environment(f"{name}_try{attempt}")
            env.update(RUN_MODE="cnn", TASK="carla_right_turn_medium", SEED="42",
                       OBS="bev", ACTION="discrete", STEPS=str(args.steps),
                       TRAIN_ARGS=f"{resume} --logdir {logdir} --set train_ratio=512 "
                       "--set eval_every=2500 --env-set reward.scales.time=0.1 "
                       "--early-stop-evals 3 --early-stop-success 0.9")
            if self.run(name, env, capture_dir=logdir):
                return logdir / "best.pt"
            if self.stop.exists():
                return None
            (self.out / f"{name}.log").rename(self.out / f"{name}.try{attempt}.log")
            if attempt < 4:
                self.status("RETRY", f"{name} attempt {attempt + 1}/4")
                time.sleep(10)
        return None

    def execute(self):
        self.status("=== BEV traffic diagnostic started ===")
        result = {"diagnostic": "cnn_bev_medium", "completed": False,
                  "init_from": str(self.args.init_from), "hard_training": False}
        try:
            while not self.idle():
                if self.check_health():
                    return
                print("Waiting for existing CARLA job to finish...", flush=True)
                time.sleep(10)
            baseline = self.evaluate("baseline_cnn_bev_medium", "cnn", self.args.init_from,
                                     "carla_right_turn_medium", 30)
            result["baseline_medium"] = baseline
            if baseline is None:
                return
            selected = self.train_bev()
            if selected is None:
                return
            result["selected_checkpoint"] = str(selected)
            traffic = self.evaluate("validate_cnn_bev_medium", "cnn", selected,
                                    "carla_right_turn_medium", 30)
            retention = self.evaluate("retention_cnn_bev_medium", "cnn", selected,
                                      "carla_right_turn_simple", 10)
            result.update(medium=traffic, simple_retention=retention,
                          completed=bool(traffic and retention),
                          gate_passed=bool(traffic and retention
                                           and promotion_gate(traffic, retention)))
        finally:
            write_json(self.out / "results.json", result)
            # Mark unattempted stages explicitly when stopped or blocked.
            status = (self.out / "status.txt").read_text()
            for step in plan(self.args):
                if not any(f" {state} {step['name']}" in status
                           for state in ("START", "DONE", "OK", "FAIL", "SKIP")):
                    self.status("SKIP", f"{step['name']} (earlier stage failed or STOP)")
            self.status("=== BEV traffic diagnostic finished ===")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/bev_traffic_diagnostic_queue"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/diagnostics/bev_traffic_medium_seed42"))
    parser.add_argument("--init-from", type=Path,
                        default=Path("outputs/logs/cnn_bev_tp01_seed42/latest.pt"))
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--max-hours", type=float, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for key in ("out", "run_dir", "init_from"):
        setattr(args, key, getattr(args, key).resolve())
    if args.steps <= 2500 or args.max_hours <= 0:
        parser.error("steps must exceed prefill (2500), and max-hours must be positive")
    if not args.init_from.is_file():
        parser.error(f"Missing checkpoint: {args.init_from}")
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return
    queue = Diagnostic(args, steps=plan(args))
    signal.signal(signal.SIGTERM, lambda *_: queue.request_stop("controller SIGTERM"))
    queue.execute()


if __name__ == "__main__":
    main()
