#!/usr/bin/env python3
"""Bounded overnight traffic curriculum with validation, retention and STOP gates.

Uses the existing trainer and CARLA launcher. Rewards and the final hard task
are unchanged. Progress is readable with watch_queue.py --out <queue dir>.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("custom_jepa", "vjepa2", "cnn")
BUSY_PATTERN = (
    r"^bash jobs/|^[^ ]*python[0-9.]* [^ ]*scripts/(train_|evaluate_|probe_)"
    r"|CarlaUE4-Linux"
)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def safe_score(summary):
    return (summary["success_rate"], -summary["collision_rate"], summary["mean_reward"])


def promotion_gate(traffic, retention):
    """Separate validation from noisy training evals; require retained turning skill."""
    return (traffic["success_rate"] >= 0.8 and traffic["collision_rate"] <= 0.2
            and retention["success_rate"] >= 0.9 and retention["collision_rate"] <= 0.1)


def capture_best(logdir):
    """Preserve weights at the best completed training eval, before latest advances."""
    path = logdir / "eval.jsonl"
    latest = logdir / "latest.pt"
    if not path.exists() or not latest.exists():
        return
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # an append may still be in progress
    if not records:
        return
    last = records[-1]
    selected = read_json(logdir / "best_eval.json")
    if selected and safe_score(last) <= safe_score(selected):
        return
    temp = logdir / "best.pt.tmp"
    shutil.copyfile(latest, temp)
    temp.replace(logdir / "best.pt")
    write_json(logdir / "best_eval.json", last)


def make_steps(args):
    steps = []
    for arm in args.arms:
        steps.append({"name": f"baseline_{arm}_medium", "hours": 0.05})
        for stage, target in (("medium", args.medium_steps), ("hard", args.hard_steps)):
            steps.append({"name": f"{arm}_{stage}_seed{args.seed}", "target": target,
                          "logdir": str(args.run_dir / stage / f"{arm}_seed{args.seed}"),
                          "hours": target / {"cnn": 21000, "custom_jepa": 21000,
                                             "vjepa2": 17500}[arm]})
            steps.extend({"name": f"{kind}_{arm}_{stage}", "hours": 0.06}
                         for kind in ("validate", "retention"))
    return steps


class Queue:
    def __init__(self, args, steps=None):
        self.args = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        # Ownership comes before changing STOP or watcher metadata.
        self.lock = (self.out / ".queue.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Queue already running: {self.out}") from None
        self.stop = self.out / "STOP"
        self.stop.unlink(missing_ok=True)
        self.deadline = time.monotonic() + args.max_hours * 3600
        self.last_health = 0
        self.hot_since = None
        write_json(self.out / "steps.json", make_steps(args) if steps is None else steps)
        write_json(self.out / "plan.json", {k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(args).items()})

    def status(self, state, message=""):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {state} {message}".rstrip()
        print(line, flush=True)
        with (self.out / "status.txt").open("a") as f:
            f.write(line + "\n")

    def request_stop(self, reason):
        if not self.stop.exists():
            self.stop.write_text(reason + "\n")

    def check_health(self):
        now = time.monotonic()
        if now >= self.deadline:
            self.request_stop("8-hour/default or configured overnight time budget reached")
        if now - self.last_health >= 30:
            self.last_health = now
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
            fields = gpu.stdout.strip().splitlines()
            values = [v.strip() for v in fields[0].split(",")] if fields else []
            memory = Path("/proc/meminfo").read_text()
            ram = int(re.search(r"MemAvailable:\s+(\d+)", memory).group(1)) / 2**20
            disk = shutil.disk_usage(ROOT).free / 2**30
            # CPU temperature is display-only; GPU/RAM/disk guards below must
            # still run when lm-sensors is unavailable or its read times out.
            try:
                sensors = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10)
                sensors_output = sensors.stdout
            except (subprocess.TimeoutExpired, OSError) as error:
                sensors_output = ""
                with (self.out / "monitor_warnings.log").open("a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} CPU telemetry unavailable: "
                            f"{type(error).__name__}: {error}\n")
            cpu_match = re.search(r"Package id 0:\s+(\S+)", sensors_output)
            cpu = cpu_match.group(1) if cpu_match else "?"
            with (self.out / "health.log").open("a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} gpu[temp,MiB,util,W]="
                        f"{','.join(values)} cpu={cpu} ram_avail={ram:.1f}G "
                        f"disk_free={disk:.0f}G\n")
            if values and values[0].isdigit() and int(values[0]) >= 92:
                self.hot_since = self.hot_since or now
                if now - self.hot_since >= 90:
                    self.request_stop("GPU >= 92 C for 90 seconds")
            else:
                self.hot_since = None
            if ram < 2 or disk < 5:
                self.request_stop(f"low resources: RAM {ram:.1f}G / disk {disk:.1f}G")
        return self.stop.exists()

    def idle(self):
        result = subprocess.run(["pgrep", "-af", BUSY_PATTERN], capture_output=True, text=True)
        return not result.stdout.strip()

    def stop_owned_client(self, launcher_pid, signaled):
        # Scope STOP to our launcher descendants. Never signal unrelated jobs
        # or DataLoader workers; main trainer SIGTERM saves its checkpoint.
        import psutil
        try:
            descendants = psutil.Process(launcher_pid).children(recursive=True)
        except psutil.NoSuchProcess:
            return
        clients = []
        for process in descendants:
            try:
                cmd = process.cmdline()
                if len(cmd) >= 2 and re.search(r"python[0-9.]*$", cmd[0]) and re.search(
                        r"scripts/(train_cardreamer|evaluate_agent)\.py$", cmd[1]):
                    clients.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        ids = {p.pid for p in clients}
        for process in clients:
            try:
                if process.ppid() not in ids and process.pid not in signaled:
                    process.send_signal(signal.SIGTERM)
                    signaled.add(process.pid)
            except psutil.NoSuchProcess:
                pass

    def run(self, name, env, capture_dir=None):
        if self.check_health():
            self.status("SKIP", f"{name} (STOP requested)")
            return False
        self.status("START", name)
        with (self.out / f"{name}.log").open("w") as log:
            child = subprocess.Popen(["bash", "jobs/slurm_run.sh"], cwd=ROOT,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
            stop_sent = False
            signaled = set()
            try:
                while child.poll() is None:
                    if capture_dir:
                        capture_best(capture_dir)
                    if self.check_health() and not stop_sent:
                        self.status("STOP:", self.stop.read_text().strip() or "manual STOP file")
                        self.stop_owned_client(child.pid, signaled)
                        stop_sent = True
                    # During startup there may not yet be a client to signal.
                    if stop_sent:
                        self.stop_owned_client(child.pid, signaled)
                    time.sleep(2)
                if capture_dir:
                    capture_best(capture_dir)
            except BaseException:
                self.request_stop("curriculum controller interrupted or failed")
                self.status("FAIL", f"{name} (controller exception; recovery requested)")
                self.stop_owned_client(child.pid, signaled)
                child.wait(timeout=120)
                raise
        passed = child.returncode == 0
        self.status("OK" if passed else "FAIL", f"{name} (exit {child.returncode})")
        return passed

    def environment(self, name):
        return dict(os.environ, CARLA_ROOT=os.environ.get(
            "CARLA_ROOT", str(Path.home() / "software/CARLA_0.9.15")),
            PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
            CARLA_LOG=str((self.out / f"carla_{name}.log").resolve()))

    def train(self, arm, stage, source):
        args = self.args
        name = f"{arm}_{stage}_seed{args.seed}"
        logdir = args.run_dir / stage / f"{arm}_seed{args.seed}"
        target = args.medium_steps if stage == "medium" else args.hard_steps
        if (logdir / "metrics.json").exists():
            self.status("DONE", name)
            best = logdir / "best.pt"
            return best if best.exists() else logdir / "latest.pt"
        for attempt in range(1, 5):
            resume = "--resume" if (logdir / "latest.pt").exists() else f"--init-from {source}"
            env = self.environment(f"{name}_try{attempt}")
            env.update(RUN_MODE=arm, TASK=f"carla_right_turn_{stage}", SEED=str(args.seed),
                       OBS="camera_route", ACTION="discrete", STEPS=str(target),
                       TRAIN_ARGS=f"{resume} --logdir {logdir} --set train_ratio=512 "
                       "--set eval_every=2500 --env-set reward.scales.time=0.1 "
                       "--early-stop-evals 3 --early-stop-success 0.9")
            if self.run(name, env, capture_dir=logdir):
                best = logdir / "best.pt"
                return best if best.exists() else logdir / "latest.pt"
            if self.stop.exists():
                return None
            (self.out / f"{name}.log").rename(self.out / f"{name}.try{attempt}.log")
            if attempt < 4:
                self.status("RETRY", f"{name} attempt {attempt + 1}/4")
                time.sleep(10)
        return None

    def evaluate(self, name, arm, source, task, episodes):
        dest = self.out / "evaluations" / name
        p = dest / "evaluation_metrics.json"
        if p.exists():
            self.status("DONE", name)
            return read_json(p)["summary"]
        for attempt in range(1, 4):
            env = self.environment(f"{name}_try{attempt}")
            env.update(RUN_MODE="evaluate", ARM=arm, TASK=task, CHECKPOINT=str(source),
                       SEED=str(self.args.eval_seed), EPISODES=str(episodes), OUTPUT_DIR=str(dest),
                       EVAL_ARGS="--no-shield --video-episodes 1 --fps 10")
            if self.run(name, env) and p.exists():
                return read_json(p)["summary"]
            if self.stop.exists():
                return None
            if attempt < 3:
                self.status("RETRY", f"{name} attempt {attempt + 1}/3")
        return None

    def skip_hard(self, arm, reason):
        for name in (f"{arm}_hard_seed{self.args.seed}", f"validate_{arm}_hard",
                     f"retention_{arm}_hard"):
            self.status("SKIP", f"{name} ({reason})")

    def execute(self):
        args = self.args
        self.status("=== overnight traffic curriculum started ===")
        # Existing short evaluations may outlive the chat turn. Wait, never
        # start another CARLA instance or stop someone else's process.
        while not self.idle():
            if self.check_health():
                self.status("=== overnight traffic curriculum finished ===")
                return
            print("Waiting for existing CARLA job/evaluation to finish...", flush=True)
            time.sleep(10)
        results = read_json(self.out / "results.json") or {}
        for arm in args.arms:
            if self.check_health():
                break
            source = args.init_runs / f"{arm}_seed{args.seed}" / "latest.pt"
            if not source.is_file():
                raise FileNotFoundError(source)
            baseline = self.evaluate(f"baseline_{arm}_medium", arm, source,
                                     "carla_right_turn_medium", 20)
            if baseline is None:
                if self.stop.exists():
                    break
                self.status("FAIL", f"{arm}_medium_seed{args.seed} (baseline evaluation failed)")
                self.skip_hard(arm, "baseline evaluation failed")
                continue
            selected = self.train(arm, "medium", source)
            if selected is None:
                self.skip_hard(arm, "medium training failed or stopped")
                continue
            traffic = self.evaluate(f"validate_{arm}_medium", arm, selected,
                                    "carla_right_turn_medium", 30)
            retention = self.evaluate(f"retention_{arm}_medium", arm, selected,
                                      "carla_right_turn_simple", 10)
            promoted = bool(traffic and retention and promotion_gate(traffic, retention))
            results[arm] = {"baseline_medium": baseline, "medium": traffic,
                            "simple_after_medium": retention, "promoted": promoted,
                            "medium_checkpoint": str(selected)}
            write_json(self.out / "results.json", results)
            if not promoted:
                self.skip_hard(arm, "medium validation/retention gate not passed")
                continue
            selected = self.train(arm, "hard", selected)
            if selected:
                results[arm]["hard_checkpoint"] = str(selected)
                results[arm]["hard"] = self.evaluate(f"validate_{arm}_hard", arm, selected,
                                                     "carla_right_turn_hard", 30)
                results[arm]["simple_after_hard"] = self.evaluate(
                    f"retention_{arm}_hard", arm, selected, "carla_right_turn_simple", 10)
                write_json(self.out / "results.json", results)
        self.status("=== overnight traffic curriculum finished ===")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/overnight_traffic_queue"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/comparison/traffic_curriculum_v1"))
    parser.add_argument("--init-runs", type=Path,
                        default=Path("outputs/comparison/carla_right_turn_simple_camera_route"))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--medium-steps", type=int, default=30000)
    parser.add_argument("--hard-steps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("out", "run_dir", "init_runs"):
        setattr(args, key, getattr(args, key).resolve())
    if min(args.medium_steps, args.hard_steps) <= 2500 or args.max_hours <= 0:
        parser.error("step budgets must exceed prefill (2500), and max-hours must be positive")
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(make_steps(args), indent=2))
        return
    queue = Queue(args)
    signal.signal(signal.SIGTERM, lambda *_: queue.request_stop("controller SIGTERM"))
    queue.execute()


if __name__ == "__main__":
    main()
