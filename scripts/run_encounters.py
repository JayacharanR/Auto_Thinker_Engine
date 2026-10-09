#!/usr/bin/env python3
"""Run the frozen encounter catalog in bounded, resumable chunks (one owned server each).

Each chunk is an ordinary run_navigation.py attempt over an evaluator route subset,
so all M0 provenance, STOP handling and process ownership apply unchanged.
Predeclared retry rule: a route with no evaluator record, or whose record reports
a simulator crash, is rerun once; both attempts are kept and reported. Agent
failures, collisions and timeouts are results and are never retried.
"""

import argparse
import fcntl
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.metrics import summarize_records, wilson_interval  # noqa: E402
from src.navigation.runtime import atomic_json, resource_shortfall  # noqa: E402

CRASH_STATUS = "Simulation crashed"


def verify_catalog(catalog_dir):
    expected = (catalog_dir / "catalog.sha256").read_text().split()[0]
    if hashlib.sha256((catalog_dir / "catalog.json").read_bytes()).hexdigest() != expected:
        raise ValueError("catalog.json differs from its frozen hash")
    return json.loads((catalog_dir / "catalog.json").read_text()), expected


def route_id(record_route):
    match = re.fullmatch(r"RouteScenario_(\d+)_rep\d+", record_route or "")
    if not match:
        raise ValueError(f"Unexpected evaluator route id {record_route!r}")
    return int(match.group(1))


def chunk_records(chunk):
    """Records of one attempt, and whether a user STOP truncated it."""
    path, raw = chunk / "summary.json", chunk / "leaderboard.json"
    if path.exists():
        episodes = json.loads(path.read_text())["episodes"]
    elif raw.exists():
        # A native abort skips summary.json; routes the evaluator already recorded
        # are results and must not be rerun, so read its raw records directly.
        raw_records = json.loads(raw.read_text()).get("_checkpoint", {}).get("records", [])
        episodes = summarize_records(raw_records)["episodes"]
    else:
        episodes = []
    records = {route_id(e["route_id"]): e for e in episodes}
    stopped = (chunk / "STOP").exists() or (
        (chunk / "stopped.json").exists()
        and json.loads((chunk / "stopped.json").read_text()).get("reason") == "user_stop"
    )
    if stopped and records:
        # The route running when STOP arrived is interrupted, not a driving result.
        records.pop(list(records)[-1])
    # An attempt killed from outside (e.g. host memory pressure) is annotated with
    # interrupted.json. Its running route never got a record, so recorded routes are
    # kept, and the unrecorded ones are reattempted without spending their retry.
    return records, stopped or (chunk / "interrupted.json").exists()


def settle_group(args, xml, group, base, slot=0, gate=None):
    """Attempt dirs base_a0, base_a1, ...; returns per-route attempts and final record."""
    attempts, finals, failures = {r: [] for r in group}, {}, Counter()
    k = not_started = 0
    while True:
        chunk = base.with_name(f"{base.name}_a{k}")
        routes_file = chunk.with_name(chunk.name + ".routes.json")
        if not routes_file.exists():
            todo = [r for r in group if r not in finals and failures[r] < 2]
            if not todo or (args.out / "STOP").exists() or not_started >= 3:
                return attempts, finals
            subset = f"{todo[0]}-{todo[-1]}" if todo == group else ",".join(map(str, todo))
            atomic_json(routes_file, todo)  # run_navigation requires an empty attempt dir
            run_chunk(args, xml, subset, chunk, slot, gate)
        asked = json.loads(routes_file.read_text())
        if not (chunk / "manifest.json").exists():
            # Refused before any server or route started (resources, lock, STOP):
            # an infrastructure non-event, not a failed driving attempt.
            for r in asked:
                attempts[r].append({"attempt": chunk.name, "status": "not started"})
            not_started += 1
            k += 1
            continue
        records, stopped = chunk_records(chunk)
        for r in asked:
            if r in records:
                attempts[r].append({"attempt": chunk.name, **records[r]})
                if CRASH_STATUS in records[r]["status"]:
                    failures[r] += 1
                else:
                    finals[r] = records[r]
            elif not stopped:
                attempts[r].append({"attempt": chunk.name, "status": "no evaluator record"})
                failures[r] += 1
        k += 1


class LaunchGate:
    """Serializes server starts and waits for resources, so parallel slots never
    race each other's memory. A start is complete only once its chunk is driving:
    the server boots on its default map and loads the requested (possibly much
    larger) map afterwards, so earlier measurements understate its footprint."""

    def __init__(self, runtime, stop_path):
        self.runtime, self.stop_path, self.lock = runtime, stop_path, threading.Lock()
        self.running, self.running_lock = {}, threading.Lock()  # chunk -> start time

    def launch(self, command, chunk, log):
        with self.lock:
            while (shortfall := resource_shortfall(self.runtime)) and not self.stop_path.exists():
                atomic_json(chunk.with_name(chunk.name + ".waiting.json"), {"reason": shortfall})
                time.sleep(10)
            if self.stop_path.exists():
                return None
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with self.running_lock:
                self.running[chunk] = time.monotonic()
            deadline = time.monotonic() + 300
            while (
                child.poll() is None
                and not any(p.stat().st_size for p in chunk.glob("episodes/*/trace.jsonl"))
                and time.monotonic() < deadline
            ):
                time.sleep(2)
            return child

    def finished(self, chunk):
        with self.running_lock:
            self.running.pop(chunk, None)

    def newest(self):
        with self.running_lock:
            return max(self.running, key=self.running.get) if self.running else None


def memory_watchdog(gate, critical_gb, events, state, done):
    """Pause (STOP) the newest running chunk when available RAM is critical. Its
    interrupted route is dropped and reattempted later through the gate."""
    last = 0.0
    while not done.wait(5):
        available = psutil.virtual_memory().available / 1e9
        chunk = gate.newest()
        if available >= critical_gb or time.monotonic() - last < 60 or chunk is None:
            continue
        (chunk / "STOP").touch()
        atomic_json(chunk / "memory_pause.json", {"available_gb": available, "at": time.time()})
        with state:
            events.append({"chunk": chunk.name, "available_gb": round(available, 2)})
        last = time.monotonic()


def run_chunk(args, xml, subset, chunk, slot=0, gate=None):
    command = [
        sys.executable, "-u", str(ROOT / "scripts/run_navigation.py"), "expert",
        "--route", str(xml), "--routes-subset", subset, "--seed", str(args.seed),
        "--seed-per-route", "--out", str(chunk), "--max-wall", str(args.chunk_wall),
        "--slot", str(slot),
    ]  # fmt: skip
    with (chunk.parent / f"{chunk.name}_console.log").open("a") as log:
        if gate is None:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        else:
            child = gate.launch(command, chunk, log)
            if child is None:
                return None
        while child.poll() is None:
            if (args.out / "STOP").exists() and chunk.exists():
                (chunk / "STOP").touch()  # the chunk owner stops its own evaluator/server
            time.sleep(2)
    if gate is not None:
        gate.finished(chunk)
    return child.returncode


def aggregate(catalog, finals, attempts, towns):
    by_id = {e["route_id"]: e for e in catalog["encounters"] if e["town"] in towns}
    families = {}
    for rid, entry in by_id.items():
        f = families.setdefault(
            entry["family"],
            Counter(encounters=0, recorded=0, completed=0, rule_safe=0, strict_safe=0,
                    collisions=0, red=0, stop=0, min_speed=0, execution_missing=0),
        )  # fmt: skip
        f["encounters"] += 1
        e = finals.get(rid)
        if e is None:
            f["execution_missing"] += 1
            continue
        f["recorded"] += 1
        f["completed"] += e["completed"]
        f["rule_safe"] += e["rule_safe_completed"]
        f["strict_safe"] += e["safe_completed"]
        f["collisions"] += e["collisions"]
        f["red"] += e["red_violations"]
        f["stop"] += e["stop_violations"]
        f["min_speed"] += e["min_speed_infractions"]
    report = {}
    for family, f in families.items():
        n = f["encounters"]
        report[family] = {
            **f,
            # Missing executions count as failures in the interval, never as removed trials.
            "rule_safe_wilson95_unclustered": wilson_interval(f["rule_safe"], n),
        }
    return {
        "families": report,
        "all_collision_events": sum(f["collisions"] for f in families.values()),
        "retried_routes": sorted(rid for rid, a in attempts.items() if len(a) > 1),
        "unsettled_routes": sorted(set(by_id) - set(finals)),
        "attempts": {str(rid): a for rid, a in sorted(attempts.items())},
        "note": "Wilson intervals assume independent encounters; final analysis must "
        "cluster by junction/stop location (see catalog location_reuse_rule).",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/encounters/dev_v1")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42, help="Traffic seed = seed + route id")
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--chunk-wall", type=float, default=1800)
    parser.add_argument("--towns", nargs="*")
    parser.add_argument("--workers", type=int, default=1, help="Parallel server slots")
    parser.add_argument("--config", default="configs/navigation.yaml")
    parser.add_argument(
        "--min-ram-gb", type=float, default=12.0, help="Available RAM needed to start a server"
    )
    parser.add_argument(
        "--critical-ram-gb", type=float, default=6.0, help="Pause the newest chunk below this"
    )
    args = parser.parse_args()
    runtime = yaml.safe_load((ROOT / args.config).read_text())["runtime"]
    if not 1 <= args.workers <= runtime["max_servers"]:
        raise ValueError(f"--workers must be within 1..{runtime['max_servers']}")
    catalog_dir = (ROOT / args.catalog).resolve()
    catalog, digest = verify_catalog(catalog_dir)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".runner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Another encounter runner owns {args.out}") from None
    # Worker count changes wall time, never chunk composition, seeds or results.
    request = {"catalog": str(catalog_dir), "catalog_sha256": digest, "seed": args.seed,
               "chunk": args.chunk, "policy_kind": "privileged_pdm_lite_modified"}  # fmt: skip
    previous = args.out / "request.json"
    if previous.exists() and json.loads(previous.read_text()) != request:
        raise ValueError("Output directory belongs to a different catalog/seed/chunk request")
    atomic_json(previous, request)
    with (args.out / "invocations.jsonl").open("a") as log:
        log.write(json.dumps({"started_at": time.time(), "workers": args.workers,
                              "min_ram_gb": args.min_ram_gb,
                              "critical_ram_gb": args.critical_ram_gb}) + "\n")  # fmt: skip
    towns = args.towns or list(catalog["towns"])
    total = sum(catalog["towns"][t]["count"] for t in towns)
    finals, attempts, errors, current = {}, {}, [], {}
    state = threading.Lock()
    stop_path = args.out / "STOP"

    def progress(status="running"):
        done = len(finals)
        atomic_json(
            args.out / "progress.json",
            {"status": status, "stage": "encounter expert ceiling", "route_progress_percent":
             100 * done / total, "frames": done, "target_frames": 0, "encounters_done": done,
             "encounters_total": total, "current_chunk": ", ".join(sorted(current.values()))
             or None, "workers": args.workers, "errors": errors, "memory_pauses":
             memory_events, "updated_at": time.time()},
        )  # fmt: skip

    work = queue.Queue()
    for town in towns:
        xml = catalog_dir / f"{town}.xml"
        ids = sorted(e["route_id"] for e in catalog["encounters"] if e["town"] == town)
        for start in range(0, len(ids), args.chunk):
            group = ids[start : start + args.chunk]
            base = args.out / town / f"chunk_{group[0]:03d}_{group[-1]:03d}"
            base.parent.mkdir(parents=True, exist_ok=True)
            work.put((xml, group, base))
    gate = LaunchGate({**runtime, "minimum_available_ram_gb": args.min_ram_gb}, stop_path)
    memory_events, done = [], threading.Event()
    watchdog = threading.Thread(
        target=memory_watchdog,
        args=(gate, args.critical_ram_gb, memory_events, state, done),
        daemon=True,
    )
    watchdog.start()

    def slot_worker(slot):
        while not stop_path.exists():
            try:
                xml, group, base = work.get_nowait()
            except queue.Empty:
                return
            with state:
                current[slot] = base.name
                progress()
            try:
                group_attempts, group_finals = settle_group(args, xml, group, base, slot, gate)
            except Exception as exc:  # preserve; other slots continue
                with state:
                    errors.append({"chunk": base.name, "error": f"{type(exc).__name__}: {exc}"})
                    current.pop(slot, None)
                    progress()
                return
            with state:
                attempts.update(group_attempts)
                finals.update(group_finals)
                current.pop(slot, None)
                progress()

    threads = [threading.Thread(target=slot_worker, args=(k,)) for k in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done.set()
    if stop_path.exists():
        status = "stopped"
    elif errors or not work.empty():
        status = "incomplete"
    else:
        status = "complete"  # unsettled routes, if any, are listed in the summary
    summary = aggregate(catalog, finals, attempts, towns)
    summary.update(
        request=request, status=status, towns=towns, errors=errors, memory_pauses=memory_events
    )
    atomic_json(args.out / "summary.json", summary)
    progress(status)
    print(json.dumps(summary["families"], indent=1))


if __name__ == "__main__":
    main()
