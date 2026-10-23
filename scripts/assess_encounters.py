#!/usr/bin/env python3
"""Assess an encounter-catalog run: per-family results with location-clustered intervals,
every non-rule-safe encounter, every crash/retry/interruption, and (optionally) exact
agreement with another run of the same catalog and seeds. Read-only on run outputs."""

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.metrics import wilson_interval  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402

spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_encounters.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def worker_is_alive(chunk):
    """Check the owned evaluator, including its output argument to reject PID reuse."""
    path = chunk / "worker.json"
    if not path.exists():
        return False
    try:
        proc = psutil.Process(json.loads(path.read_text())["pid"])
        command = proc.cmdline()
        return (
            proc.status() != psutil.STATUS_ZOMBIE
            and "--worker" in command
            and "--out" in command
            and command[command.index("--out") + 1] == str(chunk.resolve())
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
        return False


def settled(out, catalog):
    """Final record and attempt history per route, replaying the runner's own rules."""
    finals, attempts = {}, {}
    for town in catalog["towns"]:
        ids = sorted(e["route_id"] for e in catalog["encounters"] if e["town"] == town)
        size = json.loads((out / "request.json").read_text())["chunk"]
        for start in range(0, len(ids), size):
            group = ids[start : start + size]
            base = out / town / f"chunk_{group[0]:03d}_{group[-1]:03d}"
            if not base.with_name(base.name + "_a0.routes.json").exists():
                continue
            a, f = runner.settle_group(argparse.Namespace(out=out), None, group, base, launch=False)
            live = {}
            for history in a.values():
                for item in history:
                    if item["status"] != "no evaluator record":
                        continue
                    chunk = base.parent / item["attempt"]
                    if chunk not in live:
                        live[chunk] = worker_is_alive(chunk)
                    if live[chunk]:
                        # settle_group is designed to run after a child exits.
                        # A live snapshot must not label its current trial a crash.
                        item["status"] = "in progress"
            attempts.update(a)
            finals.update(f)
    return finals, attempts


def light_identity(entry, runtime_id, runtime, reference, tolerance_m=0.01):
    """CARLA actor IDs are session-local; compare physical location and declared lane."""
    lane = list(ast.literal_eval(entry["movement_key"][2]))
    actual = [
        s
        for s in runtime["signs"]
        if s["type"] == "traffic.traffic_light" and s["id"] == runtime_id
    ]
    expected = [
        s
        for s in reference["lights"]
        if s["id"] in entry["signal_ids_at_inventory"]
        and lane in [p["lane"] for p in s["stop_lanes"]]
    ]
    result = {
        "runtime_id": runtime_id,
        "entry_lane": lane,
        "inventory_ids": entry["signal_ids_at_inventory"],
        "status": "wrong_light",
    }
    if runtime["map"] != entry["town"] or len(actual) != 1 or len(expected) != 1:
        return result
    distances = [
        float(np.linalg.norm(np.asarray(s["xyz"]) - np.asarray(expected[0]["xyz"])))
        for s in runtime["signs"]
        if s["type"] == "traffic.traffic_light"
    ]
    distance = float(np.linalg.norm(np.asarray(actual[0]["xyz"]) - np.asarray(expected[0]["xyz"])))
    result.update(
        runtime_xyz=actual[0]["xyz"],
        inventory_xyz=expected[0]["xyz"],
        position_difference_m=distance,
        tolerance_m=tolerance_m,
    )
    # Ambiguous colocated actors do not establish physical identity.
    if distance <= tolerance_m and sum(d <= tolerance_m for d in distances) == 1:
        result["status"] = "setup_started"
    return result


def setup_audit(out, catalog, finals, attempts, light_audit=None):
    """Did each encounter's declared scenario start (and bind the inventory's light)?
    Upstream skips a failed scenario setup with only a printed warning."""
    result = {}
    inventories = {}
    for e in catalog["encounters"]:
        rid = e["route_id"]
        if rid not in finals:
            result[rid] = "unsettled"
            continue
        if not e["scenario"]:
            result[rid] = "no_setup_declared"
            continue
        # Use the same attempt that supplied the final result, never glob order
        # or a successful setup from an earlier crashed attempt.
        attempt = next(
            item["attempt"]
            for item in reversed(attempts[rid])
            if item.get("route_id") == finals[rid]["route_id"]
            and runner.CRASH_STATUS not in item["status"]
        )
        path = out / e["town"] / attempt / "scenario_setup" / f"{rid}.json"
        if not path.exists():
            result[rid] = "setup_missing"
            continue
        audit = json.loads(path.read_text())
        started = [s for s in audit.get("started", []) if s["type"] == e["scenario"]]
        if not started:
            result[rid] = "setup_failed"
        elif e["scenario"] == "NavigationSignalHold":
            town = e["town"]
            if town not in inventories:
                inventory_path = ROOT / catalog["inventory"] / f"{town}.json"
                data = inventory_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != catalog["inventory_sha256"][town]:
                    raise ValueError(f"Reference inventory hash mismatch for {town}")
                inventories[town] = json.loads(data)
            runtime_path = path.parent.parent / "runtime_sign_inventory.json"
            if not runtime_path.exists():
                result[rid] = "setup_missing"
                continue
            identity = light_identity(
                e,
                started[0]["ego_light_id"],
                json.loads(runtime_path.read_text()),
                inventories[town],
            )
            result[rid] = identity["status"]
            if light_audit is not None:
                light_audit[rid] = {"attempt": attempt, **identity}
        else:
            result[rid] = "setup_started"
    return result


def cluster_interval(rows, key, success, reps=10000, seed=0):
    """Percentile cluster bootstrap: resample whole locations, not encounters."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[key(row)].append(success(row))
    groups = [np.array(v, dtype=float) for v in clusters.values()]
    if not groups:
        return None, None, 0
    # A percentile bootstrap cannot discover failures absent from the sample.
    # Reporting [1, 1] would suggest certainty, especially with just 7 junctions.
    values = np.concatenate(groups)
    if len(groups) < 2 or np.all(values == values[0]):
        return None, None, len(groups)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        sample = np.concatenate([groups[i] for i in pick])
        stats.append(sample.mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(groups)


def location(entry):
    # Junction is the clustering unit for every family; stop actors share junctions too.
    return (entry["town"], entry["junction_id"])


def collision_audit(out):
    """Every collision in every raw attempt, including failed setups and retries."""
    events = []
    for path in sorted(out.glob("*/chunk_*/leaderboard.json")):
        records = json.loads(path.read_text()).get("_checkpoint", {}).get("records", [])
        for record in records:
            for kind in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"):
                for detail in record.get("infractions", {}).get(kind, []):
                    events.append(
                        {
                            "attempt": str(path.parent.relative_to(out)),
                            "route_id": runner.route_id(record["route_id"]),
                            "type": kind,
                            "detail": detail,
                        }
                    )
    return events


def planned_headline(catalog, finals, setup, all_collisions):
    """Every planned encounter stays in the headline, including invalid setups."""
    rows = catalog["encounters"]
    valid = {rid for rid, state in setup.items() if state in ("setup_started", "no_setup_declared")}

    def passes(e, key):
        return e["route_id"] in valid and bool(finals.get(e["route_id"], {}).get(key, False))

    safe = sum(passes(e, "safe_completed") for e in rows)
    rule_safe = sum(passes(e, "rule_safe_completed") for e in rows)
    lo, hi, locations = cluster_interval(rows, location, lambda e: passes(e, "rule_safe_completed"))
    return dict(
        expected=len(rows),
        recorded=len(finals),
        valid_setups=len(valid),
        setup_failures=sorted(
            e["route_id"] for e in rows if e["route_id"] in finals and e["route_id"] not in valid
        ),
        missing=sorted(e["route_id"] for e in rows if e["route_id"] not in finals),
        raw_completed=sum(bool(r["completed"]) for r in finals.values()),
        completed=sum(passes(e, "completed") for e in rows),
        rule_safe=rule_safe,
        strict_safe=safe,
        all_collision_events=all_collisions,
        rule_safe_fraction=rule_safe / len(rows),
        strict_safe_fraction=safe / len(rows),
        rule_safe_wilson95_unclustered=list(wilson_interval(rule_safe, len(rows))),
        rule_safe_cluster_bootstrap95=[lo, hi],
        locations=locations,
        note=(
            "All planned cases retained; missing/wrong setups cannot pass. "
            "Family valid-only rates are diagnostic."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compare", type=Path, help="Another run of the same catalog and seed")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--wait-for-run",
        action="store_true",
        help="Wait on the runner lock, then assess after it exits (including an incomplete exit)",
    )
    args = parser.parse_args()
    # Holding a shared lock prevents a concurrent resume from mutating the output
    # during the final audit. Ordinary live snapshots stay nonblocking/read-only.
    run_lock = None
    if args.wait_for_run:
        run_lock = (args.out / ".runner.lock").open("r")
        fcntl.flock(run_lock, fcntl.LOCK_SH)
    request = json.loads((args.out / "request.json").read_text())
    catalog, digest = runner.verify_catalog(Path(request["catalog"]))
    if digest != request["catalog_sha256"]:
        raise ValueError("Run used a different catalog version")
    by_id = {e["route_id"]: e for e in catalog["encounters"]}
    finals, attempts = settled(args.out, catalog)
    light_audit = {}
    setup = setup_audit(args.out, catalog, finals, attempts, light_audit)
    invalid = {
        rid for rid, v in setup.items() if v in ("setup_failed", "wrong_light", "setup_missing")
    }

    families = {}
    for family in catalog["coverage"]:
        entries = [e for e in catalog["encounters"] if e["family"] == family]
        planned = len(entries)
        unsettled = [e["route_id"] for e in entries if e["route_id"] not in finals]
        setup_failed = [e["route_id"] for e in entries if e["route_id"] in invalid]
        # Valid-setup family rates are diagnostics. The separate headline below
        # retains every planned case and treats missing/wrong setup as failure.
        entries = [e for e in entries if e["route_id"] not in invalid and e["route_id"] in finals]
        rows = [(e, finals.get(e["route_id"])) for e in entries]
        n = len(rows)
        ok = sum(bool(r and r["rule_safe_completed"]) for _, r in rows)
        low, high, clusters = cluster_interval(
            rows,
            lambda row: location(row[0]),
            lambda row: bool(row[1] and row[1]["rule_safe_completed"]),
        )
        families[family] = {
            "planned_encounters": planned,
            "encounters": n,
            "setup_failed": setup_failed,
            "unsettled": unsettled,
            "settled": sum(r is not None for _, r in rows),
            "rule_safe": ok,
            "strict_safe": sum(bool(r and r["safe_completed"]) for _, r in rows),
            "completed": sum(bool(r and r["completed"]) for _, r in rows),
            "collision_events": sum(r["collisions"] for _, r in rows if r),
            "red_violations": sum(r["red_violations"] for _, r in rows if r),
            "stop_violations": sum(r["stop_violations"] for _, r in rows if r),
            "min_speed_episodes": sum(bool(r and r["min_speed_infractions"]) for _, r in rows),
            "locations": clusters,
            "rule_safe_cluster_bootstrap95": [low, high],
            "rule_safe_wilson95_unclustered": list(wilson_interval(ok, n)) if n else None,
        }

    failures = []
    for rid in sorted(by_id):
        e, r = by_id[rid], finals.get(rid)
        if r is None or rid in invalid or not r["rule_safe_completed"]:
            failures.append(
                {
                    "route_id": rid,
                    "family": e["family"],
                    "town": e["town"],
                    "junction_id": e["junction_id"],
                    "turn": e["turn"],
                    "scenario": e["scenario"],
                    "status": r["status"] if r else "unsettled",
                    "setup": setup[rid],
                    "completion_percent": r["completion_percent"] if r else None,
                    "infractions": {
                        k: len(v) for k, v in (r or {}).get("infractions", {}).items() if v
                    },
                }
            )
    history = Counter()
    for rid, items in attempts.items():
        for item in items[:-1] if rid in finals else items:
            history[item["status"]] += 1
    # CARLA also segfaults during normal SIGTERM shutdown, so "signal 11" in carla.log is
    # not a crash marker. A mid-run failure kills the evaluator child instead.
    crashes = []
    for error in sorted(args.out.glob("*/chunk_*/error.json")):
        detail = json.loads(error.read_text())
        current = error.with_name("current_route.json")
        crashes.append(
            {
                "attempt": str(error.parent.relative_to(args.out)),
                "error": detail["error"],
                "running_route": json.loads(current.read_text())["route_id"]
                if current.exists()
                else None,
            }
        )
    collision_events = collision_audit(args.out)
    report = {
        "assessment_version": 4,
        "run": str(args.out),
        "catalog_sha256": digest,
        "policy_kind": request["policy_kind"],
        "settled": len(finals),
        "total": len(by_id),
        "families": families,
        "all_collision_events": len(collision_events),
        "headline": planned_headline(catalog, finals, setup, len(collision_events)),
        "collision_event_audit": collision_events,
        "valid_final_collision_events": sum(f["collision_events"] for f in families.values()),
        "setup_status_counts": dict(Counter(setup.values())),
        "light_identity_audit": light_audit,
        "not_rule_safe": failures,
        "non_final_attempt_statuses": dict(history),
        "routes_with_several_attempts": sorted(r for r, a in attempts.items() if len(a) > 1),
        "routes_charged_a_crash": sorted(
            r
            for r, a in attempts.items()
            if any(
                x["status"] == "no evaluator record" or runner.CRASH_STATUS in x["status"]
                for x in a
            )
        ),  # fmt: skip
        "evaluator_failures": crashes,
        "interrupted_attempts": sorted(
            str(p.parent.relative_to(args.out)) for p in args.out.glob("*/chunk_*/interrupted.json")
        ),
        "note": "Privileged modified PDM-Lite expert; development towns only; not a camera "
        "result. Family rates use valid settled encounters; pending and invalid setups "
        "remain explicit. All collision events include every raw attempt. Cluster bootstrap "
        "resamples junctions; null bounds mean empty/degenerate data, not zero risk. "
        "Wilson assumes independent encounters and can be optimistic for reused locations.",
    }
    if args.compare:
        other_request = json.loads((args.compare / "request.json").read_text())
        for field in ("catalog_sha256", "seed", "policy_kind"):
            if request[field] != other_request[field]:
                raise ValueError(f"Cannot compare runs with different {field}")
        other, _ = settled(args.compare, catalog)
        shared = sorted(set(finals) & set(other))
        fields = ("status", "completion_percent", "rule_safe_completed", "collisions",
                  "red_violations", "stop_violations", "min_speed_infractions")  # fmt: skip
        mismatches = [
            {"route_id": rid, **{f: [finals[rid][f], other[rid][f]] for f in fields
                                 if finals[rid][f] != other[rid][f]}}
            for rid in shared
            if any(finals[rid][f] != other[rid][f] for f in fields)
        ]  # fmt: skip
        report["comparison"] = {"other": str(args.compare), "shared": len(shared),
                                "mismatches": mismatches}  # fmt: skip
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "not_rule_safe"}, indent=1))
    print(f"not rule-safe: {len(failures)}")
    for f in failures:
        print(" ", f)
    if run_lock is not None:
        run_lock.close()


if __name__ == "__main__":
    main()
