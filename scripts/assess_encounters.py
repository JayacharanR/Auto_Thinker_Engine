#!/usr/bin/env python3
"""Assess an encounter-catalog run: per-family results with location-clustered intervals,
every non-rule-safe encounter, every crash/retry/interruption, and (optionally) exact
agreement with another run of the same catalog and seeds. Read-only on run outputs."""

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.metrics import wilson_interval  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402

spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_encounters.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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
            attempts.update(a)
            finals.update(f)
    return finals, attempts


def setup_audit(out, catalog, finals):
    """Did each encounter's declared scenario start (and bind the inventory's light)?
    Upstream skips a failed scenario setup with only a printed warning."""
    audit = {}
    for f in out.glob("*/chunk_*/scenario_setup/*.json"):
        rid = int(f.stem)
        chunk = f.parent.parent
        raw = json.loads((chunk / "leaderboard.json").read_text())["_checkpoint"]["records"]
        if rid in finals and any(r["route_id"] == f"RouteScenario_{rid}_rep0" for r in raw):
            audit[rid] = json.loads(f.read_text())  # the attempt that produced the result
    result = {}
    for e in catalog["encounters"]:
        rid = e["route_id"]
        if not e["scenario"]:
            result[rid] = "no_setup_declared"
            continue
        started = [s for s in audit.get(rid, {}).get("started", []) if s["type"] == e["scenario"]]
        if not started:
            result[rid] = "setup_failed"
        elif (
            e["scenario"] == "NavigationSignalHold"
            and started[0]["ego_light_id"] not in e["signal_ids_at_inventory"]
        ):
            result[rid] = "wrong_light"
        else:
            result[rid] = "setup_started"
    return result


def cluster_interval(rows, key, success, reps=10000, seed=0):
    """Percentile cluster bootstrap: resample whole locations, not encounters."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[key(row)].append(success(row))
    groups = [np.array(v, dtype=float) for v in clusters.values()]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compare", type=Path, help="Another run of the same catalog and seed")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    request = json.loads((args.out / "request.json").read_text())
    catalog, digest = runner.verify_catalog(Path(request["catalog"]))
    if digest != request["catalog_sha256"]:
        raise ValueError("Run used a different catalog version")
    by_id = {e["route_id"]: e for e in catalog["encounters"]}
    finals, attempts = settled(args.out, catalog)
    setup = setup_audit(args.out, catalog, finals)
    invalid = {rid for rid, v in setup.items() if v in ("setup_failed", "wrong_light")}

    families = {}
    for family in catalog["coverage"]:
        entries = [e for e in catalog["encounters"] if e["family"] == family]
        setup_failed = [e["route_id"] for e in entries if e["route_id"] in invalid]
        # Headline over encounters whose declared setup actually ran; the others were not
        # the declared test (detected before driving) and are listed, never counted as passes.
        entries = [e for e in entries if e["route_id"] not in invalid]
        rows = [(e, finals.get(e["route_id"])) for e in entries]
        n = len(rows)
        ok = sum(bool(r and r["rule_safe_completed"]) for _, r in rows)
        low, high, clusters = cluster_interval(
            rows,
            lambda row: location(row[0]),
            lambda row: bool(row[1] and row[1]["rule_safe_completed"]),
        )
        families[family] = {
            "encounters": n,
            "setup_failed": setup_failed,
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
            "rule_safe_wilson95_unclustered": list(wilson_interval(ok, n)),
        }

    failures = []
    for rid in sorted(by_id):
        e, r = by_id[rid], finals.get(rid)
        if r is None or not r["rule_safe_completed"]:
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
    report = {
        "run": str(args.out),
        "catalog_sha256": digest,
        "policy_kind": request["policy_kind"],
        "settled": len(finals),
        "total": len(by_id),
        "families": families,
        "all_collision_events": sum(f["collision_events"] for f in families.values()),
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
        "result. Cluster bootstrap resamples junctions; Wilson assumes independence.",
    }
    if args.compare:
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


if __name__ == "__main__":
    main()
