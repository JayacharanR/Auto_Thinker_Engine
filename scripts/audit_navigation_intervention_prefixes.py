#!/usr/bin/env python3
"""Supplement original actor-order audit with exact static-light correspondence."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.replay import compare_world_prefixes  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def canonical_prefix(trace):
    """Use exact first-frame static poses and unique declared dynamic roles.

    IDs are episode-local only. Ambiguous correspondence is an error, never a
    nearest-neighbour guess. This does not alter the original replay tolerances.
    """
    if len(trace) != 80:
        raise ValueError("Exactly80closed pre-decision physics snapshots required")
    identities = {}
    for actor in trace[0]["actors"]:
        base = (actor["type"], actor["role"])
        if actor["type"] == "traffic.traffic_light":
            key = base + (tuple(actor["pose"]),)
        elif actor["type"].startswith(("vehicle.", "walker.")) and actor["role"]:
            key = base
        else:
            raise ValueError("Undeclared actor correspondence")
        if key in identities or actor["id"] in identities.values():
            raise ValueError("Ambiguous static geometry or dynamic role")
        identities[key] = actor["id"]
    keys = sorted(identities, key=repr)
    result = []
    for frame in trace:
        actors = {a["id"]: a for a in frame["actors"]}
        if len(actors) != len(frame["actors"]) or set(actors) != set(identities.values()):
            raise ValueError("Actor inventory changed during common prefix")
        ordered = []
        for key in keys:
            actor = actors[identities[key]]
            if (actor["type"], actor["role"]) != key[:2]:
                raise ValueError("Episode-local actor identity changed")
            if len(key) == 3 and tuple(actor["pose"]) != key[2]:
                raise ValueError("Static traffic light moved during prefix")
            ordered.append(actor)
        result.append(dict(frame=frame["frame"], actors=ordered))
    return keys, result


def compare_geometry_prefixes(first, second):
    keys_a, ordered_a = canonical_prefix(first)
    keys_b, ordered_b = canonical_prefix(second)
    if keys_a != keys_b:
        raise ValueError("Exact static geometry and unique dynamic identities must match")
    return compare_world_prefixes(ordered_a, ordered_b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    audit = json.loads(args.audit.read_text())
    prefixes, results, sources = {}, [], {}
    with BudgetWatch(args.out):
        for record in audit["records"]:
            if (args.out / "STOP").exists() or time.monotonic() - started > 300:
                raise InterruptedError("Owned audit STOPor300scap")
            path = Path(record["physical_output"]) / "world_trace.jsonl"
            sources[str(path)] = digest(path)
            trace = [json.loads(line) for line in path.read_text().splitlines()][:80]
            case = record["case"]
            canonical_prefix(trace)  # Validate even a context's first episode.
            if case["context_id"] in prefixes:
                first_case, first = prefixes[case["context_id"]]
                results.append(
                    dict(
                        reference_case=first_case,
                        case=case,
                        **compare_geometry_prefixes(first, trace),
                    )
                )
            else:
                prefixes[case["context_id"]] = case, trace
        atomic_json(
            args.out / "assessment.json",
            dict(
                original_audit=str(args.audit.resolve()),
                original_audit_sha256=digest(args.audit),
                original_audit_preserved=True,
                correspondence="Exact static type/role/pose; unique dynamic type/role",
                tolerances_unchanged=True,
                comparisons=results,
                comparisons_count=len(results),
                all_pass=bool(results)
                and all(r["observable_prefix_within_tolerance"] for r in results),
                all_exact=bool(results)
                and all(all(v == 0 for v in r["maximum_differences"].values()) for r in results),
                traces_sha256=sources,
                source_sha256=digest(__file__),
                no_new_native_runs_or_model_fit=True,
                RGB_prefix_equality_unclaimed=True,
            ),
        )
        shutil.copyfile(__file__, args.out / "audit_source.py")
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete"
                if all(r["observable_prefix_within_tolerance"] for r in results)
                else "failed",
                stage="Geometry-corresponded prefix audit",
                frames=len(results),
                target_frames=len(results),
                phase="Original failed audit preserved; static identity fixed independently",
                updated_at=time.time(),
            ),
        )


if __name__ == "__main__":
    main()
