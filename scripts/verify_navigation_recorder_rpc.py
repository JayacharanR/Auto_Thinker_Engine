#!/usr/bin/env python3
"""Cross-check offline collision decoding against official CARLA recorder queries.

An owned server parses preserved files only: no policy, traffic or evaluation
mission is started. Run after the active GPU study releases the navigation lock.
"""

import argparse
import collections
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import CarlaServer, RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.vendor import activate  # noqa: E402


def query_rows(response):
    rows = []
    pattern = re.compile(
        r"^\s*(\d+)\s+[ovwtha]\s+[ovwtha]\s+(\d+)\s+"
        r"(?:[A-Za-z_][\w.]*\s+)?(\d+)(?:\s|$)"
    )
    for line in response.splitlines():
        if re.match(r"^\s*\d+\s+[ovwtha]\s+[ovwtha]", line):
            match = pattern.match(line)
            if not match:
                raise ValueError(f"Unparsed native collision row:{line}")
            rows.append(tuple(map(int, match.groups())))
    if not re.search(r"^Frames:\s*\d+\s*$", response, re.M):
        raise ValueError("Native query did not return a recorder frame summary")
    return rows


def expected_rows(record, side):
    return [
        (int(f"{row['time_s']:.0f}"), row["actor1"], row["actor2"])
        for row in record["ordered_starts"]
        if row[side]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain every RPC verification attempt")
    records = {}
    for audit in args.audit:
        for record in json.loads(audit.read_text())["records"]:
            path = record["path"]
            if path in records and records[path]["sha256"] != record["sha256"]:
                raise ValueError("Input audits disagree about a recorder")
            records[path] = record
    with RunLock(ROOT / "outputs/.navigation.lock"):
        check_resources(args.out, 25)
        vendor = activate()
        sources = [Path(__file__), ROOT / "src/navigation/recorder_collisions.py"]
        for source in sources:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(
            args.out / "request.json",
            dict(
                audits=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.audit],
                source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
                policy_inference=False,
                missions_started=0,
                scope=(
                    "Official recorder file queries on owned server; historical outcomes unchanged"
                ),
            ),
        )
        checked, skipped, failures = [], [], []
        stopped = False
        with CarlaServer(vendor["carla_root"], args.out / "server", town="Town03_Opt") as client:
            for path, record in records.items():
                if (args.out / "STOP").exists():
                    stopped = True
                    break
                if not record["complete"]:
                    skipped.append(
                        dict(
                            path=path,
                            reason="Structurally partial recording; keep lower-bound contacts",
                        )
                    )
                    continue
                if digest(path) != record["sha256"]:
                    raise ValueError("Preserved recorder changed before official query")
                try:
                    matches = []
                    raw_paths = []
                    for first, second, side in (("h", "a", "hero1"), ("a", "h", "hero2")):
                        response = client.show_recorder_collisions(path, first, second)
                        raw = args.out / f"{record['sha256']}_{first}_{second}.txt"
                        if raw.exists() and raw.read_text() != response:
                            raise ValueError("Same recorder returned inconsistent query text")
                        raw.write_text(response)
                        actual = query_rows(response)
                        expected = expected_rows(record, side)
                        matches.append(collections.Counter(actual) == collections.Counter(expected))
                        raw_paths.append(dict(path=str(raw), sha256=digest(raw), rows=len(actual)))
                    if digest(path) != record["sha256"]:
                        raise ValueError("Recorder changed during official query")
                    checked.append(
                        dict(
                            path=path,
                            sha256=record["sha256"],
                            all_rows_match=all(matches),
                            queries=raw_paths,
                        )
                    )
                except Exception as error:
                    failures.append(dict(path=path, error=str(error)))
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="running",
                        stage="Official recorder RPC validation",
                        files=len(checked) + len(skipped) + len(failures),
                        total_files=len(records),
                        updated_at=time.time(),
                    ),
                )
        passed = (
            not stopped
            and not failures
            and bool(checked)
            and all(row["all_rows_match"] for row in checked)
            and len(checked) + len(skipped) == len(records)
        )
        atomic_json(
            args.out / "verification.json",
            dict(
                status="complete" if passed else "stopped" if stopped else "failed",
                all_complete_recordings_match=passed,
                checked=checked,
                partial_recordings=skipped,
                failures=failures,
                policy_inference=False,
                missions_started=0,
            ),
        )
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete" if passed else "stopped" if stopped else "failed",
                stage="Official recorder RPC validation",
                files=len(checked) + len(skipped) + len(failures),
                total_files=len(records),
                updated_at=time.time(),
            ),
        )
        print(
            json.dumps(
                dict(
                    all_complete_recordings_match=passed,
                    checked=len(checked),
                    partial=len(skipped),
                    failures=len(failures),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
