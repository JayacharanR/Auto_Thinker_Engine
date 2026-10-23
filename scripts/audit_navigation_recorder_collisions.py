#!/usr/bin/env python3
"""Audit preserved native recordings; no simulator or policy inference required."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Keep prior audit; use a fresh output")
    sources = [Path(__file__), ROOT / "src/navigation/recorder_collisions.py"]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            roots=[str(p.resolve()) for p in args.root],
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )
    files = sorted({p for root in args.root for p in root.rglob("recordings/route_*_rep*.log")})
    records = []
    errors = []
    for i, path in enumerate(files):
        try:
            records.append(recorder_contacts(path))
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="Recorder contact audit",
                files=i + 1,
                total_files=len(files),
                updated_at=time.time(),
            ),
        )
    summary = dict(
        status="complete" if not errors else "complete_with_parse_failures",
        files=len(files),
        parsed=len(records),
        complete_recordings=sum(r["complete"] for r in records),
        ego_contact_starts=sum(r["ego_contact_starts"] for r in records),
        ego_raw_contact_samples=sum(r["ego_raw_contact_samples"] for r in records),
        contact_recordings=sum(r["ego_contact_starts"] > 0 for r in records),
        parse_errors=errors,
        records=records,
        official_rpc_crosscheck_complete=False,
        note=(
            "Provisional offline decoder: preserve native scores; "
            "cross-check official RPC before final headline promotion"
        ),
    )
    atomic_json(args.out / "audit.json", summary)
    atomic_json(
        args.out / "progress.json",
        dict(
            status=summary["status"],
            stage="Recorder contact audit",
            files=len(files),
            total_files=len(files),
            updated_at=time.time(),
        ),
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("records", "parse_errors")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
