#!/usr/bin/env python3
"""Resume interrupted frozen pilots; never rerun a settled driving failure."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_encounters import chunk_records  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def replay(root, names):
    records, history = {}, []
    for name in names:
        output = root / name
        if not output.exists():
            continue
        measured, stopped = chunk_records(output)
        if stopped:
            raise ValueError("Explicitly stopped old pilot needs its interruption receipt audited")
        for rid, record in measured.items():
            if "Simulation crashed" not in record["status"]:
                if rid in records:
                    raise ValueError("A settled route was executed twice")
                records[rid] = dict(record, attempt=name)
        contacts = []
        for path in sorted((output / "recordings").glob("*.log")):
            contacts.append(recorder_contacts(path))
        history.append(dict(attempt=name, records=list(measured), contacts=contacts))
    return records, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    request = json.loads((args.out / "request.json").read_text())
    protocol = json.loads((args.out / "protocol.json").read_text())
    declared = sorted(e["route_id"] for e in protocol["encounters"])
    continuation = args.out / "continuation"
    continuation.mkdir(exist_ok=True)
    sources = [Path(__file__), ROOT / "scripts/run_navigation.py"]
    receipt = dict(
        original_request_sha256=digest(args.out / "request.json"),
        original_protocol_sha256=digest(args.out / "protocol.json"),
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        reason=(
            "Host/owner interruption; no live processes. "
            "Reuse every settled outcome, including failures."
        ),
        preserve_all_raw_attempt_contacts=True,
    )
    prior = continuation / "request.json"
    if prior.exists() and json.loads(prior.read_text()) != receipt:
        raise ValueError("Continuation source/request changed")
    if not prior.exists():
        atomic_json(prior, receipt)
        for source in sources:
            target = continuation / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    completed, results, status = 0, {}, "running"
    active = None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M4 unchanged encoder driving pilots",
                encounters_done=completed,
                encounters_total=24,
                active_output=str(active) if active else None,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    try:
        with RunLock(ROOT / "outputs/.navigation.lock"):
            for candidate in request["candidates"]:
                if digest(candidate["checkpoint"]) != candidate["sha256"]:
                    raise ValueError("Frozen candidate checkpoint changed")
                name = candidate["name"]
                names = [name] + [
                    p.name for p in sorted(args.out.glob(name + "_resume_a*")) if p.is_dir()
                ]
                records, history = replay(args.out, names)
                todo = sorted(set(declared) - set(records))
                completed += len(records)
                if todo:
                    if (args.out / "STOP").exists():
                        raise InterruptedError("Encoder pilot STOP")
                    active = args.out / f"{name}_resume_a{len(names) - 1}"
                    if active.exists():
                        raise FileExistsError("Never overwrite an interrupted attempt")
                    progress("Same route/seed/checkpoint; settled failures retained")
                    ledger_path = ROOT / "reports/navigation_execution_20261018.json"
                    ledger = json.loads(ledger_path.read_text())
                    ledger["active_output"] = str(args.out.relative_to(ROOT))
                    atomic_json(ledger_path, ledger)
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/run_navigation.py",
                        "student",
                        "--route",
                        protocol["route"],
                        "--routes-subset",
                        ",".join(map(str, todo)),
                        "--seed",
                        str(protocol["seed"]),
                        "--seed-per-route",
                        "--checkpoint",
                        candidate["checkpoint"],
                        "--max-wall",
                        str(protocol["max_wall_per_candidate"]),
                        "--out",
                        str(active),
                    ]
                    with active.with_suffix(".console.log").open("x") as log:
                        child = subprocess.Popen(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                        )
                        while child.poll() is None:
                            if (args.out / "STOP").exists():
                                active.mkdir(parents=True, exist_ok=True)
                                (active / "STOP").touch()
                            progress("Native driving; all outcomes retained")
                            time.sleep(2)
                    atomic_json(
                        continuation / (name + "_exit.json"),
                        dict(exit_code=child.returncode, output=str(active)),
                    )
                    if child.returncode:
                        raise RuntimeError(
                            f"Unfinished native attempt:{name}; inspect before resuming"
                        )
                    names.append(active.name)
                    after, history = replay(args.out, names)
                    completed += len(after) - len(records)
                    records = after
                    if set(records) != set(declared):
                        raise RuntimeError(
                            f"Native resource interruption leaves missing routes:{name}"
                        )
                results[name] = dict(
                    candidate=candidate,
                    records=records,
                    all_attempts=history,
                    arrivals=sum(r["completed"] for r in records.values()),
                    native_rule_safe=sum(r["rule_safe_completed"] for r in records.values()),
                    all_attempt_contact_starts=sum(
                        c["ego_contact_starts"] for a in history for c in a["contacts"]
                    ),
                    incomplete_recordings=[
                        c["path"] for a in history for c in a["contacts"] if not c["complete"]
                    ],
                )
                atomic_json(continuation / "results.json", results)
                progress("Candidate settled; physical/video audit remains separate")
            status = "complete"
    except InterruptedError:
        status = "stopped"
    except Exception as error:
        status = "failed"
        atomic_json(continuation / "error.json", dict(error=str(error), active_output=str(active)))
        raise
    finally:
        progress("finished")
    print(
        json.dumps(
            dict(
                status=status,
                candidates={
                    name: dict(
                        arrivals=r["arrivals"],
                        native_rule_safe=r["native_rule_safe"],
                        all_attempt_contact_starts=r["all_attempt_contact_starts"],
                        incomplete_recordings=len(r["incomplete_recordings"]),
                    )
                    for name, r in results.items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation_encoder_pilot_resume.lock"):
        main()
