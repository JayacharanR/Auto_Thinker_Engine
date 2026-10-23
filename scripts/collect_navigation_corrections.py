#!/usr/bin/env python3
"""Freeze the old internal split, then collect a bounded fresh DAgger pilot."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_data import audit  # noqa: E402
from src.navigation.dagger import digest, internal_location_guard, validate_guard  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json  # noqa: E402
from src.navigation.splits import collection_geometry_eligibility  # noqa: E402


def prepare(args):
    manifest = args.internal_manifest.resolve()
    assignments = json.loads((args.catalog.parent / "locations.json").read_text())["assignments"]
    locations = internal_location_guard(json.loads(manifest.read_text()), assignments)
    geometry = collection_geometry_eligibility(
        args.catalog, additional_reserved=locations["reserved_internal_junctions"]
    )
    catalog = json.loads((args.catalog / "catalog.json").read_text())
    entries = [
        e
        for e in catalog["encounters"]
        if e["town"] == "Town03_Opt"
        and e["route_id"] in geometry["kept_route_ids"]
        and set(e["crossed_junction_ids"]) <= set(locations["allowed_junctions"][e["town"]])
    ]
    families = sorted({e["family"] for e in catalog["encounters"]})
    picks = [next(e for e in entries if e["family"] == family) for family in families]
    guard = {
        "purpose": "TRAIN-only DAgger correction pilot, never benchmark evaluation",
        "catalog_sha256": digest(args.catalog / "catalog.json"),
        "internal_manifest": str(manifest),
        "internal_manifest_sha256": digest(manifest),
        **locations,
        "geometry_eligibility": geometry,
        "selection": "First geometry-eligible/internal-TRAIN Town03 route per family; "
        "fresh observations only",
        "pilot_entries": picks,
    }
    if args.guard.exists() and json.loads(args.guard.read_text()) != guard:
        raise FileExistsError("Frozen guard already differs; use a new reviewed output")
    atomic_json(args.guard, guard)
    print(
        json.dumps(
            {
                "guard": str(args.guard),
                "sha256": digest(args.guard),
                "eligible_routes": len(geometry["kept_route_ids"]),
                "pilot": [(e["route_id"], e["family"]) for e in picks],
            },
            indent=2,
        )
    )


def committed(root):
    total = 0
    for path in Path(root).rglob("data/shard_*.zip"):
        with zipfile.ZipFile(path) as archive:
            total += len(json.loads(archive.read("manifest.json"))["samples"])
    return total


def pilot(args):
    if args.out is None or args.out.exists():
        raise FileExistsError("A fresh --out directory is required")
    if not 0 < args.max_wall <= 1800:
        raise ValueError("Correction pilot budget must be at most30 minutes")
    guard = json.loads(args.guard.read_text())
    entries = guard["pilot_entries"]
    town = "Town03_Opt"
    subset = ",".join(str(e["route_id"]) for e in entries)
    validate_guard(args.guard, args.catalog, town, subset)
    args.out.mkdir(parents=True)
    sources = [
        *sorted((ROOT / "src/navigation").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "scripts/run_navigation.py",
        ROOT / "scripts/audit_navigation_data.py",
        ROOT / "configs/navigation.yaml",
    ]
    for source in sources:
        archived = args.out / "source" / source.relative_to(ROOT)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archived)
    atomic_json(
        args.out / "request.json",
        {
            "kind": "DAgger training mixture; not camera evaluation",
            "seed": args.seed,
            "checkpoint_sha256": digest(args.checkpoint),
            "guard_sha256": digest(args.guard),
            "episodes": entries,
            "max_wall": args.max_wall,
            "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "correction_targets": "Instantaneous privileged expert only; no mixture future targets",
            "behavior": "Student plus teacher braking/hazards; "
            "3s teacher recovery after4s stationary expert-go",
        },
    )
    shutil.copyfile(args.guard, args.out / "guard.json")
    child_out = args.out / town
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_navigation.py"),
        "dagger",
        "--route",
        str(args.catalog / f"{town}.xml"),
        "--collection-catalog",
        str(args.catalog),
        "--dagger-guard",
        str(args.guard),
        "--checkpoint",
        str(args.checkpoint),
        "--routes-subset",
        subset,
        "--seed",
        str(args.seed),
        "--seed-per-route",
        "--max-wall",
        str(args.max_wall),
        "--out",
        str(child_out),
    ]
    started = time.monotonic()
    with RunLock(ROOT / "outputs/navigation/.correction_pilot.lock"):
        with (args.out / "console.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                if (args.out / "STOP").exists() or time.monotonic() - started >= args.max_wall:
                    if child_out.exists():
                        (child_out / "STOP").touch()
                ledger = child_out / "leaderboard.json"
                try:
                    done = len(
                        json.loads(ledger.read_text()).get("_checkpoint", {}).get("records", [])
                    )
                except (ValueError, OSError):
                    done = 0
                atomic_json(
                    args.out / "progress.json",
                    {
                        "status": "running",
                        "stage": "M2 DAgger pilot",
                        "updated_at": time.time(),
                        "encounters_done": done,
                        "encounters_total": len(entries),
                        "current_chunk": town,
                        "route_progress_percent": 100 * done / len(entries),
                        "correction_samples": committed(args.out),
                        "training_mixture_not_camera_evaluation": True,
                    },
                )
                time.sleep(2)
    summary_path = child_out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    complete = bool(
        child.returncode == 0
        and summary
        and summary["execution_status"] == "evaluation_complete"
        and summary["count"] == len(entries)
    )
    status = "complete" if complete else ("stopped" if (args.out / "STOP").exists() else "failed")
    atomic_json(
        args.out / "progress.json",
        {
            "status": "running",
            "phase": "auditing committed RGB and controls",
            "stage": "M2 DAgger pilot",
            "updated_at": time.time(),
            "encounters_done": summary["count"] if summary else 0,
            "encounters_total": len(entries),
            "current_chunk": "finished",
            "route_progress_percent": 100 if complete else 0,
            "correction_samples": committed(args.out),
            "training_mixture_not_camera_evaluation": True,
        },
    )
    try:
        result = audit(args.out)
    except Exception as exc:
        atomic_json(args.out / "audit_error.json", {"error": str(exc)})
        progress = json.loads((args.out / "progress.json").read_text())
        progress.update(status="failed", error=str(exc), updated_at=time.time())
        atomic_json(args.out / "progress.json", progress)
        raise
    result.update(
        status=status,
        training_mixture_not_camera_evaluation=True,
        child_exit_code=child.returncode,
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_json(args.out / "data_audit.json", result)
    progress = json.loads((args.out / "progress.json").read_text())
    progress.update(status=status, phase="audit finished", updated_at=time.time())
    atomic_json(args.out / "progress.json", progress)
    print(json.dumps({"status": status, "totals": result["totals"]}, indent=2))
    if not complete:
        raise RuntimeError("Correction collection incomplete; preserve all data and inspect")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "pilot"])
    parser.add_argument(
        "--catalog", type=Path, default="configs/encounters/location_split_v1/train"
    )
    parser.add_argument(
        "--internal-manifest",
        type=Path,
        default="outputs/navigation_m2_cache_20k_20261015/manifest.json",
    )
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default="outputs/navigation_m2_stop_go_20261016/best_policy.pt"
    )
    parser.add_argument("--seed", type=int, default=202610160)
    parser.add_argument("--max-wall", type=float, default=900)
    args = parser.parse_args()
    args.catalog, args.guard, args.checkpoint = (
        args.catalog.resolve(),
        args.guard.resolve(),
        args.checkpoint.resolve(),
    )
    if args.out is not None:
        args.out = args.out.resolve()
    (prepare if args.command == "prepare" else pilot)(args)


if __name__ == "__main__":
    main()
