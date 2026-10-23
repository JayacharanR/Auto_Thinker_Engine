#!/usr/bin/env python3
"""Corrected-data wrapper around the original unchanged matched training loop."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_visual_student import train  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    original = json.loads(
        (ROOT / "outputs/navigation_visual_study_20261019/request.json").read_text()
    )
    training = "scripts/train_navigation_visual_student.py"
    if digest(ROOT / training) != original["source_hashes"][training]:
        raise ValueError("Repair must reuse the exact original training loop")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["schema"] != "navigation-spatial-study-geography-repair-v3":
        raise ValueError("Repaired fit requires registered corrected data")
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        wrapper_sha256=digest(__file__),
        original_train_loop_sha256=digest(ROOT / training),
        data_manifest_sha256=digest(args.data / "manifest.json"),
        repair_protocol_sha256=digest(
            ROOT / "reports/navigation_m4_geography_repair_protocol_20261019.md"
        ),
    )
    path = args.out / "repair_request.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("Repair source identity changed")
    if not path.exists():
        atomic_json(path, receipt)
        shutil.copyfile(__file__, args.out / "repair_wrapper_source.py")
    args.steps, args.max_wall = 2000, 3600.0
    train(args)


if __name__ == "__main__":
    with RunLock(ROOT / "outputs/.navigation.lock"):
        main()
