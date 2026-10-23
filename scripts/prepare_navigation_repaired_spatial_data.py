#!/usr/bin/env python3
"""Remove whole contaminated/unsafe TRAIN episodes; preserve validation exactly."""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.label_budget import nested_budgets  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve prior repair attempt")
    args.out.mkdir(parents=True)
    base = ROOT / "outputs/navigation_spatial_data_20261019"
    old = json.loads((base / "manifest.json").read_text())
    audit_path = ROOT / "reports/navigation_spatial_actual_geography_20261019.json"
    audit = json.loads(audit_path.read_text())
    contacts_path = ROOT / "reports/navigation_study_source_contacts_20261019.json"
    contacts = {r["id"]: r for r in json.loads(contacts_path.read_text())["records"]}
    if audit["data_manifest_sha256"] != digest(base / "manifest.json"):
        raise ValueError("Actual-pose audit belongs to another corpus")
    geography = {r["id"]: r for r in audit["records"]}
    validation_locations = {
        (r["town"], int(j))
        for r in audit["records"]
        if r["role"] == "validation"
        for j in r["actual_rgb_junction_counts"]
    }
    removed, kept = [], []
    offset = 0
    for e in old["episodes"]:
        geo = geography[e["id"]]
        actual = {(e["town"], int(j)) for j in geo["actual_rgb_junction_counts"]}
        reasons = []
        if e["role"] == "train":
            if actual & validation_locations:
                reasons.append("Actual committed RGB enters an original validation junction")
            if geo["strict_train_violations"]:
                reasons.append("Actual committed RGB violates strict TRAIN guard")
            if not contacts[e["id"]]["complete"] or contacts[e["id"]]["contacts"]:
                reasons.append("Clean imitation requires complete contact-free recorder")
        if reasons:
            removed.append(
                dict(episode=e, reasons=reasons, geography=geo, contacts=contacts[e["id"]])
            )
            continue
        copied = copy.deepcopy(e)
        copied.update(
            parent_start=e["start"], parent_end=e["end"], start=offset, end=offset + e["labels"]
        )
        offset += e["labels"]
        kept.append(copied)
    train = [e for e in kept if e["role"] == "train"]
    val = [e for e in kept if e["role"] == "validation"]
    if (
        len(train) != 55
        or sum(e["labels"] for e in train) != 15712
        or len(val) != 13
        or offset != 18534
    ):
        raise ValueError("Repair differs from registered complete-episode exclusions")
    sources = [
        Path(__file__),
        ROOT / "reports/navigation_m4_geography_repair_protocol_20261019.md",
        audit_path,
        contacts_path,
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    request = dict(
        original_manifest_sha256=digest(base / "manifest.json"),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        removed=removed,
        original_validation_unchanged=True,
    )
    atomic_json(args.out / "request.json", request)
    for name, expected in old["array_sha256"].items():
        if digest(base / name) != expected:
            raise ValueError("Preserved original array changed")
        original = np.load(base / name, mmap_mode="r", allow_pickle=False)
        required = offset * int(np.prod(original.shape[1:])) * original.dtype.itemsize + 128
        if shutil.disk_usage(args.out).free < 25e9 + required:
            raise RuntimeError("Corrected data array would breach25GB reserve")
        output = np.lib.format.open_memmap(
            args.out / name, mode="w+", dtype=original.dtype, shape=(offset, *original.shape[1:])
        )
        for e in kept:
            output[e["start"] : e["end"]] = original[e["parent_start"] : e["parent_end"]]
        output.flush()
    updated = copy.deepcopy(old)
    updated.update(
        schema="navigation-spatial-study-geography-repair-v3",
        episodes=kept,
        samples=offset,
        annotation_budget=nested_budgets(
            [
                dict(
                    id=e["id"],
                    role="train",
                    partition="train",
                    town=e["town"],
                    labels=e["labels"],
                    family=e["family"],
                )
                for e in train
            ]
        ),
        source_manifest_sha256=old["source_manifest_sha256"],
        request_sha256=digest(args.out / "request.json"),
        array_sha256={name: digest(args.out / name) for name in old["array_sha256"]},
        actual_validation_locations=sorted(validation_locations),
        prior_contaminated_corpus_sha256=digest(base / "manifest.json"),
    )
    atomic_json(args.out / "manifest.json", updated)
    guard = json.loads(
        (ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json").read_text()
    )
    for town in guard["allowed_junctions"]:
        guard["allowed_junctions"][town] = [
            j for j in guard["allowed_junctions"][town] if (town, j) not in validation_locations
        ]
    guard["additional_actual_validation_reservation"] = sorted(validation_locations)
    guard["parent_guard_sha256"] = digest(
        ROOT / "outputs/navigation_m2_varied_waits_20261018/guard.json"
    )
    atomic_json(args.out / "actual_training_guard.json", guard)
    # Exact array segments verify unchanged validation, including its early frames.
    for e in val:
        for name in updated["array_sha256"]:
            current = np.load(args.out / name, mmap_mode="r", allow_pickle=False)[
                e["start"] : e["end"]
            ]
            parent = np.load(base / name, mmap_mode="r", allow_pickle=False)[
                e["parent_start"] : e["parent_end"]
            ]
            if not np.array_equal(current, parent):
                raise ValueError("Validation array segment changed")
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            stage="M4 split-safe corpus repair",
            frames=offset,
            target_frames=offset,
            updated_at=time.time(),
        ),
    )
    print(
        json.dumps(
            dict(
                train_episodes=len(train),
                training_labels=15712,
                validation_unchanged=13,
                removed_episodes=len(removed),
                label_allocations=updated["annotation_budget"]["allocations"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
