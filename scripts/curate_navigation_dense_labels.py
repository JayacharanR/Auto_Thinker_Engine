#!/usr/bin/env python3
"""Publish only independently verified TRAIN label sources; retain all13outcomes."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.audit / "assessment.json").read_text())
    # This first curation accepts geometry rejection only. Byte/alignment/role
    # failures require inspection; never turn a broken audit into fitting evidence.
    if any(
        not e["error"].startswith("Actual trajectory lacks1m reserved-junction clearance:")
        for e in source["errors"]
    ):
        raise ValueError("Non-geographic audit errors remain unresolved; no fitting manifest")
    if any(
        r["partition_violation"] is None or not r["no_dense_labels_eligible"]
        for r in source["failed_sources"]
    ):
        raise ValueError("Missing source is not an acknowledged zero-label geography rejection")
    args.out.mkdir(parents=True, exist_ok=False)
    records = source["results"]
    seen = {"train": {}, "calibration": {}}
    for r in records:
        item = r["item"]
        geo = r["independently_verified_actual_geography"]
        if (
            not r["all_label_RGB_matches_original"]
            or not r["all_pixel_hashes_verified"]
            or not r["zero_next_tick_control_differences"]
            or geo["minimum_reserved_geometry_clearance_m"] < 1.0
            or item["role"] not in seen
        ):
            raise ValueError(
                "Every accepted source requires complete independent "
                "pixel/actuation/geography evidence"
            )
        for p, h in (r["original_source_shards_sha256"] | r["label_shards_sha256"]).items():
            if digest(p) != h:
                raise ValueError("A verified raw source changed")
        seen[item["role"]].setdefault(item["town"], set()).update(
            item["entry"]["crossed_junction_ids"]
        )
    if any(seen["train"].get(t, set()) & v for t, v in seen["calibration"].items()):
        raise ValueError("Accepted source TRAINand calibration locations overlap")
    planned = len(records) + len(source["errors"]) + len(source["failed_sources"])
    if (
        planned != 13
        or len(records) != 11
        or len(seen["train"]) != 5
        or len(seen["calibration"]) != 2
    ):
        raise ValueError(
            "Preserve exact13source denominator,11verifiedsources/"
            "fiveTRAINtowns/two calibration groups"
        )
    manifest = dict(
        schema="navigation-verified-dense-labels-v1",
        sources=records,
        rejected_sources=source["errors"] + source["failed_sources"],
        original_planned_source_count=13,
        accepted_source_count=len(records),
        rejected_source_count=2,
        train_rgb_timestamps=sum(
            r["rgb_timestamps"] for r in records if r["item"]["role"] == "train"
        ),
        train_label_timestamps=sum(
            r["label_timestamps"] for r in records if r["item"]["role"] == "train"
        ),
        calibration_rgb_timestamps=sum(
            r["rgb_timestamps"] for r in records if r["item"]["role"] == "calibration"
        ),
        calibration_label_timestamps=sum(
            r["label_timestamps"] for r in records if r["item"]["role"] == "calibration"
        ),
        role_locations={
            role: {t: sorted(v) for t, v in towns.items()} for role, towns in seen.items()
        },
        original_audit_sha256=digest(args.audit / "assessment.json"),
        original_plan_sha256=source["collection_plan_sha256"],
        all_attempt_contact_starts=source["unfiltered_contact_starts"],
        training_started=False,
        benchmark_changes=False,
        source_geography_gate_relaxed=False,
        no_source_replacement_or_retry=True,
        encoder_exposure="Older frozen encoders and imitation heads have TRAINexposure at these "
        "locations. New dense-label heads must honor this explicit split.",
        driving_qualification=False,
        full_original_audit_retained_failed=True,
    )
    atomic_json(args.out / "manifest.json", manifest)
    shutil.copyfile(__file__, args.out / "curation_source.py")
    atomic_json(
        args.out / "assessment.json",
        dict(
            manifest_sha256=digest(args.out / "manifest.json"),
            source_sha256=digest(__file__),
            every_included_source_independently_verified=True,
            original_denominator=13,
            accepted=11,
            rejected=2,
            no_training_or_promotion=True,
        ),
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k not in ["sources", "rejected_sources"]})
    )


if __name__ == "__main__":
    main()
