#!/usr/bin/env python3
"""Audit paired bulb supervision and current geography without fitting or driving."""

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_dense_labels import offline_map  # noqa:E402
from scripts.collect_navigation_signal_pairs import validate_plan  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_learning import junction_roles  # noqa:E402
from src.navigation.signal_pixel_labels import (  # noqa:E402
    active_bulb_masks,
    current_training_locations,
)
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

CORPORA = [
    ROOT / "outputs/navigation_m2_signal_corpus_v2_20261016",
    ROOT / "outputs/navigation_m2_signal_expansion_20261016",
]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan_path = ROOT / "outputs/navigation_dense_label_collection_20261020/plan.json"
    dense = read(plan_path)
    guard_path = Path(dense["actual_training_guard"])
    if digest(guard_path) != dense["actual_training_guard_sha256"]:
        raise ValueError("Current geography guard changed")
    guard = read(guard_path)
    original = read(CORPORA[0] / "plan.json")
    roles = junction_roles(original["cases"])
    validation, calibration = defaultdict(list), defaultdict(list)
    for (town, junction), role in roles.items():
        if role == "validation":
            validation[town].append(junction)
    for item in dense["episodes"]:
        if item["role"] == "calibration":
            calibration[item["town"]].extend(item["entry"]["crossed_junction_ids"])
    # Reservations must still agree with the exact original rule case files.
    reserved = defaultdict(set)
    for file, sha in dense["rule_case_sha256"].items():
        if digest(file) != sha:
            raise ValueError("Reserved rule case changed")
        case = read(file)
        reserved[case["town"]].update(case["crossed_junction_ids"])
    if {t: sorted(v) for t, v in reserved.items()} != dense["reserved_rule_junctions"]:
        raise ValueError("Reserved rule geography mismatch")
    allowed = current_training_locations(
        guard["allowed_junctions"], reserved, calibration, validation
    )
    files = [
        Path(__file__),
        ROOT / "src/navigation/signal_pixel_labels.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "scripts/audit_navigation_dense_labels.py",
        ROOT / "scripts/collect_navigation_signal_pairs.py",
        ROOT / "reports/navigation_signal_pixel_audit_protocol_20261021.md",
        plan_path,
        guard_path,
    ]
    for file in files:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    atomic_json(
        out / "request.json",
        dict(
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in files},
            allowed_junctions=allowed,
            calibration=dict(calibration),
            component_validation=dict(validation),
            maximum_seconds=600,
            maximum_artifact_bytes=100_000_000,
            no_fit_no_native=True,
        ),
    )
    cases, labels, validation_images, validation_points = [], [], {}, set()
    maps, trees, examples, counts = {}, {}, {}, Counter()
    status, error = "running", None

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 600:
            raise InterruptedError("Owned STOP or600second limit")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("100MB artifact limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Paired active-bulb label audit",
                status=status,
                error=error,
                cases=len(cases),
                target_cases=150,
                labels=len(labels),
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            from src.navigation.vendor import activate

            activate()
            import carla

            progress()
            for root in CORPORA:
                plan, audit, request = [
                    read(root / n) for n in ("plan.json", "data_audit.json", "request.json")
                ]
                validate_plan(plan)
                if not audit["static_label_data_gate_passed"] or not audit["all_cases_complete"]:
                    raise ValueError("Only passed original static corpora")
                if digest(root / "plan.json") != audit["plan_sha256"]:
                    raise ValueError("Corpus plan changed")
                for file, sha in request["source_hashes"].items():
                    if digest(root / "source" / file) != sha:
                        raise ValueError("Archived capture source changed")
                for file, sha in audit["image_sha256"].items():
                    check()
                    if digest(root / file) != sha:
                        raise ValueError("Native image changed")
                    counts["verified_images"] += 1
                supported = defaultdict(list)
                for label in audit["labels"]:
                    if label["static_phase_response_supported"]:
                        supported[(label["town"], label["case_id"], label["view"])].append(label)
                for town in sorted({c["town"] for c in plan["cases"]}):
                    receipt = read(root / town / "assessment.json")
                    if not receipt["probe_complete"] or any(
                        digest(root / town / filename) != receipt[key]
                        for filename, key in (
                            ("cases.json", "source_cases_sha256"),
                            ("geometry.json", "geometry_sha256"),
                        )
                    ):
                        raise ValueError("Native static pose/geometry receipt changed")
                    capture = {c["route_id"]: c for c in receipt["cases"]}
                    for case in read(root / town / "cases.json"):
                        check()
                        rid, junction = case["route_id"], case["junction_id"]
                        role = roles[(town, junction)]
                        row = dict(
                            source=str(root.relative_to(ROOT)),
                            town=town,
                            case_id=rid,
                            junction_id=junction,
                            original_role=role,
                        )
                        cases.append(row)
                        if role == "validation":
                            if root != CORPORA[0]:
                                raise ValueError("Supplement contains original validation")
                            row["disposition"] = "unchanged_component_validation"
                            for view in ("front", "left", "right"):
                                for state in ("red", "green", "yellow", "red_repeat"):
                                    file = f"{town}/route{rid}_{state}_{view}.png"
                                    validation_images[file] = audit["image_sha256"][file]
                                    for lab in supported[(town, rid, view)]:
                                        if state == "yellow" and not lab["yellow_color_supported"]:
                                            continue
                                        x0, y0, x1, y1 = lab["crop_xyxy"]
                                        validation_points.add(
                                            (
                                                file,
                                                min(255, int((x0 + x1) / 4)),
                                                min(143, int((y0 + y1) / 4)),
                                            )
                                        )
                            continue
                        if junction not in allowed.get(town, []):
                            row["disposition"] = "excluded_current_reserved_location"
                            continue
                        if town not in maps:
                            maps[town] = offline_map(
                                town, guard["geometry_eligibility"]["map_sha256"][town]
                            )
                            forbidden = [
                                w.transform.location
                                for w in maps[town].generate_waypoints(0.5)
                                if w.is_junction and w.junction_id not in allowed[town]
                            ]
                            trees[town] = cKDTree([[p.x, p.y, p.z] for p in forbidden])
                        wp = maps[town].get_waypoint(carla.Location(*case["ego_pose"][:3]))
                        drift = capture[rid]["ego_position_drift_m"]
                        clearance = float(trees[town].query(case["ego_pose"][:3])[0]) - drift
                        row.update(
                            clearance_minus_native_drift_m=clearance,
                            native_position_drift_m=drift,
                            actual_pose_junction=wp.junction_id if wp.is_junction else None,
                        )
                        if clearance < 1 or (
                            wp.is_junction and wp.junction_id not in allowed[town]
                        ):
                            row["disposition"] = "excluded_actual_pose_geography"
                            continue
                        row["disposition"] = "eligible_train"
                        for view in ("front", "left", "right"):
                            lamps = supported[(town, rid, view)]
                            if not lamps:
                                continue
                            images = []
                            for state in ("red", "green", "yellow", "red_repeat"):
                                with Image.open(
                                    root / town / f"route{rid}_{state}_{view}.png"
                                ) as im:
                                    images.append(np.array(im.convert("RGB")))
                            for lab in sorted(lamps, key=lambda x: x["head"]):
                                x0, y0, x1, y1 = lab["crop_xyxy"]
                                crops = [im[y0:y1, x0:x1] for im in images]
                                masks = active_bulb_masks(*crops)
                                if not lab["yellow_color_supported"]:
                                    masks[1] = False
                                label = dict(
                                    source=row["source"],
                                    town=town,
                                    case_id=rid,
                                    junction_id=junction,
                                    view=view,
                                    head=lab["head"],
                                    crop_xyxy=lab["crop_xyxy"],
                                    original_yellow_supported=lab["yellow_color_supported"],
                                    # Native local (y,x) pixels; no validation pixel labels.
                                    pixels_yx=[np.argwhere(mask).tolist() for mask in masks],
                                    counts=masks.sum(axis=(1, 2)).tolist(),
                                )
                                labels.append(label)
                                examples.setdefault((town, junction), (label, crops, masks))
                        progress()
            if len(validation_images) != 144 or len(validation_points) != 72:
                raise ValueError("Original144image/72label validation denominator changed")
            atomic_json(out / "cases.json", cases)
            atomic_json(out / "labels.json", labels)
            atomic_json(
                out / "validation_identity.json",
                dict(
                    image_sha256=validation_images,
                    housing_center_labels=sorted(validation_points),
                    masks_computed=False,
                ),
            )
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                len(examples), 4, figsize=(12, 2 * len(examples)), squeeze=False
            )
            for row, ((town, junction), (label, crops, masks)) in enumerate(
                sorted(examples.items())
            ):
                for col, (crop, title) in enumerate(
                    zip(crops, ("Red", "Green", "Yellow", "Red repeat"))
                ):
                    ax = axes[row, col]
                    ax.imshow(crop, interpolation="nearest")
                    y, x = np.where(masks[(0, 2, 1, 0)[col]])
                    ax.scatter(x, y, s=20, facecolors="none", edgecolors="cyan", linewidths=0.6)
                    ax.set_title(
                        f"{town}/{junction} {title}\ncase{label['case_id']} {label['view']}",
                        fontsize=8,
                    )
                    ax.axis("off")
            fig.suptitle(
                "First eligible housing per TRAIN location; cyan = fixed phase-response target"
            )
            fig.tight_layout()
            fig.savefig(out / "fixed_examples.png", dpi=130)
            plt.close(fig)
            support = {}
            for phase, index in (("red", 0), ("yellow", 1), ("green", 2)):
                considered = [
                    label
                    for label in labels
                    if phase != "yellow" or label["original_yellow_supported"]
                ]
                support[phase] = dict(
                    housings=len(considered),
                    with_pixels=sum(bool(label["counts"][index]) for label in considered),
                    pixels=sum(label["counts"][index] for label in considered),
                )
            atomic_json(
                out / "assessment.json",
                dict(
                    audit_complete=True,
                    cases=len(cases),
                    verified_images=counts["verified_images"],
                    dispositions=dict(Counter(c["disposition"] for c in cases)),
                    phase_support=support,
                    eligible_locations=[list(key) for key in sorted(examples)],
                    validation_images=144,
                    validation_labels=72,
                    validation_masks_computed=False,
                    case_sha256=digest(out / "cases.json"),
                    labels_sha256=digest(out / "labels.json"),
                    validation_identity_sha256=digest(out / "validation_identity.json"),
                    no_model_fit=True,
                    no_driving_qualification=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            check()
            status = "complete"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
