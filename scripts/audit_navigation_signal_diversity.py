#!/usr/bin/env python3
"""Independent closed-case RGB/geography/phase audit; retain every200case outcome."""

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import yaml
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_dense_labels import offline_map  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.rule_geometry import match_light, project_box  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.sensors import calibrations  # noqa:E402
from src.navigation.signal_data import phase_response_v2  # noqa:E402
from src.navigation.signal_pixel_labels import active_bulb_masks  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "outputs/navigation_signal_diversity_20261021"
    request = read(source / "request.json")
    plan = read(request["plan"])
    if digest(request["plan"]) != request["plan_sha256"]:
        raise ValueError("Original200case plan changed")
    started, status, phase, error = (
        time.monotonic(),
        "running",
        "waiting for exact collector exit",
        None,
    )
    rows, labels, maps, trees = [], [], {}, {}
    if psutil.pid_exists(args.wait_pid):
        p = psutil.Process(args.wait_pid)
        if not any(
            a.endswith("/collect_navigation_signal_diversity.py")
            or a == "scripts/collect_navigation_signal_diversity.py"
            for a in p.cmdline()
        ):
            raise ValueError("Wait only on exact original collection process")
        created = p.create_time()
    else:
        created = None
    shutil.copyfile(__file__, out / "audit_source.py")
    atomic_json(
        out / "request.json",
        dict(
            source=str(source),
            source_request_sha256=digest(source / "request.json"),
            audit_source_sha256=digest(__file__),
            wait_pid=args.wait_pid,
            process_created=created,
            original_cases=200,
            wait_cap_seconds=2700,
            audit_cap_seconds=600,
            artifact_cap_bytes=100_000_000,
        ),
    )

    def check():
        if (out / "STOP").exists():
            raise InterruptedError("Owned auditSTOP")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("Audit100MB limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Strict signal independent audit",
                status=status,
                phase=phase,
                error=error,
                frames=len(rows),
                target_frames=200,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            while created is not None and psutil.pid_exists(args.wait_pid):
                p = psutil.Process(args.wait_pid)
                if p.create_time() != created or p.status() == psutil.STATUS_ZOMBIE:
                    break
                check()
                if time.monotonic() - started > 2700:
                    raise InterruptedError("Exactowner wait2700second limit")
                progress()
                time.sleep(2)
            audit_started = time.monotonic()
            rig = calibrations(
                yaml.safe_load((source / "source/configs/navigation.yaml").read_text())
            )
            phase = "independent closed-case geography/phase/RGB audit"
            for file, sha in request["source_hashes"].items():
                if digest(source / "source" / file) != sha:
                    raise ValueError("Archived capture source changed")
            # All current role sources must still identify the originally frozen roles.
            for file, sha in (
                (plan["catalog"], plan["catalog_sha256"]),
                (plan["guard"], plan["guard_sha256"]),
                (plan["eligibility_request"], plan["eligibility_request_sha256"]),
            ):
                if digest(file) != sha:
                    raise ValueError("Frozen geography source changed")
            from src.navigation.vendor import activate

            activate()
            import carla

            for task in plan["tasks"]:
                capture = source / "native" / task["task_id"] / "capture"
                geometry = (
                    read(capture / "geometry.json")
                    if (capture / "geometry.json").exists()
                    else None
                )
                setup = read(capture / "setup.json") if (capture / "setup.json").exists() else None
                town = task["town"]
                if town not in maps:
                    maps[town] = offline_map(town, plan["map_sha256"][town])
                    forbidden = [
                        w.transform.location
                        for w in maps[town].generate_waypoints(0.5)
                        if w.is_junction and w.junction_id not in plan["allowed_junctions"][town]
                    ]
                    trees[town] = cKDTree([[p.x, p.y, p.z] for p in forbidden])
                for case in task["cases"]:
                    check()
                    if time.monotonic() - audit_started > 600:
                        raise InterruptedError("600second audit limit")
                    row = dict(
                        task_id=task["task_id"],
                        case_id=case["route_id"],
                        town=town,
                        junction_id=case["junction_id"],
                        illumination=task["illumination"],
                        role="train",
                    )
                    meta = capture / "case_metadata" / f"{case['route_id']}.json"
                    row["case_metadata"] = str(meta.relative_to(ROOT))
                    if not meta.exists():
                        row.update(eligible=False, disposition="unknown_missing_closed_case")
                        rows.append(row)
                        progress()
                        continue
                    try:
                        item = read(meta)
                        if geometry is None or setup is None:
                            raise ValueError("Missing native setup or geometry")
                        if any(item["case"].get(k) != v for k, v in case.items()):
                            raise ValueError("Original static case mutated")
                        if any(
                            abs(setup["weather"][k] - v) > 0.001 for k, v in task["weather"].items()
                        ):
                            raise ValueError("Registered weather mismatch")
                        if setup["actual_map_sha256"] != plan["map_sha256"][town]:
                            raise ValueError("Native map mismatch")
                        saved = item["setup"]
                        frames = saved["frames"]
                        if (
                            saved["states"] != ["Red", "Green", "Yellow", "Red"]
                            or not saved["all_lights_frozen"]
                            or len(frames) != 4
                            or any(b <= a for a, b in zip(frames, frames[1:]))
                            or saved["ego_position_drift_m"] > 0.001
                            or max(saved["camera_matrix_max_error"].values()) > 0.001
                        ):
                            raise ValueError("Native state/frame/pose/rig audit failed")
                        lamp = match_light(item["case"]["light"], geometry)
                        if (
                            np.linalg.norm(
                                np.asarray(lamp["pose"][:3]) - case["inventory_light_xyz"]
                            )
                            >= 0.002
                        ):
                            raise ValueError("Inventory/physical signal correspondence mismatch")
                        identity = str(lamp["capture_actor_id"])
                        phase_rows = saved["phase_lights"]
                        if [p["light_states"][identity] for p in phase_rows] != [
                            "Red",
                            "Green",
                            "Yellow",
                            "Red",
                        ]:
                            raise ValueError("Physical target phase mismatch")
                        actor_ids = set(phase_rows[0]["light_states"])
                        if any(set(p["light_states"]) != actor_ids for p in phase_rows):
                            raise ValueError("Background signal inventory changed")
                        if any(
                            len({p["light_states"][i] for p in phase_rows}) != 1
                            for i in actor_ids - {identity}
                        ):
                            raise ValueError("Non-target light changed")
                        wp = maps[town].get_waypoint(carla.Location(*case["ego_pose"][:3]))
                        clearance = (
                            float(trees[town].query(case["ego_pose"][:3])[0])
                            - saved["ego_position_drift_m"]
                        )
                        if clearance < 1 or (
                            wp.is_junction and wp.junction_id not in plan["allowed_junctions"][town]
                        ):
                            raise ValueError("Actual measured staticpose violates1mgeography")
                        images, image_hashes = {}, {}
                        for state in ("red", "green", "yellow", "red_repeat"):
                            for view in ("front", "left", "right"):
                                name = f"route{case['route_id']}_{state}_{view}.png"
                                path = capture / name
                                if digest(path) != item["image_sha256"][name]:
                                    raise ValueError("NativeRGBhash changed")
                                with Image.open(path) as im:
                                    if im.mode != "RGB" or im.size != (512, 288):
                                        raise ValueError("NativeRGBformat mismatch")
                                    images[(state, view)] = np.array(im)
                                image_hashes[str(path.relative_to(ROOT))] = digest(path)
                        if len(item["image_sha256"]) != 12:
                            raise ValueError("Exact12images required")
                        own_labels = []
                        projections = {
                            (view, head): project_box(
                                box["vertices"], case["ego_pose"], calibration
                            )
                            for view, calibration in rig.items()
                            for head, box in enumerate(lamp["boxes_world"])
                        }
                        visible = {key for key, box in projections.items() if box["in_image"]}
                        if {(h["view"], h["head"]) for h in item["housing_pixels"]} != visible:
                            raise ValueError("Projected visible housing denominator changed")
                        for housing in item["housing_pixels"]:
                            box = housing["contrast"]["crop_xyxy"]
                            projection = projections[(housing["view"], housing["head"])]
                            a, b, c, d = projection["bbox_xyxy"]
                            expected_box = [
                                max(0, math.floor(a)),
                                max(0, math.floor(b)),
                                min(512, math.ceil(c)),
                                min(288, math.ceil(d)),
                            ]
                            if box != expected_box:
                                raise ValueError("Independent native housing projection differs")
                            x0, y0, x1, y1 = box
                            view = housing["view"]
                            crops = [
                                images[(state, view)][y0:y1, x0:x1]
                                for state in ("red", "green", "yellow", "red_repeat")
                            ]
                            response = phase_response_v2(crops[0], crops[1], crops[3])
                            yellow = crops[2].astype(float)
                            yellow_support = bool(
                                response["static_phase_response_supported"]
                                and (
                                    np.minimum(yellow[..., 0], yellow[..., 1]) - yellow[..., 2]
                                ).max()
                                >= 30
                            )
                            masks = active_bulb_masks(*crops)
                            if not response["static_phase_response_supported"]:
                                masks[:] = False
                            if not yellow_support:
                                masks[1] = False
                            own_labels.append(
                                dict(
                                    task_id=task["task_id"],
                                    case_id=case["route_id"],
                                    town=town,
                                    junction_id=case["junction_id"],
                                    view=view,
                                    head=housing["head"],
                                    crop_xyxy=box,
                                    **response,
                                    yellow_color_supported=yellow_support,
                                    pixels_yx=[np.argwhere(m).tolist() for m in masks],
                                    counts=masks.sum(axis=(1, 2)).tolist(),
                                )
                            )
                        row.update(
                            eligible=True,
                            disposition="verified_closed_case",
                            case_metadata_sha256=digest(meta),
                            geometry_sha256=digest(capture / "geometry.json"),
                            setup_sha256=digest(capture / "setup.json"),
                            minimum_clearance_minus_drift_m=clearance,
                            image_sha256=image_hashes,
                            frames=frames,
                            source_capture=str(capture.relative_to(ROOT)),
                            supported_housings=sum(
                                label["static_phase_response_supported"] for label in own_labels
                            ),
                        )
                        labels.extend(own_labels)
                    except Exception as exc:
                        row.update(
                            eligible=False, disposition="quarantined_audit_failure", error=repr(exc)
                        )
                    rows.append(row)
                    progress()
            atomic_json(out / "cases.json", rows)
            atomic_json(out / "labels.json", labels)
            eligible = [r for r in rows if r["eligible"]]
            phase_support = {}
            for phase_name, index in (("red", 0), ("yellow", 1), ("green", 2)):
                pool = [
                    label
                    for label in labels
                    if label["static_phase_response_supported"]
                    and (phase_name != "yellow" or label["yellow_color_supported"])
                ]
                phase_support[phase_name] = dict(
                    housings=len(pool), nonempty=sum(bool(label["counts"][index]) for label in pool)
                )
            atomic_json(
                out / "manifest.json",
                dict(
                    schema="navigation-strict-signal-supplement-v1",
                    role="train",
                    original_cases=200,
                    eligible_cases=eligible,
                    unknown_or_quarantined=[r for r in rows if not r["eligible"]],
                    original_validation_unchanged=True,
                    labels_file=str(out / "labels.json"),
                    labels_sha256=digest(out / "labels.json"),
                    source_request_sha256=digest(source / "request.json"),
                    plan_sha256=digest(request["plan"]),
                ),
            )
            atomic_json(
                out / "assessment.json",
                dict(
                    audit_complete=True,
                    original_cases=200,
                    verified_cases=len(eligible),
                    unknown_cases=sum(
                        r["disposition"] == "unknown_missing_closed_case" for r in rows
                    ),
                    quarantined_cases=sum(
                        r["disposition"] == "quarantined_audit_failure" for r in rows
                    ),
                    verified_rgb_images=12 * len(eligible),
                    phase_support=phase_support,
                    eligible_locations=sorted({(r["town"], r["junction_id"]) for r in eligible}),
                    illumination_cases=dict(Counter(r["illumination"] for r in eligible)),
                    manifest_sha256=digest(out / "manifest.json"),
                    all_attempts_successful=len(eligible) == 200,
                    original_failure_artifacts_unchanged=True,
                    no_automatic_fit=True,
                    audit_seconds=time.monotonic() - audit_started,
                ),
            )
            status = "complete"
            phase = "Independentauditfinished; originalunknowns/failuresremain"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
