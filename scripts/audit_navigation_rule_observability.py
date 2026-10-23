#!/usr/bin/env python3
"""Training-only applicable-light frustum and stop-memory audit; no model selection."""

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.contracts import VIEWS, CameraCalibration  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa: E402
from src.navigation.rule_geometry import project_rule_sample, stopped_runs  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources(root, request):
    if any(digest(root / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Archived source changed")


def validate_training_data(data, assessment):
    """Fail before projection if scope/provenance/internal partition no longer agrees."""
    meta, outcome = read(data / "episode.json"), read(data / "outcome.json")
    root = data.parents[3]
    request, guard = read(root / "request.json"), read(root / "guard.json")
    if (
        not assessment["audit_passed"]
        or not assessment["correction_data_gate_passed"]
        or digest(root / "request.json") != assessment["request_sha256"]
        or digest(root / "data_audit.json") != assessment["data_audit_sha256"]
        or digest(root / "guard.json") != request["guard_sha256"]
        or meta["internal_manifest_sha256"] != assessment["internal_manifest_sha256"]
        or digest(guard["internal_manifest"]) != assessment["internal_manifest_sha256"]
    ):
        raise ValueError("Correction audit/source provenance mismatch")
    if (
        meta["town"].removesuffix("_Opt") == "Town05"
        or meta["partition"] != "train"
        or not set(meta["junction_ids"]) <= set(guard["allowed_junctions"][meta["town"]])
        or not outcome["correction_labels_eligible"]
        or outcome["imitation_eligible"]
        or meta["mode"] != "dagger"
    ):
        raise ValueError("Only audited internal-TRAIN correction episodes allowed")
    verify_sources(root, request)
    return meta


def count_sample(row):
    counts = Counter(samples=1)
    signal = row["signal"]
    if signal is None:
        return counts
    views = signal["views"]
    known = row["next_light_state"] in ("Red", "Yellow", "Green")
    flags = [v["housing_in_frustum"] for v in views.values()]
    any_head = any(f is True for f in flags)
    all_out = all(f is False for f in flags)
    in_range = row["next_light_distance_m"] is not None and row["next_light_distance_m"] <= 80
    counts.update(
        applicable_light=1,
        known_state=int(known),
        known_within_80m=int(known and in_range),
        known_no_housing_in_any_image=int(known and all_out),
        known_within_80m_no_housing=int(known and in_range and all_out),
        red_no_housing_in_any_image=int(row["next_light_state"] == "Red" and all_out),
        known_housing_in_any_image=int(known and any_head),
        known_front_no_housing=int(known and views["front"]["housing_in_frustum"] is False),
        indeterminate_geometry=int(not any_head and not all_out),
    )
    heights = [b["height_px"] for v in views.values() for b in v["boxes"] if b["in_image"] is True]
    if heights:
        counts["known_any_head_at_least_6px"] += int(known and max(heights) >= 6)
    return counts


def representative(rows):
    # Deterministic image QA, not outcome filtering: closest route-label distance
    # to20m after5s; no-light episodes use their middle sample.
    applicable = [
        r
        for r in rows
        if r["signal"] is not None
        and r["timestamp"] >= 5
        and r["next_light_distance_m"] is not None
    ]
    return (
        min(applicable, key=lambda r: (abs(r["next_light_distance_m"] - 20), r["frame"]))
        if applicable
        else rows[len(rows) // 2]
    )


def contact_sheet(episodes, destination):
    width, tile_height = 512 * 3, 320
    canvas = Image.new("RGB", (width, tile_height * len(episodes) + 36), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        "TRAINING LABEL AUDIT | cyan: light housing; amber: evaluator lines | visibility UNKNOWN",
        fill="white",
    )
    for index, episode in enumerate(episodes):
        r = representative(episode["rows"])
        packet = load_policy_sample(r["shard"], r["record"])
        y = 36 + index * tile_height
        distance = r["next_light_distance_m"]
        distance = f"{distance:.1f}" if distance is not None else "unknown"
        caption = (
            f"route {episode['route_id']} {episode['family']} t={r['timestamp']:.1f}s "
            f"state={r['next_light_state']} route distance={distance}m"
        )
        draw.text((10, y + 3), caption, fill="white")
        for col, view in enumerate(VIEWS):
            image = Image.fromarray(packet.rgb[view]).copy()
            local = ImageDraw.Draw(image)
            if r["signal"]:
                geometry = r["signal"]["views"][view]
                for box in geometry["boxes"]:
                    if box["in_image"]:
                        local.rectangle(box["bbox_xyxy"], outline="cyan", width=2)
                for line in geometry["evaluator_line_candidates"]:
                    if line["pixels"]:
                        local.line([tuple(p) for p in line["pixels"]], fill="orange", width=2)
            local.text((8, 6), view.upper(), fill="white", stroke_width=1, stroke_fill="black")
            canvas.paste(image, (512 * col, y + 24))
    canvas.save(destination)


def plot_events(episodes, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(episodes), 1, figsize=(12, 2.1 * len(episodes)), constrained_layout=True
    )
    for ax, episode in zip(np.atleast_1d(axes), episodes):
        rows = episode["rows"]
        t = [r["timestamp"] for r in rows]
        for key, label in [
            ("teacher_target_speed_mps", "Expert target"),
            ("student_target_speed_mps", "Student request"),
            ("ego_speed_mps", "Actual speed (assisted)"),
        ]:
            ax.plot(t, [r[key] for r in rows], label=label, linewidth=1.2)
        ax.set(title=f"{episode['route_id']} — {episode['family']}", ylabel="m/s")
        ax.set_ylim(bottom=-0.4)  # Preserve high requested targets; no artificial upper ceiling.
        ax.axvspan(
            t[0], min(t[-1], t[0] + 3.2), color="grey", alpha=0.15, label="3.2s clip reference"
        )
        ax.grid(alpha=0.2)
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Episode time (s); collection has privileged assistance")
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def assess(assessment_path, geometry_root, out):
    assessment, geometry = read(assessment_path), read(geometry_root / "geometry.json")
    if out.exists():
        raise FileExistsError("Fresh audit output required")
    if read(geometry_root / "progress.json")["status"] != "complete":
        raise ValueError("Completed geometry capture required")
    verify_sources(geometry_root, read(geometry_root / "request.json"))
    if geometry["town"] != "Town03_Opt":
        raise ValueError("This pilot audit only covers Town03 TRAIN episodes")
    inventory = read(ROOT / "outputs/navigation_encounter_inventory_v2_20261009/Town03_Opt.json")
    if geometry["opendrive_sha256"] != inventory["opendrive_sha256"]:
        raise ValueError("Recorded training town OpenDRIVE changed")
    paths = [Path(p) for p in assessment["eligible_correction_paths"]]
    if not paths:
        raise ValueError("No audited corrections")
    metas = [validate_training_data(p, assessment) for p in paths]
    if any(m["town"] != geometry["town"] for m in metas):
        raise ValueError("Episode/geometry town mismatch")
    out.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/navigation/rule_geometry.py",
        ROOT / "src/navigation/policy_worker.py",
    ]
    hashes = {}
    for source in sources:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(source.relative_to(ROOT))] = digest(source)
    episodes, counts = [], Counter()
    with (out / "samples.jsonl").open("x") as stream:
        for data, meta in zip(paths, metas):
            rows, local = [], Counter()
            for shard in sorted(data.glob("shard_*.zip")):
                with zipfile.ZipFile(shard) as archive:
                    for record in shard_records(shard):
                        prefix = record["prefix"]
                        policy, targets = (
                            json.loads(archive.read(f"{prefix}/{name}.json"))
                            for name in ("policy", "targets")
                        )
                        facts = targets["facts"]
                        if (
                            targets["frame"] != policy["frame"]
                            or facts["world"]["frame"] != policy["frame"]
                        ):
                            raise ValueError(
                                "Projection requires matching camera/world/label frames"
                            )
                        if rows and (
                            policy["frame"] != rows[-1]["frame"] + 2
                            or abs(policy["timestamp"] - rows[-1]["timestamp"] - 0.1) > 1e-5
                        ):
                            raise ValueError("Stop events require contiguous10Hz samples")
                        c = {k: CameraCalibration(**v) for k, v in policy["calibration"].items()}
                        signal = project_rule_sample(facts, c, geometry)
                        row = {
                            "route_id": meta["route_id"],
                            "frame": policy["frame"],
                            "timestamp": policy["timestamp"],
                            "shard": str(shard),
                            "record": record,
                            "next_light_state": facts["next_light_state"],
                            "next_light_distance_m": facts["next_light_distance_m"],
                            "teacher_target_speed_mps": facts["teacher_target_speed_mps"],
                            "teacher_hazards": facts["teacher_hazards"],
                            "student_target_speed_mps": facts["student_target_speed_mps"],
                            "ego_speed_mps": policy["ego_motion"][0],
                            "signal": signal,
                        }
                        rows.append(row)
                        local.update(count_sample(row))
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
            episodes.append(
                {
                    "route_id": meta["route_id"],
                    "family": meta["family"],
                    "counts": dict(local),
                    "stop_events": stopped_runs(rows),
                    "rows": rows,
                }
            )
            counts.update(local)
    contact_sheet(episodes, out / "projected_images.png")
    plot_events(episodes, out / "stop_release_history.png")
    result = {
        "audit_complete": True,
        "scope": "TRAIN corrections only; no policy result, visibility certification, "
        "or state-label release",
        "counts": dict(counts),
        "episodes": [{k: v for k, v in e.items() if k != "rows"} for e in episodes],
        "provenance": {
            "assessment_sha256": digest(assessment_path),
            "geometry_sha256": digest(geometry_root / "geometry.json"),
            "samples_sha256": digest(out / "samples.jsonl"),
            "source_hashes": hashes,
        },
        "visibility_valid": None,
        "state_supervision_approved": False,
        "limitations": [
            "World box projections are geometric candidates, not pixel segmentation",
            "Occlusion, housing facing and bulb legibility remain unknown",
            "All evaluator stop-line lanes shown; no applicable lane/painted-line target selected",
            "Only six familiar TRAIN routes; frames and stops are correlated",
            "Assisted actual speed is not student driving performance",
        ],
    }
    atomic_json(out / "assessment.json", result)
    print(json.dumps({"out": str(out), "counts": dict(counts)}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assessment",
        type=Path,
        default=ROOT / "reports/navigation_m2_dagger_assessment_20261016.json",
    )
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assess(args.assessment, args.geometry, args.out)


if __name__ == "__main__":
    main()
