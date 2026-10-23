#!/usr/bin/env python3
"""Frozen camera-network response to paired TRAIN renders, not a driving score."""

import argparse
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.contracts import VIEWS  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.policy_runtime import CameraPolicy  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, resource_shortfall  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def approved_inputs(images, navigation, yaw, device, steps=32):
    """The same resize/axis/scales as CameraPolicy; stationary ego, no light facts."""
    if tuple(images) != VIEWS or np.asarray(navigation).shape != (16, 2):
        raise ValueError("Three RGB views and sixteen route waypoints required")
    for image in images.values():
        if image.shape != (288, 512, 3) or image.dtype != np.uint8:
            raise ValueError("Original512x288 RGB required")
    context = np.stack(
        [
            np.asarray(
                Image.fromarray(images[view]).resize((256, 144), Image.Resampling.BILINEAR)
            ).transpose(2, 0, 1)
            for view in VIEWS
        ]
    )
    values = {
        "rgb": context,
        "front": images["front"].transpose(2, 0, 1),
        "navigation": np.asarray(navigation, np.float32),
        "ego": np.array([0, yaw, 0], np.float32),
    }
    return {
        key: torch.from_numpy(np.ascontiguousarray(value))
        .to(device)[None, None]
        .expand(1, steps, *value.shape)
        .contiguous()
        for key, value in values.items()
    }


def decode(prediction, index):
    return {
        "light_probabilities_none_red_yellow_green_other": prediction["light"][0, index]
        .float()
        .softmax(-1)
        .cpu()
        .tolist(),
        "light_class": int(prediction["light"][0, index].argmax().item()),
        "stop_probability": float(prediction["brake"][0, index].float().sigmoid().item()),
        "target_speed_mps": float(prediction["current_speed"][0, index].float().item()),
    }


@torch.inference_mode()
def assess(probe, checkpoint, out, device="cuda"):
    if out.exists():
        raise FileExistsError("Fresh response output required")
    assessment, request = read(probe / "assessment.json"), read(probe / "request.json")
    if not assessment["probe_complete"] or read(probe / "progress.json")["status"] != "complete":
        raise ValueError("Completed paired pixel probe required")
    if any(digest(probe / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Paired-render source archive changed")
    if digest(probe / "cases.json") != assessment["source_cases_sha256"]:
        raise ValueError("Frozen TRAIN cases changed")
    expected = {case["route_id"]: case for case in read(probe / "cases.json")}
    if set(expected) != {case["route_id"] for case in assessment["cases"]}:
        raise ValueError("Incomplete render cases")
    if any(
        c["states"] != ["Red", "Green", "Red"]
        or not c["all_lights_frozen"]
        or c["ego_position_drift_m"] > 1e-3
        or max(c["camera_matrix_max_error"].values()) > 1e-3
        for c in assessment["cases"]
    ):
        raise ValueError("Static render setup did not pass")
    if device != "cpu":
        reason = resource_shortfall({"minimum_free_vram_gb": 2, "minimum_available_ram_gb": 3})
        if reason:
            raise RuntimeError(reason)
    out.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/navigation/policy_runtime.py",
        ROOT / "src/navigation/student.py",
    ]
    for source in sources:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        out / "request.json",
        {
            "kind": "Frozen policy image response; static same-pose TRAIN renders, not driving",
            "checkpoint_sha256": digest(checkpoint),
            "probe_assessment_sha256": digest(probe / "assessment.json"),
            "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "branch_history": "Shared32-frame red history; copy hidden into each32-frame branch",
            "positions_after_branch": [1, 8, 32],
            "precision": "float32 eval; no optimizer or update",
        },
    )
    # Reuse the production constructor, which performs strict state_dict loading.
    policy = CameraPolicy(checkpoint, device=device, rule_overrides=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not all(
        torch.equal(value.cpu(), saved["model"][name])
        for name, value in policy.model.state_dict().items()
    ):
        raise ValueError("Loaded parameters differ from selected checkpoint")
    rows, image_hashes = [], {}
    for rid, case in sorted(expected.items()):
        with zipfile.ZipFile(case["source_shard"]) as archive:
            record = case["source_record"]
            payload = archive.read(f"{record['prefix']}/policy.json")
            import hashlib

            if hashlib.sha256(payload).hexdigest() != record["sha256"]["policy.json"]:
                raise ValueError("Recorded sensor/navigation packet changed")
            packet = json.loads(payload)
        if case["town"] != "Town03_Opt" or case["original_frame"] != packet["frame"]:
            raise ValueError("Wrong original TRAIN pose")
        yaw = math.radians(case["ego_pose"][4])
        inputs = {}
        for state in ("red", "green", "red_repeat"):
            images = {}
            for view in VIEWS:
                path = probe / f"route{rid}_{state}_{view}.png"
                image_hashes[path.name] = digest(path)
                with Image.open(path) as image:
                    images[view] = np.asarray(image.convert("RGB")).copy()
            inputs[state] = approved_inputs(images, packet["navigation_waypoints"], yaw, device)
        _, common_hidden = policy.model(**inputs["red"])
        branches = {}
        for state, model_inputs in inputs.items():
            prediction, _ = policy.model(**model_inputs, hidden=common_hidden.clone())
            branches[state] = {str(index + 1): decode(prediction, index) for index in (0, 7, 31)}
        rows.append({"route_id": rid, "branches": branches})
    unchanged = all(
        torch.equal(value.cpu(), saved["model"][name])
        for name, value in policy.model.state_dict().items()
    )
    if not unchanged:
        raise RuntimeError("Diagnostic mutated checkpoint parameters")
    result = {
        "diagnostic_complete": True,
        "model_parameters_equal_checkpoint_before_and_after": unchanged,
        "checkpoint_sha256": digest(checkpoint),
        "rows": rows,
        "render_image_sha256": image_hashes,
        "scope": "Static TRAIN image sensitivity; repeated images are not a driving rollout",
        "limitations": [
            "Original traffic omitted",
            "Only four familiar poses",
            "Shared red history biases the initial prior but is identical across branches",
            "No stop/go behavioral pass rate or model improvement is claimed",
        ],
    }
    atomic_json(out / "assessment.json", result)
    print(
        json.dumps(
            {
                "diagnostic_complete": True,
                "out": str(out),
                "parameters_unchanged": unchanged,
                "final_predictions": [
                    {
                        "route_id": row["route_id"],
                        **{
                            state: row["branches"][state]["32"]
                            for state in ("red", "green", "red_repeat")
                        },
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/navigation_m2_stop_go_20261016/best_policy.pt",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        assess(args.probe, args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
