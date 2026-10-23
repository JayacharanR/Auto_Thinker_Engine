#!/usr/bin/env python3
"""Mine conservative TRAIN-only background ROIs with static phase controls."""

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_reference_detector import iou  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import make_signal_detector  # noqa: E402


def outside_all_housings(box, ignored):
    box, ignored = np.asarray(box), np.asarray(ignored).reshape(-1, 4)
    if not len(ignored):
        return True
    intersection = np.maximum(
        0, np.minimum(box[2:], ignored[:, 2:]) - np.maximum(box[:2], ignored[:, :2])
    ).prod(1)
    area = np.minimum(np.prod(box[2:] - box[:2]), np.prod(ignored[:, 2:] - ignored[:, :2], 1))
    return bool(np.all(intersection / np.maximum(area, 1e-8) <= 0.05))


def phase_invariant(images, box):
    x0, y0, x1, y1 = box
    pixels = {k: v[:, y0:y1, x0:x1].astype(np.float32) for k, v in images.items()}
    repeat = float(np.abs(pixels["red"] - pixels["red_repeat"]).mean())
    changes = {
        state: float(np.abs(pixels["red"] - pixels[state]).mean()) for state in ("green", "yellow")
    }
    return dict(
        repeat_mae=repeat,
        phase_mae=changes,
        passed=all(value <= 5 and value <= 2 * repeat + 1 for value in changes.values()),
    )


@torch.inference_mode()
def mine(args):
    if args.out.exists():
        raise FileExistsError("Fresh mining output required")
    started = time.monotonic()
    torch.set_num_threads(2)
    data = SignalDetectionData(args.corpus)
    parent_path = args.parent / "best_component.pt"
    parent = json.loads((args.parent / "review.json").read_text())
    expected_parent = "8df27ee218056e392cc5ddb0f50fef9263a1e8b81517b1d5ecbeba8038a6e49b"
    if parent["component_sha256"] != expected_parent:
        raise ValueError("Parent is not the predeclared selected50 component")
    if parent["selected_step"] != 50 or digest(parent_path) != parent["component_sha256"]:
        raise ValueError("Frozen parent changed")
    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    if saved["request"]["corpus_audit_sha256"] != digest(args.corpus / "data_audit.json"):
        raise ValueError("Parent and mining corpus differ")
    indices = [i for i in data.base.train if data.base.samples[i]["phase"] == "red"]
    if len(indices) != 117:
        raise ValueError("Expected original117 TRAIN red-phase images")
    lookup = {
        (s["town"], s["case_id"], s["view"], s["phase"]): i for i, s in enumerate(data.base.samples)
    }
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/navigation/signal_detector.py",
        ROOT / "src/navigation/signal_detection_data.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "reports/navigation_m2_signal_hard_negative_protocol_20261016.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            kind="TRAIN-only hard negative mining, no validation inference or driving",
            parent_sha256=parent["component_sha256"],
            corpus_audit_sha256=digest(args.corpus / "data_audit.json"),
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            training_red_images=117,
            confidence=0.5,
            maximum_candidates_per_image=2,
            maximum_wall_seconds=300,
        ),
    )
    model = make_signal_detector()
    model.load_state_dict(saved["model"], strict=True)
    model = model.eval().to("cuda")
    candidates, examined = [], 0

    def progress(status, done, **extra):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 TRAIN negative mining",
                images_done=done,
                images_total=117,
                phase="FP32 TRAIN-only mining",
                updated_at=time.time(),
                **extra,
            ),
        )

    progress("running", 0)
    try:
        for offset in range(0, len(indices), 3):
            if (args.out / "STOP").exists() or time.monotonic() - started >= 300:
                progress("stopped", offset)
                return
            batch = indices[offset : offset + 3]
            rgb, _ = data.batch(batch, "cuda")
            predictions = model(rgb)  # No privileged inputs.
            for index, prediction in zip(batch, predictions):
                sample = data.base.samples[index]
                key = sample["town"], sample["case_id"], sample["view"]
                phase_indices = {
                    phase: lookup[(*key, phase)]
                    for phase in ("red", "green", "yellow", "red_repeat")
                }
                if any(data.base.samples[i]["role"] != "train" for i in phase_indices.values()):
                    raise ValueError("Attempt to mine a non-TRAIN image")
                images = {phase: data.base.images[i] for phase, i in phase_indices.items()}
                accepted = []
                for box, score in zip(
                    prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist()
                ):
                    examined += 1
                    if score < 0.5 or not outside_all_housings(
                        box, data.targets[index]["ignore_boxes"]
                    ):
                        continue
                    if any(iou(box, other["box"]) > 0.3 for other in accepted):
                        continue
                    integer = [
                        max(0, math.floor(box[0])),
                        max(0, math.floor(box[1])),
                        min(512, math.ceil(box[2])),
                        min(288, math.ceil(box[3])),
                    ]
                    if integer[2] <= integer[0] or integer[3] <= integer[1]:
                        continue
                    response = phase_invariant(images, integer)
                    if not response["passed"]:
                        continue
                    accepted.append(
                        dict(box=box, crop_xyxy=integer, score=score, phase_control=response)
                    )
                    if len(accepted) == 2:
                        break
                if accepted:
                    candidates.append(
                        dict(
                            sample=sample,
                            phase_indices=phase_indices,
                            phase_paths={
                                p: data.base.samples[i]["path"] for p, i in phase_indices.items()
                            },
                            negatives=accepted,
                        )
                    )
            progress("running", min(offset + 3, 117))
        objects = sum(len(row["negatives"]) for row in candidates)
        groups = {(r["sample"]["town"], r["sample"]["junction_id"]) for r in candidates}
        towns = Counter(r["sample"]["town"] for r in candidates)
        passed = objects >= 20 and len(groups) >= 6 and len(towns) >= 3
        audit = dict(
            data_gate_passed=passed,
            canonical_candidates=objects,
            canonical_images=len(candidates),
            junctions=sorted(groups),
            towns=dict(towns),
            examined=examined,
            wall_seconds=time.monotonic() - started,
            parent_sha256=parent["component_sha256"],
            corpus_audit_sha256=digest(args.corpus / "data_audit.json"),
            scope="Conservative approximate TRAIN backgrounds; no exhaustive ground truth",
        )
        atomic_json(args.out / "manifest.json", dict(**audit, rows=candidates))
        all_crops = [(row, negative) for row in candidates for negative in row["negatives"]]
        selected = np.random.default_rng(20261016).permutation(len(all_crops))[:24]
        canvas = Image.new("RGB", (6 * 180, 4 * 145), "#18212d")
        draw = ImageDraw.Draw(canvas)
        for n, chosen in enumerate(selected):
            row, negative = all_crops[int(chosen)]
            image = Image.open(args.corpus / row["sample"]["path"]).convert("RGB")
            box = negative["crop_xyxy"]
            patch = image.crop(box)
            patch.thumbnail((170, 100))
            x, y = (n % 6) * 180, (n // 6) * 145
            canvas.paste(patch, (x + 5, y + 32))
            s = row["sample"]
            draw.text((x + 5, y + 4), f"{s['town']} J{s['junction_id']}", fill="white")
            draw.text(
                (x + 5, y + 18),
                f"score{negative['score']:.2f} {box[2] - box[0]}x{box[3] - box[1]}",
                fill="white",
            )
        canvas.save(args.out / "negative_contact_sheet.png")
        atomic_json(args.out / "audit.json", audit)
        progress("complete", 117, data_gate_passed=passed)
        print(json.dumps(audit, indent=2))
    except BaseException as exc:
        progress("failed", 0, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        mine(args)
