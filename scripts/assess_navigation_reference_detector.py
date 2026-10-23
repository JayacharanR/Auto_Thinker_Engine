#!/usr/bin/env python3
"""Frozen COCO detector + learned RGB crop-state baseline, unchanged TRAIN images."""

import argparse
import json
import math
import shutil
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_signal_crops import crop_feature  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_learning import SignalDataset  # noqa: E402


def iou(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    intersection = float(
        np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    )
    area_a, area_b = np.prod(np.maximum(0, a[2:] - a[:2])), np.prod(np.maximum(0, b[2:] - b[:2]))
    return intersection / max(1e-8, area_a + area_b - intersection)


@torch.inference_mode()
def assess(args):
    if args.out.exists():
        raise FileExistsError("Fresh detector output required")
    if not 0 < args.max_wall <= 900:
        raise ValueError("Detector probe must remain bounded")
    started, deadline = time.monotonic(), time.monotonic() + args.max_wall
    dataset = SignalDataset(args.corpus)
    audit = json.loads((args.corpus / "data_audit.json").read_text())
    classifier_report = json.loads((args.classifier / "assessment.json").read_text())
    if (
        not classifier_report["oracle_crops"]
        or digest(args.classifier / "oracle_rgb_linear.npz") != classifier_report["artifact_sha256"]
    ):
        raise ValueError("Verified crop-state baseline required")
    with np.load(args.classifier / "oracle_rgb_linear.npz", allow_pickle=False) as linear:
        coef, intercept, classes = (linear[k].copy() for k in ("coef", "intercept", "classes"))
    check_resources(args.out, 25)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    request = {
        "kind": "Frozen external-supervised RGB detector baseline, no driving/control",
        "weights_url": weights.url,
        "pretraining": "COCO supervised labels/compute; not a matched encoder ablation",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "native_rgb": [512, 288],
        "internal_resize": "Default min800,max1333; interpolates existing pixels, no new sensor",
        "confidence": 0.5,
        "iou_recall_thresholds": [0.3, 0.5],
        "corpus_audit_sha256": digest(args.corpus / "data_audit.json"),
        "classifier_sha256": classifier_report["artifact_sha256"],
        "maximum_wall_seconds": args.max_wall,
    }
    atomic_json(args.out / "request.json", request)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/assess_navigation_signal_crops.py",
        ROOT / "src/navigation/signal_learning.py",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "source_hashes.json", {str(p.relative_to(ROOT)): digest(p) for p in sources}
    )

    def progress(status, done, phase):
        atomic_json(
            args.out / "progress.json",
            {
                "status": status,
                "stage": "M2 reference detector",
                "images_done": done,
                "images_total": len(dataset.validation),
                "phase": phase,
                "updated_at": time.time(),
            },
        )

    progress("running", 0, "download/verify official weights")
    socket.setdefaulttimeout(30)
    cache = ROOT / "outputs/navigation_models"
    state = torch.hub.load_state_dict_from_url(
        weights.url, model_dir=str(cache), check_hash=True, progress=False, weights_only=True
    )
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=len(weights.meta["categories"])
    )
    model.load_state_dict(state, strict=True)
    model = model.eval().to(args.device)
    light_id = weights.meta["categories"].index("traffic light")
    labels = defaultdict(list)
    for label in audit["labels"]:
        if label["static_phase_response_supported"]:
            labels[(label["town"], label["case_id"], label["view"])].append(label)
    groups = defaultdict(list)
    for index in dataset.validation:
        sample = dataset.samples[index]
        groups[(sample["town"], sample["case_id"], sample["phase"])].append(index)
    counts = dict(
        centers=0,
        detected_iou3=0,
        detected_iou5=0,
        contained=0,
        detected_iou3_state_correct=0,
        images=0,
        traffic_light_predictions=0,
    )
    predictions, timing = [], []
    torch.cuda.reset_peak_memory_stats()
    for group_index, indices in enumerate(groups.values()):
        if (args.out / "STOP").exists() or time.monotonic() >= deadline:
            progress("stopped", counts["images"], "STOP or wall budget")
            return
        rgb = [
            torch.from_numpy(dataset.images[index].copy()).to(args.device).float() / 255
            for index in indices
        ]
        torch.cuda.synchronize()
        begin = time.monotonic()
        output = model(rgb)
        torch.cuda.synchronize()
        timing.append(time.monotonic() - begin)
        for index, detection in zip(indices, output):
            sample = dataset.samples[index]
            array = dataset.images[index].transpose(1, 2, 0)
            light_boxes = []
            for box, score, label in zip(
                detection["boxes"].cpu().tolist(),
                detection["scores"].cpu().tolist(),
                detection["labels"].cpu().tolist(),
            ):
                if label != light_id or score < 0.5:
                    continue
                x0, y0, x1, y1 = box
                integer = [
                    max(0, math.floor(x0)),
                    max(0, math.floor(y0)),
                    min(512, math.ceil(x1)),
                    min(288, math.ceil(y1)),
                ]
                if integer[2] <= integer[0] or integer[3] <= integer[1]:
                    continue
                scores = crop_feature(array, integer) @ coef.T + intercept
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                light_boxes.append(
                    {
                        "box": box,
                        "detector_score": score,
                        "state_index_red_yellow_green": int(classes[scores.argmax()]),
                        "state_probabilities": probability.tolist(),
                    }
                )
            local = []
            for ground_truth in labels[(sample["town"], sample["case_id"], sample["view"])]:
                if sample["phase"] == "yellow" and not ground_truth["yellow_color_supported"]:
                    continue
                box = ground_truth["crop_xyxy"]
                counts["centers"] += 1
                best = max(light_boxes, key=lambda pred: iou(pred["box"], box), default=None)
                overlap = iou(best["box"], box) if best else 0
                truth = {"red": 0, "red_repeat": 0, "yellow": 1, "green": 2}[sample["phase"]]
                counts["detected_iou3"] += int(overlap >= 0.3)
                counts["detected_iou5"] += int(overlap >= 0.5)
                counts["detected_iou3_state_correct"] += int(
                    overlap >= 0.3 and best["state_index_red_yellow_green"] == truth
                )
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                counts["contained"] += int(
                    any(
                        p["box"][0] <= cx <= p["box"][2] and p["box"][1] <= cy <= p["box"][3]
                        for p in light_boxes
                    )
                )
                local.append(
                    {
                        "head": ground_truth["head"],
                        "best_iou": overlap,
                        "truth_state": truth,
                        "matched_state": best["state_index_red_yellow_green"] if best else None,
                    }
                )
            counts["images"] += 1
            counts["traffic_light_predictions"] += len(light_boxes)
            predictions.append(
                {
                    "sample": sample,
                    "predicted_lights": light_boxes,
                    "supported_annotation_matches": local,
                }
            )
            if group_index == 0:
                image = Image.fromarray(array)
                draw = ImageDraw.Draw(image)
                for pred in light_boxes:
                    draw.rectangle(pred["box"], outline="cyan", width=2)
                draw.text(
                    (5, 5),
                    "RGB detector predictions | TRAIN diagnostic | no driving",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                image.save(args.out / f"example_{sample['view']}.png")
        progress("running", counts["images"], "frozen detector inference, no training")
    if counts["centers"] != 72:
        raise ValueError("Reference detector denominator differs from frozen component set")
    result = {
        "diagnostic_complete": True,
        "counts": counts,
        "mean_three_view_seconds": float(np.mean(timing[1:])),
        "p95_three_view_seconds": float(np.percentile(timing[1:], 95)),
        "peak_torch_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "weights_sha256": digest(cache / Path(weights.url).name),
        "wall_seconds": time.monotonic() - started,
        "scope": "RGB-only TRAIN diagnostics, no driving/lane result. "
        "Sparse projected geometry is not a full detection benchmark.",
    }
    atomic_json(args.out / "predictions.json", {"rows": predictions})
    atomic_json(args.out / "assessment.json", result)
    progress("complete", counts["images"], "diagnostic complete")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--max-wall", type=int, default=300)
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        try:
            assess(args)
        except BaseException as exc:
            path = args.out / "progress.json"
            if path.exists() and json.loads(path.read_text())["status"] == "running":
                atomic_json(
                    path,
                    {
                        **json.loads(path.read_text()),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raise


if __name__ == "__main__":
    main()
