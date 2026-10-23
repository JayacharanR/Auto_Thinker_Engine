#!/usr/bin/env python3
"""Paired FP32/FP16 latency and sparse-label parity, fixed selected RGB component."""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_crops import crop_feature  # noqa: E402
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import make_signal_detector  # noqa: E402


@torch.inference_mode()
def assess(args):
    if args.out.exists():
        raise FileExistsError("Fresh precision diagnostic output required")
    started = time.monotonic()
    torch.set_num_threads(2)
    dataset = SignalDetectionData(args.corpus)
    checkpoint_path = args.detector / "best_component.pt"
    review = json.loads((args.detector / "review.json").read_text())
    if digest(checkpoint_path) != review["component_sha256"] or review["selected_step"] != 50:
        raise ValueError("Frozen selected detector changed")
    classifier = json.loads((args.classifier / "assessment.json").read_text())
    linear_path = args.classifier / "robust_rgb_linear.npz"
    if digest(linear_path) != classifier["artifact_sha256"]:
        raise ValueError("Frozen crop classifier changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    corpus_sha = digest(args.corpus / "data_audit.json")
    if state["request"]["corpus_audit_sha256"] != corpus_sha:
        raise ValueError("Frozen detector/corpus mismatch")
    if json.loads((args.classifier / "request.json").read_text())["corpus_sha256"] != corpus_sha:
        raise ValueError("Frozen classifier/corpus mismatch")
    with np.load(linear_path, allow_pickle=False) as archive:
        linear = {k: archive[k].copy() for k in ("coef", "intercept", "classes")}
    check_resources(args.out, 25)
    model = make_signal_detector()
    model.load_state_dict(state["model"], strict=True)
    model = model.eval().to("cuda")
    variants = ["fp32", "fp16"]
    if args.include_cap200:
        variants.append("fp16_cap200")
    proposal_caps = {variant: (200 if variant == "fp16_cap200" else 1000) for variant in variants}
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/navigation/signal_detector.py",
        ROOT / "src/navigation/signal_detection_data.py",
        ROOT / "scripts/assess_navigation_signal_crops.py",
        ROOT / "scripts/assess_navigation_signal_detector.py",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            kind="FP32/FP16 controlled inference diagnostic, no fitting/driving",
            variants=variants,
            rpn_post_nms_proposals=proposal_caps,
            selected_detector_sha256=review["component_sha256"],
            classifier_sha256=classifier["artifact_sha256"],
            corpus_sha256=corpus_sha,
            confidence=0.5,
            iou_thresholds=[0.3, 0.5],
            warmup_batches_per_variant=3,
            paired_order="Rotate first mode on consecutive identical three-view batches",
            cap_seconds=300,
            inference_inputs="RGB only",
            channels_last=False,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started >= 300:
            raise InterruptedError("STOP or bounded precision cap")

    def progress(status, done, phase, **extra):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 precision diagnostic",
                images_done=done,
                images_total=144 * len(variants),
                phase=phase,
                updated_at=time.time(),
                **extra,
            ),
        )

    rows, timings, peak = {k: [] for k in variants}, {k: [] for k in variants}, {}
    progress("running", 0, "warmup both precisions")
    done = 0
    try:
        warmup, _ = dataset.batch(dataset.base.validation[:3], "cuda")
        for variant in rows:
            model.rpn._post_nms_top_n["testing"] = proposal_caps[variant]
            for _ in range(3):
                check()
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=variant.startswith("fp16")
                ):
                    model(warmup)
            torch.cuda.synchronize()
        for offset in range(0, 144, 3):
            indices = dataset.base.validation[offset : offset + 3]
            rgb, _ = dataset.batch(indices, "cuda")
            pivot = (offset // 3) % len(variants)
            order = variants[pivot:] + variants[:pivot]
            for variant in order:
                check()
                model.rpn._post_nms_top_n["testing"] = proposal_caps[variant]
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.monotonic()
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=variant.startswith("fp16")
                ):
                    outputs = model(rgb)  # Same RGB only, no target/geometry inputs.
                torch.cuda.synchronize()
                timings[variant].append(time.monotonic() - begin)
                peak[variant] = max(peak.get(variant, 0), torch.cuda.max_memory_allocated())
                for index, output in zip(indices, outputs):
                    array = dataset.base.images[index].transpose(1, 2, 0)
                    predicted = []
                    for box, score in zip(
                        output["boxes"].cpu().tolist(), output["scores"].cpu().tolist()
                    ):
                        if score < 0.5:
                            continue
                        integer = [
                            max(0, math.floor(box[0])),
                            max(0, math.floor(box[1])),
                            min(512, math.ceil(box[2])),
                            min(288, math.ceil(box[3])),
                        ]
                        if integer[2] <= integer[0] or integer[3] <= integer[1]:
                            continue
                        logits = (
                            crop_feature(array, integer) @ linear["coef"].T + linear["intercept"]
                        )
                        predicted.append(
                            dict(
                                box=box, score=score, state=int(linear["classes"][logits.argmax()])
                            )
                        )
                    rows[variant].append(
                        dict(sample=dataset.base.samples[index], predictions=predicted)
                    )
                done += 3
                progress("running", done, "paired RGB inference")
        results = {}
        for variant in rows:
            counts, assignments = audit_predictions(rows[variant], dataset.labels)
            results[variant] = dict(
                rpn_post_nms_proposals=proposal_caps[variant],
                one_to_one=counts,
                predictions=sum(len(r["predictions"]) for r in rows[variant]),
                mean_three_view_seconds=float(np.mean(timings[variant])),
                p95_three_view_seconds=float(np.percentile(timings[variant], 95)),
                peak_allocated_gb=peak[variant] / 1e9,
            )
            atomic_json(
                args.out / f"predictions_{variant}.json",
                dict(rows=rows[variant], assignments=assignments),
            )
        assert (
            results["fp32"]["one_to_one"]["iou3"] == review["validation"]["counts"]["detected_iou3"]
        )
        assert (
            results["fp32"]["one_to_one"]["iou3_state_correct"]
            == classifier["augmented_detected_state_correct"]
        )
        unchanged = all(
            torch.equal(tensor.cpu(), state["model"][name])
            for name, tensor in model.state_dict().items()
        )
        if not unchanged:
            raise ValueError("Precision audit mutated checkpoint parameters/buffers")
        atomic_json(
            args.out / "assessment.json",
            dict(
                diagnostic_complete=True,
                results=results,
                parameters_and_buffers_unchanged=True,
                wall_seconds=time.monotonic() - started,
                scope="TRAIN precision diagnostic; no driving or full pipeline timing",
            ),
        )
        progress("complete", done, "diagnostic finished")
        print(json.dumps(results, indent=2))
    except BaseException as exc:
        progress(
            "stopped" if isinstance(exc, InterruptedError) else "failed",
            done,
            "diagnostic incomplete",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-cap200", action="store_true")
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        assess(args)
