#!/usr/bin/env python3
"""Bounded small-anchor adaptation with ignored unknown lamps, never a driver."""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection.transform import resize_boxes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_reference_detector import iou  # noqa: E402
from scripts.assess_navigation_signal_crops import crop_feature  # noqa: E402
from scripts.assess_navigation_signal_detector import audit_predictions, match_boxes  # noqa: E402
from scripts.train_navigation_student import atomic_checkpoint, restore_rng, rng_state  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import (  # noqa: E402
    RunLock,
    atomic_json,
    check_resources,
    resource_shortfall,
)
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import detector_training, make_signal_detector  # noqa: E402


class BudgetStop(Exception):
    pass


def frozen_digest(model):
    trained = {k for k, v in model.named_parameters() if v.requires_grad}
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name not in trained:
            result.update(name.encode())
            result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def evaluate(model, dataset, linear, rng, stop_check, update, *, proposal_audit=False):
    saved_rng = rng_state(rng)
    torch.manual_seed(20261017)
    torch.cuda.manual_seed_all(20261017)
    losses, rows, timings = [], [], []
    counts = dict(
        centers=0,
        detected_iou3=0,
        detected_iou5=0,
        detected_iou3_state_correct=0,
        images=0,
        predictions=0,
        approximate_background_candidates=0,
    )
    height_groups = defaultdict(lambda: [0, 0])
    proposal_batches, proposal_hook = [], None
    proposal_counts = dict(denominator=0, iou3=0, iou5=0, total_proposals=0)
    if proposal_audit:

        def retain_proposals(module, inputs, output):
            if not module.training:
                proposal_batches.append((output[0], inputs[0].image_sizes))

        proposal_hook = model.rpn.register_forward_hook(retain_proposals)
    try:
        detector_training(model)
        for offset in range(0, len(dataset.base.validation), 2):
            stop_check()
            rgb, targets = dataset.batch(dataset.base.validation[offset : offset + 2], "cuda")
            error = model(rgb, targets)
            value = sum(error.values()).item()
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite validation loss")
            losses.append(value)
            if offset % 24 == 0:
                update("validation loss", offset)
        model.eval()
        for offset in range(0, len(dataset.base.validation), 3):
            stop_check()
            indices = dataset.base.validation[offset : offset + 3]
            rgb, _ = dataset.batch(indices, "cuda")
            torch.cuda.synchronize()
            begin = time.monotonic()
            outputs = model(rgb)  # RGB ONLY. No geometry/targets supplied.
            torch.cuda.synchronize()
            timings.append(time.monotonic() - begin)
            if proposal_audit:
                if len(proposal_batches) != 1:
                    raise ValueError("Exactly one RGB-only proposal batch expected")
                proposals, resized_sizes = proposal_batches.pop()
                for index, boxes, size in zip(indices, proposals, resized_sizes):
                    native = resize_boxes(boxes, size, (288, 512)).cpu().tolist()
                    truth = dataset.targets[index]["boxes"].tolist()
                    proposal_counts["denominator"] += len(truth)
                    proposal_counts["total_proposals"] += len(native)
                    proposal_counts["iou3"] += len(match_boxes(truth, native, 0.3))
                    proposal_counts["iou5"] += len(match_boxes(truth, native, 0.5))
            for index, output in zip(indices, outputs):
                sample = dataset.base.samples[index]
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
                    logits = crop_feature(array, integer) @ linear["coef"].T + linear["intercept"]
                    predicted.append(
                        dict(box=box, score=score, state=int(linear["classes"][logits.argmax()]))
                    )
                    # Diagnostic only: a prediction center outside all geometry ignore regions.
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    ignored = dataset.targets[index]["ignore_boxes"]
                    covered = (
                        (ignored[:, 0] <= cx)
                        & (cx <= ignored[:, 2])
                        & (ignored[:, 1] <= cy)
                        & (cy <= ignored[:, 3])
                    ).any()
                    counts["approximate_background_candidates"] += int(not covered)
                matches = []
                for label in dataset.annotation_rows[index]:
                    box = label["crop_xyxy"]
                    best = max(predicted, key=lambda p: iou(p["box"], box), default=None)
                    overlap = iou(best["box"], box) if best else 0
                    truth = {"red": 0, "red_repeat": 0, "yellow": 1, "green": 2}[sample["phase"]]
                    counts["centers"] += 1
                    counts["detected_iou3"] += int(overlap >= 0.3)
                    counts["detected_iou5"] += int(overlap >= 0.5)
                    counts["detected_iou3_state_correct"] += int(
                        overlap >= 0.3 and best["state"] == truth
                    )
                    height = box[3] - box[1]
                    bucket = "under8px" if height < 8 else "8to15px" if height < 16 else "16pxplus"
                    height_groups[bucket][0] += int(overlap >= 0.3)
                    height_groups[bucket][1] += 1
                    matches.append(
                        dict(
                            head=label["head"],
                            best_iou=overlap,
                            truth=truth,
                            state=best["state"] if best else None,
                        )
                    )
                counts["images"] += 1
                counts["predictions"] += len(predicted)
                rows.append(dict(sample=sample, predictions=predicted, supported_matches=matches))
            if offset % 24 == 0:
                update("RGB-only validation inference", offset)
        assert counts["centers"] == 72 and counts["images"] == 144
        return dict(
            selection_score=float(np.mean(losses)),
            counts=counts,
            mean_three_view_seconds=float(np.mean(timings[1:])),
            p95_three_view_seconds=float(np.percentile(timings[1:], 95)),
            by_native_height=dict(height_groups),
            **({"rpn_proposals": proposal_counts} if proposal_audit else {}),
        ), rows
    finally:
        if proposal_hook is not None:
            proposal_hook.remove()
        restore_rng(saved_rng, rng)


def train(args):
    hard_mode = args.parent is not None
    joint_mode = args.supplement is not None
    if joint_mode and not hard_mode:
        raise ValueError("Expanded joint adaptation requires protected parent and negatives")
    if any((args.parent, args.hard_negatives, args.negative_release)) and not all(
        (args.parent, args.hard_negatives, args.negative_release)
    ):
        raise ValueError("Parent, mined negatives and semantic release are required together")
    if args.out.exists() and not args.resume:
        raise FileExistsError("Fresh output required, or explicit provenance-checked --resume")
    if not 0 < args.max_wall <= 900:
        raise ValueError("Bounded experiment cap must be <=900 seconds")
    if args.resume and not (args.out / "latest.pt").exists():
        raise ValueError("No detector checkpoint to resume")
    reason = resource_shortfall(dict(minimum_free_vram_gb=4, minimum_available_ram_gb=4))
    if reason:
        raise RuntimeError(reason)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    random.seed(20261016)
    np.random.seed(20261016)
    rng = np.random.default_rng(20261016)
    dataset = SignalDetectionData(args.corpus)
    classifier = json.loads((args.classifier / "assessment.json").read_text())
    linear_path = args.classifier / (
        "robust_rgb_linear.npz" if hard_mode else "oracle_rgb_linear.npz"
    )
    if digest(linear_path) != classifier["artifact_sha256"]:
        raise ValueError("Crop classifier changed")
    classifier_request = json.loads((args.classifier / "request.json").read_text())
    classifier_corpus = classifier_request.get(
        "corpus_audit_sha256", classifier_request.get("corpus_sha256")
    )
    if classifier_corpus != digest(args.corpus / "data_audit.json"):
        raise ValueError("Crop classifier source corpus differs")
    with np.load(linear_path, allow_pickle=False) as archive:
        linear = {k: archive[k].copy() for k in ("coef", "intercept", "classes")}
    weights_path = ROOT / "outputs/navigation_models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    expected = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
    if digest(weights_path) != expected:
        raise ValueError("Official cached COCO weights changed")
    parent, negative_provenance = None, None
    if hard_mode:
        parent = json.loads((args.parent / "review.json").read_text())
        if (
            parent["component_sha256"] != digest(args.parent / "best_component.pt")
            or parent["selected_step"] != 50
        ):
            raise ValueError("Frozen parent changed")
        negative_provenance = dataset.attach_hard_negatives(
            args.hard_negatives, parent["component_sha256"], args.negative_release
        )
    supplement_provenance = dataset.append_training_corpus(args.supplement) if joint_mode else None
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/assess_navigation_signal_crops.py",
        ROOT / "scripts/assess_navigation_reference_detector.py",
        ROOT / "scripts/train_navigation_student.py",
        *sorted((ROOT / "src/navigation").glob("*.py")),
        ROOT / "reports/navigation_m2_signal_detector_protocol_20261016.md",
        *(
            [
                ROOT / "reports/navigation_m2_signal_hard_negative_protocol_20261016.md",
                ROOT / "reports/navigation_m2_signal_negative_semantics_protocol_20261016.md",
                ROOT / "scripts/assess_navigation_signal_detector.py",
            ]
            if hard_mode
            else []
        ),
        *(
            [ROOT / "reports/navigation_m2_signal_expansion_protocol_20261016.md"]
            if joint_mode
            else []
        ),
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    request = dict(
        kind="RGB-only perception component, no driving",
        architecture="signal-detector-v1",
        training_mode="protected-joint-expanded"
        if joint_mode
        else "protected-hard-negative-classifier"
        if hard_mode
        else "initial-RPN-adaptation",
        parent_sha256=parent["component_sha256"] if hard_mode else None,
        negative_provenance=negative_provenance,
        supplement_provenance=supplement_provenance,
        proposal_audit=joint_mode,
        corpus_audit_sha256=digest(args.corpus / "data_audit.json"),
        coco_sha256=expected,
        crop_classifier_sha256=classifier["artifact_sha256"],
        source_hashes=hashes,
        seed=20261016,
        steps=200,
        batch_size=2,
        learning_rate=0.0005,
        maximum_wall_seconds=args.max_wall,
        torch=torch.__version__,
        torchvision=torchvision.__version__,
    )
    if args.resume:
        if json.loads((args.out / "request.json").read_text()) != request:
            raise ValueError("Refuse resume with changed worker/config/data")
        if any(digest(args.out / "source" / path) != sha for path, sha in hashes.items()):
            raise ValueError("Archived worker changed")
    check_resources(args.out, 25)
    if not args.resume:
        atomic_json(args.out / "request.json", request)
        for source in sources:
            target = args.out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(args.out / "dataset.json", dataset.base.manifest)
    model = make_signal_detector(torch.load(weights_path, map_location="cpu", weights_only=True))
    if hard_mode:
        parent_state = torch.load(
            args.parent / "best_component.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(parent_state["model"], strict=True)
        model.requires_grad_(False)
        if joint_mode:
            model.rpn.requires_grad_(True)
            model.roi_heads.box_predictor.requires_grad_(True)
        else:
            model.roi_heads.box_predictor.cls_score.requires_grad_(True)
        expected_trainable = 1194265 if joint_mode else 2050
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != expected_trainable:
            raise ValueError("Unexpected protected refinement trainable-parameter count")
    model = model.to("cuda")
    frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.0005, weight_decay=0.0001
    )
    step, selected_step, best, previous_seconds = 0, 0, float("inf"), 0.0
    validation = None
    validation_step = -1
    if args.resume:
        saved = torch.load(args.out / "latest.pt", map_location="cuda", weights_only=False)
        if saved["request"] != request or saved["frozen_sha256"] != frozen:
            raise ValueError("Checkpoint provenance/frozen weights mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], rng)
        step, selected_step, best = saved["step"], saved["selected_step"], saved["best_score"]
        previous_seconds, validation = saved["elapsed_seconds"], saved["validation"]
        validation_step = saved["validation_step"]

    def elapsed():
        return previous_seconds + time.monotonic() - started

    def stop_check():
        if (args.out / "STOP").exists() or elapsed() >= args.max_wall:
            raise BudgetStop("STOP or cumulative wall cap")

    def progress(phase, validation_images=0, status="running", **extra):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 signal detector",
                steps=step,
                target_steps=200,
                phase=phase,
                validation_images=validation_images,
                validation_total=144,
                elapsed_seconds=elapsed(),
                updated_at=time.time(),
                selected_step=selected_step,
                validation=validation,
                **extra,
            ),
        )

    def save(improved=False):
        state = dict(
            kind="perception_component_only",
            architecture="signal-detector-v1",
            step=step,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(rng),
            request=request,
            validation=validation,
            validation_step=validation_step,
            frozen_sha256=frozen,
            selected_step=selected_step,
            best_score=best,
            elapsed_seconds=elapsed(),
        )
        atomic_checkpoint(args.out / "latest.pt", state)
        if improved:
            atomic_checkpoint(args.out / "best_component.pt", state)

    def validate_and_save():
        nonlocal validation, validation_step, best, selected_step
        validation, predictions = evaluate(
            model, dataset, linear, rng, stop_check, progress, proposal_audit=joint_mode
        )
        validation_step = step
        eligible = True
        if hard_mode:
            exact, _ = audit_predictions(predictions, dataset.labels)
            validation["one_to_one"] = exact
            counts = validation["counts"]
            if step == 0 and (
                exact["iou3"],
                exact["iou3_state_correct"],
                counts["approximate_background_candidates"],
            ) != (30, 24, 681):
                raise ValueError("Parent0 does not reproduce fixed selected50+robust classifier")
            eligible = (
                exact["iou3"] >= 30
                and exact["iou3_state_correct"] >= 24
                and counts["approximate_background_candidates"] <= 681
            )
            validation["parent_guard_passed"] = eligible
        improved = eligible and validation["selection_score"] < best
        if improved:
            best, selected_step = validation["selection_score"], step
        atomic_json(args.out / f"predictions_step{step}.json", dict(rows=predictions))
        with (args.out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, **validation)) + "\n")
        save(improved)
        print(json.dumps(dict(step=step, validation=validation)), flush=True)

    status = "complete"
    try:
        progress("initial validation")
        if validation is None or (step in (50, 100, 200) and validation_step != step):
            validate_and_save()
        while step < 200:
            stop_check()
            detector_training(model)
            rgb, targets = dataset.batch(dataset.sample_train(rng), "cuda")
            optimizer.zero_grad(set_to_none=True)
            losses = model(rgb, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite detector loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if step % 10 == 0:
                progress("training detector heads", loss=float(loss.detach()))
            if step in (50, 100, 200):
                validate_and_save()
            elif step % 25 == 0:
                save()
    except BudgetStop:
        status = "stopped"
    except BaseException as exc:
        save()
        progress(
            "failed; inspect before resume", status="failed", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    unchanged = frozen_digest(model) == frozen
    if not unchanged:
        progress("frozen feature mutation", status="failed")
        raise ValueError("Frozen backbone/ROI parameters or buffers changed")
    save()
    selected_path = args.out / "best_component.pt"
    if selected_path.exists():
        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
        review = dict(
            status=status,
            selected_step=selected["step"],
            validation=selected["validation"],
            wall_seconds=elapsed(),
            frozen_parameters_and_buffers_unchanged=unchanged,
            frozen_sha256=frozen,
            component_sha256=digest(selected_path),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            scope="TRAIN static RGB detector diagnostic; no driving/policy promotion",
        )
        atomic_json(args.out / "review.json", review)
    progress("component diagnostic finished", status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--hard-negatives", type=Path)
    parser.add_argument("--negative-release", type=Path)
    parser.add_argument(
        "--supplement", type=Path, help="Frozen TRAIN expansion; enables joint RPN/head fit"
    )
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        train(args)


if __name__ == "__main__":
    main()
