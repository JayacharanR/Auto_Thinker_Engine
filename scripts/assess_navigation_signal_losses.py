#!/usr/bin/env python3
"""Decompose two fixed models' losses; GT ROI scores are privileged diagnostics."""

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_navigation_signal_detector import frozen_digest  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import detector_training, make_signal_detector  # noqa: E402


@torch.no_grad()
def assess(args):
    if args.out.exists():
        raise FileExistsError("Fresh diagnostic output required")
    started = time.monotonic()
    torch.set_num_threads(2)
    paths = dict(parent=args.parent / "best_component.pt", candidate=args.candidate / "latest.pt")
    saved = {k: torch.load(p, map_location="cpu", weights_only=False) for k, p in paths.items()}
    if saved["parent"]["step"] != 50 or saved["candidate"]["step"] != 200:
        raise ValueError("Only frozen parent50 and latest200")
    if saved["candidate"]["validation_step"] != 200:
        raise ValueError("Candidate scoring incomplete")
    if digest(paths["parent"]) != saved["candidate"]["request"]["parent_sha256"]:
        raise ValueError("Fixed parent changed")
    data = SignalDetectionData(args.corpus)
    expected = saved["candidate"]["request"]["corpus_audit_sha256"]
    if digest(args.corpus / "data_audit.json") != expected:
        raise ValueError("Original validation corpus changed")
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_navigation_signal_detector.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "scripts/collect_navigation_signal_pairs.py",
        *sorted((ROOT / "src/navigation").glob("*.py")),
        ROOT / "reports/navigation_m2_signal_loss_audit_protocol_20261016.md",
        *(
            [ROOT / "reports/navigation_m2_signal_sampling_audit_protocol_20261016.md"]
            if args.prior_audit is not None
            else []
        ),
    ]
    for p in sources:
        target = args.out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(
        args.out / "request.json",
        dict(
            model_sha256={k: digest(p) for k, p in paths.items()},
            corpus_audit_sha256=digest(args.corpus / "data_audit.json"),
            maximum_wall_seconds=300,
            prior_audit_sha256=digest(args.prior_audit / "assessment.json")
            if args.prior_audit is not None
            else None,
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        ),
    )
    positive_indices = [i for i in data.base.validation if len(data.targets[i]["boxes"])]
    prior = None
    if args.prior_audit is not None:
        prior_request = json.loads((args.prior_audit / "request.json").read_text())
        if prior_request["model_sha256"] != {k: digest(p) for k, p in paths.items()}:
            raise ValueError("Prior GT-ROI model identity differs")
        if prior_request["corpus_audit_sha256"] != digest(args.corpus / "data_audit.json"):
            raise ValueError("Prior GT-ROI corpus differs")
        prior = json.loads((args.prior_audit / "assessment.json").read_text())
    done, total = 0, 288 + (0 if prior is not None else len(positive_indices))

    def update(phase):
        if (args.out / "STOP").exists() or time.monotonic() - started >= 300:
            raise InterruptedError("STOP or diagnostic cap")
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                phase=phase,
                images_done=done,
                images_total=total,
                updated_at=time.time(),
            ),
        )

    results, model = {}, None
    for name, state in saved.items():
        if model is not None:
            del model
        model = make_signal_detector().cuda()
        model.load_state_dict(state["model"], strict=True)
        frozen = frozen_digest(model)
        detector_training(model)
        sampling = []
        if prior is not None:
            select = model.roi_heads.select_training_samples

            def counted(proposals, targets):
                output = select(proposals, targets)
                labels = output[2]
                sampling.append(
                    (sum(len(v) for v in labels), sum(int((v > 0).sum()) for v in labels))
                )
                return output

            model.roi_heads.select_training_samples = counted
        torch.manual_seed(20261017)
        torch.cuda.manual_seed_all(20261017)
        pieces, totals = defaultdict(list), []
        all_roi, positive_roi, box_sum = 0, 0, 0.0
        for offset in range(0, 144, 2):
            update(f"{name} loss components")
            images, targets = data.batch(data.base.validation[offset : offset + 2], "cuda")
            losses = model(images, targets)
            if prior is not None:
                assert len(sampling) == 1
                count, positives = sampling.pop()
                all_roi += count
                positive_roi += positives
                box_sum += float(losses["loss_box_reg"]) * count
            totals.append(float(sum(losses.values())))
            for key, value in losses.items():
                pieces[key].append(float(value))
            done += len(images)
        score = float(np.mean(totals))
        if not np.isclose(score, state["validation"]["selection_score"], atol=1e-6, rtol=0):
            raise ValueError(f"{name} loss did not reproduce: {score}")
        if frozen_digest(model) != frozen:
            raise ValueError("Frozen features or normalization changed")
        results[name] = dict(
            total=score,
            components={k: float(np.mean(v)) for k, v in pieces.items()},
            saved_total_reproduced=True,
            frozen_sha256=frozen,
        )
        if prior is not None:
            results[name]["roi_sampling"] = dict(
                all_samples=all_roi,
                positive_samples=positive_roi,
                box_smooth_l1_sum=box_sum,
                mean_box_smooth_l1_per_positive=box_sum / positive_roi,
            )
            results[name]["privileged_gt_roi"] = prior["results"][name]["privileged_gt_roi"]
    if prior is not None:
        result = dict(
            results=results,
            wall_seconds=time.monotonic() - started,
            reused_gt_roi_audit_sha256=digest(args.prior_audit / "assessment.json"),
            scope="Fixed sampler-normalization audit; no fitting or selection change",
        )
        atomic_json(args.out / "assessment.json", result)
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                images_done=done,
                images_total=total,
                phase="Sampling audit complete",
                updated_at=time.time(),
            ),
        )
        print(json.dumps(result, indent=2))
        return
    model.eval()
    rows, features = [], []
    for offset in range(0, len(positive_indices), 3):
        update("Privileged GT ROI features")
        indices = positive_indices[offset : offset + 3]
        images, targets = data.batch(indices, "cuda")
        images, targets = model.transform(images, targets)
        encoded = model.backbone(images.tensors)
        pooled = model.roi_heads.box_roi_pool(
            encoded, [t["boxes"] for t in targets], images.image_sizes
        )
        features.append(model.roi_heads.box_head(pooled).cpu())
        rows.extend(
            dict(sample=data.base.samples[i], head=label["head"])
            for i in indices
            for label in data.annotation_rows[i]
        )
        done += len(indices)
    x = torch.cat(features).numpy()
    assert len(rows) == len(x) == 72
    for name, state in saved.items():
        prefix = "roi_heads.box_predictor.cls_score."
        w, b = (state["model"][prefix + key].numpy() for key in ("weight", "bias"))
        margins = x @ (w[1] - w[0]) + b[1] - b[0]
        towns = defaultdict(lambda: [0, 0])
        for row, value in zip(rows, margins):
            row[name + "_margin"] = float(value)
            town = row["sample"]["town"]
            towns[town][0] += int(value > 0)
            towns[town][1] += 1
        results[name]["privileged_gt_roi"] = dict(
            accepted_at_half=int((margins > 0).sum()), total=72, by_town=dict(towns)
        )
    prefix = ("rpn.", "roi_heads.box_predictor.")
    if not all(
        torch.equal(t, saved["candidate"]["model"][k])
        for k, t in saved["parent"]["model"].items()
        if not k.startswith(prefix)
    ):
        raise ValueError("Shared frozen-feature assumption failed")
    result = dict(
        results=results,
        wall_seconds=time.monotonic() - started,
        scope="Fixed descriptive loss/GT-ROI audit; no fitting or selection change",
    )
    atomic_json(args.out / "gt_roi_rows.json", dict(rows=rows))
    atomic_json(args.out / "assessment.json", result)
    atomic_json(
        args.out / "progress.json",
        dict(
            status="complete",
            images_done=done,
            images_total=total,
            phase="Loss/GT-ROI audit complete",
            updated_at=time.time(),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "candidate", "corpus", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--prior-audit", type=Path, help="Reuse GT ROI scores and count loss sampling"
    )
    args = parser.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        fresh = not args.out.exists()
        try:
            assess(args)
        except BaseException as exc:
            if fresh and (args.out / "request.json").exists():
                atomic_json(
                    args.out / "progress.json",
                    dict(
                        status="stopped" if isinstance(exc, InterruptedError) else "failed",
                        error=f"{type(exc).__name__}: {exc}",
                        updated_at=time.time(),
                    ),
                )
            raise
