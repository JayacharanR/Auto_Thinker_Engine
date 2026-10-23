#!/usr/bin/env python3
"""Bounded cached TRAIN ROI residual fit; unchanged RGB-only detector evaluation."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import resize_boxes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from scripts.train_navigation_signal_detector import (  # noqa: E402
    BudgetStop,
    evaluate,
    frozen_digest,
)
from scripts.train_navigation_student import atomic_checkpoint, rng_state  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import (  # noqa: E402
    RunLock,
    atomic_json,
    check_resources,
    resource_shortfall,
)
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import install_roi_adapter, make_signal_detector  # noqa: E402
from src.navigation.signal_roi_learning import (  # noqa: E402
    operational_key,
    supervised_rois,
    validation_payload_digest,
)

INIT_SHA = "a44f65acd4e3d2941c7f36e6dcc7e0f5d6a3853e3be6a7d1fbd691ed0370f1f9"
ARCHITECTURE = "signal-detector-roi-residual-v1"


@torch.no_grad()
def extract_cache(model, data, stop, progress):
    rng = np.random.default_rng(20261016)
    features, logits, records = [], [], []
    model.eval()
    predictor = model.roi_heads.box_predictor
    for offset in range(0, len(data.base.train), 3):
        stop()
        indices = data.base.train[offset : offset + 3]
        rows = [supervised_rois(data, i, rng) for i in indices]
        images, targets = data.batch(indices, "cuda")
        transformed, _ = model.transform(images, targets)
        encoded = model.backbone(transformed.tensors)
        proposals, _ = model.rpn(transformed, encoded, None)
        boxes = []
        for index, selected, size, row in zip(indices, proposals, transformed.image_sizes, rows):
            if len(selected) < 16:
                raise ValueError("TRAIN image has fewer than16 distillation proposals")
            native = np.array([r["box"] for r in row], np.float32).reshape(-1, 4)
            fixed = resize_boxes(torch.as_tensor(native, device="cuda"), (288, 512), size)
            boxes.append(torch.cat([fixed, selected[:16]]))
            proposal_native = resize_boxes(selected[:16], size, (288, 512)).cpu().tolist()
            row.extend(
                dict(
                    sample_index=index,
                    label=-1,
                    source_id=n,
                    variant=0,
                    box=box,
                    domain=int("supplement_corpus" in data.base.samples[index]),
                )
                for n, box in enumerate(proposal_native)
            )
        pooled = model.roi_heads.box_roi_pool(encoded, boxes, transformed.image_sizes)
        x = model.roi_heads.box_head(pooled)
        old, old_boxes = predictor.base(x)
        same, same_boxes = predictor(x)
        if not torch.equal(old, same) or not torch.equal(old_boxes, same_boxes):
            raise ValueError("Zero adapter changed source logits or boxes")
        features.append(x.cpu())
        logits.append(old.cpu())
        records.extend(r for row in rows for r in row)
        progress(
            "Caching TRAIN ROI features",
            images_done=min(offset + 3, len(data.base.train)),
            images_total=len(data.base.train),
        )
    x, old = torch.cat(features), torch.cat(logits)
    labels = torch.tensor([r["label"] for r in records], dtype=torch.long)
    domains = torch.tensor([r["domain"] for r in records], dtype=torch.long)
    assert len(x) == len(records) == 30824
    assert int((labels == 1).sum()) == 642 * 4
    assert int((labels == 0).sum()) == 440 * 4
    assert int((labels == -1).sum()) == 1656 * 16
    return dict(features=x, base_logits=old, labels=labels, domains=domains), records


def train(args):
    if args.out.exists():
        raise FileExistsError("Fresh ROI experiment required; preserve completed/partial runs")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    rng = np.random.default_rng(20261016)
    reason = resource_shortfall(dict(minimum_free_vram_gb=4, minimum_available_ram_gb=4))
    if reason:
        raise RuntimeError(reason)
    init_path = args.init / "latest.pt"
    if digest(init_path) != INIT_SHA:
        raise ValueError("Only fixed completed joint200 initialization allowed")
    source = torch.load(init_path, map_location="cpu", weights_only=False)
    if source["step"] != 200 or source["validation_step"] != 200:
        raise ValueError("Completed joint200 required")
    data = SignalDetectionData(args.corpus)
    data.attach_hard_negatives(args.negatives, source["request"]["parent_sha256"], args.release)
    validation_sha = validation_payload_digest(data)
    supplement = data.append_training_corpus(args.supplement)
    if validation_payload_digest(data) != validation_sha:
        raise ValueError("Validation changed on append")
    if validation_sha != "9ab66584c129e8a84a676512bad8b693dd9f82d2df5de2f1a8c74a3edcb0bd2f":
        raise ValueError("Original validation payload changed")
    linear_path = args.classifier / "robust_rgb_linear.npz"
    if digest(linear_path) != source["request"]["crop_classifier_sha256"]:
        raise ValueError("Fixed crop classifier changed")
    with np.load(linear_path, allow_pickle=False) as values:
        linear = {k: values[k].copy() for k in ("coef", "intercept", "classes")}
    baseline = source["validation"]
    initial_predictions = args.init / "predictions_step200.json"
    prior_summary_path = ROOT / "reports/navigation_m2_signal_expansion_20261016.json"
    prior_summary = json.loads(prior_summary_path.read_text())
    prior_key = str(initial_predictions.resolve().relative_to(ROOT))
    if (
        prior_summary["model_sha256"]["candidate"] != INIT_SHA
        or digest(initial_predictions) != prior_summary["evidence_sha256"][prior_key]
    ):
        raise ValueError("Previously archived initialization prediction evidence changed")
    baseline_rows = json.loads(initial_predictions.read_text())["rows"]
    exact, _ = audit_predictions(baseline_rows, data.labels)
    if exact != baseline["one_to_one"]:
        raise ValueError("Initialization predictions do not reproduce saved one-to-one counts")
    assert (
        exact["iou3"],
        exact["iou3_state_correct"],
        baseline["counts"]["approximate_background_candidates"],
    ) == (36, 26, 209)
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        *sorted((ROOT / "src/navigation").glob("*.py")),
        ROOT / "scripts/train_navigation_signal_detector.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "scripts/assess_navigation_signal_detector.py",
        ROOT / "scripts/assess_navigation_signal_crops.py",
        ROOT / "scripts/assess_navigation_reference_detector.py",
        ROOT / "scripts/collect_navigation_signal_pairs.py",
        ROOT / "reports/navigation_m2_signal_roi_adapter_protocol_20261016.md",
    ]
    request = dict(
        architecture=ARCHITECTURE,
        kind="TRAIN perception candidate only; no driving",
        initialization_sha256=INIT_SHA,
        initialization_is_unselected_development_candidate=True,
        corpus_audit_sha256=digest(args.corpus / "data_audit.json"),
        supplement=supplement,
        release_sha256=digest(args.release),
        classifier_sha256=digest(linear_path),
        validation_payload_sha256=validation_sha,
        initial_predictions_sha256=digest(initial_predictions),
        initialization_summary_sha256=digest(prior_summary_path),
        seed=20261016,
        steps=1000,
        torch_version=torch.__version__,
        maximum_wall_seconds=900,
        trainable_parameters=65665,
        objective="Balanced ROI cross-entropy +1.0 KL(old||new) on unlabelled proposals",
        operational_selection="Strict recall/joint improvement, baseline global/town/tiny floors; "
        "rank joint,recall,-approximate_background; no loss reselection",
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    for p in sources:
        target = args.out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(args.out / "request.json", request)
    model = make_signal_detector()
    model.load_state_dict(source["model"], strict=True)
    install_roi_adapter(model)
    initial_state = model.state_dict()
    for name, tensor in source["model"].items():
        mapped = name.replace("roi_heads.box_predictor.", "roi_heads.box_predictor.base.")
        assert torch.equal(tensor, initial_state[mapped]), name
    model = model.cuda()
    frozen = frozen_digest(model)
    predictor = model.roi_heads.box_predictor
    optimizer = torch.optim.AdamW(predictor.adapter.parameters(), lr=0.001, weight_decay=0.0001)
    step, selected_step, validation_step = 0, 0, 0
    validation, selected_validation = baseline, baseline
    best_key = (26, 36, -209)
    improved_candidate = False
    cache_sha = None

    def elapsed():
        return time.monotonic() - started

    def stop():
        if (args.out / "STOP").exists() or elapsed() >= 900:
            raise BudgetStop("STOP or900s cumulative budget")

    def progress(phase, status="running", **extra):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M2 ROI adapter",
                steps=step,
                target_steps=1000,
                selected_step=selected_step,
                candidate_improved=improved_candidate,
                phase=phase,
                updated_at=time.time(),
                elapsed_seconds=elapsed(),
                validation=validation,
                **extra,
            ),
        )

    def save(best=False):
        state = dict(
            kind="perception_candidate_only",
            architecture=ARCHITECTURE,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(rng),
            request=request,
            step=step,
            validation_step=validation_step,
            validation=validation,
            selected_step=selected_step,
            candidate_improved=improved_candidate,
            best_key=best_key,
            frozen_sha256=frozen,
            cache_sha256=cache_sha,
            elapsed_seconds=elapsed(),
        )
        atomic_checkpoint(args.out / "latest.pt", state)
        if best:
            atomic_checkpoint(args.out / "best_candidate.pt", state)

    def validate():
        nonlocal validation, validation_step, selected_step, best_key
        nonlocal improved_candidate, selected_validation
        validation, rows = evaluate(
            model, data, linear, rng, stop, lambda phase, n: progress(phase, validation_images=n)
        )
        validation_step = step
        exact, _ = audit_predictions(rows, data.labels)
        validation["one_to_one"] = exact
        key = operational_key(validation, baseline)
        validation["prospective_operational_guard_passed"] = key is not None
        better = key is not None and key > best_key
        if better:
            best_key, selected_step = key, step
            improved_candidate, selected_validation = True, validation
        atomic_json(args.out / f"predictions_step{step}.json", dict(rows=rows))
        with (args.out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, **validation)) + "\n")
        save(best=better)
        print(
            json.dumps(dict(step=step, validation=validation, selected_step=selected_step)),
            flush=True,
        )

    status = "complete"
    try:
        progress("Preparing TRAIN-only cache")
        cache, records = extract_cache(model, data, stop, progress)
        cache_path = args.out / "train_roi_cache.pt"
        atomic_checkpoint(cache_path, dict(**cache, initialization_sha256=INIT_SHA))
        cache_sha = digest(cache_path)
        atomic_json(
            args.out / "cache_manifest.json",
            dict(
                samples=records,
                features_shape=list(cache["features"].shape),
                cache_sha256=cache_sha,
                train_images=1656,
                positive_source_labels=642,
                negative_source_phase_copies=440,
                contains_validation=False,
                zero_adapter_logit_and_box_parity=True,
            ),
        )
        # Frozen zero correction and identical base prove source inference parity.
        # Preserve its completed evaluation; do not rerun an identical baseline.
        atomic_json(args.out / "predictions_step0.json", dict(rows=baseline_rows))
        with (args.out / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=0, reused_exact_joint200=True, **baseline)) + "\n")
        save(best=True)
        x = cache["features"].cuda()
        old_logits = cache["base_logits"].cuda()
        y = cache["labels"].cuda()
        labels = cache["labels"].numpy()
        domain = cache["domains"].numpy()
        groups = [
            np.flatnonzero((labels == 1) & (domain == 0)),
            np.flatnonzero((labels == 1) & (domain == 1)),
            np.flatnonzero(labels == 0),
            np.flatnonzero(labels == -1),
        ]
        if any(not len(group) for group in groups):
            raise ValueError("Cache lacks fixed supervised/distillation strata")
        model.eval()  # No dropout/BN in the adapter; fixed base remains in eval.
        while step < 1000:
            stop()
            batch = np.concatenate(
                [rng.choice(groups[0], 16), rng.choice(groups[1], 16), rng.choice(groups[2], 32)]
            )
            distill = rng.choice(groups[3], 64)
            optimizer.zero_grad(set_to_none=True)
            scores, _ = predictor(x[batch])
            soft_scores, _ = predictor(x[distill])
            ce = F.cross_entropy(scores, y[batch])
            consistency = F.kl_div(
                F.log_softmax(soft_scores, dim=1),
                F.softmax(old_logits[distill], dim=1),
                reduction="batchmean",
            )
            loss = ce + consistency
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ROI loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor.adapter.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if step % 100 == 0:
                progress(
                    "Fitting cached TRAIN classifier",
                    loss=float(loss.detach()),
                    supervised_ce=float(ce.detach()),
                    consistency_kl=float(consistency.detach()),
                )
            if step in (250, 500, 1000):
                validate()
            elif step % 100 == 0:
                save()
        if frozen_digest(model) != frozen:
            raise ValueError("Frozen detector or normalization changed")
    except BudgetStop:
        status = "stopped"
    except BaseException as exc:
        save()
        progress(
            "Failed; inspect before any new run",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    unchanged = frozen_digest(model) == frozen
    if not unchanged:
        raise ValueError("Frozen detector changed")
    save()
    review = dict(
        status=status,
        selected_step=selected_step,
        candidate_improved=improved_candidate,
        selected_validation=selected_validation,
        latest_validation=validation,
        initialization_validation=baseline,
        frozen_parameters_and_buffers_unchanged=True,
        frozen_sha256=frozen,
        cache_sha256=cache_sha,
        best_candidate_sha256=digest(args.out / "best_candidate.pt")
        if (args.out / "best_candidate.pt").exists()
        else None,
        wall_seconds=elapsed(),
        trainable_parameters=65665,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        scope="Prospectively selected perception candidate only; original driver unchanged",
    )
    atomic_json(args.out / "review.json", review)
    progress("ROI experiment finished", status=status)
    print(json.dumps(review, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "supplement", "negatives", "release", "init", "classifier", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    with RunLock(ROOT / "outputs/navigation/.gpu.lock"):
        train(args)
