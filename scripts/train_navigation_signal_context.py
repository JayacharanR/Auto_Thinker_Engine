#!/usr/bin/env python3
"""Freeze TRAIN-only RGB context mining, fit one verifier, preserve all candidates."""

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_student import atomic_checkpoint, rng_state  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_context import (  # noqa:E402
    SignalContextVerifier,
    context_patches,
    label_proposals,
    verified_candidates,
)
from src.navigation.signal_hard_pixels import protected_signal_metrics  # noqa:E402
from src.navigation.signal_pixel_study import (  # noqa:E402
    NativeSignalDataset,
    NativeSignalGrid,
    point_candidates,
)
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def score(peaks_by_image, data):
    result = dict(
        labels=0, localized=0, joint=0, background_peaks=0, unknown_peaks=0, towns={}, classes={}
    )
    for index in data.validation:
        available, matched = set(range(len(data.boxes[index]))), {}
        for peak in peaks_by_image[index]:
            eligible = [
                i
                for i in sorted(available)
                if data.boxes[index][i][0] <= peak["x"] < data.boxes[index][i][2]
                and data.boxes[index][i][1] <= peak["y"] < data.boxes[index][i][3]
            ]
            if eligible:
                match = eligible[0]
                available.remove(match)
                matched[match] = peak
            elif data.background[index, peak["y"], peak["x"]]:
                result["background_peaks"] += 1
            else:
                result["unknown_peaks"] += 1
        for i, box in enumerate(data.boxes[index]):
            localized = i in matched
            joint = localized and matched[i]["cls"] == box[4]
            for row in (
                result,
                result["towns"].setdefault(
                    data.samples[index]["town"], dict(labels=0, localized=0, joint=0)
                ),
                result["classes"].setdefault(str(box[4]), dict(labels=0, localized=0, joint=0)),
            ):
                row["labels"] += 1
                row["localized"] += int(localized)
                row["joint"] += int(joint)
    if result["labels"] != 72:
        raise ValueError("Original72label denominator required")
    return result


@torch.inference_mode()
def infer(model, patches, check):
    result = []
    model.eval()
    for offset in range(0, len(patches), 256):
        check()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(torch.from_numpy(patches[offset : offset + 256]).cuda())
        result.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, 4), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    parent_root = ROOT / "outputs/navigation_signal_pixel_study_20261021"
    parent = read(parent_root / "assessment.json")["arms"]["bulb"]
    checkpoint = Path(parent["selected_checkpoint"])
    expected = "27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193"
    if digest(checkpoint) != expected:
        raise ValueError("Exact frozen proposal parent required")
    # Frozen predictions were scored by this archived generator/checkpoint.
    original_request = read(parent_root / "request.json")
    for file, sha in original_request["source_hashes"].items():
        if digest(parent_root / "source" / file) != sha:
            raise ValueError("Original frozen-proposal provenance changed")
    files = [
        Path(__file__),
        ROOT / "src/navigation/signal_context.py",
        ROOT / "src/navigation/signal_pixel_study.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "src/navigation/signal_hard_pixels.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "reports/navigation_signal_context_protocol_20261021.md",
        parent_root / "assessment.json",
        parent_root / "bulb/predictions1000.json",
        ROOT / "outputs/navigation_signal_pixel_audit_20261021/assessment.json",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    for file in files:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    atomic_json(
        out / "request.json",
        dict(
            source_hashes=hashes,
            parent_sha256=expected,
            cache_cap_seconds=600,
            fit_cap_seconds=1200,
            stage_cap_seconds=2400,
            artifact_cap_bytes=250_000_000,
            seed=42,
            updates=1000,
            no_native=True,
        ),
    )
    step, status, error, phase, images_done = 0, "running", None, "verifying original RGB", 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 2400:
            raise InterruptedError("Owned STOP or2400second stage limit")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 250_000_000:
            raise InterruptedError("250MB artifact limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="RGB signal context verifier",
                status=status,
                error=error,
                phase=phase,
                frames=step,
                target_frames=1000,
                images_done=images_done,
                images_total=1020,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            progress()
            cache_started = time.monotonic()
            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            frozen_predictions = read(parent_root / "bulb/predictions1000.json")
            validation_peaks = {row["sample"]: row["peaks"] for row in frozen_predictions}
            if set(validation_peaks) != set(data.validation):
                raise ValueError("Frozen validation sample identities changed")
            baseline = score(validation_peaks, data)
            for key in ("localized", "joint", "background_peaks", "towns", "classes"):
                if baseline[key] != parent["selected_metrics"][key]:
                    raise ValueError("Independent original proposal recount mismatch")
            atomic_json(out / "baseline.json", baseline)
            atomic_json(
                out / "dataset.json",
                dict(samples=data.samples, train=data.train, validation=data.validation),
            )
            proposal = NativeSignalGrid().cuda().eval()
            proposal.load_state_dict(
                torch.load(checkpoint, map_location="cuda", weights_only=False)["model"]
            )
            train_peaks, patch_rows, patches = {}, [], []
            counts, rng = Counter(), np.random.default_rng(42)
            phase = "TRAIN-only proposal mining and RGB context cache"
            for start in range(0, len(data.train), 3):
                check()
                if time.monotonic() - cache_started > 600:
                    raise InterruptedError("600second cache limit")
                indices = data.train[start : start + 3]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    probabilities = (
                        proposal(torch.from_numpy(data.images[indices]).cuda())
                        .float()
                        .softmax(1)
                        .cpu()
                        .numpy()
                    )
                for index, prob in zip(indices, probabilities):
                    peaks = point_candidates(prob)
                    train_peaks[index] = peaks
                    labels = label_proposals(peaks, data.boxes[index], data.background[index])
                    counts.update(
                        total_proposals=len(peaks),
                        unknown_proposals=int((labels == -1).sum()),
                        positive_proposals=int((labels > 0).sum()),
                        background_proposals=int((labels == 0).sum()),
                    )
                    selected = {
                        (p["x"], p["y"], int(label)) for p, label in zip(peaks, labels) if label > 0
                    }
                    negatives = np.flatnonzero(labels == 0)
                    chosen = list(negatives[:4])
                    if len(negatives) > 4:
                        chosen.extend(
                            rng.choice(
                                negatives[4:], min(12, len(negatives) - 4), replace=False
                            ).tolist()
                        )
                    for pick in chosen:
                        p = peaks[pick]
                        selected.add((p["x"], p["y"], 0))
                    for cls, anchors in data.anchors["bulb"].items():
                        for sample, positions in anchors:
                            if sample == index:
                                for pick in np.linspace(
                                    0, len(positions) - 1, min(4, len(positions)), dtype=int
                                ):
                                    x, y = positions[pick]
                                    selected.add((int(x), int(y), cls))
                    # Reject ambiguous points even if independent positive sources disagree.
                    by_point = {}
                    for x, y, label in selected:
                        by_point.setdefault((x, y), set()).add(label)
                    selected = sorted(
                        (x, y, next(iter(labels)))
                        for (x, y), labels in by_point.items()
                        if len(labels) == 1
                    )
                    if selected:
                        points = np.array([[x, y] for x, y, _ in selected])
                        patches.append(
                            context_patches(data.images[index].transpose(1, 2, 0), points)
                        )
                        patch_rows.extend(
                            dict(sample=index, x=x, y=y, label=label, role="train")
                            for x, y, label in selected
                        )
                    images_done += 1
                progress()
            train_patches = np.concatenate(patches)
            del patches, proposal
            labels = np.array([row["label"] for row in patch_rows], np.int64)
            np.savez_compressed(out / "train_patches.npz", patches=train_patches, labels=labels)
            atomic_json(out / "train_patch_rows.json", patch_rows)
            atomic_json(out / "train_proposals.json", train_peaks)
            validation_patches, offsets = [], [0]
            for index in data.validation:
                check()
                points = np.array([[p["x"], p["y"]] for p in validation_peaks[index]], np.int64)
                validation_patches.append(
                    context_patches(data.images[index].transpose(1, 2, 0), points)
                )
                offsets.append(offsets[-1] + len(points))
                images_done += 1
            validation_patches = np.concatenate(validation_patches)
            check()
            if time.monotonic() - cache_started > 600:
                raise InterruptedError("600second cache limit")
            atomic_json(
                out / "cache.json",
                dict(
                    train_images=876,
                    validation_images=144,
                    train_patches=len(labels),
                    classes={str(c): int((labels == c).sum()) for c in range(4)},
                    mined_counts=dict(counts),
                    train_patch_sha256=digest(out / "train_patches.npz"),
                    train_rows_sha256=digest(out / "train_patch_rows.json"),
                    validation_candidates=len(validation_patches),
                    validation_offsets=offsets,
                    validation_patch_array_sha256=hashlib.sha256(
                        validation_patches.tobytes()
                    ).hexdigest(),
                    validation_patches_persistent=False,
                    validation_targets_used_in_mining=False,
                    elapsed_seconds=time.monotonic() - cache_started,
                ),
            )
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            rng = np.random.default_rng(42)
            model = SignalContextVerifier().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
            pools = [np.flatnonzero(labels == c) for c in range(4)]
            if not all(len(p) for p in pools):
                raise ValueError("All TRAIN classes required")
            history, selected, rank = [], None, None
            fit_started = time.monotonic()

            def fit_check():
                check()
                if time.monotonic() - fit_started > 1200:
                    raise InterruptedError("1200second fit limit")

            for step in range(1001):
                fit_check()
                if step:
                    phase = "RGB context training"
                    model.train()
                    picks = np.concatenate([rng.choice(pool, 16) for pool in pools])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(torch.from_numpy(train_patches[picks]).cuda())
                    loss = F.cross_entropy(logits.float(), torch.from_numpy(labels[picks]).cuda())
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite verifier loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if step in (0, 100, 250, 500, 1000):
                    phase = "all-proposal context validation"
                    progress()
                    probabilities = infer(model, validation_patches, fit_check)
                    # Commit every probability before consulting component targets.
                    np.savez_compressed(
                        out / f"probabilities{step}.npz", probabilities=probabilities
                    )
                    verified = {
                        index: verified_candidates(
                            validation_peaks[index], probabilities[offsets[i] : offsets[i + 1]]
                        )
                        for i, index in enumerate(data.validation)
                    }
                    metric = score(verified, data)
                    protected = protected_signal_metrics(metric, baseline)
                    candidate_rank = (metric["background_peaks"], -metric["joint"])
                    if protected and (rank is None or candidate_rank < rank):
                        selected, rank = step, candidate_rank
                    history.append(dict(step=step, metrics=metric, coverage_protected=protected))
                    atomic_json(out / "history.json", history)
                    atomic_checkpoint(
                        out / f"step{step}.pt",
                        dict(
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            rng=rng_state(rng),
                            scaler=None,
                            precision="BF16",
                            step=step,
                            metrics=metric,
                            request_sha256=digest(out / "request.json"),
                            architecture="signal-context-verifier-v1",
                            driving_qualified=False,
                        ),
                    )
                if step % 25 == 0:
                    progress()
            metric = next((r["metrics"] for r in history if r["step"] == selected), None)
            passed = (
                metric is not None
                and selected > 0
                and metric["background_peaks"] <= 0.1 * baseline["background_peaks"]
            )
            for file, sha in hashes.items():
                if digest(ROOT / file) != sha:
                    raise ValueError("Registered source changed during study")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    selected_step=selected,
                    selected_metrics=metric,
                    selected_checkpoint=str(out / f"step{selected}.pt")
                    if selected is not None
                    else None,
                    checkpoint_sha256=digest(out / f"step{selected}.pt")
                    if selected is not None
                    else None,
                    countermeasure_pass=passed,
                    driving_qualified=False,
                    baseline=baseline,
                    parameters=sum(p.numel() for p in model.parameters()),
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "context study settled; no automatic driving"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
