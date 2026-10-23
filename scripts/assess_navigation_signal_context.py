#!/usr/bin/env python3
"""Frozen context learning diagnosis; no training, selection or threshold change."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_signal_context import infer, score  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_context import SignalContextVerifier, label_proposals  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def classification(probabilities, labels):
    valid = labels >= 0
    p, y = probabilities[valid], labels[valid]
    prediction = p.argmax(1)
    matrix = np.zeros((4, 4), np.int64)
    np.add.at(matrix, (y, prediction), 1)
    recall = np.diag(matrix) / np.maximum(1, matrix.sum(1))
    return dict(
        confusion=matrix.tolist(),
        recall=recall.tolist(),
        balanced_accuracy=float(recall.mean()),
        supported=int(valid.sum()),
        unknown=int((~valid).sum()),
        foreground_binary_recall=float((p[y > 0, 0] < 0.5).mean()),
        foreground_maxclass_recall=float((p[y > 0, 1:].max(1) >= 0.5).mean()),
        foreground_phase_accuracy=float((p[y > 0, 1:].argmax(1) + 1 == y[y > 0]).mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = ROOT / "outputs/navigation_signal_context_20261021"
    started = time.monotonic()
    torch.set_num_threads(2)
    shutil.copyfile(__file__, out / "assessment_source.py")
    cache = json.loads((root / "cache.json").read_text())
    if digest(root / "train_patches.npz") != cache["train_patch_sha256"]:
        raise ValueError("TRAIN patch cache changed")
    atomic_json(
        out / "request.json",
        dict(
            source_sha256=digest(__file__),
            study_assessment_sha256=digest(root / "assessment.json"),
            scope="posthoc diagnosis only; no reselection",
            maximum_seconds=600,
            maximum_bytes=50_000_000,
        ),
    )
    status, error, done = "running", None, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 600:
            raise InterruptedError("STOP or600second limit")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 50_000_000:
            raise InterruptedError("50MB diagnostic limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Frozen context learning diagnosis",
                status=status,
                error=error,
                frames=done,
                target_frames=5,
                updated_at=time.time(),
            ),
        )

    results = []
    try:
        with BudgetWatch(out):
            progress()
            with np.load(root / "train_patches.npz") as arrays:
                train_x, train_y = arrays["patches"], arrays["labels"]
            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            original = json.loads(
                (
                    ROOT
                    / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
                ).read_text()
            )
            peaks = {row["sample"]: row["peaks"] for row in original}
            labels = np.concatenate(
                [
                    label_proposals(peaks[i], data.boxes[i], data.background[i])
                    for i in data.validation
                ]
            )
            offsets = cache["validation_offsets"]
            for step in (0, 100, 250, 500, 1000):
                model = SignalContextVerifier().cuda()
                model.load_state_dict(
                    torch.load(root / f"step{step}.pt", map_location="cuda", weights_only=False)[
                        "model"
                    ]
                )
                train_p = infer(model, train_x, check)
                np.savez_compressed(out / f"train_probabilities{step}.npz", probabilities=train_p)
                with np.load(root / f"probabilities{step}.npz") as arrays:
                    val_p = arrays["probabilities"]
                # Diagnostic separates background rejection from phase uncertainty;
                # never replace the registered four-class acceptance/selection.
                binary = {}
                for n, index in enumerate(data.validation):
                    p = val_p[offsets[n] : offsets[n + 1]]
                    binary[index] = [
                        dict(peak)
                        for peak, probability in zip(peaks[index], p)
                        if probability[0] < 0.5
                    ]
                results.append(
                    dict(
                        step=step,
                        train=classification(train_p, train_y),
                        validation=classification(val_p, labels),
                        diagnostic_binary_gate_original_state=score(binary, data),
                    )
                )
                done += 1
                progress()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    rows=results,
                    posthoc_no_reselection=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
