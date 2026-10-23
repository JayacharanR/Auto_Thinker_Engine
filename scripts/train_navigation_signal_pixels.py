#!/usr/bin/env python3
"""Bounded paired native-pixel signal study; no driving-policy export."""

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_student import atomic_checkpoint, rng_state  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_model import signal_loss  # noqa:E402
from src.navigation.signal_pixel_study import (  # noqa:E402
    NativeSignalDataset,
    NativeSignalGrid,
    point_candidates,
)
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


@torch.inference_mode()
def evaluate(model, data, check):
    model.eval()
    foreground, background_sum, background_n, active_n = [], 0.0, 0, 0
    totals = dict(labels=0, localized=0, joint=0, center_correct=0, background_peaks=0)
    towns, classes, predictions = {}, {}, []
    for start in range(0, len(data.validation), 3):
        check()
        indices = data.validation[start : start + 3]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(torch.from_numpy(data.images[indices]).cuda()).float().softmax(1)
        probabilities = p.cpu().numpy()
        for index, prob in zip(indices, probabilities):
            bg = data.background[index]
            background_sum += float(-np.log(np.maximum(prob[0][bg], 1e-9)).sum())
            background_n += int(bg.sum())
            active_n += int(((prob[1:].max(0) >= 0.5) & bg).sum())
            peaks = point_candidates(prob)
            boxes = data.boxes[index]
            available = set(range(len(boxes)))
            matches = {}
            for peak in peaks:
                candidates = [
                    i
                    for i in sorted(available)
                    if boxes[i][0] <= peak["x"] < boxes[i][2]
                    and boxes[i][1] <= peak["y"] < boxes[i][3]
                ]
                if candidates:
                    match = candidates[0]
                    available.remove(match)
                    matches[match] = peak
                elif bg[peak["y"], peak["x"]]:
                    totals["background_peaks"] += 1
            for i, (x0, y0, x1, y1, cls) in enumerate(boxes):
                region = prob[:, y0:y1, x0:x1]
                y, x = np.unravel_index(region[1:].max(0).argmax(), region.shape[1:])
                foreground.append(-math.log(max(float(region[cls, y, x]), 1e-9)))
                correct = i in matches and matches[i]["cls"] == cls
                totals["labels"] += 1
                totals["localized"] += int(i in matches)
                totals["joint"] += int(correct)
                totals["center_correct"] += int(
                    prob[:, (y0 + y1) // 2, (x0 + x1) // 2].argmax() == cls
                )
                for mapping, key in ((towns, data.samples[index]["town"]), (classes, str(cls))):
                    row = mapping.setdefault(key, dict(labels=0, localized=0, joint=0))
                    row["labels"] += 1
                    row["localized"] += int(i in matches)
                    row["joint"] += int(correct)
            predictions.append(dict(sample=index, peaks=peaks, matched=matches))
    if totals["labels"] != 72:
        raise ValueError("Original72known labels required")
    return dict(
        **totals,
        foreground_loss=float(np.mean(foreground)),
        background_loss=background_sum / background_n,
        selection_loss=float(np.mean(foreground)) + 0.2 * background_sum / background_n,
        background_active_fraction=active_n / background_n,
        towns=towns,
        classes=classes,
    ), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    files = [
        Path(__file__),
        ROOT / "src/navigation/signal_pixel_study.py",
        ROOT / "src/navigation/signal_model.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "reports/navigation_signal_pixel_study_protocol_20261021.md",
    ]
    audit = ROOT / "outputs/navigation_signal_pixel_audit_20261021"
    files.extend(
        audit / name
        for name in ("assessment.json", "cases.json", "labels.json", "validation_identity.json")
    )
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    for file in files:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    atomic_json(
        out / "request.json",
        dict(
            source_hashes=hashes,
            seed=42,
            updates_each=1000,
            fit_cap_seconds=1200,
            stage_cap_seconds=2800,
            artifact_cap_bytes=250_000_000,
            no_native=True,
        ),
    )
    arm, step, completed, status, error, phase = None, 0, 0, "running", None, "verifying sources"

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 2800:
            raise InterruptedError("OwnedSTOP or2800second stage limit")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 250_000_000:
            raise InterruptedError("250MB artifact limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Native bulb-label paired study",
                status=status,
                phase=phase,
                error=error,
                arm=arm,
                steps=step,
                target_steps=1000,
                fits_done=completed,
                fits_total=2,
                frames=completed * 1000 + step if completed < 2 else 2000,
                target_frames=2000,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    results = {}
    try:
        with BudgetWatch(out):
            progress()
            data = NativeSignalDataset(ROOT, audit)
            check()
            atomic_json(
                out / "dataset.json",
                dict(
                    samples=data.samples,
                    train=data.train,
                    validation=data.validation,
                    positive_housings={str(k): len(v) for k, v in data.anchors["bulb"].items()},
                    labels_from_paired_training_images_only=True,
                ),
            )
            for arm in ("center", "bulb"):
                step = 0
                directory = out / arm
                directory.mkdir()
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                rng = np.random.default_rng(42)
                model = NativeSignalGrid().cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                best, selected, history = math.inf, 0, []
                fit_started = time.monotonic()

                def fit_check():
                    check()
                    if time.monotonic() - fit_started > 1200:
                        raise InterruptedError("1200second per-fit limit")

                for step in range(1001):
                    fit_check()
                    if step:
                        phase = "training"
                        model.train()
                        rgb, targets = data.batch(arm, rng, "cuda")
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(rgb)
                        loss = signal_loss(logits, targets)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite signal loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    if step in (0, 100, 250, 500, 1000):
                        phase = "full-image component validation"
                        progress()
                        metric, predictions = evaluate(model, data, fit_check)
                        if metric["selection_loss"] < best:
                            best, selected = metric["selection_loss"], step
                        history.append(dict(step=step, metrics=metric))
                        atomic_checkpoint(
                            directory / f"step{step}.pt",
                            dict(
                                model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                rng=rng_state(rng),
                                scaler=None,
                                precision="BF16",
                                step=step,
                                request_sha256=digest(out / "request.json"),
                                architecture="native-signal-grid-v1",
                                metrics=metric,
                                driving_qualified=False,
                            ),
                        )
                        atomic_json(directory / f"predictions{step}.json", predictions)
                        atomic_json(directory / "history.json", history)
                    if step % 25 == 0:
                        progress()
                # Same precision, batch ofthree views; no simulator orIPC included.
                model.eval()
                rgb = torch.from_numpy(data.images[data.validation[:3]]).cuda()
                times = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for repeat in range(25):
                        torch.cuda.synchronize()
                        clock = time.perf_counter()
                        model(rgb)
                        torch.cuda.synchronize()
                        if repeat >= 5:
                            times.append((time.perf_counter() - clock) * 1000)
                results[arm] = dict(
                    selected_step=selected,
                    selected_checkpoint=str(directory / f"step{selected}.pt"),
                    checkpoint_sha256=digest(directory / f"step{selected}.pt"),
                    selected_metrics=next(r["metrics"] for r in history if r["step"] == selected),
                    parameters=sum(p.numel() for p in model.parameters()),
                    elapsed_seconds=time.monotonic() - fit_started,
                    three_view_forward_ms=dict(
                        mean=float(np.mean(times)), p95=float(np.quantile(times, 0.95))
                    ),
                )
                completed += 1
                step = 0
                atomic_json(out / "results.json", results)
                progress()
            control, bulb = [results[a]["selected_metrics"] for a in ("center", "bulb")]
            criteria = dict(
                joint_improved=bulb["joint"] > control["joint"],
                every_town_protected=all(
                    bulb["towns"][t]["joint"] >= m["joint"] for t, m in control["towns"].items()
                ),
                background_fraction_protected=bulb["background_active_fraction"]
                <= control["background_active_fraction"],
                background_peaks_protected=bulb["background_peaks"] <= control["background_peaks"],
            )
            for file, sha in hashes.items():
                if digest(ROOT / file) != sha:
                    raise ValueError("Source changed during registered study")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    arms=results,
                    criteria=criteria,
                    label_countermeasure_pass=all(criteria.values()),
                    driving_qualified=False,
                    results_sha256=digest(out / "results.json"),
                    scope="Inspected component development; not boxAP or driving",
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "paired study settled; no automatic driving"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
