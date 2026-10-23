#!/usr/bin/env python3
"""Same native bulb recipe on independently verified additional TRAIN data."""

import argparse
import json
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

from scripts.train_navigation_signal_pixels import evaluate  # noqa:E402
from scripts.train_navigation_student import atomic_checkpoint, rng_state  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa:E402
from src.navigation.signal_hard_pixels import protected_signal_metrics  # noqa:E402
from src.navigation.signal_model import signal_loss  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalGrid  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    supplement = ROOT / "outputs/navigation_signal_diversity_audit_20261021"
    if (
        digest(supplement / "manifest.json")
        != "f400408ae0ce7f53bbf16326592742b4df0ccdc4e79c3f16cc864cb3a7a8bdac"
    ):
        raise ValueError("Exact164case supplement required")
    parent_file = ROOT / "outputs/navigation_signal_pixel_study_20261021/assessment.json"
    parent = json.loads(parent_file.read_text())["arms"]["bulb"]
    files = [
        Path(__file__),
        ROOT / "src/navigation/signal_diversity_data.py",
        ROOT / "src/navigation/signal_pixel_study.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "src/navigation/signal_model.py",
        ROOT / "src/navigation/signal_hard_pixels.py",
        ROOT / "scripts/train_navigation_signal_pixels.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "reports/navigation_signal_diverse_fit_protocol_20261021.md",
        supplement / "manifest.json",
        supplement / "assessment.json",
        supplement / "labels.json",
        parent_file,
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
            updates=1000,
            seed=42,
            parent_reference=parent,
            no_native=True,
            fit_cap_seconds=1200,
            stage_cap_seconds=1500,
            artifact_cap_bytes=150_000_000,
        ),
    )
    step, status, error, phase = 0, "running", None, "source/RGB audit and unchangedVAL preflight"

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1500:
            raise InterruptedError("OwnedSTOP or1500second stage cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 150_000_000:
            raise InterruptedError("150MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Diverse native bulb study",
                status=status,
                error=error,
                phase=phase,
                frames=step,
                target_frames=1000,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            progress()
            data = DiverseSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021", supplement
            )
            if len(data.train) != 2844 or len(data.validation) != 144:
                raise ValueError("Frozen2844TRAIN/144VALrequired")
            check()
            atomic_json(
                out / "dataset.json",
                dict(
                    samples=data.samples,
                    train=data.train,
                    validation=data.validation,
                    positive_housings={str(k): len(v) for k, v in data.anchors["bulb"].items()},
                    original_validation_rgb_background_byte_identical=True,
                    train_junctions=sorted(
                        {
                            (data.samples[i]["town"], data.samples[i]["junction_id"])
                            for i in data.train
                        }
                    ),
                ),
            )
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            rng = np.random.default_rng(42)
            model = NativeSignalGrid().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            selected, best, history = 0, math.inf, []
            fit_started = time.monotonic()

            def fit_check():
                check()
                if time.monotonic() - fit_started > 1200:
                    raise InterruptedError("1200second fit cap")

            for step in range(1001):
                fit_check()
                if step:
                    phase = "same-recipe bulb training on broaderTRAINdata"
                    model.train()
                    rgb, target = data.batch("bulb", rng, "cuda")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(rgb)
                    loss = signal_loss(logits, target)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if step in (0, 100, 250, 500, 1000):
                    phase = "unchanged144image/72label validation"
                    progress()
                    metric, predictions = evaluate(model, data, fit_check)
                    if metric["selection_loss"] < best:
                        selected, best = step, metric["selection_loss"]
                    history.append(dict(step=step, metrics=metric))
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
                            architecture="native-signal-grid-v1",
                            driving_qualified=False,
                        ),
                    )
                    atomic_json(out / f"predictions{step}.json", predictions)
                    atomic_json(out / "history.json", history)
                if step % 25 == 0:
                    progress()
            metric = next(r["metrics"] for r in history if r["step"] == selected)
            reference = parent["selected_metrics"]
            criteria = dict(
                trained_candidate=selected > 0,
                coverage_protected=protected_signal_metrics(metric, reference),
                background_peaks_reduced90=metric["background_peaks"]
                <= 0.1 * reference["background_peaks"],
                background_fraction_reduced90=metric["background_active_fraction"]
                <= 0.1 * reference["background_active_fraction"],
            )
            for file, sha in hashes.items():
                if digest(ROOT / file) != sha:
                    raise ValueError("Registered source changed during fit")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    selected_step=selected,
                    selected_metrics=metric,
                    criteria=criteria,
                    countermeasure_pass=all(criteria.values()),
                    driving_qualified=False,
                    selected_checkpoint=str(out / f"step{selected}.pt"),
                    checkpoint_sha256=digest(out / f"step{selected}.pt"),
                    elapsed_seconds=time.monotonic() - started,
                    fit_seconds=time.monotonic() - fit_started,
                ),
            )
            status, phase = "complete", "data-diversity study settled; no automatic driving"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
