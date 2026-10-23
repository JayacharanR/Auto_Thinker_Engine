#!/usr/bin/env python3
"""One registered hard-background correction; preserve all parent failures."""

import argparse
import json
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
from src.navigation.signal_hard_pixels import hard_pixel_loss, protected_signal_metrics  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalDataset, NativeSignalGrid  # noqa:E402
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
    source_root = ROOT / "outputs/navigation_signal_pixel_study_20261021"
    parent = json.loads((source_root / "assessment.json").read_text())["arms"]["bulb"]
    checkpoint = Path(parent["selected_checkpoint"])
    expected = "27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193"
    if digest(checkpoint) != expected or parent["checkpoint_sha256"] != expected:
        raise ValueError("Only registered parent weights")
    files = [
        Path(__file__),
        ROOT / "src/navigation/signal_hard_pixels.py",
        ROOT / "src/navigation/signal_pixel_study.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "scripts/train_navigation_signal_pixels.py",
        ROOT / "scripts/train_navigation_student.py",
        ROOT / "reports/navigation_signal_hard_pixels_protocol_20261021.md",
        source_root / "assessment.json",
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
            parent_checkpoint=str(checkpoint),
            parent_sha256=expected,
            updates=1000,
            seed=42,
            stage_cap_seconds=1500,
            fit_cap_seconds=1200,
            artifact_cap_bytes=150_000_000,
            no_native=True,
        ),
    )
    step, status, error, phase = 0, "running", None, "verifying sources"

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1500:
            raise InterruptedError("Owned STOP or1500second stage limit")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 150_000_000:
            raise InterruptedError("150MB artifact limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Hard background signal study",
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
            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            rng = np.random.default_rng(42)
            model = NativeSignalGrid().cuda()
            model.load_state_dict(
                torch.load(checkpoint, map_location="cuda", weights_only=False)["model"],
                strict=True,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
            reference = parent["selected_metrics"]
            selected, rank, history = (
                0,
                (
                    reference["background_peaks"],
                    reference["background_active_fraction"],
                    -reference["joint"],
                ),
                [],
            )
            fit_started = time.monotonic()

            def fit_check():
                check()
                if time.monotonic() - fit_started > 1200:
                    raise InterruptedError("1200second fit limit")

            for step in range(1001):
                fit_check()
                if step:
                    phase = "hard-background training"
                    model.train()
                    rgb, targets = data.batch("bulb", rng, "cuda")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(rgb)
                    loss = hard_pixel_loss(logits, targets)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite hard-background loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if step in (0, 100, 250, 500, 1000):
                    phase = "full-image component validation"
                    progress()
                    metric, predictions = evaluate(model, data, fit_check)
                    if step == 0 and metric != reference:
                        raise ValueError("Exact parent inference parity required")
                    eligible = protected_signal_metrics(metric, reference)
                    candidate_rank = (
                        metric["background_peaks"],
                        metric["background_active_fraction"],
                        -metric["joint"],
                    )
                    if eligible and candidate_rank < rank:
                        rank, selected = candidate_rank, step
                    history.append(dict(step=step, metrics=metric, coverage_protected=eligible))
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
            metric = next(row["metrics"] for row in history if row["step"] == selected)
            criteria = dict(
                trained_candidate=selected > 0,
                coverage_protected=protected_signal_metrics(metric, reference),
                background_peak_reduction90=metric["background_peaks"]
                <= 0.1 * reference["background_peaks"],
                background_fraction_reduction90=metric["background_active_fraction"]
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
                    selected_checkpoint=str(out / f"step{selected}.pt"),
                    checkpoint_sha256=digest(out / f"step{selected}.pt"),
                    criteria=criteria,
                    countermeasure_pass=all(criteria.values()),
                    driving_qualified=False,
                    parent_checkpoint=str(checkpoint),
                    parent_sha256=expected,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "registered hard-background study settled"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
