#!/usr/bin/env python3
"""TRAIN-only coverage of exact audited bulb pixels by frozen candidate supervision."""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    out = ROOT / "outputs/navigation_signal_positive_coverage_20261022"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, error, done = "running", None, 0
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="TRAIN positive proposal coverage",
                status=status,
                frames=done,
                target_frames=2844,
                error=error,
                phase="Exact audited active-pixel coverage; no model inference",
                elapsed_seconds=time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
            inventory = ROOT / "outputs/navigation_signal_plateau_inventory_20261022"
            rows = read(cache / "train_rows.json")
            keep = np.load(
                ROOT / "outputs/navigation_signal_background_fit_20261022/training_keep.npy"
            )
            for name, sha in read(inventory / "assessment.json")["artifact_sha256"].items():
                if digest(inventory / name) != sha:
                    raise ValueError("Frozen candidate inventory changed")
            files = [
                Path(__file__),
                cache / "train_rows.json",
                cache / "dataset.json",
                inventory / "assessment.json",
                ROOT / "outputs/navigation_signal_background_fit_20261022/training_keep.npy",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    no_model_inference=True,
                    no_fit=True,
                    no_component_val=True,
                    stage_seconds=180,
                    artifact_cap_bytes=10_000_000,
                ),
            )
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            original = defaultdict(set)
            all_original = defaultdict(set)
            for j, r in enumerate(rows):
                if r["role"] != "train" or r["sample"] not in data.train:
                    raise ValueError("Original TRAIN rows required")
                all_original[r["sample"]].add((r["x"], r["y"]))
                if r["label"] and keep[j]:
                    original[r["sample"]].add((r["x"], r["y"]))
            points = np.load(inventory / "points.npy", mmap_mode="r")
            with np.load(inventory / "offsets.npz") as a:
                offsets, samples = a["offsets"], a["samples"]
            if not np.array_equal(samples, data.train):
                raise ValueError("Original TRAIN ordering changed")
            totals = Counter()
            phases = defaultdict(Counter)
            extra, details = [], []
            for n, i in enumerate(data.train):
                if (out / "STOP").exists() or time.monotonic() - started > 180:
                    raise InterruptedError("Owned STOP or 180 second cap")
                xy = points[offsets[n] : offsets[n + 1]]
                target = data.targets["bulb"][i]
                cls = {"red": 1, "red_repeat": 1, "yellow": 2, "green": 3}[data.samples[i]["phase"]]
                eligible = xy[target[xy[:, 1], xy[:, 0]] == cls]
                c = Counter(
                    active_pixels=int((target == cls).sum()), active_candidates=len(eligible)
                )
                for x, y in eligible:
                    key = (int(x), int(y))
                    c["already_supervised"] += key in original[i]
                    if key not in all_original[i]:
                        if data.background[i, y, x]:
                            raise ValueError("Active pixel also marked known background")
                        extra.append(
                            dict(sample=int(i), x=int(x), y=int(y), label=cls, role="train")
                        )
                        c["new_exact_positive_candidates"] += 1
                totals.update(c)
                phases[str(cls)].update(c)
                details.append(dict(sample=int(i), **c))
                done += 1
                if done % 100 == 0:
                    progress()
            atomic_json(out / "new_candidate_rows.json", extra)
            atomic_json(out / "images.json", details)
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Source changed")
            if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 10_000_000:
                raise InterruptedError("10 MB cap")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    totals=dict(totals),
                    phases={k: dict(v) for k, v in phases.items()},
                    new_rows_sha256=digest(out / "new_candidate_rows.json"),
                    no_training_release=True,
                    no_model_inference=True,
                    no_component_val=True,
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
