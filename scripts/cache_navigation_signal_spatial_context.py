#!/usr/bin/env python3
"""Cache registered spatial RGB features on the fixed TRAIN rows only."""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_rgb_augmentation import GAMMAS, gamma_rgb  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def direct_features(rgb, xy):
    """Independent direct-pixel moments; no integral image implementation."""
    padded = np.pad(rgb, ((48, 48), (48, 48), (0, 0)), mode="edge")
    result = []
    for x, y in xy:
        values = []
        for scale in (16, 32, 96):
            cell = scale // 4
            for row in range(4):
                for col in range(4):
                    left = int(x) + 48 - scale // 2 + col * cell
                    top = int(y) + 48 - scale // 2 + row * cell
                    patch = padded[top : top + cell, left : left + cell].astype(np.float64)
                    values.extend((patch.mean((0, 1)) / 255).tolist())
                    values.extend((patch.std((0, 1)) / 255).tolist())
        result.append(values)
    return np.asarray(result, np.float32).reshape(-1, 288)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    original = ROOT / "outputs/navigation_signal_pretrained_20261021"
    extra = ROOT / "outputs/navigation_signal_background_cache_20261022"
    fit = ROOT / "outputs/navigation_signal_background_fit_20261022"
    fold_path = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
    module_file = ROOT / "patches/navigation_signal_spatial_context_candidate.py"
    started = time.monotonic()
    status, error, done, phase = "running", None, 0, "verify original TRAIN sources"

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1200:
            raise InterruptedError("Owned STOP or 1200 second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 500_000_000:
            raise InterruptedError("500 MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Spatial signal TRAIN cache",
                status=status,
                frames=done,
                target_frames=2844,
                phase=phase,
                error=error,
                elapsed_seconds=time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            files = [
                Path(__file__),
                module_file,
                ROOT / "reports/navigation_signal_spatial_context_protocol_20261022.md",
                ROOT / "tests/test_navigation_staged_signal_spatial_context.py",
                original / "train_rows.json",
                original / "dataset.json",
                original / "cache.json",
                extra / "rows.json",
                extra / "assessment.json",
                fit / "cache_verification.json",
                fold_path,
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for p in files:
                target = out / "source" / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
            for name, sha in read(extra / "assessment.json")["artifacts"].items():
                if digest(extra / name) != sha:
                    raise ValueError("Extra cache changed")
            old_request = read(fit / "request.json")
            if digest(fold_path) != old_request["folds_sha256"]:
                raise ValueError("Original whole-junction folds changed")
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    no_component_val=True,
                    features=288,
                    stage_seconds=1200,
                    artifact_cap_bytes=500_000_000,
                ),
            )
            old_rows, extra_rows = read(original / "train_rows.json"), read(extra / "rows.json")
            rows = old_rows + extra_rows
            if len(old_rows) != 51094 or len(extra_rows) != 182016:
                raise ValueError("Fixed original/extra row counts required")
            with np.load(fold_path) as a:
                old_fold = a["fold"]
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            identity = read(original / "dataset.json")
            if (
                json.dumps(identity["samples"], sort_keys=True)
                != json.dumps(data.samples, sort_keys=True)
                or identity["train"] != data.train
            ):
                raise ValueError("Original ordered dataset changed")
            train = set(data.train)
            grouped, location_folds = defaultdict(list), defaultdict(set)
            folds = np.concatenate([old_fold, np.array([r["fold"] for r in extra_rows])])
            seen = set()
            for j, row in enumerate(rows):
                i, x, y = row["sample"], row["x"], row["y"]
                key = (i, x, y)
                if i not in train or row["role"] != "train":
                    raise ValueError("TRAIN-only fixed rows required")
                if j >= len(old_rows) and (
                    key in seen or row["label"] != 0 or not data.background[i, y, x]
                ):
                    raise ValueError("Added negative eligibility/overlap invalid")
                seen.add(key)
                grouped[i].append(j)
                s = data.samples[i]
                location_folds[(s["town"], s["junction_id"])].add(int(folds[j]))
            if len(location_folds) != 21 or any(len(f) != 1 for f in location_folds.values()):
                raise ValueError("Whole-junction fold consistency required")
            selected = set(read(fit / "cache_verification.json")["selected_images"])
            spec = importlib.util.spec_from_file_location("staged_spatial", module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            features = np.lib.format.open_memmap(
                out / "features.npy", mode="w+", dtype=np.float16, shape=(3, len(rows), 288)
            )
            verified = 0
            for i in data.train:
                check()
                ids = grouped[i]
                xy = np.array([[rows[j]["x"], rows[j]["y"]] for j in ids], np.int64)
                rgb = data.images[i].transpose(1, 2, 0)
                for version, image in enumerate([rgb, *[gamma_rgb(rgb, g) for g in GAMMAS]]):
                    values = module.spatial_context(image, xy)
                    if i in selected:
                        reference = direct_features(image, xy)
                        if not np.array_equal(values, reference):
                            raise ValueError(f"Direct-pixel FP32 parity failed sample {i}")
                        verified += len(ids)
                    features[version, ids] = values.astype(np.float16)
                done += 1
                if done % 20 == 0:
                    features.flush()
                    phase = f"Spatial TRAIN features {done}/2844"
                    progress()
            features.flush()
            np.save(out / "fold.npy", folds)
            atomic_json(out / "rows.json", rows)
            if not np.isfinite(features).all():
                raise ValueError("Nonfinite features")
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Source changed during cache")
            check()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    no_component_val=True,
                    rows=len(rows),
                    original_rows=len(old_rows),
                    extra_rows=len(extra_rows),
                    features=288,
                    whole_junction_folds_verified=True,
                    negative_eligibility_verified=True,
                    direct_pixel_images=sorted(selected),
                    exact_direct_pixel_rows=verified,
                    elapsed_seconds=time.monotonic() - started,
                    artifacts={
                        name: digest(out / name)
                        for name in ("features.npy", "fold.npy", "rows.json")
                    },
                ),
            )
            status, phase = "complete", "Spatial TRAIN cache complete; no fitting or component VAL"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
