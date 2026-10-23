#!/usr/bin/env python3
"""Independently release and cache exact additional TRAIN active-pixel candidates."""

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

from scripts.cache_navigation_signal_spatial_context import direct_features  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_context import context_patches  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_pretrained_verifier import rgb_features  # noqa: E402
from src.navigation.signal_rgb_augmentation import GAMMAS, gamma_rgb  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    out = ROOT / "outputs/navigation_signal_positive_cache_20261022"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, error, done = "running", None, 0
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Exact positive TRAIN cache",
                status=status,
                frames=done,
                target_frames=1596,
                error=error,
                phase="Independent eligibility and feature verification",
                elapsed_seconds=time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            diagnosis = ROOT / "outputs/navigation_signal_positive_coverage_20261022"
            source = ROOT / "outputs/navigation_signal_pretrained_20261021"
            inventory = ROOT / "outputs/navigation_signal_plateau_inventory_20261022"
            fold_path = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
            module_file = ROOT / "patches/navigation_signal_spatial_context_candidate.py"
            assessment = read(diagnosis / "assessment.json")
            if (
                read(diagnosis / "progress.json")["status"] != "complete"
                or not assessment["no_training_release"]
            ):
                raise ValueError("Original closed diagnostic required")
            row_path = diagnosis / "new_candidate_rows.json"
            if digest(row_path) != assessment["new_rows_sha256"]:
                raise ValueError("Exact diagnostic rows required")
            files = [
                Path(__file__),
                ROOT / "scripts/cache_navigation_signal_spatial_context.py",
                ROOT / "reports/navigation_signal_positive_protocol_20261022.md",
                module_file,
                row_path,
                diagnosis / "assessment.json",
                source / "train_rows.json",
                source / "dataset.json",
                fold_path,
                inventory / "assessment.json",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for p in files:
                target = out / "source" / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    stage_seconds=300,
                    artifact_cap_bytes=30_000_000,
                    no_component_val=True,
                ),
            )
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            identity = read(source / "dataset.json")
            if (
                json.dumps(identity["samples"], sort_keys=True)
                != json.dumps(data.samples, sort_keys=True)
                or data.train != identity["train"]
            ):
                raise ValueError("Original ordered dataset changed")
            rows = read(row_path)
            old_rows = read(source / "train_rows.json")
            used = {(r["sample"], r["x"], r["y"]) for r in old_rows}
            used.update(
                (r["sample"], r["x"], r["y"])
                for r in read(
                    ROOT / "outputs/navigation_signal_background_cache_20261022/rows.json"
                )
            )
            with np.load(fold_path) as a:
                old_fold = a["fold"]
            by_sample = defaultdict(set)
            for r, f in zip(old_rows, old_fold):
                by_sample[r["sample"]].add(int(f))
            for name, sha in read(inventory / "assessment.json")["artifact_sha256"].items():
                if digest(inventory / name) != sha:
                    raise ValueError("Candidate inventory changed")
            points = np.load(inventory / "points.npy", mmap_mode="r")
            with np.load(inventory / "offsets.npz") as a:
                offsets, samples = a["offsets"], a["samples"]
            if not np.array_equal(samples, data.train) or len(rows) != 1596:
                raise ValueError("Original TRAIN ordering/positive count required")
            position = {int(i): j for j, i in enumerate(samples)}
            grouped = defaultdict(list)
            folds = []
            seen = set()
            for j, r in enumerate(rows):
                i, x, y, c = r["sample"], r["x"], r["y"], r["label"]
                key = (i, x, y)
                if (
                    i not in position
                    or r["role"] != "train"
                    or key in used
                    or key in seen
                    or len(by_sample[i]) != 1
                    or c not in (1, 2, 3)
                    or data.targets["bulb"][i, y, x] != c
                    or data.background[i, y, x]
                ):
                    raise ValueError("Independent exact positive eligibility failed")
                seen.add(key)
                folds.append(next(iter(by_sample[i])))
                grouped[i].append(j)
            spec = importlib.util.spec_from_file_location("staged_positive_context", module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            features = np.empty((3, len(rows), 375), np.float16)
            for i, ids in grouped.items():
                if (out / "STOP").exists() or time.monotonic() - started > 300:
                    raise InterruptedError("Owned STOP or 300 second cap")
                n = position[i]
                allowed = {tuple(p) for p in points[offsets[n] : offsets[n + 1]]}
                xy = np.array([[rows[j]["x"], rows[j]["y"]] for j in ids], np.int64)
                if any(tuple(p) not in allowed for p in xy):
                    raise ValueError("Positive is not a frozen candidate")
                rgb = data.images[i].transpose(1, 2, 0)
                for version, image in enumerate([rgb, *[gamma_rgb(rgb, g) for g in GAMMAS]]):
                    raw = rgb_features(context_patches(image, xy))
                    spatial = module.spatial_context(image, xy)
                    if not np.array_equal(spatial, direct_features(image, xy)):
                        raise ValueError("Direct-pixel spatial parity failed")
                    features[version, ids] = np.concatenate([raw, spatial], 1).astype(np.float16)
                done += len(ids)
                if done % 50 < len(ids):
                    progress()
            np.savez_compressed(
                out / "features.npz",
                features=features,
                fold=np.array(folds),
                labels=np.array([r["label"] for r in rows]),
            )
            atomic_json(out / "rows.json", rows)
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Registered source changed")
            if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 30_000_000:
                raise InterruptedError("30 MB cap")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    training_release=True,
                    rows=1596,
                    exact_feature_rows=4788,
                    whole_junction_folds_verified=True,
                    original_diagnostic_unchanged=True,
                    no_component_val=True,
                    elapsed_seconds=time.monotonic() - started,
                    artifacts={name: digest(out / name) for name in ("features.npz", "rows.json")},
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
