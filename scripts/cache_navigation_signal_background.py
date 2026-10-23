#!/usr/bin/env python3
"""Model-independent TRAIN background sampling with original RGB feature recipe."""

import argparse
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
from src.navigation.signal_background_sampling import select_background  # noqa: E402
from src.navigation.signal_context import context_patches  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_pretrained_verifier import rgb_features  # noqa: E402
from src.navigation.signal_rgb_augmentation import GAMMAS, gamma_rgb  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    original = ROOT / "outputs/navigation_signal_pretrained_20261021"
    inventory = ROOT / "outputs/navigation_signal_plateau_inventory_20261022"
    folds_path = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
    started = time.monotonic()
    status, phase, error, done = "running", "verify original TRAIN sources", None, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1200:
            raise InterruptedError("Owned STOP or1200second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 200_000_000:
            raise InterruptedError("200MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Expanded proposal TRAIN background cache",
                status=status,
                phase=phase,
                error=error,
                frames=done,
                target_frames=2844,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            result = read(inventory / "assessment.json")
            if read(inventory / "progress.json")["status"] != "complete":
                raise ValueError("Closed original candidate inventory required")
            for name, sha in result["artifact_sha256"].items():
                if digest(inventory / name) != sha:
                    raise ValueError("Original candidate inventory changed")
            files = [
                Path(__file__),
                ROOT / "reports/navigation_signal_background_protocol_20261022.md",
                inventory / "assessment.json",
                folds_path,
                original / "train_rows.json",
                original / "dataset.json",
                original / "cache.json",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                if path.suffix in (".py", ".md", ".json"):
                    target = out / "source" / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    inventory_points_sha256=digest(inventory / "points.npy"),
                    created_at=time.time(),
                    seed=42,
                    per_image_limit=64,
                    model_outputs_used=False,
                    no_component_val=True,
                    gammas=GAMMAS,
                    stage_seconds=1200,
                    artifact_cap_bytes=200_000_000,
                ),
            )
            rows = read(original / "train_rows.json")
            dataset = read(original / "dataset.json")
            with np.load(folds_path) as a:
                folds = a["fold"]
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            if (
                json.dumps(data.samples, sort_keys=True)
                != json.dumps(dataset["samples"], sort_keys=True)
                or data.train != dataset["train"]
                or data.validation != dataset["validation"]
            ):
                raise ValueError("Original ordered dataset changed")
            old = defaultdict(list)
            sample_folds = defaultdict(set)
            location_folds = defaultdict(set)
            train = set(data.train)
            for j, row in enumerate(rows):
                i = row["sample"]
                if i not in train or row["role"] != "train":
                    raise ValueError("Only original TRAIN rows allowed")
                old[i].append((row["x"], row["y"]))
                sample_folds[i].add(int(folds[j]))
                s = data.samples[i]
                location_folds[(s["town"], s["junction_id"])].add(int(folds[j]))
            if len(location_folds) != 21 or any(len(v) != 1 for v in location_folds.values()):
                raise ValueError("Original21whole-junction folds required")
            xy = np.load(inventory / "points.npy", mmap_mode="r")
            with np.load(inventory / "offsets.npz") as a:
                offsets, indices = a["offsets"], a["samples"]
            if not np.array_equal(indices, data.train):
                raise ValueError("Original TRAIN candidate ordering required")
            selected, extra_rows = [], []
            for n, i in enumerate(data.train):
                check()
                candidates = xy[offsets[n] : offsets[n + 1]]
                chosen = select_background(candidates, data.background[i], old[i], i)
                selected.append(candidates[chosen].copy())
                if len(sample_folds[i]) != 1:
                    raise ValueError("One original fold per TRAIN image required")
                fold = next(iter(sample_folds[i]))
                for x, y in candidates[chosen]:
                    extra_rows.append(
                        dict(sample=i, x=int(x), y=int(y), fold=fold, label=0, role="train")
                    )
            atomic_json(out / "rows.json", extra_rows)
            features = np.empty((3, len(extra_rows), 87), np.float16)
            cursor = 0
            for i, points in zip(data.train, selected):
                check()
                image = data.images[i].transpose(1, 2, 0)
                for version, transformed in enumerate(
                    [image, *[gamma_rgb(image, g) for g in GAMMAS]]
                ):
                    if len(points):
                        features[version, cursor : cursor + len(points)] = rgb_features(
                            context_patches(transformed, points)
                        )
                cursor += len(points)
                done += 1
                if done % 20 == 0:
                    phase = f"TRAIN background features: {done}/2844"
                    progress()
            if cursor != len(extra_rows) or not np.isfinite(features).all():
                raise ValueError("Incomplete finite features")
            np.savez_compressed(
                out / "features.npz",
                features=features,
                fold=np.array([r["fold"] for r in extra_rows], np.int64),
            )
            check()
            # Independent eligibility, overlap and per-fold geography recount.
            seen = set()
            for row in extra_rows:
                i, x, y = row["sample"], row["x"], row["y"]
                identity = (i, x, y)
                s = data.samples[i]
                if (
                    identity in seen
                    or i not in train
                    or not data.background[i, y, x]
                    or (x, y) in old[i]
                    or row["fold"] not in location_folds[(s["town"], s["junction_id"])]
                ):
                    raise ValueError("Extra negative eligibility/identity/fold audit failed")
                seen.add(identity)
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Pinned source changed")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    TRAIN_only=True,
                    no_model_score_mining=True,
                    no_component_val=True,
                    images=done,
                    extra_rows=len(extra_rows),
                    feature_shape=list(features.shape),
                    all_known_background=True,
                    no_original_coordinate_overlap=True,
                    whole_junction_folds_verified=True,
                    per_fold={str(f): sum(r["fold"] == f for r in extra_rows) for f in range(3)},
                    artifacts={name: digest(out / name) for name in ("rows.json", "features.npz")},
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = (
                "complete",
                "Expanded background cache complete; no fitting or component VAL",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
