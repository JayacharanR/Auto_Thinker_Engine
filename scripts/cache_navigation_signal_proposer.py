#!/usr/bin/env python3
"""Hash-verified frozen proposer features, retaining every original RGB point."""

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_context import context_patches  # noqa:E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa:E402
from src.navigation.signal_pretrained_verifier import rgb_features  # noqa:E402
from src.navigation.signal_proposer_features import (  # noqa:E402
    PROPOSER_SHA256,
    FrozenBulbFeatures,
    point_features,
    verify_ordered_dataset,
)
from src.navigation.signal_rgb_augmentation import GAMMAS, gamma_rgb  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

CACHE = ROOT / "outputs/navigation_signal_pretrained_20261021"
AUGMENTED = ROOT / "outputs/navigation_signal_rgb_boost_v2_20261021"
PARENT = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--deadline-from", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    deadline = time.time() + 1800
    if args.deadline_from:
        old = read(args.deadline_from)
        if old["status"] != "failed" or old["frames"] != 0:
            raise ValueError("Only zero-feature preflight failure may supply original deadline")
        deadline = old["updated_at"] - old["elapsed_seconds"] + 1800
    status, error, phase, done = "running", None, "verify original cache and sources", 0

    def check():
        if (out / "STOP").exists() or time.time() > deadline:
            raise InterruptedError("Owned STOP or 1800-second cache limit")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 150_000_000:
            raise InterruptedError("150 MB cache artifact limit")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Frozen proposer visual feature cache",
                status=status,
                error=error,
                phase=phase,
                frames=done,
                target_frames=2988,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            progress()
            if not torch.cuda.is_available():
                raise RuntimeError("Verified CUDA readiness required; no silent CPU fallback")
            files = [
                Path(__file__),
                ROOT / "src/navigation/signal_proposer_features.py",
                ROOT / "src/navigation/signal_pixel_study.py",
                ROOT / "src/navigation/signal_learning.py",
                ROOT / "src/navigation/signal_diversity_data.py",
                ROOT / "src/navigation/signal_context.py",
                ROOT / "src/navigation/signal_pretrained_verifier.py",
                ROOT / "src/navigation/signal_rgb_augmentation.py",
                ROOT / "reports/navigation_signal_proposer_fusion_protocol_20261021.md",
                CACHE / "cache.json",
                CACHE / "train_rows.json",
                CACHE / "dataset.json",
                AUGMENTED / "cache.json",
                AUGMENTED / "assessment.json",
                PARENT / "predictions1000.json",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021/manifest.json",
                ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz",
            ]
            # Dataset constructors also use geometry/calibration helpers. Pin the
            # whole navigation source generation so an indirect edit is detected.
            files.extend((ROOT / "src/navigation").glob("*.py"))
            files.append(ROOT / "scripts/collect_navigation_signal_pairs.py")
            if args.deadline_from:
                files.append(args.deadline_from.resolve())
            files = sorted(set(files))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            cache, augmentation = read(CACHE / "cache.json"), read(AUGMENTED / "cache.json")
            feature_hashes = {
                str(CACHE / "train_features.npz"): cache["train_features_sha256"],
                str(CACHE / "validation_features.npz"): cache["validation_features_sha256"],
                str(AUGMENTED / "augmented.npz"): augmentation["augmented_sha256"],
            }
            for path, sha in feature_hashes.items():
                if digest(path) != sha:
                    raise ValueError("Original verified feature cache changed")
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    original_feature_hashes=feature_hashes,
                    proposer_sha256=PROPOSER_SHA256,
                    stage_seconds=1800,
                    artifact_bytes=150_000_000,
                    inference_batch_images=1,
                    inference_dtype="bfloat16",
                    cache_dtype="float16",
                    feature_dim=36,
                    gammas=[1.0, *GAMMAS],
                    no_fit=True,
                    no_native=True,
                    downstream_fold_exposure_limit_disclosed=True,
                    original_deadline=deadline,
                ),
            )
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            rows = read(CACHE / "train_rows.json")
            original_data = read(CACHE / "dataset.json")
            verify_ordered_dataset(data.samples, data.train, data.validation, original_data)
            with np.load(CACHE / "train_features.npz") as a:
                train_rgb, labels = a["rgb"], a["labels"]
            with np.load(AUGMENTED / "augmented.npz") as a:
                augmented_rgb = a["features"]
            with np.load(CACHE / "validation_features.npz") as a:
                val_rgb, offsets = a["rgb"], a["offsets"]
            if len(rows) != 51094 or offsets[-1] != 38543:
                raise ValueError("Original point denominators changed")
            if not np.array_equal(labels, [r["label"] for r in rows]):
                raise ValueError("Original TRAIN labels changed")
            grouped = defaultdict(list)
            for j, row in enumerate(rows):
                if row["sample"] not in data.train or row["role"] != "train":
                    raise ValueError("Non-TRAIN point in TRAIN cache")
                grouped[row["sample"]].append(j)
            model = FrozenBulbFeatures(PARENT / "step1000.pt").cuda()
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            train = np.empty((3, len(rows), 36), np.float16)
            validation = np.empty((int(offsets[-1]), 36), np.float16)
            filled = np.zeros(len(rows), bool)
            parity_count = 0

            def extract(image, points):
                nonlocal parity_count
                check()
                tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    maps, probabilities = model(tensor)
                    reference = model.model(tensor).float().softmax(1)
                if not torch.equal(probabilities, reference):
                    raise ValueError("Feature extractor differs from original same-batch forward")
                parity_count += 1
                pts = torch.from_numpy(points).cuda()
                return (
                    point_features(maps[0], probabilities[0], pts).to(torch.float16).cpu().numpy()
                )

            phase = "Original and gamma TRAIN points; same full-image forward parity"
            for index in data.train:
                check()
                indices = grouped[index]
                if indices:
                    points = np.array([[rows[j]["x"], rows[j]["y"]] for j in indices], np.int64)
                    image = data.images[index].transpose(1, 2, 0)
                    for gi, gamma in enumerate((1.0, *GAMMAS)):
                        transformed = image if gi == 0 else gamma_rgb(image, gamma)
                        rgb = rgb_features(context_patches(transformed, points)).astype(np.float16)
                        expected = train_rgb[indices] if gi == 0 else augmented_rgb[gi - 1, indices]
                        if not np.array_equal(rgb, expected):
                            raise ValueError("Original or augmented RGB parity failed")
                        train[gi, indices] = extract(transformed, points)
                    filled[indices] = True
                done += 1
                if done % 30 == 0:
                    progress()
            if not filled.all() or not np.isfinite(train).all():
                raise ValueError("Incomplete TRAIN feature cache")
            np.savez_compressed(out / "train_proposer_features.npz", features=train)
            del train, train_rgb, augmented_rgb
            peaks = {r["sample"]: r["peaks"] for r in read(PARENT / "predictions1000.json")}
            discrepancies, state_disagreements = [], 0
            phase = "All original VAL points; no classifier fitting or labels used"
            for n, index in enumerate(data.validation):
                check()
                points = np.array([[p["x"], p["y"]] for p in peaks[index]], np.int64).reshape(-1, 2)
                image = data.images[index].transpose(1, 2, 0)
                if offsets[n + 1] - offsets[n] != len(points):
                    raise ValueError("VAL point offsets changed")
                rgb = rgb_features(context_patches(image, points)).astype(np.float16)
                if not np.array_equal(rgb, val_rgb[offsets[n] : offsets[n + 1]]):
                    raise ValueError("Original VAL RGB features changed")
                if len(points):
                    features = extract(image, points)
                    validation[offsets[n] : offsets[n + 1]] = features
                    for row, feature in zip(peaks[index], features):
                        scores = feature[32:].astype(float)
                        discrepancies.append(abs(scores[row["cls"]] - row["score"]))
                        state_disagreements += int(scores[1:].argmax() + 1 != row["cls"])
                done += 1
                if done % 12 == 0:
                    progress()
            if not np.isfinite(validation).all():
                raise ValueError("Incomplete VAL feature cache")
            np.savez_compressed(
                out / "validation_proposer_features.npz", features=validation, offsets=offsets
            )
            for key, value in before.items():
                if not torch.equal(value, model.state_dict()[key].detach().cpu()):
                    raise ValueError("Frozen model weight or buffer changed")
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Registered source changed during extraction")
            for path, sha in feature_hashes.items():
                if digest(path) != sha:
                    raise ValueError("Original feature cache changed during extraction")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_images=2844,
                    validation_images=144,
                    train_points=51094,
                    train_versions=3,
                    validation_points=38543,
                    same_forward_exact_checks=parity_count,
                    frozen_model_verified=True,
                    original_and_augmented_rgb_exact=True,
                    original_val_score_max_abs_difference=float(max(discrepancies, default=0)),
                    original_val_point_class_disagreements=state_disagreements,
                    discrepancy_scope="Batch1 BF16 plus FP16 cache versus original saved scores",
                    train_sha256=digest(out / "train_proposer_features.npz"),
                    validation_sha256=digest(out / "validation_proposer_features.npz"),
                    feature_dim=36,
                    no_fit=True,
                    no_deployment=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = (
                "complete",
                "Frozen cache complete; independent audit required before fits",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
