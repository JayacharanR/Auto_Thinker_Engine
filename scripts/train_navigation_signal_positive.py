#!/usr/bin/env python3
"""Three fixed TRAIN-junction fits with newly verified exact active-pixel positives."""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_phase_supervision import (  # noqa: E402
    phase_balanced_weights,
    phase_targets,
)
from src.navigation.signal_supported_supervision import sampled_housing_score  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402

CACHE = ROOT / "outputs/navigation_signal_pretrained_20261021"
SPATIAL = ROOT / "outputs/navigation_signal_spatial_cache_20261022"
EXTRA = ROOT / "outputs/navigation_signal_background_cache_20261022"
BASELINE = ROOT / "outputs/navigation_signal_background_fit_20261022"
POSITIVE = ROOT / "outputs/navigation_signal_positive_cache_20261022"
CONTROL = ROOT / "outputs/navigation_signal_spatial_fit_20261022"
AUGMENT = ROOT / "outputs/navigation_signal_rgb_boost_v2_20261021"
FOLDS = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"


def read(path):
    return json.loads(Path(path).read_text())


def verify(request):
    for path, sha in request["source_hashes"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Pinned source changed: {path}")
    for path, sha in request["cache_hashes"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Pinned cache changed: {path}")


def worker(out, fold):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    if fold not in (0, 1, 2):
        raise ValueError("Only three internal TRAIN folds; no final fit")
    started = time.monotonic()
    request = read(out.parent / "request.json")
    verify(request)
    if (out / "STOP").exists():
        raise InterruptedError("Owned STOP")
    with np.load(CACHE / "train_features.npz") as a:
        raw, binary = a["rgb"], a["labels"]
    labels = phase_targets(read(CACHE / "train_rows.json"), read(CACHE / "dataset.json"), binary)
    with np.load(FOLDS) as a:
        old_fold = a["fold"]
    with np.load(EXTRA / "features.npz") as a:
        extra, extra_fold = a["features"], a["fold"]
    with np.load(AUGMENT / "augmented.npz") as a:
        augmented = a["features"]
    spatial = np.load(SPATIAL / "features.npy", mmap_mode="r")
    keep = np.load(BASELINE / "training_keep.npy")
    if keep.shape != (51094,) or keep.dtype != np.bool_ or keep.sum() != 49088:
        raise ValueError("Original supported-supervision mask required")
    if spatial.shape != (3, 233110, 288) or extra.shape != (3, 182016, 87):
        raise ValueError("Original aligned feature shapes required")
    mask, emask = keep & (old_fold != fold), extra_fold != fold
    # Preserve the raw control's version-major original rows followed by version-major extras.
    x = np.concatenate(
        [
            *[
                np.concatenate([v[mask], spatial[k, :51094][mask]], 1).astype(np.float32)
                for k, v in enumerate([raw, *augmented])
            ],
            *[
                np.concatenate([v[emask], spatial[k, 51094:][emask]], 1).astype(np.float32)
                for k, v in enumerate(extra)
            ],
        ]
    )
    y = np.concatenate(
        [np.tile(labels[mask], 3), np.zeros(3 * int(emask.sum()), dtype=labels.dtype)]
    )
    with np.load(POSITIVE / "features.npz") as positive:
        positive_features, positive_labels, positive_fold = (
            positive["features"],
            positive["labels"],
            positive["fold"],
        )
    if positive_features.shape != (3, 1596, 375) or not np.isin(positive_labels, [1, 2, 3]).all():
        raise ValueError("Exact released positive feature/label shape required")
    pmask = positive_fold != fold
    x = np.concatenate([x, *[v[pmask].astype(np.float32) for v in positive_features]])
    y = np.concatenate([y, np.tile(positive_labels[pmask], 3)])
    all_x = np.concatenate(
        [
            np.concatenate([raw, spatial[0, :51094]], 1),
            np.concatenate([extra[0], spatial[0, 51094:]], 1),
            positive_features[0],
        ]
    ).astype(np.float32)
    with threadpool_limits(limits=2):
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=10,
            class_weight=None,
            early_stopping=False,
            random_state=42,
            max_bins=255,
        ).fit(x, y, sample_weight=phase_balanced_weights(y))
        probability = model.predict_proba(all_x)
        blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        if not np.array_equal(pickle.loads(blob).predict_proba(all_x), probability):
            raise ValueError("Saved model prediction parity failed")
    if (out / "STOP").exists():
        raise InterruptedError("Owned STOP after fit")
    if not np.array_equal(model.classes_, [0, 1, 2, 3]) or model.n_features_in_ != 375:
        raise ValueError("Exact class/feature contract required")
    (out / "model.pkl").write_bytes(blob)
    np.save(out / "probability.npy", probability)
    atomic_json(
        out / "fit.json",
        dict(
            excluded_fold=fold,
            original_examples=int(mask.sum()),
            extra_examples=int(emask.sum()),
            positive_examples=int(pmask.sum()),
            augmented_examples=len(y),
            features=375,
            trees=sum(len(r) for r in model._predictors),
            no_component_val=True,
            model_sha256=digest(out / "model.pkl"),
            probability_sha256=digest(out / "probability.npy"),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fit-worker", action="store_true")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    if args.fit_worker:
        return worker(out, args.fold)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, error, done, phase = "running", None, 0, "verify released positive cache"

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Positive-coverage RGB TRAIN fits",
                status=status,
                phase=phase,
                error=error,
                fits_done=done,
                fits_total=3,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1200:
            raise InterruptedError("Owned STOP or 1200 second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("100 MB artifact cap")

    try:
        progress()
        with BudgetWatch(out):
            positive_result = read(POSITIVE / "assessment.json")
            if (
                read(POSITIVE / "progress.json")["status"] != "complete"
                or not positive_result["training_release"]
                or positive_result["exact_feature_rows"] != 4788
                or not positive_result["whole_junction_folds_verified"]
            ):
                raise ValueError("Complete independent positive release required")
            result = read(SPATIAL / "assessment.json")
            if (
                read(SPATIAL / "progress.json")["status"] != "complete"
                or not result["whole_junction_folds_verified"]
                or len(result["direct_pixel_images"]) != 63
            ):
                raise ValueError("Complete audited TRAIN feature cache required")
            files = [
                Path(__file__),
                ROOT / "scripts/train_navigation_signal_supported.py",
                ROOT / "reports/navigation_signal_positive_protocol_20261022.md",
                SPATIAL / "assessment.json",
                SPATIAL / "request.json",
                BASELINE / "assessment.json",
                CONTROL / "oof.npy",
                CONTROL / "assessment.json",
                POSITIVE / "assessment.json",
                POSITIVE / "request.json",
                BASELINE / "training_keep.npy",
                CACHE / "dataset.json",
                CACHE / "train_rows.json",
                FOLDS,
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            files += [BASELINE / f"joint100_fold{fold}/fit.json" for fold in range(3)]
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for p in files:
                target = out / "source" / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
            cache_hashes = {
                str((SPATIAL / name).relative_to(ROOT)): sha
                for name, sha in result["artifacts"].items()
            }
            cache_hashes.update(
                {
                    str((POSITIVE / name).relative_to(ROOT)): sha
                    for name, sha in positive_result["artifacts"].items()
                }
            )
            cache_hashes.update(
                {
                    str((EXTRA / name).relative_to(ROOT)): sha
                    for name, sha in read(EXTRA / "assessment.json")["artifacts"].items()
                }
            )
            cache_hashes[str((CACHE / "train_features.npz").relative_to(ROOT))] = read(
                CACHE / "cache.json"
            )["train_features_sha256"]
            cache_hashes[str((AUGMENT / "augmented.npz").relative_to(ROOT))] = read(
                AUGMENT / "cache.json"
            )["augmented_sha256"]
            request = dict(
                source_hashes=hashes,
                cache_hashes=cache_hashes,
                created_at=time.time(),
                no_component_val=True,
                no_final_fit=True,
                stage_seconds=1200,
                fit_seconds=300,
                artifact_cap_bytes=100_000_000,
            )
            verify(request)
            atomic_json(out / "request.json", request)
            with np.load(POSITIVE / "features.npz") as positive:
                folds = np.concatenate([np.load(SPATIAL / "fold.npy"), positive["fold"]])
            probability = np.full((len(folds), 4), np.nan)
            for fold in range(3):
                check()
                destination = out / f"fold{fold}"
                destination.mkdir()
                phase = f"Positive-coverage fold {fold}; no final fit"
                progress()
                with (destination / "worker.log").open("xb") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(Path(__file__).resolve()),
                            "--fit-worker",
                            "--out",
                            str(destination),
                            "--fold",
                            str(fold),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(
                            os.environ,
                            OMP_NUM_THREADS="2",
                            OPENBLAS_NUM_THREADS="2",
                            MKL_NUM_THREADS="2",
                            CUDA_VISIBLE_DEVICES="",
                        ),
                    )
                    atomic_json(
                        destination / "worker_identity.json",
                        dict(pid=child.pid, command=child.args, started_at=time.time()),
                    )
                    launch = time.monotonic()
                    try:
                        while child.poll() is None:
                            check()
                            if time.monotonic() - launch > 300:
                                raise InterruptedError("300 second fold cap")
                            time.sleep(0.25)
                    finally:
                        if child.poll() is None:
                            (destination / "STOP").touch()
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.terminate()
                                child.wait(timeout=20)
                                atomic_json(
                                    destination / "normal_cleanup.json",
                                    dict(STOP_present=True, signal="SIGTERM", force_kill=False),
                                )
                    if child.returncode:
                        raise RuntimeError(f"Fold {fold} failed; no retry")
                receipt = read(destination / "fit.json")
                control_receipt = read(BASELINE / f"joint100_fold{fold}/fit.json")
                if (
                    receipt["original_examples"] != control_receipt["original_train_examples"]
                    or receipt["extra_examples"] * 3 != control_receipt["extra_background_examples"]
                    or receipt["augmented_examples"] - 3 * receipt["positive_examples"]
                    != control_receipt["fitted_examples"]
                    or digest(destination / "model.pkl") != receipt["model_sha256"]
                    or digest(destination / "probability.npy") != receipt["probability_sha256"]
                ):
                    raise ValueError("Raw-control fit counts or saved model identity differ")
                if (
                    receipt["excluded_fold"] != fold
                    or receipt["trees"] != 400
                    or not receipt["no_component_val"]
                ):
                    raise ValueError("Fold contract mismatch")
                p = np.load(destination / "probability.npy")
                probability[folds == fold] = p[folds == fold]
                done += 1
                progress()
            if not np.isfinite(probability).all():
                raise ValueError("Incomplete OOF predictions")
            np.save(out / "oof.npy", probability)
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            with np.load(FOLDS) as a:
                predicted = a["predicted"]
            rows = read(CACHE / "train_rows.json")
            results = {
                name: sampled_housing_score(
                    p, rows, predicted, data.samples, data.boxes, data.train
                )
                for name, p in (
                    ("control", np.load(CONTROL / "oof.npy")[:51094]),
                    ("spatial", probability[:51094]),
                )
            }
            verify(request)
            check()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    no_component_val=True,
                    no_candidate_qualification=True,
                    sampled_counts_are_not_qualification=True,
                    results=results,
                    oof_sha256=digest(out / "oof.npy"),
                    requires_separately_registered_full_proposal_audit=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = (
                "complete",
                "Three positive-coverage TRAIN fits complete; full audit required",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
