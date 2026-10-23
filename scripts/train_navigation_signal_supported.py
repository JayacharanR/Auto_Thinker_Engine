#!/usr/bin/env python3
"""Three bounded TRAIN-only fits of locally supported signal supervision."""

import argparse
import json
import os
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
from src.navigation.signal_group_selection import macro_location_logloss  # noqa: E402
from src.navigation.signal_supported_supervision import (  # noqa: E402
    sampled_housing_score,
    supported_training_rows,
)
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned project output required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, phase, error, done = "running", "verify TRAIN sources", None, 0
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    fold_root = ROOT / "outputs/navigation_signal_regularized_20261021"
    augmented = ROOT / "outputs/navigation_signal_rgb_boost_v2_20261021"
    baseline = ROOT / "outputs/navigation_signal_phase_boost_20261021"
    support = ROOT / "reports/navigation_signal_support_geometry_20261022.json"
    worker = ROOT / "scripts/train_navigation_signal_rgb_boost.py"

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Owned STOP or 900-second stage cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("100 MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="TRAIN-only locally supported supervision",
                status=status,
                phase=phase,
                error=error,
                fits_done=done,
                fits_total=3,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            files = [
                Path(__file__),
                worker,
                support,
                ROOT / "reports/navigation_signal_supported_protocol_20261022.md",
                cache / "train_rows.json",
                cache / "dataset.json",
                cache / "cache.json",
                augmented / "cache.json",
                fold_root / "request.json",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            files += [baseline / f"joint100_fold{f}/train_class_probability.npy" for f in range(3)]
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            request = dict(
                source_hashes=hashes,
                train_features_sha256=read(cache / "cache.json")["train_features_sha256"],
                folds_sha256=read(fold_root / "request.json")["folds_sha256"],
                training_keep_path=str(out / "training_keep.npy"),
                no_component_validation=True,
                stage_seconds=900,
                fit_seconds=180,
                artifact_cap_bytes=100_000_000,
                created_at=time.time(),
            )
            if digest(cache / "train_features.npz") != request["train_features_sha256"]:
                raise ValueError("TRAIN cache changed")
            if digest(fold_root / "folds.npz") != request["folds_sha256"]:
                raise ValueError("TRAIN folds changed")
            aug_hash = read(augmented / "cache.json")["augmented_sha256"]
            if digest(augmented / "augmented.npz") != aug_hash:
                raise ValueError("TRAIN augmentation cache changed")
            atomic_json(
                out / "cache.json",
                dict(
                    augmented_path=str(augmented / "augmented.npz"),
                    augmented_sha256=aug_hash,
                ),
            )
            rows = read(cache / "train_rows.json")
            with np.load(fold_root / "folds.npz") as a:
                folds, predicted, groups = a["fold"], a["predicted"], a["groups"]
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            recorded = read(cache / "dataset.json")
            if (
                json.dumps(data.samples, sort_keys=True)
                != json.dumps(recorded["samples"], sort_keys=True)
                or data.train != recorded["train"]
                or data.validation != recorded["validation"]
            ):
                raise ValueError("Ordered dataset changed")
            keep = supported_training_rows(rows, read(support)["details"], data.train, predicted)
            anchor_count = 0
            classes = {"red": 1, "red_repeat": 1, "yellow": 2, "green": 3}
            for j in np.flatnonzero(~predicted):
                row = rows[j]
                cls = classes[data.samples[row["sample"]]["phase"]]
                if (
                    row["label"] != 1
                    or data.targets["bulb"][row["sample"], row["y"], row["x"]] != cls
                ):
                    raise ValueError("GT-only anchor is not an audited active TRAIN pixel")
                anchor_count += 1
            if len(rows) != 51094 or int((~keep).sum()) != 2006 or anchor_count != 2229:
                raise ValueError("Registered TRAIN denominator changed")
            np.save(out / "training_keep.npy", keep)
            request.update(
                training_keep_sha256=digest(out / "training_keep.npy"),
                kept_rows=int(keep.sum()),
                excluded_unknown_positive=2006,
                verified_gt_anchors=anchor_count,
            )
            atomic_json(out / "request.json", request)
            samples, boxes, train_indices = data.samples, data.boxes, data.train
            del data
            control, candidate = np.full((len(rows), 4), np.nan), np.full((len(rows), 4), np.nan)
            for fold in range(3):
                check()
                phase = f"joint100 supported fold {fold}"
                progress()
                destination = out / f"joint100_fold{fold}"
                destination.mkdir()
                with (destination / "worker.log").open("xb") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(worker),
                            "--fit-worker",
                            "--out",
                            str(destination),
                            "--arm",
                            "joint100",
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
                        dict(
                            pid=child.pid,
                            command=child.args,
                            started_at=time.time(),
                        ),
                    )
                    launched = time.monotonic()
                    try:
                        while child.poll() is None:
                            check()
                            if time.monotonic() - launched > 180:
                                raise InterruptedError("180-second owned fit cap")
                            progress()
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
                        raise RuntimeError(f"Fold {fold} failed with {child.returncode}; no retry")
                fit = read(destination / "fit.json")
                if fit["trees"] != 400 or fit["component_validation_read"]:
                    raise ValueError("TRAIN-only fixed-budget fit contract violated")
                if fit["original_train_examples"] != int((keep & (folds != fold)).sum()):
                    raise ValueError("Fold exclusion or supervision mask mismatch")
                held = folds == fold
                candidate[held] = np.load(destination / "train_class_probability.npy")[held]
                control[held] = np.load(
                    baseline / f"joint100_fold{fold}/train_class_probability.npy"
                )[held]
                done += 1
                progress()
            if not np.isfinite(candidate).all() or not np.isfinite(control).all():
                raise ValueError("Incomplete OOF probabilities")
            np.save(out / "oof_supported.npy", candidate)
            np.save(out / "oof_control.npy", control)
            results = {}
            labels = np.array([r["label"] for r in rows])
            for arm, p in (("control", control), ("supported", candidate)):
                results[arm] = sampled_housing_score(
                    p, rows, predicted, samples, boxes, train_indices
                )
                results[arm]["macro_point_logloss"] = macro_location_logloss(
                    p[:, 1:].sum(1), labels, groups, predicted
                )[0]
            a, b = results["supported"], results["control"]
            criteria = dict(
                joint_preserved=a["joint"] >= b["joint"],
                each_town_preserved=all(
                    a["towns"][k]["joint"] >= v["joint"] for k, v in b["towns"].items()
                ),
                each_phase_preserved=all(
                    a["classes"][k]["joint"] >= v["joint"] for k, v in b["classes"].items()
                ),
                sampled_background_lower=a["sampled_background"] < b["sampled_background"],
            )
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError(f"Pinned source changed: {path}")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    results=results,
                    criteria=criteria,
                    proceed_to_full_train_audit=all(criteria.values()),
                    no_component_val=True,
                    no_deployment=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = (
                "complete",
                "TRAIN-only comparison complete; no deployment or component VAL",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
