#!/usr/bin/env python3
"""One preregistered supported-supervision fit and unchanged component evaluation."""

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
from src.navigation.signal_hard_pixels import protected_signal_metrics  # noqa: E402
from src.navigation.signal_phase_supervision import joint_candidates  # noqa: E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, phase, error, done = "running", "verify passing TRAIN audit", None, 0
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    study = ROOT / "outputs/navigation_signal_supported_20261022"
    audit = ROOT / "outputs/navigation_signal_supported_full_20261022"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    aug = ROOT / "outputs/navigation_signal_rgb_boost_v2_20261021"
    worker = ROOT / "scripts/train_navigation_signal_rgb_boost.py"

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Owned STOP or900second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("100MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Supported signal component evaluation",
                status=status,
                phase=phase,
                error=error,
                fits_done=done,
                fits_total=1,
                elapsed_seconds=time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            audited = read(audit / "assessment.json")
            if (
                read(audit / "progress.json")["status"] != "complete"
                or not audited["train_audit_pass"]
                or audited["exact_sampled_feature_and_probability_rows"] != 48865
                or not audited["independent_sampled_recount_exact"]
            ):
                raise ValueError("Passing complete full TRAIN audit required")
            previous = read(study / "request.json")
            mask = Path(previous["training_keep_path"])
            if digest(mask) != previous["training_keep_sha256"] or np.load(mask).sum() != 49088:
                raise ValueError("Original support mask changed")
            files = [
                Path(__file__),
                worker,
                mask,
                ROOT / "scripts/train_navigation_signal_context.py",
                ROOT / "reports/navigation_signal_supported_final_protocol_20261022.md",
                audit / "request.json",
                audit / "assessment.json",
                study / "request.json",
                study / "assessment.json",
                cache / "train_rows.json",
                cache / "dataset.json",
                cache / "cache.json",
                aug / "cache.json",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            request = dict(
                source_hashes=hashes,
                training_keep_path=str(mask),
                training_keep_sha256=previous["training_keep_sha256"],
                train_features_sha256=previous["train_features_sha256"],
                folds_sha256=previous["folds_sha256"],
                supported_full_audit=str(audit),
                supported_full_audit_sha256=digest(audit / "assessment.json"),
                registered_protocol_sha256=digest(
                    ROOT / "reports/navigation_signal_supported_final_protocol_20261022.md"
                ),
                created_at=time.time(),
                child_seconds=180,
                stage_seconds=900,
                artifact_bytes=100_000_000,
                no_driving=True,
            )
            atomic_json(out / "request.json", request)
            aug_hash = read(aug / "cache.json")["augmented_sha256"]
            if digest(aug / "augmented.npz") != aug_hash:
                raise ValueError("Original gamma features changed")
            atomic_json(
                out / "cache.json",
                dict(
                    augmented_path=str(aug / "augmented.npz"),
                    augmented_sha256=aug_hash,
                ),
            )
            atomic_json(
                out / "selection.json",
                dict(
                    selected="supported_joint100",
                    validation_used=False,
                    selected_at=time.time(),
                    full_train_audit_sha256=digest(audit / "assessment.json"),
                    criteria=audited["criteria"],
                    no_hyperparameter_sweep=True,
                ),
            )
            target = out / "final"
            target.mkdir()
            phase = "One registered all-TRAIN supported fit"
            progress()
            with (target / "worker.log").open("xb") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(worker),
                        "--fit-worker",
                        "--out",
                        str(target),
                        "--arm",
                        "joint100",
                        "--fold",
                        "-1",
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
                    target / "worker_identity.json",
                    dict(pid=child.pid, command=child.args, started_at=time.time()),
                )
                launched = time.monotonic()
                try:
                    while child.poll() is None:
                        check()
                        if time.monotonic() - launched > 180:
                            raise InterruptedError("180second fit cap")
                        progress()
                        time.sleep(0.25)
                finally:
                    if child.poll() is None:
                        (target / "STOP").touch()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.terminate()
                            child.wait(timeout=20)
                            atomic_json(
                                target / "normal_cleanup.json",
                                dict(STOP_present=True, signal="SIGTERM", force_kill=False),
                            )
                if child.returncode:
                    raise RuntimeError(f"Final fit failed{child.returncode}; no retry")
            fit = read(target / "fit.json")
            if (
                fit["fitted_examples"] != 147264
                or fit["trees"] != 400
                or not fit["serialization_parity"]
                or not fit["component_validation_read"]
            ):
                raise ValueError("Registered final fit contract mismatch")
            done = 1
            phase = "Unchanged72housing component score"
            progress()
            from scripts.train_navigation_signal_context import score

            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            parent = (
                ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
            )
            peaks = {r["sample"]: r["peaks"] for r in read(parent)}
            original = score(peaks, data)
            offsets = read(cache / "cache.json")["validation_offsets"]
            p = np.load(target / "validation_class_probability.npy")
            if p.shape != (38543, 4):
                raise ValueError("Original component proposal denominator changed")
            predictions = {
                i: joint_candidates(peaks[i], p[offsets[n] : offsets[n + 1]])
                for n, i in enumerate(data.validation)
            }
            result = score(predictions, data)
            criteria = dict(
                coverage_protected=protected_signal_metrics(result, original),
                background_reduced90=result["background_peaks"]
                <= 0.1 * original["background_peaks"],
            )
            atomic_json(out / "predictions.json", predictions)
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Pinned source changed")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    metrics=result,
                    baseline=original,
                    criteria=criteria,
                    component_pass=all(criteria.values()),
                    driving_qualified=False,
                    repeatedly_inspected_development_set=True,
                    no_validation_selection=True,
                    model_sha256=fit["model_sha256"],
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "Original component assessed once; no runtime deployment"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
