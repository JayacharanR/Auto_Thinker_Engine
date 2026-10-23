#!/usr/bin/env python3
"""Select one penalty using TRAIN locations; score component VAL only after selection."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_group_selection import (  # noqa:E402
    location_folds,
    macro_location_logloss,
)
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

CANDIDATES = (0.0001, 0.001, 0.01, 0.1, 1.0)
CACHE = ROOT / "outputs/navigation_signal_pretrained_20261021"


def read(path):
    return json.loads(Path(path).read_text())


def verify(request):
    for path, sha in request["source_hashes"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Registered source changed: {path}")
    if digest(CACHE / "train_features.npz") != request["train_features_sha256"]:
        raise ValueError("Original TRAIN feature cache changed")


def predict(x, head):
    logit = ((x - head["mean"]) / head["scale"]) @ head["coefficient"] + head["intercept"][0]
    return 1 / (1 + np.exp(-np.clip(logit, -700, 700)))


def worker(args):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    if args.penalty not in CANDIDATES or args.fold not in (-1, 0, 1, 2):
        raise ValueError("Only registered penalties and folds are allowed")
    request = read(args.out.parent / "request.json")
    verify(request)
    if digest(args.out.parent / "folds.npz") != request["folds_sha256"]:
        raise ValueError("Internal fold membership changed")
    with np.load(CACHE / "train_features.npz") as arrays:
        x, y = arrays["resnet"].astype(np.float64), arrays["labels"]
    with np.load(args.out.parent / "folds.npz") as arrays:
        fold = arrays["fold"]
    mask = np.ones(len(y), bool) if args.fold == -1 else fold != args.fold
    if (args.out / "STOP").exists():
        raise InterruptedError("Owned fit STOP")
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = StandardScaler().fit(x[mask])
        model = LogisticRegression(
            C=args.penalty,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            tol=1e-4,
            random_state=42,
        ).fit(scaler.transform(x[mask]), y[mask])
        p = model.predict_proba(scaler.transform(x))[:, 1]
        head = dict(
            mean=scaler.mean_,
            scale=scaler.scale_,
            coefficient=model.coef_[0],
            intercept=model.intercept_,
        )
        if not np.allclose(predict(x, head), p, rtol=0, atol=1e-12):
            raise ValueError("Exported head parity failed")
    if (args.out / "STOP").exists():
        raise InterruptedError("Owned fit STOP after fitting")
    np.savez_compressed(args.out / "head.npz", **head, train_probability=p)
    atomic_json(
        args.out / "fit.json",
        dict(
            complete=True,
            converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            warnings=[str(w.message) for w in caught],
            iterations=model.n_iter_.tolist(),
            C=args.penalty,
            excluded_fold=args.fold,
            train_examples=int(mask.sum()),
            no_component_validation_read=True,
            exported_parity=True,
            head_sha256=digest(args.out / "head.npz"),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def main(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, error, phase, done = "running", None, "verify cached TRAIN identities", 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1800:
            raise InterruptedError("Owned STOP or 1800-second stage cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 60_000_000:
            raise InterruptedError("60 MB stage cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="TRAIN-junction regularization",
                status=status,
                error=error,
                phase=phase,
                fits_done=done,
                fits_total=16,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    def fit(name, penalty, fold):
        nonlocal done, phase
        check()
        phase = f"{name}: TRAIN-only fit C={penalty}, excluded fold={fold}"
        directory = out / name
        directory.mkdir()
        with (directory / "worker.log").open("xb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--fit-worker",
                    "--out",
                    str(directory),
                    "--penalty",
                    str(penalty),
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
                directory / "worker_identity.json",
                dict(
                    pid=child.pid,
                    command=child.args,
                    started_at=time.time(),
                ),
            )
            fit_started = time.monotonic()
            try:
                while child.poll() is None:
                    check()
                    if time.monotonic() - fit_started > 180:
                        raise InterruptedError("180-second fit cap")
                    progress()
                    time.sleep(0.25)
            finally:
                if child.poll() is None:
                    (directory / "STOP").touch()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        child.wait(timeout=20)
                        atomic_json(
                            directory / "normal_cleanup.json",
                            dict(STOP_present=True, signal="SIGTERM", force_kill=False),
                        )
            if child.returncode != 0:
                raise RuntimeError(f"Fit {name} exited {child.returncode}; no retry")
        receipt = read(directory / "fit.json")
        if not receipt["converged"]:
            raise ValueError("Nonconverged fold: do not select from incomplete candidates")
        with np.load(directory / "head.npz") as arrays:
            head = dict(arrays)
        done += 1
        progress()
        return head

    try:
        with BudgetWatch(out):
            progress()
            cache = read(CACHE / "cache.json")
            if digest(CACHE / "train_features.npz") != cache["train_features_sha256"]:
                raise ValueError("Unverified TRAIN features")
            rows, dataset = read(CACHE / "train_rows.json"), read(CACHE / "dataset.json")
            fold, groups, assignment = location_folds(rows, dataset)
            proposals = read(CACHE / "train_proposals.json")
            points = {int(i): {(p["x"], p["y"]) for p in ps} for i, ps in proposals.items()}
            predicted = np.array([(r["x"], r["y"]) in points[r["sample"]] for r in rows])
            with np.load(CACHE / "train_features.npz") as arrays:
                labels = arrays["labels"]
            if len(assignment) != 21 or len(rows) != 51094:
                raise ValueError("Original TRAIN denominator changed")
            if not np.array_equal(labels, [r["label"] for r in rows]):
                raise ValueError("Original TRAIN labels changed")
            macro_location_logloss(np.full(len(labels), 0.5), labels, groups, predicted)
            np.savez_compressed(out / "folds.npz", fold=fold, groups=groups, predicted=predicted)
            atomic_json(
                out / "folds.json",
                dict(assignment=assignment, seed=42, predicted_points=int(predicted.sum())),
            )
            files = [
                Path(__file__),
                ROOT / "src/navigation/signal_group_selection.py",
                ROOT / "reports/navigation_signal_regularization_protocol_20261021.md",
                ROOT / "scripts/train_navigation_signal_context.py",
                ROOT / "src/navigation/signal_pretrained_verifier.py",
                ROOT / "src/navigation/signal_pixel_study.py",
                ROOT / "src/navigation/signal_learning.py",
                ROOT / "src/navigation/signal_hard_pixels.py",
                CACHE / "cache.json",
                CACHE / "train_rows.json",
                CACHE / "dataset.json",
            ]
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for file in files:
                target = out / "source" / file.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
            request = dict(
                source_hashes=hashes,
                train_features_sha256=cache["train_features_sha256"],
                folds_sha256=digest(out / "folds.npz"),
                candidates=CANDIDATES,
                seconds_per_fit=180,
                stage_seconds=1800,
                artifact_bytes=60_000_000,
                no_native=True,
                selection_uses_component_validation=False,
            )
            atomic_json(out / "request.json", request)
            candidates = []
            for ci, penalty in enumerate(CANDIDATES):
                oof = np.full(len(labels), np.nan)
                for fi in range(3):
                    head = fit(f"c{ci}_fold{fi}", penalty, fi)
                    oof[fold == fi] = head["train_probability"][fold == fi]
                if not np.isfinite(oof).all():
                    raise ValueError("Incomplete OOF coverage")
                np.save(out / f"oof_c{ci}.npy", oof)
                loss, locations = macro_location_logloss(oof, labels, groups, predicted)
                candidates.append(dict(C=penalty, macro_logloss=loss, by_location=locations))
                atomic_json(out / "candidates.json", candidates)
            selected = min(candidates, key=lambda row: (row["macro_logloss"], row["C"]))
            atomic_json(
                out / "selection.json",
                dict(
                    selected=selected,
                    candidates=candidates,
                    component_validation_used=False,
                    selected_at=time.time(),
                    selector="macro TRAIN-location predicted-point logloss",
                ),
            )
            head = fit("final", selected["C"], -1)
            # Original component VAL is first loaded only after TRAIN selection and final fit.
            phase = "selected head: fixed component validation"
            progress()
            if digest(CACHE / "validation_features.npz") != cache["validation_features_sha256"]:
                raise ValueError("Original component VAL cache changed")
            with np.load(CACHE / "validation_features.npz") as arrays:
                p = predict(arrays["resnet"].astype(np.float64), head)
                offsets = arrays["offsets"]
            np.save(out / "validation_probability.npy", p)
            if selected["C"] == 1 and not np.array_equal(
                p, np.load(CACHE / "resnet/validation_probability.npy")
            ):
                raise ValueError("Repeated C=1 final probabilities changed")
            from scripts.train_navigation_signal_context import score
            from src.navigation.signal_hard_pixels import protected_signal_metrics
            from src.navigation.signal_pixel_study import NativeSignalDataset
            from src.navigation.signal_pretrained_verifier import retain_original_states

            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            source = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb"
            peaks = {r["sample"]: r["peaks"] for r in read(source / "predictions1000.json")}
            baseline = score(peaks, data)
            metrics = score(
                {
                    i: retain_original_states(peaks[i], p[offsets[n] : offsets[n + 1]])
                    for n, i in enumerate(data.validation)
                },
                data,
            )
            criteria = dict(
                coverage_protected=protected_signal_metrics(metrics, baseline),
                background_reduced90=metrics["background_peaks"]
                <= 0.1 * baseline["background_peaks"],
            )
            verify(request)
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    selected_C=selected["C"],
                    metrics=metrics,
                    baseline=baseline,
                    criteria=criteria,
                    component_pass=all(criteria.values()),
                    driving_qualified=False,
                    no_component_validation_selection=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "TRAIN-selected head assessed; no driving promotion"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-worker", action="store_true")
    parser.add_argument("--penalty", type=float)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    worker(args) if args.fit_worker else main(args)
