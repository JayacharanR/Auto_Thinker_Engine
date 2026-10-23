#!/usr/bin/env python3
"""TRAIN-junction selection of nonlinear RGB verification with paired augmentation."""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_background_sampling import append_fold_background  # noqa:E402
from src.navigation.signal_group_selection import macro_location_logloss  # noqa:E402
from src.navigation.signal_opponent_features import opponent_features  # noqa:E402
from src.navigation.signal_phase_supervision import (  # noqa:E402
    joint_candidates,
    phase_balanced_weights,
    phase_targets,
)
from src.navigation.signal_rgb_augmentation import GAMMAS, gamma_rgb  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

CACHE = ROOT / "outputs/navigation_signal_pretrained_20261021"
FOLDS = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
AUGMENTED = ROOT / "outputs/navigation_signal_rgb_boost_v2_20261021"


def read(path):
    return json.loads(Path(path).read_text())


def require_supported_final_audit(request):
    """Final supported fit is allowed only after its registered full TRAIN audit."""
    if not request.get("supported_full_audit"):
        raise ValueError("Supported final fit requires a passing full TRAIN audit")
    audit = Path(request["supported_full_audit"])
    result = read(audit / "assessment.json")
    if (
        digest(audit / "assessment.json") != request["supported_full_audit_sha256"]
        or read(audit / "progress.json")["status"] != "complete"
        or not result["train_audit_pass"]
        or result["exact_sampled_feature_and_probability_rows"] != 48865
        or not result["independent_sampled_recount_exact"]
        or not result["no_component_val"]
    ):
        raise ValueError("Completed matching full TRAIN audit required")


def verify(request):
    for path, sha in request["source_hashes"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Registered source changed: {path}")
    if digest(CACHE / "train_features.npz") != request["train_features_sha256"]:
        raise ValueError("Original TRAIN features changed")
    if digest(FOLDS) != request["folds_sha256"]:
        raise ValueError("Registered TRAIN folds changed")
    for path, sha in request.get("proposer_feature_hashes", {}).items():
        if digest(path) != sha:
            raise ValueError("Independently audited proposer cache changed")
    for path, sha in request.get("extra_background_hashes", {}).items():
        if digest(path) != sha:
            raise ValueError("Registered extra background cache changed")


def fuse_proposer_training(original, augmented_fit, learned, mask):
    """Pair each original/gamma RGB row with the same image-version CNN row."""
    if (
        original.ndim != 2
        or original.shape[1] != 87
        or learned.shape != (3, len(original), 36)
        or mask.shape != (len(original),)
        or mask.dtype != np.bool_
        or augmented_fit.shape != (3 * int(mask.sum()), 87)
    ):
        raise ValueError("Exact original/gamma RGB and proposer row alignment required")
    return (
        np.concatenate([augmented_fit, np.concatenate([v[mask] for v in learned])], axis=1),
        np.concatenate([original, learned[0]], axis=1),
    )


def worker(args):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    if args.arm not in (
        "plain",
        "gamma",
        "binary400",
        "joint100",
        "opponent100",
        "fused100",
    ) or args.fold not in (
        -1,
        0,
        1,
        2,
    ):
        raise ValueError("Only registered arms and folds")
    started = time.monotonic()
    root = args.out.parent
    request = read(root / "request.json")
    verify(request)
    joint = args.arm in ("joint100", "opponent100", "fused100")
    cache = read(root / "cache.json")
    with np.load(CACHE / "train_features.npz") as a:
        original, labels = a["rgb"].astype(np.float32), a["labels"]
    if joint:
        labels = phase_targets(
            read(CACHE / "train_rows.json"), read(CACHE / "dataset.json"), labels
        )
    with np.load(FOLDS) as a:
        fold = a["fold"]
    mask = np.ones(len(labels), bool) if args.fold == -1 else fold != args.fold
    if request.get("training_keep_path"):
        keep_path = Path(request["training_keep_path"])
        if digest(keep_path) != request["training_keep_sha256"]:
            raise ValueError("Registered TRAIN supervision mask changed")
        keep = np.load(keep_path)
        if keep.dtype != np.bool_ or keep.shape != mask.shape:
            raise ValueError("Exact supported-supervision eligibility array required")
        if args.fold == -1:
            require_supported_final_audit(request)
        mask &= keep
    x, y = original[mask], labels[mask]
    if args.arm in ("gamma", "binary400", "joint100", "opponent100", "fused100"):
        augmented_path = Path(cache.get("augmented_path", root / "augmented.npz"))
        if digest(augmented_path) != cache["augmented_sha256"]:
            raise ValueError("Registered augmented features changed")
        with np.load(augmented_path) as a:
            x = np.concatenate([x, *[v[mask].astype(np.float32) for v in a["features"]]])
        y = np.tile(y, 3)
    if args.arm == "opponent100":
        x, original = opponent_features(x), opponent_features(original)
    if args.arm == "fused100":
        feature_root = Path(request["proposer_features"])
        with np.load(feature_root / "train_proposer_features.npz") as a:
            learned = a["features"].astype(np.float32)
        x, original = fuse_proposer_training(original, x, learned, mask)
    extra_examples = 0
    if request.get("extra_background_cache"):
        if args.arm != "joint100" or args.fold not in (0, 1, 2):
            raise ValueError("Expanded backgrounds allow only registered internal joint100 folds")
        extra_root = Path(request["extra_background_cache"])
        if (
            read(extra_root / "progress.json")["status"] != "complete"
            or not read(extra_root / "assessment.json")["whole_junction_folds_verified"]
        ):
            raise ValueError("Complete verified background cache required")
        before = len(y)
        with np.load(extra_root / "features.npz") as extra:
            x, y = append_fold_background(x, y, extra["features"], extra["fold"], args.fold)
        extra_examples = len(y) - before
    if (args.out / "STOP").exists():
        raise InterruptedError("Owned fit STOP")
    with threadpool_limits(limits=2):
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=400 if args.arm == "binary400" else 100,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=10,
            class_weight=None if joint else "balanced",
            early_stopping=False,
            random_state=42,
            max_bins=255,
        ).fit(x, y, sample_weight=phase_balanced_weights(y) if joint else None)
        all_p = model.predict_proba(original)
        if joint and not np.array_equal(model.classes_, [0, 1, 2, 3]):
            raise ValueError("Joint class order changed")
        p = all_p[:, 1:].sum(1) if joint else all_p[:, 1]
        repeat_raw = (
            request.get("opponent_colors") or request.get("proposer_features")
        ) and args.arm == "joint100"
        if repeat_raw:
            name = "final" if args.fold == -1 else f"joint100_fold{args.fold}"
            prior = ROOT / "outputs/navigation_signal_phase_boost_20261021" / name
            if not np.array_equal(all_p, np.load(prior / "train_class_probability.npy")):
                raise ValueError("Repeated raw-feature control TRAIN predictions changed")
        blob = pickle.dumps(model, protocol=5)
        recovered = pickle.loads(blob)
        if not np.array_equal(recovered.predict_proba(original), all_p):
            raise ValueError("Serialized classifier parity failed")
        if args.fold == -1:
            if not (root / "selection.json").exists():
                raise ValueError("Final fit requires prior TRAIN-only selection")
            expected = read(CACHE / "cache.json")["validation_features_sha256"]
            if digest(CACHE / "validation_features.npz") != expected:
                raise ValueError("Original component VAL changed")
            with np.load(CACHE / "validation_features.npz") as a:
                val_x = a["rgb"].astype(np.float32)
                if args.arm == "opponent100":
                    val_x = opponent_features(val_x)
                if args.arm == "fused100":
                    with np.load(feature_root / "validation_proposer_features.npz") as features:
                        learned_val = features["features"].astype(np.float32)
                    if learned_val.shape != (len(val_x), 36):
                        raise ValueError("Original VAL proposer feature count changed")
                    val_x = np.concatenate([val_x, learned_val], axis=1)
                all_val_p = recovered.predict_proba(val_x)
                val_p = all_val_p[:, 1:].sum(1) if joint else all_val_p[:, 1]
            if repeat_raw:
                if not np.array_equal(
                    all_val_p, np.load(prior / "validation_class_probability.npy")
                ):
                    raise ValueError("Repeated raw-feature control VAL predictions changed")
            np.save(args.out / "validation_probability.npy", val_p)
            if joint:
                np.save(args.out / "validation_class_probability.npy", all_val_p)
    if (args.out / "STOP").exists():
        raise InterruptedError("Owned fit STOP after fitting")
    (args.out / "model.pkl").write_bytes(blob)
    np.save(args.out / "train_probability.npy", p)
    if joint:
        np.save(args.out / "train_class_probability.npy", all_p)
    atomic_json(
        args.out / "fit.json",
        dict(
            complete=True,
            arm=args.arm,
            excluded_fold=args.fold,
            iterations=model.n_iter_,
            trees=int(model.n_trees_per_iteration_ * model.n_iter_),
            original_train_examples=int(mask.sum()),
            fitted_examples=len(y),
            extra_background_examples=extra_examples,
            input_features=x.shape[1],
            component_validation_read=args.fold == -1,
            serialization_parity=True,
            sklearn_version=sklearn.__version__,
            model_sha256=digest(args.out / "model.pkl"),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def main(args):
    if bool(args.proposer_features) != bool(args.fusion_audit):
        raise ValueError("Both independently audited proposer cache and audit required")
    if args.proposer_features and args.opponent_colors:
        raise ValueError("Distinct registered experiments cannot be combined")
    if args.proposer_features:
        args.state_aware = True
    if args.opponent_colors:
        args.state_aware = True
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = time.time() + 1800
    if args.deadline_from:
        previous = read(args.deadline_from)
        if previous["status"] != "failed" or previous["fits_done"] or previous["frames"]:
            raise ValueError("Only an empty failed preflight can supply the original deadline")
        deadline = previous["updated_at"] - previous["elapsed_seconds"] + 1800
    status, error, phase, done, images_done = "running", None, "verify sources", 0, 0

    def check():
        if (out / "STOP").exists() or time.time() > deadline:
            raise InterruptedError("Owned STOP or 1800-second stage cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("100 MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Frozen proposer RGB fusion"
                if args.proposer_features
                else "Opponent RGB representation"
                if args.opponent_colors
                else "State-aware RGB supervision"
                if args.state_aware
                else "Nonlinear RGB and TRAIN augmentation",
                status=status,
                error=error,
                phase=phase,
                fits_done=done,
                fits_total=7,
                frames=images_done,
                target_frames=2844,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    def fit(name, arm, fold):
        nonlocal done, phase
        check()
        phase = f"{name}: {arm}, excluded TRAIN fold {fold}"
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
                    "--arm",
                    arm,
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
            launched = time.monotonic()
            try:
                while child.poll() is None:
                    check()
                    if time.monotonic() - launched > 180:
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
                raise RuntimeError(f"{name} exited {child.returncode}; no retry")
        receipt = read(directory / "fit.json")
        iterations = 400 if arm == "binary400" else 100
        if receipt["iterations"] != iterations or not receipt["serialization_parity"]:
            raise ValueError("Registered fixed iteration budget/parity not met")
        if arm in ("binary400", "joint100", "opponent100", "fused100") and receipt["trees"] != 400:
            raise ValueError("Matched tree count not met")
        done += 1
        progress()
        return np.load(directory / "train_probability.npy")

    try:
        with BudgetWatch(out):
            from src.navigation.signal_context import context_patches
            from src.navigation.signal_diversity_data import DiverseSignalDataset
            from src.navigation.signal_pretrained_verifier import rgb_features

            files = [
                Path(__file__),
                ROOT / "src/navigation/signal_rgb_augmentation.py",
                ROOT / "src/navigation/signal_context.py",
                ROOT / "src/navigation/signal_diversity_data.py",
                ROOT / "src/navigation/signal_pretrained_verifier.py",
                ROOT / "src/navigation/signal_group_selection.py",
                ROOT / "src/navigation/signal_pixel_study.py",
                ROOT / "src/navigation/signal_learning.py",
                ROOT / "src/navigation/signal_hard_pixels.py",
                ROOT / "scripts/train_navigation_signal_context.py",
                ROOT / "reports/navigation_signal_rgb_boost_protocol_20261021.md",
                CACHE / "train_rows.json",
                CACHE / "dataset.json",
                CACHE / "cache.json",
            ]
            if args.deadline_from:
                files.append(args.deadline_from.resolve())
            if args.state_aware:
                files.extend(
                    [
                        ROOT / "src/navigation/signal_phase_supervision.py",
                        ROOT / "reports/navigation_signal_phase_boost_protocol_20261021.md",
                        AUGMENTED / "cache.json",
                        AUGMENTED / "assessment.json",
                    ]
                )
            if args.opponent_colors:
                files.extend(
                    [
                        ROOT / "src/navigation/signal_opponent_features.py",
                        ROOT / "reports/navigation_signal_opponent_protocol_20261021.md",
                        ROOT / "outputs/navigation_signal_phase_boost_20261021/assessment.json",
                    ]
                )
            if args.opponent_colors or args.proposer_features:
                prior_root = ROOT / "outputs/navigation_signal_phase_boost_20261021"
                for name in ("joint100_fold0", "joint100_fold1", "joint100_fold2", "final"):
                    files.append(prior_root / name / "train_class_probability.npy")
                files.append(prior_root / "final/validation_class_probability.npy")
            proposer_hashes = {}
            if args.proposer_features:
                feature_root, audit_root = (
                    args.proposer_features.resolve(),
                    args.fusion_audit.resolve(),
                )
                feature_result = read(feature_root / "assessment.json")
                audit_request, audit_result = (
                    read(audit_root / "request.json"),
                    read(audit_root / "assessment.json"),
                )
                if (
                    read(feature_root / "progress.json")["status"] != "complete"
                    or read(audit_root / "progress.json")["status"] != "complete"
                    or not audit_result["complete"]
                    or audit_result["independent_exact_forwards"] != 67
                    or not audit_result["all_array_shapes_ranges_hashes_verified"]
                    or audit_request["cache"] != str(feature_root)
                    or audit_request["cache_assessment_sha256"]
                    != digest(feature_root / "assessment.json")
                    or audit_request["cache_request_sha256"]
                    != digest(feature_root / "request.json")
                ):
                    raise ValueError("Complete independent cache audit required before fitting")
                feature_request = read(feature_root / "request.json")
                for item in ("train_rows.json", "dataset.json", "cache.json"):
                    key = str((CACHE / item).relative_to(ROOT))
                    if feature_request["source_hashes"][key] != digest(CACHE / item):
                        raise ValueError(
                            "Proposer cache does not correspond to current original RGB rows"
                        )
                for file, key in (
                    ("train_proposer_features.npz", "train_sha256"),
                    ("validation_proposer_features.npz", "validation_sha256"),
                ):
                    proposer_hashes[str(feature_root / file)] = feature_result[key]
                files.extend(
                    [
                        ROOT / "reports/navigation_signal_proposer_fusion_protocol_20261021.md",
                        feature_root / "request.json",
                        feature_root / "assessment.json",
                        audit_root / "request.json",
                        audit_root / "assessment.json",
                    ]
                )
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            request = dict(
                source_hashes=hashes,
                folds_sha256=digest(FOLDS),
                train_features_sha256=read(CACHE / "cache.json")["train_features_sha256"],
                fit_seconds=180,
                stage_seconds=1800,
                artifact_bytes=100_000_000,
                no_native=True,
                gammas=GAMMAS,
                original_deadline=deadline,
                state_aware=args.state_aware,
                opponent_colors=args.opponent_colors,
                proposer_features=str(args.proposer_features.resolve())
                if args.proposer_features
                else None,
                proposer_feature_hashes=proposer_hashes,
            )
            if request["folds_sha256"] != read(FOLDS.parent / "request.json")["folds_sha256"]:
                raise ValueError("Prior registered TRAIN fold identity changed")
            atomic_json(out / "request.json", request)
            verify(request)
            if args.state_aware:
                previous_cache = read(AUGMENTED / "cache.json")
                path = AUGMENTED / "augmented.npz"
                if digest(path) != previous_cache["augmented_sha256"]:
                    raise ValueError("Existing verified augmentation cache changed")
                with np.load(CACHE / "train_features.npz") as a:
                    labels = a["labels"]
                atomic_json(
                    out / "cache.json",
                    dict(
                        augmented_path=str(path),
                        augmented_sha256=digest(path),
                        original_rgb_exact=previous_cache["original_rgb_exact"],
                        train_points=previous_cache["train_points"],
                        no_feature_copy=True,
                    ),
                )
                images_done = 2844
                phase = "Existing augmented RGB features verified"
                progress()
            else:
                phase = "TRAIN-only gamma features; exact original RGB parity"
                progress()
                data = DiverseSignalDataset(
                    ROOT,
                    ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                    ROOT / "outputs/navigation_signal_diversity_audit_20261021",
                )
                rows = read(CACHE / "train_rows.json")
                with np.load(CACHE / "train_features.npz") as a:
                    original, labels = a["rgb"], a["labels"]
                augmented = np.empty((2, len(rows), 87), np.float16)
                grouped = defaultdict(list)
                for j, row in enumerate(rows):
                    if row["sample"] not in data.train or row["role"] != "train":
                        raise ValueError("Non-TRAIN point entered augmentation")
                    grouped[row["sample"]].append(j)
                if len(rows) != 51094 or not np.array_equal(labels, [r["label"] for r in rows]):
                    raise ValueError("Original TRAIN points/labels changed")
                for index in data.train:
                    check()
                    indices = grouped[index]
                    points = np.array([[rows[j]["x"], rows[j]["y"]] for j in indices])
                    image = data.images[index].transpose(1, 2, 0)
                    if indices:
                        control = rgb_features(context_patches(image, points)).astype(np.float16)
                        if not np.array_equal(control, original[indices]):
                            raise ValueError("Original RGB features not reproduced")
                        for gi, gamma in enumerate(GAMMAS):
                            transformed = gamma_rgb(image, gamma)
                            augmented[gi, indices] = rgb_features(
                                context_patches(transformed, points)
                            )
                    images_done += 1
                    if images_done % 30 == 0:
                        progress()
                del data
                np.savez_compressed(out / "augmented.npz", features=augmented)
                del augmented
                atomic_json(
                    out / "cache.json",
                    dict(
                        augmented_sha256=digest(out / "augmented.npz"),
                        original_rgb_exact=True,
                        train_points=len(rows),
                    ),
                )
            with np.load(FOLDS) as a:
                fold, groups, predicted = a["fold"], a["groups"], a["predicted"]
            candidates = []
            arms = (
                ("joint100", "fused100")
                if args.proposer_features
                else ("joint100", "opponent100")
                if args.opponent_colors
                else ("binary400", "joint100")
                if args.state_aware
                else ("plain", "gamma")
            )
            for arm in arms:
                oof = np.full(len(labels), np.nan)
                for fi in range(3):
                    p = fit(f"{arm}_fold{fi}", arm, fi)
                    oof[fold == fi] = p[fold == fi]
                if not np.isfinite(oof).all():
                    raise ValueError("Incomplete OOF predictions")
                np.save(out / f"oof_{arm}.npy", oof)
                loss, location_losses = macro_location_logloss(oof, labels, groups, predicted)
                candidates.append(dict(arm=arm, macro_logloss=loss, by_location=location_losses))
                atomic_json(out / "candidates.json", candidates)
            selected = min(candidates, key=lambda r: (r["macro_logloss"], r["arm"] != arms[0]))
            atomic_json(
                out / "selection.json",
                dict(
                    selected=selected,
                    candidates=candidates,
                    selected_at=time.time(),
                    validation_used=False,
                ),
            )
            fit("final", selected["arm"], -1)
            from scripts.train_navigation_signal_context import score
            from src.navigation.signal_hard_pixels import protected_signal_metrics
            from src.navigation.signal_pixel_study import NativeSignalDataset
            from src.navigation.signal_pretrained_verifier import retain_original_states

            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            source = (
                ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
            )
            peaks = {r["sample"]: r["peaks"] for r in read(source)}
            baseline = score(peaks, data)
            offsets = read(CACHE / "cache.json")["validation_offsets"]
            p = np.load(out / "final/validation_probability.npy")
            joint_p = (
                np.load(out / "final/validation_class_probability.npy")
                if selected["arm"] in ("joint100", "opponent100", "fused100")
                else None
            )
            metrics = score(
                {
                    i: joint_candidates(peaks[i], joint_p[offsets[n] : offsets[n + 1]])
                    if joint_p is not None
                    else retain_original_states(peaks[i], p[offsets[n] : offsets[n + 1]])
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
                    selected_arm=selected["arm"],
                    metrics=metrics,
                    baseline=baseline,
                    criteria=criteria,
                    component_pass=all(criteria.values()),
                    driving_qualified=False,
                    no_validation_selection=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "TRAIN-selected nonlinear RGB head assessed; no deployment"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-worker", action="store_true")
    parser.add_argument(
        "--arm", choices=("plain", "gamma", "binary400", "joint100", "opponent100", "fused100")
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--deadline-from", type=Path)
    parser.add_argument("--state-aware", action="store_true")
    parser.add_argument("--opponent-colors", action="store_true")
    parser.add_argument("--proposer-features", type=Path)
    parser.add_argument("--fusion-audit", type=Path)
    args = parser.parse_args()
    worker(args) if args.fit_worker else main(args)
