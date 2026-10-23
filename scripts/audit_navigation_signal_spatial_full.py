#!/usr/bin/env python3
"""Full frozen-proposal audit of spatial RGB TRAIN heads; no fitting."""

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_signal_supported_full import add_counts, count_image  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_context import context_patches  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_pretrained_verifier import rgb_features  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    study = ROOT / "outputs/navigation_signal_spatial_fit_20261022"
    extra_root = ROOT / "outputs/navigation_signal_background_cache_20261022"
    spatial_root = ROOT / "outputs/navigation_signal_spatial_cache_20261022"
    full = ROOT / "outputs/navigation_signal_plateau_train_20261022"
    inventory = ROOT / "outputs/navigation_signal_plateau_inventory_20261022"
    raw = ROOT / "outputs/navigation_signal_supported_full_20261022"
    folds_path = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
    started = time.monotonic()
    status, phase, error, images_done = "running", "verify sources", None, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 1200:
            raise InterruptedError("Owned STOP or1200second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 600_000_000:
            raise InterruptedError("600MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Spatial RGB full TRAIN audit",
                status=status,
                phase=phase,
                error=error,
                frames=images_done,
                target_frames=2844,
                elapsed_seconds=time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    try:
        progress()
        with BudgetWatch(out), threadpool_limits(limits=2):
            completed = read(full / "assessment.json")
            old_request = read(study / "request.json")
            if read(full / "progress.json")["status"] != "complete" or not completed["complete"]:
                raise ValueError("Completed full TRAIN verification required")
            if (
                read(study / "progress.json")["status"] != "complete"
                or not read(study / "assessment.json")["no_candidate_qualification"]
                or read(spatial_root / "assessment.json")["exact_direct_pixel_rows"] != 15417
            ):
                raise ValueError(
                    "All three completed fits and independent cache verification required"
                )
            models = []
            files = [
                Path(__file__),
                ROOT / "scripts/audit_navigation_signal_supported_full.py",
                ROOT / "reports/navigation_signal_spatial_full_protocol_20261022.md",
                study / "assessment.json",
                spatial_root / "assessment.json",
                spatial_root / "request.json",
                ROOT / "patches/navigation_signal_context_fast_candidate.py",
                ROOT / "patches/navigation_signal_context_fused_candidate.py",
                ROOT / "patches/navigation_signal_spatial_context_candidate.py",
                extra_root / "assessment.json",
                extra_root / "rows.json",
                inventory / "assessment.json",
                inventory / "request.json",
                raw / "assessment.json",
                full / "assessment.json",
                full / "offsets.npz",
                folds_path,
                cache / "train_rows.json",
                cache / "train_proposals.json",
                cache / "dataset.json",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            for fold in range(3):
                directory = study / f"fold{fold}"
                model_file = directory / "model.pkl"
                receipt = read(directory / "fit.json")
                if (
                    digest(model_file) != receipt["model_sha256"]
                    or receipt["excluded_fold"] != fold
                ):
                    raise ValueError("Exact original held-out-junction model required")
                files += [model_file, directory / "fit.json"]
                models.append(pickle.loads(model_file.read_bytes()))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                if path.suffix in (".py", ".md", ".pkl"):
                    target = out / "source" / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            if digest(full / "probability.npy") != completed["artifact_sha256"]["probability.npy"]:
                raise ValueError("Original full TRAIN probabilities changed")
            if (
                digest(cache / "train_features.npz")
                != read(cache / "cache.json")["train_features_sha256"]
            ):
                raise ValueError("Original sampled RGB features changed")
            if (
                digest(folds_path)
                != old_request["source_hashes"][str(folds_path.relative_to(ROOT))]
            ):
                raise ValueError("Original TRAIN folds changed")
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    no_new_fits=True,
                    no_component_val=True,
                    stage_seconds=1200,
                    artifact_cap_bytes=600_000_000,
                    baseline_probability_sha256=digest(full / "probability.npy"),
                ),
            )
            with np.load(folds_path) as a:
                folds, predicted = a["fold"], a["predicted"]
            with np.load(full / "offsets.npz") as a:
                old_offsets, old_indices = a["offsets"], a["samples"]
            with np.load(cache / "train_features.npz") as a:
                original_features = a["rgb"]
            original_p = np.load(full / "probability.npy", mmap_mode="r")
            sampled_candidate_p = np.load(study / "oof.npy")
            if digest(study / "oof.npy") != read(study / "assessment.json")["oof_sha256"]:
                raise ValueError("OOF probability cache changed")
            spatial_result = read(spatial_root / "assessment.json")
            for name, sha in spatial_result["artifacts"].items():
                if digest(spatial_root / name) != sha:
                    raise ValueError("Spatial cache changed")
            spatial_features = np.load(spatial_root / "features.npy", mmap_mode="r")

            def staged(name):
                spec = importlib.util.spec_from_file_location(
                    name, ROOT / "patches" / (name + ".py")
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

            prepare = staged("navigation_signal_context_fast_candidate").PreparedContext
            fused = staged("navigation_signal_context_fused_candidate").fused_features
            spatial = staged("navigation_signal_spatial_context_candidate").spatial_context
            fused(prepare(np.zeros((1, 1, 3), np.uint8)), np.array([[0, 0]], np.int64))
            extra_result = read(extra_root / "assessment.json")
            for name, sha in extra_result["artifacts"].items():
                if digest(extra_root / name) != sha:
                    raise ValueError("Original expanded-background feature cache changed")
            with np.load(extra_root / "features.npz") as extra:
                extra_features = extra["features"][0]
            extra_rows = read(extra_root / "rows.json")
            extra_group = defaultdict(list)
            for j, row in enumerate(extra_rows):
                extra_group[row["sample"]].append(j)
            exact_extra = 0
            rows = read(cache / "train_rows.json")
            previous_points = np.load(full / "points.npy", mmap_mode="r")
            if digest(full / "points.npy") != completed["artifact_sha256"]["points.npy"]:
                raise ValueError("Original refinement coordinates changed")
            by_sample = defaultdict(list)
            for j, row in enumerate(rows):
                if row["role"] != "train":
                    raise ValueError("TRAIN only")
                by_sample[row["sample"]].append(j)
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            identity = read(cache / "dataset.json")
            if (
                not np.array_equal(old_indices, data.train)
                or data.train != identity["train"]
                or json.dumps(data.samples, sort_keys=True)
                != json.dumps(identity["samples"], sort_keys=True)
            ):
                raise ValueError("Original ordered TRAIN identity changed")
            inventoried = read(inventory / "assessment.json")
            if (
                read(inventory / "progress.json")["status"] != "complete"
                or not inventoried["proceed_to_registered_inference"]
                or inventoried["union_points"] != 12362313
            ):
                raise ValueError("Exact completed plateau inventory required")
            for name, sha in inventoried["artifact_sha256"].items():
                if digest(inventory / name) != sha:
                    raise ValueError("Registered inventory artifact changed")
            with np.load(inventory / "offsets.npz") as a:
                offsets, inventory_samples = a["offsets"], a["samples"]
            if not np.array_equal(inventory_samples, data.train):
                raise ValueError("Original TRAIN inventory ordering changed")
            candidates = np.load(inventory / "points.npy", mmap_mode="r")
            phase = "Classify all registered plateau candidates"
            progress()
            output_p = np.lib.format.open_memmap(
                out / "probability.npy", mode="w+", dtype=np.float64, shape=(offsets[-1], 4)
            )
            output_xy = np.lib.format.open_memmap(
                out / "points.npy", mode="w+", dtype=np.uint16, shape=(offsets[-1], 2)
            )
            total, baseline_total = dict(towns={}), dict(towns={})
            image_receipts, exact_rows, added_points, times = [], 0, 0, []
            for n, i in enumerate(data.train):
                check()
                rgb, points, peaks = (
                    data.images[i].transpose(1, 2, 0),
                    candidates[offsets[n] : offsets[n + 1]],
                    [
                        dict(x=int(x), y=int(y))
                        for x, y in previous_points[old_offsets[n] : old_offsets[n + 1]]
                    ],
                )
                old = original_p[old_offsets[n] : old_offsets[n + 1]]
                old_lookup = {(p["x"], p["y"]): j for j, p in enumerate(peaks)}
                lookup = {tuple(p): j for j, p in enumerate(points)}
                if not set(old_lookup) <= set(lookup):
                    raise ValueError("Every original proposal must remain")
                row_ids = by_sample[i]
                fi = {int(folds[j]) for j in row_ids}
                if len(fi) != 1:
                    raise ValueError("One held-out fold per TRAIN image required")
                fold = fi.pop()
                tick = time.perf_counter()
                features = np.concatenate(
                    [fused(prepare(rgb), points), spatial(rgb, points)], axis=1
                ).astype(np.float16)
                p = models[fold].predict_proba(features.astype(np.float32))
                times.append(time.perf_counter() - tick)
                new_ids = []  # Same frozen candidate coordinates; only the trained head changes.
                extra_ids = extra_group[i]
                extra_point_ids = [
                    lookup[(extra_rows[j]["x"], extra_rows[j]["y"])] for j in extra_ids
                ]
                if not np.array_equal(features[extra_point_ids, :87], extra_features[extra_ids]):
                    raise ValueError("Extra TRAIN original-version feature parity failed")
                if not np.array_equal(
                    features[extra_point_ids, 87:], spatial_features[0, 51094 + np.array(extra_ids)]
                ):
                    raise ValueError("Spatial extra feature parity failed")
                if not np.array_equal(
                    p[extra_point_ids], sampled_candidate_p[51094 + np.array(extra_ids)]
                ):
                    raise ValueError("Spatial extra OOF probability parity failed")
                exact_extra += len(extra_ids)
                # Existing sampled coordinates are recomputed independently, not just copied.
                check_rows = [j for j in row_ids if predicted[j]]
                check_xy = np.array(
                    [[rows[j]["x"], rows[j]["y"]] for j in check_rows], np.int64
                ).reshape(-1, 2)
                check_features = features[[lookup[tuple(point)] for point in check_xy]]
                if not np.array_equal(check_features[:, :87], original_features[check_rows]):
                    raise ValueError("Original sampled RGB features changed")
                if not np.array_equal(check_features[:, 87:], spatial_features[0, check_rows]):
                    raise ValueError("Spatial sampled feature parity failed")
                # Original CPU implementation remains the independent raw-feature reference.
                reference = rgb_features(context_patches(rgb, check_xy)).astype(np.float16)
                if not np.array_equal(reference, check_features[:, :87]):
                    raise ValueError("Fast raw extractor differs from original sampled reference")
                check_p = (
                    models[fold].predict_proba(check_features.astype(np.float32))
                    if len(check_rows)
                    else np.empty((0, 4))
                )
                if not np.array_equal(check_p, p[[lookup[tuple(xy)] for xy in check_xy]]):
                    raise ValueError("Original sampled OOF predictions changed")
                if not np.array_equal(check_p, sampled_candidate_p[check_rows]):
                    raise ValueError("New model does not reproduce its original-row OOF cache")
                if not np.isfinite(p).all():
                    raise ValueError("Every old/new coordinate needs a prediction")
                exact_rows += len(check_rows)
                added_points += len(new_ids)
                output_p[offsets[n] : offsets[n + 1]] = p
                output_xy[offsets[n] : offsets[n + 1]] = points
                all_peaks = [dict(x=int(x), y=int(y)) for x, y in points]
                counted = count_image(all_peaks, p, data.boxes[i], data.background[i])
                before = count_image(peaks, old, data.boxes[i], data.background[i])
                town = data.samples[i]["town"]
                for result, count in ((total, counted), (baseline_total, before)):
                    add_counts(result, count)
                    add_counts(result["towns"].setdefault(town, {}), count)
                image_receipts.append(
                    dict(
                        sample=i,
                        fold=fold,
                        original_points=len(peaks),
                        refined_points=len(points),
                        metrics=counted,
                    )
                )
                with (out / "partial_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(image_receipts[-1]) + "\n")
                if images_done % 20 == 0:
                    output_p.flush()
                    output_xy.flush()
                images_done += 1
                if images_done % 20 == 0:
                    phase = f"Spatial RGB TRAIN images: {images_done}/2844"
                    progress()
            if (
                exact_rows != 48865
                or exact_extra != 182016
                or baseline_total != completed["metrics"]
            ):
                raise ValueError("Original sampled/full control recount differs")
            output_xy.flush()
            output_p.flush()
            np.savez(out / "offsets.npz", offsets=np.array(offsets), samples=np.array(data.train))
            atomic_json(out / "image_metrics.json", image_receipts)
            criteria = dict(
                joint_preserved=total["joint"] >= baseline_total["joint"],
                every_town_protected=all(
                    total["towns"][k]["joint"] >= v["joint"]
                    for k, v in baseline_total["towns"].items()
                ),
                every_phase_protected=all(
                    total["classes"][k]["joint"] >= v["joint"]
                    for k, v in baseline_total["classes"].items()
                ),
                background_reduced90_vs_raw=total["background_peaks"]
                <= 0.1 * read(raw / "assessment.json")["results"]["control"]["background_peaks"],
            )
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Registered source/model changed")
            check()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    no_new_fit=True,
                    no_component_val=True,
                    metrics=total,
                    baseline=baseline_total,
                    criteria=criteria,
                    train_pass=all(criteria.values()),
                    exact_sampled_rows=exact_rows,
                    exact_extra_background_feature_rows=exact_extra,
                    original_points=12362313,
                    added_points=added_points,
                    total_points=int(offsets[-1]),
                    all_point_classification_seconds=dict(
                        total=float(sum(times)),
                        median=float(np.median(times)),
                        p95=float(np.quantile(times, 0.95)),
                        excludes_proposer_candidate_generation=True,
                        not_a_runtime_profile=True,
                    ),
                    elapsed_seconds=time.monotonic() - started,
                    artifact_sha256={
                        name: digest(out / name)
                        for name in ("probability.npy", "points.npy", "offsets.npz")
                    },
                ),
            )
            status, phase = (
                "complete",
                "Spatial RGB full TRAIN audit complete; no component VAL or deployment",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
