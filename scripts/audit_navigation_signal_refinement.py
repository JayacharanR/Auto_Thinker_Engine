#!/usr/bin/env python3
"""TRAIN-only RGB proposal refinement with frozen out-of-fold supported heads."""

import argparse
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
from src.navigation.signal_rgb_refinement import refine_rgb_points  # noqa: E402
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
    study = ROOT / "outputs/navigation_signal_supported_20261022"
    full = ROOT / "outputs/navigation_signal_supported_full_20261022"
    folds_path = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
    started = time.monotonic()
    status, phase, error, images_done = "running", "verify sources", None, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Owned STOP or900second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 120_000_000:
            raise InterruptedError("120MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="RGB local proposal refinement",
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
            if (
                read(full / "progress.json")["status"] != "complete"
                or not completed["train_audit_pass"]
            ):
                raise ValueError("Completed full TRAIN verification required")
            models = []
            files = [
                Path(__file__),
                ROOT / "scripts/audit_navigation_signal_supported_full.py",
                ROOT / "reports/navigation_signal_refinement_protocol_20261022.md",
                full / "assessment.json",
                full / "offsets.npz",
                folds_path,
                cache / "train_rows.json",
                cache / "train_proposals.json",
                cache / "dataset.json",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            for fold in range(3):
                directory = study / f"joint100_fold{fold}"
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
            if (
                digest(full / "supported_probability.npy")
                != completed["probability_sha256"]["supported"]
            ):
                raise ValueError("Original full TRAIN probabilities changed")
            if digest(cache / "train_features.npz") != old_request["train_features_sha256"]:
                raise ValueError("Original sampled RGB features changed")
            if digest(folds_path) != old_request["folds_sha256"]:
                raise ValueError("Original TRAIN folds changed")
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    no_new_fits=True,
                    no_component_val=True,
                    stage_seconds=900,
                    artifact_cap_bytes=120_000_000,
                    baseline_probability_sha256=digest(full / "supported_probability.npy"),
                ),
            )
            with np.load(folds_path) as a:
                folds, predicted = a["fold"], a["predicted"]
            with np.load(full / "offsets.npz") as a:
                old_offsets, old_indices = a["offsets"], a["samples"]
            with np.load(cache / "train_features.npz") as a:
                original_features = a["rgb"]
            original_p = np.load(full / "supported_probability.npy", mmap_mode="r")
            rows, proposals = read(cache / "train_rows.json"), read(cache / "train_proposals.json")
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
            refined, offsets, generation_seconds = [], [0], []
            phase = "Generate RGB-only points for all TRAIN images"
            progress()
            for i in data.train:
                check()
                points = np.array([[p["x"], p["y"]] for p in proposals[str(i)]], np.int64).reshape(
                    -1, 2
                )
                tick = time.perf_counter()
                refined.append(refine_rgb_points(data.images[i].transpose(1, 2, 0), points))
                generation_seconds.append(time.perf_counter() - tick)
                offsets.append(offsets[-1] + len(refined[-1]))
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
                    refined[n],
                    proposals[str(i)],
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
                p = np.full((len(points), 4), np.nan)
                for xy, j in old_lookup.items():
                    p[lookup[xy]] = old[j]
                new_ids = [j for j, xy in enumerate(points) if tuple(xy) not in old_lookup]
                tick = time.perf_counter()
                if new_ids:
                    features = rgb_features(context_patches(rgb, points[new_ids])).astype(
                        np.float16
                    )
                    p[new_ids] = models[fold].predict_proba(features.astype(np.float32))
                times.append(time.perf_counter() - tick + generation_seconds[n])
                # Existing sampled coordinates are recomputed independently, not just copied.
                check_rows = [j for j in row_ids if predicted[j]]
                check_xy = np.array(
                    [[rows[j]["x"], rows[j]["y"]] for j in check_rows], np.int64
                ).reshape(-1, 2)
                check_features = rgb_features(context_patches(rgb, check_xy)).astype(np.float16)
                if not np.array_equal(check_features, original_features[check_rows]):
                    raise ValueError("Original sampled RGB features changed")
                check_p = (
                    models[fold].predict_proba(check_features.astype(np.float32))
                    if len(check_rows)
                    else np.empty((0, 4))
                )
                if not np.array_equal(check_p, p[[lookup[tuple(xy)] for xy in check_xy]]):
                    raise ValueError("Original sampled OOF predictions changed")
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
                images_done += 1
                if images_done % 20 == 0:
                    phase = f"Refined TRAIN images: {images_done}/2844"
                    progress()
            if exact_rows != 48865 or baseline_total != completed["results"]["supported"]:
                raise ValueError("Original sampled/full control recount differs")
            output_xy.flush()
            output_p.flush()
            np.savez(out / "offsets.npz", offsets=np.array(offsets), samples=np.array(data.train))
            atomic_json(out / "image_metrics.json", image_receipts)
            criteria = dict(
                joint_improves=total["joint"] > baseline_total["joint"],
                every_town_protected=all(
                    total["towns"][k]["joint"] >= v["joint"]
                    for k, v in baseline_total["towns"].items()
                ),
                every_phase_protected=all(
                    total["classes"][k]["joint"] >= v["joint"]
                    for k, v in baseline_total["classes"].items()
                ),
                background_reduced90_vs_raw=total["background_peaks"]
                <= 0.1 * completed["results"]["control"]["background_peaks"],
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
                    original_points=715825,
                    added_points=added_points,
                    total_points=offsets[-1],
                    candidate_generation_plus_new_points_seconds=dict(
                        total=float(sum(times)),
                        median=float(np.median(times)),
                        p95=float(np.quantile(times, 0.95)),
                        excludes_original_proposer_and_original_point_scoring=True,
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
                "Refinement TRAIN audit complete; no component VAL or deployment",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
