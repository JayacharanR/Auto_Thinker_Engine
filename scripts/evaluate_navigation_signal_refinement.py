#!/usr/bin/env python3
"""One conditional component check; RGB refinement only, no new fit."""

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_signal_supported_full import add_counts, count_image  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_context import context_patches  # noqa: E402
from src.navigation.signal_hard_pixels import protected_signal_metrics  # noqa: E402
from src.navigation.signal_pixel_study import NativeSignalDataset  # noqa: E402
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
    train = ROOT / "outputs/navigation_signal_refinement_train_20261022"
    previous = ROOT / "outputs/navigation_signal_supported_final_20261022"
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    started = time.monotonic()
    status, phase, error, done = "running", "verify passing TRAIN refinement", None, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 300:
            raise InterruptedError("Owned STOP or300second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 40_000_000:
            raise InterruptedError("40MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="RGB refinement component evaluation",
                status=status,
                phase=phase,
                error=error,
                frames=done,
                target_frames=144,
                elapsed_seconds=time.monotonic() - started,
                updated_at=time.time(),
            ),
        )

    try:
        progress()
        with BudgetWatch(out), threadpool_limits(limits=2):
            assessment = read(train / "assessment.json")
            if (
                read(train / "progress.json")["status"] != "complete"
                or not assessment["train_pass"]
                or assessment["exact_sampled_rows"] != 48865
            ):
                raise ValueError("Passing complete TRAIN refinement audit required")
            prior = read(previous / "assessment.json")
            model_file = previous / "final/model.pkl"
            expected = "ed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191"
            if digest(model_file) != expected or prior["model_sha256"] != expected:
                raise ValueError("Exact previously fitted supported model required")
            model = pickle.loads(model_file.read_bytes())
            parent = (
                ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb/predictions1000.json"
            )
            files = [
                Path(__file__),
                ROOT / "scripts/audit_navigation_signal_supported_full.py",
                ROOT / "reports/navigation_signal_refinement_component_protocol_20261022.md",
                train / "assessment.json",
                train / "request.json",
                previous / "assessment.json",
                model_file,
                previous / "final/validation_class_probability.npy",
                cache / "cache.json",
                parent,
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    no_new_fit=True,
                    TRAIN_decision_sha256=digest(train / "assessment.json"),
                    model_sha256=expected,
                    no_driving=True,
                    seconds_cap=300,
                    artifact_cap_bytes=40_000_000,
                ),
            )
            meta = read(cache / "cache.json")
            if digest(cache / "validation_features.npz") != meta["validation_features_sha256"]:
                raise ValueError("Original component feature cache changed")
            with np.load(cache / "validation_features.npz") as a:
                original_features = a["rgb"]
            old_offsets = meta["validation_offsets"]
            old_probability = np.load(previous / "final/validation_class_probability.npy")
            original_peaks = {r["sample"]: r["peaks"] for r in read(parent)}
            data = NativeSignalDataset(
                ROOT, ROOT / "outputs/navigation_signal_pixel_audit_20261021"
            )
            total, before = dict(towns={}), dict(towns={})
            points_all, probability_all, offsets, rows, times, exact = [], [], [0], [], [], 0
            for n, i in enumerate(data.validation):
                check()
                rgb, peaks = data.images[i].transpose(1, 2, 0), original_peaks[i]
                xy = np.array([[p["x"], p["y"]] for p in peaks], np.int64).reshape(-1, 2)
                old = old_probability[old_offsets[n] : old_offsets[n + 1]]
                tick = time.perf_counter()
                refined = refine_rgb_points(rgb, xy)
                lookup = {tuple(p): j for j, p in enumerate(refined)}
                old_lookup = {tuple(p): j for j, p in enumerate(xy)}
                if not set(old_lookup) <= set(lookup):
                    raise ValueError("Original component proposal dropped")
                p = np.full((len(refined), 4), np.nan)
                for point, j in old_lookup.items():
                    p[lookup[point]] = old[j]
                new = [j for j, point in enumerate(refined) if tuple(point) not in old_lookup]
                if new:
                    features = rgb_features(context_patches(rgb, refined[new])).astype(np.float16)
                    p[new] = model.predict_proba(features.astype(np.float32))
                times.append(time.perf_counter() - tick)
                recomputed = rgb_features(context_patches(rgb, xy)).astype(np.float16)
                if not np.array_equal(
                    recomputed, original_features[old_offsets[n] : old_offsets[n + 1]]
                ):
                    raise ValueError("Original RGB feature parity failed")
                probability = (
                    model.predict_proba(recomputed.astype(np.float32))
                    if len(xy)
                    else np.empty((0, 4))
                )
                if not np.array_equal(probability, old):
                    raise ValueError("Original model probability parity failed")
                exact += len(xy)
                if not np.isfinite(p).all():
                    raise ValueError("Missing candidate probability")
                new_peaks = [dict(x=int(x), y=int(y)) for x, y in refined]
                counted = count_image(new_peaks, p, data.boxes[i], data.background[i])
                original = count_image(peaks, old, data.boxes[i], data.background[i])
                town = data.samples[i]["town"]
                for accumulator, result in ((total, counted), (before, original)):
                    add_counts(accumulator, result)
                    add_counts(accumulator["towns"].setdefault(town, {}), result)
                rows.append(
                    dict(
                        sample=i,
                        metrics=counted,
                        original_points=len(xy),
                        refined_points=len(refined),
                    )
                )
                points_all.append(refined.astype(np.uint16))
                probability_all.append(p)
                offsets.append(offsets[-1] + len(refined))
                done += 1
                if done % 10 == 0:
                    phase = f"Refined component images: {done}/144"
                    progress()
            if exact != 38543:
                raise ValueError("Original component count changed")
            for key in ("labels", "localized", "joint", "background_peaks", "unknown_peaks"):
                if before[key] != prior["metrics"][key]:
                    raise ValueError("Original component control recount disagrees")
            np.save(out / "points.npy", np.concatenate(points_all))
            np.save(out / "probability.npy", np.concatenate(probability_all))
            np.savez(
                out / "offsets.npz", offsets=np.array(offsets), samples=np.array(data.validation)
            )
            atomic_json(out / "image_metrics.json", rows)
            criteria = dict(
                coverage_protected=protected_signal_metrics(total, prior["baseline"]),
                background_reduced90=total["background_peaks"]
                <= 0.1 * prior["baseline"]["background_peaks"],
            )
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Registered source changed")
            check()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    no_new_fit=True,
                    metrics=total,
                    supported_baseline=before,
                    original_parent=prior["baseline"],
                    criteria=criteria,
                    component_pass=all(criteria.values()),
                    driving_qualified=False,
                    exact_original_feature_probability_rows=exact,
                    total_points=offsets[-1],
                    added_points=offsets[-1] - exact,
                    repeatedly_inspected_development_set=True,
                    incremental_refinement_timing=dict(
                        total_seconds=float(sum(times)),
                        median_seconds=float(np.median(times)),
                        p95_seconds=float(np.quantile(times, 0.95)),
                        not_full_runtime=True,
                    ),
                    artifact_sha256={
                        name: digest(out / name)
                        for name in ("points.npy", "probability.npy", "offsets.npz")
                    },
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "Refined component evaluated once; no deployment"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
