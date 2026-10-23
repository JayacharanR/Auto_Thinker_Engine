#!/usr/bin/env python3
"""All frozen TRAIN proposals, existing out-of-fold heads, no new fits."""

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

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_context import context_patches  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_pretrained_verifier import rgb_features  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def count_image(peaks, p, boxes, background):
    """Independent direct recount using original score-ordered housing matching."""
    remaining, matched = set(range(len(boxes))), {}
    fg = p[:, 1:].sum(1)
    order = sorted(np.flatnonzero(fg >= 0.5), key=lambda j: (-fg[j], peaks[j]["y"], peaks[j]["x"]))
    bg, unknown = 0, 0
    for j in order:
        x, y = peaks[j]["x"], peaks[j]["y"]
        k = next(
            (
                k
                for k in sorted(remaining)
                if boxes[k][0] <= x < boxes[k][2] and boxes[k][1] <= y < boxes[k][3]
            ),
            None,
        )
        if k is not None:
            remaining.remove(k)
            matched[k] = int(p[j, 1:].argmax() + 1)
        elif background[y, x]:
            bg += 1
        else:
            unknown += 1
    counts = dict(
        labels=len(boxes),
        localized=len(matched),
        joint=sum(matched.get(k) == b[4] for k, b in enumerate(boxes)),
        background_peaks=bg,
        unknown_peaks=unknown,
        classes={},
    )
    for k, b in enumerate(boxes):
        c = counts["classes"].setdefault(str(b[4]), dict(labels=0, localized=0, joint=0))
        c["labels"] += 1
        c["localized"] += int(k in matched)
        c["joint"] += int(matched.get(k) == b[4])
    return counts


def add_counts(total, current):
    for key in ("labels", "localized", "joint", "background_peaks", "unknown_peaks"):
        total[key] = total.get(key, 0) + current.get(key, 0)
    for cls, c in current.get("classes", {}).items():
        target = total.setdefault("classes", {}).setdefault(cls, {})
        for key, value in c.items():
            target[key] = target.get(key, 0) + value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    cache = ROOT / "outputs/navigation_signal_pretrained_20261021"
    candidate = ROOT / "outputs/navigation_signal_supported_20261022"
    baseline = ROOT / "outputs/navigation_signal_phase_boost_20261021"
    folds_path = ROOT / "outputs/navigation_signal_regularized_20261021/folds.npz"
    started = time.monotonic()
    status, phase, error, images_done, points_done = "running", "verify sources", None, 0, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("Owned STOP or900second stage cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("100MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Full TRAIN proposal verification",
                status=status,
                phase=phase,
                error=error,
                frames=images_done,
                target_frames=2844,
                points_done=points_done,
                total_points=715825,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        progress()
        with BudgetWatch(out), threadpool_limits(limits=2):
            request = read(candidate / "request.json")
            if (
                read(candidate / "progress.json")["status"] != "complete"
                or not read(candidate / "assessment.json")["proceed_to_full_train_audit"]
            ):
                raise ValueError("Completed passing sampled TRAIN diagnostic required")
            for path, sha in request["source_hashes"].items():
                if digest(candidate / "source" / path) != sha:
                    raise ValueError("Archived study source changed")
            files = [
                Path(__file__),
                ROOT / "reports/navigation_signal_supported_full_protocol_20261022.md",
                cache / "train_proposals.json",
                cache / "train_rows.json",
                cache / "dataset.json",
                candidate / "request.json",
                candidate / "assessment.json",
                folds_path,
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            models = {}
            for arm, parent in (("control", baseline), ("supported", candidate)):
                models[arm] = []
                for fold in range(3):
                    directory = parent / f"joint100_fold{fold}"
                    file = directory / "model.pkl"
                    if digest(file) != read(directory / "fit.json")["model_sha256"]:
                        raise ValueError("Exact completed model required")
                    files += [file, directory / "fit.json"]
                    models[arm].append(pickle.loads(file.read_bytes()))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                # Large existing proposal identities are hash referenced, not duplicated.
                if path.suffix in (".py", ".md", ".pkl"):
                    target = out / "source" / path.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    created_at=time.time(),
                    no_new_fits=True,
                    no_component_validation=True,
                    cpu_threads=2,
                    seconds_cap=900,
                    artifact_cap=100_000_000,
                ),
            )
            if digest(folds_path) != request["folds_sha256"]:
                raise ValueError("TRAIN folds changed")
            with np.load(folds_path) as a:
                folds, predicted = a["fold"], a["predicted"]
            rows, proposals = read(cache / "train_rows.json"), read(cache / "train_proposals.json")
            groups = defaultdict(list)
            for j, row in enumerate(rows):
                if row["role"] != "train":
                    raise ValueError("Non-TRAIN cache row")
                groups[row["sample"]].append(j)
            if digest(cache / "train_features.npz") != request["train_features_sha256"]:
                raise ValueError("Original feature cache changed")
            with np.load(cache / "train_features.npz") as a:
                original = a["rgb"]
            old_p = {arm: np.load(candidate / f"oof_{arm}.npy") for arm in ("control", "supported")}
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            identity = read(cache / "dataset.json")
            if (
                json.dumps(data.samples, sort_keys=True)
                != json.dumps(identity["samples"], sort_keys=True)
                or data.train != identity["train"]
            ):
                raise ValueError("Original ordered TRAIN data changed")
            if (
                set(map(int, proposals)) != set(data.train)
                or sum(map(len, proposals.values())) != 715825
            ):
                raise ValueError("All original TRAIN proposals required")
            output_p = {
                arm: np.lib.format.open_memmap(
                    out / f"{arm}_probability.npy", mode="w+", dtype=np.float64, shape=(715825, 4)
                )
                for arm in models
            }
            totals = {arm: dict(towns={}) for arm in models}
            sampled_totals = {arm: dict(towns={}) for arm in models}
            offsets, sample_receipts, exact_rows = [0], [], 0
            for i in data.train:
                check()
                peaks = proposals[str(i)]
                row_ids = groups[i]
                fi = set(int(folds[j]) for j in row_ids)
                if len(fi) != 1:
                    raise ValueError("Every image needs one held-out TRAIN fold")
                fold = fi.pop()
                xy = np.array([[p["x"], p["y"]] for p in peaks], dtype=np.int64).reshape(-1, 2)
                features = rgb_features(
                    context_patches(data.images[i].transpose(1, 2, 0), xy)
                ).astype(np.float16)
                lookup = {tuple(point): k for k, point in enumerate(xy)}
                if len(lookup) != len(peaks):
                    raise ValueError("Duplicate original proposal coordinates")
                selected_rows = [j for j in row_ids if predicted[j]]
                indices = [lookup[(rows[j]["x"], rows[j]["y"])] for j in selected_rows]
                if not np.array_equal(features[indices], original[selected_rows]):
                    raise ValueError("Sampled RGB feature parity failed")
                exact_rows += len(indices)
                sample_result = {}
                for arm in models:
                    p = (
                        models[arm][fold].predict_proba(features.astype(np.float32))
                        if len(features)
                        else np.empty((0, 4))
                    )
                    if not np.array_equal(p[indices], old_p[arm][selected_rows]):
                        raise ValueError("Original sampled OOF predictions changed")
                    output_p[arm][points_done : points_done + len(p)] = p
                    counted = count_image(peaks, p, data.boxes[i], data.background[i])
                    sample_result[arm] = counted
                    town = data.samples[i]["town"]
                    add_counts(totals[arm], counted)
                    add_counts(totals[arm]["towns"].setdefault(town, {}), counted)
                    counted_sample = count_image(
                        [peaks[k] for k in indices], p[indices], data.boxes[i], data.background[i]
                    )
                    add_counts(sampled_totals[arm], counted_sample)
                    add_counts(sampled_totals[arm]["towns"].setdefault(town, {}), counted_sample)
                sample_receipts.append(dict(sample=i, fold=fold, results=sample_result))
                images_done += 1
                points_done += len(peaks)
                offsets.append(points_done)
                if images_done % 20 == 0:
                    phase = f"All TRAIN proposals: {images_done}/2844 images"
                    progress()
            for arm, array in output_p.items():
                array.flush()
                expected = read(candidate / "assessment.json")["results"][arm]
                actual = sampled_totals[arm]
                for key in ("labels", "localized", "joint"):
                    if actual[key] != expected[key]:
                        raise ValueError("Independent sampled metric recount failed")
                if actual["background_peaks"] != expected["sampled_background"]:
                    raise ValueError("Independent sampled background recount failed")
                for group in ("towns", "classes"):
                    for k, v in expected[group].items():
                        for metric in ("labels", "localized", "joint"):
                            if actual[group][k][metric] != v[metric]:
                                raise ValueError("Independent subgroup recount failed")
            if exact_rows != 48865 or images_done != 2844 or points_done != 715825:
                raise ValueError("Full coverage/parity denominator mismatch")
            atomic_json(out / "image_metrics.json", sample_receipts)
            np.savez(out / "offsets.npz", offsets=np.array(offsets), samples=np.array(data.train))
            a, b = totals["supported"], totals["control"]
            criteria = dict(
                joint_preserved=a["joint"] >= b["joint"],
                each_town_preserved=all(
                    a["towns"][k]["joint"] >= v["joint"] for k, v in b["towns"].items()
                ),
                each_phase_preserved=all(
                    a["classes"][k]["joint"] >= v["joint"] for k, v in b["classes"].items()
                ),
                background_lower=a["background_peaks"] < b["background_peaks"],
            )
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Registered source/model changed during full audit")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    no_new_fits=True,
                    no_component_val=True,
                    results=totals,
                    criteria=criteria,
                    train_audit_pass=all(criteria.values()),
                    exact_sampled_feature_and_probability_rows=exact_rows,
                    independent_sampled_recount_exact=True,
                    probability_sha256={
                        arm: digest(out / f"{arm}_probability.npy") for arm in models
                    },
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = (
                "complete",
                "All TRAIN proposals verified; no component VAL or deployment",
            )
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
