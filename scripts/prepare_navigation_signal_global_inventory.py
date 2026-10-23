#!/usr/bin/env python3
"""Bounded TRAIN-only candidate inventory; no model prediction or fit."""

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_global_proposals import (  # noqa: E402
    global_rgb_points,
    plateau_rgb_points,
)
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plateau", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    prior = ROOT / "outputs/navigation_signal_refinement_train_20261022"
    diagnosis = ROOT / "reports/navigation_signal_refinement_20261022.json"
    started = time.monotonic()
    status, phase, error, done = "running", "verify TRAIN sources", None, 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 600:
            raise InterruptedError("Owned STOP or600second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 120_000_000:
            raise InterruptedError("120MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Independent RGB proposal TRAIN inventory",
                status=status,
                phase=phase,
                error=error,
                frames=done,
                target_frames=2844,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        progress()
        with BudgetWatch(out):
            prior_result = read(prior / "assessment.json")
            for name in ("points.npy", "offsets.npz"):
                if digest(prior / name) != prior_result["artifact_sha256"][name]:
                    raise ValueError("Prior closed candidate identity changed")
            files = [
                Path(__file__),
                diagnosis,
                prior / "assessment.json",
                ROOT / "reports/navigation_signal_global_inventory_protocol_20261022.md",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            if args.plateau:
                files.append(
                    ROOT / "reports/navigation_signal_plateau_inventory_protocol_20261022.md"
                )
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
                    no_fit_or_prediction=True,
                    no_component_val=True,
                    max_points=15_000_000,
                    stage_seconds=600,
                    artifact_cap_bytes=120_000_000,
                    prior_points_sha256=digest(prior / "points.npy"),
                    plateau_representatives=args.plateau,
                ),
            )
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            old_points = np.load(prior / "points.npy", mmap_mode="r")
            with np.load(prior / "offsets.npz") as a:
                old_offsets, old_samples = a["offsets"], a["samples"]
            if not np.array_equal(old_samples, data.train):
                raise ValueError("Only original ordered TRAIN samples allowed")
            residuals = defaultdict(list)
            for row in read(diagnosis)["residuals"]:
                residuals[row["sample"]].append(row)
            all_points, offsets, times, records, totals = [], [0], [], [], Counter()
            for n, i in enumerate(data.train):
                check()
                tick = time.perf_counter()
                proposed = (plateau_rgb_points if args.plateau else global_rgb_points)(
                    data.images[i].transpose(1, 2, 0)
                )
                old = old_points[old_offsets[n] : old_offsets[n + 1]]
                joined = np.unique(np.concatenate([old, proposed]), axis=0)
                joined = joined[np.lexsort((joined[:, 0], joined[:, 1]))]
                times.append(time.perf_counter() - tick)
                # Union must retain every earlier coordinate without truncation.
                encoded_old = old[:, 1].astype(np.int64) * 512 + old[:, 0]
                encoded_new = joined[:, 1] * 512 + joined[:, 0]
                if not np.isin(encoded_old, encoded_new).all():
                    raise ValueError("Original refinement point dropped")
                all_points.append(joined.astype(np.uint16))
                offsets.append(offsets[-1] + len(joined))
                # Preserve useful inventory evidence even if a later resource cap stops it.
                with (out / "partial_counts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                sample=i,
                                old=len(old),
                                proposed=len(proposed),
                                union=len(joined),
                                cumulative=offsets[-1],
                            )
                        )
                        + "\n"
                    )
                if offsets[-1] > 15_000_000:
                    raise InterruptedError(
                        "15millionpoint cap; no truncation or changed proposal rule"
                    )
                coverage = []
                for k, (x0, y0, x1, y1, cls) in enumerate(data.boxes[i]):
                    original = (
                        (old[:, 0] >= x0) & (old[:, 0] < x1) & (old[:, 1] >= y0) & (old[:, 1] < y1)
                    ).any()
                    current = (
                        (joined[:, 0] >= x0)
                        & (joined[:, 0] < x1)
                        & (joined[:, 1] >= y0)
                        & (joined[:, 1] < y1)
                    ).any()
                    totals["labels"] += 1
                    totals["old_candidate_coverage"] += int(original)
                    totals["new_candidate_coverage"] += int(current)
                    coverage.append(
                        dict(box_index=k, cls=cls, old=bool(original), new=bool(current))
                    )
                misses = []
                for row in residuals[i]:
                    exact = [
                        x for x in row["oracle_coordinates"] if x[1] * 512 + x[0] in encoded_new
                    ]
                    old_covered, now_covered = (
                        coverage[row["box_index"]]["old"],
                        coverage[row["box_index"]]["new"],
                    )
                    totals["missed_housings"] += 1
                    totals["newly_covered_missed_housings"] += int(now_covered and not old_covered)
                    totals["missed_with_exact_recognized_oracle_in_new_set"] += bool(exact)
                    misses.append(
                        dict(
                            box_index=row["box_index"],
                            previously_no_candidate=not old_covered,
                            now_candidate=now_covered,
                            recognized_oracle_hits=exact,
                        )
                    )
                records.append(
                    dict(
                        sample=i,
                        original_points=len(old),
                        global_points=len(proposed),
                        union_points=len(joined),
                        coverage=coverage,
                        residuals=misses,
                    )
                )
                done += 1
                if done % 20 == 0:
                    phase = f"TRAIN RGB inventory: {done}/2844"
                    progress()
            np.save(out / "points.npy", np.concatenate(all_points))
            np.savez(out / "offsets.npz", offsets=np.array(offsets), samples=np.array(data.train))
            atomic_json(out / "image_inventory.json", records)
            if totals["labels"] != 1248 or totals["missed_housings"] != 59:
                raise ValueError("Original TRAIN labels or residual count changed")
            criteria = dict(
                previously_no_candidate_gain=totals["newly_covered_missed_housings"] > 0,
                recognized_oracle_gain=totals["missed_with_exact_recognized_oracle_in_new_set"] > 0,
                bounded_points=offsets[-1] <= 15_000_000,
            )
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Pinned source changed")
            check()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    TRAIN_only=True,
                    no_model_predictions=True,
                    no_component_val=True,
                    totals=dict(totals),
                    original_points=len(old_points),
                    union_points=offsets[-1],
                    added_points=offsets[-1] - len(old_points),
                    criteria=criteria,
                    proceed_to_registered_inference=all(criteria.values()),
                    generation_seconds=dict(
                        total=sum(times),
                        median=float(np.median(times)),
                        p95=float(np.quantile(times, 0.95)),
                    ),
                    elapsed_seconds=time.monotonic() - started,
                    artifact_sha256={
                        name: digest(out / name)
                        for name in ("points.npy", "offsets.npz", "image_inventory.json")
                    },
                ),
            )
            status, phase = "complete", "RGB proposal inventory complete; no predictions or fit"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
