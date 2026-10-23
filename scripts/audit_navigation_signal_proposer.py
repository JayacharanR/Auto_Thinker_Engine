#!/usr/bin/env python3
"""Audit a closed frozen-proposer cache with an independent original-model hook."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalGrid  # noqa:E402
from src.navigation.signal_rgb_augmentation import GAMMAS  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def validate_arrays(train, validation, offsets, expected_offsets, train_points):
    if train.dtype != np.float16 or validation.dtype != np.float16:
        raise ValueError("FP16 feature arrays required")
    if train.shape != (3, train_points, 36) or validation.shape != (int(expected_offsets[-1]), 36):
        raise ValueError("Exact ordered point denominators required")
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("Original validation offsets changed")
    for array in (train, validation):
        if not np.isfinite(array).all():
            raise ValueError("Finite complete features required")
        probabilities = array[..., 32:].astype(np.float32)
        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=0.001)
        ):
            raise ValueError("Four normalized cached probabilities required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache, out = args.cache.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, error, done = "running", None, 0
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False

    def read(path):
        return json.loads(Path(path).read_text())

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 600:
            raise InterruptedError("Owned STOP or 600-second audit cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 10_000_000:
            raise InterruptedError("10 MB audit cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                status=status,
                error=error,
                stage="Independent frozen-proposer cache audit",
                frames=done,
                target_frames=67,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            progress()
            if read(cache / "progress.json")["status"] != "complete":
                raise ValueError("Only a terminal complete cache may be audited")
            result, request = read(cache / "assessment.json"), read(cache / "request.json")
            for path, sha in request["source_hashes"].items():
                if digest(ROOT / path) != sha or digest(cache / "source" / path) != sha:
                    raise ValueError("Recorded cache source generation changed")
            for path, sha in request["original_feature_hashes"].items():
                if digest(path) != sha:
                    raise ValueError("Original RGB feature source changed")
            for file, key in (
                ("train_proposer_features.npz", "train_sha256"),
                ("validation_proposer_features.npz", "validation_sha256"),
            ):
                if digest(cache / file) != result[key]:
                    raise ValueError("Closed feature artifact changed")
            shutil.copyfile(__file__, out / "audit_source.py")
            atomic_json(
                out / "request.json",
                dict(
                    source_sha256=digest(__file__),
                    cache=str(cache),
                    cache_request_sha256=digest(cache / "request.json"),
                    cache_assessment_sha256=digest(cache / "assessment.json"),
                    feature_checks="All arrays verified; fixed first image per location recomputed",
                    maximum_seconds=600,
                    maximum_bytes=10_000_000,
                ),
            )
            original = ROOT / "outputs/navigation_signal_pretrained_20261021"
            rows, original_data = (
                read(original / "train_rows.json"),
                read(original / "dataset.json"),
            )
            expected_offsets = read(original / "cache.json")["validation_offsets"]
            with np.load(cache / "train_proposer_features.npz") as a:
                train = a["features"]
            with np.load(cache / "validation_proposer_features.npz") as a:
                validation, offsets = a["features"], a["offsets"]
            validate_arrays(train, validation, offsets, expected_offsets, len(rows))
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            actual_data = dict(samples=data.samples, train=data.train, validation=data.validation)
            if json.dumps(actual_data, sort_keys=True, allow_nan=False) != json.dumps(
                original_data, sort_keys=True, allow_nan=False
            ):
                raise ValueError("Ordered RGB sources differ")
            by_sample = {}
            for j, row in enumerate(rows):
                if row["role"] != "train" or row["sample"] not in data.train:
                    raise ValueError("Non-TRAIN feature row")
                by_sample.setdefault(row["sample"], []).append(j)
            parent = ROOT / "outputs/navigation_signal_pixel_study_20261021/bulb"
            checkpoint = parent / "step1000.pt"
            if digest(checkpoint) != request["proposer_sha256"]:
                raise ValueError("Original frozen model identity changed")
            model = NativeSignalGrid().cuda().eval().requires_grad_(False)
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            )
            captured = []
            handle = model.features.register_forward_hook(
                lambda _m, _i, output: captured.append(output)
            )
            cases, seen = [], set()
            for index in data.train:
                sample = data.samples[index]
                location = (sample["town"], sample["junction_id"])
                if location in seen or index not in by_sample:
                    continue
                seen.add(location)
                indices = by_sample[index]
                points = [[rows[j]["x"], rows[j]["y"]] for j in indices]
                for gi, gamma in enumerate((1.0, *GAMMAS)):
                    cases.append((index, gamma, np.array(points), train[gi, indices], "train"))
            if len(seen) != 21:
                raise ValueError("All 21 TRAIN locations required in audit sample")
            peaks = {r["sample"]: r["peaks"] for r in read(parent / "predictions1000.json")}
            seen_val = set()
            for n, index in enumerate(data.validation):
                town = data.samples[index]["town"]
                if town in seen_val or not peaks[index]:
                    continue
                seen_val.add(town)
                points = np.array([[p["x"], p["y"]] for p in peaks[index]])
                cases.append(
                    (index, 1.0, points, validation[offsets[n] : offsets[n + 1]], "validation")
                )
            if len(cases) != 67 or len(seen_val) != 4:
                raise ValueError("Fixed 67-forward audit sample changed")
            receipts = []
            for index, gamma, points, expected, role in cases:
                check()
                image = data.images[index].transpose(1, 2, 0)
                # Independent transform: same registered quantized channelwise formula.
                lut = np.round(255 * (np.arange(256, dtype=np.float64) / 255) ** gamma).astype(
                    np.uint8
                )
                transformed = lut[image]
                tensor = torch.tensor(transformed.transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
                captured.clear()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    probabilities = model(tensor).float().softmax(1)
                x, y = points[:, 0], points[:, 1]
                f = captured[0][0].float().cpu().numpy()[:, y, x].T
                p = probabilities[0].cpu().numpy()[:, y, x].T
                actual = np.concatenate([f, p], axis=1).astype(np.float16)
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"Independent model-hook cache mismatch sample{index} gamma{gamma}"
                    )
                receipts.append(
                    dict(sample=index, gamma=gamma, role=role, points=len(points), exact=True)
                )
                done += 1
                progress()
            handle.remove()
            for path, sha in request["source_hashes"].items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Source changed during independent audit")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    all_array_shapes_ranges_hashes_verified=True,
                    independent_exact_forwards=done,
                    sampled_train_locations=21,
                    sampled_validation_towns=4,
                    receipts=receipts,
                    all_points_recomputed=False,
                    no_fit=True,
                    no_deployment=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status = "complete"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
