#!/usr/bin/env python3
"""Profile staged exact context extraction on 63 predeclared TRAIN images."""

import argparse
import importlib.util
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

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_context import context_patches  # noqa: E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa: E402
from src.navigation.signal_pretrained_verifier import rgb_features  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, error, done, phase = "running", None, 0, "verify sources"
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Exact signal context profile",
                status=status,
                phase=phase,
                frames=done,
                target_frames=63,
                error=error,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 600:
            raise InterruptedError("Owned STOP or 600 second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 20_000_000:
            raise InterruptedError("20 MB artifact cap")

    try:
        progress()
        with BudgetWatch(out), threadpool_limits(limits=2):
            module_file = ROOT / "patches/navigation_signal_context_fast_candidate.py"
            model_file = ROOT / "outputs/navigation_signal_supported_final_20261022/final/model.pkl"
            inventory = ROOT / "outputs/navigation_signal_plateau_inventory_20261022"
            selection = (
                ROOT / "outputs/navigation_signal_background_fit_20261022/cache_verification.json"
            )
            selected = read(selection)["selected_images"]
            if len(selected) != 63 or len(set(selected)) != 63:
                raise ValueError("Original fixed 63 TRAIN images required")
            if (
                digest(model_file)
                != "ed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191"
            ):
                raise ValueError("Exact supported-supervision head required")
            files = [
                Path(__file__),
                module_file,
                model_file,
                selection,
                ROOT / "reports/navigation_signal_context_profile_protocol_20261022.md",
                ROOT / "tests/test_navigation_staged_signal_context_fast.py",
            ]
            files += sorted((ROOT / "src/navigation").glob("*.py"))
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for p in files:
                target = out / "source" / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
            for name, sha in read(inventory / "assessment.json")["artifact_sha256"].items():
                if digest(inventory / name) != sha:
                    raise ValueError("Frozen proposal coordinates changed")
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    selected_images=selected,
                    chunk_size=256,
                    no_fit=True,
                    no_VAL=True,
                    no_deployment=True,
                    created_at=time.time(),
                    concurrent_cpu_audit=True,
                    stage_seconds=600,
                    artifact_cap_bytes=20_000_000,
                    inventory_hashes=read(inventory / "assessment.json")["artifact_sha256"],
                ),
            )
            spec = importlib.util.spec_from_file_location("staged_context", module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = pickle.loads(model_file.read_bytes())
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            with np.load(inventory / "offsets.npz") as a:
                offsets, samples = a["offsets"], a["samples"]
            if not np.array_equal(samples, data.train) or not set(selected) <= set(data.train):
                raise ValueError("TRAIN identity mismatch")
            coords = np.load(inventory / "points.npy", mmap_mode="r")
            positions = {int(s): j for j, s in enumerate(samples)}
            receipts = []
            for ordinal, i in enumerate(selected):
                check()
                phase = f"Compare all proposals in fixed TRAIN image {i}"
                rgb = data.images[i].transpose(1, 2, 0)
                n = positions[i]
                points = coords[offsets[n] : offsets[n + 1]]
                features, seconds = {}, {}
                for label in ["original", "staged"] if ordinal % 2 == 0 else ["staged", "original"]:
                    tick = time.perf_counter()
                    features[label] = (
                        rgb_features(context_patches(rgb, points))
                        if label == "original"
                        else module.exact_rgb_features(rgb, points)
                    )
                    seconds[label + "_features"] = time.perf_counter() - tick
                if not np.array_equal(features["original"], features["staged"]):
                    raise ValueError(f"FP32 feature mismatch at sample {i}")
                # Separate patch-byte verification; excluded from performance timings.
                prepared = module.PreparedContext(rgb)
                for k in range(0, len(points), 256):
                    check()
                    xy = points[k : k + 256]
                    if not np.array_equal(prepared.patches(xy), context_patches(rgb, xy)):
                        raise ValueError(f"Patch-byte mismatch at sample {i}")
                probabilities = {}
                for label in ("original", "staged"):
                    tick = time.perf_counter()
                    probabilities[label] = model.predict_proba(
                        features[label].astype(np.float16).astype(np.float32)
                    )
                    seconds[label + "_head"] = time.perf_counter() - tick
                    seconds[label + "_total"] = (
                        seconds[label + "_features"] + seconds[label + "_head"]
                    )
                if not np.array_equal(probabilities["original"], probabilities["staged"]):
                    raise ValueError("Probability mismatch")
                receipt = dict(
                    sample=i,
                    points=len(points),
                    exact_patches=True,
                    exact_fp32_features=True,
                    exact_fp16_features=True,
                    exact_probabilities=True,
                    seconds=seconds,
                )
                receipts.append(receipt)
                with (out / "images.jsonl").open("a") as stream:
                    stream.write(json.dumps(receipt) + "\n")
                done += 1
                progress()
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Pinned source changed")
            check()
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    train_only=True,
                    no_fit=True,
                    no_deployment=True,
                    no_VAL=True,
                    images=done,
                    exact_points=sum(r["points"] for r in receipts),
                    timings={
                        key: dict(
                            median=float(np.median([r["seconds"][key] for r in receipts])),
                            p95=float(np.quantile([r["seconds"][key] for r in receipts], 0.95)),
                        )
                        for key in receipts[0]["seconds"]
                    },
                    not_full_runtime=True,
                    concurrent_cpu_audit=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "Exact context profile complete; not deployed"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
