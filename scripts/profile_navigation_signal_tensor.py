#!/usr/bin/env python3
"""Bounded TRAIN-only export parity/profile; no deployed model changes."""

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.storage_budget import BudgetWatch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--deadline-from", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Owned output required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status, phase, error, done = "running", "verify staged export and TRAIN cache", None, 0
    read = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    deadline = time.time() + 300
    if args.deadline_from:
        failed = read(args.deadline_from)
        if (
            failed["status"] != "failed"
            or failed["frames"] != 0
            or "CUDA unavailable" not in failed["error"]
        ):
            raise ValueError("Only the unperformed sandbox CUDA stage may continue")
        deadline = failed["updated_at"] - failed["elapsed_seconds"] + 300

    def check():
        if (out / "STOP").exists() or time.time() > deadline:
            raise InterruptedError("Owned STOP or300second cap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 30_000_000:
            raise InterruptedError("30MB artifact cap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Staged numeric signal tree profile",
                status=status,
                phase=phase,
                error=error,
                frames=done,
                target_frames=4096,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        progress()
        with BudgetWatch(out), threadpool_limits(limits=2):
            module_file = ROOT / "patches/navigation_signal_tree_tensor_candidate.py"
            model_file = ROOT / "outputs/navigation_signal_supported_final_20261022/final/model.pkl"
            features_file = (
                ROOT / "outputs/navigation_signal_pretrained_20261021/train_features.npz"
            )
            if (
                digest(model_file)
                != "ed533241a872943c928d725ac06399c7bc2c78ce73e1f05cd4d359929e583191"
            ):
                raise ValueError("Exact previously fitted head required")
            if (
                digest(features_file)
                != read(features_file.parent / "cache.json")["train_features_sha256"]
            ):
                raise ValueError("Exact original TRAIN cache required")
            files = [
                Path(__file__),
                module_file,
                ROOT / "reports/navigation_signal_tensor_profile_protocol_20261022.md",
                ROOT / "tests/test_navigation_staged_signal_tree_tensor.py",
                features_file.parent / "cache.json",
                model_file,
            ]
            if args.deadline_from:
                files.append(args.deadline_from.resolve())
            hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
            for path in files:
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            atomic_json(
                out / "request.json",
                dict(
                    source_hashes=hashes,
                    TRAIN_features_sha256=digest(features_file),
                    rows=list(range(4096)),
                    created_at=time.time(),
                    no_fit=True,
                    no_VAL=True,
                    no_deployment=True,
                    parity_atol=2e-13,
                    batch=256,
                    warmups=2,
                    repeats=10,
                    concurrent_cpu_audit=True,
                    original_absolute_deadline=deadline,
                    failed_sandbox_profile=str(args.deadline_from) if args.deadline_from else None,
                ),
            )
            spec = importlib.util.spec_from_file_location("staged_tree_profile", module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = pickle.loads(model_file.read_bytes())
            bundle = module.export_numeric_classifier(model)
            np.savez(
                out / "export.npz",
                **bundle["arrays"],
                baseline=bundle["baseline"],
                maximum_depth=bundle["maximum_depth"],
                classes=bundle["classes"],
            )
            with np.load(features_file) as a:
                x = a["rgb"][:4096].astype(np.float32)
            torch.set_num_threads(2)
            cpu = module.NumericSignalTrees(bundle)
            reference = model.predict_proba(x)
            phase = "CPU numerical parity"
            progress()
            cpu_p = []
            for offset in range(0, len(x), 256):
                check()
                cpu_p.append(cpu(torch.from_numpy(x[offset : offset + 256])).numpy())
            cpu_p = np.concatenate(cpu_p)
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; keep CPU profile incomplete, no fallback speed claim"
                )
            gpu = module.NumericSignalTrees(bundle).cuda()
            torch.cuda.reset_peak_memory_stats()
            phase = "GPU numerical parity"
            progress()
            gpu_p = []
            for offset in range(0, len(x), 256):
                check()
                gpu_p.append(gpu(torch.from_numpy(x[offset : offset + 256]).cuda()).cpu().numpy())
                done = min(len(x), offset + 256)
                progress()
            gpu_p = np.concatenate(gpu_p)
            parity = {}
            for name, p in (("cpu_tensor", cpu_p), ("gpu_tensor", gpu_p)):
                parity[name] = dict(
                    max_absolute_difference=float(np.abs(p - reference).max()),
                    acceptance_changes=int(
                        ((p[:, 1:].sum(1) >= 0.5) != (reference[:, 1:].sum(1) >= 0.5)).sum()
                    ),
                    phase_changes=int((p[:, 1:].argmax(1) != reference[:, 1:].argmax(1)).sum()),
                )
            np.savez(out / "probabilities.npz", reference=reference, cpu=cpu_p, gpu=gpu_p)
            if any(
                r["max_absolute_difference"] > 2e-13
                or r["acceptance_changes"]
                or r["phase_changes"]
                for r in parity.values()
            ):
                atomic_json(out / "failed_parity.json", parity)
                raise ValueError("Registered export parity tolerance or decision parity failed")
            phase = "Same256row kernel timing"
            progress()
            cpu_x, gpu_x = torch.from_numpy(x[:256]), torch.from_numpy(x[:256]).cuda()
            timings = {}
            for name, call in (
                ("sklearn_cpu", lambda: model.predict_proba(x[:256])),
                ("tensor_cpu", lambda: cpu(cpu_x)),
                ("tensor_gpu", lambda: gpu(gpu_x)),
            ):
                for _ in range(2):
                    check()
                    call()
                values = []
                for _ in range(10):
                    check()
                    if name == "tensor_gpu":
                        torch.cuda.synchronize()
                    tick = time.perf_counter()
                    call()
                    if name == "tensor_gpu":
                        torch.cuda.synchronize()
                    values.append(time.perf_counter() - tick)
                timings[name] = dict(seconds=values, median=float(np.median(values)))
            for path, sha in hashes.items():
                if digest(ROOT / path) != sha:
                    raise ValueError("Registered profile source changed")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    parity=parity,
                    timings=timings,
                    gpu=torch.cuda.get_device_name(),
                    cuda_version=torch.version.cuda,
                    torch_version=torch.__version__,
                    peak_torch_bytes=torch.cuda.max_memory_allocated(),
                    no_full_runtime_claim=True,
                    no_deployment=True,
                    no_new_fit=True,
                    TRAIN_only=True,
                    tested_rows=len(x),
                    tested_batch=256,
                    concurrent_cpu_audit=True,
                    elapsed_seconds=time.monotonic() - started,
                    artifacts={
                        name: digest(out / name) for name in ("export.npz", "probabilities.npz")
                    },
                ),
            )
            status, phase = "complete", "Staged export parity and kernel timing only; not deployed"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
