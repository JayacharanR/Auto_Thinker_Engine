#!/usr/bin/env python3
"""Cache frozen-parent features and causal RGB-only facts, without fitting."""

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

from scripts.train_navigation_causal_student import BASE, BASE_SHA, PARENT, PARENT_SHA  # noqa: E402
from src.navigation.causal_learning import EpisodeCache, original_training_episodes  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_perception import FrozenSignalPerception  # noqa: E402
from src.navigation.signal_policy import MAX_OBJECTS, TOKEN_DIM, CausalSignalFacts  # noqa: E402
from src.navigation.student import make_student  # noqa: E402

BUNDLE = ROOT / "outputs/navigation_m2_signal_state_expansion_20261016/perception_bundle.json"
EXTRA = ROOT / "outputs/navigation_m2_causal_approach_cache_20261018"
COLLECTION = ROOT / "outputs/navigation_m2_varied_waits_20261018"
PROTOCOL = ROOT / "reports/navigation_m2_signal_residual_protocol_20261018.md"


def selected_episodes(base, allowed):
    train, _ = original_training_episodes(base, allowed)
    return [e for e in base["episodes"] if e["role"] == "validation" or e in train]


@torch.inference_mode()
def cache(args):
    if args.out.exists():
        raise FileExistsError("Preserve prior attempt; cache requires a fresh output")
    if digest(PARENT) != PARENT_SHA or digest(BASE / "manifest.json") != BASE_SHA:
        raise ValueError("Protected parent/cache identity changed")
    position = json.loads(
        (ROOT / "reports/navigation_m2_varied_wait_positions_20261018.json").read_text()
    )
    if not position["audit_passed"] or position["collection_audit_sha256"] != digest(
        COLLECTION / "data_audit.json"
    ):
        raise ValueError("Supplement position audit changed")
    guard = json.loads((COLLECTION / "guard.json").read_text())
    sources = [
        Path(__file__),
        PROTOCOL,
        ROOT / "src/navigation/signal_policy.py",
        ROOT / "src/navigation/signal_perception.py",
        ROOT / "src/navigation/signal_detector.py",
        ROOT / "src/navigation/signal_tracking.py",
        ROOT / "src/navigation/student.py",
        ROOT / "src/navigation/dataset.py",
        ROOT / "src/navigation/causal_learning.py",
    ]
    parents = {"original": EpisodeCache(BASE), "approach": EpisodeCache(EXTRA)}
    pools = {
        "original": selected_episodes(parents["original"].manifest, guard["allowed_junctions"]),
        "approach": parents["approach"].manifest["episodes"],
    }
    total = sum(e["end"] - e["start"] for pool in pools.values() for e in pool)
    required = total * ((MAX_OBJECTS + 1) * (TOKEN_DIM * 4 + 1) + 1200)
    if shutil.disk_usage(args.out.parent).free < 25e9 + required:
        raise RuntimeError("Feature cache would violate25GB reserve")
    check_resources(args.out, 25)
    request = dict(
        parent_sha256=PARENT_SHA,
        base_manifest_sha256=BASE_SHA,
        extra_manifest_sha256=digest(EXTRA / "manifest.json"),
        bundle_sha256=digest(BUNDLE),
        protocol_sha256=digest(PROTOCOL),
        maximum_wall_seconds=3600,
        fact_cadence_hz=2.5,
        samples=total,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    for p in sources:
        target = args.out / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    atomic_json(args.out / "request.json", request)
    torch.set_num_threads(2)
    saved = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = make_student(saved["architecture"], saved["hidden"]).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    perception = FrozenSignalPerception(BUNDLE)
    started = time.monotonic()
    done = infer_count = 0
    manifest = dict(request_sha256=digest(args.out / "request.json"), domains={})

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 3600:
            raise InterruptedError("Cache STOP/wall cap; partial evidence retained")

    try:
        for domain, episodes in pools.items():
            source = parents[domain]
            root = args.out / domain
            root.mkdir()
            n = sum(e["end"] - e["start"] for e in episodes)
            specs = dict(
                recurrent=((192,), np.float32),
                stop_logits=((9,), np.float32),
                cruise_speeds=((9,), np.float32),
                signal_facts=((MAX_OBJECTS + 1, TOKEN_DIM), np.float32),
                signal_valid=((MAX_OBJECTS + 1,), bool),
            )
            specs.update(
                {
                    k: (a.shape[1:], a.dtype)
                    for k, a in source.arrays.items()
                    if k not in ("rgb", "front")
                }
            )
            arrays = {
                k: np.lib.format.open_memmap(
                    root / f"{k}.npy", mode="w+", dtype=dtype, shape=(n, *shape)
                )
                for k, (shape, dtype) in specs.items()
            }
            entries, offset = [], 0
            for e in episodes:
                check()
                if e["town"].removesuffix("_Opt") == "Town05" or e["partition"] != "train":
                    raise ValueError("Held-out data cannot enter cache")
                length = e["end"] - e["start"]
                for k, a in source.arrays.items():
                    if k not in ("rgb", "front"):
                        arrays[k][offset : offset + length] = a[e["start"] : e["end"]]
                hidden = None
                for start in range(0, length, 8):
                    check()
                    inputs = [
                        torch.from_numpy(
                            np.asarray(
                                source.arrays[k][
                                    e["start"] + start : min(e["end"], e["start"] + start + 8)
                                ]
                            ).copy()
                        ).cuda()[None]
                        for k in ("rgb", "front", "navigation", "ego")
                    ]
                    recurrent, hidden = model.encode_history(*inputs, hidden)
                    prediction = model.decode(recurrent)
                    for k, v in dict(
                        recurrent=recurrent,
                        stop_logits=prediction["stop_logits"],
                        cruise_speeds=prediction["cruise_speeds"],
                    ).items():
                        arrays[k][offset + start : offset + start + v.shape[1]] = v[0].cpu().numpy()
                facts = CausalSignalFacts(perception)
                local = 0
                shard_hashes = {}
                for shard in sorted(Path(e["data"]).glob("shard_*.zip")):
                    check()
                    shard_hashes[str(shard)] = digest(shard)
                    for record in shard_records(shard):
                        packet = load_policy_sample(shard, record)
                        # Bind cached navigation/ego to the exact lossless packet.
                        at = offset + local
                        if not np.array_equal(
                            packet.navigation_waypoints, arrays["navigation"][at]
                        ) or not np.array_equal(packet.ego_motion, arrays["ego"][at]):
                            raise ValueError("Raw packet and original cache misalignment")
                        tokens, valid = facts.step(packet)
                        arrays["signal_facts"][at], arrays["signal_valid"][at] = tokens, valid
                        infer_count += int(local % 4 == 0)
                        local += 1
                        done += 1
                        if local % 8 == 0:
                            check()
                            elapsed = time.monotonic() - started
                            atomic_json(
                                args.out / "progress.json",
                                dict(
                                    status="running",
                                    stage="M2 RGB fact cache",
                                    frames=done,
                                    target_frames=total,
                                    elapsed_s=elapsed,
                                    frames_per_s=done / elapsed,
                                    eta_seconds=(total - done) * elapsed / done,
                                    updated_at=time.time(),
                                ),
                            )
                    for a in arrays.values():
                        a.flush()
                if local != length:
                    raise ValueError("Raw/cache episode length mismatch")
                entries.append(
                    dict(
                        e,
                        source_start=e["start"],
                        source_end=e["end"],
                        start=offset,
                        end=offset + length,
                        shard_sha256=shard_hashes,
                    )
                )
                offset += length
            atomic_json(root / "manifest.json", dict(samples=n, episodes=entries))
            manifest["domains"][domain] = dict(
                manifest_sha256=digest(root / "manifest.json"),
                array_sha256={p.name: digest(p) for p in root.glob("*.npy")},
            )
        manifest.update(
            samples=done, detector_triplets=infer_count, elapsed_s=time.monotonic() - started
        )
        atomic_json(args.out / "manifest.json", manifest)
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="M2 RGB fact cache",
                frames=done,
                target_frames=total,
                elapsed_s=manifest["elapsed_s"],
                updated_at=time.time(),
            ),
        )
        print(json.dumps({k: manifest[k] for k in ("samples", "detector_triplets", "elapsed_s")}))
    except Exception as error:
        atomic_json(
            args.out / "progress.json",
            dict(
                status="stopped" if isinstance(error, InterruptedError) else "failed",
                error=str(error),
                frames=done,
                target_frames=total,
                updated_at=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/navigation_m2_signal_policy_cache_20261018"
    )
    with RunLock(ROOT / "outputs/.navigation.lock"):
        cache(parser.parse_args())
