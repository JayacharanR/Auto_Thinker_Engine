#!/usr/bin/env python3
"""Fixed pretrained/RGB binary proposal verifiers; TRAIN-only fits, no native run."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_signal_context import score  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.signal_context import context_patches, label_proposals  # noqa:E402
from src.navigation.signal_diversity_data import DiverseSignalDataset  # noqa:E402
from src.navigation.signal_hard_pixels import protected_signal_metrics  # noqa:E402
from src.navigation.signal_pixel_study import NativeSignalGrid, point_candidates  # noqa:E402
from src.navigation.signal_pretrained_verifier import (  # noqa:E402
    FrozenSignalResNet,
    retain_original_states,
    rgb_features,
)
from src.navigation.signal_tight_verifier import TightSignalResNet  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def fit_worker(args):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    request = read(args.out.parent / "request.json")
    for file, sha in request["source_hashes"].items():
        if digest(ROOT / file) != sha:
            raise ValueError("Registered source changed before fitting")
    if (args.out / "STOP").exists():
        raise InterruptedError("OwnedfitSTOP")
    with np.load(args.out.parent / "train_features.npz") as arrays:
        x, y = arrays[args.arm].astype(np.float64), arrays["labels"]
    started = time.monotonic()
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            tol=1e-4,
            random_state=42,
        ).fit(z, y)
        train = model.predict_proba(z)[:, 1]
    if (args.out / "STOP").exists():
        raise InterruptedError("OwnedfitSTOPafterfit")
    np.savez_compressed(
        args.out / "head.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=model.coef_[0],
        intercept=model.intercept_,
        train_probability=train,
    )
    atomic_json(
        args.out / "fit.json",
        dict(
            complete=True,
            converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            warnings=[str(w.message) for w in caught],
            iterations=model.n_iter_.tolist(),
            train_examples=len(y),
            train_features=x.shape[1],
            elapsed_seconds=time.monotonic() - started,
            head_sha256=digest(args.out / "head.npz"),
            validation_used=False,
        ),
    )


@torch.inference_mode()
def encode(model, patches, check):
    semantic = []
    for start in range(0, len(patches), 128):
        check()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(torch.from_numpy(patches[start : start + 128]).cuda())
        semantic.append(y.to(torch.float16).cpu().numpy())
    return (
        np.concatenate(semantic) if semantic else np.empty((0, 1024), np.float16),
        rgb_features(patches).astype(np.float16),
    )


def probability(features, head):
    z = (features.astype(np.float64) - head["mean"]) / head["scale"]
    logits = z @ head["coefficient"] + head["intercept"][0]
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def main(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    parent_root = ROOT / "outputs/navigation_signal_pixel_study_20261021"
    parent = read(parent_root / "assessment.json")["arms"]["bulb"]
    if (
        parent["checkpoint_sha256"]
        != "27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193"
    ):
        raise ValueError("Exactoriginalproposalparentrequired")
    if digest(parent["selected_checkpoint"]) != parent["checkpoint_sha256"]:
        raise ValueError("Proposalcheckpointchanged")
    registry = ROOT / "outputs/navigation_models/navigation_backbones_20261019.json"
    descriptor = read(registry)["imagenet_resnet18"]
    if descriptor["sha256"] != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
        raise ValueError("ExactoriginalImageNetweightsrequired")
    files = [
        Path(__file__),
        registry,
        ROOT / "src/navigation/signal_pretrained_verifier.py",
        ROOT / "src/navigation/signal_context.py",
        ROOT / "src/navigation/signal_diversity_data.py",
        ROOT / "src/navigation/signal_pixel_study.py",
        ROOT / "src/navigation/signal_learning.py",
        ROOT / "src/navigation/signal_hard_pixels.py",
        ROOT / "scripts/train_navigation_signal_context.py",
        ROOT / "reports/navigation_signal_pretrained_verifier_protocol_20261021.md",
        parent_root / "assessment.json",
        parent_root / "bulb/predictions1000.json",
        ROOT / "outputs/navigation_signal_diversity_audit_20261021/manifest.json",
    ]
    if args.tight_crops:
        files.extend([
            ROOT / "src/navigation/signal_tight_verifier.py",
            ROOT / "reports/navigation_signal_tight_verifier_protocol_20261021.md",
            ROOT / "outputs/navigation_signal_pretrained_20261021/assessment.json",
            ROOT / "outputs/navigation_signal_pretrained_20261021/cache.json",
            ROOT / "outputs/navigation_signal_pretrained_20261021/train_rows.json",
        ])
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    for file in files:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    atomic_json(
        out / "request.json",
        dict(
            source_hashes=hashes,
            encoder=descriptor,
            proposal=parent,
            cache_seconds=1800,
            head_seconds=600,
            stage_seconds=3000,
            artifact_bytes=400_000_000,
            cpu_threads=2,
            no_native=True,
            crop_scales_native_pixels=[8, 16] if args.tight_crops else [32, 96],
            tight_crops=args.tight_crops,
        ),
    )
    status, error, phase, images_done = "running", None, "verifyingoriginalsources", 0
    fits_done = 0

    def check():
        if (out / "STOP").exists() or time.monotonic() - started > 3000:
            raise InterruptedError("OwnedSTOPor3000secondstagecap")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 400_000_000:
            raise InterruptedError("400MBartifactcap")

    def progress():
        atomic_json(
            out / "progress.json",
            dict(
                stage="Tight-crop signal verifier" if args.tight_crops
                else "Frozen pretrained signal verifier",
                status=status,
                error=error,
                phase=phase,
                frames=images_done,
                target_frames=2988,
                fits_done=fits_done,
                fits_total=2,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    try:
        with BudgetWatch(out):
            progress()
            cache_started = time.monotonic()
            data = DiverseSignalDataset(
                ROOT,
                ROOT / "outputs/navigation_signal_pixel_audit_20261021",
                ROOT / "outputs/navigation_signal_diversity_audit_20261021",
            )
            if len(data.train) != 2844 or len(data.validation) != 144:
                raise ValueError("Exact2844TRAIN/144VALrequired")
            val_peaks = {
                r["sample"]: r["peaks"] for r in read(parent_root / "bulb/predictions1000.json")
            }
            baseline = score(val_peaks, data)
            if any(
                baseline[k] != parent["selected_metrics"][k]
                for k in ("joint", "localized", "background_peaks", "towns", "classes")
            ):
                raise ValueError("Originalbaselinechanged")
            atomic_json(out / "baseline.json", baseline)
            atomic_json(
                out / "dataset.json",
                dict(samples=data.samples, train=data.train, validation=data.validation),
            )
            proposal = NativeSignalGrid().cuda().eval()
            proposal.load_state_dict(
                torch.load(parent["selected_checkpoint"], map_location="cuda", weights_only=False)[
                    "model"
                ]
            )
            encoder = (TightSignalResNet if args.tight_crops else FrozenSignalResNet)(
                descriptor
            ).cuda()
            encoder_before = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            rng = np.random.default_rng(42)
            train_peaks, rows, semantic, rgb, counts = {}, [], [], [], Counter()
            phase = "TRAINonlyproposalminingandfrozenRGBfeatures"

            def cache_check():
                check()
                if time.monotonic() - cache_started > 1800:
                    raise InterruptedError("1800secondfeaturecap")

            for start in range(0, len(data.train), 3):
                cache_check()
                indices = data.train[start : start + 3]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    probabilities = (
                        proposal(torch.from_numpy(data.images[indices]).cuda())
                        .float()
                        .softmax(1)
                        .cpu()
                        .numpy()
                    )
                for index, p in zip(indices, probabilities):
                    peaks = point_candidates(p)
                    train_peaks[index] = peaks
                    labels = label_proposals(peaks, data.boxes[index], data.background[index])
                    counts.update(
                        total_proposals=len(peaks),
                        positive_proposals=int((labels > 0).sum()),
                        background_proposals=int((labels == 0).sum()),
                        unknown_proposals=int((labels < 0).sum()),
                    )
                    selected = {(p["x"], p["y"], 1) for p, label in zip(peaks, labels) if label > 0}
                    negatives = np.flatnonzero(labels == 0)
                    chosen = list(negatives[:4])
                    if len(negatives) > 4:
                        chosen.extend(
                            rng.choice(
                                negatives[4:], min(12, len(negatives) - 4), replace=False
                            ).tolist()
                        )
                    selected.update((peaks[j]["x"], peaks[j]["y"], 0) for j in chosen)
                    for anchors in data.anchors["bulb"].values():
                        for sample, positions in anchors:
                            if sample == index:
                                for pick in np.linspace(
                                    0, len(positions) - 1, min(4, len(positions)), dtype=int
                                ):
                                    x, y = positions[pick]
                                    selected.add((int(x), int(y), 1))
                    bypoint = {}
                    for x, y, label in selected:
                        bypoint.setdefault((x, y), set()).add(label)
                    selected = sorted(
                        (x, y, next(iter(labels)))
                        for (x, y), labels in bypoint.items()
                        if len(labels) == 1
                    )
                    if selected:
                        patches = context_patches(
                            data.images[index].transpose(1, 2, 0),
                            np.array([[x, y] for x, y, _ in selected]),
                        )
                        a, b = encode(encoder, patches, cache_check)
                        semantic.append(a)
                        rgb.append(b)
                        rows.extend(
                            dict(sample=index, x=x, y=y, label=label, role="train")
                            for x, y, label in selected
                        )
                    images_done += 1
                if images_done % 30 == 0:
                    progress()
            train_a, train_b = np.concatenate(semantic), np.concatenate(rgb)
            train_y = np.array([r["label"] for r in rows], np.int64)
            np.savez_compressed(
                out / "train_features.npz", resnet=train_a, rgb=train_b, labels=train_y
            )
            atomic_json(out / "train_rows.json", rows)
            atomic_json(out / "train_proposals.json", train_peaks)
            del semantic, rgb, train_peaks, proposal
            semantic, rgb, offsets = [], [], [0]
            phase = "allfrozenVALproposalsfeatures; nolabelfitting"
            for index in data.validation:
                cache_check()
                patches = context_patches(
                    data.images[index].transpose(1, 2, 0),
                    np.array([[p["x"], p["y"]] for p in val_peaks[index]], np.int64),
                )
                a, b = encode(encoder, patches, cache_check)
                semantic.append(a)
                rgb.append(b)
                offsets.append(offsets[-1] + len(patches))
                images_done += 1
                if images_done % 12 == 0:
                    progress()
            val_a, val_b = np.concatenate(semantic), np.concatenate(rgb)
            np.savez_compressed(
                out / "validation_features.npz", resnet=val_a, rgb=val_b, offsets=offsets
            )
            if any(
                not torch.equal(v, encoder.state_dict()[k].detach().cpu())
                for k, v in encoder_before.items()
            ):
                raise ValueError("Frozenencoderweightorbufferchanged")
            atomic_json(
                out / "cache.json",
                dict(
                    train_images=2844,
                    validation_images=144,
                    train_examples=len(train_y),
                    positive_examples=int(train_y.sum()),
                    background_examples=int((train_y == 0).sum()),
                    validation_candidates=offsets[-1],
                    validation_offsets=offsets,
                    mined_counts=dict(counts),
                    train_features_sha256=digest(out / "train_features.npz"),
                    validation_features_sha256=digest(out / "validation_features.npz"),
                    encoder_frozen_all_tensors_verified=True,
                    elapsed_seconds=time.monotonic() - cache_started,
                ),
            )
            del encoder, encoder_before, semantic, rgb
            torch.cuda.empty_cache()
            if args.tight_crops:
                prior = ROOT / "outputs/navigation_signal_pretrained_20261021"
                prior_cache = read(prior / "cache.json")
                if rows != read(prior / "train_rows.json"):
                    raise ValueError("Paired scale comparison changed TRAIN points")
                for split, values in (("train", train_b), ("validation", val_b)):
                    path = prior / f"{split}_features.npz"
                    if digest(path) != prior_cache[f"{split}_features_sha256"]:
                        raise ValueError("Original feature cache changed")
                    with np.load(path) as old:
                        if not np.array_equal(values, old["rgb"]):
                            raise ValueError("Paired scale comparison changed RGB control")
                        if split == "train" and not np.array_equal(train_y, old["labels"]):
                            raise ValueError("Paired scale comparison changed TRAIN labels")
                        if split == "validation" and offsets != old["offsets"].tolist():
                            raise ValueError("Paired scale comparison changed VAL points")
                atomic_json(out / "paired_parity.json", dict(
                    train_points_labels_exact=True, rgb_features_exact=True,
                    validation_offsets_exact=True, only_resnet_crop_scale_changed=True,
                ))
            results = {}
            for arm in ("rgb", "resnet"):
                check()
                phase = f"{arm} TRAINonlylogisticfit"
                directory = out / arm
                directory.mkdir()
                with (directory / "worker.log").open("xb") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(Path(__file__).resolve()),
                            "--fit-worker",
                            "--arm",
                            arm,
                            "--out",
                            str(directory),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(
                            os.environ,
                            OMP_NUM_THREADS="2",
                            OPENBLAS_NUM_THREADS="2",
                            MKL_NUM_THREADS="2",
                        ),
                    )
                    atomic_json(
                        directory / "worker_identity.json",
                        dict(pid=child.pid, command=child.args, started_at=time.time()),
                    )
                    fit_started = time.monotonic()
                    try:
                        while child.poll() is None:
                            check()
                            if time.monotonic() - fit_started > 600:
                                raise InterruptedError("600secondfitcap")
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
                        raise RuntimeError(f"{arm}fitworkerexit{child.returncode}; retainfailure")
                head = dict(np.load(directory / "head.npz"))
                train_x = train_b if arm == "rgb" else train_a
                val_x = val_b if arm == "rgb" else val_a
                exported = probability(train_x, head)
                if not np.allclose(exported, head["train_probability"], rtol=0, atol=1e-12):
                    raise ValueError("Exportedlinearheadparityfailed")
                pred = probability(val_x, head)
                np.save(directory / "validation_probability.npy", pred)
                if args.tight_crops and arm == "rgb":
                    original_p = np.load(
                        ROOT / "outputs/navigation_signal_pretrained_20261021/rgb"
                        / "validation_probability.npy"
                    )
                    if not np.array_equal(pred, original_p):
                        raise ValueError("Repeated RGB control predictions changed")
                verified = {
                    index: retain_original_states(
                        val_peaks[index], pred[offsets[n] : offsets[n + 1]]
                    )
                    for n, index in enumerate(data.validation)
                }
                metrics = score(verified, data)
                fit = read(directory / "fit.json")
                criteria = dict(
                    converged=fit["converged"],
                    coverage_protected=protected_signal_metrics(metrics, baseline),
                    background_reduced90=metrics["background_peaks"]
                    <= 0.1 * baseline["background_peaks"],
                )
                results[arm] = dict(
                    metrics=metrics,
                    criteria=criteria,
                    component_pass=all(criteria.values()),
                    fit=fit,
                    linear_export_parity=True,
                    driving_qualified=False,
                )
                fits_done += 1
                atomic_json(out / "results.json", results)
                progress()
            for file, sha in hashes.items():
                if digest(ROOT / file) != sha:
                    raise ValueError("Registeredsourcechangedduringstudy")
            atomic_json(
                out / "assessment.json",
                dict(
                    complete=True,
                    arms=results,
                    baseline=baseline,
                    no_validation_tuning=True,
                    driving_qualified=False,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, phase = "complete", "twopredeclaredheadssettled; nodrivingpromotion"
    except Exception as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-worker", action="store_true")
    parser.add_argument("--arm", choices=("rgb", "resnet"))
    parser.add_argument("--tight-crops", action="store_true")
    args = parser.parse_args()
    fit_worker(args) if args.fit_worker else main(args)
