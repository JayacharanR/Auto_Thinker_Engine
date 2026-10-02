"""
Phase 2 pipeline without the real dataset: preprocessing, the memmap
dataset, and short CPU training runs for both regularisers.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import preprocess_comma2k19 as prep  # noqa: E402
import train_phase2_jepa as phase2  # noqa: E402

from src.data.comma2k19_dataset import Comma2k19Dataset, split_segments  # noqa: E402


def _write_raw_segment(segment: Path, num_frames: int = 24, size=(64, 48)):
    """A comma2k19-style segment: encoded video plus CAN logs."""
    import av

    segment.mkdir(parents=True)
    with av.open(str(segment / "video.hevc"), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(num_frames):
            image = np.full((size[1], size[0], 3), i * 10, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    t = 100.0 + np.arange(0, num_frames / 20 + 1, 0.01)  # 100 Hz CAN on the log clock
    for name, value in (("steering_angle", t - 100.0), ("speed", 2 * (t - 100.0))):
        (segment / "processed_log" / "CAN" / name).mkdir(parents=True)
        np.save(segment / "processed_log" / "CAN" / name / "t.npy", t)
        np.save(segment / "processed_log" / "CAN" / name / "value.npy", value)
        for key in ("t", "value"):  # comma2k19 files have no extension
            (segment / "processed_log" / "CAN" / name / f"{key}.npy").rename(
                segment / "processed_log" / "CAN" / name / key
            )
    (segment / "global_pose").mkdir()
    np.save(segment / "global_pose" / "frame_times.npy", 100.0 + np.arange(num_frames) / 20)
    (segment / "global_pose" / "frame_times.npy").rename(segment / "global_pose" / "frame_times")


def test_preprocess_segment(tmp_path):
    segment = tmp_path / "raw" / "Chunk_1" / "abc|2018-01-01--00-00-00" / "3"
    _write_raw_segment(segment)
    assert prep.discover_segments(tmp_path / "raw") == [segment]

    meta = prep.process_segment((segment, tmp_path / "out", 2, 32))
    assert "error" not in meta, meta
    assert meta["num_frames"] == 12 and meta["time_source"] == "global_pose/frame_times"
    out = tmp_path / "out" / meta["id"]
    assert "|" not in meta["id"]
    frames = np.load(out / "frames.npy")
    telemetry = np.load(out / "telemetry.npy")
    assert frames.shape == (12, 32, 32, 3) and frames.dtype == np.uint8
    # Kept frames are source frames 0, 2, 4, ... at t = 0, 0.1, 0.2 s.
    np.testing.assert_allclose(telemetry[:, 0], np.arange(12) * 0.1, atol=1e-4)
    np.testing.assert_allclose(telemetry[:, 1], np.arange(12) * 0.2, atol=1e-4)
    # Re-running skips finished segments.
    assert prep.process_segment((segment, tmp_path / "out", 2, 32)) == meta


def make_processed_dataset(root: Path, num_segments=10, num_frames=30, size=32):
    rng = np.random.RandomState(0)
    segments = []
    for i in range(num_segments):
        seg_dir = root / f"seg{i:02d}"
        seg_dir.mkdir(parents=True)
        np.save(seg_dir / "frames.npy", rng.randint(0, 255, (num_frames, size, size, 3), np.uint8))
        telemetry = np.stack([rng.randn(num_frames) * 5 + 1, rng.rand(num_frames) * 30], 1)
        np.save(seg_dir / "telemetry.npy", telemetry.astype(np.float32))
        segments.append({"id": seg_dir.name, "num_frames": num_frames})
    (root / "manifest.json").write_text(json.dumps({"size": size, "segments": segments}))
    return root


def test_dataset_split_and_samples(tmp_path):
    root = make_processed_dataset(tmp_path / "data")
    train = Comma2k19Dataset(str(root), num_frames=8, split="train", sample_step=4)
    val = Comma2k19Dataset(str(root), num_frames=8, split="val", sample_step=4)
    assert not set(train.segment_ids) & set(val.segment_ids)
    assert len(train.segment_ids) == 9 and len(val.segment_ids) == 1
    assert len(train) == 9 * len(range(0, 30 - 8 + 1, 4))

    item = train[0]
    assert item["video"].shape == (3, 8, 32, 32)
    assert 0.0 <= item["video"].min() and item["video"].max() <= 1.0
    assert item["telemetry"].shape == (8, 2)
    # Both splits normalise with training-split statistics.
    np.testing.assert_allclose(train.telemetry_mean, val.telemetry_mean)

    frames = np.load(root / val.segment_ids[0] / "frames.npy")
    first = val[0]
    expected = torch.from_numpy(frames[0:8]).permute(3, 0, 1, 2).float() / 255
    torch.testing.assert_close(first["video"], expected)


def test_missing_manifest_names_preprocessor(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_comma2k19"):
        Comma2k19Dataset(str(tmp_path))


def test_split_is_deterministic():
    ids = [f"s{i}" for i in range(20)]
    assert split_segments(ids, "train", 0.8) == split_segments(list(reversed(ids)), "train", 0.8)


def tiny_config(tmp_path: Path, regularizer: str) -> dict:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "phase2_jepa_laptop.yaml").read_text())
    config = copy.deepcopy(config)
    config["experiment"].update(
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / f"ckpt_{regularizer}")
    )
    config["data"].update(dataset_root=str(tmp_path / "data"), num_workers=0, pin_memory=False)
    config["data"]["tubelet"].update(num_frames=4, spatial_size=32, sample_step=8)
    config["model"]["regularizer"] = regularizer
    config["model"]["context_encoder"].update(patch_size=8, embed_dim=48, depth=1, num_heads=2)
    config["model"]["predictor"].update(embed_dim=24, depth=1, num_heads=2)
    config["model"]["sigreg"].update(num_slices=16, max_samples=256)
    config["training"].update(batch_size=4, max_steps=4, warmup_steps=1, log_every=1)
    return config


@pytest.mark.parametrize("regularizer", ["ema", "sigreg"])
def test_short_training_and_resume(tmp_path, regularizer):
    make_processed_dataset(tmp_path / "data")
    config = tiny_config(tmp_path, regularizer)
    ckpt_dir = Path(config["experiment"]["checkpoint_dir"])

    phase2.train(config)
    latest = torch.load(ckpt_dir / "latest.pt", weights_only=False)
    assert (ckpt_dir / "best.pt").is_file()
    assert latest["global_step"] == 4 and latest["regularizer"] == regularizer
    assert ("target_encoder" in latest) == (regularizer == "ema")

    with pytest.raises(FileExistsError):
        phase2.train(config)

    config["training"]["max_steps"] = 6
    phase2.train(config, resume=True)
    assert torch.load(ckpt_dir / "latest.pt", weights_only=False)["global_step"] == 6


def test_sigreg_gradient_reaches_encoder_through_targets(tmp_path):
    """SIGReg mode trains the encoder through the full-clip targets as well."""
    config = tiny_config(tmp_path, "sigreg")
    encoder, predictor, target = phase2.build_models(config, "cpu")
    assert target is None
    geometry = phase2.patch_geometry(config)
    masks = phase2.MaskGenerator(
        geometry["total"], geometry["spatial"], geometry["temporal"], "tube", config["masking"]
    )(2)
    out = phase2.compute_losses(
        encoder, predictor, None, phase2.JEPALoss(detach_targets=False),
        phase2.SIGReg(num_slices=8), 0.05, torch.rand(2, 3, 4, 32, 32), torch.zeros(2, 4, 2),
        masks["context_indices"], masks["mask_indices"], step=0,
    )
    assert torch.isfinite(out["loss"]) and out["sigreg"] > 0
    out["loss"].backward()
    assert encoder.patch_embed.proj.weight.grad.abs().sum() > 0


def test_probe_uses_checkpoint_config(tmp_path, monkeypatch):
    import probe_phase2

    make_processed_dataset(tmp_path / "data")
    config = tiny_config(tmp_path, "ema")
    config["linear_probe"].update(probe_epochs=1, probe_batch_size=4)
    phase2.train(config)
    checkpoint = Path(config["experiment"]["checkpoint_dir"]) / "best.pt"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["probe", "--checkpoint", str(checkpoint)])
    probe_phase2.main()
    result = json.loads((tmp_path / "outputs/probe_results/probe_ckpt_ema_best_seed42.json")
                        .read_text())
    assert result["regularizer"] == "ema" and set(result["results"]) >= {"trained", "random"}


def test_ridge_probe_recovers_linear_signal_and_ignores_noise():
    from src.eval.linear_probe import fit_linear_probe

    rng = np.random.RandomState(0)
    w = rng.randn(32)
    train_x, val_x = rng.randn(2000, 32), rng.randn(500, 32)
    signal = fit_linear_probe(train_x, train_x @ w + 0.1 * rng.randn(2000),
                              val_x, val_x @ w + 0.1 * rng.randn(500))
    assert signal["r2"] > 0.99 and signal["alpha"] > 0
    noise = fit_linear_probe(train_x, rng.randn(2000), val_x, rng.randn(500))
    assert noise["r2"] < 0.05


def test_sigterm_saves_and_resumes(tmp_path, monkeypatch):
    """SIGTERM mid-run stops after the current step with a resumable checkpoint."""
    import os
    import signal

    make_processed_dataset(tmp_path / "data", num_frames=60)
    config = tiny_config(tmp_path, "ema")
    config["training"]["max_steps"] = 50
    ckpt_dir = Path(config["experiment"]["checkpoint_dir"])

    original = phase2.compute_losses
    calls = {"n": 0}

    def compute_and_maybe_stop(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)

    monkeypatch.setattr(phase2, "compute_losses", compute_and_maybe_stop)
    assert phase2.train(config) is False
    assert torch.load(ckpt_dir / "latest.pt", weights_only=False)["global_step"] == 3
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL

    monkeypatch.setattr(phase2, "compute_losses", original)
    assert phase2.train(config, resume=True) is True
    assert torch.load(ckpt_dir / "latest.pt", weights_only=False)["global_step"] == 50
