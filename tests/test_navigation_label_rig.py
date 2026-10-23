"""Raw semantic/depth decoding, explicit label separation and atomic byte preservation."""

import io
import json
import zipfile

import numpy as np
import pytest

from scripts.navigation_label_rig import LabelShards, decode_depth, decode_instances


def test_depth_channel_order_and_semantic_render_instance_are_lossless():
    raw = np.array([[[0, 0, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]]], np.uint8)
    depth = decode_depth(raw)
    assert depth.dtype == np.float32
    assert depth[0, 0] == 0 and depth[0, 2] == 1000
    assert np.isclose(depth[0, 1], 1000 * 255 / 16777215)
    raw = np.array([[[55, 20, 14, 255], [1, 2, 7, 255]]], np.uint8)
    semantic, component = decode_instances(raw)
    assert semantic.tolist() == [[14, 7]] and component.tolist() == [[[20, 55], [2, 1]]]
    with pytest.raises(ValueError):
        decode_depth(raw[..., :3])


def test_labels_remain_separate_and_committed_shard_survives_truncated_next_part(tmp_path):
    rgb = {v: np.zeros((288, 512, 3), np.uint8) for v in ("front", "left", "right")}
    labels = {
        f"{v}_{kind}": np.full((288, 512, 4), 7, np.uint8)
        for v in rgb
        for kind in ("instance", "depth")
    }
    writer = LabelShards(tmp_path / "labels", labels=True, samples_per_shard=1)
    writer.append(10, 0.5, rgb, labels)
    shard = next(writer.root.glob("*.zip"))
    with zipfile.ZipFile(shard) as archive:
        meta = json.loads(archive.read("00000000/metadata.json"))
        assert meta["privileged_labels_not_policy_inputs"] is True
        assert len(meta["pixel_sha256"]) == 9
        assert np.array_equal(
            np.load(io.BytesIO(archive.read("00000000/front_instance.npy")), allow_pickle=False),
            labels["front_instance"],
        )
    (writer.root / "shard_000001.zip.part").write_bytes(b"truncated uncommitted fixture")
    assert shard.exists() and len(writer.shards) == 1
    baseline = LabelShards(tmp_path / "rgb", labels=False)
    with pytest.raises(ValueError, match="privileged label"):
        baseline.append(10, 0.5, rgb, labels)
    baseline.append(10, 0.5, rgb, {})
    baseline.close()
    assert baseline.count == 1 and not list(baseline.root.glob("*.part"))
    with pytest.raises(FileExistsError):
        LabelShards(tmp_path / "labels", labels=True)
