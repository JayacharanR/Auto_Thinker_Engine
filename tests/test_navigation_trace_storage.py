import gzip
import json

import pytest

from src.navigation.trace_storage import (
    archive_paths,
    compress_trace,
    materialized_traces,
    open_trace,
    restore_trace,
)


def test_verified_trace_roundtrip_stream_and_legacy_paths(tmp_path):
    path = tmp_path / "world_trace.jsonl"
    content = b'{"frame":1,"actors":[]}\n' * 100
    path.write_bytes(content)
    receipt = compress_trace(path, minimum_free_bytes=0, minimum_age_seconds=0)
    assert receipt["verified_roundtrip"] and not path.exists()
    archive, record = archive_paths(path)
    with open_trace(path, "rb") as stream:
        assert stream.read() == content
    with materialized_traces([record], temporary_root=tmp_path, minimum_free_ram_bytes=0):
        assert path.is_symlink() and path.read_bytes() == content
    assert not path.exists() and archive.exists() and record.exists()
    assert (
        compress_trace(path, minimum_free_bytes=0)["original_sha256"] == receipt["original_sha256"]
    )


def test_bad_existing_archive_never_removes_original(tmp_path):
    path = tmp_path / "world_trace.jsonl"
    path.write_bytes(b"original")
    archive, receipt = archive_paths(path)
    with gzip.open(archive, "wb") as stream:
        stream.write(b"different")
    with pytest.raises(ValueError, match="round-trip"):
        compress_trace(path, minimum_free_bytes=0, minimum_age_seconds=0)
    assert path.read_bytes() == b"original" and not receipt.exists()


def test_recent_trace_and_insufficient_restore_space_are_rejected(tmp_path):
    path = tmp_path / "world_trace.jsonl"
    path.write_bytes(b"source")
    with pytest.raises(ValueError, match="recently"):
        compress_trace(path, minimum_free_bytes=0)
    compress_trace(path, minimum_free_bytes=0, minimum_age_seconds=0)
    _, receipt = archive_paths(path)
    with pytest.raises(RuntimeError, match="Insufficient"):
        with materialized_traces([receipt], temporary_root=tmp_path, minimum_free_ram_bytes=10**20):
            pass


def test_changed_archive_never_materializes_bad_bytes(tmp_path):
    path = tmp_path / "world_trace.jsonl"
    path.write_bytes(b"truth")
    compress_trace(path, minimum_free_bytes=0, minimum_age_seconds=0)
    archive, receipt = archive_paths(path)
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        with materialized_traces([receipt], temporary_root=tmp_path, minimum_free_ram_bytes=0):
            pass
    assert not path.exists()
    assert json.loads(receipt.read_text())["original_bytes"] == 5


def test_restoration_returns_exact_original_layout_and_mtime(tmp_path):
    path = tmp_path / "world_trace.jsonl"
    path.write_bytes(b"exact original bytes\n" * 100)
    before = path.stat()
    data = path.read_bytes()
    compress_trace(path, minimum_free_bytes=0, minimum_age_seconds=0)
    archive, receipt = archive_paths(path)
    restored = restore_trace(receipt, minimum_free_bytes=0)
    assert restored["original_restored"] and path.read_bytes() == data
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert not archive.exists() and not receipt.exists()


def test_restoration_never_overwrites_a_changed_original(tmp_path):
    path = tmp_path / "world_trace.jsonl"
    path.write_bytes(b"truth")
    compress_trace(path, minimum_free_bytes=0, minimum_age_seconds=0)
    archive, receipt = archive_paths(path)
    path.write_bytes(b"new user data")
    with pytest.raises(ValueError, match="never overwrite"):
        restore_trace(receipt, minimum_free_bytes=0)
    assert path.read_bytes() == b"new user data" and archive.exists() and receipt.exists()
