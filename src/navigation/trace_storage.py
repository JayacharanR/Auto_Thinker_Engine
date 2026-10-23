"""Lossless completed-trace storage with verified, reversible legacy paths."""

import gzip
import hashlib
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from src.navigation.runtime import atomic_json

BLOCK = 1024 * 1024


def stream_digest(stream):
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(BLOCK), b""):
        count += len(block)
        digest.update(block)
    return digest.hexdigest(), count


def archive_paths(path):
    path = Path(path)
    return path.with_name(path.name + ".gz"), path.with_name(path.name + ".archive.json")


def compress_trace(path, minimum_free_bytes=25_000_000_000, minimum_age_seconds=60):
    """Unlink only after archive round-trip SHA/size and durable receipt succeed."""
    import json

    path = Path(path)
    archive, receipt = archive_paths(path)
    if path.is_symlink():
        raise ValueError("Do not compress a materialized/foreign symlink")
    if not path.exists():
        if receipt.exists() and archive.exists():
            saved = json.loads(receipt.read_text())
            with archive.open("rb") as stream:
                if stream_digest(stream)[0] != saved["archive_sha256"]:
                    raise ValueError("Compressed trace changed")
            return saved
        raise FileNotFoundError(path)
    before = path.stat()
    if time.time() - before.st_mtime < minimum_age_seconds:
        raise ValueError("Trace was modified recently; may still be active")
    if shutil.disk_usage(path.parent).free < minimum_free_bytes + before.st_size + BLOCK:
        raise RuntimeError("Trace compression temporary would breach disk reserve")
    temporary = archive.with_name(archive.name + ".tmp")
    if archive.exists():
        with path.open("rb") as stream:
            original_sha, original_bytes = stream_digest(stream)
    else:
        if temporary.exists():
            raise FileExistsError("Interrupted archive temporary retained; inspect before retry")
        original = hashlib.sha256()
        original_bytes = 0
        with path.open("rb") as source, temporary.open("xb") as target:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=target, compresslevel=1, mtime=0
            ) as zipped:
                for block in iter(lambda: source.read(BLOCK), b""):
                    zipped.write(block)
                    original.update(block)
                    original_bytes += len(block)
            target.flush()
            os.fsync(target.fileno())
        original_sha = original.hexdigest()
        os.rename(temporary, archive)
    with gzip.open(archive, "rb") as stream:
        verified_sha, verified_bytes = stream_digest(stream)
    if (verified_sha, verified_bytes) != (original_sha, original_bytes):
        raise ValueError("Lossless round-trip failed; original trace retained")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Trace changed during compression; original retained")
    with archive.open("rb") as stream:
        archive_sha, archive_bytes = stream_digest(stream)
    saved = dict(
        schema="navigation-lossless-trace-v1",
        original_path=str(path.resolve()),
        original_sha256=original_sha,
        original_bytes=original_bytes,
        original_mtime_ns=before.st_mtime_ns,
        original_mode=before.st_mode & 0o777,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        archive_path=str(archive.resolve()),
        verified_roundtrip=True,
        timestamp=time.time(),
        restore="scripts/navigation_trace_storage.py materialize --root <run> -- <legacy command>",
    )
    atomic_json(receipt, saved)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
        path.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return saved


def restore_trace(receipt, minimum_free_bytes=25_000_000_000):
    """Restore original bytes/mode/mtime before removing redundant archive files."""
    import json

    receipt = Path(receipt)
    saved = json.loads(receipt.read_text())
    original, archive = Path(saved["original_path"]), Path(saved["archive_path"])
    if original.is_symlink():
        raise ValueError("Do not restore over a materialized/foreign symlink")
    with archive.open("rb") as stream:
        if stream_digest(stream)[0] != saved["archive_sha256"]:
            raise ValueError("Archive identity changed; preserve for inspection")
    if original.exists():
        with original.open("rb") as stream:
            if stream_digest(stream) != (saved["original_sha256"], saved["original_bytes"]):
                raise ValueError("Original path contains changed data; never overwrite")
    else:
        if (
            shutil.disk_usage(original.parent).free
            < minimum_free_bytes + saved["original_bytes"] + BLOCK
        ):
            raise RuntimeError("Restoration temporary would breach disk reserve")
        temporary = original.with_name(original.name + ".restore.tmp")
        if temporary.exists():
            raise FileExistsError("Interrupted restoration temporary retained")
        restored = hashlib.sha256()
        count = 0
        with gzip.open(archive, "rb") as source, temporary.open("xb") as target:
            for block in iter(lambda: source.read(BLOCK), b""):
                target.write(block)
                restored.update(block)
                count += len(block)
            target.flush()
            os.fsync(target.fileno())
        if (restored.hexdigest(), count) != (saved["original_sha256"], saved["original_bytes"]):
            raise ValueError("Restored bytes differ; archive retained")
        os.chmod(temporary, saved["original_mode"])
        os.utime(temporary, ns=(saved["original_mtime_ns"], saved["original_mtime_ns"]))
        os.link(temporary, original)
        temporary.unlink()
    directory = os.open(str(original.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
        archive.unlink()
        receipt.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(saved, original_restored=True, restored_at=time.time())


@contextmanager
def open_trace(path, mode="rt"):
    """New analysis can stream original logical bytes without materializing."""
    import json

    path = Path(path)
    if path.exists():
        with path.open(mode) as stream:
            yield stream
        return
    archive, receipt = archive_paths(path)
    if not receipt.exists():
        raise FileNotFoundError(f"No original or verified trace archive:{path}")
    saved = json.loads(receipt.read_text())
    with archive.open("rb") as stream:
        if stream_digest(stream)[0] != saved["archive_sha256"]:
            raise ValueError("Compressed trace hash changed")
    with gzip.open(archive, mode) as stream:
        yield stream


@contextmanager
def materialized_traces(receipts, temporary_root="/tmp", minimum_free_ram_bytes=12_000_000_000):
    """Provide exact legacy paths using temporary raw bytes; retain durable archives."""
    import json

    import psutil

    entries = [json.loads(Path(p).read_text()) for p in receipts]
    needed = sum(e["original_bytes"] for e in entries if not Path(e["original_path"]).exists())
    if (
        shutil.disk_usage(temporary_root).free < needed + 100_000_000
        or psutil.virtual_memory().available < needed + minimum_free_ram_bytes
    ):
        raise RuntimeError("Insufficient tmpfs/RAM for exact legacy-path materialization")
    created = []
    with tempfile.TemporaryDirectory(
        prefix="auto_thinker_traces_", dir=temporary_root
    ) as directory:
        try:
            for entry in entries:
                original = Path(entry["original_path"])
                if original.exists():
                    continue
                if original.is_symlink():
                    target = Path(os.readlink(original))
                    if not str(target).startswith(
                        str(Path(temporary_root) / "auto_thinker_traces_")
                    ):
                        raise ValueError("Foreign broken link at a legacy trace path")
                    original.unlink()
                target = Path(directory) / hashlib.sha256(str(original).encode()).hexdigest()
                archive = Path(entry["archive_path"])
                with archive.open("rb") as stream:
                    if stream_digest(stream)[0] != entry["archive_sha256"]:
                        raise ValueError("Archive identity changed; do not restore")
                digest = hashlib.sha256()
                count = 0
                with gzip.open(archive, "rb") as source, target.open("xb") as destination:
                    for block in iter(lambda: source.read(BLOCK), b""):
                        destination.write(block)
                        digest.update(block)
                        count += len(block)
                if (digest.hexdigest(), count) != (
                    entry["original_sha256"],
                    entry["original_bytes"],
                ):
                    raise ValueError("Restored trace bytes do not match receipt")
                os.chmod(target, entry["original_mode"])
                os.utime(target, ns=(entry["original_mtime_ns"], entry["original_mtime_ns"]))
                original.symlink_to(target)
                created.append((original, target))
            yield entries
        finally:
            for original, target in reversed(created):
                if original.is_symlink() and Path(os.readlink(original)) == target:
                    original.unlink()
