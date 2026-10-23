#!/usr/bin/env python3
"""Read-only inventory of identical aligned tensor bytes in historical checkpoints."""

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve each range inventory")
    excluded = set()
    for receipt in [
        "navigation_checkpoint_cow_20261019",
        "navigation_checkpoint_cow_more_20261020",
    ]:
        for r in json.loads((ROOT / "outputs" / receipt / "summary.json").read_text())["records"]:
            excluded.add(Path(r["target"]).resolve())
    files = [
        f
        for f in (ROOT / "outputs").rglob("*.pt")
        if f.is_file()
        and not f.is_symlink()
        and f.stat().st_size > 50_000_000
        and f.resolve() not in excluded
        and "navigation_repaired_study_v4_" not in str(f)
    ]
    groups, identities, errors = defaultdict(list), {}, []
    for path in files:
        try:
            before = path.stat()
            with zipfile.ZipFile(path) as z, path.open("rb") as raw:
                for entry in z.infolist():
                    if entry.file_size < 262144 or entry.compress_type != zipfile.ZIP_STORED:
                        continue
                    raw.seek(entry.header_offset + 26)
                    namesize, extrasize = struct.unpack("<HH", raw.read(4))
                    offset = entry.header_offset + 30 + namesize + extrasize
                    trim = (-offset) % 4096
                    length = ((entry.file_size - trim) // 4096) * 4096
                    if length < 262144:
                        continue
                    raw.seek(offset + trim)
                    sha, remaining = hashlib.sha256(), length
                    while remaining:
                        chunk = raw.read(min(remaining, 8 << 20))
                        if not chunk:
                            raise ValueError("Partial tensor storage")
                        sha.update(chunk)
                        remaining -= len(chunk)
                    groups[(length, sha.hexdigest())].append(
                        dict(
                            path=str(path.relative_to(ROOT)),
                            offset=offset + trim,
                            bytes=length,
                            tensor_storage=entry.filename,
                            original_data_offset=offset,
                        )
                    )
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Historical checkpoint changed during read-only inventory")
            identities[str(path.relative_to(ROOT))] = dict(
                bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest(path)
            )
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    duplicates = [dict(bytes=n, sha256=h, ranges=r) for (n, h), r in groups.items() if len(r) > 1]
    result = dict(
        read_only=True,
        source_sha256=digest(__file__),
        files=identities,
        groups=duplicates,
        errors=errors,
        logical_duplicate_range_bytes=sum(g["bytes"] * (len(g["ranges"]) - 1) for g in duplicates),
        note="Identical4KiB-aligned bytes only; logical potential does not promise physical savings.",
    )
    atomic_json(args.out, result)
    print(
        json.dumps(
            dict(
                files=len(identities),
                groups=len(duplicates),
                errors=errors,
                logical_potential_gb=result["logical_duplicate_range_bytes"] / 1e9,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
