#!/usr/bin/env python3
"""Release a registered semantic TRAIN subset, preserving original mining/veto."""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def semantic_crop_check(tags, source_rgb, fresh_rgb, box):
    x0, y0, x1, y1 = box
    padded = tags[max(0, y0 - 2) : min(288, y1 + 2), max(0, x0 - 2) : min(512, x1 + 2)]
    values, counts = np.unique(padded, return_counts=True)
    mae = float(
        np.abs(
            source_rgb[y0:y1, x0:x1].astype(float) - fresh_rgb[y0:y1, x0:x1].astype(float)
        ).mean()
    )
    return dict(
        approved=bool(not np.isin(padded, [0, 7]).any() and mae <= 10),
        padded_tag_counts={str(int(k)): int(v) for k, v in zip(values, counts)},
        fresh_source_rgb_mae=mae,
    )


def verify(args):
    if args.out.exists():
        raise FileExistsError("Fresh verification output required")
    manifest = json.loads((args.mining / "manifest.json").read_text())
    corpus_audit = json.loads((args.corpus / "data_audit.json").read_text())
    if digest(args.corpus / "data_audit.json") != manifest["corpus_audit_sha256"]:
        raise ValueError("Original corpus audit changed")
    evidence, checked, rows, allowed, groups, towns = {}, {}, [], [], set(), Counter()
    canonical = [(r, n) for r in manifest["rows"] for n in r["negatives"]]
    for index, (row, negative) in enumerate(canonical):
        sample = row["sample"]
        if sample["role"] != "train":
            raise ValueError("Non-TRAIN mining source")
        town = sample["town"].removesuffix("_Opt").lower()
        town_root = ROOT / f"outputs/navigation_m2_negative_semantics_{town}_20261016"
        directory = town_root / f"case{sample['case_id']}"
        key = sample["town"], sample["case_id"]
        if key not in checked:
            request = json.loads((town_root / "request.json").read_text())
            for path, expected in request["source_hashes"].items():
                if digest(town_root / "source" / path) != expected:
                    raise ValueError("Semantic worker archive changed")
            batch = json.loads((town_root / "pixel_batch.json").read_text())
            if batch["mining_manifest_sha256"] != digest(args.mining / "manifest.json"):
                raise ValueError("Semantic selection changed")
            meta = json.loads((directory / "pixel_probe.json").read_text())
            if (
                meta["case"]["route_id"] != sample["case_id"]
                or meta["semantic_enum"]["TrafficLight"] != 7
            ):
                raise ValueError("Semantic case or tag mapping changed")
            if len({s["frame"] for s in meta["sensors"].values()}) != 1:
                raise ValueError("RGB/semantic frame mismatch")
            if max(s["matrix_error"] for s in meta["sensors"].values()) > 1e-3:
                raise ValueError("Semantic projection pose mismatch")
            for view in ("front", "left", "right"):
                if meta["optics"][f"rgb_{view}"] != meta["optics"][f"semantic_{view}"]:
                    raise ValueError("Semantic/RGB optics mismatch")
            checked[key] = meta
            for path in (
                directory / "pixel_probe.json",
                town_root / "request.json",
                town_root / "geometry.json",
            ):
                evidence[str(path.relative_to(ROOT))] = digest(path)
        source_path = args.corpus / sample["path"]
        if digest(source_path) != corpus_audit["image_sha256"][sample["path"]]:
            raise ValueError("Original RGB changed")
        fresh_path = directory / f"probe_rgb_{sample['view']}.png"
        tags_path = directory / f"probe_tags_{sample['view']}.png"
        for path in (source_path, fresh_path, tags_path):
            evidence[str(path.resolve().relative_to(ROOT))] = digest(path)
        tags = np.array(Image.open(tags_path))
        source, fresh = (np.array(Image.open(p).convert("RGB")) for p in (source_path, fresh_path))
        if tags.shape != (288, 512) or source.shape != fresh.shape or source.shape != (288, 512, 3):
            raise ValueError("Native pixel dimensions changed")
        result = semantic_crop_check(tags, source, fresh, negative["crop_xyxy"])
        rows.append(dict(candidate=index, sample=sample, **result))
        if result["approved"]:
            allowed.append(index)
            groups.add((sample["town"], sample["junction_id"]))
            towns[sample["town"]] += 1
    passed = len(allowed) >= 20 and len(groups) >= 6 and len(towns) >= 3
    args.out.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(),
        ROOT / "reports/navigation_m2_signal_negative_semantics_protocol_20261016.md",
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    release = dict(
        approved=passed,
        manifest_sha256=digest(args.mining / "manifest.json"),
        approved_candidate_indices=allowed,
        canonical_approved=len(allowed),
        canonical_rejected=len(canonical) - len(allowed),
        junctions=sorted(groups),
        towns=dict(towns),
        semantic_evidence_sha256=evidence,
        source_hashes={str(s.relative_to(ROOT)): digest(s) for s in sources},
        scope="TRAIN-only semantic verification; additional labels, not driving evidence",
    )
    atomic_json(args.out / "release.json", release)
    atomic_json(args.out / "candidate_checks.json", dict(rows=rows))
    omit = ["semantic_evidence_sha256", "source_hashes", "approved_candidate_indices"]
    print(json.dumps({k: v for k, v in release.items() if k not in omit}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mining", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    verify(parser.parse_args())
