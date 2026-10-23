#!/usr/bin/env python3
"""Re-audit closed branch evidence without rewriting provisional collector results."""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.recorder_actions import align_world_frames, vehicle_frames  # noqa:E402
from src.navigation.recorder_collisions import recorder_contacts  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(root, out):
    if out.exists():
        raise FileExistsError("Preserve previous independent branch audits")
    if read(root / "progress.json")["status"] != "complete":
        raise ValueError("Only terminal collections can be independently re-audited")
    out.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "src/navigation/recorder_actions.py",
        ROOT / "src/navigation/recorder_collisions.py",
        ROOT / "src/navigation/dataset.py",
    ]
    for source in sources:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    request = read(root / "request.json")
    if not all(digest(root / "source" / p) == h for p, h in request["source_hashes"].items()):
        raise ValueError("Collection's archived source identity changed")
    records, errors = [], []
    for row in read(root / "records.json"):
        name = f"repeat{row['repeat']}_{row['branch']}"
        episode = root / name
        try:
            world = lines(episode / "world_trace.jsonl")
            actions = lines(episode / "actions.jsonl")
            frames = [w["frame"] for w in world]
            if (
                len(frames) != 160
                or frames != [a["frame"] for a in actions]
                or not all(b == a + 1 for a, b in zip(frames, frames[1:]))
            ):
                raise ValueError("Expected160 contiguous aligned world/control frames")
            hero = next(a["id"] for a in world[0]["actors"] if a["role"] == "hero")
            native, alignment = align_world_frames(
                vehicle_frames(episode / "native.log", hero), world
            )
            contacts = recorder_contacts(episode / "native.log")
            if not contacts["complete"]:
                raise ValueError("Closed native recorder is incomplete")
            differences = []
            for action in actions:
                actual = native[action["frame"] + 1]["control"]
                if not np.allclose(actual, action["requested_control"], atol=1e-5, rtol=0):
                    differences.append(action["frame"])
            sample_frames, shard_hashes = [], {}
            censored = 0
            for shard in sorted((episode / "data").glob("shard_*.zip")):
                shard_hashes[str(shard)] = digest(shard)
                with zipfile.ZipFile(shard) as archive:
                    for sample in shard_records(shard):
                        packet = load_policy_sample(shard, sample)  # Verify every RGB pixel hash.
                        target = json.loads(archive.read(sample["prefix"] + "/targets.json"))
                        ack = target["facts"]["actuator_feedback_frame"]
                        if ack != packet.frame + 1 or not np.allclose(
                            native[ack]["control"], target["executed_control"], atol=1e-5, rtol=0
                        ):
                            raise ValueError(
                                "Shard's executed control lacks exact next-tick "
                                "native acknowledgment"
                            )
                        if target["facts"]["imitation_eligible"] is not False:
                            raise ValueError(
                                "Alternative actions must never be clean imitation data"
                            )
                        sample_frames.append(packet.frame)
                        censored += int(target["intervention"] is not None)
            if sample_frames != frames[::2] or len(sample_frames) != 80 or differences:
                raise ValueError(
                    "Expected80 frame-exact samples and zero actual-control differences"
                )
            identity = read(episode / "world_identity.json")
            if (
                identity["actual_town"] != request["entry"]["town"]
                or contacts["town"] != identity["actual_town"]
            ):
                raise ValueError("Actual map identity mismatch")
            comparisons = row["observable_comparisons"]
            records.append(
                dict(
                    name=name,
                    samples=80,
                    native_controls_checked=160,
                    alignment=alignment,
                    control_differences=0,
                    all_rgb_pixel_hashes_verified=True,
                    closed_recorder_complete=True,
                    recorder_sha256=contacts["sha256"],
                    provisional_recorder_complete=row["recorder"]["complete"],
                    provisional_recorder_sha256=row["recorder"]["sha256"],
                    unfiltered_contact_starts=contacts["ego_contact_starts"],
                    contact_times_s=[s["time_s"] for s in contacts["starts"]],
                    assisted_rgb_samples=censored,
                    world_identity=identity,
                    evidence_sha256={
                        str(episode / p): digest(episode / p)
                        for p in (
                            "world_trace.jsonl",
                            "actions.jsonl",
                            "outcome.json",
                            "world_identity.json",
                        )
                    },
                    shard_sha256=shard_hashes,
                    observable_comparisons=comparisons,
                )
            )
        except Exception as error:
            errors.append(dict(name=name, error=str(error)))
        atomic_json(
            out / "progress.json",
            dict(
                status="running",
                frames=len(records) + len(errors),
                target_frames=6,
                stage="M5 closed branch evidence audit",
                updated_at=time.time(),
            ),
        )
    comparisons = [c for r in records for c in r["observable_comparisons"]]
    comparable = bool(comparisons) and all(
        c["observable_prefix_within_tolerance"] for c in comparisons
    )
    limitations = [
        "Pilot-only stationary lead; no moving/crossing-vehicle coverage or red positives.",
        "Post-contact privileged stops censor alternative-action tails.",
    ]
    if any(r["provisional_recorder_sha256"] != r["recorder_sha256"] for r in records):
        limitations.append(
            "Collector recorder hashes preceded final flush; "
            "original outcomes remain intact and closed-file identities are authoritative."
        )
    if not comparable:
        limitations.append(
            "Registered observable replay comparisons fail; "
            "do not claim deterministic counterfactual calibration."
        )
    result = dict(
        status="complete" if len(records) == 6 and not errors else "complete_with_failures",
        records=records,
        errors=errors,
        samples=sum(r["samples"] for r in records),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
        collection_request_sha256=digest(root / "request.json"),
        collection_records_sha256=digest(root / "records.json"),
        unfiltered_contact_starts=sum(r["unfiltered_contact_starts"] for r in records),
        all_registered_observable_comparisons_pass=comparable,
        physical_replay_comparability_verified=comparable,
        calibration_qualified=False,
        no_model_updates=True,
        imitation_eligible=False,
        limitations=limitations,
    )
    atomic_json(out / "assessment.json", result)
    atomic_json(
        out / "progress.json",
        dict(
            status=result["status"],
            stage="M5 closed branch evidence audit",
            frames=len(records) + len(errors),
            target_frames=6,
            updated_at=time.time(),
        ),
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("records", "source_hashes")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.out.resolve())
