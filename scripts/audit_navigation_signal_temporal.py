#!/usr/bin/env python3
"""One bounded, TRAIN-only causal replay of the frozen signal perception bundle."""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_navigation_rule_observability import (  # noqa: E402
    validate_training_data,
    verify_sources,
)
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa: E402
from src.navigation.rule_geometry import stopped_runs  # noqa: E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa: E402
from src.navigation.signal_learning import junction_roles  # noqa: E402
from src.navigation.signal_perception import FrozenSignalPerception  # noqa: E402
from src.navigation.signal_temporal_audit import (  # noqa: E402
    episode_scope,
    score_frame,
    summarize_frames,
    transition_delays,
)
from src.navigation.signal_tracking import SignalTracker  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    corpus = ROOT / "outputs/navigation_m2_signal_corpus_v2_20261016"
    if digest(corpus / "data_audit.json") != (
        "ec1632bab904092d595371b0b3d1cdb2c49c5e0caa2b65cbce58af4005844a48"
    ):
        raise ValueError("Original component corpus identity changed")
    plan = read(corpus / "plan.json")
    if digest(corpus / "plan.json") != read(corpus / "data_audit.json")["plan_sha256"]:
        raise ValueError("Component plan changed")
    verify_sources(corpus, read(corpus / "request.json"))
    roles = junction_roles(plan["cases"])
    assessment_path = ROOT / "reports/navigation_m2_dagger_assessment_20261016.json"
    assessment = read(assessment_path)
    audit_root = ROOT / "outputs/navigation_m2_rule_audit_20261016"
    audit = read(audit_root / "assessment.json")
    if (
        not audit["audit_complete"]
        or digest(assessment_path) != audit["provenance"]["assessment_sha256"]
        or digest(audit_root / "samples.jsonl") != audit["provenance"]["samples_sha256"]
    ):
        raise ValueError("Original projection/correction audit changed")
    verify_sources(audit_root, audit["provenance"])
    all_rows = [
        json.loads(line) for line in (audit_root / "samples.jsonl").read_text().splitlines()
    ]
    paths = [Path(p) for p in assessment["eligible_correction_paths"]]
    included, excluded = [], []
    for path in paths:
        meta = validate_training_data(path, assessment)
        guard = read(path.parents[3] / "guard.json")
        scope = episode_scope(meta, guard, roles)
        rows = [r for r in all_rows if r["route_id"] == meta["route_id"]]
        if not rows or any(Path(r["shard"]).parent != path for r in rows):
            raise ValueError("Projection rows do not match episode identity")
        for a, b in zip(rows, rows[1:]):
            if b["frame"] != a["frame"] + 2 or abs(b["timestamp"] - a["timestamp"] - 0.1) > 1e-5:
                raise ValueError("Noncontiguous episode projection audit")
        by_shard = {}
        for row in rows:
            by_shard.setdefault(row["shard"], []).append(row["record"])
        for shard, records in by_shard.items():
            if list(shard_records(shard)) != records:
                raise ValueError("Committed shard/label/media manifest differs from original audit")
        item = dict(
            route_id=meta["route_id"],
            family=meta["family"],
            data=str(path),
            junction_ids=meta["junction_ids"],
            samples=len(rows),
            **scope,
        )
        if scope["eligible"]:
            item["rows"] = rows
            included.append(item)
        else:
            excluded.append(item)
    if {e["route_id"] for e in included} != {86, 101, 123, 143}:
        raise ValueError("Fixed TRAIN replay selection changed")
    selected = [e for e in included if any(r["signal"] for r in e["rows"])]
    if {e["route_id"] for e in selected} != {101, 123} or sum(
        e["samples"] for e in selected
    ) != 518:
        raise ValueError("Fixed complete signal episodes changed")
    return included, excluded, selected


def event_audit(episode, evaluated):
    rows = episode["rows"]
    by_frame = {r["frame"]: r for r in evaluated}

    def evidence(frame):
        r = by_frame.get(frame)
        if r is None:
            return None
        return dict(
            label_state=r["next_light_state"],
            matched_housings=len(r["scored"]["matches"]),
            raw_states=[m["raw_state"] for m in r["scored"]["matches"]],
            tracked_states=[m["tracked_state"] for m in r["scored"]["matches"]],
        )

    events = []
    for event in stopped_runs(rows):
        start, end = event["start_frame"], event["end_frame"]
        during = [r for r in evaluated if start <= r["frame"] <= end]
        events.append(
            dict(
                **event,
                left_censored=start == rows[0]["frame"],
                at_start=evidence(start),
                at_release=evidence(event["release_frame"]),
                matched_housing_frames_during_stop=sum(
                    bool(r["scored"]["matches"]) for r in during
                ),
                rgb_replayed=bool(evaluated),
            )
        )
    return dict(
        route_id=episode["route_id"],
        family=episode["family"],
        samples=episode["samples"],
        stop_events=events,
    )


def plot(episodes, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(episodes), 1, figsize=(12, 3.5 * len(episodes)), constrained_layout=True
    )
    for ax, episode in zip(axes, episodes):
        rows = episode["rows"]
        t = [r["timestamp"] for r in rows]
        ax.plot(t, [r["teacher_target_speed_mps"] for r in rows], label="Teacher target m/s")
        ax.plot(t, [r["ego_speed_mps"] for r in rows], label="Assisted actual speed m/s")
        states = ax.twinx()
        for field, marker, label in (
            ("raw_state", "x", "Raw matched RGB"),
            ("tracked_state", ".", "Confident tracked RGB"),
        ):
            for name, level, color in (
                ("Red", -1, "#c22"),
                ("Yellow", 0, "#b90"),
                ("Green", 1, "#292"),
            ):
                times = [
                    r["timestamp"]
                    for r in rows
                    if any(m[field] == name for m in r["scored"]["matches"])
                ]
                states.scatter(
                    times,
                    [level + (0.1 if marker == "." else -0.1)] * len(times),
                    marker=marker,
                    s=9,
                    color=color,
                    label=label if name == "Red" else None,
                )
        states.set(yticks=[-1, 0, 1], yticklabels=["Red", "Yellow", "Green"], ylim=(-1.6, 1.6))
        for row in rows:
            if row["next_light_state"] == "Red":
                ax.axvspan(row["timestamp"], row["timestamp"] + 0.1, color="red", alpha=0.04)
        ax.set(
            title=f"TRAIN route {episode['route_id']} | oracle association for scoring ONLY",
            ylabel="m/s",
            xlabel="Episode time (s)",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
        states.legend(loc="upper right", fontsize=8)
    figure.suptitle("Causal replay: existing assisted TRAIN images; not driving performance")
    figure.savefig(out / "timeline.png", dpi=140)
    plt.close(figure)


def run(args):
    if args.out.exists():
        raise FileExistsError("Fresh temporal audit output required")
    started = time.monotonic()
    included, excluded, selected = prepare()
    bundle_path = (
        ROOT / "outputs/navigation_m2_signal_state_expansion_20261016/perception_bundle.json"
    )
    bundle = read(bundle_path)
    if (
        bundle["detector_sha256"]
        != "2dbd3e56c7de29c5f1b79c284f7fe0f704ec659e43b0a4885583ca2f108ee507"
        or bundle["state_sha256"]
        != "f4171de1e652e1f53b64089145e38dacd4a353199a4a097c3712d6c2259c17c4"
    ):
        raise ValueError("Fixed selected perception bundle changed")
    check_resources(args.out, 25)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/audit_navigation_rule_observability.py",
        ROOT / "reports/navigation_m2_signal_temporal_protocol_20261016.md",
        *sorted((ROOT / "src/navigation").glob("*.py")),
    ]
    for source in sources:
        target = args.out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(
        args.out / "request.json",
        dict(
            kind="Fixed TRAIN temporal replay, no fitting or driver integration",
            source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
            perception_bundle=str(bundle_path),
            bundle_sha256=digest(bundle_path),
            projection_samples_sha256=digest(
                ROOT / "outputs/navigation_m2_rule_audit_20261016/samples.jsonl"
            ),
            included=[{k: v for k, v in e.items() if k != "rows"} for e in included],
            excluded=excluded,
            inference_routes=[e["route_id"] for e in selected],
            timestamps=518,
            native_images=1554,
            maximum_wall_seconds=600,
            approved_inputs="RGB/calibration/timestamps/navigation/ego-motion SensorPacket only",
            visibility_certified=False,
            applicability_confirmed=False,
            driving_policy_changed=False,
        ),
    )

    def progress(done, route_id, status="running"):
        if status == "running" and (
            (args.out / "STOP").exists() or time.monotonic() - started >= 600
        ):
            raise InterruptedError("STOP or600s temporal replay cap")
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                phase=f"Causal TRAIN RGB replay route{route_id}",
                images_done=done * 3,
                images_total=1554,
                timestamps_done=done,
                timestamps_total=518,
                updated_at=time.time(),
                elapsed_s=time.monotonic() - started,
            ),
        )

    progress(0, None)
    import torch

    torch.set_num_threads(2)
    perception = FrozenSignalPerception(bundle_path, args.device)
    evaluated, latency, done = [], [], 0
    with (args.out / "predictions.jsonl").open("x") as stream:
        for episode in selected:
            tracker, rows = SignalTracker(), []
            for source in episode["rows"]:
                progress(done, episode["route_id"])
                packet = load_policy_sample(source["shard"], source["record"])
                if (
                    packet.frame != source["frame"]
                    or abs(packet.timestamp - source["timestamp"]) > 1e-6
                ):
                    raise ValueError("Projection/RGB packet timestamp mismatch")
                tick = time.monotonic()
                observations = perception.infer(packet)
                tracked = tracker.step(packet, observations)
                latency.append(time.monotonic() - tick)
                # Privileged projection/phase enters only after sensor-only inference.
                scored = score_frame(source, tracked["observations"])
                row = dict(**source, **tracked, scored=scored)
                rows.append(row)
                stream.write(
                    json.dumps(
                        dict(
                            route_id=episode["route_id"],
                            frame=packet.frame,
                            timestamp=packet.timestamp,
                            **tracked,
                            scored=scored,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                done += 1
            evaluated.append(
                dict(
                    route_id=episode["route_id"],
                    rows=rows,
                    metrics=summarize_frames(rows),
                    transitions=transition_delays(rows),
                )
            )
    all_counts = Counter()
    for e in evaluated:
        all_counts.update(e["metrics"]["counts"])
    import numpy as np

    plot(evaluated, args.out)
    result = dict(
        diagnostic_complete=True,
        scope="Familiar assisted TRAIN component replay only",
        counts=dict(all_counts),
        excluded=excluded,
        episodes=[{k: v for k, v in e.items() if k != "rows"} for e in evaluated],
        stop_events=[
            event_audit(
                e, next((r["rows"] for r in evaluated if r["route_id"] == e["route_id"]), [])
            )
            for e in included
        ],
        component_and_tracking_seconds=dict(
            mean=float(np.mean(latency)),
            p95=float(np.percentile(latency, 95)),
            excludes="Shard decode, projection scoring and IO",
        ),
        wall_seconds=time.monotonic() - started,
        predictions_sha256=digest(args.out / "predictions.jsonl"),
        request_sha256=digest(args.out / "request.json"),
        applicability_confirmed=False,
        visibility_certified=False,
        state_labels_released=False,
        driving_policy_changed=False,
        tracker_selected_for_driver=False,
        limitations=[
            "GT association is scoring only, not a runtime applicable-light selector",
            "Projected overlap/phase are not verified facing, visibility or legibility labels",
            "Per-view identities; no cross-view identity, depth, stop line or applicability",
            "Angular association ignores translation/ego tilt; nearby heads may merge/switch",
            "Teacher stop/release and assisted actual speed are not student driving",
            "State probabilities are uncalibrated; no threshold tuning or independent test here",
        ],
    )
    atomic_json(args.out / "assessment.json", result)
    progress(done, None, "complete")
    print(
        json.dumps(
            dict(out=str(args.out), counts=result["counts"], wall_seconds=result["wall_seconds"]),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        included, excluded, selected = prepare()
        print(
            json.dumps(
                dict(
                    included=[{k: v for k, v in e.items() if k != "rows"} for e in included],
                    excluded=excluded,
                    inference_timestamps=sum(e["samples"] for e in selected),
                ),
                indent=2,
            )
        )
        return
    with RunLock(ROOT / "outputs/.navigation.lock"):
        try:
            run(args)
        except BaseException as error:
            if args.out.exists():
                atomic_json(args.out / "failure.json", dict(error=repr(error)))
                progress_path = args.out / "progress.json"
                p = read(progress_path) if progress_path.exists() else {}
                atomic_json(
                    progress_path,
                    dict(
                        p,
                        status="stopped" if isinstance(error, InterruptedError) else "failed",
                        error=repr(error),
                        updated_at=time.time(),
                    ),
                )
            raise


if __name__ == "__main__":
    main()
