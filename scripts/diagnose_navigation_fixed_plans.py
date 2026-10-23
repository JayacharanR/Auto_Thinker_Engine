#!/usr/bin/env python3
"""CPU-only frozen-model diagnosis on independently audited prospective TRAIN plans."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def first_collision_labels(prefix):
    """A witnessed first event remains true; censored no-event tails stay unknown."""
    observed = prefix["matching_commands_before_safety"]
    hit = prefix["first_contact_tick"]
    if type(observed) is not int or not 0 <= observed <= 80:
        raise ValueError("Bounded independently verified executed prefix required")
    if hit is not None and (type(hit) is not int or not 1 <= hit <= observed):
        raise ValueError("First contact must be witnessed within the actual matched prefix")
    horizon = np.arange(10, 81, 10)
    positive = np.zeros(8, bool) if hit is None else horizon >= hit
    valid = (horizon <= observed) | positive
    labels = np.where(valid, positive.astype(float), np.nan)
    return labels, valid


def score_hazards(prediction, labels, valid):
    prediction, labels, valid = np.asarray(prediction), np.asarray(labels), np.asarray(valid)
    if (
        prediction.shape != (8,)
        or labels.shape != (8,)
        or valid.shape != (8,)
        or valid.dtype != bool
        or not np.isfinite(prediction).all()
        or (prediction < 0).any()
        or (prediction > 1).any()
        or not np.isin(labels[valid], [0.0, 1.0]).all()
    ):
        raise ValueError("Eight finite hazard scores and explicit known-event masks required")
    # Subset before arithmetic: NaN*0 is still NaN.
    p, y = prediction[valid], labels[valid]
    return dict(
        labeled_horizons=len(p),
        positive_horizons=int(y.sum()),
        missed_positive_at_half=int(((p < 0.5) & (y == 1)).sum()),
        false_positive_at_half=int(((p >= 0.5) & (y == 0)).sum()),
        sum_squared_error=float(np.square(p - y).sum()),
    )


def past_context(episode, decision_frame):
    from src.navigation.dataset import load_policy_sample, shard_records

    packets, sources = [], {}
    for shard in sorted((episode / "data").glob("shard_*.zip")):
        sources[str(shard)] = digest(shard)
        for row in shard_records(shard):
            if row["frame"] <= decision_frame:
                packets.append(load_policy_sample(shard, row))
    packets = packets[-8:]
    if (
        len(packets) != 8
        or packets[-1].frame != decision_frame
        or any(b.frame != a.frame + 2 for a, b in zip(packets, packets[1:]))
    ):
        raise ValueError(
            "Exactly eight contiguous10Hz past-only packets ending at decision required"
        )
    return packets, sources


def predict_episode(episode, encoder, detector, ensembles, encoder_sha, ego_model, check):
    import torch
    from PIL import Image

    from src.navigation.candidate_controls import control_candidates
    from src.navigation.constant_velocity import collision_risk
    from src.navigation.contracts import VIEWS
    from src.navigation.trajectory_candidates import common_cost
    from src.navigation.vehicle_perception import CameraVehicleTracker

    plan = read(episode / "frozen_plan.json")
    packets, shards = past_context(episode, plan["branch_frame"])
    packet = packets[-1]
    if not np.array_equal(packet.navigation_waypoints, plan["navigation"]):
        raise ValueError("Diagnostic context differs from the archived decision")
    slate = control_candidates(
        plan["navigation"],
        np.zeros(8),
        np.full(8, 5.0),
        plan["speed_mps"],
        plan["actual_previous_controls"],
        ego_model,
    )
    if len(plan["slate"]) != 15 or any(
        not np.array_equal(p.ordered_controls, r["ordered_controls"])
        for p, r in zip(slate, plan["slate"])
    ):
        raise ValueError("Only the exact independently audited prospective actions may be scored")
    check()
    video = np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        Image.fromarray(p.rgb[v]).resize((256, 144), Image.Resampling.BILINEAR)
                    ).transpose(2, 0, 1)
                    for v in VIEWS
                ]
            )
            for p in packets
        ]
    )
    with torch.inference_mode():
        tokens = encoder.video_features(torch.from_numpy(video.transpose(1, 0, 2, 3, 4)[None]))
    tokens = tokens[0].reshape(48, 384).half().cpu().numpy()
    tracker = CameraVehicleTracker()
    detection_metadata = []
    tracks = []
    # Three samples all lie inside the same eight-frame past context.
    for index in (1, 4, 7):
        check()
        detections, metadata = detector.detect(packets[index])
        tracks = tracker.step(packets[index], detections)
        detection_metadata.append(dict(frame=packets[index].frame, **metadata))
    check()
    controls = np.stack([p.ordered_controls for p in slate])
    cv = np.stack([np.maximum.accumulate(collision_risk(p.realized, tracks)) for p in slate])
    predictions = dict(
        cv=dict(
            collision_mean=cv.tolist(),
            calibrated_probability=False,
            score_kind="cumulative_maximum_camera_track_proximity",
        )
    )
    for name, ensemble in ensembles.items():
        check()
        out = ensemble.predict(
            tokens, controls, packet.navigation_waypoints, packet.ego_motion, encoder_sha
        )
        predictions[name] = dict(
            collision_mean=out["collision_mean"].tolist(),
            collision_members=out["collision_members"].tolist(),
            learned_progress_mean_m=out["progress_mean_m"].tolist(),
            calibrated_probability=False,
            score_kind="uncalibrated_collision_balanced_sigmoid",
        )
    for result in predictions.values():
        risks = np.asarray(result["collision_mean"])
        result["common_collision_only_cost"] = [
            common_cost(p.realized, risk, np.zeros(8)) for p, risk in zip(slate, risks)
        ]
        result["minimum_cost_index_unexecuted"] = int(
            np.argmin(result["common_collision_only_cost"])
        )
    return dict(
        episode=episode.name,
        plan_sha256=digest(episode / "frozen_plan.json"),
        input_frames=[p.frame for p in packets],
        source_shards=shards,
        encoder_tokens_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
        tracks=[
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(track).items()}
            for track in tracks
        ],
        detector_metadata=detection_metadata,
        predictions=predictions,
        candidates=[dict(path=p.desired.path_index, speed=p.desired.speed_index) for p in slate],
        executed_candidate_index=5 * plan["selected_path"] + plan["selected_speed"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This concurrent read-only diagnostic requires CUDA_VISIBLE_DEVICES=''")
    args.out.mkdir(parents=True, exist_ok=False)
    descriptors = ROOT / "outputs/navigation_dynamics_ensembles_20261020"
    encoder_path = ROOT / read(descriptors / "jepa.json")["encoder_descriptor"]
    detector_path = ROOT / "outputs/navigation_models/navigation_tracking_fpn_20261019.json"
    # The official descriptor location is a registration parameter, never a download.
    if not detector_path.exists():
        raise FileNotFoundError(f"Registered FPN descriptor unavailable:{detector_path}")
    sources = [
        Path(__file__),
        ROOT / "reports/navigation_fixed_plan_diagnosis_protocol_20261020.md",
        ROOT / "src/jepa/encoder.py",
        *[
            ROOT / "src/navigation" / name
            for name in (
                "grid_encoders.py",
                "rectangular_jepa.py",
                "road_user_fpn.py",
                "vehicle_perception.py",
                "constant_velocity.py",
                "candidate_controls.py",
                "trajectory_candidates.py",
                "frozen_ego.py",
                "dataset.py",
                "contracts.py",
                "dynamics_inference.py",
                "ordered_dynamics.py",
                "rssm_dynamics.py",
            )
        ],
    ]
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    model_sha = {
        str(p): digest(p)
        for p in (encoder_path, detector_path, descriptors / "jepa.json", descriptors / "rssm.json")
    }
    atomic_json(
        args.out / "queued_request.json",
        dict(
            branches=str(args.branches.resolve()),
            audit=str(args.audit.resolve()),
            source_hashes=frozen,
            descriptors_sha256=model_sha,
            CPU_only=True,
            maximum_compute_wall_seconds=900,
            threads=2,
            no_updates=True,
            planner_qualified=False,
            single_training_location=True,
        ),
    )
    for path in sources:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    status, done, started = "running", 0, None

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="M5 frozen-plan diagnosis",
                frames=done,
                target_frames=6,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Frozen-plan diagnostic STOP")
        if started is not None and time.monotonic() - started > 900:
            raise InterruptedError("Frozen-plan diagnostic900s compute cap")

    try:
        with BudgetWatch(args.out):
            while True:
                check()
                path = args.audit / "assessment.json"
                if path.exists():
                    audited = read(path)
                    if (
                        audited["errors"]
                        or not audited["executable_prefixes_verified"]
                        or not audited["replay_observables_verified"]
                    ):
                        raise ValueError("Prospective control/replay evidence failed; no diagnosis")
                    break
                if (args.branches / "progress.json").exists() and read(
                    args.branches / "progress.json"
                )["status"] not in ("running", "complete"):
                    raise RuntimeError(
                        "Branch collection failed/stopped; preserve diagnostic request"
                    )
                progress(
                    "Waiting for closed independent prospective-plan audit; no GPU/native owner"
                )
                time.sleep(3)
            if any(digest(ROOT / p) != h for p, h in frozen.items()) or any(
                digest(p) != h for p, h in model_sha.items()
            ):
                raise ValueError("Registered diagnostic sources/models changed while queued")
            atomic_json(
                args.out / "request.json",
                dict(
                    audit_sha256=digest(args.audit / "assessment.json"),
                    collection_request_sha256=digest(args.branches / "request.json"),
                    registration_sha256=digest(args.out / "queued_request.json"),
                    precision="CPU FP32 encoder/ensemble forward,FP16 token export; "
                    "differs from registered GPU BF16 cache forward",
                    shared_cost_progress="identical frozen-ego forecast; "
                    "each learned progress head is omitted from ranking",
                    red_cost_disabled_not_observed_safe=True,
                    perception_not_matched="CVuses fixed FPN tracks; "
                    "old learned models use JEPA tokens. "
                    "Diagnostic only; not an isolated same-perception planner comparison.",
                ),
            )
            import torch

            from src.navigation.dynamics_inference import FactualDynamicsEnsemble
            from src.navigation.frozen_ego import FrozenExpertEgo
            from src.navigation.grid_encoders import FrozenGridEncoder
            from src.navigation.road_user_fpn import FrozenRoadUsersFPN

            torch.set_num_threads(2)
            started = time.monotonic()
            encoder = FrozenGridEncoder(read(encoder_path)).cpu()
            detector = FrozenRoadUsersFPN(detector_path, "cpu")
            ensembles = {
                name: FactualDynamicsEnsemble(descriptors / f"{name}.json", "cpu")
                for name in ("jepa", "rssm")
            }
            ego_model = FrozenExpertEgo(
                args.branches / "source/outputs/navigation_models/frozen_expert_ego_20261020.json"
            )
            predictions = []
            for name in sorted(r["name"] for r in audited["records"]):
                check()
                progress(f"Past-only CPU prediction at {name}; no outcome inputs")
                predictions.append(
                    predict_episode(
                        args.branches / name,
                        encoder,
                        detector,
                        ensembles,
                        digest(encoder_path),
                        ego_model,
                        check,
                    )
                )
                done += 1
                atomic_json(args.out / "predictions_in_progress.json", predictions)
            # Freeze all predictions before accessing the audit's labels for scoring.
            atomic_json(args.out / "predictions.json", predictions)
            audit_rows = {r["name"]: r for r in read(args.audit / "assessment.json")["records"]}
            results = []
            for prediction in predictions:
                outcome = audit_rows[prediction["episode"]]
                labels, valid = first_collision_labels(outcome["prefix"])
                index = prediction["executed_candidate_index"]
                scored = {
                    name: score_hazards(np.array(v["collision_mean"])[index], labels, valid)
                    for name, v in prediction["predictions"].items()
                }
                results.append(
                    dict(
                        episode=prediction["episode"],
                        observed_index=index,
                        prefix=outcome["prefix"],
                        scores=scored,
                        unfiltered_contact_starts=outcome["unfiltered_contact_starts"],
                    )
                )
            totals = {
                name: {
                    key: sum(r["scores"][name][key] for r in results)
                    for key in results[0]["scores"][name]
                }
                for name in ("cv", "jepa", "rssm")
            }
            for value in totals.values():
                value["mean_squared_error_descriptive"] = value["sum_squared_error"] / max(
                    1, value["labeled_horizons"]
                )
            atomic_json(
                args.out / "assessment.json",
                dict(
                    results=results,
                    totals=totals,
                    predictions_sha256=digest(args.out / "predictions.json"),
                    actual_native_branch_recordings=6,
                    independent_scenarios=1,
                    unfiltered_contact_starts=audited["unfiltered_all_attempt_contact_starts"],
                    inference_precision_differs_from_training=True,
                    red_risk_qualified=False,
                    calibration_qualified=False,
                    planner_qualified=False,
                    all15way_ground_truth_available=False,
                    limitation="Three observed options, two repeats at one stationary TRAINlead. "
                    "No tuning or independent-scenario safety inference. CVproximity is "
                    "not a calibrated probability. No full15way ranking success claim.",
                ),
            )
            status = "complete"
            progress("Finished diagnostic only; no model changes or planner promotion")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(args.out / "error.json", dict(error=str(error), at=time.time()))
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
