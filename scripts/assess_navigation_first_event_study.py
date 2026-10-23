#!/usr/bin/env python3
"""Frozen15-option predictions and explicit rejection gates; never native driving."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_first_event_baselines import (  # noqa:E402
    load_inputs,
    load_targets,
    metrics,
    read,
)
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.first_event_dynamics import FirstEventDynamics  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "study", "baselines", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    if read(args.study / "progress.json")["status"] != "complete":
        raise ValueError("Allsixoriginalfixedfitsmustfinish")
    request = read(args.study / "request.json")
    if any(digest(ROOT / p) != sha for p, sha in request["source_hashes"].items()):
        raise ValueError("Selectedmemberinference/sourcechanged")
    members = read(args.study / "members.json")
    records = read(args.features / "records.json")
    if (
        len(members) != 6
        or digest(args.features / "records.json") != request["feature_records_sha256"]
    ):
        raise ValueError("Originalsixmembers/68featuresonly")
    atomic_json(
        args.out / "request.json",
        dict(
            study_request_sha256=digest(args.study / "request.json"),
            source_sha256=digest(__file__),
            all_prospective_candidates=15,
            no_native_driving=True,
            maximum_wall_seconds=900,
            maximum_artifact_bytes=100_000_000,
            primary_metrics="Originalselectedmemberpredictions;neverreselect",
            full_slate_outcomes="Onlyrecordedoptionsobserved;allothersunknown",
            ties="Reportallminimumrisktiesandtheirbest/worstobservedoutcomes",
        ),
    )
    started, done = time.monotonic(), 0

    def check():
        if (args.out / "STOP").exists() or time.monotonic() - started > 900:
            raise InterruptedError("OwnedpredictionSTOPor900scap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 100_000_000:
            raise InterruptedError("Prediction100MBartifactcap")

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status="running",
                stage="First-event frozen candidate audit",
                frames=done,
                target_frames=408,
                phase=phase,
                updated_at=time.time(),
            ),
        )

    with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
        check_resources(args.out, 10)
        torch.set_num_threads(2)
        inputs = []
        selected = []
        for record in records:
            value, plans, _ = load_inputs(record)
            condition = np.r_[
                value["navigation"].ravel() / 32,
                value["ego"] / [10.0, np.pi, 1.0],
                value["own_controls"].ravel(),
            ]
            inputs.append(
                dict(
                    tokens=torch.as_tensor(
                        value["current_tokens"].astype(np.float32), device="cuda"
                    )[None].expand(15, -1, -1),
                    controls=torch.as_tensor(plans["all_controls"], device="cuda"),
                    conditioning=torch.as_tensor(condition.astype(np.float32), device="cuda")[
                        None
                    ].expand(15, -1),
                    objects=torch.as_tensor(value["road_users"], device="cuda")[None].expand(
                        15, -1, -1, -1, -1
                    ),
                    presence=torch.as_tensor(value["presence"], device="cuda")[None].expand(
                        15, -1, -1, -1
                    ),
                    past_ego=torch.as_tensor(
                        (value["past_ego"] / [10.0, np.pi, 1.0]).astype(np.float32), device="cuda"
                    )[None].expand(15, -1, -1),
                )
            )
            matches = np.flatnonzero(
                np.all(plans["all_controls"] == plans["ordered_controls"][None], axis=(1, 2))
            )
            if len(matches) != 1:
                raise ValueError("Uniqueexecutedcandidate")
            selected.append(int(matches[0]))
        all_logits, parity = [], []
        for member in members:
            check()
            path = Path(member["output"]) / "selected.pt"
            if digest(path) != member["selected_sha256"]:
                raise ValueError("Selectedmemberartifactchanged")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["request_sha256"] != digest(args.study / "request.json"):
                raise ValueError("Selectedmemberrequestnotbound")
            model = FirstEventDynamics(member["kind"]).cuda().eval().requires_grad_(False)
            model.load_state_dict(saved["model"], strict=True)
            logits = []
            for value in inputs:
                check()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(**value)
                logits.append(prediction["logits"].cpu().numpy())
                done += 1
                progress("All15frozenplans;RGBhistoryandownstateonly;outcomesnotread")
            logits = np.array(logits)
            original = np.load(Path(member["output"]) / "selected_predictions.npz")["logits"]
            observed = logits[np.arange(68), selected]
            parity.append(
                dict(
                    kind=member["kind"],
                    seed=member["seed"],
                    maximum_logit_difference=float(abs(observed - original).max()),
                    same_BF16_different_batch_shapes=True,
                )
            )
            all_logits.append(logits)
            del model
        all_logits = np.array(all_logits)
        np.savez_compressed(
            args.out / "all15_predictions.npz",
            logits=all_logits,
            selected_indices=np.array(selected),
        )
        atomic_json(
            args.out / "prediction_receipt.json",
            dict(
                predictions_sha256=digest(args.out / "all15_predictions.npz"),
                member_order=[dict(kind=m["kind"], seed=m["seed"]) for m in members],
                outcome_targets_not_deserialized_during_inference=True,
                selected_parity=parity,
            ),
        )
        # Prediction generation has ended. Only now load audited event labels.
        targets = [load_targets(record) for record in records]
        baseline = read(args.baselines / "assessment.json")["metrics"]["calibration"]
        study = read(args.study / "assessment.json")["metrics"]
        criteria = {}
        for kind in ("jepa", "rssm"):
            score = study[kind]["calibration"]
            checks = dict(
                beats_constant_brier=score["endpoint_brier"]
                < baseline["constant"]["endpoint_brier"],
                beats_CV_brier=score["endpoint_brier"] < baseline["cv"]["endpoint_brier"],
                beats_action_only_brier=score["endpoint_brier"]
                < baseline["action_only"]["endpoint_brier"],
                beats_constant_NLL=score["sequence_nll"] < baseline["constant"]["sequence_nll"],
                beats_CV_NLL=score["sequence_nll"] < baseline["cv"]["sequence_nll"],
                misses_no_worse_than_CV=score["misses"] <= baseline["cv"]["misses"],
                false_alarms_no_worse_than_CV=score["false_alarms"]
                <= baseline["cv"]["false_alarms"],
            )
            criteria[kind] = dict(
                checks=checks,
                offline_countermeasure_passed=all(checks.values()),
                planner_qualified=False,
            )
        per_family = {}
        per_location = {}
        for kind in ("jepa", "rssm"):
            prediction = np.load(args.study / f"{kind}_ensemble.npz")["cumulative"]
            for key, grouping in (("mechanism", per_family), ("route_id", per_location)):
                groups = sorted({str(r[key]) for r in records if r["source_role"] == "calibration"})
                for group in groups:
                    indices = [
                        i
                        for i, r in enumerate(records)
                        if r["source_role"] == "calibration" and str(r[key]) == group
                    ]
                    grouping.setdefault(kind, {})[group] = metrics(
                        prediction[indices], [targets[i] for i in indices]
                    )
        atomic_json(
            args.out / "assessment.json",
            dict(
                criteria=criteria,
                primary_metrics=study,
                baselines=baseline,
                by_mechanism=per_family,
                by_calibration_route=per_location,
                original_attempts=73,
                eligible=68,
                unknown=5,
                original_contact_starts=16,
                all15_predictions_retained=True,
                unexecuted_outcomes_unknown=True,
                no_automatic_driving_or_promotion=True,
                M5_qualified=False,
                M6_not_run=True,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        shutil.copyfile(__file__, args.out / "assessment_source.py")
        atomic_json(
            args.out / "progress.json",
            dict(
                status="complete",
                stage="First-event frozen candidate audit",
                frames=408,
                target_frames=408,
                phase="Allmembersassessed; all15predictionsretained; no native rollout",
                updated_at=time.time(),
            ),
        )
        print(json.dumps(criteria), flush=True)


if __name__ == "__main__":
    main()
