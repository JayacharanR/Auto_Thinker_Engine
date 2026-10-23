#!/usr/bin/env python3
"""Six fixed-budget first-event fits; no native deployment or automatic promotion."""

import argparse
import copy
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_first_event_baselines import (  # noqa:E402
    load_inputs,
    load_targets,
    metrics,
    read,
)
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.first_event_dynamics import FirstEventDynamics, hazard_nll  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402


def write_checkpoint(path, state):
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def mixture_sequence_nll(member_logits, targets):
    """Exact mixture event/survival likelihood, avoiding rounded CDF differences."""
    logits = np.asarray(member_logits, np.float64)
    log_survival = np.cumsum(-np.logaddexp(0.0, logits), axis=-1)
    before = np.concatenate([np.zeros_like(log_survival[..., :1]), log_survival[..., :-1]], -1)
    log_event = before - np.logaddexp(0.0, -logits)
    losses = []
    for i, target in enumerate(targets):
        length = int(target["valid"].sum())
        if not length:
            continue
        density = (
            log_event[:, i, length - 1] if target["event"].any() else log_survival[:, i, length - 1]
        )
        losses.append(-float(np.logaddexp.reduce(density) - np.log(len(logits))))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    if (
        read(args.features / "progress.json")["status"] != "complete"
        or read(args.baselines / "progress.json")["status"] != "complete"
    ):
        raise ValueError("Both frozen features and simple baselines must be terminal/complete")
    records = read(args.features / "records.json")
    if len(records) != 68:
        raise ValueError("Exactly68eligible recordings; keep73attemptdenominator")
    sources = [
        Path(__file__),
        ROOT / "scripts/assess_navigation_first_event_baselines.py",
        ROOT / "src/navigation/first_event_dynamics.py",
        ROOT / "src/navigation/first_event_data.py",
        ROOT / "reports/navigation_first_event_study_protocol_20261021.md",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for source in sources:
        dest = args.out / "source" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    atomic_json(
        args.out / "request.json",
        dict(
            source_hashes=hashes,
            feature_records_sha256=digest(args.features / "records.json"),
            feature_request_sha256=digest(args.features / "request.json"),
            baseline_assessment_sha256=digest(args.baselines / "assessment.json"),
            seeds=[42, 1051, 2060],
            model_kinds=["jepa", "rssm"],
            batch_size=12,
            updates=400,
            learning_rate=0.0003,
            weight_decay=0.0001,
            gradient_clip=1.0,
            hidden=96,
            evaluation_steps=[0, 100, 200, 300, 400],
            selection="LowestcalibrationsequenceNLL;earlytie",
            maximum_member_seconds=600,
            maximum_study_seconds=3900,
            maximum_artifact_bytes=500_000_000,
            no_native_or_automatic_promotion=True,
            positive_class_balancing=False,
        ),
    )
    status, completed, step, active = "running", [], 0, None
    started, member_started = time.monotonic(), time.monotonic()

    def progress(phase):
        atomic_json(
            args.out / "progress.json",
            dict(
                status=status,
                stage="First-event matched model study",
                frames=len(completed) * 400 + step,
                target_frames=2400,
                active_member=active,
                step=step,
                target_steps=400,
                members_done=len(completed),
                members_total=6,
                phase=phase,
                updated_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )

    def check():
        if (
            (args.out / "STOP").exists()
            or time.monotonic() - started > 3900
            or time.monotonic() - member_started > 600
        ):
            raise InterruptedError("Owned study STOPorregistered wallcap")
        if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > 500_000_000:
            raise InterruptedError("Study500MBartifactcap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation.lock"):
            check_resources(args.out, 10)
            torch.set_num_threads(2)
            progress("Verifyallfeatures; inputs strictly separate from targets")
            values, targets, latent, conditions = [], [], [], []
            for row in records:
                inputs, plans, _ = load_inputs(row)
                values.append(
                    dict(
                        tokens=inputs["current_tokens"].astype(np.float32),
                        controls=plans["ordered_controls"],
                        objects=inputs["road_users"],
                        presence=inputs["presence"],
                        past_ego=(inputs["past_ego"] / [10.0, np.pi, 1.0]).astype(np.float32),
                    )
                )
                conditions.append(
                    np.r_[
                        inputs["navigation"].ravel() / 32,
                        inputs["ego"] / [10.0, np.pi, 1.0],
                        inputs["own_controls"].ravel(),
                    ].astype(np.float32)
                )
                targets.append(load_targets(row))
                with np.load(Path(row["output"]) / "latent_targets.npz") as future:
                    latent.append(
                        np.concatenate(
                            [
                                future["tokens"].astype(np.float32),
                                np.zeros((1, 48, 384), np.float32),
                            ]
                        )
                    )
            tensors = {
                k: torch.from_numpy(np.stack([v[k] for v in values])).cuda() for k in values[0]
            }
            tensors["conditioning"] = torch.from_numpy(np.stack(conditions)).cuda()
            event = torch.from_numpy(np.stack([t["event"] for t in targets])).cuda()
            valid = torch.from_numpy(np.stack([t["valid"] for t in targets])).cuda()
            latent_valid = torch.from_numpy(np.stack([t["latent_valid"] for t in targets])).cuda()
            future = torch.from_numpy(np.stack(latent)).cuda()
            train = np.array([r["source_role"] == "train" for r in records])
            ti, vi = np.flatnonzero(train), np.flatnonzero(~train)
            if len(ti) != 50 or len(vi) != 18:
                raise ValueError("Frozen50TRAIN/18calibration partition")
            prior = float(event[ti].sum() / valid[ti].sum())

            def forward(model, index):
                return model(**{k: v[index] for k, v in tensors.items()})

            def evaluate(model):
                prediction, logits = [], []
                model.eval()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for start in range(0, 68, 8):
                        value = forward(model, slice(start, start + 8))
                        prediction.append(value["cumulative"].cpu().numpy())
                        logits.append(value["logits"].cpu().numpy())
                probability, raw_logits = np.concatenate(prediction), np.concatenate(logits)
                if not np.isfinite(probability).all():
                    raise FloatingPointError("Nonfinitefirsteventpredictions")
                score = dict(
                    train=metrics(probability[ti], [targets[i] for i in ti]),
                    calibration=metrics(probability[vi], [targets[i] for i in vi]),
                )
                for role, indices in (("train", ti), ("calibration", vi)):
                    score[role]["sequence_nll"] = mixture_sequence_nll(
                        raw_logits[None, indices], [targets[i] for i in indices]
                    )
                return probability, raw_logits, score

            for kind in ("jepa", "rssm"):
                for seed in (42, 1051, 2060):
                    member_started, step = time.monotonic(), 0
                    active = f"{kind}_seed{seed}"
                    check()
                    if any(digest(ROOT / p) != sha for p, sha in hashes.items()):
                        raise ValueError("Registered study source changed")
                    member = args.out / active
                    member.mkdir()
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    rng = np.random.default_rng(seed)
                    model = FirstEventDynamics(kind).cuda()
                    torch.nn.init.zeros_(model.hazard[-1].weight)
                    torch.nn.init.constant_(model.hazard[-1].bias, math.log(prior / (1 - prior)))
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=0.0003, weight_decay=0.0001
                    )
                    best, best_step, history = float("inf"), None, []
                    best_state, best_prediction, best_logits = None, None, None

                    def checkpoint(model, optimizer, step):
                        return dict(
                            model={
                                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                            },
                            optimizer=optimizer.state_dict(),
                            step=step,
                            seed=seed,
                            kind=kind,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(),
                            numpy_rng=rng.bit_generator.state,
                            python_rng=random.getstate(),
                            grad_scaler=None,
                            precision="BF16nodynamicscalerrequired",
                            request_sha256=digest(args.out / "request.json"),
                            planner_qualified=False,
                            red_risk_qualified=False,
                        )

                    for step in range(401):
                        check()
                        if step:
                            model.train()
                            index = rng.choice(ti, 12, replace=True)
                            optimizer.zero_grad(set_to_none=True)
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                prediction = forward(model, index)
                                risk = hazard_nll(prediction["logits"], event[index], valid[index])
                                mask = latent_valid[index]
                                safe_target = torch.where(mask[..., None, None], future[index], 0.0)
                                latent_error = F.smooth_l1_loss(
                                    prediction["future_tokens"].float(),
                                    safe_target,
                                    reduction="none",
                                ).mean((-1, -2))
                                latent_loss = (latent_error * mask).sum() / mask.sum().clamp_min(1)
                                kl = model.posterior_kl(prediction, future[index], mask)
                                loss = risk + latent_loss + 0.01 * kl
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Nonfinitefirsteventobjective")
                            loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            if not torch.isfinite(norm):
                                raise FloatingPointError("Nonfinitefirsteventgradient")
                            optimizer.step()
                        if step % 100 == 0:
                            probability, logits, score = evaluate(model)
                            history.append(dict(step=step, **score))
                            np.savez_compressed(
                                member / f"predictions_{step:04d}.npz",
                                cumulative=probability,
                                logits=logits,
                            )
                            atomic_json(member / "history.json", history)
                            current = score["calibration"]["sequence_nll"]
                            if current < best:
                                best, best_step = current, step
                                best_state = copy.deepcopy(checkpoint(model, optimizer, step))
                                best_prediction = probability.copy()
                                best_logits = logits.copy()
                                write_checkpoint(member / "selected.pt", best_state)
                            write_checkpoint(
                                member / "latest.pt", checkpoint(model, optimizer, step)
                            )
                        if step % 10 == 0:
                            progress("PropercensoredNLL;fixed400updates;no native driving")
                    np.savez_compressed(
                        member / "selected_predictions.npz",
                        cumulative=best_prediction,
                        logits=best_logits,
                    )
                    completed.append(
                        dict(
                            kind=kind,
                            seed=seed,
                            output=str(member),
                            selected_step=best_step,
                            selected_nll=best,
                            trainable_parameters=sum(p.numel() for p in model.parameters()),
                            elapsed_seconds=time.monotonic() - member_started,
                            selected_sha256=digest(member / "selected.pt"),
                            latest_sha256=digest(member / "latest.pt"),
                            history=history,
                        )
                    )
                    atomic_json(args.out / "members.json", completed)
                    del model, optimizer, best_state
                    torch.cuda.empty_cache()
                    step = 0
                    progress("Completedmemberretained;nextfixedseed, no extensions")
            ensembles = {}
            for kind in ("jepa", "rssm"):
                member_logits = [
                    np.load(Path(r["output"]) / "selected_predictions.npz")["logits"]
                    for r in completed
                    if r["kind"] == kind
                ]
                member_logits = np.asarray(member_logits, np.float64)
                members = -np.expm1(np.cumsum(-np.logaddexp(0.0, member_logits), axis=-1))
                probability = np.mean(members, axis=0)
                np.savez_compressed(
                    args.out / f"{kind}_ensemble.npz",
                    cumulative=probability,
                    members=np.array(members),
                )
                ensembles[kind] = dict(
                    train=metrics(probability[ti], [targets[i] for i in ti]),
                    calibration=metrics(probability[vi], [targets[i] for i in vi]),
                )
                for role, indices in (("train", ti), ("calibration", vi)):
                    ensembles[kind][role]["sequence_nll"] = mixture_sequence_nll(
                        member_logits[:, indices], [targets[i] for i in indices]
                    )
            atomic_json(
                args.out / "assessment.json",
                dict(
                    completed_members=6,
                    metrics=ensembles,
                    original_attempts=73,
                    eligible_TRAIN=50,
                    eligible_calibration=18,
                    unknown_attempts=5,
                    no_planner_qualification=True,
                    no_native_driving=True,
                    no_red_cost_model=True,
                    scripted_not_reactive=True,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            status, active = "complete", None
            progress("Sixfixedfitscomplete;independentassessmentrequiredbeforeanydeployment")
    except BaseException as error:
        status = "stopped" if isinstance(error, InterruptedError) else "failed"
        atomic_json(
            args.out / "error.json",
            dict(error=str(error), at=time.time(), member=active, step=step),
        )
        progress(str(error))
        raise


if __name__ == "__main__":
    main()
