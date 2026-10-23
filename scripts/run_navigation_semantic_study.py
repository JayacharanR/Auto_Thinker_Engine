#!/usr/bin/env python3
"""Queue/extract/fit paired semantic stop corrections after the frozen rule study."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_navigation_causal_student import score_stop_metrics  # noqa:E402
from scripts.train_navigation_fact_student import FactCorpus  # noqa:E402
from scripts.train_navigation_student import atomic_checkpoint, rng_state  # noqa:E402
from scripts.train_navigation_visual_student import inputs_for  # noqa:E402
from src.navigation.contracts import VIEWS  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.dataset import load_policy_sample, shard_records  # noqa:E402
from src.navigation.fact_student import FactStudent  # noqa:E402
from src.navigation.runtime import RunLock, atomic_json, check_resources  # noqa:E402
from src.navigation.semantic_stop_residual import (  # noqa:E402
    SemanticStopResidual,
    semantic_features,
)
from src.navigation.spatial_data import TARGET_ARRAYS, validation_positions  # noqa:E402
from src.navigation.storage_budget import BudgetWatch  # noqa:E402

PARENT = ROOT / "outputs/navigation_fact_study_20261020/cnn_scratch_facts/best_policy.pt"
PARENT_SHA = "e3a688576d007158ee9d61527de83c4eed839b2bd6ddb0d1bc34e5f88181ea48"
DATA = ROOT / "outputs/navigation_spatial_data_repaired_20261019"
LABELS = ROOT / "outputs/navigation_fact_labels_20261020"
PREDICTIONS = ("stop_logits", "cruise_speeds", "light", "presence", "distance", "hazards")


def read(path):
    return json.loads(Path(path).read_text())


def admitted(candidate, parent):
    return bool(
        candidate["counts"] == parent["counts"]
        and candidate["controller_stop_recall"] >= parent["controller_stop_recall"] + .02 - 1e-10
        and candidate["controller_false_hold_fraction"]
        <= parent["controller_false_hold_fraction"] + .01 + 1e-10
        and candidate["go_speed_mae_mps"] <= parent["go_speed_mae_mps"] + .2 + 1e-10
        and candidate["selection_score"] < parent["selection_score"] - .005
        and all(
            candidate["families"][family]["stop_hits"] / values["stopped"]
            >= values["stop_hits"] / values["stopped"] - .02 - 1e-10
            for family, values in parent["families"].items() if values["stopped"] >= 30
        )
    )


def episode(cache, entry):
    start, end = entry["start"], entry["end"]
    return (
        {k: torch.from_numpy(cache["p_" + k][start:end].copy())[None] for k in PREDICTIONS},
        {k: torch.from_numpy(cache["t_" + k][start:end].copy())[None] for k in TARGET_ARRAYS},
    )


def verify_original_pixels(entry, corpus, check):
    """Compare the retained original mmap with independently decoded native RGB."""
    if entry["domain"] != "original":
        return 0  # Supplement restoration separately verifies its complete original NPY hash.
    stored = corpus.raw.rgb(entry)
    offset = 0
    for shard in sorted(Path(entry["data"]).glob("shard_*.zip")):
        check()
        for record in shard_records(shard):
            packet = load_policy_sample(shard, record)
            rgb = np.stack([
                np.asarray(Image.fromarray(packet.rgb[v]).resize(
                    (256, 144), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
                for v in VIEWS
            ])
            if offset >= len(stored) or not np.array_equal(stored[offset], rgb):
                raise ValueError("Original RGB cache differs from independently decoded source")
            offset += 1
    if offset != len(stored):
        raise ValueError("Original source/cache sample counts differ")
    return offset


@torch.inference_mode()
def evaluate(model, cache, manifest, mean, arm, check):
    model.eval()
    sums = dict(speed=0., go=0., held_stopped=0, held_go=0)
    counts = dict(speed=0, go=0, stopped=0, controller_go=0)
    families = {}
    for entry in manifest["episodes"]:
        if entry["role"] != "validation":
            continue
        check()
        base, target = episode(cache, entry)
        facts = semantic_features(base)
        if arm == "constant":
            facts = mean[None, None].expand_as(facts)
        prediction, _ = model(base, facts)
        mask = validation_positions(entry["end"] - entry["start"], "cpu")
        p, t = ({k: v[:, mask] for k, v in values.items()}
                for values in (prediction, target))
        error = (p["current_speed"] - t["current_speed"]).abs()
        stopped, go = t["current_speed"] < .1, t["current_speed"] > 1
        held = (p["current_speed"] < .15) | (p["brake"] >= 0)
        moving = go & (t["brake"] < .5)
        sums["speed"] += float(error.sum())
        sums["go"] += float(error[go].sum())
        sums["held_stopped"] += int((held & stopped).sum())
        sums["held_go"] += int((held & moving).sum())
        counts["speed"] += error.numel()
        counts["go"] += int(go.sum())
        counts["stopped"] += int(stopped.sum())
        counts["controller_go"] += int(moving.sum())
        family = families.setdefault(entry["family"], dict(stopped=0, stop_hits=0))
        family["stopped"] += int(stopped.sum())
        family["stop_hits"] += int((held & stopped).sum())
    return dict(score_stop_metrics(sums, counts), families=families)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if digest(PARENT) != PARENT_SHA:
        raise ValueError("The predeclared frozen CNN-facts parent changed")
    if (args.after / "student/request.json").exists():
        raise ValueError("This protocol must be registered before the first student rule trial")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "reports/navigation_semantic_residual_protocol_20261020.md",
               ROOT / "src/navigation/semantic_stop_residual.py",
               ROOT / "src/navigation/fact_student.py", ROOT / "src/navigation/spatial_student.py",
               ROOT / "src/navigation/visual_decision_student.py",
               ROOT / "src/navigation/spatial_data.py",
               ROOT / "scripts/train_navigation_fact_student.py",
               ROOT / "scripts/train_navigation_visual_student.py",
               ROOT / "scripts/train_navigation_student.py",
               ROOT / "scripts/train_navigation_causal_student.py",
               ROOT / "scripts/restore_navigation_shared_ram_rgb.py",
               ROOT / "src/navigation/dataset.py", ROOT / "src/navigation/contracts.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    request = dict(
        after=str(args.after.resolve()), parent=str(PARENT), parent_sha256=PARENT_SHA,
        data_manifest_sha256=digest(DATA / "manifest.json"),
        fact_manifest_sha256=digest(LABELS / "manifest.json"),
        source_hashes=hashes, seed=42, steps_per_arm=1000, arms=["constant", "facts"],
        validations=[0, 250, 500, 750, 1000], maximum_wall_seconds=1800,
        source_inputs="Frozen parent predictions only; targets loss/scoring only",
        train_timestamps=15712, no_new_annotation_timestamps=True,
        architecture="camera-semantic-stop-residual-v4", no_automatic_deployment=True,
        registered_before_first_student_rule_trial=True,
        torch_version=str(torch.__version__), numpy_version=str(np.__version__),
        checkpoint_recovery="Complete optimizer/Torch/NumPy/sampler state saved every100updates; "
                            "interrupted fits require inspected explicit continuation",
    )
    atomic_json(args.out / "request.json", request)
    for path in sources:
        target = args.out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    active, status, started, fitted = None, "running", None, 0

    def progress(phase, **values):
        atomic_json(args.out / "progress.json", dict(
            status=status, stage="M2 structured semantic residual", phase=phase,
            fits_done=fitted, fits_total=2, active_output=str(active) if active else None,
            updated_at=time.time(), **values,
        ))

    def check():
        if (args.out / "STOP").exists():
            raise InterruptedError("Semantic residual STOP")
        if started is not None and time.monotonic() - started >= 1800:
            raise InterruptedError("Whole cache/paired-fit1800s cap")

    try:
        with BudgetWatch(args.out), RunLock(ROOT / "outputs/.navigation_semantic_residual.lock"):
            while True:
                check()
                previous = args.after / "progress.json"
                if previous.exists() and read(previous)["status"] != "running":
                    if read(previous)["status"] != "complete":
                        raise RuntimeError("Frozen rule study did not complete; inspect first")
                    break
                progress("Waiting for original rule study; no concurrent GPU/native work")
                time.sleep(3)
            if any(digest(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError("Queued source changed before extraction")
            started = time.monotonic()
            with RunLock(ROOT / "outputs/.navigation.lock"):
                check_resources(args.out, 10)
                torch.set_num_threads(2)
                torch.manual_seed(42)
                supplement = ROOT / "outputs/navigation_m2_causal_approach_cache_20261018/rgb.npy"
                if not supplement.exists():
                    progress("Restore exact original supplement RGB into owned RAM cache")
                    subprocess.run([
                        sys.executable, "scripts/restore_navigation_shared_ram_rgb.py",
                        "--out", str(args.out / "rgb_restore"),
                    ], cwd=ROOT, check=True)
                corpus = FactCorpus(DATA, LABELS, enabled=True)
                if any(digest(DATA / p) != h for p, h in corpus.manifest["array_sha256"].items()):
                    raise ValueError("Frozen corrected arrays changed")
                state = torch.load(PARENT, map_location="cpu", weights_only=True)
                if state["data_manifest_sha256"] != digest(DATA / "manifest.json"):
                    raise ValueError("Parent was not fitted on this corrected corpus")
                parent = FactStudent(384, 192, scratch=True).cuda().eval().requires_grad_(False)
                parent.load_state_dict(state["model"], strict=True)
                progress("Reconstruct only missing raw TRAINsupplement into bounded RAM")
                corpus.raw.preload_missing(check)
                arrays, verified_original = {}, 0
                for index, entry in enumerate(corpus.manifest["episodes"]):
                    check()
                    progress("Verify original RGB cache against independently decoded source",
                             episodes_done=index, episodes_total=68)
                    verified_original += verify_original_pixels(entry, corpus, check)
                    with torch.inference_mode():
                        inputs, targets = inputs_for(parent, corpus, entry, "cuda", None)
                        predicted, _ = parent(**inputs)
                    for prefix, values in (("p_", predicted), ("t_", targets)):
                        names = PREDICTIONS if prefix == "p_" else TARGET_ARRAYS
                        for name in names:
                            value = values[name][0].float().cpu().numpy()
                            key = prefix + name
                            if key not in arrays:
                                arrays[key] = np.empty((corpus.manifest["samples"],
                                                        *value.shape[1:]), np.float32)
                            arrays[key][entry["start"]:entry["end"]] = value
                    progress("FP32 frozen-parent extraction; no new perception training",
                             frames=entry["end"], target_frames=18534,
                             episodes_done=index + 1, episodes_total=68)
                np.savez_compressed(args.out / "parent_predictions.npz", **arrays)
                manifest = corpus.manifest
                del parent, corpus, inputs, targets, predicted
                torch.cuda.empty_cache()
                train_ids = np.concatenate([np.arange(e["start"], e["end"])
                                            for e in manifest["episodes"] if e["role"] == "train"])
                base = {k: torch.from_numpy(arrays["p_" + k][train_ids])[None]
                        for k in PREDICTIONS}
                mean = semantic_features(base).mean((0, 1))
                del base
                baseline = evaluate(SemanticStopResidual(), arrays, manifest, mean, "facts", check)
                reference = read(PARENT.parent / "review.json")["latest"]
                if baseline["counts"] != reference["counts"] or any(
                    abs(baseline[k] - reference[k]) > 1e-6
                    for k in ("selection_score", "controller_stop_recall",
                              "controller_false_hold_fraction", "go_speed_mae_mps")
                ):
                    raise ValueError("Cached cold parent does not reproduce selected validation")
                atomic_json(args.out / "cache_assessment.json", dict(
                    parent_sha256=PARENT_SHA,
                    cache_sha256=digest(args.out / "parent_predictions.npz"),
                    selected_parent_metrics_reproduced=True, baseline=baseline,
                    original_rgb_timestamps_independently_verified=verified_original,
                    constant_facts_train_only=mean.tolist(), samples=18534,
                    no_model_updates=True,
                ))
                pool = [e for e in manifest["episodes"] if e["role"] == "train"]
                families = sorted({e["family"] for e in pool})
                results, draws = {}, {}
                for arm in ("constant", "facts"):
                    active = args.out / arm
                    active.mkdir()
                    rng = np.random.default_rng(42)
                    torch.manual_seed(42)
                    model = SemanticStopResidual()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
                    selected, best, step = 0, baseline, 0
                    draws[arm] = []

                    def save():
                        atomic_checkpoint(active / "latest.pt", dict(
                            model=model.state_dict(), optimizer=optimizer.state_dict(),
                            rng=rng_state(rng), step=step, selected_step=selected,
                            request_sha256=digest(args.out / "request.json"),
                            best=best, parent_sha256=PARENT_SHA, arm=arm,
                        ))

                    for step in range(1001):
                        check()
                        if step > 0:
                            model.train()
                            family = families[int(rng.integers(len(families)))]
                            choices = [e for e in pool if e["family"] == family]
                            entry = choices[int(rng.integers(len(choices)))]
                            draws[arm].append(entry["id"])
                            base, target = episode(arrays, entry)
                            facts = semantic_features(base)
                            if arm == "constant":
                                facts = mean[None, None].expand_as(facts)
                            prediction, _ = model(base, facts)
                            current = (target["current_speed"] < .1).float()
                            future = (target["speeds"] < .1).float()
                            valid = target["speed_mask"]
                            loss = 2 * F.binary_cross_entropy_with_logits(
                                prediction["stop_logits"][..., 0], current)
                            loss = loss + (F.binary_cross_entropy_with_logits(
                                prediction["stop_logits"][..., 1:], future, reduction="none")
                                * valid).sum() / valid.sum().clamp_min(1)
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Nonfinite semantic residual loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1,
                                                           error_if_nonfinite=True)
                            optimizer.step()
                            with (active / "train.jsonl").open("a") as log:
                                log.write(json.dumps(dict(step=step, episode_id=entry["id"],
                                                          family=family, loss=float(loss.detach())))
                                          + "\n")
                        if step % 250 == 0:
                            metrics = evaluate(model, arrays, manifest, mean, arm, check)
                            accept = admitted(metrics, baseline)
                            if step == 0 or (accept and metrics["selection_score"]
                                             < best["selection_score"] - 1e-8):
                                selected, best = step, metrics
                                atomic_checkpoint(active / "best_residual.pt", dict(
                                    architecture="camera-semantic-stop-residual-v4",
                                    model=model.state_dict(), selected_step=selected, arm=arm,
                                    constant_facts=mean, parent=str(PARENT),
                                    parent_sha256=PARENT_SHA,
                                    request_sha256=digest(args.out / "request.json"),
                                    native_deployable=False,
                                ))
                            with (active / "validation.jsonl").open("a") as log:
                                log.write(json.dumps(dict(step=step, metrics=metrics,
                                                          admitted=accept, selected_step=selected))
                                          + "\n")
                            save()
                        elif step % 100 == 0:
                            save()
                        if step % 25 == 0:
                            progress("Paired full-history CPUstop correction; no native deployment",
                                     arm=arm, steps=step, target_steps=1000, selected_step=selected)
                    results[arm] = dict(selected_step=selected, metrics=best,
                                        checkpoint_sha256=digest(active / "best_residual.pt"))
                    atomic_json(active / "review.json", results[arm])
                    fitted += 1
                if draws["constant"] != draws["facts"]:
                    raise ValueError("Paired arms did not see identical TRAINepisode draws")
                a, b = results["facts"], results["constant"]
                eligible = bool(a["selected_step"] > 0
                                and a["metrics"]["selection_score"]
                                <= b["metrics"]["selection_score"] - .01
                                and a["metrics"]["controller_stop_recall"]
                                >= b["metrics"]["controller_stop_recall"])
                atomic_json(args.out / "assessment.json", dict(
                    results=results, parent=baseline, exact_paired_draws=draws,
                    fact_candidate_eligible_for_new_native_pilot=eligible,
                    native_driving_qualified=False, no_automatic_deployment=True,
                ))
                status = "complete"
    except InterruptedError as error:
        status = "stopped"
        atomic_json(args.out / "error.json", dict(error=str(error)))
    except Exception as error:
        status = "failed"
        atomic_json(args.out / "error.json", dict(error=str(error)))
        raise
    finally:
        progress("Finished; original driving/rule results retained, no automatic promotion")
    print(json.dumps(dict(status=status, fitted=fitted)), flush=True)


if __name__ == "__main__":
    main()
