#!/usr/bin/env python3
"""Verify immutable negative-refinement evidence and rebuild its report figures."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/navigation_models/matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402


def main():
    torch.set_num_threads(2)
    evidence = {}

    def read(path):
        evidence[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def archive(directory):
        request = read(directory / "request.json")
        for name, expected in request["source_hashes"].items():
            assert digest(directory / "source" / name) == expected, name
        return request

    outputs = ROOT / "outputs"
    original = outputs / "navigation_m2_signal_detector_20261016"
    mining = outputs / "navigation_m2_signal_hard_negatives_20261016"
    released = outputs / "navigation_m2_signal_negative_release_20261016"
    fitted = outputs / "navigation_m2_signal_hard_negative_fit_retry_20261016"
    calibrated = outputs / "navigation_m2_signal_train_calibration_20261016"
    corpus = outputs / "navigation_m2_signal_corpus_v2_20261016"
    parent_path = original / "best_component.pt"
    latest_path = fitted / "latest.pt"
    selected_path = fitted / "best_component.pt"
    requests = {p.name: archive(p) for p in [original, mining, fitted, calibrated]}
    manifest = read(mining / "manifest.json")
    veto = read(mining / "training_release.json")
    release = read(released / "release.json")
    checks = read(released / "candidate_checks.json")["rows"]
    review = read(fitted / "review.json")
    calibration = read(calibrated / "assessment.json")
    data_audit = read(corpus / "data_audit.json")
    assert not veto["approved"] and release["approved"]
    assert release["manifest_sha256"] == digest(mining / "manifest.json")
    for name, expected in release["source_hashes"].items():
        assert digest(released / "source" / name) == expected, name
    for name, expected in release["semantic_evidence_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in data_audit["image_sha256"].items():
        assert digest(corpus / name) == expected, name
    batches = {}
    for town in ("town01", "town02", "town03", "town04"):
        directory = outputs / f"navigation_m2_negative_semantics_{town}_20261016"
        archive(directory)
        batch = read(directory / "pixel_batch.json")
        assert batch["mining_manifest_sha256"] == digest(mining / "manifest.json")
        batches[town] = batch
    dataset = SignalDetectionData(corpus)
    attached = dataset.attach_hard_negatives(mining, digest(parent_path), released / "release.json")
    negative_copies = sum(len(t["hard_negative_boxes"]) for t in dataset.targets)
    negative_images = sum(bool(len(t["hard_negative_boxes"])) for t in dataset.targets)
    assert negative_copies == 440 and negative_images == 264
    assert not any(len(dataset.targets[i]["hard_negative_boxes"]) for i in dataset.base.validation)
    assert review["selected_step"] == 0 and review["frozen_parameters_and_buffers_unchanged"]
    assert digest(selected_path) == review["component_sha256"]
    assert digest(parent_path) == requests[fitted.name]["parent_sha256"]
    assert digest(latest_path) == requests[calibrated.name]["candidate_sha256"]
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert parent["step"] == 50 and selected["step"] == 0
    assert latest["step"] == latest["validation_step"] == 200
    assert parent["model"].keys() == selected["model"].keys() == latest["model"].keys()
    assert all(torch.equal(t, selected["model"][k]) for k, t in parent["model"].items())
    prefix = "roi_heads.box_predictor.cls_score."
    assert all(
        torch.equal(t, latest["model"][k])
        for k, t in parent["model"].items()
        if not k.startswith(prefix)
    )
    assert any(
        not torch.equal(t, latest["model"][k])
        for k, t in parent["model"].items()
        if k.startswith(prefix)
    )
    assert calibration["foreground_bias_addition"] == 0
    assert calibration["identical_candidate_evaluation_reused"]
    assert calibration["validation"] == latest["validation"]
    assert not calibration["eligible_vs_parent"]
    assert not (calibrated / "calibrated_component.pt").exists()
    preserved = {
        "outputs/navigation_m2_stop_go_20261016/best_policy.pt": (
            "80bee4e338eec8f55f57d96ad4cf0bf27c736e47bc641e2b1dd4e8bb0cfdc3e4"
        ),
        "outputs/navigation_m2_cache_20k_20261015/manifest.json": (
            "b9e09499fa0c0158a62849369f44a2c942d90aa9c2dcdff1b7dffdd73e6710ae"
        ),
        "outputs/navigation_m2_signal_crop_alignment_20261016/robust_rgb_linear.npz": (
            "ca08ac5a7c301c76489a9fc8cde692a9da213c0fa679fd7307e7dc0d2a6b5de2"
        ),
    }
    for name, expected in preserved.items():
        assert digest(ROOT / name) == expected, name
    labels = defaultdict(list)
    for label in data_audit["labels"]:
        if label["static_phase_response_supported"]:
            labels[label["town"], label["case_id"], label["view"]].append(label)
    history = [json.loads(line) for line in (fitted / "validation.jsonl").read_text().splitlines()]
    evidence[str((fitted / "validation.jsonl").relative_to(ROOT))] = digest(
        fitted / "validation.jsonl"
    )
    matched = {}
    for entry in history:
        rows = read(fitted / f"predictions_step{entry['step']}.json")["rows"]
        exact, assignments = audit_predictions(rows, labels)
        assert exact == entry["one_to_one"]
        matched[entry["step"]] = {
            (rows[a["row"]]["sample"]["path"], a["head"]) for a in assignments
        }
    retained = len(matched[0] & matched[200])
    comparison = dict(
        retained_parent_positives=retained,
        lost_parent_positives=len(matched[0] - matched[200]),
        gained_positives=len(matched[200] - matched[0]),
    )
    # Record the separate invalid/stopped diagnostics without altering their evidence.
    invalid_optics = read(
        outputs / "navigation_m2_semantic_probe_town01_20261016/optics_validity.json"
    )
    stopped = read(outputs / "navigation_m2_signal_hard_negative_fit_20261016/progress.json")
    assert not invalid_optics["approved_for_pixel_registration"]
    assert stopped["status"] == "stopped" and stopped["steps"] == 0
    summary = dict(
        status=(
            "All trained candidates rejected; original detector and driving policy retained; "
            "M2 open"
        ),
        mining={k: v for k, v in manifest.items() if k != "rows"},
        release={
            k: v
            for k, v in release.items()
            if k not in ("semantic_evidence_sha256", "source_hashes", "approved_candidate_indices")
        },
        negative_phase_copies=negative_copies,
        negative_training_images=negative_images,
        rejected_source_indices=[r["candidate"] for r in checks if not r["approved"]],
        semantic_capture_metadata=batches,
        provenance=attached,
        refinement=review,
        history=history,
        paired_known_light_diagnostic=comparison,
        train_only_calibration=calibration,
        selected_weights_bit_identical_to_original_parent=True,
        latest_non_classifier_tensors_bit_identical_to_parent=True,
        verified_corpus_images=len(data_audit["image_sha256"]),
        preserved_artifacts=preserved,
        evidence_sha256=evidence,
        summary_source_sha256=digest(Path(__file__)),
        limitations=[
            "Four previously inspected component-development junctions;72 correlated labels",
            "Sparse projected regions: background candidate counts are approximate, not full AP",
            "Semantic labels are privileged additional TRAIN supervision, not runtime inputs",
            "Zero TRAIN-derived bias rejects one remedy, not every calibration or model change",
            "No driving integration, policy promotion, heldout_dev or Town05 result",
        ],
    )
    atomic_json(ROOT / "reports/navigation_m2_signal_hard_negatives_20261016.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    steps = [r["step"] for r in history]
    axes[0, 0].plot(steps, [r["selection_score"] for r in history], "o-", color="#427da5")
    axes[0, 0].set(
        xlabel="Classifier updates", ylabel="Fixed validation detector loss", title="Loss improves"
    )
    for key, label, color in [
        ("iou3", "Detected", "#427da5"),
        ("iou3_state_correct", "Detected + correct state", "#cf8a35"),
    ]:
        axes[0, 1].plot(
            steps, [r["one_to_one"][key] for r in history], "o-", label=label, color=color
        )
    axes[0, 1].axhline(30, ls="--", color="gray", label="Parent recall floor")
    axes[0, 1].set(
        xlabel="Classifier updates",
        ylabel="Known labels /72",
        ylim=(0, 40),
        title="Every trained candidate fails recall guard",
    )
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].plot(
        steps,
        [r["counts"]["approximate_background_candidates"] for r in history],
        "o-",
        color="#538359",
    )
    axes[1, 0].set(
        xlabel="Classifier updates",
        ylabel="Candidates /144 images",
        title="Approximate background candidates fall",
    )
    towns = list(history[0]["one_to_one"]["by_town"])
    positions = list(range(len(towns)))
    for offset, entry, label, color in [
        (-0.18, history[0], "Parent (retained)", "#427da5"),
        (0.18, history[-1], "Latest200 (rejected)", "#cf8a35"),
    ]:
        values = [entry["one_to_one"]["by_town"][town] for town in towns]
        bars = axes[1, 1].bar(
            [x + offset for x in positions],
            [100 * n / total for n, total in values],
            0.36,
            label=label,
            color=color,
        )
        axes[1, 1].bar_label(bars, labels=[f"{n}/{total}" for n, total in values], padding=3)
    axes[1, 1].set(
        xticks=positions,
        xticklabels=[t.removesuffix("_Opt") for t in towns],
        ylabel="Detected known labels (%)",
        ylim=(0, 118),
        title="Town03 coverage regresses",
    )
    axes[1, 1].legend(fontsize=9)
    fig.suptitle("M2 hard negatives: protected rejection, no driving improvement", fontsize=14)
    fig.savefig(ROOT / "reports/figures/navigation_m2_signal_hard_negatives_20261016.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                status=summary["status"],
                verified_images=summary["verified_corpus_images"],
                approved_boxes=release["canonical_approved"],
                negative_copies=negative_copies,
                calibration_bias=calibration["foreground_bias_addition"],
                **comparison,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
