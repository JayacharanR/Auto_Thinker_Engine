#!/usr/bin/env python3
"""Rebuild the expansion report from immutable data, checkpoints and audits."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/navigation_models/matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402


def validation_digest(data):
    value = hashlib.sha256()
    for i in data.base.validation:
        value.update(data.base.images[i].tobytes())
        value.update(json.dumps(data.base.samples[i], sort_keys=True).encode())
        for key, array in sorted(data.targets[i].items()):
            value.update(key.encode())
            value.update(array.tobytes())
        value.update(json.dumps(data.annotation_rows[i], sort_keys=True).encode())
    return value.hexdigest()


def main():
    torch.set_num_threads(2)
    outputs = ROOT / "outputs"
    base = outputs / "navigation_m2_signal_corpus_v2_20261016"
    original = outputs / "navigation_m2_signal_detector_20261016"
    expanded = outputs / "navigation_m2_signal_expansion_20261016"
    fitted = outputs / "navigation_m2_signal_joint_expansion_20261016"
    losses = outputs / "navigation_m2_signal_joint_loss_audit_20261016"
    sampling = outputs / "navigation_m2_signal_joint_sampling_audit_20261016"
    evidence = {}

    def read(path):
        evidence[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def archive(directory):
        request = read(directory / "request.json")
        for path, expected in request["source_hashes"].items():
            assert digest(directory / "source" / path) == expected, path
        return request

    requests = {p.name: archive(p) for p in [base, original, expanded, fitted, losses, sampling]}
    data_audit = read(expanded / "data_audit.json")
    attachment = read(expanded / "attachment_audit.json")
    review = read(fitted / "review.json")
    loss_audit = read(losses / "assessment.json")
    sampling_audit = read(sampling / "assessment.json")
    read(losses / "gt_roi_rows.json")
    assert data_audit["static_label_data_gate_passed"] and data_audit["all_cases_complete"]
    assert review["status"] == "complete" and review["selected_step"] == 0
    assert review["frozen_parameters_and_buffers_unchanged"]
    parent_path, latest_path = original / "best_component.pt", fitted / "latest.pt"
    data = SignalDetectionData(base)
    data.attach_hard_negatives(
        outputs / "navigation_m2_signal_hard_negatives_20261016",
        digest(parent_path),
        outputs / "navigation_m2_signal_negative_release_20261016/release.json",
    )
    before = validation_digest(data)
    provenance = data.append_training_corpus(expanded)
    assert before == validation_digest(data) == attachment["validation_payload_sha256"]
    assert len(data.base.train) == 1656 and len(data.base.validation) == 144
    assert sum(len(data.targets[i]["boxes"]) for i in data.base.train) == 642
    assert sum(len(t["hard_negative_boxes"]) for t in data.targets) == 440
    assert provenance == requests[fitted.name]["supplement_provenance"]
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    selected = torch.load(fitted / "best_component.pt", map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert latest["step"] == latest["validation_step"] == 200
    assert selected["step"] == 0 and parent["step"] == 50
    assert all(torch.equal(t, selected["model"][k]) for k, t in parent["model"].items())
    prefixes = ("rpn.", "roi_heads.box_predictor.")
    assert all(
        torch.equal(t, latest["model"][k])
        for k, t in parent["model"].items()
        if not k.startswith(prefixes)
    )
    assert digest(fitted / "best_component.pt") == review["component_sha256"]
    pair = dict(parent=digest(parent_path), candidate=digest(latest_path))
    for directory in [losses, sampling]:
        assert requests[directory.name]["model_sha256"] == pair
    assert sampling_audit["reused_gt_roi_audit_sha256"] == digest(losses / "assessment.json")
    for name in pair:
        assert sampling_audit["results"][name]["total"] == loss_audit["results"][name]["total"]
        assert sampling_audit["results"][name]["saved_total_reproduced"]
    history = [json.loads(line) for line in (fitted / "validation.jsonl").read_text().splitlines()]
    evidence[str((fitted / "validation.jsonl").relative_to(ROOT))] = digest(
        fitted / "validation.jsonl"
    )
    assignments = {}
    for entry in history:
        rows = read(fitted / f"predictions_step{entry['step']}.json")["rows"]
        exact, pairs = audit_predictions(rows, data.labels)
        assert exact == entry["one_to_one"]
        assignments[entry["step"]] = {(rows[a["row"]]["sample"]["path"], a["head"]) for a in pairs}
    retained = len(assignments[0] & assignments[200])
    paired = dict(
        retained=retained,
        lost=len(assignments[0] - assignments[200]),
        gained=len(assignments[200] - assignments[0]),
    )
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
    for path, expected in preserved.items():
        assert digest(ROOT / path) == expected, path
    summary = dict(
        status="Parent retained by frozen loss; joint200 component metrics improve; M2 open",
        data={k: v for k, v in data_audit.items() if k not in ("labels", "image_sha256")},
        attachment=attachment,
        refinement=review,
        history=history,
        fixed_loss_audit=loss_audit,
        sampling_audit=sampling_audit,
        paired_final_label_diagnostic=paired,
        selected_tensors_equal_original_parent=True,
        latest_frozen_features_and_buffers_equal_parent=True,
        verified_images=1800,
        preserved_artifacts=preserved,
        model_sha256=pair,
        evidence_sha256=evidence,
        summary_source_sha256=digest(Path(__file__)),
        limitations=[
            "72 correlated labels at four previously inspected component-development junctions",
            "Combined new data and joint RPN/head intervention, not isolated data ablation",
            "Approximate background candidates are not full detection AP/false-positive truth",
            "GT ROI scores and sampled residuals are privileged diagnostics, not runtime scores",
            "Loss selection preserved; no candidate promotion or driving/JEPA/held-out result",
        ],
    )
    atomic_json(ROOT / "reports/navigation_m2_signal_expansion_20261016.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    steps = [r["step"] for r in history]
    for key, label, color in [
        ("iou3", "Detected", "#377da8"),
        ("iou3_state_correct", "Also correct state", "#c68135"),
    ]:
        axes[0, 0].plot(
            steps, [r["one_to_one"][key] for r in history], "o-", label=label, color=color
        )
    axes[0, 0].set(
        xlabel="Updates",
        ylabel="Known labels /72",
        ylim=(0, 72),
        title="Final detections improve; loss retains parent",
    )
    axes[0, 0].legend()
    for key, label, color in [
        ("iou3", "Proposals IoU ≥ .3", "#56855d"),
        ("iou5", "Proposals IoU ≥ .5", "#8466a4"),
    ]:
        axes[0, 1].plot(
            steps, [r["rpn_proposals"][key] for r in history], "o-", label=label, color=color
        )
    axes[0, 1].set(
        xlabel="Updates",
        ylabel="Known labels /72",
        ylim=(0, 72),
        title="RPN proposals before classification/regression",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(
        steps,
        [r["counts"]["approximate_background_candidates"] for r in history],
        "o-",
        color="#c68135",
    )
    axes[1, 0].set(
        xlabel="Updates", ylabel="Candidates /144 images", title="Approximate backgrounds decrease"
    )
    bottoms = np.zeros(2)
    for key, label, color in [
        ("loss_classifier", "ROI classification", "#377da8"),
        ("loss_box_reg", "ROI box regression", "#c68135"),
        ("loss_objectness", "RPN objectness", "#56855d"),
        ("loss_rpn_box_reg", "RPN box regression", "#8466a4"),
    ]:
        values = [
            loss_audit["results"][name]["components"][key] for name in ("parent", "candidate")
        ]
        axes[1, 1].bar([0, 1], values, bottom=bottoms, color=color, label=label)
        bottoms += values
    axes[1, 1].set(
        xticks=[0, 1],
        xticklabels=["Parent (selected)", "Joint200 (unselected)"],
        ylabel="Fixed detector loss",
        title="Sampling changes loss interpretation",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("M2 approach expansion — component gains, no driving promotion", fontsize=14)
    fig.savefig(ROOT / "reports/figures/navigation_m2_signal_expansion_20261016.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                status=summary["status"],
                verified_images=1800,
                validation_payload_sha256=before,
                paired_label_diagnostic=paired,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
