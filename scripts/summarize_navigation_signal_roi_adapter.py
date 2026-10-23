#!/usr/bin/env python3
"""Verify and summarize the ROI adapter/state bundle without GPU or refitting."""

import hashlib
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.assess_navigation_signal_detector import audit_predictions  # noqa: E402
from src.navigation.dagger import digest  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402
from src.navigation.signal_detection_data import SignalDetectionData  # noqa: E402
from src.navigation.signal_detector import ResidualSignalPredictor  # noqa: E402
from src.navigation.signal_roi_learning import (  # noqa: E402
    operational_key,
    supervised_rois,
    validation_payload_digest,
)


def frozen_state_hash(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not name.startswith("roi_heads.box_predictor.adapter."):
            result.update(name.encode())
            result.update(tensor.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def cache_diagnostic(cache, records, checkpoint=None):
    x, old = cache["features"], cache["base_logits"]
    if checkpoint is None:
        logits = old
    else:
        head = ResidualSignalPredictor(FastRCNNPredictor(1024, 2)).eval()
        prefix = "roi_heads.box_predictor."
        head.load_state_dict(
            {k[len(prefix) :]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)},
            strict=True,
        )
        logits = torch.cat([head(x[i : i + 1024])[0] for i in range(0, len(x), 1024)])
    y, domains = cache["labels"], cache["domains"]
    supervised = y >= 0
    errors = F.cross_entropy(logits[supervised], y[supervised], reduction="none")
    masked_y, masked_domain = y[supervised], domains[supervised]
    ce = sum(
        weight * float(errors[mask].mean())
        for weight, mask in [
            (0.25, (masked_y == 1) & (masked_domain == 0)),
            (0.25, (masked_y == 1) & (masked_domain == 1)),
            (0.5, masked_y == 0),
        ]
    )
    teacher = y == -1
    kl = float(
        F.kl_div(
            F.log_softmax(logits[teacher], 1), F.softmax(old[teacher], 1), reduction="batchmean"
        )
    )
    original_roi = torch.tensor([r["variant"] == 0 for r in records])
    foreground = logits[:, 1] > logits[:, 0]
    positives, negatives = (y == 1) & original_roi, (y == 0) & original_roi
    return dict(
        balanced_train_ce=ce,
        unlabelled_teacher_kl=kl,
        objective=ce + kl,
        positive_gt_accepted=int(foreground[positives].sum()),
        positive_gt_total=642,
        certified_negative_accepted=int(foreground[negatives].sum()),
        certified_negative_total=440,
        scope="CPU cached TRAIN ROI diagnostic, not validation/driving accuracy",
    )


def main():
    torch.set_num_threads(2)
    outputs = ROOT / "outputs"
    roi = outputs / "navigation_m2_signal_roi_adapter_20261016"
    state_root = outputs / "navigation_m2_signal_state_expansion_20261016"
    original = outputs / "navigation_m2_signal_detector_20261016"
    joint = outputs / "navigation_m2_signal_joint_expansion_20261016"
    base = outputs / "navigation_m2_signal_corpus_v2_20261016"
    supplement = outputs / "navigation_m2_signal_expansion_20261016"
    evidence = {}

    def read(path):
        evidence[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def archive(directory):
        request = read(directory / "request.json")
        for relative, expected in request["source_hashes"].items():
            assert digest(directory / "source" / relative) == expected, relative
        return request

    request = archive(roi)
    state_request = archive(state_root)
    for directory in [base, supplement, original, joint]:
        archive(directory)
    review = read(roi / "review.json")
    assert state_request["fixed_detector_sha256"] == review["best_candidate_sha256"]
    assert state_request["fixed_prediction_sha256"] == digest(roi / "predictions_step500.json")
    state = read(state_root / "assessment.json")
    bundle = read(state_root / "perception_bundle.json")
    manifest = read(roi / "cache_manifest.json")
    assert review["status"] == "complete" and review["candidate_improved"]
    assert review["selected_step"] == 500 and state["candidate_accepted"]
    assert (
        digest(roi / "best_candidate.pt")
        == review["best_candidate_sha256"]
        == bundle["detector_sha256"]
    )
    assert (
        digest(state_root / "expanded_rgb_linear.npz")
        == state["artifact_sha256"]
        == bundle["state_sha256"]
    )
    assert bundle["selected_joint_state"] == 31 and not bundle["driving_policy_changed"]
    assert digest(state_root / "assessment.json") == bundle["evaluation_sha256"]
    assert digest(roi / "train_roi_cache.pt") == manifest["cache_sha256"] == review["cache_sha256"]
    source = torch.load(joint / "latest.pt", map_location="cpu", weights_only=False)
    assert digest(joint / "latest.pt") == request["initialization_sha256"]
    selected = torch.load(roi / "best_candidate.pt", map_location="cpu", weights_only=False)
    latest = torch.load(roi / "latest.pt", map_location="cpu", weights_only=False)
    assert selected["step"] == selected["validation_step"] == 500
    assert latest["step"] == latest["validation_step"] == 1000
    assert selected["architecture"] == latest["architecture"] == "signal-detector-roi-residual-v1"
    for checkpoint in [selected, latest]:
        assert frozen_state_hash(checkpoint["model"]) == review["frozen_sha256"]
        for name, tensor in source["model"].items():
            mapped = name.replace("roi_heads.box_predictor.", "roi_heads.box_predictor.base.")
            assert torch.equal(tensor, checkpoint["model"][mapped]), name
    data = SignalDetectionData(base)
    data.attach_hard_negatives(
        outputs / "navigation_m2_signal_hard_negatives_20261016",
        source["request"]["parent_sha256"],
        outputs / "navigation_m2_signal_negative_release_20261016/release.json",
    )
    before = validation_payload_digest(data)
    data.append_training_corpus(supplement)
    assert before == validation_payload_digest(data) == request["validation_payload_sha256"]
    cache = torch.load(roi / "train_roi_cache.pt", map_location="cpu", weights_only=False)
    records = manifest["samples"]
    assert list(cache["features"].shape) == [30824, 1024] == manifest["features_shape"]
    grouped = defaultdict(list)
    for record, label, domain in zip(records, cache["labels"].tolist(), cache["domains"].tolist()):
        index = record["sample_index"]
        assert data.base.samples[index]["role"] == "train"
        assert record["label"] == label and record["domain"] == domain
        grouped[index].append(record)
    rng = np.random.default_rng(20261016)
    for index in data.base.train:
        expected = supervised_rois(data, index, rng)
        assert [r for r in grouped[index] if r["label"] >= 0] == expected
        assert len([r for r in grouped[index] if r["label"] == -1]) == 16
    assert set(grouped) == set(data.base.train)
    history = [json.loads(line) for line in (roi / "validation.jsonl").read_text().splitlines()]
    evidence[str((roi / "validation.jsonl").relative_to(ROOT))] = digest(roi / "validation.jsonl")
    matches, selected_rows, selected_pairs = {}, None, None
    for entry in history:
        rows = read(roi / f"predictions_step{entry['step']}.json")["rows"]
        counts, pairs = audit_predictions(rows, data.labels)
        assert counts == entry["one_to_one"]
        if entry["step"] > 0:
            assert (operational_key(entry, history[0]) is not None) == entry[
                "prospective_operational_guard_passed"
            ]
        matches[entry["step"]] = {(rows[m["row"]]["sample"]["path"], m["head"]) for m in pairs}
        if entry["step"] == 500:
            selected_rows, selected_pairs = rows, pairs
    initial_rows = read(joint / "predictions_step200.json")["rows"]
    assert digest(joint / "predictions_step200.json") == request["initial_predictions_sha256"]
    assert initial_rows == json.loads((roi / "predictions_step0.json").read_text())["rows"]
    new_rows = read(state_root / "predictions_expanded_state.json")["rows"]
    combined, combined_pairs = audit_predictions(new_rows, data.labels)
    assert combined == state["one_to_one"] and combined["iou3_state_correct"] == 31
    assert combined_pairs == selected_pairs
    confusions = {}
    for label, rows in [
        ("old_state_on_selected44", selected_rows),
        ("expanded_state_on_selected44", new_rows),
    ]:
        matrix = np.zeros((3, 3), dtype=int)
        for item in selected_pairs:
            row = rows[item["row"]]
            matrix[item["truth"], row["predictions"][item["prediction"]]["state"]] += 1
        confusions[label] = dict(
            matrix=matrix.tolist(),
            matched_red_classified_green=int(matrix[0, 2]),
            class_order=["Red", "Yellow", "Green"],
            matched_only=True,
        )
    for a, b in zip(selected_rows, new_rows):
        assert a["sample"] == b["sample"]
        assert [(p["box"], p["score"]) for p in a["predictions"]] == [
            (p["box"], p["score"]) for p in b["predictions"]
        ]
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
        "outputs/navigation_m2_signal_detector_20261016/best_component.pt": (
            "8df27ee218056e392cc5ddb0f50fef9263a1e8b81517b1d5ecbeba8038a6e49b"
        ),
    }
    for path, expected in preserved.items():
        assert digest(ROOT / path) == expected, path
    paired = dict(
        retained=len(matches[0] & matches[500]),
        lost=len(matches[0] - matches[500]),
        gained=len(matches[500] - matches[0]),
    )
    train_diagnostic = dict(
        initial=cache_diagnostic(cache, records),
        selected=cache_diagnostic(cache, records, selected),
        latest_rejected=cache_diagnostic(cache, records, latest),
    )
    summary = dict(
        status="Selected ROI500 and expanded state component; M2 open, no driver promotion",
        roi_adapter=review,
        roi_history=history,
        state_expansion=state,
        bundle=bundle,
        validation_payload_sha256=before,
        cache_rows_verified=len(records),
        verified_native_images=1800,
        paired_detection_diagnostic=paired,
        matched_state_confusions=confusions,
        cached_train_diagnostic=train_diagnostic,
        frozen_base_parameters_and_buffers_equal_joint200=True,
        preserved_artifacts=preserved,
        evidence_sha256=evidence,
        summary_source_sha256=digest(Path(__file__)),
        limitations=[
            "72 correlated labels at four previously inspected development junctions",
            "COCO supervised features, geometry/semantic labels and unequal stage budgets; "
            "not JEPA ablation",
            "Negative candidates approximate; not exhaustive precision or detection AP",
            "Only top16 TRAIN proposals/image distilled; "
            "not a complete proposal-distribution guarantee",
            "28 known lights missed and13 matched states wrong; no ego-lane/temporal/driving claim",
            "Oracle exact-crop state68->66/72 is privileged and separate from31/72 pipeline",
            "No heldout_dev/Town05 use or changed historical selection",
        ],
    )
    atomic_json(ROOT / "reports/navigation_m2_signal_roi_adapter_20261016.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    steps = [r["step"] for r in history]
    for key, label in [
        ("iou3", "Detected"),
        ("iou3_state_correct", "Also state-correct (old head)"),
    ]:
        axes[0, 0].plot(steps, [r["one_to_one"][key] for r in history], "o-", label=label)
    axes[0, 0].axvline(500, ls="--", color="gray", label="Selected500")
    axes[0, 0].set(
        xlabel="Adapter updates",
        ylabel="Known labels /72",
        ylim=(0, 72),
        title="Cached ROI residual: selected versus latest",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(
        steps,
        [r["counts"]["approximate_background_candidates"] for r in history],
        "o-",
        color="#c17d36",
    )
    axes[0, 1].axhline(209, ls="--", color="gray", label="Initial background ceiling")
    axes[0, 1].set(
        xlabel="Adapter updates",
        ylabel="Candidates /144 images",
        title="Latest has fewer backgrounds but fails town guard",
    )
    axes[0, 1].legend(fontsize=8)
    vmax = max(max(max(row) for row in c["matrix"]) for c in confusions.values())
    for ax, (label, result), title in zip(
        axes[1], confusions.items(), ["Old state head:29/44", "Expanded TRAIN state head:31/44"]
    ):
        matrix = np.asarray(result["matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > vmax / 2 else "black",
            )
        ax.set(
            xticks=[0, 1, 2],
            yticks=[0, 1, 2],
            xticklabels=["R", "Y", "G"],
            yticklabels=["R", "Y", "G"],
            xlabel="Predicted state",
            ylabel="True state",
            title=title + "\nSame44 matched boxes;28 misses excluded",
        )
    fig.suptitle(
        "M2 perception components — fixed development data, no driving result", fontsize=13
    )
    fig.savefig(ROOT / "reports/figures/navigation_m2_signal_roi_adapter_20261016.png", dpi=170)
    plt.close(fig)
    # Illustrations chosen by largest supported red housing per junction, never success.
    examples = {}
    for row in new_rows:
        s = row["sample"]
        if s["phase"] != "red":
            continue
        labels = data.labels[s["town"], s["case_id"], s["view"]]
        if not labels:
            continue
        area = max(
            (r["crop_xyxy"][2] - r["crop_xyxy"][0]) * (r["crop_xyxy"][3] - r["crop_xyxy"][1])
            for r in labels
        )
        key = s["town"], s["junction_id"]
        if key not in examples or area > examples[key][0]:
            examples[key] = area, row
    canvas = Image.new("RGB", (1024, 624), "#19212d")
    for n, (_, row) in enumerate(examples.values()):
        s = row["sample"]
        index = next(i for i in data.base.validation if data.base.samples[i]["path"] == s["path"])
        image = Image.fromarray(data.base.images[index].transpose(1, 2, 0))
        draw = ImageDraw.Draw(image)
        for label in data.labels[s["town"], s["case_id"], s["view"]]:
            draw.rectangle(label["crop_xyxy"], outline="cyan", width=1)
        for p in row["predictions"]:
            draw.rectangle(p["box"], outline=["red", "yellow", "lime"][p["state"]], width=1)
        x, y = n % 2 * 512, n // 2 * 312
        canvas.paste(image, (x, y + 24))
        ImageDraw.Draw(canvas).text(
            (x + 6, y + 5),
            f"{s['town']} junction{s['junction_id']} | cyan GT, colors predicted",
            fill="white",
        )
    canvas.save(ROOT / "reports/figures/navigation_m2_signal_roi_examples_20261016.png")
    print(
        json.dumps(
            dict(
                selected_step=500,
                detected="44/72",
                combined_state="31/72",
                approximate_background=160,
                paired_detection_diagnostic=paired,
                matched_state_confusions=confusions,
                cached_train_diagnostic=train_diagnostic,
                cache_rows_verified=len(records),
                source_and_image_hashes_verified=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
