#!/usr/bin/env python3
"""Verify and report the fixed TRAIN replay; never modifies its original evidence."""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.signal_temporal_audit import (  # noqa: E402
    iou,
    summarize_frames,
    transition_delays,
)


def read(path):
    return json.loads(Path(path).read_text())


def diagnose_identity_and_range(predicted, source):
    """Post-hoc decomposition only; no tracker/threshold changes or reselection."""
    previous, totals, by_route = {}, Counter(), {}
    for row in predicted:
        rid, frame = row["route_id"], row["frame"]
        local = by_route.setdefault(rid, Counter())
        for match in row["scored"]["matches"]:
            key = rid, match["view"], match["actor"], match["head"]
            old = previous.get(key)
            if old and old["frame"] + 2 == frame and old["track_id"] != match["track_id"]:
                target = source[rid, frame]["signal"]["views"][match["view"]]["boxes"][
                    match["head"]
                ]["bbox_xyxy"]
                target = np.clip(target, [0] * 4, [512, 288, 512, 288])
                active = [o for o in row["observations"] if o["track_id"] == old["track_id"]]
                if active and iou(active[0]["box"], target) >= 0.3:
                    label = "old_id_still_overlaps_same_housing"
                elif active:
                    label = "old_id_active_elsewhere"
                else:
                    label = "old_id_absent_current_detections"
                local[label] += 1
                totals[label] += 1
            previous[key] = dict(frame=frame, track_id=match["track_id"])
    ranges = []
    for rid in sorted(by_route):
        known = [
            source[r["route_id"], r["frame"]]
            for r in predicted
            if r["route_id"] == rid and r["scored"]["known"]
        ]
        heights = [
            max(
                b["height_px"]
                for view in r["signal"]["views"].values()
                for b in view["boxes"]
                if b["in_image"] is True
            )
            for r in known
        ]
        ranges.append(
            dict(
                route_id=rid,
                known_timestamps=len(known),
                first_last_label_distance_m=[
                    known[0]["next_light_distance_m"],
                    known[-1]["next_light_distance_m"],
                ],
                largest_native_housing_height_min_median_max_px=np.percentile(
                    heights, [0, 50, 100]
                ).tolist(),
                assigned_track_change_decomposition=dict(by_route[rid]),
            )
        )
    return dict(
        kind="Post-hoc descriptive analysis; no tuning, fitting or changed selection",
        change_decomposition=dict(totals),
        ranges=ranges,
        interpretation="Competing overlapping predictions can change scoring-associated "
        "track ID even when the older ID still covers that housing. This is not "
        "a calibrated physical identity-switch rate.",
    )


def main():
    out = ROOT / "outputs/navigation_m2_signal_temporal_20261016"
    request, assessment = read(out / "request.json"), read(out / "assessment.json")
    if (
        read(out / "progress.json")["status"] != "complete"
        or not assessment["diagnostic_complete"]
        or digest(out / "request.json") != assessment["request_sha256"]
        or digest(out / "predictions.jsonl") != assessment["predictions_sha256"]
    ):
        raise ValueError("Incomplete or modified replay evidence")
    for path, sha in request["source_hashes"].items():
        if digest(out / "source" / path) != sha:
            raise ValueError("Archived replay source changed")
    for path in ("src/navigation/signal_temporal_audit.py",):
        if digest(ROOT / path) != request["source_hashes"][path]:
            raise ValueError("Reproduction helpers differ from original replay")
    bundle_path = Path(request["perception_bundle"])
    if digest(bundle_path) != request["bundle_sha256"]:
        raise ValueError("Selected component bundle changed")
    bundle = read(bundle_path)
    for path_key, sha_key in (
        ("detector_checkpoint", "detector_sha256"),
        ("state_classifier", "state_sha256"),
    ):
        if digest(bundle[path_key]) != bundle[sha_key]:
            raise ValueError("Selected artifact changed")
    projection = ROOT / "outputs/navigation_m2_rule_audit_20261016/samples.jsonl"
    if digest(projection) != request["projection_samples_sha256"]:
        raise ValueError("Source projection data changed")
    source = {
        (r["route_id"], r["frame"]): r
        for r in (json.loads(line) for line in projection.read_text().splitlines())
    }
    predicted = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    if len(predicted) != 518 or len({(r["route_id"], r["frame"]) for r in predicted}) != 518:
        raise ValueError("Changed replay denominator or duplicate frames")
    totals = Counter()
    for episode in assessment["episodes"]:
        rows = [
            dict(source[r["route_id"], r["frame"]], **r)
            for r in predicted
            if r["route_id"] == episode["route_id"]
        ]
        if (
            summarize_frames(rows) != episode["metrics"]
            or transition_delays(rows) != episode["transitions"]
        ):
            raise ValueError("Temporal metrics fail exact reproduction")
        totals.update(episode["metrics"]["counts"])
    if dict(totals) != assessment["counts"]:
        raise ValueError("Aggregate count mismatch")
    protected = {
        "outputs/navigation_m2_stop_go_20261016/best_policy.pt": (
            "80bee4e338eec8f55f57d96ad4cf0bf27c736e47bc641e2b1dd4e8bb0cfdc3e4"
        ),
        "outputs/navigation_m2_cache_20k_20261015/manifest.json": (
            "b9e09499fa0c0158a62849369f44a2c942d90aa9c2dcdff1b7dffdd73e6710ae"
        ),
    }
    if any(digest(ROOT / path) != sha for path, sha in protected.items()):
        raise ValueError("Original driving policy/cache identity changed")
    diagnosis = diagnose_identity_and_range(predicted, source)
    if sum(diagnosis["change_decomposition"].values()) != totals["consecutive_track_id_changes"]:
        raise ValueError("Track-change decomposition does not sum to original count")
    report = dict(
        assessment,
        posthoc_identity_and_range=diagnosis,
        evidence_verified=True,
        preserved_policy_and_cache=protected,
        summarizer_sha256=digest(Path(__file__)),
    )
    destination = ROOT / "reports/navigation_m2_signal_temporal_20261016.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    figures = ROOT / "reports/figures"
    figures.mkdir(exist_ok=True)
    shutil.copyfile(out / "timeline.png", figures / "navigation_m2_signal_temporal_20261016.png")
    c = assessment["counts"]
    matched = c["matched_housings"]
    text = f"""# Causal signal replay — 16 October 2026

The fixed TRAIN replay completed in{assessment["wall_seconds"]:.2f}s. This is
perception/tracking evidence on existing assisted rollouts. **M2 remains open;
the camera driver is unchanged and no tracker is promoted.** No fitting,
simulator, new collection, held-out evaluation or threshold search was performed.

## Scope and split protection

Preserve all six original correction episodes. Exclude routes111/133 and their491
timestamps because Town03/655 is now component-validation. Four safe routes
remain (1,238 timestamps). Replay the two complete signal-bearing routes101/123:
518 timestamps/1,554 native RGB images. They contain{c["known_timestamps"]} known
applicable-state timestamps. These are correlated, familiar TRAIN examples,
not independent encounters or a new driving-validation result.

The frozen [protocol](navigation_m2_signal_temporal_protocol_20261016.md) predates
inference. The RGB wrapper verifies selected detector500 and expanded-state
artifact hashes. A causal, color-blind per-view geometric tracker compensates
ego heading and smooths probabilities. It reports Unknown before confirmation
or after stale evidence. It never confirms lane applicability or commands control.
Training/scoring imports are forbidden in the camera worker; inference modules
pass a fresh-process privilege-guard test.

## Identical-match state comparison

IoU>=.3 scoring matches current RGB predictions to clipped, projected housings
of the privileged applicable actor. There are{c["projected_housings"]} projected
housing/view/time targets and{matched} one-to-one matches. **Projection and
recorded actor phase do not certify visibility, facing or legibility.** No new
state labels are approved for fitting. GT association is scoring-only, so these
numbers do not measure selecting the correct light from the scene.

| State estimate | Correct / same matched targets | Additional coverage |
|---|---:|---|
| Current RGB argmax | {c.get("raw_state_correct", 0)}/{matched} | Every match |
| Causal smoothed argmax | {c.get("tracked_argmax_correct", 0)}/{matched} | Every match |
| Confident smoothed state | {c.get("tracked_confident_correct", 0)}/{matched} | \
{c.get("tracked_confident", 0)} confident; {c.get("tracked_unknown", 0)} Unknown |

Matched red targets:{c.get("matched_truth_Red", 0)}. Raw red-called-green:
{c.get("raw_red_called_green", 0)}; confident tracked red-called-green:
{c.get("tracked_red_called_green", 0)}. Every Unknown stays in the denominator.
Smoothed argmax still calls one matched red target green; zero confident
red-called-green includes abstention and is not a safety result.
Consecutive matched housing pairs:{c.get("consecutive_matched_housing_pairs", 0)},
track-ID changes:{c.get("consecutive_track_id_changes", 0)}. These diagnostics do
not certify physical identity, lane applicability or safe driving.

| Complete TRAIN rollout | Known timestamps with a match | Raw correct / matches | \
Smoothed correct / matches | Longest known detection gap |
|---|---:|---:|---:|---:|
"""
    for e in assessment["episodes"]:
        m, counts = e["metrics"], e["metrics"]["counts"]
        text += (
            f"| {e['route_id']} | {counts.get('known_timestamps_with_match', 0)}/"
            f"{counts['known_timestamps']} | {counts.get('raw_state_correct', 0)}/"
            f"{counts['matched_housings']} | {counts.get('tracked_argmax_correct', 0)}/"
            f"{counts['matched_housings']} | {m['maximum_known_no_detection_run_s']:.1f}s |\n"
        )
    text += """
![Causal state evidence](figures/navigation_m2_signal_temporal_20261016.png)

Red shading is the privileged applicable actor's phase, not perceived truth.
Markers show states only on scoring-associated detections; different heads may
disagree at one time. Absence of a marker means missing evidence, not permission
to proceed. The timeline is not a camera-policy rollout.

## Coverage and identity interpretation

The following analysis was added after inspecting the frozen replay; it does
not retune, select or rerun the tracker. Of133 scoring-associated track changes,
87 still have the older ID overlapping the same GT housing atIoU>=.3,21 have
the older ID active elsewhere, and25 have no current detection for the older
ID. Thus133/734 is **not** a calibrated physical identity-switch rate. Duplicate
overlapping hypotheses contribute substantially; stronger identity arbitration
needs its own prospective experiment.

Following route101 starts with its upcoming-light distance label113.1m and ends
at37.2m. The largest projected native housing is only2.0–4.1px tall (median2.7).
It records vehicle-caused stops, not a complete red-light approach/stop/release.
Route123 starts4.6m before its API reference, with8.1–13.8px housings, already
inside the teacher's red-stop intent. Its wait/release has evidence on179/179
known timestamps. The original labels use API/planner distance semantics, not
verified painted/evaluator stop-line distance. No depth enters inference.

**These two sequences do not cover a complete approach into the light stop.**
Coverage conditional on these distant/near cases must not become a new safety
gate or a claim that memory fixes the missing middle. Projection overlap still
does not certify facing or visibility. The JSON retains measured ranges and
the exact per-route identity decomposition.

## Stop/release and transition evidence

Teacher zero-speed runs are audited on all four split-safe episodes. Causes are
recorded diagnostics, not a complete causal explanation. In particular, the14.1s
light wait on123 begins at the first frame: **left-censored**. Its release is
recorded, but the pre-stop approach/onset is absent. Stop-sign/following holds
must not be relabelled as light behavior. See JSON for every event and endpoint.

| Route / phase transition | Raw first correct evidence | Smoothed argmax | \
Confident tracked state | Observed window |
|---|---:|---:|---:|---:|
"""

    def delay(value):
        return "not observed" if value is None else f"{value:.2f}s"

    for e in assessment["episodes"]:
        for transition in e["transitions"]:
            delays = transition["first_correct_associated_evidence_delay_s"]
            text += (
                f"| {e['route_id']} {transition['previous_state']}→{transition['next_state']} | "
                f"{delay(delays['raw_state'])} | {delay(delays['tracked_argmax'])} | "
                f"{delay(delays['tracked_state'])} | {transition['observed_window_s']:.2f}s |\n"
            )
    latency = assessment["component_and_tracking_seconds"]
    text += f"""
Transition delays use scoring-only GT association and any matching head. They
are not runtime applicable-light decisions or vehicle release times. Unobserved
means no correct associated evidence within the recorded window (up to3s).

FP32 perception plus tracking averages{latency["mean"] * 1000:.1f}ms per three-camera
timestamp,p95{latency["p95"] * 1000:.1f}ms. Shard decoding, scoring and IO are excluded;
this does not establish full-pipeline10Hz operation.

## Decision and remaining work

Keep the tracker as a diagnostic prototype. Next prepare a split-safe TRAIN
collection plan that records a moving approach, braking onset, complete wait
and release in the SAME episode. Begin before the40m region and retain at least
3s after release; include green/no-lead-car and red/green cases, crossing traffic
and stop signs as separate families. Audit actual encounter coverage and
left/right censoring before accepting data. Preserve collisions and interventions.

Before another policy fit, verify facing/visibility and ego-route applicability
targets on those TRAIN sequences. Do not turn unknown/occluded lamps into
negative state labels. Make near-identical duplicate-housing arbitration and
causal memory separate bounded hypotheses. At197ms per triplet, the perception
cadence/budget also needs a measured implementation; do not call this10Hz.
No repeated tracker tuning or long fit is scheduled. Keep the selected driving
parent/cache, heldout_dev and Town05 unchanged. The audit never promotes the tracker.

Reproduce this report with:

```bash
.venv/bin/python scripts/summarize_navigation_signal_temporal.py
```

It checks archived source/input hashes, selected artifacts, all518 predictions,
exact metric/transition reproduction and unchanged original driver/cache. Original
request, predictions and assessment remain in
`outputs/navigation_m2_signal_temporal_20261016`; the JSON report retains every
confusion matrix, transition, stop event and limitation.
"""
    (ROOT / "reports/navigation_m2_signal_temporal_20261016.md").write_text(text)
    print(json.dumps(dict(report=str(destination), counts=c), indent=2))


if __name__ == "__main__":
    main()
