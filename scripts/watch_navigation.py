#!/usr/bin/env python3
"""Read-only progress for bounded navigation runs; Ctrl-C only stops watching."""

import argparse
import json
import time
from pathlib import Path


def render(output):
    path = Path(output) / "progress.json"
    if not path.exists():
        return "Waiting for progress.json", False
    p = json.loads(path.read_text())
    percent = min(100, max(0, p.get("route_progress_percent", 0)))
    target = p.get("target_frames", 0)
    fraction = min(1, p.get("frames", 0) / target) if target else percent / 100
    filled = int(fraction * 20)
    age = max(0, time.time() - p.get("updated_at", 0))
    bar = "█" * filled + "░" * (20 - filled)
    measure = "samples" if p.get("stage") == "M1 collection" else "frames"
    phase = "collection" if measure == "samples" else ("profile" if target else "route")
    text = (
        f"{p['status']:16} [{bar}] {phase} {fraction * 100:5.1f}% | "
        f"route {percent:5.1f}% | "
        f"{p.get('frames', 0):,} {measure} | {p.get('frames_per_second', 0):.1f} fps | "
        f"{p.get('speed_kmh', 0):.1f} km/h | update {age:.0f}s ago"
    )
    if p.get("stage") in (
        "Approved compact runtime activation",
        "Approved ego-only feedback interface",
    ):
        text = f"{p['status']:16} | {p.get('phase', '')} | update {age:.0f}s ago"
    if p.get("stage") == "Verified TRAINlabel sources":
        text = (
            f"{p['status']:16} [{bar}] 13/13 original sources settled; "
            f"{p['accepted_sources']} verified, {p['quarantined_sources']} quarantined | "
            "No training or model promotion"
        )
    if p.get("stage") == "Verified factual action sources":
        text = (
            f"{p['status']:16} [{bar}] {p.get('frames', 0)}/{target} original attempts settled; "
            f"{p['accepted_sources']} verified, {p['unknown_sources']} unknown | "
            "No dynamics fitting or driving promotion"
        )
    if p.get("stage") == "Verified raw RGB in RAM":
        text = (
            f"{p['status']:16} [{bar}] {p.get('frames', 0):,}/{target:,} "
            f"verified raw RGB cache | update {age:.0f}s ago"
        )
    if measure == "samples":
        text = (
            f"{p['status']:16} [{bar}] {p.get('frames', 0):,}/{target:,} samples | "
            f"episode {p.get('current_episode')} | update {age:.0f}s ago"
        )
    if p.get("stage") == "M2 RGB fact cache":
        text = (
            f"{p['status']:16} [{bar}] RGB fact cache {p.get('frames', 0):,}/{target:,} "
            f"({fraction:.1%}) | {p.get('frames_per_s', 0):.1f} frames/s | "
            f"ETA {p.get('eta_seconds', 0) / 60:.1f} min | update {age:.0f}s ago"
        )
    if p.get("stage") in ("M4 optical data cache", "M4 frozen encoder cache"):
        text = (
            f"{p['status']:16} [{bar}] {p.get('stage')} {p.get('encoder', '')} "
            f"{p.get('frames', 0):,}/{target:,} ({fraction:.1%}) | "
            f"{p.get('frames_per_second', 0):.1f} frames/s | "
            f"ETA {p.get('eta_seconds', 0) / 60:.1f} min"
        )
    if p.get("stage") in (
        "M4 spatial student",
        "M2 visual decision ablation",
        "M2 cold-start and fact supervision",
    ):
        done, total = p.get("steps", 0), p.get("target_steps", 2000)
        filled = int(20 * min(1, done / total))
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] {p.get('encoder')} seed{p.get('seed')} "
            f"{done}/{total} updates | labels {p.get('labels'):,} | "
            f"selected {p.get('selected_step')} | {p.get('phase')}"
        )
        if p.get("validation"):
            v = p["validation"]
            text += (
                f"\nValidation at{p.get('validation_step')}: "
                f"stop recall {v['controller_stop_recall']:.1%}, "
                f"false holds {v['controller_false_hold_fraction']:.1%}; offline only."
            )
    if p.get("stage") == "M4 TRAIN-only JEPA":
        done, total = p.get("steps", 0), p.get("target_steps", 10000)
        filled = int(20 * min(1, done / total))
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] JEPA {done}/{total} unlabeled updates | "
            f"{p.get('phase')} | update {age:.0f}s ago"
        )
    if p.get("stage") == "M5 factual dynamics member":
        done, total = p.get("steps", 0), p.get("target_steps", 2000)
        filled = int(20 * min(1, done / total))
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] {p.get('model')} seed{p.get('seed')} "
            f"{done}/{total} factual updates | {p.get('phase')} | "
            f"selected {p.get('selected_step')}; offline only"
        )
    if p.get("stage") in (
        "M5 past-only factual encoder cache",
        "M5 GPU road-user profile",
        "M5 closed branch evidence audit",
        "Verified COW storage recovery",
        "M5 frozen-plan diagnosis",
        "TRAIN label-rig profile",
        "Dense-label independent data audit",
        "M5 visible-instance perception audit",
        "TRAIN metric depth convention",
        "Shared RGB road-user cache",
        "Encoded box-depth baselines",
        "Intervention geometry",
        "Intervention independent audit",
        "Diverse causal-action TRAINpilot",
        "Native callback lifecycle instrument",
        "Native process isolation instrument",
        "Frozen multi-location action data",
        "Native factual action worker",
        "Expansion independent action audit",
        "Geometry-corresponded prefix audit",
        "Shared causal intervention features",
        "First-event simple baselines",
        "First-event matched model study",
        "First-event frozen candidate audit",
        "Hard background signal study",
        "RGB signal context verifier",
        "Strict TRAINsignal diversity",
        "Strict signal independent audit",
        "Frozen context learning diagnosis",
        "Diverse native bulb study",
        "Frozen proposer visual feature cache",
        "Independent frozen-proposer cache audit",
        "Full TRAIN proposal verification",
        "RGB local proposal refinement",
        "RGB refinement component evaluation",
        "Independent RGB proposal TRAIN inventory",
        "Independent RGB plateau classifier audit",
        "Expanded background full TRAIN audit",
        "Spatial signal TRAIN cache",
        "Spatial RGB full TRAIN audit",
        "Exact signal context profile",
        "Fused exact signal context profile",
        "CUDA exact signal context profile",
    ):
        text = (
            f"{p['status']:16} [{bar}] {p.get('frames', 0):,}/{target:,} "
            f"{p['stage']} | {p.get('phase', '')} | update {age:.0f}s ago"
        )
        if p.get("stage") == "RGB signal context verifier":
            text += (
                f"\n  RGB cache {p.get('images_done', 0)}/{p.get('images_total', 1020)} images; "
                "all validation proposals scored, no runtime top-K filter"
            )
        if p.get("stage") == "Strict TRAINsignal diversity":
            text += (
                f"\n  {p.get('tasks_done', 0)}/8 native tasks settled; "
                f"{p.get('captured_case_receipts', 0)}/200 case receipts; independent audit pending"
            )
        if p.get("stage") == "Frozen multi-location action data":
            text += (
                f"\n  Captured {p.get('captured', 0)}/{target}; "
                f"abnormal native exits {p.get('abnormal_native_exits', 0)}; "
                "settled attempts are not verified data or driving successes"
            )
        if p.get("stage") == "M5 GPU road-user profile" and p["status"] == "complete":
            assessment = Path(output) / "assessment.json"
            if assessment.exists():
                measured = json.loads(assessment.read_text()).get("frames", 0)
                measured_fraction = min(1, measured / target) if target else 0
                filled = int(measured_fraction * 20)
                measured_bar = "█" * filled + "░" * (20 - filled)
                text = (
                    f"{p['status']:16} [{measured_bar}] {measured}/{target} "
                    f"{p['stage']} | assessment count; raw queue counter {p.get('frames', 0)}"
                )
    if p.get("fits_total"):
        done, total = p.get("fits_done", 0), p.get("fits_total", 66)
        filled = int(20 * min(1, done / total))
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] {done}/{total} registered fits | "
            f"{p.get('phase')} | update {age:.0f}s ago"
        )
        if p.get("stage") in (
            "Frozen pretrained signal verifier",
            "Tight-crop signal verifier",
            "Nonlinear RGB and TRAIN augmentation",
        ):
            images = p.get("frames", 0)
            image_total = p.get("target_frames", 2988)
            filled = int(20 * min(1, images / image_total))
            text += (
                f"\n  [{'█' * filled}{'░' * (20 - filled)}] "
                f"{images}/{image_total} images encoded; original proposals retained"
            )
        if p.get("stage") == "Native bulb-label paired study":
            step, count = p.get("steps", 0), p.get("target_steps", 1000)
            filled = int(20 * min(1, step / count))
            text += (
                f"\n  [{'█' * filled}{'░' * (20 - filled)}] "
                f"{p.get('arm')} {step}/{count} updates; perception component only"
            )
        if p.get("stage") == "M2 structured semantic residual":
            if p.get("episodes_total"):
                verified, total_episodes = p.get("episodes_done", 0), p["episodes_total"]
                filled = int(20 * min(1, verified / total_episodes))
                text += (
                    f"\n  [{'█' * filled}{'░' * (20 - filled)}] "
                    f"{verified}/{total_episodes} source episodes checked"
                )
            elif "steps" in p:
                step, target_steps = p["steps"], p["target_steps"]
                filled = int(20 * min(1, step / target_steps))
                text += (
                    f"\n  [{'█' * filled}{'░' * (20 - filled)}] "
                    f"{p.get('arm')} {step}/{target_steps} updates; "
                    f"selected {p.get('selected_step')}; offline only"
                )
    if p.get("stage") == "Lossless completed-trace archive":
        done, total = p.get("files", 0), p.get("total_files", 1)
        filled = int(20 * min(1, done / max(1, total)))
        bar = "█" * filled + "░" * (20 - filled)
        saved = (p.get("original_bytes", 0) - p.get("archive_bytes", 0)) / 1e9
        text = (
            f"{p['status']:16} [{bar}] {done}/{total} verified trace archives | "
            f"{saved:.2f} GB logical reduction"
        )
    if p.get("stage") == "Exact original-trace restoration":
        done, total = p.get("files", 0), p.get("total_files", 1)
        filled = int(20 * min(1, done / max(1, total)))
        bar = "█" * filled + "░" * (20 - filled)
        text = f"{p['status']:16} [{bar}] {done}/{total} original traces restored exactly"
    if p.get("stage") == "M2 stop residual":
        done, total = p.get("steps", 0), p.get("target_steps", 1000)
        filled = int(20 * min(1, done / total))
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] {p.get('arm')} residual {done}/{total} updates | "
            f"{p.get('phase')} | selected {p.get('selected_step')} | update {age:.0f}s ago"
        )
        if p.get("validation"):
            v = p["validation"]
            text += (
                f"\nLast evaluated: stop recall {v['controller_stop_recall']:.2%}, "
                f"false holds {v['controller_false_hold_fraction']:.2%}; offline only."
            )
    if p.get("images_total") and p.get("stage") != "RGB signal context verifier":
        done, total = p.get("images_done", 0), p["images_total"]
        filled = int(20 * min(1, done / total))
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] {done}/{total} image evaluations | "
            f"{p.get('phase')} | update {age:.0f}s ago\n"
            "Perception diagnostic; no driving evaluation."
        )
    if p.get("encounters_total"):
        static = p.get("stage") == "M2 static signal supervision"
        fraction = min(1, p["encounters_done"] / p["encounters_total"])
        filled = int(fraction * 20)
        bar = "█" * filled + "░" * (20 - filled)
        unit = "static TRAIN poses" if static else "encounters"
        text = (
            f"{p['status']:16} [{bar}] {unit} {p['encounters_done']}/{p['encounters_total']} "
            f"| current {p.get('current_chunk')} | update {age:.0f}s ago\n"
            + (
                "Perception data only; no driving policy evaluation."
                if static
                else "Live driving detail: watch the current chunk directory with --out."
            )
        )
        if p.get("stage") == "M3 native setup pilot":
            text = (
                f"{p['status']:16} [{bar}] Expert setup probes "
                f"{p['encounters_done']}/{p['encounters_total']} | {p.get('phase')}"
            )
        if p.get("stage") in (
            "M3 complete physical reference",
            "M5 factual branch pilot",
            "M4 unchanged encoder driving pilots",
            "M2 corrected camera pilots",
            "M2 corrected camera pilots, exact unstarted continuation",
            "M3 override-disabled camera rules",
            "M3 frozen-primary followthrough",
            "TRAIN expert/camera long integration",
            "TRAIN dense-label collection",
        ):
            text = (
                f"{p['status']:16} [{bar}] {p['encounters_done']}/{p['encounters_total']} "
                f"{p['stage']} | {p.get('phase')}"
            )
        if "unknown_cases" in p:
            text += (
                f"\nNative cases observed {p['native_cases_observed']}; "
                f"unknown failures retained {p['unknown_cases']}. Bar includes both."
            )
    if p.get("stage") == "M2 camera student":
        preparing = p.get("phase") == "preparing raw image cache"
        done = p.get("frames", 0) if preparing else p.get("steps", 0)
        total = p.get("target_frames", 0) if preparing else p.get("target_steps", 0)
        fraction = done / total if total else 0
        filled = int(min(1, fraction) * 20)
        bar = "█" * filled + "░" * (20 - filled)
        rate = p.get("steps_per_second", 0)
        eta = (total - done) / rate if rate and not preparing else None
        text = (
            f"{p['status']:16} [{bar}] {done:,}/{total:,} "
            f"{'cached timestamps' if preparing else 'optimizer steps'} | {p.get('phase')} "
            f"| update {age:.0f}s ago"
        )
        if eta is not None:
            text += f" | {rate:.2f} steps/s ETA {eta / 60:.1f}m"
        validation = p.get("validation")
        if validation:
            text += f"\nInternal validation speed MAE {validation['speed_mae_mps']:.3f}m/s"
            if "go_predicted_stop_fraction" in validation:
                text += f"; go predicted stop {validation['go_predicted_stop_fraction']:.1%}"
            if "controller_stop_recall" in validation:
                text += (
                    f"\nController stop recall {validation['controller_stop_recall']:.1%}; "
                    f"false holds on go {validation['controller_false_hold_fraction']:.1%}"
                )
        history_path = Path(output) / "validation.jsonl"
        if history_path.exists():
            records = history_path.read_text().strip().splitlines()
            last = json.loads(records[-1]) if records else {}
            if "causal" in last:
                text += (
                    f"\nCausal metrics last evaluated at step {last['step']}; "
                    f"selected checkpoint step {p.get('selected_step', 0)}"
                )
    if p.get("stage") == "M2 ROI adapter" and not p.get("images_total"):
        done, total = p.get("steps", 0), p.get("target_steps", 1000)
        filled = int(20 * min(1, done / total)) if total else 0
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] ROI classifier {done}/{total} updates | "
            f"{p.get('phase')} | selected step {p.get('selected_step')} | "
            f"improved candidate: {p.get('candidate_improved', False)} | update {age:.0f}s ago"
        )
        if p.get("validation"):
            counts = p["validation"]["counts"]
            text += (
                f"\nLatest evaluated detection {counts['detected_iou3']}/{counts['centers']}; "
                f"joint state {counts['detected_iou3_state_correct']}; "
                "perception only, driving policy unchanged."
            )
        review_path = Path(output) / "review.json"
        if p["status"] in ("complete", "stopped") and review_path.exists():
            review = json.loads(review_path.read_text())
            counts = review["selected_validation"]["counts"]
            text += (
                f"\nSelected component step {review['selected_step']}: "
                f"{counts['detected_iou3']}/{counts['centers']} detected, "
                f"{counts['detected_iou3_state_correct']} joint with the frozen state head."
            )
    assessment_path = Path(output) / "assessment.json"
    if p["status"] == "complete" and assessment_path.exists():
        assessment = json.loads(assessment_path.read_text())
        if "new_joint_state" in assessment:
            selected_joint = (
                assessment["new_joint_state"]
                if assessment["candidate_accepted"]
                else assessment["old_joint_state"]
            )
            text += (
                f"\nFixed detections {assessment['fixed_detections']}/{assessment['denominator']}; "
                f"selected state component {selected_joint}/{assessment['denominator']} joint."
            )
    if p.get("stage") == "M2 signal localizer":
        done, total = p.get("steps", 0), p.get("target_steps", 0)
        filled = int(20 * min(1, done / total)) if total else 0
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] signal component {done:,}/{total:,} optimizer steps "
            f"| {p.get('phase', 'training')} | update {age:.0f}s ago"
        )
        if p.get("validation"):
            v = p["validation"]
            text += (
                f"\nDevelopment center class accuracy {v['center_class_accuracy']:.1%}; "
                f"background active cells@0.5 {v['background_active_fraction_at_half']:.2%}"
            )
        text += "\nPerception component only; driving policy unchanged."
    if p.get("error"):
        text += f"\n{p['error']}"
    if p.get("stage") == "M2 signal detector":
        done, total = p.get("steps", 0), p.get("target_steps", 200)
        filled = int(20 * min(1, done / total)) if total else 0
        bar = "█" * filled + "░" * (20 - filled)
        text = (
            f"{p['status']:16} [{bar}] detector {done:,}/{total:,} optimizer steps | "
            f"{p.get('phase')} | validation images {p.get('validation_images', 0)}/144 | "
            f"update {age:.0f}s ago"
        )
        if p.get("validation"):
            counts = p["validation"]["counts"]
            text += (
                f"\nLatest evaluated sparse housing recall: {counts['detected_iou3']}/"
                f"{counts['centers']} at IoU>=.3; best selected step {p.get('selected_step')}."
            )
        review_path = Path(output) / "review.json"
        if p["status"] in ("complete", "stopped") and review_path.exists():
            review = json.loads(review_path.read_text())
            counts = review["validation"]["counts"]
            text += (
                f"\nSelected step {review['selected_step']}: {counts['detected_iou3']}/"
                f"{counts['centers']}; perception only, driving policy unchanged."
            )
        if p.get("error"):
            text += f"\n{p['error']}"
    if p.get("training_mixture_not_camera_evaluation"):
        text += "\nTRAINING mixture; privileged assistance ON."
        samples = p.get("correction_samples", p.get("collection_samples"))
        if samples is not None:
            text += f" {samples:,} correction samples."
        if "privileged_intervention_ticks" in p:
            text += f" {p['privileged_intervention_ticks']:,} teacher-selected ticks."
    if p.get("summary"):
        s = p["summary"]
        text += (
            f"\nSafe completed: {s['safe_completed']}/{s['count']}; "
            f"all collision events: {s['all_collisions']}; "
            f"stop reason: {s.get('stop_reason')}"
        )
    return text, p["status"] not in ("starting", "running")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/navigation_m0")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            text, done = render(args.out)
            print(text if args.once else "\033[2J\033[H" + text, flush=True)
            if args.once or done:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
