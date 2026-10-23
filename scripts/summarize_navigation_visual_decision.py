#!/usr/bin/env python3
"""Reproduce the spatial ego-speed ablation and its native driving evidence."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/auto_thinker_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402
from src.navigation.metrics import summarize_records  # noqa: E402
from src.navigation.runtime import atomic_json  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def fit(root):
    progress, review = read(root / "progress.json"), read(root / "review.json")
    if progress["status"] != "complete" or progress["steps"] != 2000:
        raise ValueError("Incomplete ablation")
    if digest(root / "best_policy.pt") != review["policy_sha256"]:
        raise ValueError("Changed selected ablation weights")
    history = [json.loads(row) for row in (root / "validation.jsonl").read_text().splitlines()]
    if [row["step"] for row in history] != [0, 500, 1000, 1500, 2000]:
        raise ValueError("Changed validation schedule")
    selected = next(row for row in history if row["step"] == review["selected_step"])
    return dict(
        path=str(root),
        sha256=review["policy_sha256"],
        selected_step=selected["step"],
        metrics=selected["metrics"],
        elapsed_s=progress["elapsed_s"],
    )


def driving(root):
    assessment = read(root / "assessment.json")
    if not assessment["audit_passed"]:
        raise ValueError("Native pilot audit failed")
    result = next(iter(assessment["candidates"].values()))
    return dict(
        path=str(root),
        audit_sha256=digest(root / "assessment.json"),
        arrivals=sum(row["completed"] for row in result["episodes"]),
        rule_safe=result["rule_safe_completed"],
        strict_safe=result["safe_completed"],
        all_attempt_collisions=assessment["all_attempt_collisions"],
        red_violations=sum(row["red_violations"] for row in result["episodes"]),
        stop_violations=sum(row["stop_violations"] for row in result["episodes"]),
        episodes=result["episodes"],
    )


def audit_long(root, expected_policy):
    if not (root / "summary.json").exists():
        return dict(status="running_or_not_started", path=str(root))
    manifest = read(root / "manifest.json")
    sources = read(root / "source_hashes.json")
    if any(digest(root / "source" / p) != h for p, h in sources.items()):
        raise ValueError("Long-drive archived source changed")
    if (
        manifest["seed"] != 42
        or manifest["seed_per_route"]
        or manifest["route"]["route_sha256"]
        != "a6670df4402816eed44b673a21e7f53a86768e024ff13d8a7e2691bacb9eee99"
        or manifest["student"]["sha256"] != expected_policy
        or manifest["student"]["rule_overrides"]
    ):
        raise ValueError("Long-drive protocol changed")
    records = read(root / "leaderboard.json")["_checkpoint"]["records"]
    measured = summarize_records(records)
    if measured["count"] != 1:
        raise ValueError("Expected one original integration route")
    episodes = list((root / "episodes").iterdir())
    if len(episodes) != 1:
        raise ValueError("Long-drive episode missing/duplicated")
    episode = episodes[0]
    profile = read(episode / "profile.json")
    if not profile["all_camera_frames_aligned"] or profile.get("intervention_ticks", 0):
        raise ValueError("Long-drive sensor/override audit failed")
    frames, previous = 0, None
    with (episode / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if (
                (previous is not None and row["frame"] != previous + 1)
                or row["interventions"]
                or row["rule_overrides_enabled"]
            ):
                raise ValueError("Long-drive trace gap/intervention")
            frames += 1
            previous = row["frame"]
    if frames != profile["frames"]:
        raise ValueError("Long-drive frame count mismatch")
    video = episode / "three_cameras.mp4"
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,nb_frames",
                "-of",
                "json",
                str(video),
            ]
        )
    )["streams"][0]
    if (info["codec_name"], info["pix_fmt"], info["width"], info["height"]) != (
        "h264",
        "yuv420p",
        1536,
        288,
    ):
        raise ValueError("Unexpected long-drive video codec")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-threads", "2", "-i", str(video), "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=180,
    )
    recorders = list((root / "recordings").glob("*.log"))
    if len(recorders) != 1 or b"CARLA_RECORDER" not in recorders[0].open("rb").read(30):
        raise ValueError("Native recorder missing/corrupt")
    return dict(
        status=read(root / "summary.json")["execution_status"],
        path=str(root),
        measured=measured,
        profile=profile,
        video=info,
        video_path=str(video),
        video_fully_decoded=True,
        sources_verified=True,
        native_recorder_verified=True,
    )


def main():
    conditioned = fit(ROOT / "outputs/navigation_spatial_resnet_f100_seed42_20261019")
    visual = fit(ROOT / "outputs/navigation_visual_resnet_f100_seed42_20261019")
    names = ("Speed input", "Visual decision")
    pilots = [
        driving(ROOT / path)
        for path in (
            "outputs/navigation_spatial_resnet_pilot_v2_20261019",
            "outputs/navigation_visual_resnet_pilot_20261019",
        )
    ]
    draws = []
    for model in (conditioned, visual):
        draws.append(
            [
                (row["episode_id"], row["family"], row["frames"])
                for row in map(
                    json.loads, (Path(model["path"]) / "train.jsonl").read_text().splitlines()
                )
            ]
        )
    if draws[0] != draws[1] or len(draws[0]) != 2000:
        raise ValueError("Ablation label sampling differs")
    diagnosis = read(ROOT / "reports/navigation_spatial_speed_diagnosis_20261019.json")
    diagnostic_source = (ROOT / "scripts/diagnose_navigation_spatial_speed.py").read_text()
    if hashlib.sha256(diagnostic_source.encode()).hexdigest() != diagnosis["script_sha256"]:
        # Preserve the exact executed source, before a cosmetic long-line fix.
        diagnostic_source = diagnostic_source.replace(
            '"Artificial input sensitivity, not a physical counterfactual "\n'
            '                "or valid driving result. "',
            '"Artificial input sensitivity, not a physical counterfactual '
            'or valid driving result. "',
        )
    if hashlib.sha256(diagnostic_source.encode()).hexdigest() != diagnosis["script_sha256"]:
        raise ValueError("Cannot reproduce exact executed diagnostic source")
    archived = (
        ROOT
        / "reports/sources/navigation_spatial_speed_20261019/diagnose_navigation_spatial_speed.py"
    )
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text(diagnostic_source)
    long = audit_long(ROOT / "outputs/navigation_visual_resnet_long_20261019", visual["sha256"])
    contact_audit_path = ROOT / "outputs/navigation_recorder_contact_pilots_20261019/audit.json"
    rpc_path = ROOT / "outputs/navigation_recorder_rpc_20261019/verification.json"
    contact_audit = read(contact_audit_path)
    rpc = read(rpc_path)
    if not rpc["all_complete_recordings_match"]:
        raise ValueError("Official contact decoding cross-check is incomplete")
    checked = {r["path"]: r["sha256"] for r in rpc["checked"] if r["all_rows_match"]}
    for trial in [*pilots, long]:
        if "path" not in trial:
            continue
        root = Path(trial["path"])
        contacts = [r for r in contact_audit["records"] if Path(r["path"]).is_relative_to(root)]
        expected = list(root.rglob("recordings/route_*_rep*.log"))
        if len(contacts) != len(expected) or not contacts:
            raise ValueError("Incomplete contact evidence for retained camera attempt")
        if any(not r["complete"] or checked.get(r["path"]) != r["sha256"] for r in contacts):
            raise ValueError("Camera contact evidence is not official-query verified")
        trial["unfiltered_contact_starts"] = sum(r["ego_contact_starts"] for r in contacts)
        trial["native_collision_counter_has_stationary_filter"] = True
    output = dict(
        conditioned=conditioned,
        visual=visual,
        pilots=pilots,
        long_route=long,
        identical_2000_label_draws=True,
        speed_probe=diagnosis,
        scope=(
            "Post-hoc M2 feasibility; six familiar TRAIN-location cases, not heldout/final evidence"
        ),
        M2_passed=False,
        M3_M4_M5_M6_complete=False,
        contact_audit_sha256=digest(contact_audit_path),
        official_contact_verification_sha256=digest(rpc_path),
    )
    atomic_json(ROOT / "reports/navigation_visual_decision_20261019.json", output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    positions = [0, 1]
    for offset, (key, label, color) in enumerate(
        (
            ("arrivals", "Arrival", "#285e8e"),
            ("rule_safe", "Rule-compliant", "#2b8c76"),
            ("strict_safe", "No infraction", "#d79536"),
        )
    ):
        axes[0].bar(
            [x + (offset - 1) * 0.23 for x in positions],
            [p[key] for p in pilots],
            width=0.23,
            label=label,
            color=color,
        )
    axes[0].set(
        xticks=positions,
        xticklabels=names,
        ylim=(0, 6.7),
        ylabel="Cases out of six",
        title="Native driving: retain all failures",
    )
    axes[0].legend(fontsize=8)
    variants = ("observed", "zero_speed", "speed_5mps")
    axes[1].bar(
        range(3),
        [diagnosis["metrics"][v]["controller_false_hold_fraction"] * 100 for v in variants],
        color="#a44c40",
        label="False holds",
    )
    axes[1].plot(
        range(3),
        [diagnosis["metrics"][v]["controller_stop_recall"] * 100 for v in variants],
        "o-",
        color="#285e8e",
        label="Stop recall",
    )
    axes[1].set(
        xticks=range(3),
        xticklabels=["Measured", "Zero", "5 m/s"],
        ylim=(0, 105),
        ylabel="Percent",
        title="Original model: artificial speed probe",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Camera policy feasibility • overrides off • no safety/generalization claim", fontsize=11
    )
    figure = ROOT / "reports/figures/navigation_visual_decision_20261019.png"
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    collisions = [p["unfiltered_contact_starts"] for p in pilots]
    violations = [f"{p['red_violations']} / {p['stop_violations']}" for p in pilots]
    long_details = "The long-route audit is pending."
    if "measured" in long:
        row = long["measured"]["episodes"][0]
        long_details = (
            f"Ordered completion{row['completion_percent']}%; "
            f"{long['unfiltered_contact_starts']} unfiltered contact starts "
            f"({row['collisions']} native collision events), "
            f"{row['red_violations']} red violations and{row['stop_violations']} stop violations. "
            "This is a failed safety result. The uninterrupted video fully decodes."
        )
    report = f"""# Camera speed-input ablation —19 October2026

The visual-decision model reaches all six pilot destinations, but still runs one
red and one stop sign. It is not promoted. The speed-conditioned model reached
none; its excellent offline stopping score hid a stationary closed-loop failure.

![Matched pilot and artificial speed diagnostic](figures/navigation_visual_decision_20261019.png)

| Same six-case pilot | Speed input | Visual decision |
|---|---:|---:|
| Arrivals | {pilots[0]["arrivals"]}/6 | {pilots[1]["arrivals"]}/6 |
| Rule-compliant arrivals | {pilots[0]["rule_safe"]}/6 | {pilots[1]["rule_safe"]}/6 |
| Any-infraction-free arrivals | {pilots[0]["strict_safe"]}/6 | {pilots[1]["strict_safe"]}/6 |
| Unfiltered recorder contact starts | {collisions[0]} | {collisions[1]} |
| Red / stop violations | {violations[0]} | {violations[1]} |

Exactly2000 identical labeled-episode draws, fresh identical initial weights,
same labels/loss/validation/controller and native scenario seeds. Only the neural
ego-speed coordinate is omitted in the ablation; the physical controller always
uses actual measured speed. The common selector chooses updates
{conditioned["selected_step"]} and{visual["selected_step"]}, respectively.
Offline stop recall{conditioned["metrics"]["controller_stop_recall"]:.2%} versus
{visual["metrics"]["controller_stop_recall"]:.2%}; false holds
{conditioned["metrics"]["controller_false_hold_fraction"]:.2%} versus
{visual["metrics"]["controller_false_hold_fraction"]:.2%}. Lower teacher-forced
error did not imply better autonomous behavior.

The artificial input probe is not a physical counterfactual or a driving score.
TRAIN contains392 stationary-go labels versus8910 stationary-stop;
273/392 launch labels are omitted by burn-in8. This remains a disclosed data/loss
limitation, not silently repaired in one encoder arm.

Both native pilot audits verify all declared setups, bound physical lights,
camera/trace alignment, source archives, distinct CARLA recorders and full H.264
video decoding. Six familiar TRAIN-location cases cannot establish generalization;
the historical stop/go model achieved3/6 rule-compliant, expert6/6. No heldout_dev
or Town05 result is consumed here. Every failed case remains in its denominator.

Original1233m integration route status: **{long["status"]}**.
{long_details}
Detailed outcomes and audit evidence: [reproduced JSON](navigation_visual_decision_20261019.json).
The pinned native criterion filters near-stationary collisions; preserved recorder
queries supply contact counts independently. All313 complete recordings match
official queries. See [collision correction](navigation_collision_correction_20261019.md).

The old integration route traversed two now-reserved heldout_dev locations,
Town03 junctions1135/103 (142/105 recorded junction ticks). This engineering
exposure must be disclosed; the17-location development geography is no longer
wholly unseen. Keep all300 frozen cases and full denominators, avoid tuning on
those sites and never remove them after seeing this result. The common encoder
architecture/protocol was fixed before the long-route outcomes. TRAIN arrays
exclude both sites; Town05 student outcomes remain unused.

The first broad encoder matrix never started. The corrected common architecture
is frozen in navigation_m4_visual_protocol_20261019.md before other encoder fits;
the ResNet feasibility history is disclosed. M2 encounter/reliable-driving,
M3 physical rule tests, full M4 study, M5 planning and M6 final driving remain open.

Reproduce: `.venv/bin/python scripts/summarize_navigation_visual_decision.py`.
"""
    (ROOT / "reports/navigation_visual_decision_20261019.md").write_text(report)
    print(
        json.dumps(
            dict(pilot_rule_safe=[p["rule_safe"] for p in pilots], long_status=long["status"])
        )
    )


if __name__ == "__main__":
    main()
