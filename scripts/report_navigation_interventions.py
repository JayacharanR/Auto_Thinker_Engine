#!/usr/bin/env python3
"""Publish all ten original action outcomes and their independently verified scope."""

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.storage_budget import require_budget  # noqa:E402


def main():
    require_budget()
    verified = ROOT / "outputs/navigation_interventions_verified_20261020"
    manifest = json.loads((verified / "manifest.json").read_text())
    receipt = json.loads((verified / "assessment.json").read_text())
    if digest(verified / "manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("Verified source manifest changed")
    colors = {"go": "#db593e", "brake": "#477db0", "wait_go": "#389a74"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.7), constrained_layout=True)
    for col, kind in enumerate(("lead", "crossing", "rear")):
        episodes = [
            e for e in manifest["episodes"] if e["mechanism"] == kind and e["case"]["repeat"] == 0
        ]
        for episode in episodes:
            rows = [
                json.loads(line)
                for line in (Path(episode["physical_output"]) / "world_trace.jsonl")
                .read_text()
                .splitlines()
            ]
            ego = [next(a for a in r["actors"] if a["role"] == "hero") for r in rows]
            other = [
                next(a for a in r["actors"] if a["role"] == "intervention_" + kind) for r in rows
            ]
            speed = np.linalg.norm([a["velocity"] for a in ego], axis=1) * 3.6
            gap = np.linalg.norm(
                np.array([a["pose"][:2] for a in ego]) - np.array([a["pose"][:2] for a in other]),
                axis=1,
            )
            t = np.arange(160) * 0.05
            branch = episode["case"]["branch"]
            color = colors[branch]
            axes[0, col].plot(t, speed, label=branch, color=color)
            axes[1, col].plot(t, gap, label=branch, color=color)
            tick = episode["prefix_support"].get("first_contact_tick")
            if tick is not None:
                event_time = 4 + tick * 0.05
                for ax in axes[:, col]:
                    ax.axvline(event_time, color=color, linestyle=":", alpha=0.7)
        axes[0, col].set_title(kind.title() + " · observed alternatives")
        axes[0, col].set_ylabel("Actual ego speed (km/h)")
        axes[1, col].set(xlabel="Recorded episode time (s)", ylabel="Actor-centre separation (m)")
        for ax in axes[:, col]:
            ax.axvline(4, color="#5b616b", linestyle="--", linewidth=1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Scripted TRAIN data · dashed: decision · dotted: first native contact\n"
        "After-contact motion is assisted/censored; centre separation is not collision geometry"
    )
    fig.savefig(ROOT / "reports/navigation_intervention_pilot_20261020.png", dpi=150)
    plt.close(fig)
    report = """# Causal-action data pilot — 20 October 2026

Eight original rollouts are independently verified; two reload failures remain
unknown in the ten-attempt denominator. This is factual TRAINdata, not a camera
policy result or an improved world model. The whole pilot audit stays FAILED.

![Actual speed and separation](navigation_intervention_pilot_20261020.png)

| Context | Go | Brake | Wait then go |
|---|---|---|---|
| Lead · Town01/j306 | Contact at 3.25s; repeat matches | Unknown | No contact in 4s |
| Crossing · Town03/j238 | Contact at 3.15s | No contact in 4s | Unknown |
| Rear · Town01/j306 | No contact in 4s | No contact in 4s | No contact in 4s |

There are3observed unfiltered contact starts: two correlated repeats of lead/go,
and crossing/go. Two scenarios have positive outcomes, not three independent
positive encounters. The two missing recorders remain unknown, never zero hits.
No observed contact within4sdoes not mean that the same action is safe later.

The closed-file audit verifies640RGBtimestamps and1,280next-tick actual ego
controls, every RGBpixel hash, both actual prior commands, all15prospective plans
at each decision and both actors'actual geography. All six registered observable
comparisons have EXACTzero differences in position/rotation/velocity/light phase,
including one complete go repeat. This establishes observable replay agreement
for these conditions; no hidden-state or general CARLAdeterminism guarantee.
The original1mreserved-location clearance remains unchanged. Town01ego minimum
is only1.000012m, so do not present that numerical margin as physical certainty.

All candidate controls were saved before first execution, using only current
navigation, own speed and two actual prior commands. Proposed and acknowledged
actions match through native contact:65ticks for lead/go,63for crossing/go.
Privileged post-contact stopping censors future latent/progress targets. Every
first event remains counted. Native event index is tick−1for the80next-tick
first-event hazards; observations at/after an earlier event cannot become new
first-event labels. The verified manifest retains the explicit eligible prefix.

All data here is TRAIN:3contexts at2locations,8rollouts including a repeat.
There are NOindependent calibration episodes in this pilot. Count all640RGB
timestamps and1,280labeled physics ticks in future supervision accounting. Other
actors are scripted, not reactive traffic. All lights are green/frozen: NOred
positives or traffic-rule learning evidence. Rear actors can be outside the rig.
No rear contact occurred, so this pilot has not produced the desired examples
where braking is worse than going. The final rear/brake centre separation is
7.10m; the observed horizon ended before any contact. Do not extend or relabel it.

The pilot originally allowed900s/600MB. Every native capture receipt precedes the
original deadline. Explicit STOP-based cleanup and the separate read-only audit
finished afterwards; that lifecycle overrun is not a successful whole-stage
completion. All four original owner generations remain FAILED. The first short
timeout leak was corrected, but subsequent20stimeouts and fresh-engine failures
showed that was not the full cause. Every orphan was recovered only through its
owned STOPand verified normal SIGTERMretry; no force kill or unrelated job.

The retained core of the final client identifies native LightManagerlight-state
query failure in a streaming callback followed by std::terminate. The stack
is in `outputs/navigation_native_lifecycle_diagnosis_20261020`.
The [pinned LightManager implementation](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/client/LightManager.cpp)
registers that callback; the
[Episode listener](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/client/detail/Episode.cpp)
dispatches it from the native stream. This explains the observed client abort
path, not every engine segmentation fault. A later two-cycle client-release
instrument also FAILED: one captured cycle ended with engine−11; the second
engine crashed. Python survived that instrument, but infrastructure is unqualified.
Use per-episode child-process isolation and an external owner before scaling up.

The separate verified manifest is
`outputs/navigation_interventions_verified_20261020/manifest.json`, SHA
`ef35b75311dd6d683c8f98713eb3cf919e0606d00865dafd79625fe93b1680fc`.
It includes the8audited sources and both explicit rejections; original audit
`outputs/navigation_intervention_audit_20261020` remains FAILED.
No data is silently promoted to expert imitation, no model has been fit on this
corpus, no gate has passed, and M6has not run. The next useful work is reliable
process ownership, diverse predeclared action conditions including closer rear
approaches and reactive traffic, shared perception, then matched first-event fits.
"""
    (ROOT / "reports/navigation_intervention_pilot_20261020.md").write_text(report)
    atomic_json(
        ROOT / "reports/navigation_intervention_pilot_20261020.json",
        dict(
            verified_manifest_sha256=receipt["manifest_sha256"],
            original_planned=10,
            verified=8,
            unknown=2,
            observed_contacts=3,
            unique_positive_contexts=2,
            calibration_episodes=0,
            generating_source_sha256=digest(__file__),
            no_model_or_driving_qualification=True,
        ),
    )
    shutil.copyfile(__file__, verified / "report_source.py")


if __name__ == "__main__":
    main()
