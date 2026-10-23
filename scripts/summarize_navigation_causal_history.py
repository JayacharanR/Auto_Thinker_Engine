#!/usr/bin/env python3
"""Reproduce the failed causal-history fit summary and figure from saved evidence."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.navigation.dagger import digest  # noqa: E402


def main():
    out = ROOT / "outputs/navigation_m2_causal_history_20261018"
    request = json.loads((out / "request.json").read_text())
    if any(digest(out / "source" / p) != h for p, h in request["source_hashes"].items()):
        raise ValueError("Archived fit source changed")
    rows = [json.loads(s) for s in (out / "validation.jsonl").read_text().splitlines()]
    if [r["step"] for r in rows] != [0, 250, 500, 750, 1000]:
        raise ValueError("Expected all five evaluations, including disclosed extra750")
    if (
        any(r["accepted"] for r in rows)
        or digest(out / "best_policy.pt") != request["parent_sha256"]
    ):
        raise ValueError("Rejection/parent retention evidence differs")
    diagnosis = ROOT / "reports/navigation_m2_causal_diagnosis_20261018.json"
    result = dict(
        run=str(out.relative_to(ROOT)),
        selected_step=0,
        parent_sha256=request["parent_sha256"],
        candidate_improved=False,
        evaluations=rows,
        diagnosis=json.loads(diagnosis.read_text()),
        request_sha256=digest(out / "request.json"),
        diagnosis_sha256=digest(diagnosis),
        scope="Internal validation only; no new driving rollout; every trained update rejected",
        protocol_deviation=(
            "750 was evaluated in addition to declared0/250/500/1000; "
            "rejected with no selection effect"
        ),
    )
    destination = ROOT / "reports/navigation_m2_causal_history_20261018.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    keys = [
        ("controller_stop_recall", "Required-stop recall (%)", 100),
        ("controller_false_hold_fraction", "False holds on go (%)", 100),
        ("go_speed_mae_mps", "Go-speed MAE (m/s)", 1),
    ]
    for ax, (key, title, scale) in zip(axes, keys):
        ax.plot(
            [r["step"] for r in rows],
            [scale * r["causal"][key] for r in rows],
            marker="o",
            color="#315a9b",
        )
        ax.axhline(
            scale * rows[0]["causal"][key], color="#777777", linestyle="--", label="Selected parent"
        )
        ax.set(xlabel="Training updates", title=title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Continuous-data / full-history fit: every trained checkpoint rejected")
    figure = ROOT / "reports/figures/navigation_m2_causal_history_20261018.png"
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    table = "\n".join(
        f"|{r['step']}|{r['causal']['controller_stop_recall']:.2%}|{r['causal']['controller_false_hold_fraction']:.2%}|{r['causal']['go_speed_mae_mps']:.3f}|{r['accepted']}|"
        for r in rows
    )
    (
        ROOT / "reports/navigation_m2_causal_history_20261018.md"
    ).write_text(f"""# Continuous expert data and causal-history fit: rejected

The1000-update countermeasure did not improve the protected parent. Selected0
remains byte-identical `{request["parent_sha256"]}`; no redundant camera driving
run was made. Rejected latest and full recovery state are preserved.

|Update|Required-stop recall|False holds|Go-speed MAE m/s|Accepted|
|---:|---:|---:|---:|---|
{table}

![All evaluations, including750](figures/navigation_m2_causal_history_20261018.png)

Original13 validation episodes and1312 label positions are unchanged. Parent
causal/clipped results are nearly identical. Continuous TRAIN supplement has
2550 timestamps from12 episodes/four locations with7/11/15s waits; no unverified
perception labels receive a loss. All10296 control positions are TRAIN-audited.
Exact replay of1000 sampler draws finds47.35% stop labels. Latest stop hits fall
33→0/278 in signal-memory scenes,146→88/201 signal-stop and41→25/63 stop-sign.
This combined experiment does not identify a single failure mechanism.

Protocol deviation: code also evaluated750 beyond declared0/250/500/1000;750 was
rejected and had no selection effect. All five evaluations remain visible.
The watcher optional-field KeyError was fixed without modifying active training.
Next hypothesis: frozen-parent learned stop residual, localized RGB facts versus
matched null-token arm. No parent promotion, safe-driving claim or M2 completion.
M3–M6 remain open. Town05 was not used in fitting or model selection.

Reproduce with `.venv/bin/python scripts/summarize_navigation_causal_history.py`.
Machine-readable companion contains exact metrics, hashes and sampler diagnosis.
""")
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
