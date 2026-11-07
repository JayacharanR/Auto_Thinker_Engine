#!/usr/bin/env python3
"""Render an actual CPU software-check transcript as the README demo GIF.

Runs the doctor and the existing synthetic Dreamer integration tests. This does
not start CARLA, consume research data, or resume the paused experiments.
Requires the source checkout, patched Dreamer submodule and dev dependencies.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#0d1424"
FOREGROUND = "#d8e2ef"
GREEN = "#6ee7b7"
MUTED = "#94a3b8"


def font(size: int):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        "DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def capture() -> list[str]:
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", NO_COLOR="1"
    )
    lines = []
    commands = [
        ["run.py", "doctor"],
        [
            "-m", "pytest", "tests/test_synthetic_training.py", "-v",
            "--no-header", "--disable-warnings", "--color=no",
        ],
    ]
    for arguments in commands:
        lines.extend(["", "$ python " + shlex.join(arguments)])
        completed = subprocess.run(
            [sys.executable, *arguments], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=180,
        )
        # stdout records the checks; stderr is retained on failure for diagnosis.
        lines.extend(completed.stdout.replace(str(ROOT), ".").splitlines())
        if completed.returncode:
            raise RuntimeError(
                f"Demo check failed ({completed.returncode}):\n"
                + completed.stdout + completed.stderr
            )
    return lines


def render(lines: list[str], output: Path) -> None:
    body_font, title_font, small_font = font(17), font(25), font(15)
    width, height = 1120, 650
    row_height, visible_rows = 24, 18
    wrapped = []
    for line in lines:
        line = line.removeprefix("tests/test_synthetic_training.py::")
        wrapped.extend(
            textwrap.wrap(line, width=103, subsequent_indent="  ")
            or [""]
        )
    frames, durations = [], []
    for end in range(1, len(wrapped) + 1):
        canvas = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 20), "AUTO THINKER ENGINE", font=title_font, fill=FOREGROUND)
        draw.text(
            (30, 59), "CPU SOFTWARE DEMO  /  REAL EXECUTION  /  SYNTHETIC INPUTS",
            font=small_font, fill=GREEN,
        )
        draw.rounded_rectangle((25, 97, width - 25, 575), radius=12, fill="#151f32")
        for index, color in enumerate(("#fb7185", "#fbbf24", "#4ade80")):
            x = 44 + 23 * index
            draw.ellipse((x, 112, x + 11, 123), fill=color)
        draw.text((137, 108), "doctor + Dreamer CNN / JEPA integration checks", font=small_font,
                  fill=MUTED)
        for index, line in enumerate(wrapped[max(0, end - visible_rows):end]):
            color = GREEN if "PASSED" in line or "passed" in line else FOREGROUND
            if line.startswith("$ "):
                color = "#7dd3fc"
            draw.text((43, 140 + index * row_height), line, font=body_font, fill=color)
        draw.text((30, 595), "No CARLA session. No driving qualification. Research remains paused.",
                  font=small_font, fill=MUTED)
        draw.text((30, 620), "Recorded command output; playback condensed for the README.",
                  font=small_font, fill=MUTED)
        frames.append(canvas)
        durations.append(700 if wrapped[end - 1].startswith("$ ") else 180)
    durations[-1] = 3500
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=durations,
        loop=0, optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/demo.gif")
    args = parser.parse_args()
    lines = capture()
    render(lines, args.output)
    args.output.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
