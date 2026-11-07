# README demo provenance

`demo.gif` is rendered from the actual command output in `demo.txt`, captured on
7 November 2026. The display omits the repeated test-file prefix and condenses
playback timing; the transcript retains the complete stdout. It contains no
generated driving footage or claimed CARLA execution.

## What executed

1. `python run.py doctor`: passed required checks; CUDA unavailable and CARLA_ROOT
   unset are disclosed as warnings.
2. `python -m pytest tests/test_synthetic_training.py -v --no-header --disable-warnings --color=no`:
   **8 passed**. These existing tests exercise synthetic observations, CNN and
   frozen-JEPA model updates, repeated optimizer updates and checkpoint save/load.

The generator runs these commands using the same Python interpreter as itself,
with OMP/MKL threads capped at two. A failed command prevents GIF generation.
The test environment uses random synthetic images; it is not a driving simulator.
No CARLA server, research training queue or data collection was launched.

## Environment and broader checks

Release preparation started from GitHub `main` at
`74ca1b4824c5cd01a28bde1f96088327e0b0c442`, after fast-forwarding the local checkout
by 30 commits. Both pinned vendor submodules have their existing tracked
compatibility patches applied locally; the upstream gitlinks are unchanged.

This machine's existing environment uses Python 3.10.20 and PyTorch 2.4.1+cu121
on CPU. It differs from the GPU research environment pinned in `uv.lock`.

The full suite after applying both vendor patches returned **534 passed,
18 skipped, 10 failed**. Failures depend on missing ignored navigation manifests,
datasets/checkpoints, the optional navigation vendor, or `numba`. These are
disclosed checkout limitations; no skipped/failed result was relabeled as a pass.
The release does not include those local research artifacts. Historical original-
machine results remain separately identified in the main README.

`ruff check scripts/make_readme_demo.py`, Python compilation, `git diff --check`
and `uv lock --check --offline` pass. The GIF is decoded and representative
frames are visually inspected for legibility and clipping before publication.

## Regeneration

After installing the development dependencies and applying the tracked Dreamer
compatibility patch, run from the source checkout:

```bash
uv run --extra dev python scripts/make_readme_demo.py
```

The script uses Pillow, writes both `demo.gif` and `demo.txt`, and accepts
`--output /path/to/demo.gif`. Timing and dependency warnings can differ by host.
