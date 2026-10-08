# Approved overnight BEV continuation — 2026-10-06

The user explicitly requested a long unattended training run using current findings.
This is a BEV traffic diagnostic; it does not establish any JEPA encoder advantage.
Improvement is a hypothesis to test, not a promised outcome.

The current 60k reference and fresh-replay control both scored51/60 collision-free
medium arrivals on streams2029/2030. The fixed-cost pilot scored48/60 after only
21,278/30,000 steps. The optional CPU telemetry timeout that stopped it has been
repaired without changing the original experiment. The earlier continuation
from30k to60k improved47/90 to67/90 safe medium arrivals while retaining10/10 empty
turns; preserving replay/optimizers is therefore a supported continuation to test.

## Scheduled work

1. Resume the original fixed300 pilot for its remaining8,722 steps and complete
   its original canonical checks. Reuse the reference/control checks. Apply the
   explicit recorded runtime recovery; original provenance is preserved.
2. Evaluate all three completed candidates on the known difficult stream2028,
   30 medium episodes each. Choose a candidate retaining at least9/10 collision-free
   simple turns. Rank by the weakest medium stream, pooled collision-free arrivals,
   fewer collisions, then fewer lane departures. Preserve the reference on ties.
3. Copy the selected checkpoint **including optimizer state** and its saved replay
   into a separate run directory. No repeated random prefill or reset of optimizers.
   Preserve all source checkpoints and equal-budget pilot results.
4. Train up to four blocks of30,000 additional medium steps, ratio512, evaluation
   every2,500 steps, checkpoint cadence2,500 and no ten-episode early-stop shortcut.
   Keep the selected source's existing training reward; no stronger penalty sweep.
5. After each block, freeze its checkpoint and run canonical medium30 episodes on
   each of2028/2029/2030 plus simple10 on2029. Accept a checkpoint only if it gains
   at least3/90 safe arrivals, does not worsen the weakest-stream count, pooled
   collisions or lane failures, loses no more than2 arrivals on any stream and
   retains at least9/10 safe simple turns. Reward magnitude is not a selector.
6. Keep the best validated checkpoint independent of `latest.pt`. Stop further
   training after three blocks without an accepted gain, simple retention below8/10,
   or a medium stream below15/30. Reserve time for final checks rather than starting
   a block too close to the deadline.
7. Evaluate the immutable starting baseline and validation-selected best on fresh
   medium streams3031/4031 (30 episodes each), simple3031 (10), and zero-shot
   hard5031 (30). These streams do not select training checkpoints. Write results
   and a comparison table automatically. No hard-task training is scheduled.

## Bounds, reliability and interpretation

Queue: `outputs/overnight_bev_20261006`; continuation run:
`outputs/diagnostics/overnight_bev_20261006`. Expected full duration about7.25h;
hard time cap8h, with earlier stagnation/regression stops. Existing scoped STOP,
GPU/RAM/disk guards and up to four checkpoint-resuming training attempts apply.
Detached launch inhibits automatic sleep, idle suspension and lid-switch sleep
while the queue runs. It cannot prevent manual shutdown or loss of power.

Original pilot lock and new queue lock prevent duplicate ownership. New queue STOP
propagates via the old queue's STOP file while recovery is active. No jobs scripts
are edited. CPU display telemetry cannot abort training. Canonical evaluation
explicitly restores collision_fixed0 and time0.1, with no shield; task definitions,
arrival/collision flags and benchmark rewards remain unchanged. Joint safe-arrival
counts are supplementary selection statistics.

Streams2028/2029/2030 are reused diagnostics, not held-out evidence. Nearby seed
IDs do not guarantee independent or paired traffic scenes. The final streams
were selected before training and are separate from checkpoint selection, but
this remains one training seed in stochastic CARLA. Modest count differences do
not prove a repeatable generalization gain. The controller logs stage decisions
and results to the project journal and keeps source snapshots/code hashes.

Live bar:
```fish
cd ~/Project/Auto_Thinker_Engine; and python3 scripts/watch_queue.py --out outputs/overnight_bev_20261006
```

Clean manual stop: `touch outputs/overnight_bev_20261006/STOP`.
Final artifacts: `results.json`, `source_selection.json`, `best_validation.json`,
`baseline.pt`, `best.pt`, `checkpoints/`, `evaluations/` and `summary.md` in the
queue directory. `completed:true` is required before treating the run as finished.
