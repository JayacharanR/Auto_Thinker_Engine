# Factual dynamics sources — before model fitting

Use strict-TRAIN geography and preserve the driving/component/heldout_dev/Town05
reservations. Start with the59 TRAIN episodes in the unchanged shared spatial
manifest. Add committed M1v2 noisy/failed episodes omitted from imitation and
the older DAgger collection only where every actual ego pose satisfies the
current strict guard. Do not add its component-validation junction655. Exclude
all13 original driving-validation episodes even though they are globally TRAIN.
Every original source episode, role, mode, native outcome, contact record,
shard hash, actuator acknowledgment and inclusion/rejection stays reported.

This is a new prediction dataset; it does not modify the completed66-cell encoder
study or borrow these additional annotations into its low-label arms. No M3 setup
RGB, encoder driving pilot, long drive or final/development evaluation enters
training. The six-rollout factual branch pilot is calibration only in this first
model study. No counterfactual success is inferred for an unexecuted command.

Native20Hz actions are read from the official recorder and aligned by all poses
to the20Hz world trace; input-framef uses command acknowledgment atf+1. Verify
every available10Hz shard execution label independently. Partial/misaligned or
geography-violating episodes are excluded with a reason, never guessed.

Future token targets use the same past-only frozen CARLA-JEPA encoder as current
tokens, three views of8frames at256x144,48 spatial tokens. The RSSM and ordered
JEPA models get the identical past grid and80 ordered actuator commands. Future
targets/positions/native rules enter only training/scoring code. Latent targets
stop at committed RGB endpoints. Safety-intervened factual observations remain
available with their actual controls, but survival/zero-risk claims beyond an
intervention or unknown end are masked. Observed first contacts stay positive;
do not use native fault-filtered collision counts as labels.

Freeze whole-location dynamics validation before any dynamics fit using SHA256
of the town/junction key and seed62019: first20% (rounded up) of locations in
each town. This only holds out locations from these dynamics sources; report
that the encoder has already seen TRAIN video. Original camera validation,
heldout_dev and Town05 remain excluded from dynamics fitting AND model selection.
Use episode-bootstrap3 independent members for each small learned model, same
data draws/update budget and separate factual calibration from deployment.
Register training budget/selector and cost-label observability after the raw
source audit, before the first update. No final M5 qualification from latent loss.
