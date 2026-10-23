# Semantic verification supplement — before complete-set capture

The118 automatically mined candidates remain unreleased. Enlarged RGB review
suspected real lamps, but matched-optics semantics at Town01case7 show building
features (not TrafficLight); Town04case43 red objects are TrafficSigns. Preserve
the mistaken visual interpretation and its correction in the journal. Do not
turn a visual suspicion into a confirmed inventory defect.

Capture all distinct component-TRAIN poses represented by the mined manifest,
Town01–04 only. Same RGB mount/FOV512x288, source pose and seeded background
states, targetRed. Add semantic sensors for training-label diagnosis only; no
policy input or driving benchmark changes. Diagnostic20Hz all-frame capture,
six sensors at exact same world frame, two further render ticks before retrieval,
and recorded image poses must match calibration within1e-3. Copy lens settings
numerically and verify actual spawned RGB/semantic values agree. Preserve the
first invalid optics-copy attempt; do not use it as matched-registration proof.

For each frozen mined box, test a2-native-pixel padded crop in the new semantic
mask. Reject if it contains TrafficLight tag7 or Unknown tag0. Other known
categories, including TrafficSigns8, are negatives for the binary LIGHT housing
detector; they are not negatives for a future stop-sign task. Also require
fresh/source RGB MAE<=10 inside the original candidate crop to verify appearance
correspondence across stationary captures. Keep all tag/pixel statistics and
unknown/rejected candidates. Never use validation pixels for this annotation.

Release a subset only if>=20 canonical boxes,>=6 original TRAIN junctions and
>=3 TRAIN towns pass, with every source/frame/optics/role/map check valid.
Preserve original manifest and initial veto; training_release.json records a
versioned allowed-candidate-index list plus all source hashes. No retroactive
claim that the automatic gate alone certified background. Keep the original
hard-negative training recipe and parent guards if release succeeds; otherwise
do not fit. These additional privileged labels count toward supervision budgets.
