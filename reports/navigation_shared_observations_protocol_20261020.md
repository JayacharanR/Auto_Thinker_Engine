# Shared camera road-user observations — before caching

Preserve every road-user detection accepted by the original COCO FPN classes and
score0.35, including boxes without a valid flat-ground intersection. Make no
model/threshold/ground-plane change, merge or hidden truncation. This is an
optional new representation derived from the same declared RGB and calibration,
not a policy input extension, learned depth, collision probability or new driver.

Version navigation-camera-road-users-v1: three cameras ×100detector slots ×20
features with separate presence masks. Features contain detector confidence,
six COCOclass bits, normalized box corners, unit ego-frame bottom-centre ray,
ground-valid bit, approximateXY/60when valid, image truncation and horizon/range
unknown bits. Keep explicit ground-status strings, including bottom-image
failure. Padding and unlocated coordinates must never mean empty/free space.
All raw accepted boxes retained; exceeding100per view raises instead of dropping.
No simulator, teacher, other-actor identity, depth target or future frame enters
observation construction. CV, RSSM and JEPA receive the identical version when
future experiments adopt it. None has adopted it yet.

Cache the separate verified manifest's complete351TRAIN and78calibration
timestamps across9TRAIN/2predeclared calibration sources. Verify raw source-shard
and manifest hashes. Use original policyRGBreaders and pixel hashes, never read
label-shard arrays. Reuse exact previously committed raw calibration predictions
instead of repeating inference. Bind their raw prediction SHA and unchanged
FPNdescriptor source identity. For TRAINuse the same frozen FP32forward with three
unscored warmups. On the first TRAINpacket independently compare the original
production forward's raw hook pack against the new observation wrapper; require
bitwise equal values/presence and equal raw observations before publishing cache.

Archive raw metadata/predictions, explicit source roles, feature names, descriptor
and source hashes. Small exclusive atomic chunks survive an interruption. No
model fitting, label selection, data substitution or driving promotion. This
cache does not add local supervision; future fitting must still charge every
supervised timestamp and disclose COCOexternal labels/oldencoder exposure. The
two calibration locations already had historical driver/backbone exposure and
remain calibration-only for new fits. Both rejected dense sources remain excluded.

One GPU/global navigation lock, two CPUthreads,900swhole-stage/200MBartifact cap,
same14GBadditional/10GBminimum reserve guardian. Check owned STOPbefore each
packet; no duplicate queues. Native depth probe failure remains failed/unknown;
these observations do not substitute learned geometry or qualify M5planning.
