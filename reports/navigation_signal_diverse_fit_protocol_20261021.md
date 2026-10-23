# Frozen data-diversity countermeasure

The supplement audit accepts164/200cases,1968native images,221red/green and219
yellow housings;36Town01clear-noon cases remain unknown after weather-state failure.
Collection owner stays FAILED. All21plannedTRAINjunctions have accepted cases,
overcast100/clear64; noTown10TRAINsignals, no reserved locations added.

Append this exact audit manifest SHA
f400408ae0ce7f53bbf16326592742b4df0ccdc4e79c3f16cc864cb3a7a8bdac
to876previouseligibleTRAINimages:2844TRAIN, same144VAL/72labels byte-identical.
Reuse fixed active-bulb thresholds and original broad projected-light ignore zones.
Do not fit unknown cases or construct VAL pixel masks. Background-label approximations
and previously inspected development-set limitations remain.

One fresh native-stride1/RF21CNN37772parameters, seed42,1000updates,
AdamWlr.001/wd.0001,clip1,BF16forward/FP32foregroundCE+.2meanbackgroundCE,
samebatch12/class-balancedhousing/randomcrop recipe as original bulb study.
This isolates additional TRAIN data, not an architecture/loss change. Do not resume
any rejected checkpoint. No hard-negative correction or context verifier in this fit.
Checkpoint0/100/250/500/1000, model/optimizer/RNG retained. Common selection loss
unchanged: most confident foreground pixel in existing housing state NLL plus.2mean
backgroundCE. Lowestloss/earliertie; no reselect by operational diagnostic.

Compare selected new model against unchanged original selected bulb1000:
44joint, town7/15/12/10, states22/15/7,backgroundpeaks37241,activefraction.01654080755.
For useful component improvement preserve joint and all town/state joint counts,
reduce both background peak count and active fraction by≥90%. Report all checkpoints
and training-data/count differences. Point metric is NOT boxAP or driving reliability.
If gate fails, retain failure/no extension/no automatic native pilot. M4primary and
all M2/M3/M5/M6driving criteria unchanged; this is posthoc engineering development.

Stage1500s including data audit,fit1200s,150MBartifacts,original storage guardian
baseline/STOP. Original raw images referenced, not duplicated; RAM cache temporary.
Record two-lighting data imbalance, failed weather check and raw native exit counts.
